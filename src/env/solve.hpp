#pragma once

#include "core/physics.hpp"
#include "env/episode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <unordered_set>
#include <vector>

// Macro-action breadth-first search: an exact planner in the POLICY'S action
// space. tools/solve.cpp searches per tick (its output is the deploy-format
// existence proof); this one branches on the env's macro decisions
// (EnvInstance::step: a hop is one action, walks/ship control are
// ticksPerAction ticks), so a found path is a stream of decisions the policy
// could have taken, and its recorded per-tick holds replay bit-exactly
// through gdenv_replay. Used by the trainer to manufacture demonstrations
// at walls the on-policy sampler cannot cross (a wall needing ~15
// consecutive no-flips against a saturated prior is a 1e-6 event for the
// sampler and a few thousand node expansions here).
//
// States are expanded in tick order (x is a function of the tick, so every
// state in a bucket shares x); duplicates within a bucket are dropped by
// the same quantized key tools/solve.cpp uses (an approximation only for
// dedup — every kept state is exact). Buckets over maxStates are thinned
// deterministically by key hash.
namespace gdenv {

struct SolveKey {
    int32_t y, yvel;
    uint32_t flags;
    bool operator==(const SolveKey& o) const {
        return y == o.y && yvel == o.yvel && flags == o.flags;
    }
};
struct SolveKeyHash {
    size_t operator()(const SolveKey& k) const {
        uint64_t h = static_cast<uint32_t>(k.y);
        h = h * 0x9E3779B97F4A7C15ull ^ static_cast<uint32_t>(k.yvel);
        h = h * 0x9E3779B97F4A7C15ull ^ k.flags;
        return static_cast<size_t>(h ^ (h >> 29));
    }
};

inline SolveKey solveKeyOf(const gdcore::PlayerState& s) {
    SolveKey k;
    k.y = static_cast<int32_t>(std::lround(s.y * 4.0f));       // 0.25 units
    k.yvel = static_cast<int32_t>(std::lround(s.yvel * 20.0)); // 0.05
    k.flags = (s.onGround ? 1u : 0u) | (s.ship ? 2u : 0u) |
              (s.upsideDown ? 4u : 0u) | (s.holding ? 8u : 0u) |
              (s.hasJustHeld ? 16u : 0u) |
              (gdcore::pressableRing(s) >= 0 ? 32u : 0u) |
              (s.ball ? 64u : 0u) | (s.inputHeld ? 128u : 0u) |
              (static_cast<uint32_t>(s.specialDone & 0xffu) << 8) |
              (s.mini ? (1u << 16) : 0u);
    return k;
}

struct SolveResult {
    bool found = false;
    float reachX = 0.0f;         // deepest x any expanded state reached
    int nodes = 0;               // states expanded
    std::vector<uint8_t> holds;  // per-tick holds from the start state
};

// Search from `start` (at sim tick startTick) for a decision sequence that
// reaches x >= targetX alive (the level win counts too). maxNodes bounds
// the total work; maxStates the per-tick-bucket population.
inline SolveResult solveMacro(const gdcore::PlayerState& start, int startTick,
                              float targetX, const gdcore::PhysicsConfig& c,
                              const gdcore::LevelGeometry& g,
                              const EnvConfig& ecfg, float endX,
                              int maxNodes = 200000, int maxStates = 4096) {
    struct Node {
        gdcore::PlayerState s;
        int tick;
        int parent;
        int presses; // action-1 decisions on the path (tie-breaker)
        std::vector<uint8_t> holds; // ticks of the step that produced it
    };
    SolveResult res;
    std::vector<Node> nodes;
    nodes.reserve(1024);
    nodes.push_back({start, startTick, -1, 0, {}});
    std::map<int, std::vector<int>> open; // tick -> node indices
    open[startTick].push_back(0);
    std::unordered_set<SolveKey, SolveKeyHash> seen;
    int winIdx = -1;
    EnvInstance e;
    while (!open.empty() && winIdx < 0) {
        auto it = open.begin();
        std::vector<int> bucket = std::move(it->second);
        open.erase(it);
        // Every path to a given x takes the same ticks, so the search is
        // indifferent between them; expand fewest-press states first so
        // the returned demo is the laziest one (dedup keeps the first
        // arrival) — demonstrations then differ from the policy's own
        // behavior only where a press actually matters.
        std::stable_sort(bucket.begin(), bucket.end(), [&](int a, int b) {
            return nodes[a].presses < nodes[b].presses;
        });
        if (static_cast<int>(bucket.size()) > maxStates) {
            std::vector<int> keep(bucket.begin(), bucket.begin() + maxStates / 2);
            std::vector<int> rest(bucket.begin() + maxStates / 2, bucket.end());
            std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) {
                return SolveKeyHash{}(solveKeyOf(nodes[a].s)) <
                       SolveKeyHash{}(solveKeyOf(nodes[b].s));
            });
            rest.resize(maxStates - keep.size());
            keep.insert(keep.end(), rest.begin(), rest.end());
            bucket.swap(keep);
        }
        for (int ni : bucket) {
            for (int a = 0; a < 2 && winIdx < 0; ++a) {
                e.s = nodes[ni].s;
                e.tick = nodes[ni].tick;
                e.holds.clear();
                StepResult r = e.step(a != 0, c, g, ecfg, endX);
                ++res.nodes;
                if (e.s.dead) continue;
                res.reachX = std::max(res.reachX, e.s.x);
                const bool win = r.won || e.s.x >= targetX;
                if (!win) {
                    if (r.truncated) continue;
                    // Dedup per tick bucket: (tick, key). The set is shared
                    // across buckets, so fold the tick into the key.
                    SolveKey k = solveKeyOf(e.s);
                    k.flags ^= static_cast<uint32_t>(e.tick) * 0x9E3779B1u;
                    if (!seen.insert(k).second) continue;
                }
                nodes.push_back({e.s, e.tick, ni, nodes[ni].presses + a,
                                 e.holds});
                if (win) winIdx = static_cast<int>(nodes.size()) - 1;
                else open[e.tick].push_back(static_cast<int>(nodes.size()) - 1);
            }
            if (winIdx >= 0 || res.nodes >= maxNodes) break;
        }
        if (res.nodes >= maxNodes) break;
    }
    if (winIdx < 0) return res;
    std::vector<const std::vector<uint8_t>*> chain;
    for (int i = winIdx; i > 0; i = nodes[i].parent) chain.push_back(&nodes[i].holds);
    for (auto ci = chain.rbegin(); ci != chain.rend(); ++ci)
        res.holds.insert(res.holds.end(), (*ci)->begin(), (*ci)->end());
    res.found = true;
    return res;
}

} // namespace gdenv
