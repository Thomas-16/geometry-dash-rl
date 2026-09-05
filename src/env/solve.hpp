#pragma once

#include "core/physics.hpp"
#include "env/episode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <unordered_set>
#include <vector>

// Breadth-first search in the POLICY'S action space: an exact planner over
// EnvInstance::act (macro or precision, whichever the env runs), so a found
// path is a stream of decisions the policy could have taken and its
// recorded per-tick holds replay bit-exactly through gdenv_replay.
// tools/solve.cpp searches per tick (its output is the deploy-format
// existence proof); this one manufactures demonstrations at walls the
// on-policy sampler cannot cross (a wall needing ~15 consecutive no-flips
// against a saturated prior is a 1e-6 event for the sampler and a few
// thousand node expansions here).
//
// States are expanded in tick order (x is a function of the tick, so every
// state in a bucket shares x). Duplicates are dropped by FIELDWISE EXACT
// equality of the transition state (tick, position, velocity, every mode /
// gravity / size flag, activation records, ring contact lists, band floor,
// landing memory, the flip-grace counter saturated at the grace it gates;
// hash collisions resolve through the equality check) — the old quantized
// key (0.25-unit y, 0.05 yvel, a subset of the flags) silently merged
// states that later diverged. Buckets over maxStates are thinned
// deterministically while preserving diversity across height, vertical
// velocity and mode (round-robin over strata, cheapest first inside each);
// pruning, the expansion budget and the memory budget are reported apart
// from an exhausted search, and search failure is never a proof that a
// section is impossible.
//
// Every returned path is replayed through the executor from the start
// state before it is admitted (the replay also produces the raw holds).
namespace gdenv {

enum SolveStatus : int {
    kSolveFound = 0,
    kSolveExhausted = 1,   // open set empty, no budget hit, no pruning
    kSolveNodeBudget = 2,  // expansion budget exhausted
    kSolveMemoryBudget = 3,// stored-node budget exhausted
    kSolveVerifyFailed = 4,// found path did not replay (never expected)
    kSolvePrunedOut = 5,   // open set empty but states were pruned
};

struct SolveResult {
    bool found = false;
    int status = kSolveExhausted;
    float reachX = 0.0f;         // deepest x any expanded state reached
    int nodes = 0;               // executor calls (expansions)
    int stored = 0;              // nodes kept in memory
    int pruned = 0;              // states dropped by bucket thinning
    std::vector<uint8_t> holds;      // per-tick holds from the start state
    std::vector<uint8_t> actions;    // per-decision action ids
    std::vector<uint16_t> durations; // ticks per decision
};

namespace detail {

// The exact dedup identity of a state. Everything gdcore::step reads, with
// the two tick counters canonicalized: sinceSwitch is bookkeeping the
// dynamics never read, sinceFlip only matters below the flip grace.
struct ExactKey {
    int32_t tick;
    float x, y;
    double yvel;
    float bandFloor, prevLandX, prevLandY;
    double landOffset;
    int32_t specialCursor;
    uint64_t specialDone;
    int32_t sinceFlip;
    int32_t ringNow[gdcore::PlayerState::kMaxTouchRings];
    int32_t ringPrev[gdcore::PlayerState::kMaxTouchRings];
    int32_t nRingNow, nRingPrev;
    uint32_t flags;

    bool operator==(const ExactKey& o) const {
        if (tick != o.tick || x != o.x || y != o.y || yvel != o.yvel ||
            bandFloor != o.bandFloor || specialCursor != o.specialCursor ||
            specialDone != o.specialDone || sinceFlip != o.sinceFlip ||
            nRingNow != o.nRingNow || nRingPrev != o.nRingPrev ||
            flags != o.flags)
            return false;
        for (int i = 0; i < nRingNow; ++i)
            if (ringNow[i] != o.ringNow[i]) return false;
        for (int i = 0; i < nRingPrev; ++i)
            if (ringPrev[i] != o.ringPrev[i]) return false;
        if (flags & 1u) { // hasPrevLand
            if (prevLandX != o.prevLandX || prevLandY != o.prevLandY ||
                landOffset != o.landOffset)
                return false;
        }
        return true;
    }
};

inline ExactKey exactKeyOf(const gdcore::PlayerState& s, int tick) {
    ExactKey k{};
    k.tick = tick;
    k.x = s.x;
    k.y = s.y;
    k.yvel = s.yvel;
    k.bandFloor = s.bandFloor;
    k.prevLandX = s.hasPrevLand ? s.prevLandX : 0.0f;
    k.prevLandY = s.hasPrevLand ? s.prevLandY : 0.0f;
    k.landOffset = s.hasPrevLand ? s.landOffset : 0.0;
    k.specialCursor = s.specialCursor;
    k.specialDone = s.specialDone;
    k.sinceFlip = std::min(s.sinceFlip, gdcore::pad::kFlipFloorGraceTicks);
    for (int i = 0; i < gdcore::PlayerState::kMaxTouchRings; ++i) {
        k.ringNow[i] = i < s.nRingNow ? s.ringNow[i] : 0;
        k.ringPrev[i] = i < s.nRingPrev ? s.ringPrev[i] : 0;
    }
    k.nRingNow = s.nRingNow;
    k.nRingPrev = s.nRingPrev;
    k.flags = (s.hasPrevLand ? 1u : 0u) | (s.onGround ? 2u : 0u) |
              (s.holding ? 4u : 0u) | (s.inputHeld ? 8u : 0u) |
              (s.dead ? 16u : 0u) | (s.ship ? 32u : 0u) | (s.ball ? 64u : 0u) |
              (s.bird ? 128u : 0u) | (s.upsideDown ? 256u : 0u) |
              (s.mini ? 512u : 0u) | (s.rectMini ? 1024u : 0u) |
              (s.overMaxY ? 2048u : 0u) | (s.hasJustHeld ? 4096u : 0u);
    return k;
}

struct ExactKeyHash {
    size_t operator()(const ExactKey& k) const {
        uint64_t h = 0xcbf29ce484222325ull;
        auto mix = [&](const void* p, size_t n) {
            const unsigned char* b = static_cast<const unsigned char*>(p);
            for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ull; }
        };
        mix(&k.tick, sizeof k.tick);
        mix(&k.x, sizeof k.x);
        mix(&k.y, sizeof k.y);
        mix(&k.yvel, sizeof k.yvel);
        mix(&k.bandFloor, sizeof k.bandFloor);
        mix(&k.specialCursor, sizeof k.specialCursor);
        mix(&k.specialDone, sizeof k.specialDone);
        mix(&k.sinceFlip, sizeof k.sinceFlip);
        mix(&k.flags, sizeof k.flags);
        mix(&k.nRingNow, sizeof k.nRingNow);
        mix(&k.nRingPrev, sizeof k.nRingPrev);
        for (int i = 0; i < k.nRingNow; ++i) mix(&k.ringNow[i], sizeof(int32_t));
        for (int i = 0; i < k.nRingPrev; ++i) mix(&k.ringPrev[i], sizeof(int32_t));
        if (k.flags & 1u) {
            mix(&k.prevLandX, sizeof k.prevLandX);
            mix(&k.prevLandY, sizeof k.prevLandY);
            mix(&k.landOffset, sizeof k.landOffset);
        }
        return static_cast<size_t>(h ^ (h >> 31));
    }
};

// Diversity stratum for bucket thinning: 30-unit height band, yvel sign,
// mode, gravity, grounded.
inline uint32_t stratumOf(const gdcore::PlayerState& s) {
    const uint32_t band = static_cast<uint32_t>(std::max(0.0f, s.y) / 30.0f) & 0xffu;
    const uint32_t vs = s.yvel > 0.0 ? 2u : s.yvel < 0.0 ? 1u : 0u;
    return band | (vs << 8) | (static_cast<uint32_t>(modeOf(s)) << 10) |
           (s.upsideDown ? 1u << 12 : 0u) | (s.onGround ? 1u << 13 : 0u) |
           (s.mini ? 1u << 14 : 0u);
}

} // namespace detail

// Exact identity hash of a state at a tick (exposed for tests: two states
// that differ in any dynamics-relevant field must hash / compare apart).
inline uint64_t exactStateHash(const gdcore::PlayerState& s, int tick) {
    return static_cast<uint64_t>(detail::ExactKeyHash{}(detail::exactKeyOf(s, tick)));
}
inline bool exactStateEqual(const gdcore::PlayerState& a, int ta,
                            const gdcore::PlayerState& b, int tb) {
    return detail::exactKeyOf(a, ta) == detail::exactKeyOf(b, tb);
}

// Search from `start` (at sim tick startTick) for a decision sequence that
// reaches x >= targetX alive (the level win counts too). maxNodes bounds
// the expansions, maxStored the nodes kept in memory, maxStates the
// per-tick-bucket population.
inline SolveResult solveMacro(const gdcore::PlayerState& start, int startTick,
                              float targetX, const gdcore::PhysicsConfig& c,
                              const gdcore::LevelGeometry& g,
                              const EnvConfig& ecfg, float endX,
                              int maxNodes = 200000, int maxStates = 4096,
                              int maxStored = 0) {
    struct Node {
        gdcore::PlayerState s;
        int tick;
        int parent;
        uint8_t action;
        uint16_t dur;
        int cost; // hold decisions on the path (tie-breaker: laziest first)
    };
    if (maxStored <= 0) maxStored = std::max(maxNodes, 1024);
    SolveResult res;
    std::vector<Node> nodes;
    nodes.reserve(1024);
    nodes.push_back({start, startTick, -1, 0, 0, 0});
    std::map<int, std::vector<int>> open; // tick -> node indices
    open[startTick].push_back(0);
    std::unordered_set<detail::ExactKey, detail::ExactKeyHash> seen;
    seen.insert(detail::exactKeyOf(start, startTick));
    int winIdx = -1;
    bool nodeBudget = false, memBudget = false;
    EnvInstance e;
    const int mode = ecfg.actionMode;
    const int nAct = numActions(mode);
    // Expansion order: long / lazy actions first so the first arrival (the
    // one dedup keeps) is the canonical lazy path; one-tick actions add
    // precision only where a longer one does not reach the same state.
    static const int kPrecOrder[kNumPrecisionActions] = {kR4, kR2, kR1, kH4,
                                                         kH2, kH1, kHop};
    uint8_t mask[kNumPrecisionActions];
    while (!open.empty() && winIdx < 0) {
        auto it = open.begin();
        std::vector<int> bucket = std::move(it->second);
        open.erase(it);
        std::stable_sort(bucket.begin(), bucket.end(), [&](int a, int b) {
            return nodes[a].cost < nodes[b].cost;
        });
        if (static_cast<int>(bucket.size()) > maxStates) {
            // Round-robin over strata (cheapest first inside each) until
            // maxStates survive.
            std::map<uint32_t, std::vector<int>> strata;
            for (int ni : bucket) strata[detail::stratumOf(nodes[ni].s)].push_back(ni);
            std::vector<int> keep;
            keep.reserve(static_cast<size_t>(maxStates));
            size_t round = 0;
            while (static_cast<int>(keep.size()) < maxStates) {
                bool any = false;
                for (auto& [k, v] : strata) {
                    if (round < v.size()) {
                        keep.push_back(v[round]);
                        any = true;
                        if (static_cast<int>(keep.size()) >= maxStates) break;
                    }
                }
                if (!any) break;
                ++round;
            }
            res.pruned += static_cast<int>(bucket.size() - keep.size());
            std::stable_sort(keep.begin(), keep.end(), [&](int a, int b) {
                return nodes[a].cost < nodes[b].cost; // lazy-first again
            });
            bucket.swap(keep);
        }
        for (int ni : bucket) {
            actionMask(nodes[ni].s, mode, mask);
            for (int ai = 0; ai < nAct && winIdx < 0; ++ai) {
                const int a = mode == kPrecision ? kPrecOrder[ai] : ai;
                if (!mask[a]) continue;
                e.s = nodes[ni].s;
                e.tick = nodes[ni].tick;
                e.holds.clear();
                e.actions.clear();
                StepResult r = e.act(a, c, g, ecfg, endX);
                ++res.nodes;
                if (e.s.dead) continue;
                res.reachX = std::max(res.reachX, e.s.x);
                const bool win = r.won || e.s.x >= targetX;
                if (!win) {
                    if (r.truncated) continue;
                    if (!seen.insert(detail::exactKeyOf(e.s, e.tick)).second)
                        continue;
                }
                if (static_cast<int>(nodes.size()) >= maxStored) {
                    memBudget = true;
                    break;
                }
                const bool isHold = mode == kPrecision ? (a >= kH1) : (a != 0);
                nodes.push_back({e.s, e.tick, ni, static_cast<uint8_t>(a),
                                 static_cast<uint16_t>(r.ticks),
                                 nodes[ni].cost + (isHold ? 1 : 0)});
                if (win) winIdx = static_cast<int>(nodes.size()) - 1;
                else open[e.tick].push_back(static_cast<int>(nodes.size()) - 1);
            }
            if (winIdx >= 0 || memBudget) break;
            if (res.nodes >= maxNodes) { nodeBudget = true; break; }
        }
        if (nodeBudget || memBudget) break;
    }
    res.stored = static_cast<int>(nodes.size());
    if (winIdx < 0) {
        res.status = memBudget    ? kSolveMemoryBudget
                     : nodeBudget ? kSolveNodeBudget
                     : res.pruned ? kSolvePrunedOut
                                  : kSolveExhausted;
        return res;
    }
    // Chain -> action ids, then replay through the executor from the start
    // state: verification and the raw holds in one pass.
    std::vector<int> chain;
    for (int i = winIdx; i > 0; i = nodes[i].parent) chain.push_back(i);
    std::reverse(chain.begin(), chain.end());
    e.reset();
    e.s = start;
    e.tick = startTick;
    for (int i : chain) {
        const int a = nodes[i].action;
        StepResult r = e.act(a, c, g, ecfg, endX);
        res.actions.push_back(static_cast<uint8_t>(a));
        res.durations.push_back(static_cast<uint16_t>(r.ticks));
        if (r.ticks != nodes[i].dur || e.s.dead) {
            res.status = kSolveVerifyFailed;
            res.actions.clear();
            res.durations.clear();
            return res;
        }
    }
    if (!(e.s.x >= targetX || e.s.x >= endX)) {
        res.status = kSolveVerifyFailed;
        res.actions.clear();
        res.durations.clear();
        return res;
    }
    res.holds = e.holds;
    res.found = true;
    res.status = kSolveFound;
    return res;
}

// Translate a per-tick hold sequence into a precision decision stream that
// reproduces it EXACTLY through the executor: HOP where the grounded hop
// macro yields the same ticks, otherwise release / hold runs of 4, 2, 1
// (an interrupt cutting a run short is harmless — the consumed ticks all
// carried the same bit). Every hold pattern is expressible with the raw
// actions, so this always succeeds; the executor's own record is compared
// tick by tick as it goes. Returns the decisions and their durations.
inline bool holdsToPrecision(const gdcore::PlayerState& start, int startTick,
                             const std::vector<uint8_t>& holds,
                             const gdcore::PhysicsConfig& c,
                             const gdcore::LevelGeometry& g,
                             const EnvConfig& ecfgPrecision, float endX,
                             std::vector<uint8_t>& actions,
                             std::vector<uint16_t>& durations) {
    actions.clear();
    durations.clear();
    EnvInstance e;
    e.reset();
    e.s = start;
    e.tick = startTick;
    const int64_t len = static_cast<int64_t>(holds.size());
    int64_t cursor = 0;
    uint8_t mask[kNumPrecisionActions];
    auto matches = [&](const EnvInstance& t, int64_t from, int n) {
        if (from + n > len) return false;
        for (int i = 0; i < n; ++i)
            if (t.holds[t.holds.size() - n + i] != holds[from + i]) return false;
        return true;
    };
    while (cursor < len) {
        const bool bit = holds[cursor] != 0;
        actionMask(e.s, kPrecision, mask);
        int chosen = -1;
        StepResult r;
        if (bit && mask[kHop]) {
            EnvInstance trial = e;
            StepResult tr = trial.act(kHop, c, g, ecfgPrecision, endX);
            if (matches(trial, cursor, tr.ticks)) {
                e = trial;
                r = tr;
                chosen = kHop;
            }
        }
        if (chosen < 0) {
            int run = 1;
            while (cursor + run < len && run < 4 && (holds[cursor + run] != 0) == bit)
                ++run;
            const int n = run >= 4 ? 4 : run >= 2 ? 2 : 1;
            chosen = bit ? (n == 4 ? kH4 : n == 2 ? kH2 : kH1)
                         : (n == 4 ? kR4 : n == 2 ? kR2 : kR1);
            r = e.act(chosen, c, g, ecfgPrecision, endX);
            if (!matches(e, cursor, r.ticks)) return false;
        }
        actions.push_back(static_cast<uint8_t>(chosen));
        durations.push_back(static_cast<uint16_t>(r.ticks));
        cursor += r.ticks;
        if (r.terminated || r.truncated) break;
    }
    return cursor >= len || e.s.x >= endX;
}

// The planner entry point for either action mode. Precision mode searches
// in two stages: first the macro space (walk / hop: a hop is one decision,
// so the graph is tiny — Stereo Madness' first 800 units take ~1k
// expansions) and translates the result into precision decisions; only if
// that fails within the budget does it search the precision space itself
// (6 raw actions per tick — the same 800 units cost ~200k expansions),
// which can express the one-tick timings the macro space cannot. Both
// results are expressible by the precision policy and both are
// executor-verified. `stage` reports which one produced the path (1 / 2).
inline SolveResult solveAny(const gdcore::PlayerState& start, int startTick,
                            float targetX, const gdcore::PhysicsConfig& c,
                            const gdcore::LevelGeometry& g,
                            const EnvConfig& ecfg, float endX, int maxNodes,
                            int maxStates, int& stage) {
    stage = 1;
    if (ecfg.actionMode != kPrecision)
        return solveMacro(start, startTick, targetX, c, g, ecfg, endX,
                          maxNodes, maxStates);
    EnvConfig macro = ecfg;
    macro.actionMode = kMacro;
    SolveResult r1 = solveMacro(start, startTick, targetX, c, g, macro, endX,
                                maxNodes, maxStates);
    if (r1.found) {
        std::vector<uint8_t> acts;
        std::vector<uint16_t> durs;
        if (holdsToPrecision(start, startTick, r1.holds, c, g, ecfg, endX,
                             acts, durs)) {
            r1.actions = acts;
            r1.durations = durs;
            return r1;
        }
        r1.found = false;
        r1.status = kSolveVerifyFailed;
    }
    stage = 2;
    SolveResult r2 = solveMacro(start, startTick, targetX, c, g, ecfg, endX,
                                maxNodes, maxStates);
    r2.nodes += r1.nodes;
    r2.reachX = std::max(r2.reachX, r1.reachX);
    return r2;
}

} // namespace gdenv
