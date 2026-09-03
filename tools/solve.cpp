// solve: sim-only breadth-first input search for an official level.
//
// Advances a deduplicated set of exact PlayerStates one tick at a time,
// branching on hold/release each tick, until some state crosses the level's
// win line (max object x + 40, the env's end). Any input sequence it finds is
// an exact sim trajectory (states are never approximated — the dedup key is
// only used to drop near-duplicates), so replaying it in the real game via
// the Geode replayer is a hard pass/fail test of the replica for levels that
// have no human traces yet (Dry Out, Base After Base).
//
//   solve <level.txt> [solution.json]
//
// Prints the hold ranges in env_check `run` syntax (inclusive sim ticks) and
// writes a gdrl.eval-compatible solution.json when a path is given
// (`python -m gdrl.deploy <json> --level <level.txt>` installs it).
#include "core/geometry.hpp"
#include "core/level.hpp"
#include "core/physics.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

// Bit 15 of a history entry = hold, bits 0..14 = parent index.
constexpr int kMaxStates = 16384;
constexpr int kMaxTicks = 30000;

struct Key {
    int32_t y, yvel;
    uint32_t flags;
    bool operator==(const Key& o) const {
        return y == o.y && yvel == o.yvel && flags == o.flags;
    }
};
struct KeyHash {
    size_t operator()(const Key& k) const {
        uint64_t h = static_cast<uint32_t>(k.y);
        h = h * 0x9E3779B97F4A7C15ull ^ static_cast<uint32_t>(k.yvel);
        h = h * 0x9E3779B97F4A7C15ull ^ k.flags;
        return static_cast<size_t>(h ^ (h >> 29));
    }
};

Key keyOf(const gdcore::PlayerState& s) {
    Key k;
    k.y = static_cast<int32_t>(std::lround(s.y * 4.0f));         // 0.25 units
    k.yvel = static_cast<int32_t>(std::lround(s.yvel * 20.0));   // 0.05
    k.flags = (s.onGround ? 1u : 0u) | (s.ship ? 2u : 0u) |
              (s.upsideDown ? 4u : 0u) | (s.holding ? 8u : 0u) |
              (s.hasJustHeld ? 16u : 0u) |
              (gdcore::pressableRing(s) >= 0 ? 32u : 0u) |
              (s.ball ? 64u : 0u) | (s.inputHeld ? 128u : 0u) |
              (static_cast<uint32_t>(s.specialDone & 0xffu) << 8);
    return k;
}

std::string levelIdOf(const std::string& path) {
    size_t a = path.find_last_of("/\\");
    std::string stem = path.substr(a == std::string::npos ? 0 : a + 1);
    size_t dot = stem.find('.');
    return dot == std::string::npos ? stem : stem.substr(0, dot);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: solve <level.txt> [solution.json]\n");
        return 2;
    }
    const std::string levelPath = argv[1];
    gd::Level level;
    try {
        level = gd::Level::loadFromFile(levelPath);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed to load level %s: %s\n", argv[1], e.what());
        return 1;
    }
    gdcore::LevelGeometry geom = gdcore::LevelGeometry::build(level);
    gdcore::PhysicsConfig cfg = gdcore::stereoMadnessCube();
    float maxObjX = 0.0f;
    for (const auto& o : level.objects) maxObjX = std::max(maxObjX, o.x);
    const float endX = maxObjX + 40.0f;

    std::vector<gdcore::PlayerState> cur(1), next;
    std::vector<std::vector<uint16_t>> hist; // hist[t][i] = parent|hold<<15
    hist.reserve(kMaxTicks);
    std::unordered_set<Key, KeyHash> seen;
    float bestX = 0.0f;
    int winTick = -1, winIdx = -1;
    auto t0 = std::chrono::steady_clock::now();

    for (int tick = 0; tick < kMaxTicks && winTick < 0; ++tick) {
        next.clear();
        seen.clear();
        std::vector<uint16_t> h;
        h.reserve(cur.size() * 2);
        for (size_t i = 0; i < cur.size(); ++i) {
            for (int hold = 0; hold < 2; ++hold) {
                gdcore::PlayerState s = cur[i];
                gdcore::step(s, cfg, geom, hold != 0);
                if (s.dead) continue;
                const bool win = s.x >= endX;
                if (!win && !seen.insert(keyOf(s)).second) continue;
                if (win) {
                    winTick = tick;
                    winIdx = static_cast<int>(next.size());
                }
                next.push_back(s);
                h.push_back(static_cast<uint16_t>(i) |
                            static_cast<uint16_t>(hold << 15));
                if (win) break;
            }
            if (winTick >= 0) break;
        }
        if (next.empty()) {
            std::printf("dead end at tick %d, x=%.2f (%.1f%%) — no surviving "
                        "state\n", tick + 1, cur[0].x + cfg.dx,
                        100.0 * (cur[0].x + cfg.dx) / endX);
            return 1;
        }
        if (winTick < 0 && static_cast<int>(next.size()) > kMaxStates) {
            // deterministic thinning: keep a hash-spread subset
            std::vector<size_t> idx(next.size());
            for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
            std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                return KeyHash{}(keyOf(next[a])) < KeyHash{}(keyOf(next[b]));
            });
            idx.resize(kMaxStates);
            std::sort(idx.begin(), idx.end());
            std::vector<gdcore::PlayerState> n2;
            std::vector<uint16_t> h2;
            for (size_t i : idx) { n2.push_back(next[i]); h2.push_back(h[i]); }
            next.swap(n2);
            h.swap(h2);
        }
        hist.push_back(std::move(h));
        cur.swap(next);
        bestX = cur[0].x;
        if ((tick + 1) % 2000 == 0)
            std::printf("tick %5d x=%8.1f (%5.1f%%) states=%zu\n", tick + 1,
                        bestX, 100.0 * bestX / endX, cur.size());
    }
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (winTick < 0) {
        std::printf("no win within %d ticks (x=%.1f)\n", kMaxTicks, bestX);
        return 1;
    }
    // reconstruct holds: hist[t][i] describes the state at index i after step t
    const int nTicks = winTick + 1;
    std::vector<uint8_t> holds(static_cast<size_t>(nTicks));
    int idx = winIdx;
    for (int t = winTick; t >= 0; --t) {
        uint16_t e = hist[static_cast<size_t>(t)][static_cast<size_t>(idx)];
        holds[static_cast<size_t>(t)] = (e >> 15) & 1;
        idx = e & 0x7fff;
    }
    // exact re-simulation as a self-check
    gdcore::PlayerState s;
    for (int t = 0; t < nTicks; ++t) gdcore::step(s, cfg, geom, holds[static_cast<size_t>(t)]);
    if (s.dead || s.x < endX) {
        std::printf("INTERNAL: replay mismatch (dead=%d x=%.2f)\n", s.dead, s.x);
        return 1;
    }
    std::vector<std::pair<int, int>> ranges; // [start, end) sim ticks
    for (int t = 0; t < nTicks; ++t) {
        if (!holds[static_cast<size_t>(t)]) continue;
        if (!ranges.empty() && ranges.back().second == t) ranges.back().second = t + 1;
        else ranges.push_back({t, t + 1});
    }
    std::printf("WIN at tick %d, x=%.2f (endX %.1f), %zu hold ranges, %.1fs\n",
                nTicks, s.x, endX, ranges.size(), dt);
    std::printf("env_check run \"");
    for (size_t i = 0; i < ranges.size(); ++i)
        std::printf("%s%d-%d", i ? "," : "", ranges[i].first, ranges[i].second - 1);
    std::printf("\"\n");
    if (argc > 2) {
        std::ofstream f(argv[2]);
        f << "{\n  \"tps\": 240,\n  \"level_id\": " << levelIdOf(levelPath)
          << ",\n  \"level_end_x\": " << endX << ",\n  \"n_ticks\": " << nTicks
          << ",\n  \"deterministic_policy\": false,\n  \"source\": \"solve\","
          << "\n  \"holds\": [";
        for (size_t i = 0; i < ranges.size(); ++i)
            f << (i ? ", " : "") << "[" << ranges[i].first << ", "
              << ranges[i].second << "]";
        f << "]\n}\n";
        std::printf("wrote %s\n", argv[2]);
    }
    return 0;
}
