// Headless checks for the RL env layer: ASCII observation dumps, scripted
// runs, and determinism verification. Links gdcore + env headers directly
// (not through the DLL).
//
//   env_check obs <x> [y]        print the obs grid/scalars at a position
//   env_check run <spec>         run holds like "40-60,97-99" (tick ranges,
//                                inclusive), print death/win tick and x
//   env_check det                two identical pseudo-random runs must give
//                                bytewise-identical obs/reward streams
//   env_check trace <spec> [x0 [x1]]
//                                like run, but print per-tick state while
//                                x is inside [x0, x1] (default: everything)
//
// The level comes from GD_LEVEL_PATH (default Stereo Madness); Back On
// Track is <install>/Resources/levels/2.txt, Polargeist 3.txt, Dry Out
// 4.txt, Base After Base 5.txt.
#include "core/level.hpp"
#include "core/physics.hpp"
#include "env/episode.hpp"
#include "env/obs.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* kDefaultLevel =
    "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels/1.txt";

std::string levelPath() {
    if (const char* env = std::getenv("GD_LEVEL_PATH")) return env;
    return kDefaultLevel;
}

void printObs(const float* obs) {
    std::printf("y=%.3f yvel=%.3f ground=%.0f ship=%.0f hold=%.0f "
                "prog=%.4f pdist=%.3f ptype=%.0f\n",
                obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6],
                obs[7]);
    std::printf("haz=(%.3f,%.3f) haz2=(%.3f,%.3f) step=(%.3f,%.3f) "
                "ceil=(%.3f,%.3f) dx=%.4f\n",
                obs[8], obs[9], obs[14], obs[15], obs[10], obs[11], obs[12],
                obs[13], obs[16]);
    std::printf("grav=%.0f sp1=(%.3f,%.3f,%.2f) sp2=(%.3f,%.3f,%.2f) "
                "ringPress=%.0f bufArmed=%.0f\n",
                obs[17], obs[18], obs[19], obs[20], obs[21], obs[22], obs[23],
                obs[24], obs[25]);
    const float* solid = obs + gdenv::kPlayerDim;
    const float* hazard = solid + gdenv::kGridDim;
    // top row first; player column is col 0
    for (int row = gdenv::kGridRows - 1; row >= 0; --row) {
        for (int col = 0; col < gdenv::kGridCols; ++col) {
            float sv = solid[row * gdenv::kGridCols + col];
            float hv = hazard[row * gdenv::kGridCols + col];
            char ch = '.';
            if (hv > 0.5f) ch = 'X';
            else if (hv > 0.0f) ch = 'x';
            else if (sv > 0.5f) ch = '#';
            else if (sv > 0.0f) ch = '+';
            std::putchar(ch);
        }
        std::printf("  row y=[%g,%g)\n",
                    gdenv::kGridYBase + row * gdenv::kCellH,
                    gdenv::kGridYBase + (row + 1) * gdenv::kCellH);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: env_check obs|run|det ...\n");
        return 2;
    }
    gd::Level level = gd::Level::loadFromFile(levelPath());
    gdcore::LevelGeometry geom = gdcore::LevelGeometry::build(level);
    gdcore::PhysicsConfig cfg = gdcore::stereoMadnessCube();
    float maxObjX = 0.0f;
    for (const auto& o : level.objects)
        if (o.x > maxObjX) maxObjX = o.x;
    const float endX = maxObjX + 40.0f;
    std::vector<float> obs(gdenv::kObsDim);

    std::string cmd = argv[1];
    if (cmd == "obs") {
        gdcore::PlayerState s;
        if (argc > 2) s.x = std::strtof(argv[2], nullptr);
        if (argc > 3) s.y = std::strtof(argv[3], nullptr);
        gdenv::buildObs(s, geom, endX, cfg.dx, obs.data());
        std::printf("player x=%.2f y=%.2f  endX=%.2f\n", s.x, s.y, endX);
        printObs(obs.data());
        return 0;
    }

    if (cmd == "run" || cmd == "trace") {
        // parse "a-b,c-d,..." inclusive tick ranges into per-tick holds
        std::vector<std::pair<int, int>> ranges;
        const bool trace = cmd == "trace";
        float x0 = argc > 3 ? std::strtof(argv[3], nullptr) : -1e9f;
        float x1 = argc > 4 ? std::strtof(argv[4], nullptr) : 1e9f;
        if (argc > 2) {
            const char* p = argv[2];
            while (*p) {
                char* q;
                long a = std::strtol(p, &q, 10), b = a;
                if (*q == '-') b = std::strtol(q + 1, &q, 10);
                ranges.push_back({static_cast<int>(a), static_cast<int>(b)});
                p = (*q == ',') ? q + 1 : q;
            }
        }
        // raw per-tick stepping (bypasses the macro-action layer on purpose)
        gdcore::PlayerState s;
        for (int tick = 0; tick < 30000; ++tick) {
            bool hold = false;
            for (auto& r : ranges)
                if (tick >= r.first && tick <= r.second) hold = true;
            gdcore::step(s, cfg, geom, hold);
            if (trace && s.x >= x0 && s.x <= x1)
                std::printf("t=%5d x=%9.3f y=%9.4f yvel=%8.3f%s%s%s%s%s%s\n",
                            tick + 1, s.x, s.y, s.yvel,
                            s.onGround ? " ground" : "",
                            s.ship ? " ship" : "",
                            s.upsideDown ? " FLIPPED" : "",
                            gdcore::pressableRing(s) >= 0 ? " RING" : "",
                            (s.holding && s.hasJustHeld) ? " armed" : "",
                            s.dead ? " DEAD" : "");
            if (s.dead || s.x >= endX) {
                std::printf("%s at tick %d, x=%.2f (%.1f%%)\n",
                            s.dead ? "DEATH" : "WIN", tick + 1, s.x,
                            100.0 * s.x / endX);
                return 0;
            }
        }
        std::printf("no end within 30000 ticks\n");
        return 0;
    }

    if (cmd == "det") {
        auto run = [&](std::vector<float>& stream) {
            gdenv::EnvConfig ecfg; // default action repeat 4
            gdenv::EnvInstance e;
            e.reset();
            uint32_t rng = 12345;
            for (int step = 0; step < 4000; ++step) {
                rng = rng * 1664525u + 1013904223u;
                bool hold = (rng >> 16) & 1;
                gdenv::StepResult r = e.step(hold, cfg, geom, ecfg, endX);
                gdenv::buildObs(e.s, geom, endX, cfg.dx, obs.data());
                stream.insert(stream.end(), obs.begin(), obs.end());
                stream.push_back(r.reward);
                if (r.terminated || r.truncated) e.reset();
            }
        };
        std::vector<float> a, b;
        run(a);
        run(b);
        if (a.size() != b.size() ||
            std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
            std::printf("DETERMINISM FAIL\n");
            return 1;
        }
        std::printf("determinism OK (%zu floats identical)\n", a.size());
        return 0;
    }

    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
