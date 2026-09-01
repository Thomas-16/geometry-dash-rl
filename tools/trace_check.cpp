// Regression check: replay a recorded trace's inputs through gdcore physics
// and diff float32 positions tick by tick, including collisions with level
// geometry (block landings, spike/pit deaths, block-side deaths).
//
// Usage: trace_check <trace.csv> [<trace.csv> ...]
// Reads format=3 traces from the fang.trace-logger Geode mod, uses only the
// src=0 (end-of-tick) rows. Simulation starts from the tick-0 row and each
// subsequent row is compared bit-exactly (x, y as float32). The sim must
// also die on exactly the tick the real game died.
//
// Level geometry comes from the real install at runtime (never committed);
// override the level path with the GD_LEVEL_PATH environment variable
// (Back On Track: GD_LEVEL_PATH=<install>/Resources/levels/2.txt).

#include "core/geometry.hpp"
#include "core/level.hpp"
#include "core/physics.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr const char* kDefaultLevel =
    "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels/1.txt";

struct Row {
    uint64_t tick;
    double yvel;
    int holding, onground, upsidedown, ship, dead;
    float realx, realy;
};

std::vector<Row> loadTrace(const char* path) {
    std::vector<Row> rows;
    FILE* f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    char line[512];
    if (!std::fgets(line, sizeof line, f) || std::strncmp(line, "format=3", 8) != 0) {
        std::fprintf(stderr, "%s: not a format=3 trace\n", path);
        std::exit(2);
    }
    std::fgets(line, sizeof line, f); // header row
    while (std::fgets(line, sizeof line, f)) {
        // src,tick,x,y,yvel,holding,onground,upsidedown,ship,dead,speed,dt,halftick,lasttick,nodex,realx,realy,totaltime
        int src, holding, onground, upsidedown, ship, dead, halftick, lasttick;
        unsigned long long tick;
        double x, y, yvel, speed, dt, nodex, realx, realy, totaltime;
        int n = std::sscanf(line,
            "%d,%llu,%lf,%lf,%lf,%d,%d,%d,%d,%d,%lf,%lf,%d,%d,%lf,%lf,%lf,%lf",
            &src, &tick, &x, &y, &yvel, &holding, &onground, &upsidedown,
            &ship, &dead, &speed, &dt, &halftick, &lasttick, &nodex, &realx,
            &realy, &totaltime);
        if (n != 18 || src != 0) continue;
        rows.push_back(Row{tick, yvel, holding, onground, upsidedown, ship,
                           dead, (float)realx, (float)realy});
    }
    std::fclose(f);
    return rows;
}

int checkTrace(const char* path, const gdcore::LevelGeometry& geom,
               float maxObjX) {
    auto rows = loadTrace(path);
    if (rows.size() < 2) {
        std::fprintf(stderr, "%s: too few rows\n", path);
        return 1;
    }

    auto cfg = gdcore::stereoMadnessCube();
    gdcore::PlayerState s;
    s.x = rows[0].realx;
    s.y = rows[0].realy;
    s.yvel = rows[0].yvel;
    s.onGround = rows[0].onground != 0;
    s.holding = rows[0].holding != 0;
    s.ship = rows[0].ship != 0;
    s.upsideDown = rows[0].upsidedown != 0;

    uint64_t matched = 0;
    for (size_t i = 1; i < rows.size(); ++i) {
        const Row& r = rows[i];
        // Past the level's last object the real game hands the player to the
        // end-of-level auto-walk (observed trigger: last object + ~40 units;
        // physics freezes and yvel stops updating). In gameplay x advances
        // every tick, so the first frozen-x row past the last object marks
        // the takeover exactly; comparison must be bit-exact until then.
        if (r.realx > maxObjX && r.realx == rows[i - 1].realx && !r.dead) {
            std::printf(
                "%s: MATCHED all %llu gameplay ticks — LEVEL COMPLETE "
                "(end sequence from tick %llu, x=%.9g)\n",
                path, (unsigned long long)matched,
                (unsigned long long)r.tick, r.realx);
            return 0;
        }
        gdcore::step(s, cfg, geom, r.holding != 0);
        if (s.ship != (r.ship != 0)) {
            std::printf(
                "%s: MODE MISMATCH at tick %llu after %llu matched ticks: "
                "sim ship=%d real ship=%d (x=%.9g y=%.9g)\n",
                path, (unsigned long long)r.tick, (unsigned long long)matched,
                (int)s.ship, r.ship, s.x, s.y);
            return 1;
        }
        if (s.upsideDown != (r.upsidedown != 0)) {
            std::printf(
                "%s: GRAVITY MISMATCH at tick %llu after %llu matched ticks: "
                "sim upsidedown=%d real upsidedown=%d (x=%.9g y=%.9g)\n",
                path, (unsigned long long)r.tick, (unsigned long long)matched,
                (int)s.upsideDown, r.upsidedown, s.x, s.y);
            return 1;
        }
        if (s.x != r.realx || s.y != r.realy) {
            std::printf(
                "%s: DIVERGED at tick %llu after %llu matched ticks\n"
                "  sim  x=%.9g y=%.9g yvel=%.17g dead=%d\n"
                "  real x=%.9g y=%.9g yvel=%.17g (holding=%d onground=%d dead=%d)\n",
                path, (unsigned long long)r.tick, (unsigned long long)matched,
                s.x, s.y, s.yvel, (int)s.dead, r.realx, r.realy, r.yvel,
                r.holding, r.onground, r.dead);
            return 1;
        }
        if (s.dead != (r.dead != 0)) {
            std::printf(
                "%s: DEATH MISMATCH at tick %llu after %llu matched ticks: "
                "sim dead=%d real dead=%d (x=%.9g y=%.9g)\n",
                path, (unsigned long long)r.tick, (unsigned long long)matched,
                (int)s.dead, r.dead, s.x, s.y);
            return 1;
        }
        ++matched;
        if (r.dead) {
            std::printf("%s: MATCHED all %llu ticks incl. death at tick %llu\n",
                path, (unsigned long long)matched, (unsigned long long)r.tick);
            return 0;
        }
    }
    std::printf("%s: MATCHED all %llu ticks (no death in trace)\n",
        path, (unsigned long long)matched);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: trace_check <trace.csv>...\n");
        return 2;
    }
    const char* levelPath = std::getenv("GD_LEVEL_PATH");
    if (!levelPath) levelPath = kDefaultLevel;
    gdcore::LevelGeometry geom;
    float maxObjX = 0.0f;
    try {
        gd::Level level = gd::Level::loadFromFile(levelPath);
        geom = gdcore::LevelGeometry::build(level);
        for (const auto& obj : level.objects) maxObjX = std::max(maxObjX, obj.x);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed to load level %s: %s\n", levelPath, e.what());
        return 2;
    }
    int rc = 0;
    for (int i = 1; i < argc; ++i) rc |= checkTrace(argv[i], geom, maxObjX);
    return rc;
}
