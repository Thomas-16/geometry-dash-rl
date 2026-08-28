// Regression check: replay a recorded trace's inputs through gdcore physics
// and diff float32 positions tick by tick.
//
// Usage: trace_check <trace.csv> [<trace.csv> ...]
// Reads format=3 traces from the fang.trace-logger Geode mod, uses only the
// src=0 (end-of-tick) rows. Simulation starts from the tick-0 row and each
// subsequent row is compared bit-exactly (x, y as float32).
//
// Divergence is expected once non-floor collisions matter (blocks, spikes,
// ship); the tool reports the first mismatching tick and what changed.

#include "core/physics.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Row {
    uint64_t tick;
    double yvel;
    int holding, onground, ship, dead;
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
        rows.push_back(Row{tick, yvel, holding, onground, ship, dead,
                           (float)realx, (float)realy});
    }
    std::fclose(f);
    return rows;
}

int checkTrace(const char* path) {
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

    uint64_t matched = 0;
    for (size_t i = 1; i < rows.size(); ++i) {
        const Row& r = rows[i];
        if (r.dead) {
            std::printf("%s: MATCHED all %llu alive ticks (death at tick %llu)\n",
                path, (unsigned long long)matched, (unsigned long long)r.tick);
            return 0;
        }
        if (r.ship) {
            std::printf("%s: matched %llu ticks up to ship portal at tick %llu\n",
                path, (unsigned long long)matched, (unsigned long long)r.tick);
            return 0;
        }
        gdcore::step(s, cfg, rows[i].holding != 0);
        if (s.x != r.realx || s.y != r.realy) {
            std::printf(
                "%s: DIVERGED at tick %llu after %llu matched ticks\n"
                "  sim  x=%.9g y=%.9g yvel=%.17g\n"
                "  real x=%.9g y=%.9g yvel=%.17g (holding=%d onground=%d)\n",
                path, (unsigned long long)r.tick, (unsigned long long)matched,
                s.x, s.y, s.yvel, r.realx, r.realy, r.yvel, r.holding,
                r.onground);
            return 1;
        }
        ++matched;
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
    int rc = 0;
    for (int i = 1; i < argc; ++i) rc |= checkTrace(argv[i]);
    return rc;
}
