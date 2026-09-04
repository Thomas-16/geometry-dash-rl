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
// (Back On Track: GD_LEVEL_PATH=<install>/Resources/levels/2.txt; levels
// 3-5 likewise by file stem).

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
    // m_vehicleSize != 1 (the `mini` column, mods built from 2026-09-04
    // on; -1 when the trace predates it and the sim's flag is not checked)
    int mini;
};

// Index of a named column in the CSV header row, or -1.
int columnIndex(const char* header, const char* name) {
    int idx = 0;
    const char* p = header;
    const size_t n = std::strlen(name);
    while (*p) {
        const char* e = p;
        while (*e && *e != ',' && *e != '\n' && *e != '\r') ++e;
        if ((size_t)(e - p) == n && std::strncmp(p, name, n) == 0) return idx;
        if (*e != ',') break;
        p = e + 1;
        ++idx;
    }
    return -1;
}

// The idx-th comma-separated field of a line as an int (-1 if absent).
int intField(const char* line, int idx) {
    if (idx < 0) return -1;
    const char* p = line;
    for (int i = 0; i < idx; ++i) {
        p = std::strchr(p, ',');
        if (!p) return -1;
        ++p;
    }
    return std::atoi(p);
}

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
    // Trailing columns are looked up by name so the two mods can append
    // diagnostics in different orders.
    const int miniCol = columnIndex(line, "mini");
    while (std::fgets(line, sizeof line, f)) {
        // src,tick,x,y,yvel,holding,onground,upsidedown,ship,dead,speed,dt,halftick,lasttick,nodex,realx,realy,totaltime[,...]
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
                           dead, (float)realx, (float)realy,
                           intField(line, miniCol)});
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
    s.inputHeld = s.holding;
    // The `ship` column is the mode: 0 cube, 1 ship, 2 ball, 3 UFO
    // (trace-logger / replay-mod builds from 2026-09-02 on, UFO from
    // 2026-09-04; older traces only have 0/1).
    s.ship = rows[0].ship == 1;
    s.ball = rows[0].ship == 2;
    s.bird = rows[0].ship == 3;
    s.upsideDown = rows[0].upsidedown != 0;
    s.mini = rows[0].mini == 1;
    s.rectMini = s.mini;

    // GD_TRACE_DEBUG=<from>:<to> prints the sim state for ticks in that
    // range (before the comparison) plus the solids/hazards within 60 units
    // of the player's x at the first tick of the window.
    uint64_t dbgFrom = 0, dbgTo = 0;
    if (const char* d = std::getenv("GD_TRACE_DEBUG")) {
        unsigned long long a = 0, b = 0;
        if (std::sscanf(d, "%llu:%llu", &a, &b) == 2) { dbgFrom = a; dbgTo = b; }
    }
    bool dbgGeomPrinted = false;

    uint64_t matched = 0;
    // Secondary state that positions alone do not pin down: m_isOnGround and
    // the post-input yvel. Mismatches are reported (not fatal) so a run that
    // is position-exact but wrong about the flag still shows up.
    uint64_t ogMismatch = 0, ogFirst = 0, yvMismatch = 0, yvFirst = 0;
    auto secondary = [&]() {
        if (!ogMismatch && !yvMismatch) return std::string();
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      " [onground mismatches: %llu (first tick %llu); yvel mismatches: %llu (first tick %llu)]",
                      (unsigned long long)ogMismatch, (unsigned long long)ogFirst,
                      (unsigned long long)yvMismatch, (unsigned long long)yvFirst);
        return std::string(buf);
    };
    for (size_t i = 1; i < rows.size(); ++i) {
        const Row& r = rows[i];
        if (dbgTo && r.tick >= dbgFrom && r.tick <= dbgTo) {
            if (!dbgGeomPrinted) {
                dbgGeomPrinted = true;
                auto dumpBoxes = [&](const char* name, const std::vector<gdcore::Box>& v) {
                    for (const auto& b : v)
                        if (b.right >= s.x - 60.0f && b.left <= s.x + 60.0f)
                            std::printf("  %s [%.9g,%.9g]x[%.9g,%.9g] order=%d\n",
                                        name, b.left, b.right, b.bottom, b.top, b.order);
                };
                std::printf("geometry near x=%.9g:\n", s.x);
                dumpBoxes("solid ", geom.solids);
                dumpBoxes("hazard", geom.hazards);
            }
        }
        // Past the level's last object the real game hands the player to the
        // end-of-level auto-walk (observed trigger: last object + ~40 units;
        // physics freezes and yvel stops updating). In gameplay x advances
        // every tick, so the first frozen-x row past the last object marks
        // the takeover exactly; comparison must be bit-exact until then.
        if (r.realx > maxObjX && r.realx == rows[i - 1].realx && !r.dead) {
            std::printf(
                "%s: MATCHED all %llu gameplay ticks — LEVEL COMPLETE "
                "(end sequence from tick %llu, x=%.9g)%s\n",
                path, (unsigned long long)matched,
                (unsigned long long)r.tick, r.realx, secondary().c_str());
            return 0;
        }
        gdcore::step(s, cfg, geom, r.holding != 0);
        if (dbgTo && r.tick >= dbgFrom && r.tick <= dbgTo)
            std::printf("t=%llu hold=%d | sim x=%.9g y=%.9g yvel=%.17g og=%d ud=%d hjh=%d ih=%d mh=%d mode=%s%s dead=%d"
                        " | real x=%.9g y=%.9g yvel=%.17g og=%d mini=%d\n",
                        (unsigned long long)r.tick, r.holding, s.x, s.y, s.yvel,
                        (int)s.onGround, (int)s.upsideDown, (int)s.hasJustHeld,
                        (int)s.inputHeld, (int)s.holding,
                        s.ship ? "ship" : s.ball ? "ball" : s.bird ? "ufo" : "cube",
                        s.mini ? "(mini)" : "", (int)s.dead,
                        r.realx, r.realy, r.yvel, r.onground, r.mini);
        const int simMode = s.ship ? 1 : s.ball ? 2 : s.bird ? 3 : 0;
        if (simMode != r.ship) {
            std::printf(
                "%s: MODE MISMATCH at tick %llu after %llu matched ticks: "
                "sim mode=%d real mode=%d (0 cube, 1 ship, 2 ball, 3 UFO; x=%.9g y=%.9g)\n",
                path, (unsigned long long)r.tick, (unsigned long long)matched,
                simMode, r.ship, s.x, s.y);
            return 1;
        }
        if (r.mini >= 0 && (s.mini ? 1 : 0) != r.mini) {
            std::printf(
                "%s: SIZE MISMATCH at tick %llu after %llu matched ticks: "
                "sim mini=%d real mini=%d (x=%.9g y=%.9g)\n",
                path, (unsigned long long)r.tick, (unsigned long long)matched,
                (int)s.mini, r.mini, s.x, s.y);
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
        if ((s.onGround ? 1 : 0) != r.onground) {
            if (!ogMismatch) ogFirst = r.tick;
            ++ogMismatch;
        }
        if (s.yvel != r.yvel) {
            if (!yvMismatch) yvFirst = r.tick;
            ++yvMismatch;
        }
        ++matched;
        if (r.dead) {
            std::printf("%s: MATCHED all %llu ticks incl. death at tick %llu%s\n",
                path, (unsigned long long)matched, (unsigned long long)r.tick,
                secondary().c_str());
            return 0;
        }
    }
    std::printf("%s: MATCHED all %llu ticks (no death in trace)%s\n",
        path, (unsigned long long)matched, secondary().c_str());
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
