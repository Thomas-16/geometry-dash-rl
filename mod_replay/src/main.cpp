// fang.rl-replay — open-loop replay of an RL-solved input sequence.
//
// Loads <solutions-dir>/<levelID>.json (written by `python -m gdrl.deploy`,
// schema from python/gdrl/eval.py: [start,end) tick ranges of held input) and
// injects the per-tick hold state at the exact site the engine drains real
// input (GJBaseGameLayer::processQueuedButtons), so ordering relative to the
// physics step is identical to a human click. Real tick t == the t-th
// processCommands call after PlayLayer::resetLevel (trace-logger indexing);
// sim tick t == real tick t+1 (see injectForTick).
//
// Also writes a format=3 per-tick trace (same schema as fang.trace-logger) so
// build/trace_check.exe can pinpoint the first divergent tick on a desync.

#include <Geode/Geode.hpp>
#include <Geode/modify/CCKeyboardDispatcher.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/utils/cocos.hpp>
#include <Geode/utils/file.hpp>
#include <matjson.hpp>

#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

using namespace geode::prelude;

namespace {

// ---------------------------------------------------------------- trace rows
// Identical layout/format to mod/src/main.cpp so trace_check reads it as-is.
struct TickRow {
    uint8_t src;
    uint64_t tick;
    float x;
    float y;
    double yVel;
    uint8_t holding;
    uint8_t onGround;
    uint8_t upsideDown;
    uint8_t ship;
    uint8_t dead;
    float speed;
    float dt;
    uint8_t halfTick;
    uint8_t lastTick;
    float nodeX;
    float realX;
    float realY;
    double totalTime;
    uint8_t mini;  // m_vehicleSize != 1 (trace_check reads it by header name)
};

// ------------------------------------------------------------------ solution
struct Solution {
    std::filesystem::path path;
    std::filesystem::file_time_type mtime{};
    int levelId = -1;      // -1 = legacy file without level_id
    int tps = 240;
    float endX = 0.f;
    int nTicks = 0;
    std::vector<uint8_t> holds;  // expanded per-tick bits, size nTicks
};

struct Replay {
    std::optional<Solution> sol;
    bool armed = false;
    bool finished = false;
    bool injectedHold = false;   // last state pushed through handleButton
    bool injecting = false;      // re-entrancy: let our own handleButton through
    bool injectedThisTick = false;
    int pqbCallsThisTick = 0;    // processQueuedButtons calls seen this tick
    int pqbCallsLastTick = 0;
    bool pqbDiagLogged = false;
    bool prehookFallback = false; // inject in processCommands pre-hook instead
    uint64_t tick = 0;
    int attempt = 0;
    std::string outcome;         // "", "win", "death"
    uint64_t outcomeTick = 0;
    float outcomeX = 0.f;
    std::vector<TickRow> rows;
    WeakRef<CCLabelBMFont> hud;
    bool warnedMods = false;

    bool enabled() const { return Mod::get()->getSettingValue<bool>("enabled"); }
    bool blockInput() const { return Mod::get()->getSettingValue<bool>("block-input"); }
    bool logTrace() const { return Mod::get()->getSettingValue<bool>("log-trace"); }
    bool autoRetry() const { return Mod::get()->getSettingValue<bool>("auto-retry"); }

    std::filesystem::path solutionsDir() const {
        auto s = Mod::get()->getSettingValue<std::string>("solutions-dir");
        if (!s.empty()) return std::filesystem::path(s);
        return Mod::get()->getSaveDir() / "solutions";
    }

    std::filesystem::path tracesDir() const {
        auto dir = Mod::get()->getSaveDir() / "traces";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        return dir;
    }

    void flushTrace() {
        if (rows.empty()) return;
        if (logTrace()) {
            auto path = tracesDir() / fmt::format("attempt_{:04}.csv", attempt);
            if (auto* f = std::fopen(path.string().c_str(), "wb")) {
                std::fputs("format=3\n", f);
                std::fputs("src,tick,x,y,yvel,holding,onground,upsidedown,ship,dead,speed,dt,halftick,lasttick,nodex,realx,realy,totaltime,mini\n", f);
                for (auto const& r : rows) {
                    std::fprintf(f, "%u,%llu,%.9g,%.9g,%.17g,%u,%u,%u,%u,%u,%.9g,%.9g,%u,%u,%.9g,%.9g,%.9g,%.17g,%u\n",
                        r.src, (unsigned long long)r.tick, r.x, r.y, r.yVel, r.holding,
                        r.onGround, r.upsideDown, r.ship, r.dead, r.speed,
                        r.dt, r.halfTick, r.lastTick, r.nodeX, r.realX, r.realY, r.totalTime,
                        r.mini);
                }
                std::fclose(f);
                log::info("wrote {} ticks -> {}", rows.size(), path.string());
            }
        }
        rows.clear();
    }

    void writeResult(std::string const& out, uint64_t t, float x) {
        outcome = out;
        outcomeTick = t;
        outcomeX = x;
        auto path = tracesDir() / fmt::format("result_{:04}.json", attempt);
        if (auto* f = std::fopen(path.string().c_str(), "wb")) {
            float endX = sol ? sol->endX : 0.f;
            std::fprintf(f,
                "{\n  \"level_id\": %d,\n  \"outcome\": \"%s\",\n  \"tick\": %llu,\n"
                "  \"x\": %.9g,\n  \"pct\": %.3f,\n  \"solution_path\": \"%s\",\n"
                "  \"n_ticks\": %d\n}\n",
                sol ? sol->levelId : -1, out.c_str(), (unsigned long long)t, x,
                endX > 0 ? 100.0 * x / endX : 0.0,
                sol ? sol->path.generic_string().c_str() : "", sol ? sol->nTicks : 0);
            std::fclose(f);
        }
    }

    void finish(std::string const& out, uint64_t t, float x) {
        if (finished) return;
        finished = true;
        writeResult(out, t, x);
        flushTrace();
        if (out == "death" && !autoRetry()) {
            armed = false;
            log::warn("auto-retry off: disarmed (toggle with F8 or the Enabled setting)");
        }
    }
};

Replay g_rp;

// Parse solution.json -> Solution. Returns error string on failure.
Result<Solution> parseSolution(std::filesystem::path const& path) {
    GEODE_UNWRAP_INTO(auto text, file::readString(path));
    auto parsed = matjson::parse(text);
    if (!parsed) return Err(fmt::format("json parse error: {}", parsed.unwrapErr()));
    auto j = std::move(parsed).unwrap();

    Solution s;
    s.path = path;
    std::error_code ec;
    s.mtime = std::filesystem::last_write_time(path, ec);

    if (j.contains("tps")) {
        GEODE_UNWRAP_INTO(auto tps, j["tps"].asInt());
        s.tps = (int)tps;
    }
    if (s.tps != 240) return Err(fmt::format("tps {} != 240", s.tps));
    if (j.contains("level_id")) {
        GEODE_UNWRAP_INTO(auto lid, j["level_id"].asInt());
        s.levelId = (int)lid;
    }
    if (j.contains("level_end_x")) {
        GEODE_UNWRAP_INTO(auto ex, j["level_end_x"].asDouble());
        s.endX = (float)ex;
    }
    GEODE_UNWRAP_INTO(auto nt, j["n_ticks"].asInt());
    s.nTicks = (int)nt;
    if (s.nTicks <= 0 || s.nTicks > 1 << 20) return Err(fmt::format("bad n_ticks {}", s.nTicks));
    s.holds.assign((size_t)s.nTicks, 0);

    GEODE_UNWRAP_INTO(auto const& ranges, j["holds"].asArray());
    int prevEnd = 0;
    for (auto const& r : ranges) {
        GEODE_UNWRAP_INTO(auto const& pr, r.asArray());
        if (pr.size() != 2) return Err("hold range is not a pair");
        GEODE_UNWRAP_INTO(auto a, pr[0].asInt());
        GEODE_UNWRAP_INTO(auto b, pr[1].asInt());
        if (a < prevEnd || b <= a || b > s.nTicks)
            return Err(fmt::format("bad hold range [{},{}) (n_ticks {})", a, b, s.nTicks));
        for (auto t = a; t < b; ++t) s.holds[(size_t)t] = 1;
        prevEnd = (int)b;
    }
    return Ok(std::move(s));
}

// (Re)load the solution for a level if the file is new or changed.
void loadSolution(int levelId) {
    auto path = g_rp.solutionsDir() / fmt::format("{}.json", levelId);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        if (!g_rp.sol || g_rp.sol->path != path)
            log::warn("no solution for level {} at {}", levelId, path.string());
        g_rp.sol.reset();
        return;
    }
    auto mtime = std::filesystem::last_write_time(path, ec);
    if (g_rp.sol && g_rp.sol->path == path && g_rp.sol->mtime == mtime) return;
    auto res = parseSolution(path);
    if (!res) {
        log::error("failed to load {}: {}", path.string(), res.unwrapErr());
        g_rp.sol.reset();
        return;
    }
    g_rp.sol = std::move(res).unwrap();
    int held = 0;
    for (auto h : g_rp.sol->holds) held += h;
    log::info("loaded solution {} ({} ticks, {} held, end_x {})",
              path.string(), g_rp.sol->nTicks, held, g_rp.sol->endX);
}

void logRow(PlayerObject* p, float dt, bool halfTick, bool lastTick) {
    g_rp.rows.push_back(TickRow{
        0, g_rp.tick,
        p->m_position.x, p->m_position.y, p->m_yVelocity,
        (uint8_t)g_rp.injectedHold, (uint8_t)p->m_isOnGround,
        (uint8_t)p->m_isUpsideDown,
        // mode column ("ship" in the header): 0 cube, 1 ship, 2 ball, 3 UFO
        (uint8_t)(p->m_isShip ? 1 : (p->m_isBall ? 2 : (p->m_isBird ? 3 : 0))),
        (uint8_t)p->m_isDead,
        p->m_playerSpeed, dt, (uint8_t)halfTick, (uint8_t)lastTick,
        p->getPositionX(), p->getRealPosition().x, p->getRealPosition().y,
        p->m_totalTime,
        (uint8_t)(p->m_vehicleSize != 1.0f),
    });
}

void updateHud() {
    auto hud = g_rp.hud.lock();
    if (!hud) return;
    std::string s;
    if (!g_rp.sol) {
        s = "RL REPLAY: no solution";
        hud->setColor({255, 160, 0});
    } else if (g_rp.outcome == "win") {
        s = fmt::format("RL REPLAY: WIN @ tick {}", g_rp.outcomeTick);
        hud->setColor({0, 255, 0});
    } else if (g_rp.outcome == "death") {
        s = fmt::format("RL REPLAY: DIED @ tick {} x={:.1f}", g_rp.outcomeTick, g_rp.outcomeX);
        hud->setColor({255, 60, 60});
    } else if (!g_rp.armed) {
        s = fmt::format("RL REPLAY: OFF (F8)  {} ticks loaded", g_rp.sol->nTicks);
        hud->setColor({180, 180, 180});
    } else {
        s = fmt::format("RL REPLAY: tick {}/{} hold={}", g_rp.tick, g_rp.sol->nTicks,
                        (int)g_rp.injectedHold);
        hud->setColor({255, 255, 255});
    }
    hud->setString(s.c_str());
}

// Push the desired hold state for the current tick (idempotent per tick).
void injectForTick(GJBaseGameLayer* bgl) {
    if (g_rp.injectedThisTick) return;
    g_rp.injectedThisTick = true;
    // Sim tick t == real tick t+1: the real game's tick-0 processCommands
    // runs before any movement (row 0 has x=0), while sim step 0 already
    // moves the player. So the hold the sim applied on its step t must be
    // what real tick t+1 sees (trace_check encodes the same offset).
    bool want = false;
    if (!g_rp.finished && g_rp.sol && g_rp.tick >= 1 &&
        g_rp.tick - 1 < (uint64_t)g_rp.sol->nTicks)
        want = g_rp.sol->holds[(size_t)(g_rp.tick - 1)] != 0;
    if (want != g_rp.injectedHold) {
        g_rp.injecting = true;
        bgl->handleButton(want, 1, true);
        g_rp.injecting = false;
        g_rp.injectedHold = want;
    }
}

} // namespace

class $modify(RlBGL, GJBaseGameLayer) {
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        bool live = PlayLayer::get() == static_cast<GJBaseGameLayer*>(this);
        if (live && g_rp.armed) {
            // If the engine never drained the queue during the previous tick,
            // processQueuedButtons isn't where we assumed; inject here.
            if (g_rp.tick > 0 && g_rp.pqbCallsLastTick == 0 && !g_rp.prehookFallback) {
                g_rp.prehookFallback = true;
                log::warn("processQueuedButtons not observed per tick — injecting in processCommands pre-hook");
            }
            if (g_rp.prehookFallback) injectForTick(this);
        }
        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
        if (!live || g_rp.finished) return;
        auto* p = m_player1;
        if (!p) return;
        if (g_rp.armed) {
            logRow(p, dt, isHalfTick, isLastTick);
            if (!g_rp.pqbDiagLogged && g_rp.tick == 200) {
                g_rp.pqbDiagLogged = true;
                log::info("diag: processQueuedButtons calls/tick = {} ({})",
                          g_rp.pqbCallsThisTick,
                          g_rp.prehookFallback ? "pre-hook injection" : "drain-site injection");
            }
            if ((g_rp.tick & 15) == 0) updateHud();
            // Past the end of the solution with no verdict yet: keep ticking
            // (the level-complete trigger comes from the engine).
        }
        ++g_rp.tick;
        g_rp.pqbCallsLastTick = g_rp.pqbCallsThisTick;
        g_rp.pqbCallsThisTick = 0;
        g_rp.injectedThisTick = false;
    }

    void processQueuedButtons(float dt, bool clearInputQueue) {
        if (PlayLayer::get() != static_cast<GJBaseGameLayer*>(this) || !g_rp.armed) {
            GJBaseGameLayer::processQueuedButtons(dt, clearInputQueue);
            return;
        }
        ++g_rp.pqbCallsThisTick;
        if (g_rp.blockInput()) m_queuedButtons.clear();
        if (!g_rp.prehookFallback) injectForTick(this);
        GJBaseGameLayer::processQueuedButtons(dt, clearInputQueue);
    }

    void handleButton(bool down, int button, bool isPlayer1) {
        if (g_rp.armed && !g_rp.injecting && g_rp.blockInput() &&
            PlayLayer::get() == static_cast<GJBaseGameLayer*>(this)) {
            return;  // swallow human input while replaying
        }
        GJBaseGameLayer::handleButton(down, button, isPlayer1);
    }
};

class $modify(RlPL, PlayLayer) {
    void resetLevel() {
        g_rp.flushTrace();
        ++g_rp.attempt;
        g_rp.tick = 0;
        g_rp.finished = false;
        g_rp.injectedHold = false;
        g_rp.injectedThisTick = false;
        g_rp.pqbCallsThisTick = g_rp.pqbCallsLastTick = 0;
        g_rp.outcome.clear();
        g_rp.armed = false;  // never inject during the engine's reset
        PlayLayer::resetLevel();

        int levelId = m_level ? (int)m_level->m_levelID.value() : -1;
        loadSolution(levelId);

        if (!g_rp.hud.lock()) {
            auto* lbl = CCLabelBMFont::create("RL REPLAY", "bigFont.fnt");
            lbl->setAnchorPoint({0.f, 1.f});
            lbl->setScale(0.35f);
            auto win = CCDirector::sharedDirector()->getWinSize();
            lbl->setPosition({6.f, win.height - 4.f});
            lbl->setZOrder(1000);
            this->addChild(lbl);
            g_rp.hud = lbl;
        }

        if (!g_rp.warnedMods) {
            g_rp.warnedMods = true;
            for (auto id : {"syzzi.click_between_frames", "eclipse.eclipse-menu"})
                if (Loader::get()->isModLoaded(id))
                    log::warn("{} is loaded — disable it (or its physics/input options) for tick-exact replay", id);
        }

        std::string why;
        if (!g_rp.enabled()) why = "disabled";
        else if (!g_rp.sol) why = "no solution";
        else if (g_rp.sol->levelId >= 0 && g_rp.sol->levelId != levelId)
            why = fmt::format("solution is for level {} not {}", g_rp.sol->levelId, levelId);
        else if (m_isPracticeMode) why = "practice mode";
        else if (m_isTestMode) why = "test mode";
        else if (m_startPosObject) why = "start position set";
        else if (!m_player1) why = "no player";
        else if (m_player1->m_position.x != 0.f)
            why = fmt::format("player x={} at reset (expected 0)", m_player1->m_position.x);
        g_rp.armed = why.empty();
        if (g_rp.armed)
            log::info("armed: attempt {} level {} ({} ticks)", g_rp.attempt, levelId, g_rp.sol->nTicks);
        else
            log::info("not armed: {}", why);
        updateHud();
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        PlayLayer::destroyPlayer(player, object);
        if (player != m_player1 || !g_rp.armed || g_rp.finished || !player->m_isDead) return;
        float x = player->m_position.x;
        float endX = g_rp.sol ? g_rp.sol->endX : 0.f;
        log::warn("DESYNC: died at tick {} x={:.3f} ({:.1f}%) on obj {} (id {}) — sim predicted WIN at x={}",
                  g_rp.tick, x, endX > 0 ? 100.0 * x / endX : 0.0,
                  object ? object->m_uniqueID : -1, object ? object->m_objectID : -1, endX);
        log::warn("run: build/trace_check.exe \"{}\" to find the first divergent tick",
                  (g_rp.tracesDir() / fmt::format("attempt_{:04}.csv", g_rp.attempt)).string());
        g_rp.finish("death", g_rp.tick, x);
        updateHud();
    }

    void levelComplete() {
        PlayLayer::levelComplete();
        if (!g_rp.armed || g_rp.finished) return;
        float x = m_player1 ? m_player1->m_position.x : 0.f;
        log::info("WIN at tick {} x={:.3f} — open-loop replay succeeded", g_rp.tick, x);
        g_rp.finish("win", g_rp.tick, x);
        updateHud();
    }

    void onQuit() {
        g_rp.flushTrace();
        g_rp.armed = false;
        PlayLayer::onQuit();
    }
};

class $modify(RlKeys, CCKeyboardDispatcher) {
    bool dispatchKeyboardMSG(enumKeyCodes key, bool down, bool repeat, double ts) {
        if (key == KEY_F8 && down && !repeat) {
            bool now = !g_rp.enabled();
            Mod::get()->setSettingValue<bool>("enabled", now);
            if (!now) g_rp.armed = false;
            log::info("F8: replay {}", now ? "enabled (takes effect on next reset)" : "disabled");
            updateHud();
            return true;
        }
        return CCKeyboardDispatcher::dispatchKeyboardMSG(key, down, repeat, ts);
    }
};
