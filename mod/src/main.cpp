#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace geode::prelude;

// Per-physics-tick trace of player 1 state. Format v1.
// One CSV per attempt, flushed on reset/complete/quit.
namespace {

// src: which hook sampled the row. 0 = post-processCommands (input phase),
// 1 = post-PlayerObject::update (movement phase).
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
};

struct Tracer {
    std::vector<TickRow> rows;
    uint64_t tick = 0;
    bool holding = false;
    int attempt = 0;

    void start() {
        flush();
        ++attempt;
        tick = 0;
        rows.clear();
        rows.reserve(1 << 16);
    }

    void flush() {
        if (rows.empty()) return;
        auto dir = Mod::get()->getSaveDir() / "traces";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        auto path = dir / fmt::format("attempt_{:04}.csv", attempt);
        if (auto* f = std::fopen(path.string().c_str(), "wb")) {
            std::fputs("format=3\n", f);
            std::fputs("src,tick,x,y,yvel,holding,onground,upsidedown,ship,dead,speed,dt,halftick,lasttick,nodex,realx,realy,totaltime\n", f);
            for (auto const& r : rows) {
                std::fprintf(f, "%u,%llu,%.9g,%.9g,%.17g,%u,%u,%u,%u,%u,%.9g,%.9g,%u,%u,%.9g,%.9g,%.9g,%.17g\n",
                    r.src, (unsigned long long)r.tick, r.x, r.y, r.yVel, r.holding,
                    r.onGround, r.upsideDown, r.ship, r.dead, r.speed,
                    r.dt, r.halfTick, r.lastTick, r.nodeX, r.realX, r.realY, r.totalTime);
            }
            std::fclose(f);
            log::info("wrote {} ticks -> {}", rows.size(), path.string());
        }
        rows.clear();
    }
};

Tracer g_tracer;

void logRow(uint8_t src, PlayerObject* p, float dt, bool halfTick, bool lastTick) {
    g_tracer.rows.push_back(TickRow{
        src,
        g_tracer.tick,
        p->m_position.x,
        p->m_position.y,
        p->m_yVelocity,
        (uint8_t)g_tracer.holding,
        (uint8_t)p->m_isOnGround,
        (uint8_t)p->m_isUpsideDown,
        (uint8_t)p->m_isShip,
        (uint8_t)p->m_isDead,
        p->m_playerSpeed,
        dt,
        (uint8_t)halfTick,
        (uint8_t)lastTick,
        p->getPositionX(),
        p->getRealPosition().x,
        p->getRealPosition().y,
        p->m_totalTime,
    });
}

} // namespace

class $modify(TraceBGL, GJBaseGameLayer) {
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
        // Only trace real gameplay, not editor playtest
        if (PlayLayer::get() != static_cast<GJBaseGameLayer*>(this)) return;
        auto* p = m_player1;
        if (!p) return;
        logRow(0, p, dt, isHalfTick, isLastTick);
        ++g_tracer.tick;
    }

    void handleButton(bool down, int button, bool isPlayer1) {
        if (isPlayer1 && button == 1) g_tracer.holding = down;
        GJBaseGameLayer::handleButton(down, button, isPlayer1);
    }
};

class $modify(TracePlayer, PlayerObject) {
    void update(float dt) {
        PlayerObject::update(dt);
        auto* pl = PlayLayer::get();
        if (pl && this == pl->m_player1) logRow(1, this, dt, false, false);
    }
};

class $modify(TracePL, PlayLayer) {
    void resetLevel() {
        g_tracer.start();
        PlayLayer::resetLevel();
    }

    void levelComplete() {
        PlayLayer::levelComplete();
        g_tracer.flush();
    }

    void onQuit() {
        g_tracer.flush();
        PlayLayer::onQuit();
    }
};
