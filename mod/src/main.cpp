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

// One row per PlayerObject::collidedWithObject call (kind=0, rect is the
// player rect the engine tested, result its return) or per
// PlayLayer::destroyPlayer call (kind=1, rect/result unused).
struct CollisionRow {
    uint64_t tick;
    uint8_t kind;
    int uniqueID;
    int objectID;
    int objType;
    float rx, ry, rw, rh;
    uint8_t result;
};

struct Tracer {
    std::vector<TickRow> rows;
    std::vector<CollisionRow> collisions;
    uint64_t tick = 0;
    bool holding = false;
    int attempt = 0;

    void start() {
        flush();
        ++attempt;
        tick = 0;
        rows.clear();
        rows.reserve(1 << 16);
        collisions.clear();
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
        if (!collisions.empty()) {
            auto cpath = dir / fmt::format("collisions_{:04}.csv", attempt);
            if (auto* f = std::fopen(cpath.string().c_str(), "wb")) {
                std::fputs("format=1\n", f);
                std::fputs("tick,kind,uniqueid,objid,objtype,rx,ry,rw,rh,result\n", f);
                for (auto const& c : collisions) {
                    std::fprintf(f, "%llu,%u,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%u\n",
                        (unsigned long long)c.tick, c.kind, c.uniqueID, c.objectID,
                        c.objType, c.rx, c.ry, c.rw, c.rh, c.result);
                }
                std::fclose(f);
            }
        }
        rows.clear();
        collisions.clear();
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

// Ground-truth hitbox dump: every object as the engine sees it. Written once
// per session (the level is identical across attempts). Player rect is the
// row with uniqueid=-1.
void dumpObjects(PlayLayer* pl) {
    static bool done = false;
    if (done || !pl->m_objects) return;
    done = true;
    auto dir = Mod::get()->getSaveDir() / "traces";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    auto path = dir / "objects.csv";
    auto* f = std::fopen(path.string().c_str(), "wb");
    if (!f) return;
    std::fputs("format=1\n", f);
    std::fputs("uniqueid,objid,type,x,y,rx,ry,rw,rh,w,h,rot,flipx,flipy,scalex,scaley,"
               "obb,c1x,c1y,c2x,c2y,c3x,c3y,c4x,c4y\n", f);
    auto writeRow = [f](int uniqueID, int objID, int type, CCPoint pos,
                        CCRect const& r, float w, float h, float rot,
                        bool fx, bool fy, float sx, float sy, OBB2D* obb) {
        std::fprintf(f, "%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%u,%u,%.9g,%.9g",
            uniqueID, objID, type, pos.x, pos.y,
            r.origin.x, r.origin.y, r.size.width, r.size.height,
            w, h, rot, (unsigned)fx, (unsigned)fy, sx, sy);
        if (obb) {
            std::fprintf(f, ",1");
            for (auto const& c : obb->m_corners)
                std::fprintf(f, ",%.9g,%.9g", c.x, c.y);
        } else {
            std::fputs(",0,0,0,0,0,0,0,0,0", f);
        }
        std::fputc('\n', f);
    };
    for (auto* obj : CCArrayExt<GameObject*>(pl->m_objects)) {
        writeRow(obj->m_uniqueID, obj->m_objectID, (int)obj->m_objectType,
                 obj->getPosition(), obj->getObjectRect(),
                 obj->m_width, obj->m_height, obj->m_startRotationX,
                 obj->m_startFlipX, obj->m_startFlipY,
                 obj->m_startScaleX, obj->m_startScaleY, obj->m_orientedBox);
    }
    if (auto* p = pl->m_player1) {
        writeRow(-1, 0, -1, p->getPosition(), p->getObjectRect(),
                 p->m_width, p->m_height, 0.0f, false, false,
                 p->m_vehicleSize, p->m_vehicleSize, nullptr);
    }
    std::fclose(f);
    log::info("wrote object dump -> {}", path.string());
}

// Raw hook on the unnamed landing-snap helper at win 2.2081 base+0x393cb0:
// called from collidedWithObjectInternal after a bottom/top resolution with
// (PlayerObject* this, GameObject* object). It keeps this+0x888 = last
// landed object and this+0x838 = double(f32(player.x - object.x)), and
// nudges x toward objX+offset (clamped +-1) when the 90/120/150 jump
// pattern matches. We log every call to pin down its exact semantics.
using SnapFn = void(__fastcall*)(void*, void*);
SnapFn g_snapOrig = nullptr;

void __fastcall snapHook(void* player, void* object) {
    auto* p = reinterpret_cast<PlayerObject*>(player);
    auto* o = reinterpret_cast<GameObject*>(object);
    auto* pl = PlayLayer::get();
    bool trace = pl && p == pl->m_player1;
    float x0 = 0.f;
    int prevUid = -1;
    double d0 = 0.0;
    if (trace) {
        x0 = p->getRealPosition().x;
        auto* prev = *reinterpret_cast<GameObject**>((uintptr_t)player + 0x888);
        prevUid = prev ? prev->m_uniqueID : -1;
        d0 = *reinterpret_cast<double*>((uintptr_t)player + 0x838);
    }
    g_snapOrig(player, object);
    if (trace) {
        double d1 = *reinterpret_cast<double*>((uintptr_t)player + 0x838);
        g_tracer.collisions.push_back(CollisionRow{
            g_tracer.tick, 5,
            o ? o->m_uniqueID : -1, prevUid, 0,
            x0, p->getRealPosition().x, (float)d0, (float)d1, 0});
    }
}

} // namespace

$execute {
    auto addr = geode::base::get() + 0x393cb0;
    // Calling the hooked address from inside the detour re-enters the
    // TulipHook handler, which dispatches to the original.
    g_snapOrig = reinterpret_cast<SnapFn>(addr);
    auto hook = Mod::get()->hook(
        reinterpret_cast<void*>(addr), &snapHook, "landing-snap-helper",
        tulip::hook::TulipConvention::Default);
    if (!hook) log::error("failed to hook snap helper: {}", hook.unwrapErr());
}

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

    bool collidedWithObject(float dt, GameObject* object, cocos2d::CCRect rect, bool skipCheck) {
        auto* pl = PlayLayer::get();
        bool trace = pl && this == pl->m_player1 && object;
        float x0 = trace ? getRealPosition().x : 0.f;
        bool res = PlayerObject::collidedWithObject(dt, object, rect, skipCheck);
        if (trace) {
            // kind 0: rx/ry = player x before/after the call (spots position
            // changes inside the engine's per-object resolution)
            g_tracer.collisions.push_back(CollisionRow{
                g_tracer.tick, 0, object->m_uniqueID, object->m_objectID,
                (int)object->m_objectType, x0, getRealPosition().x,
                rect.size.width, rect.size.height, (uint8_t)res});
        }
        return res;
    }

    void hitGround(GameObject* object, bool notFlipped) {
        auto* pl = PlayLayer::get();
        bool trace = pl && this == pl->m_player1;
        float x0 = trace ? getRealPosition().x : 0.f;
        double v0 = m_yVelocity;
        PlayerObject::hitGround(object, notFlipped);
        if (trace) {
            // kind 2: rx/ry = x before/after, rw = yvel at call, rh = notFlipped
            g_tracer.collisions.push_back(CollisionRow{
                g_tracer.tick, 2,
                object ? object->m_uniqueID : -1,
                object ? object->m_objectID : -1, 0,
                x0, getRealPosition().x, (float)v0, (float)notFlipped, 0});
        }
    }

    void updateCollide(PlayerCollisionDirection direction, GameObject* object) {
        auto* pl = PlayLayer::get();
        bool trace = pl && this == pl->m_player1;
        float x0 = trace ? getRealPosition().x : 0.f;
        PlayerObject::updateCollide(direction, object);
        if (trace) {
            // kind 3: objtype = direction, rx/ry = x before/after
            g_tracer.collisions.push_back(CollisionRow{
                g_tracer.tick, 3,
                object ? object->m_uniqueID : -1,
                object ? object->m_objectID : -1, (int)direction,
                x0, getRealPosition().x, 0.f, 0.f, 0});
        }
    }

    void postCollision(float dt, bool betweenSteps) {
        auto* pl = PlayLayer::get();
        bool trace = pl && this == pl->m_player1;
        float x0 = trace ? getRealPosition().x : 0.f;
        PlayerObject::postCollision(dt, betweenSteps);
        if (trace && x0 != getRealPosition().x) {
            // kind 4: only when x changed inside postCollision
            g_tracer.collisions.push_back(CollisionRow{
                g_tracer.tick, 4, -1, -1, (int)betweenSteps,
                x0, getRealPosition().x, 0.f, 0.f, 0});
        }
    }
};

class $modify(TracePL, PlayLayer) {
    void resetLevel() {
        g_tracer.start();
        PlayLayer::resetLevel();
        dumpObjects(this);
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        if (player == m_player1) {
            g_tracer.collisions.push_back(CollisionRow{
                g_tracer.tick, 1,
                object ? object->m_uniqueID : -1,
                object ? object->m_objectID : -1,
                object ? (int)object->m_objectType : -1,
                0.f, 0.f, 0.f, 0.f, 0});
        }
        PlayLayer::destroyPlayer(player, object);
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
