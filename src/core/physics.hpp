#pragma once

#include "core/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace gdcore {

// Per-tick cube physics at 240 TPS, derived empirically from real-game traces
// (see traces/ and tools/). All constants are for the 0.9x ("normal") speed
// that Stereo Madness uses throughout.
//
// Verified bit-exact against traces at float32 position level. The game's
// internal per-tick order is:
//   1. movement (PlayerObject::update): apply gravity to yvel, integrate x/y
//   2. collision resolution (snap onto solids, deaths)
//   3. input (processCommands): a jump registers here, so it takes effect
//      on the *next* tick's movement
//
// x and y are float32 accumulators in the real game; yvel is a double that
// the game ROUNDS TO 3 DECIMALS on every write (PlayerObject's velocity
// setter at 2.2081 win 0x388d10: trunc toward zero, then round(frac*1000,
// half away from zero)/1000, re-added — note trunc+0.94 != double(7.94),
// which is why yvel values look decimal-perfect but occasionally sit 1 ulp
// off the correctly-rounded decimal). Gravity is applied at float precision
// promoted to double (f32(0.216)), and falls clamp at yvel = -15.0.
//
// Collision rules (derived in traces/session2, see tools/fit_collision.py):
//   - The player's solid box is 30x30 centered on (x, y); the inner box is
//     9x9 (the classic third). Boxes touch-count as overlapping (cocos2d
//     CCRect::intersectsRect semantics).
//   - Landing on a solid: only while falling (yvel < 0), snap is an exact
//     assignment y = top + 15, yvel = 0. Observed clip depths up to ~1.2
//     units land fine; the 10-unit eligibility window below is the community
//     value and is non-binding for Stereo Madness (max fall is ~4.5/tick).
//   - Death on a solid: the inner 9x9 box intersects it. Full-box scraping
//     is harmless — the player can rise through a block corner with up to
//     15 units of overlap as long as the inner box never intersects.
//   - Death on a hazard: the full 30x30 box intersects its hitbox.
struct PhysicsConfig {
    float dx;        // per-tick x advance
    double gravity;  // per-tick yvel decrement (float precision, as double)
    double minYVel;  // fall-speed cap
    float yFactor;   // yvel -> dy factor per tick (dt 0.25 frame-units * 0.9)
    double jumpVel;  // yvel set on jump
    float groundY;   // player center y when standing on the floor
    float half;      // half extent of the solid player box
    float inner;     // half extent of the inner (death) box
    float landClip;  // max depth below a solid top that still lands
};

inline PhysicsConfig stereoMadnessCube() {
    return PhysicsConfig{
        static_cast<float>(5.770002 * 0.9 * 0.25),
        static_cast<double>(0.216f),
        -15.0,
        0.225f,
        11.18,
        105.0f,
        15.0f,
        4.5f,
        10.0f,
    };
}

// Ship (flying) physics, lifted from PlayerObject::updateJump's flying branch
// (GD 2.2081 win, base+0x38b900 -> 0x38c4f7) and verified bit-exact against
// the session3 attempt_0005 ship segment (1119 ticks, yvel to the ulp).
//
// PlayerObject::update multiplies the incoming dt (0.25 frame-units) by a
// hard-coded 0.9f before calling updateJump, so the per-tick velocity unit is
// A = f32(f32(0.25*0.9f) * f32(0.958199...)). Per tick:
//   bugged = playerIsFallingBugged() = yvel < f64(2*f32(gravity)) (normal g)
//   factor = holding ? -1.0 : (bugged ? 0.8 : 1.2)
//   scale  = (holding && bugged) ? 0.5 : 0.4
//   yvel   = round3(yvel - f32(f32(f32(1*A)*factor)*scale))   [/0.85 if mini]
// then, unless an external boost flag is set (never in Stereo Madness), two
// clamp writes through the rounding setter:
//   yvel = round3(max(yvel, f64(f32(0.8f*-8.0f))))   // -6.4
//   yvel = round3(min(yvel, f64(8.0f)))              // +8.0
namespace ship {
// float bits 0x3f754c88, the gravity constant at .rdata 0x140622bf8
constexpr float kGrav = 0.95819902420043945f;
constexpr float kDtGrav = 0.25f * 0.9f * kGrav;             // A
constexpr double kBuggedThresh = static_cast<double>(kGrav + kGrav);
constexpr double kMinYVel = static_cast<double>(0.8f * -8.0f); // f32(-6.4)
constexpr double kMaxYVel = 8.0;
// Flying band (GJBaseGameLayer::checkCollisions head): floor at 90, ceiling
// at floor + 300 (getGroundHeightForMode, ship portal type 5 -> 300.0).
// Player-center clamp lines are +-half the 30-unit box.
constexpr float kCeilY = 90.0f + 300.0f - 15.0f; // 375
// Note: checkCollisions has a 0.2s post-portal snap grace, but only on the
// no-mode/dual path — the ship path snaps unconditionally (0x140213dea goes
// straight to the snap; the [0x7d8]/[0x800] time checks are on the sil=1
// branch reachable only with every mode flag clear).
// Solid top/bottom snap window for flying modes (collidedWithObjectInternal
// prologue at 0x140391c11: ship/bird/dart/swing in classic mode replace the
// default 10.0 window with f64(f32(sign*6.0)); the pure cube keeps 10.0).
// Session5 attempt_0005 tick 18358 discriminates: a 22-unit squeeze where
// the floor block sits 8.5 under the player — w=10 would land, the real
// game (w=6) passes through and dies by inner box.
constexpr double kSnapWindow = 6.0;
} // namespace ship

// Per-tick position dt in double: f64(f32(0.25f * 0.9f)). update() scales the
// 0.25 frame-unit dt by a hard-coded 0.9f in float, then promotes to double
// for the dy multiply (0.22499999403953552246).
constexpr double kDtSpeed = static_cast<double>(0.25f * 0.9f);

// The game's velocity write: round to 3 decimals, keeping the integer part
// exact (trunc toward zero + C round(), half away from zero).
inline double roundVel(double v) {
    double i = std::trunc(v);
    if (v == i) return v;
    return i + std::round((v - i) * 1000.0) / 1000.0;
}

struct PlayerState {
    float x = 0.0f;
    float y = 105.0f;
    double yvel = 0.0;
    bool onGround = false;
    bool holding = false; // input state after the last processCommands
    bool dead = false;
    bool ship = false;
    // Ticks since the last mode-portal activation (bookkeeping; the ship
    // band snap itself has no post-portal grace — see ship:: notes).
    int sinceSwitch = 1 << 20;
    // Landing-pattern snap state (collidedWithObjectInternal tail, win
    // 0x393d00..0x393fe0): the game remembers the last solid object landed
    // on and the player's x offset to it.
    bool hasPrevLand = false;
    float prevLandX = 0.0f;  // previous landed object's center
    float prevLandY = 0.0f;
    double landOffset = 0.0; // f64(f32(player.x - object.x)) at that landing
};

namespace detail {

inline bool overlaps(float l, float b, float r, float t, const Box& o) {
    return !(r < o.left || o.right < l || t < o.bottom || o.top < b);
}

inline float leftEdge(const Box& b) { return b.left; }
inline float leftEdge(const Portal& p) { return p.box.left; }

// Entries are sorted by left edge and at most 34 wide (portals), so
// candidates for a player box spanning [x-half, x+half] start at
// left >= x - half - 34.
template <typename T, typename F>
inline void forCandidates(const std::vector<T>& boxes, float x, float half,
                          F&& f) {
    auto it = std::lower_bound(boxes.begin(), boxes.end(), x - half - 34.0f,
        [](const T& b, float v) { return leftEdge(b) < v; });
    for (; it != boxes.end() && leftEdge(*it) <= x + half; ++it) f(*it);
}

} // namespace detail

inline void stepCubeCollisions(PlayerState& s, const PhysicsConfig& c,
                               const LevelGeometry& g);
inline void stepShipCollisions(PlayerState& s, const PhysicsConfig& c,
                               const LevelGeometry& g, float prevY);
inline void checkDeaths(PlayerState& s, const PhysicsConfig& c,
                        const LevelGeometry& g);

// Advance one 1/240s tick. `holding` is the input state processCommands sees
// at the end of this tick.
//
// Jump semantics (from traces): a fresh click registers in processCommands at
// the end of the tick (yvel set, position unchanged; gravity applies on the
// following tick). A *sustained* hold re-jumps in the movement phase on the
// tick after landing, replacing gravity for that tick.
inline void step(PlayerState& s, const PhysicsConfig& c,
                 const LevelGeometry& g, bool holding) {
    if (s.dead) return;
    float prevY = s.y; // pre-movement position (PlayerObject stores it at the
                       // top of update(); collision eligibility uses it)
    // movement (sees input state from the previous tick's processCommands)
    if (!s.ship) {
        if (s.onGround && s.holding) {
            s.yvel = c.jumpVel;
            s.onGround = false;
        } else {
            s.yvel = roundVel(s.yvel - c.gravity);
            if (s.yvel < c.minYVel) s.yvel = c.minYVel;
        }
    } else {
        // updateJump flying branch; see the ship:: constants above
        bool bugged = s.yvel < ship::kBuggedThresh;
        float factor = s.holding ? -1.0f : (bugged ? 0.8f : 1.2f);
        float scale = (s.holding && bugged) ? 0.5f : 0.4f;
        float inc = ((1.0f * ship::kDtGrav) * factor) * scale;
        s.yvel = roundVel(s.yvel + static_cast<double>(-inc));
        s.yvel = roundVel(std::max(s.yvel, ship::kMinYVel));
        s.yvel = roundVel(std::min(s.yvel, ship::kMaxYVel));
    }
    // Position integration, exactly as PlayerObject::update does it: the
    // per-tick dy is computed in DOUBLE as f64(f32(dt*0.9f)) * m_yVelocity
    // (full double yvel, not pre-rounded to float), rounded to float32 once,
    // then added to the float32 position (cocos2d ccpAdd). Rounding yvel to
    // float first agrees on almost every tick but loses the occasional ulp
    // (caught by session5 ship traces at ticks 7840/8092).
    s.y = s.y + static_cast<float>(kDtSpeed * s.yvel);
    s.x = s.x + c.dx;

    if (s.ship) {
        stepShipCollisions(s, c, g, prevY);
    } else {
        stepCubeCollisions(s, c, g);
    }
    if (s.dead) return;

    // portals: rect overlap with the player's full box, checked in the
    // collision phase. toggleFlyMode on an actual flag change halves yvel
    // (raw write, no 3-decimal rounding) and clears onGround.
    detail::forCandidates(g.portals, s.x, c.half, [&](const Portal& p) {
        if (p.toShip != s.ship &&
            detail::overlaps(s.x - c.half, s.y - c.half, s.x + c.half,
                             s.y + c.half, p.box)) {
            s.ship = p.toShip;
            s.yvel *= 0.5;
            s.onGround = false;
            s.sinceSwitch = 0;
        }
    });

    // input: fresh click jumps immediately if grounded (cube only)
    if (!s.ship && holding && !s.holding && s.onGround) {
        s.yvel = c.jumpVel;
        s.onGround = false;
    }
    s.holding = holding;
    if (s.sinceSwitch < (1 << 20)) ++s.sinceSwitch;
}

inline void stepCubeCollisions(PlayerState& s, const PhysicsConfig& c,
                               const LevelGeometry& g) {
    // solid resolution: pick the highest qualifying top, snap once
    bool grounded = false;
    if (s.yvel < 0.0) {
        float bottom = s.y - c.half;
        bool found = false;
        float best = 0.0f;
        bool onObject = false;
        const Box* contacts[8];
        int nContacts = 0;
        if (s.y <= c.groundY) {
            best = c.groundY - c.half;
            found = true;
        }
        detail::forCandidates(g.solids, s.x, c.half, [&](const Box& b) {
            if (detail::overlaps(s.x - c.half, bottom, s.x + c.half,
                                 s.y + c.half, b) &&
                bottom >= b.top - c.landClip) {
                if (!found || b.top > best) best = b.top, onObject = true;
                else if (b.top == best) onObject = true;
                found = true;
                if (nContacts < 8) contacts[nContacts++] = &b;
            }
        });
        if (found) {
            s.y = best + c.half;
            s.yvel = 0.0;
            grounded = true;
            // Landing-pattern x snap (GD 2.2 jump-consistency mechanic; the
            // helper at win 2.2081 0x393cb0 runs once per bottom-contact
            // solid, in DESCENDING level-load order — the section object
            // lists append in load order and checkCollisions walks them
            // backwards; session4's 573-then-572 and session5's
            // 1060-then-1051 pairs both confirm it, and the latter rules
            // out right-to-left x order). Each call first tests whether
            // this object sits at one of the standard jump offsets from the
            // previously contacted object P (0.9x speed set: +120/+30,
            // +150/-30, +90/+60, each +-1 as float compares); on a match it
            // pulls x toward objX + the saved landing offset, clamped to
            // +-1.0. It then unconditionally replaces P with this object and
            // recaptures the offset — so a contact processed first can
            // consume P and suppress the next contact's snap. Floor contact
            // never updates P.
            if (onObject) {
                for (int i = 1; i < nContacts; ++i)
                    for (int j = i; j > 0 &&
                         contacts[j]->order > contacts[j - 1]->order; --j)
                        std::swap(contacts[j], contacts[j - 1]);
                float bottom2 = s.y - c.half;
                for (int i = 0; i < nContacts; ++i) {
                    const Box& b = *contacts[i];
                    if (b.top != best) continue;
                    // overlap re-test with the current (possibly snapped) x
                    if (!detail::overlaps(s.x - c.half, bottom2,
                                          s.x + c.half, s.y + c.half, b))
                        continue;
                    float ox = (b.left + b.right) * 0.5f;
                    float oy = (b.bottom + b.top) * 0.5f;
                    if (s.hasPrevLand &&
                        (s.prevLandX != ox || s.prevLandY != oy)) {
                        bool pat =
                            (std::fabs(ox - (s.prevLandX + 120.0f)) <= 1.0f &&
                             std::fabs(oy - (s.prevLandY + 30.0f)) <= 1.0f) ||
                            (std::fabs(ox - (s.prevLandX + 150.0f)) <= 1.0f &&
                             std::fabs(oy - (s.prevLandY - 30.0f)) <= 1.0f) ||
                            (std::fabs(ox - (s.prevLandX + 90.0f)) <= 1.0f &&
                             std::fabs(oy - (s.prevLandY + 60.0f)) <= 1.0f);
                        if (pat) {
                            double target =
                                static_cast<double>(ox) + s.landOffset;
                            double cur = static_cast<double>(s.x);
                            if (std::fabs(target - cur) > 1.0)
                                target = target > cur ? cur + 1.0 : cur - 1.0;
                            s.x = static_cast<float>(target);
                        }
                    }
                    s.prevLandX = ox;
                    s.prevLandY = oy;
                    s.landOffset = static_cast<double>(s.x - ox);
                    s.hasPrevLand = true;
                }
            }
        }
    }
    s.onGround = grounded;
    checkDeaths(s, c, g);
}

// Ship collision phase, from GJBaseGameLayer::checkCollisions (band) and
// collidedWithObjectInternal (solids). Differences from cube:
//   - a flying band applies: floor at 90 and ceiling at 390 (player-center
//     lines 105/375). Snapping is suppressed for 0.2s after a mode switch.
//     Floor contact zeroes yvel only while yvel <= 0 (hitGround); ceiling
//     contact only while yvel > 0.
//   - solid tops land with a 6-unit window (cube uses 10) evaluated at both
//     the current and the pre-movement position, gated on yvel <= 0, and
//     only when the object is not "overhead" (its bottom above the player
//     bottom + window). No landing-pattern x snap (checkSnapJumpToObject is
//     called for the pure cube only).
//   - solid bottoms are symmetric ceilings: while rising (yvel > 0), an
//     overhead object whose bottom is within 6 units of the player top
//     (now or pre-movement) snaps y to bottom - 15 and zeroes yvel.
//   - deep overlaps resolve nothing; the inner-box / hazard death rules are
//     shared with cube.
//   - m_isOnGround is sticky in ship mode: nothing clears it in flight (the
//     real game only clears it on mode switches), which is why traces show
//     onground=1 for the whole flight.
inline void stepShipCollisions(PlayerState& s, const PhysicsConfig& c,
                               const LevelGeometry& g, float prevY) {
    // flying band (checkCollisions head: ceiling branch first, then floor)
    if (s.y > ship::kCeilY) {
        s.y = ship::kCeilY;
        if (s.yvel > 0.0) s.yvel = 0.0;
    } else if (s.y < c.groundY) {
        s.y = c.groundY;
        if (s.yvel <= 0.0) {
            s.yvel = 0.0;
            s.onGround = true;
        }
    }

    // solids: landing / head-hit, window 10 at current or pre-movement pos
    double bottomNow = static_cast<double>(s.y - c.half);
    double bottomPrev = static_cast<double>(prevY - c.half);
    double topNow = static_cast<double>(s.y + c.half);
    double topPrev = static_cast<double>(prevY + c.half);
    const double w = ship::kSnapWindow;
    if (s.yvel <= 0.0) {
        bool found = false;
        float best = 0.0f;
        detail::forCandidates(g.solids, s.x, c.half, [&](const Box& b) {
            if (!detail::overlaps(s.x - c.half, s.y - c.half, s.x + c.half,
                                  s.y + c.half, b))
                return;
            double oTop = static_cast<double>(b.top);
            double oBot = static_cast<double>(b.bottom);
            bool landElig = bottomNow + w >= oTop || bottomPrev + w >= oTop;
            bool overhead = oBot >= bottomNow + w || oBot >= bottomPrev + w;
            if (landElig && !overhead) {
                if (!found || b.top > best) best = b.top;
                found = true;
            }
        });
        if (found) {
            s.y = best + c.half;
            s.yvel = 0.0;
            s.onGround = true;
        }
    } else {
        bool found = false;
        float best = 0.0f;
        detail::forCandidates(g.solids, s.x, c.half, [&](const Box& b) {
            if (!detail::overlaps(s.x - c.half, s.y - c.half, s.x + c.half,
                                  s.y + c.half, b))
                return;
            double oBot = static_cast<double>(b.bottom);
            bool overhead = oBot >= bottomNow + w || oBot >= bottomPrev + w;
            bool headElig = oBot >= topNow - w || oBot >= topPrev - w;
            if (overhead && headElig) {
                if (!found || b.bottom < best) best = b.bottom;
                found = true;
            }
        });
        if (found) {
            s.y = best - c.half;
            s.yvel = 0.0;
        }
    }
    checkDeaths(s, c, g);
}

// Deaths, evaluated at the resolved position: inner 9x9 box against solids,
// full 30x30 box against hazards. Shared by cube and ship.
inline void checkDeaths(PlayerState& s, const PhysicsConfig& c,
                        const LevelGeometry& g) {
    detail::forCandidates(g.solids, s.x, c.inner, [&](const Box& b) {
        if (detail::overlaps(s.x - c.inner, s.y - c.inner, s.x + c.inner,
                             s.y + c.inner, b))
            s.dead = true;
    });
    detail::forCandidates(g.hazards, s.x, c.half, [&](const Box& b) {
        if (detail::overlaps(s.x - c.half, s.y - c.half, s.x + c.half,
                             s.y + c.half, b))
            s.dead = true;
    });
}

} // namespace gdcore
