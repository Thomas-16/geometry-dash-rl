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

// Boxes are sorted by left edge and at most 30 wide, so candidates for a
// player box spanning [x-half, x+half] start at left >= x - half - 30.
template <typename F>
inline void forCandidates(const std::vector<Box>& boxes, float x, float half,
                          F&& f) {
    auto it = std::lower_bound(boxes.begin(), boxes.end(), x - half - 30.0f,
        [](const Box& b, float v) { return b.left < v; });
    for (; it != boxes.end() && it->left <= x + half; ++it) f(*it);
}

} // namespace detail

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
    // movement (sees input state from the previous tick's processCommands)
    if (s.onGround && s.holding) {
        s.yvel = c.jumpVel;
        s.onGround = false;
    } else {
        s.yvel = roundVel(s.yvel - c.gravity);
        if (s.yvel < c.minYVel) s.yvel = c.minYVel;
    }
    // pure float32 pipeline, like the game: yvel cast to float first
    s.y = s.y + static_cast<float>(s.yvel) * c.yFactor;
    s.x = s.x + c.dx;

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
            // solid, iterated RIGHT to LEFT). Each call first tests whether
            // this object sits at one of the standard jump offsets from the
            // previously contacted object P (0.9x speed set: +120/+30,
            // +150/-30, +90/+60, each +-1 as float compares); on a match it
            // pulls x toward objX + the saved landing offset, clamped to
            // +-1.0. It then unconditionally replaces P with this object and
            // recaptures the offset — so a second contact box processed
            // first can consume P and suppress the snap the left box would
            // have made. Floor contact never updates P.
            if (onObject) {
                // engine order: descending x
                for (int i = 1; i < nContacts; ++i)
                    for (int j = i; j > 0 &&
                         contacts[j]->left > contacts[j - 1]->left; --j)
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

    // deaths, evaluated at the resolved position
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
    if (s.dead) return;

    // input: fresh click jumps immediately if grounded
    if (holding && !s.holding && s.onGround) {
        s.yvel = c.jumpVel;
        s.onGround = false;
    }
    s.holding = holding;
}

} // namespace gdcore
