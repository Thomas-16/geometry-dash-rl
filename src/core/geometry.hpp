#pragma once

#include "core/level.hpp"

#include <cstdint>
#include <vector>

namespace gdcore {

// Axis-aligned collision box in engine coordinates (engine y = level y + 90;
// the floor top is at y = 90, player center rests at 105).
// `order` is the object's level-file position (= engine m_uniqueID order):
// checkCollisions processes a tick's contacts in DESCENDING load order
// (section lists are appended in load order and iterated backwards), which
// decides which contact the landing-pattern x snap sees first.
struct Box {
    float left, bottom, right, top;
    int order;
};

// Collision geometry for one level, built from the level string plus the
// engine-extracted hitbox table (traces/session2/objects.csv, GD 2.2081).
// Boxes are centered on the object position; rotation by an odd multiple of
// 90 degrees swaps width/height; flips never change the box.
// Mode portal (ship 13 / cube 12). Engine hitbox is 34x86 centered on the
// object position (GameObject::getObjectRect from the trace-logger dump);
// activation is plain rect overlap with the player's 30x30 box.
struct Portal {
    Box box;
    bool toShip;
};

// Activatable objects handled inside GJBaseGameLayer::collisionCheckObjects
// (win 2.2081 base+0x214960): they act immediately when the player's 30x30
// rect overlaps their rect (touch-inclusive), in object order, BEFORE the
// tick's solid resolution and hazard deaths. Pads are one-shot per attempt
// (GameObject::activatedByPlayer / hasBeenActivatedByPlayer gate the loop);
// gravity portals are idempotent (GJBaseGameLayer::flipGravity is a no-op
// when the player already has the requested gravity).
//
// Rings (jump orbs, Polargeist id 36 = type 11 YellowJumpRing) are also
// dispatched here (jump table 0x215fe0 -> GJBaseGameLayer::playerTouchedRing
// 0x217e40) but do NOT act on contact: the ring is recorded in the player's
// m_touchingRings and PlayerObject::ringJump (0x398c00) only fires when the
// player has an unconsumed press (see PlayerState::hasJustHeld). A fired
// ring is marked activatedByPlayer, so it is one-shot per attempt exactly
// like a pad.
enum class SpecialKind : uint8_t {
    YellowPad,      // id 35, type 8: propellPlayer(bumpMod 1.0) -> |yvel| 16
    GravityPad,     // id 67, type 10: propellPlayer(0.8) -> 12.8, then flip
    GravityNormal,  // id 10, type 4: flipGravity(false)
    GravityFlip,    // id 11, type 3: flipGravity(true)
    YellowRing,     // id 36, type 11: ringJump -> yvel = +-jump (11.18)
    PinkPad,        // id 140, type 9: propellPlayer(0.65 cube / 0.35 ship)
                    //   -> |yvel| 10.4 / 5.6 (Can't Let Go)
};

struct Special {
    Box box;
    SpecialKind kind;
    // Pad orientation as GameObject's "is upside-down pad" helper computes it
    // (base+0x1a1910): rotation in (90,270) or exactly 180, XOR flipY. A
    // gravity pad sends the player to gravity = !flipped; yellow pads ignore
    // it (propellPlayer only looks at the player's gravity).
    bool flipped;
};

struct LevelGeometry {
    std::vector<Box> solids;      // sorted by left
    std::vector<Box> hazards;     // sorted by left
    std::vector<Portal> portals;  // mode portals, sorted by left
    std::vector<Special> specials;  // pads, gravity portals, rings; by left

    static LevelGeometry build(const gd::Level& level);
};

} // namespace gdcore
