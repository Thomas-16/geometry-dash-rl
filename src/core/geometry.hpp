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
    // Circular hazards (Clutterfunk sawblades 88 / 89 / 98): radius > 0
    // means the box is only the circle's bounding square (used by the obs
    // raster and the candidate scan) and the death test is
    // GJBaseGameLayer::playerCircleCollision (win 2.2081 base+0x211df0),
    // see gdcore::circleHitsPlayer in physics.hpp. 0 for every rectangle.
    // cx / cy are the circle's center = the object position as
    // getRealPosition returns it (kept exactly, not re-derived from the
    // rounded box edges).
    float radius = 0.0f;
    float cx = 0.0f, cy = 0.0f;
};

// Collision geometry for one level, built from the level string plus the
// engine-extracted hitbox table (traces/session2/objects.csv, GD 2.2081).
// Boxes are centered on the object position; rotation by an odd multiple of
// 90 degrees swaps width/height; flips never change the box.
// Mode portal (cube 12 / ship 13 / ball 47 / UFO 111). Engine hitbox is
// 34x86 centered on the object position: GameObject::setupSpriteSize
// (0x1401a36a0, called from customSetup's tail 0x1401946af for every
// non-decoration object, indexed by id) sends all four ids to the same case
// 0x1401a3acc, m_width = 34, m_height = 86 — the same table gives every
// verified size (9 -> 30x27, 61 -> 30x18, 88 -> 44x85, 98 -> 40x40, 99 /
// 101 -> 31x90, 140 -> 25x5, 103 -> 20x19 ...). Activation is plain rect
// overlap with the player's box (collisionCheckObjects types 5 / 6 / 16 /
// 19, all gated by canBeActivatedByPlayer).
enum class PortalMode : uint8_t { Cube, Ship, Ball, Ufo };

// Flying-band floor set by a ship / ball portal (xStep, 2026-09-03). The
// engine centers the flying band on the PORTAL, not on the ground:
// GJBaseGameLayer::playerWillSwitchMode (0x212ef0) -> updateDualGround
// (0x213180) -> animateInDualGroundNew (0x213370) computes
//   center = max(90, floorf((portal.startPos.y - H/2) / 30) * 30) + H/2
// (0x140213429..0x140213471, floorf = the CRT import, H = the mode's
// getGroundHeightForMode: ship 300, UFO 300, ball 240) into GJGameState +0x308 and
// H into +0x2ec, and checkCollisions' band snaps read getMinPortalY /
// getMaxPortalY (0x213690 / 0x213770): floor = max(90, center - H/2),
// ceiling = floor + H. Verified 2026-09-03 by logging both getters per
// tick (trace-logger columns bandmin/bandmax): xStep's second ball portal
// (x 18405, engine y 301) gives [180, 420] and its last ship portal (x
// 23025, y 301) [150, 450]; every level-1..9 portal sits low enough that
// the floor stays 90, which is why the fixed [90, 90+H] band matched all
// 77 earlier traces. Cube portals leave the band alone (the cube snaps
// against the real floor only).
struct Portal {
    Box box;
    PortalMode mode;
    float bandFloor; // Ship / Ball / Ufo: the band floor this portal installs
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
// like a pad. The pink ring (Time Machine id 141, type 12) shares the same
// dispatch (jump table entry 0x140215a32) and the same ringJump path; only
// the velocity multiplier differs (0x140399298). The gravity ring (xStep id
// 84, type 13) is dispatched by the same entry (the table at 0x140215fe0 is
// indexed by type - 3; entries 8/9/10 = types 11/12/13 all go to
// 0x140215a32 -> playerTouchedRing) and runs the same ringJump path with the
// 0.8 multiplier (0x14039924f) plus a gravity flip at the end (0x140399608).
//
// Mirror portals (Time Machine ids 45/46, types 14/15) are also dispatched
// here (0x140215723 / 0x140215690) but only call GJBaseGameLayer::
// toggleFlipped (0x2467d0), which flips the camera (m_isFlipped +0x418 and
// the +0x41c/+0x420 transition) and plays effects; no PlayerObject physics
// field is touched, so they are deliberately not part of LevelGeometry.
enum class SpecialKind : uint8_t {
    YellowPad,      // id 35, type 8: propellPlayer(bumpMod 1.0) -> |yvel| 16
    GravityPad,     // id 67, type 10: propellPlayer(0.8) -> 12.8, then flip
    GravityNormal,  // id 10, type 4: flipGravity(false)
    GravityFlip,    // id 11, type 3: flipGravity(true)
    YellowRing,     // id 36, type 11: ringJump -> yvel = +-jump (11.18)
    PinkPad,        // id 140, type 9: propellPlayer(0.65 cube / 0.35 ship)
                    //   -> |yvel| 10.4 / 5.6 (Can't Let Go)
    PinkRing,       // id 141, type 12: ringJump with the jump scaled by
                    //   0.72 cube / 0.37 ship -> |yvel| 8.05 / 4.137
                    //   (Time Machine); same press/buffer semantics as 36
    GravityRing,    // id 84, type 13 (xStep): ringJump with the jump scaled
                    //   by 0.8 in EVERY mode -> yvel = +-8.944 toward the
                    //   current "up", then flipGravity (yvel *= 0.5 raw,
                    //   m_isOnGround = 0) -> 4.472 away from the NEW
                    //   gravity; same press/buffer semantics as 36
    // Size portals (Clutterfunk). customSetup gives id 99 m_objectType 17
    // (0x140192a7f) and id 101 type 18 (0x140192af4), with the mode-portal
    // field pattern (+0x420 = 10, +0x464 = 5, no rect scale); the
    // collisionCheckObjects entries for types 17 / 18 (jump table
    // 0x140215fe0 slots 14 / 15 -> 0x140215584 / 0x14021560a) both run
    // canBeActivatedByPlayer, store the portal in m_lastPortalPos /
    // +0xa80, then PlayerObject::togglePlayerScale(enable, noEffects) with
    // enable = 0 for type 17 and 1 for type 18, then activatedByPlayer and
    // presentation. So 99 = back to NORMAL size and 101 = MINI (the level
    // confirms it: 101 precedes Clutterfunk's mini sections). togglePlayerScale
    // (0x1403a0700) is idempotent (early return when the size already
    // matches), sets m_vehicleSize (+0x9f0) to 0.6f / 1.0f and copies it
    // into the player's rect scale (+0x394 / +0x398); it touches no
    // velocity, no m_isOnGround and no position. Rect: 31x90 (OpenGD's
    // engine-extracted table; the portal_08/09 front frames are 32x90
    // trimmed the way portal_03's 35x86 gives the engine's 34x86) - to be
    // confirmed by a level-11 objects.csv dump.
    SizeNormal,     // id 99, type 17: togglePlayerScale(false)
    SizeMini,       // id 101, type 18: togglePlayerScale(true)
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
