#pragma once

#include "core/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace gdcore {

// Per-tick cube physics at 240 TPS, derived empirically from real-game traces
// (see traces/ and tools/). All constants are for the 0.9x ("normal") speed
// that the first seven official levels use throughout (Dry Out / Base After
// Base add nothing beyond the slab id 68 and reachable flip portals; Can't
// Let Go adds the pink jump pad id 140 — see geometry.cpp and pad::; Jumper
// adds nothing to the object set but is the first level that flips gravity
// WHILE FLYING — see the ship:: notes on the upside-down ship; Time Machine
// adds the pink jump ring id 141 — see ring:: — and mirror portals, which
// are camera-only; Cycles adds the ball portal id 47 and with it BALL MODE,
// see ball::; xStep adds the blue GRAVITY RING id 84 — a ring jump followed
// by a gravity flip, see the ring notes and fireRing — plus new solid /
// hazard ids that only touch geometry.cpp; Clutterfunk adds the SIZE
// PORTALS 99 / 101 and with them MINI MODE, see mini::, and the circular
// SAWBLADE hazards 88 / 89 / 98, see circleHitsPlayer).
//
// Verified bit-exact against traces at float32 position level. The game's
// internal per-tick order is:
//   1. movement (PlayerObject::update): apply gravity to yvel, integrate x/y
//   2. collision (GJBaseGameLayer::checkCollisions):
//        a. floor / flying-band snap (function head)
//        b. collisionCheckObjects per x-section: pads, portals and other
//           activatable objects act IMMEDIATELY; solids and hazards are only
//           collected into lists
//        c. the solid list, in descending load order (collidedWithObject)
//        d. the hazard list (destroyPlayer)
//   3. input (processCommands): a jump registers here, so it takes effect
//      on the *next* tick's movement
//
// x and y are float32 accumulators in the real game; yvel is a double that
// the game ROUNDS TO 3 DECIMALS on every write (PlayerObject's velocity
// setter at 2.2081 win 0x388d10: trunc toward zero, then round(frac*1000,
// half away from zero)/1000, re-added — note trunc+0.94 != double(7.94),
// which is why yvel values look decimal-perfect but occasionally sit 1 ulp
// off the correctly-rounded decimal). Gravity is applied at float precision
// promoted to double (f32(0.225f*0.96f) = 0.2159999907), and falls clamp at
// yvel = -15.0.
//
// Collision rules (derived in traces/session2, see tools/fit_collision.py):
//   - The player's solid box is 30x30 centered on (x, y); the inner box is
//     9x9 (the classic third). Boxes touch-count as overlapping (cocos2d
//     CCRect::intersectsRect semantics).
//   - Landing on a solid: only while not rising (yvel <= 0), snap is an exact
//     assignment y = top + 15, yvel = 0, eligible while the player bottom is
//     within 10 units (cube) of the object top at the current OR the
//     pre-movement position (collidedWithObjectInternal, win 0x391a70).
//   - No head hit for the cube: rising passes through solids freely (the
//     ship does get a mirrored 6-unit head snap, see stepFlyingSolids).
//   - Death on a solid: the inner 9x9 box intersects it. Full-box scraping
//     is harmless — the player can rise through a block corner with up to
//     15 units of overlap as long as the inner box never intersects.
//   - Death on a hazard: the full 30x30 box intersects its hitbox.
//
// Gravity flip (Back On Track: gravity pad id 67 + normal-gravity portal
// id 10). Every vertical rule above is written against the gravity direction
// s = upsideDown ? -1 : +1 (PlayerObject reads m_isUpsideDown at each site):
// gravity becomes yvel += g, the jump is yvel = -11.18, the fall clamp is
// +15 and "landing" is on an object's BOTTOM face at y = bottom - 15. There
// is no ceiling for the flipped classic cube (checkCollisions' band snap is
// ship/dual only): the player "falls" upward until it meets a solid.
//
// Jump rings (Polargeist: yellow orb id 36, type 11). From the 2.2081
// disassembly (PlayerObject::pushButton 0x397f40, releaseButton 0x3981d0,
// ringJump 0x398c00, resetTouchedRings 0x3982e0, update() tail 0x389f09,
// GJBaseGameLayer::playerTouchedRing 0x217e40):
//   - pushButton sets m_isHolding (+0x985) AND m_hasJustHeld (+0x986);
//     releaseButton clears both (one 16-bit store). A cube jump (updateJump
//     0x38bbef) and a ring jump consume m_hasJustHeld. The end of update()
//     snapshots both into +0x989/+0x98a and zeroes the per-tick "ring used"
//     byte +0x98b — so within one tick at most one ring fires.
//   - contact phase: playerTouchedRing adds the ring to m_touchingRings and
//     calls ringJump(ring), which fires only if the snapshots say the player
//     is holding with an unconsumed press (a press BEFORE reaching the ring,
//     i.e. the classic "buffered" orb) and the mode is not flying.
//   - input phase: pushButton with m_touchingRings non-empty calls ringJump
//     on the touching rings (36-type custom rings first, then the rest;
//     only the first passes the +0x98b guard) INSTEAD of the ground jump.
//     m_touchingRings is pruned by resetTouchedRings(false) only after the
//     next tick's input (GJBaseGameLayer::update: processCommands ->
//     resetTouchedRings -> ... -> checkCollisions), so a press can still
//     fire a ring the player overlapped on the previous tick.
//   - ringJump (type 11 default branch 0x3991fe): m_isOnGround (+0x9c1;
//     the bindings layout puts it 33 bytes after m_yVelocity +0x9a0, after
//     m_isUpsideDown +0x9bf and m_isDead +0x9c0 — earlier notes called this
//     byte m_canJump) = 0, yvel = round3(f64(f32(sign) * f32(m_jumpAccel
//     +0x7c0) * sizeMul)) — the same 11.18 as the ground jump — then
//     activatedByPlayer marks the ring used for the rest of the attempt
//     (collisionCheckObjects skips objects whose hasBeenActivatedByPlayer
//     is set).
//   - ring type multiplier (0x14039924f..0x1403993aa, applied to the
//     f32(m_jumpAccel) value before the sign): type 13 (gravity ring) 0.8,
//     type 12 (pink ring, Time Machine id 141) 0.37 ship / 0.42 bird /
//     0.77 ball / 0.72 otherwise (.rdata 0x140622ac4 / ad4 / b94 / b80),
//     type 11 (yellow) 1.0 unless robot (0.9). Everything else in ringJump
//     is type-independent for 11 and 12, so the pink ring is the yellow
//     ring with yvel = round3(f64(f32(sign) * f32(f32(m_jumpAccel) *
//     0.72f))) = +-8.05 for the cube (4.137 for a ship pressing on it).
//   - GRAVITY RING (xStep id 84, type 13, .rdata 0x140622ba4 = 0.8f): the
//     0.8 multiplier is taken at 0x14039924f BEFORE (and instead of) the
//     per-mode pink-ring table, i.e. it is 0.8 in every mode; the velocity
//     write at 0x14039951b is the common one, yvel = round3(f64(f32(sign)
//     * f32(f32(m_jumpAccel) * 0.8f) * sizeMul)) = +-8.944 with the sign
//     of the CURRENT gravity (an upward kick under normal gravity), then
//     the ball's raw f64(0.7f) multiply + m_isHolding = 0 (0x1403995be,
//     mode-generic), and only THEN (0x140399608, type == 13) the flip:
//     GJBaseGameLayer::flipGravity(player, !m_isUpsideDown, noEffects = 1)
//     (0x140212b00; it just forwards to PlayerObject::flipGravity when the
//     state differs, plus the dual-player mirror), so yvel *= 0.5 raw and
//     m_isOnGround = 0 (0x14039a3cd) with the +0x800 flip stamp. Net for
//     the cube: yvel = 4.472 away from the NEW gravity, i.e. the player
//     keeps moving toward the old "up" while the new gravity accelerates
//     it the same way - the blue orb sends a normal-gravity cube to the
//     ceiling. Ball: 8.944 * 0.7 * 0.5 = 3.1304 raw. The virtual called
//     afterwards with the new flag (vtable +0x500 = PlayLayer::
//     playGravityEffect) is presentation only. The ring itself is the
//     yellow ring's twin in customSetup (36x36 box, same playerTouchedRing
//     dispatch, same press / buffer / one-shot semantics).
// PlayerObject::m_gravity (+0x7c8) and m_jumpAccel (+0x7c0) are set by
// PlayerObject::updateTimeMod (win 2.2081 base+0x3a0d50) from the speed:
// speed 0.9f -> m_jumpAccel 0x40265c2d20000000 (11.18003..., which the
// 3-decimal velocity setter turns into exactly 11.18) and m_gravity
// 0x3feea99100000000 = 0.95819902420043945 = the float 0x3f754c88 widened
// (0.7 -> 0.940199, 1.1 -> 0.957199, 1.3 -> 0.961199). PlayerObject::init
// calls updateTimeMod(0.9f) once and nothing in a plain level rewrites the
// field, so EVERY mode runs with this gravity: updateJump reads
// f32(m_gravity) for the cube (0x14038ba20) and the flying modes load the
// same value as a float constant from .rdata 0x140622bf8 (0x14038ba86).
// Earlier notes fitted the cube gravity as f32(0.225f * 0.96f) =
// 0.21599999; the engine value is f32(0.225f * 0.95819902f) = 0.21559478.
// The two are indistinguishable through the 3-decimal velocity setter
// unless yvel carries a 4th decimal (no recorded cube tick does, and the
// half-thousandth ties after a portal halving round the same way), but the
// disassembly is unambiguous, so the engine value is used.
constexpr float kGravity = 0.95819902420043945f; // m_gravity at speed 0.9

// Vertical bound (traces/session17, Cycles: a flipped cube that missed the
// normal portal at x 2733 "fell" upward to y 2791 and the real game killed
// it there, after the sim had matched 2837 ticks of that flight). The
// classic cube has no ceiling snap, but GJBaseGameLayer::checkCollisions
// (0x140213bfe, evaluated for every mode when y >= 105, BEFORE the band
// snaps and objects) flags position.y > m_maxGameplayY (+0x36a8) + (1 -
// scale) * h/2, and kills the player (destroyPlayer, 0x14021417d) when the
// flag was ALSO set on the previous tick (+0xc38 = last tick's flag).
// GJBaseGameLayer::updateMaxGameplayY (0x23aec0) sets the field to 2790.0f
// (.rdata 0x140623594) unless a level-settings flag (+0x155 / +0x1cd of the
// LevelSettingsObject) is set, in which case it is the highest object y +
// 90 + 300; the classic levels take the constant. The session17 replay
// died at tick 2838 with y 2791.1 (tick 2837) and 2794.5 (tick 2838).
constexpr float kMaxGameplayY = 2790.0f;

// Ball mode (Cycles: ball portal id 47, type 16). From the 2.2081
// disassembly — the ball is NOT a flying mode: updateJump (0x38b900) sends
// only ship/bird/dart/swing to the flying branch (0x14038bad1), the ball
// runs the cube's branch with these differences:
//   - gravity multiplier: xmm7 = 0.6f (.rdata 0x140622b38, 0x14038bb0d;
//     spider shares it, robot 0.9f) applied as
//     yvel -= f64(f32(f32(sign * f32(dt*g)) * 0.6f)) (0x14038c2d0..2ed), the
//     same round3 setter and the same +-15 clamp (0x140623778 / 0x140622ef0)
//     as the cube; playerIsFallingBugged is the unscaled 2*m_gravity test.
//   - the "jump" (0x14038bb8a, m_isHolding && m_isOnGround, dashing off):
//     yvel = round3(f32(sign) * f32(m_jumpAccel) * sizeMul) exactly like the
//     cube, THEN (0x14038be5f, m_isBall) flipGravity(!m_isUpsideDown,
//     noEffects) — which halves yvel raw (0x140622d98) and clears
//     m_isOnGround — then yvel *= f64(0.6f) raw (0x140622db0) and
//     m_isHolding (+0x985) = 0. So a click on a surface is a gravity flip
//     with a 11.18 * 0.5 * 0.6 = 3.354 kick toward the NEW gravity, and the
//     press is consumed: holding the button does not flip again on the next
//     landing (the real game needs a release + press; the sim therefore
//     keeps the physical button state, PlayerState::inputHeld, apart from
//     m_isHolding).
//   - pushButton (0x397f40): with no touching ring, a ball that is
//     m_isOnGround calls updateJump(0) immediately (0x140398036), i.e. the
//     flip happens in the input phase like the cube's click jump; otherwise
//     the held m_isHolding flips in the movement phase of the first grounded
//     tick (a buffered flip), exactly the cube's buffered re-jump.
//   - rings (ringJump 0x3995be): after the yellow/pink ring velocity the
//     ball multiplies yvel by f64(0.7f) raw (0x140622dd0) and clears
//     m_isHolding; no gravity flip. Pink ring multiplier 0.77f for the ball.
//   - pads (propellPlayer 0x14039f8eb): after round3(sign * bumpMod * 16)
//     the ball multiplies yvel by f64(0.6f) raw (0x140622db0); pink pad
//     bumpMod 0.7 for the ball (bumpPlayer 0x2179d0).
//   - checkCollisions (0x2137f0) treats the ball like the flying modes for
//     the band (sil at 0x1402139e5): ceiling = getGroundHeightForMode(16) =
//     240 (jump table 0x140211dac: ship/UFO/wave/swing 300, ball 240,
//     default 270) -> y clamps to [105, 315]. Ceiling (0x140213dea ->
//     0x140213da1): y = 315, and if yvel > 0 hitGround(null). Floor
//     (0x140214090 -> 0x140213fff -> 0x1402140fc): y = 105 and hitGround
//     UNCONDITIONALLY (the yvel < 0 gate at 0x1402140f1 is ship-only, the
//     swing skips it). hitGround (0x39bf30) zeroes yvel and sets
//     m_isOnGround unless sign * yvel > 5.0 (0x140622e98) at entry.
//   - collidedWithObjectInternal: the ball keeps the cube's 10-unit snap
//     window (0x140391c11 lists only ship/bird/dart/swing for 6.0) but is in
//     the flying lists for the blocks (0x140392385 flipped block 1,
//     0x1403928d3 normal block 2), so its solid contacts are the ship's two
//     blocks with w = 10: normal gravity lands on tops (hitGround) and bumps
//     its head on bottoms (hitHead, 0x140392f6f: bl = overhead), flipped
//     gravity bumps on tops (0x140393821 with bl = 1 -> hitHead) and lands
//     on bottoms (block 2, bl = 0 -> hitGround). checkSnapJumpToObject is
//     pure-cube only ([rsp+0x36]) so there is no landing-pattern x snap.
//   - mode portals: the ball portal case (0x1402151b5) calls
//     toggleFlyMode(0) first (halving yvel only if the player WAS a ship)
//     then toggleRollMode(1) (0x39b570), which stamps +0x7d8, sets m_isBall
//     and resets rotation state — no velocity or m_isOnGround change. The
//     cube portal case likewise ends ball mode via toggleRollMode(0) with
//     no physics side effect; the ship portal goes through switchedToMode(5)
//     = toggleFlyMode(1) (halve, m_isOnGround = 0) + toggleRollMode(0).
//   - flipGravity (0x39a1d0) is mode-independent: yvel *= 0.5 raw,
//     m_isOnGround = 0 (the ball tail 0x14039a3d4 only resets rotation).
namespace ball {
constexpr float kGravMul = 0.6f;                              // 0x140622b38
constexpr double kFlipKick = static_cast<double>(0.6f);       // 0x140622db0 (f64 0.6000000238)
constexpr double kRingMul = static_cast<double>(0.7f);        // 0x140622dd0 (f64 0.6999999881)
constexpr double kPadMul = static_cast<double>(0.6f);         // 0x140622db0
constexpr float kBandHeight = 240.0f;                         // 0x140623408
constexpr float kCeilY = 90.0f + kBandHeight - 15.0f;         // 315 when the floor is 90 (see bandFloor)
constexpr double kSnapWindow = 10.0;                          // cube window
} // namespace ball

namespace ring {
// m_jumpAccel (+0x7c0) at speed 0.9: 0x40265c2d20000000, exactly
// representable as a float (ringJump reads it with cvtpd2ps). The
// 3-decimal setter turns it into 11.18 for the yellow ring / ground jump.
constexpr double kJumpAccel = 11.180031776428223;
constexpr float kPinkMulCube = 0.72f;  // .rdata 0x140622b80 (default branch)
constexpr float kPinkMulShip = 0.37f;  // .rdata 0x140622ac4 (m_isShip)
constexpr float kPinkMulBird = 0.42f;  // .rdata 0x140622ad4 (m_isBird)
constexpr float kPinkMulBall = 0.77f;  // .rdata 0x140622b94 (m_isBall)
constexpr float kGravityMul = 0.8f;    // .rdata 0x140622ba4 (type 13, every mode)
} // namespace ring

// Mini mode (Clutterfunk: size portals 99 / 101, SpecialKind::SizeNormal /
// SizeMini). PlayerObject::togglePlayerScale (win 2.2081 base+0x3a0700)
// writes m_vehicleSize (+0x9f0) = 0.6f (0x1403a076c) or 1.0f (0x1403a07cc)
// and copies it into the player's rect scale +0x394 / +0x398 (0x1403a085d);
// it changes no velocity, no m_isOnGround and no position (the rest of the
// function scales child sprites, plays the scale effect and, for a ball,
// resets the rotation state). Every physics site that reads +0x9f0 was
// enumerated (96 references, 29 functions) and these are the ones that
// change gameplay:
//   - player rect: GameObject::getObjectRect() (0x1401976a0) = the cached
//     getObjectRect2(m_spriteWidthScale, m_spriteHeightScale) (0x140197850
//     -> 0x1401976c0): size = m_scaleX * m_width * scale = 30 * 0.6f = 18
//     (float-exact), so the outer box is 18x18, half = 9. The INNER death
//     box is collidedWithObjectInternal's direct getObjectRect(0.3f, 0.3f)
//     (0x140393658, vtable +0x488), which multiplies m_scaleX * m_width by
//     the 0.3 argument and never sees the vehicle size: it stays 9x9. (The
//     player's own m_scaleX +0x488 stays 1.0: PlayerObject::setScaleX is
//     GameObject::setScaleX, which only forwards to CCNode and the child
//     sprites.)
//   - floor / band lines (checkCollisions 0x140213966..0x1402139b4 and
//     0x140213c72..0x140213c84): xmm6 = f32(f32(1 - scale) * h) * 0.5 with
//     h = +0xab0 = 30 (PlayerObject::init 0x14038740e), i.e. 5.9999995 for
//     the mini; cube floor line = 90 + 15 - xmm6 -> 99.0 (float-exact after
//     the add), band floor = minPortalY + 15 - xmm6, ceiling = maxPortalY
//     - 15 + xmm6 -> floor + 9 / ceiling - 9 (also float-exact), i.e. the
//     lines sit at +-half in every mode. The kMaxGameplayY test
//     (0x140213c0f) likewise flags y > maxY + xmm6 = 2796.
//   - solid snaps (collidedWithObjectInternal 0x14039386b: y = objTop +
//     rect.h * 0.5, and the block-2 mirror): +-9 for the mini; the snap
//     windows (10 cube / ball, 6 flying; prologue 0x140391b90..) do not
//     read the scale.
//   - jump: updateJump loads xmm9 = (scale == 1) ? 1.0f : 0.8f
//     (0x14038bab5, .rdata 0x140622ba4) and the cube / ball jump velocity
//     is f64(f32(sign) * f32(m_jumpAccel) * xmm9) (0x14038bcfd..0x14038bd06)
//     -> 8.944 through the 3-decimal setter (the ball then flips and
//     scales exactly as before). ringJump (0x14039921f) and propellPlayer
//     (0x14039f89e) use the same 0.8 "sizeMul"; bumpPlayer's pink-pad table
//     and the ring type multipliers do not read the scale (only the red pad
//     / red ring, types 34 / 35, do).
//   - ship: the flying branch sets xmm9 = 0.85f when scale != 1
//     (0x14038c4ff, .rdata 0x140622bc4) and DIVIDES the per-tick increment
//     by it (0x14038c692: f32(f32(f32(f32(sign * A) * factor) * scale) /
//     0.85f)) as well as both clamp bounds (0x14038cac5..0x14038cb0c:
//     f32(f32(0.8f * -8) / 0.85f) = -7.5294118, f32(8 / 0.85f) = 9.4117647
//     and the mirrored pair for the flipped ship). playerIsFallingBugged
//     and the gravity constant are unchanged, as is the cube / ball gravity
//     (no scale read on the fall path).
//   - landing-pattern snap (checkSnapJumpToObject 0x140393de2): the first
//     pattern's dx becomes 90 instead of 120 (xmm1 = xmm2 = 90.0f); the
//     (+150, -30) and (+90, +60) patterns and the +-1 tolerance stay.
//   - PlayerObject::update's three reads are the wave's x-speed doubling,
//     the cube rotation speed and a force-counter path (+0xb88, never set
//     in a plain level); hitGround's read only positions the landing
//     particle. Nothing else changes.
namespace mini {
constexpr float kScale = 0.6f;    // m_vehicleSize, 0x3f19999a
constexpr float kSizeMul = 0.8f;  // jump / ring / pad multiplier, 0x140622ba4
constexpr float kShipDiv = 0.85f; // flying increment and clamp divisor, 0x140622bc4
constexpr float kSnapDx1 = 90.0f; // checkSnapJumpToObject pattern 1 dx (120 normally)
} // namespace mini

// GJBaseGameLayer::playerCircleCollision (0x140211df0) as a classic level
// runs it (LevelSettingsObject::m_fixRadiusCollision +0x1cf is a 2.2
// level option, clear for the official levels, so the 0x140211ec5 branch):
// the player's rect (getObjectRect, i.e. the outer box) CONTAINS the
// object's position (CCRect::containsPoint, inclusive), or any of the four
// rect corners lies strictly inside the circle (ccpDistance(corner,
// center) < radius, float sqrt; 0x140211fa5..0x140212062). The radius is
// m_objectRadius times max(m_scaleX, m_scaleY) unless both are 1.0. This
// is NOT a true circle-rectangle test: a circle that crosses an edge
// without containing a corner or being centered inside the rect is a miss.
inline bool circleHitsPlayer(float px, float py, float half, float cx,
                             float cy, float radius) {
    const float l = px - half, r = px + half, b = py - half, t = py + half;
    if (cx >= l && cx <= r && cy >= b && cy <= t) return true;
    auto inside = [&](float x, float y) {
        const float dx = x - cx, dy = y - cy;
        return radius > std::sqrt(dx * dx + dy * dy);
    };
    // corner order as the engine tests them: (maxX, maxY), (maxX, minY),
    // (minX, minY), (minX, maxY) — irrelevant for the outcome
    return inside(r, t) || inside(r, b) || inside(l, b) || inside(l, t);
}

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
    // m_gravity (+0x7c8): kGravity for every mode at speed 0.9. The
    // playerIsFallingBugged test reads it as a double (2 * m_gravity).
    double gravityMul = static_cast<double>(kGravity);
    // Ball per-tick gravity: f32(f32(dt * g) * 0.6f) (updateJump
    // 0x14038c2d0..2de, xmm7 = ball::kGravMul), promoted to double.
    double ballGravity =
        static_cast<double>(static_cast<float>((0.25f * 0.9f) * kGravity) *
                            ball::kGravMul);
};

inline PhysicsConfig stereoMadnessCube() {
    return PhysicsConfig{
        static_cast<float>(5.770002 * 0.9 * 0.25),
        // updateJump 0x14038c2d0: f64(f32(f32(dt * 0.9f) * f32(m_gravity)))
        // = 0.21559478 (see kGravity). Rounding ties after a mode portal
        // halves an odd-thousandth yvel raw (session7 attempt_0002 tick
        // 12894: -1.2995 - g -> -1.515, not -1.516) resolve the same way as
        // with the old fitted 0.21599999 constant.
        static_cast<double>((0.25f * 0.9f) * kGravity),
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
// A = f32(f32(0.25*0.9f) * f32(0.958199...)). Per tick, with
// sign = m_isUpsideDown ? -1 : +1 (0x14038c665: the int -1/1 converted to
// float and multiplied into the unit before factor and scale):
//   bugged = playerIsFallingBugged() (0x39a430, the same helper as the cube:
//            normal gravity yvel < f64(f32(2*m_gravity)), flipped
//            yvel > 2*m_gravity — the same 1.916398 both ways)
//   factor = holding ? -1.0 : (bugged ? 0.8 : 1.2)
//   scale  = (holding && bugged) ? 0.5 : 0.4
//   yvel   = round3(yvel - f32(f32(f32(sign*A)*factor)*scale))   [/0.85 mini]
// then, unless an external boost flag is set (never in these levels), two
// clamp writes through the rounding setter (0x14038ca9f; xmm7 = 0.8f,
// xmm13 = 8.0f, .rdata 0x1406237fc = -8.0f):
//   normal:  yvel = round3(max(yvel, f64(f32(0.8f*-8.0f))))   // -6.4
//            yvel = round3(min(yvel, f64(8.0f)))              // +8.0
//   flipped: yvel = round3(max(yvel, f64(-8.0f)))             // -8.0
//            yvel = round3(min(yvel, f64(f32(0.8f*8.0f))))    // +6.4
// so holding pushes the upside-down ship DOWN (toward the floor) and gravity
// pulls it up, the exact mirror. Session5 attempt_0003 tick 7593 (yvel
// 1.918 while holding gained 0.086, not 0.108) pins the falling threshold
// below 1.918, i.e. m_gravity is the 0.9582 constant, not 0.96.
namespace ship {
constexpr float kGrav = kGravity; // .rdata 0x140622bf8, bits 0x3f754c88
constexpr float kDtGrav = 0.25f * 0.9f * kGrav;             // A
constexpr double kMinYVel = static_cast<double>(0.8f * -8.0f); // f32(-6.4)
constexpr double kMaxYVel = 8.0;
constexpr double kMinYVelFlipped = static_cast<double>(-8.0f);
constexpr double kMaxYVelFlipped = static_cast<double>(0.8f * 8.0f); // 6.4f
// Flying band (GJBaseGameLayer::checkCollisions head): floor at the band
// floor the last ship / ball portal installed (PlayerState::bandFloor,
// see Portal::bandFloor in geometry.hpp — 90 in levels 1-9, raised by
// xStep's high portals), ceiling at floor + 300 (getGroundHeightForMode,
// ship portal type 5 -> 300.0). Player-center clamp lines are +-half the
// 30-unit box. kCeilY is the level-1..9 value, kept for reference.
constexpr float kCeilY = 90.0f + 300.0f - 15.0f; // 375 when the floor is 90
// Note: checkCollisions has a 0.2s post-portal snap grace, but only on the
// all-flags-clear (dual) path — the ship path snaps unconditionally
// (0x140213dea / 0x140214090 go straight to the snap; the [0x7d8]/[0x800]
// time checks sit on the branch reachable only with every mode flag clear).
// The snap itself (both gravities, 0x140213da1 / 0x140214042):
//   ceiling: y = 375; if yvel > 0: hitGround(null, !upsideDown)
//   floor:   y = 105; if yvel < 0: hitGround(null, upsideDown)
// hitGround (0x39bf30) zeroes yvel and sets m_isOnGround unless
// sign*yvel > 5.0 at entry (its r12 gate; the second argument only picks
// the landing animation). A "restore the pre-snap yvel" path exists for the
// upside-down ship (r12 / [rbp+0x67] at 0x140213e6c) but is gated on the
// ceiling height having changed since the previous tick ([r14+0x764] is
// refreshed after checkCollisions every update), which never happens in a
// level without camera/ceiling triggers.
// Solid top/bottom snap window for flying modes (collidedWithObjectInternal
// prologue at 0x140391c11: ship/bird/dart/swing in classic mode replace the
// default 10.0 window with f64(f32(sign*6.0)); the pure cube keeps 10.0).
// Session5 attempt_0005 tick 18358 discriminates: a 22-unit squeeze where
// the floor block sits 8.5 under the player — w=10 would land, the real
// game (w=6) passes through and dies by inner box.
constexpr double kSnapWindow = 6.0;
// Object contacts for a flying player, per object in descending load order
// (collidedWithObjectInternal, static objects; the window w = 6 is signed
// by gravity inside the engine, the lines below are already in world y):
//   block 1 (0x14039247f, normal gravity any mode / flipped flying modes):
//     taken when playerBottom + 6 >= objTop at the current OR pre-movement
//     y; if yvel <= 0 it snaps y = objTop + 15. The object is then "handled"
//     even when yvel > 0 (al = 0 at 0x140392893), so block 2 never runs for
//     it.
//   block 2 (0x140392a05, normal flying / flipped any mode): taken when
//     objBottom >= playerTop - 6 (current or pre-movement); if yvel >= 0 it
//     snaps y = objBottom - 15 (0x140392ec8).
//   After either snap: if the object is "overhead" in gravity terms (the
//   bl flag, 0x14039239b / 0x1403923df) the call is hitHead (0x39bed0 =
//   hitGround with m_isOnGround / +0xa0c / +0xa10 saved and restored: yvel
//   = 0 only), otherwise hitGround(obj) which also grounds the player. For
//   axis-aligned static boxes the overhead flag is implied by the block:
//   normal gravity lands in block 1 and bumps its head in block 2; the
//   upside-down ship bumps its head on object TOPS in block 1 (moving down,
//   yvel <= 0) and lands on object BOTTOMS in block 2 (falling upward,
//   yvel >= 0). The "remembered object" bypass ([0x5e0]/[0x5e4] = last
//   tick's contacts, 0x140392a53) only re-admits an object that fails the
//   window test, which a static box a player rests against never does.
} // namespace ship

// Pads and gravity (GJBaseGameLayer::bumpPlayer 0x2179d0, PlayerObject::
// propellPlayer 0x39f850, PlayerObject::flipGravity 0x39a1d0):
//   propellPlayer: yvel = round3(f64(f32(sign) * f32(bumpMod * 16.0f) *
//                  sizeMul)), sign from the player's CURRENT gravity,
//                  sizeMul 1.0 (0.8 mini); clears m_isOnGround / can-jump.
//   yellow pad (type 8): bumpMod 1.0 -> 16.0 (every mode)
//   pink pad (type 9, id 140, Can't Let Go): bumpPlayer 0x2179d0 picks the
//                  bumpMod by player mode (0x140217a6f..: m_isShip +0x9b9
//                  0.35, m_isBird +0x9ba 0.4, m_isBall +0x9bb / m_isSpider
//                  +0x9be 0.7, otherwise (cube) 0.65 -> 10.4); then
//                  the same wrapper 0x39f6a0 -> propellPlayer path as the
//                  yellow pad (boost flag +0x952 cleared, only the red pad
//                  type 34 sets it).
//   gravity pad (type 10): only when the player's gravity differs from the
//                  pad's target; bumpMod 0.8 -> 12.8, then flipGravity(true)
//   flipGravity: m_isUpsideDown = flip, then yvel *= 0.5 as a RAW double
//                write (no 3-decimal rounding; the "skip damping" flag
//                +0x7e1 is only set while loading a checkpoint).
namespace pad {
constexpr float kBase = 16.0f;                  // .rdata 0x1406230f8
constexpr float kGravityBumpMod = 0.8f;         // .rdata 0x140622ba4
constexpr float kPinkBumpModCube = 0.65f;       // .rdata 0x140622b5c
constexpr float kPinkBumpModShip = 0.35f;       // .rdata 0x140622abc
constexpr float kPinkBumpModBall = 0.7f;        // bumpPlayer 0x140217a6f.. (ball / spider)
constexpr double kFlipDamp = 0.5;               // .rdata 0x140622d98
// Upside-down cube below the floor line: checkCollisions snaps it to y=105
// and skips the rest of the collision pass — unless within 0.1 s of the
// flip ([0x800] vs m_totalTime, .rdata 0x140622cf0). Unreachable in Back On
// Track (the flip happens on a raised slab and the player rises); modeled
// from the disassembly only.
constexpr int kFlipFloorGraceTicks = 24;
} // namespace pad

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
    // m_isHolding (+0x985): set by pushButton, cleared by releaseButton and
    // by the ball's flip / ring jump (ringJump 0x140399601, updateJump
    // 0x14038be9d). For the cube and ship it equals the physical button.
    bool holding = false;
    // The physical button state after the last processCommands: pushButton
    // / releaseButton fire on its transitions (the mods call handleButton
    // only when the wanted state changes). Diverges from `holding` only in
    // ball mode, where a flip consumes the press while the button stays
    // down.
    bool inputHeld = false;
    bool dead = false;
    bool ship = false;
    bool ball = false;       // m_isBall (+0x9bb): Cycles, see ball::
    bool upsideDown = false; // m_isUpsideDown (gravity flipped)
    // m_vehicleSize (+0x9f0) != 1: mini mode (Clutterfunk size portals,
    // see mini::). Scales the outer box, the floor / band lines, the jump
    // and pad velocities and the ship's increment; the inner death box
    // stays 9x9.
    bool mini = false;
    // Flying-band floor installed by the last ship / ball portal
    // (Portal::bandFloor; getMinPortalY). The ceiling is floor + 300
    // (ship) / + 240 (ball). 90 until a portal raises it (xStep).
    float bandFloor = 90.0f;
    // Ticks since the last mode-portal activation (bookkeeping; the ship
    // band snap itself has no post-portal grace — see ship:: notes).
    int sinceSwitch = 1 << 20;
    // Ticks since the last gravity flip (pad::kFlipFloorGraceTicks).
    int sinceFlip = 1 << 20;
    // +0xc38: last tick's "above kMaxGameplayY" flag (two in a row kill).
    bool overMaxY = false;
    // Landing-pattern snap state (collidedWithObjectInternal tail, win
    // 0x393d00..0x393fe0): the game remembers the last solid object landed
    // on and the player's x offset to it.
    bool hasPrevLand = false;
    float prevLandX = 0.0f;  // previous landed object's center
    float prevLandY = 0.0f;
    double landOffset = 0.0; // f64(f32(player.x - object.x)) at that landing
    // One-shot activation record for LevelGeometry::specials (pads). Entries
    // are sorted by left edge and x only increases, so a rolling window is
    // exact: specialCursor is the first entry whose right edge the player
    // has not yet passed, bit k of specialDone marks entry cursor+k as
    // activated (64 entries of look-ahead is far more than any level packs
    // into 30 units of x).
    int specialCursor = 0;
    uint64_t specialDone = 0;
    // m_hasJustHeld (+0x986): the current press has not been spent on a
    // jump or ring yet. Set by a fresh press, cleared by release, by any
    // cube jump and by a ring jump.
    bool hasJustHeld = false;
    // m_touchingRings as the input phase sees it: rings (indices into
    // LevelGeometry::specials) the player box overlapped in this tick's
    // contact phase (ringNow) and in the previous tick's (ringPrev, not yet
    // pruned — see the ring notes above). Fired rings are removed.
    static constexpr int kMaxTouchRings = 4;
    int ringNow[kMaxTouchRings] = {};
    int ringPrev[kMaxTouchRings] = {};
    int nRingNow = 0;
    int nRingPrev = 0;
};

namespace detail {

inline bool overlaps(float l, float b, float r, float t, const Box& o) {
    return !(r < o.left || o.right < l || t < o.bottom || o.top < b);
}

inline float leftEdge(const Box& b) { return b.left; }
inline float leftEdge(const Portal& p) { return p.box.left; }
inline float leftEdge(const Special& s) { return s.box.left; }

// Entries are sorted by left edge and at most `maxW` wide (34 for solids
// and mode portals; a rotated gravity portal is 76; the widest hazard is
// the 64.6 bounding square of sawblade 88), so candidates for a player box
// spanning [x-half, x+half] start at left >= x-half-maxW.
template <typename T, typename F>
inline void forCandidates(const std::vector<T>& boxes, float x, float half,
                          F&& f, float maxW = 34.0f) {
    auto it = std::lower_bound(boxes.begin(), boxes.end(), x - half - maxW,
        [](const T& b, float v) { return leftEdge(b) < v; });
    for (; it != boxes.end() && leftEdge(*it) <= x + half; ++it) f(*it);
}

constexpr float kSpecialMaxW = 80.0f;
constexpr float kHazardMaxW = 66.0f;

} // namespace detail

// Player size (mini::). The outer box is the engine's getObjectRect():
// f32(m_width * m_vehicleSize) = 18 for the mini, 30 otherwise, half of
// that per side; the inner death box (PhysicsConfig::inner) never scales.
inline float playerScale(const PlayerState& s) {
    return s.mini ? mini::kScale : 1.0f;
}
inline float playerHalf(const PlayerState& s, const PhysicsConfig& c) {
    return ((c.half + c.half) * playerScale(s)) * 0.5f;
}
// The 0.8 "sizeMul" of updateJump / ringJump / propellPlayer.
inline float sizeMul(const PlayerState& s) {
    return s.mini ? mini::kSizeMul : 1.0f;
}

// One-shot record helpers (PlayerState::specialCursor/specialDone).
inline bool specialDone(const PlayerState& s, int i) {
    const int k = i - s.specialCursor;
    return k >= 0 && k < 64 && ((s.specialDone >> k) & 1u);
}
inline void markSpecialDone(PlayerState& s, int i) {
    const int k = i - s.specialCursor;
    if (k >= 0 && k < 64) s.specialDone |= uint64_t{1} << k;
}

inline void removeRing(int* list, int& n, int idx) {
    for (int i = 0; i < n; ++i)
        if (list[i] == idx) {
            for (int j = i + 1; j < n; ++j) list[j - 1] = list[j];
            --n;
            return;
        }
}

// The ring pushButton would fire right now: m_touchingRings order is
// insertion order, so last tick's contacts come first. Returns the special
// index or -1.
inline int pressableRing(const PlayerState& s) {
    for (int i = 0; i < s.nRingPrev; ++i)
        if (!specialDone(s, s.ringPrev[i])) return s.ringPrev[i];
    for (int i = 0; i < s.nRingNow; ++i)
        if (!specialDone(s, s.ringNow[i])) return s.ringNow[i];
    return -1;
}

// PlayerObject::ringJump, yellow (type 11), pink (type 12) and gravity
// (type 13) ring branch.
inline void fireRing(PlayerState& s, const PhysicsConfig& c,
                     const LevelGeometry& g, int idx) {
    (void)c;
    const SpecialKind kind = g.specials[static_cast<size_t>(idx)].kind;
    const float sign = s.upsideDown ? -1.0f : 1.0f;
    // 0x399237: xmm6 = f32(m_jumpAccel); 0x39924f: type 13 scales it by
    // 0.8f (no mode table); 0x399298: type 12 scales it by the mode
    // multiplier (ship / bird / ball / default); 0x3994ef: f32(sign) * xmm6
    // * sizeMul (1.0 unless mini), through the 3-decimal setter.
    float j = static_cast<float>(ring::kJumpAccel);
    if (kind == SpecialKind::GravityRing)
        j *= ring::kGravityMul;
    else if (kind == SpecialKind::PinkRing)
        j *= s.ship ? ring::kPinkMulShip
             : s.ball ? ring::kPinkMulBall
                      : ring::kPinkMulCube;
    const float v = sign * j * sizeMul(s);
    s.yvel = roundVel(static_cast<double>(v));
    s.onGround = false;    // m_canJump cleared (0x39920c)
    s.hasJustHeld = false; // 0x3991ee
    if (s.ball) {
        // 0x1403995be..601: m_isBall -> yvel *= f64(0.7f) as a raw double
        // write, then m_isHolding = 0 (the press is spent; the ball's own
        // flip does not happen here - a gravity ring flips below).
        s.yvel *= ball::kRingMul;
        s.holding = false;
    }
    if (kind == SpecialKind::GravityRing) {
        // 0x140399608: type 13 -> GJBaseGameLayer::flipGravity(player,
        // !m_isUpsideDown, 1) -> PlayerObject::flipGravity: the flag, the
        // +0x800 stamp, yvel *= 0.5 raw (0x14039a2d4, +0x7e1 clear) and
        // m_isOnGround = 0 (0x14039a3cd). Same effect as a gravity portal.
        s.upsideDown = !s.upsideDown;
        s.yvel *= pad::kFlipDamp;
        s.onGround = false;
        s.sinceFlip = 0;
    }
    markSpecialDone(s, idx); // activatedByPlayer (0x399cbd)
    removeRing(s.ringNow, s.nRingNow, idx);   // removeObject (0x398e3a)
    removeRing(s.ringPrev, s.nRingPrev, idx);
}

inline void stepCubeSolids(PlayerState& s, const PhysicsConfig& c,
                           const LevelGeometry& g, float prevY);
inline void stepShipBand(PlayerState& s, const PhysicsConfig& c);
inline void stepBallBand(PlayerState& s, const PhysicsConfig& c);
inline void stepFlyingSolids(PlayerState& s, const PhysicsConfig& c,
                             const LevelGeometry& g, float prevY, double w);
inline void stepSpecials(PlayerState& s, const PhysicsConfig& c,
                         const LevelGeometry& g);
inline void checkDeaths(PlayerState& s, const PhysicsConfig& c,
                        const LevelGeometry& g, float prevY);

// PlayerObject::hitGround (0x39bf30) as far as physics goes: yvel = 0 and
// m_isOnGround = 1 unless sign * yvel > 5.0 at entry (its r12 gate, .rdata
// 0x140622e98). hitHead (hitGroundNoJump 0x39bed0) is the same with
// m_isOnGround saved and restored, i.e. yvel = 0 only.
inline void hitGround(PlayerState& s) {
    const double sign = s.upsideDown ? -1.0 : 1.0;
    if (!(sign * s.yvel > 5.0)) s.onGround = true;
    s.yvel = 0.0;
}

// The ball's jump: PlayerObject::updateJump 0x14038be5f — after the cube
// jump velocity is set, flipGravity(!m_isUpsideDown) (yvel *= 0.5 raw,
// m_isOnGround = 0, +0x800 stamped), then yvel *= f64(0.6f) raw, then
// m_isHolding = 0 (the press is spent; +0x7e8 = 0 is the ball-rotation
// flag).
inline void ballFlipAfterJump(PlayerState& s) {
    s.upsideDown = !s.upsideDown;
    s.yvel *= pad::kFlipDamp;
    s.onGround = false;
    s.sinceFlip = 0;
    s.yvel *= ball::kFlipKick;
    s.holding = false;
}

// PlayerObject::playerIsFallingBugged (win 2.2081 base+0x39a430), with no
// sideways/swing/platformer flags: normal gravity ->
// yvel < f64(f32(2 * f32(m_gravity))); flipped -> yvel > 2 * m_gravity (as a
// double). Both thresholds are 1.916398 (m_gravity is a widened float).
// The cube's updateJump fall path (0x38c27b..0x38c2ae) clears m_isOnGround
// ONLY when this returns true, evaluated on the yvel BEFORE this tick's
// gravity is added. The threshold makes the flag asymmetric: a normal cube
// walking off a ledge is cleared at once (yvel 0 < 1.916), but a flipped
// cube keeps m_isOnGround - and stays jumpable - for 9 ticks after leaving
// the underside of a block (yvel 0.216 .. 1.944, cleared once it exceeds
// the threshold). Seen as onground=1 streaks in every Dry Out / Base After
// Base trace and as the replay-mod death at BAB tick 15202 (session12): the
// sim used to drop the flag at the head of every collision pass, so the
// press that the real game turned into a jump off the slab was a no-op
// here. The flying branch calls the same helper for its "bugged" factor.
inline bool isFallingBugged(const PlayerState& s, const PhysicsConfig& c) {
    if (!s.upsideDown) {
        const float g = static_cast<float>(c.gravityMul);
        return static_cast<double>(g + g) > s.yvel;
    }
    return s.yvel > c.gravityMul + c.gravityMul;
}

// The ground jump velocity write (updateJump 0x14038bce0..0x14038bd0c):
// f64(f32(sign) * f32(m_jumpAccel) * sizeMul) through the 3-decimal
// setter: +-11.18, or +-8.944 for the mini (sizeMul 0.8f). PhysicsConfig's
// jumpVel is the normal-size result.
inline double groundJumpVel(const PlayerState& s, const PhysicsConfig& c) {
    const double sign = s.upsideDown ? -1.0 : 1.0;
    if (!s.mini) return sign * c.jumpVel;
    const float v = static_cast<float>(sign) *
                    static_cast<float>(ring::kJumpAccel) * mini::kSizeMul;
    return roundVel(static_cast<double>(v));
}

// Cube / floor line for the player center: 90 + half (checkCollisions'
// xmm7 = 90 + 15 - (1 - scale) * 15, float-exact at 105 / 99).
inline float floorLine(const PlayerState& s, const PhysicsConfig& c) {
    return c.groundY - c.half + playerHalf(s, c);
}

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
    const double sign = s.upsideDown ? -1.0 : 1.0;
    // movement (sees input state from the previous tick's processCommands)
    if (!s.ship) {
        // The cube AND the ball run this branch (updateJump sends only the
        // flying modes to 0x14038c4f7).
        if (s.onGround && s.holding) {
            s.yvel = groundJumpVel(s, c);
            s.onGround = false;
            s.hasJustHeld = false; // updateJump 0x38bbef
            if (s.ball) ballFlipAfterJump(s); // 0x14038be5f
        } else {
            // updateJump fall path: first the m_isOnGround clear, gated on
            // cubeIsFalling() at the pre-gravity yvel (0x38c27b..0x38c2ae),
            // then (0x14038c2b5) yvel -= f64(sign * f32(g)) and the clamp
            // toward the gravity direction, both through the 3-decimal
            // rounding. The ball's gravity is scaled by 0.6f (ball::).
            if (isFallingBugged(s, c)) s.onGround = false;
            s.yvel = roundVel(s.yvel - sign * (s.ball ? c.ballGravity : c.gravity));
            if (!s.upsideDown) {
                if (s.yvel < c.minYVel) s.yvel = c.minYVel;
            } else {
                if (s.yvel > -c.minYVel) s.yvel = -c.minYVel;
            }
        }
    } else {
        // updateJump flying branch; see the ship:: notes above. The gravity
        // sign enters as the float -1/+1 at 0x14038c665 (Jumper flips the
        // ship at x 7565 / 8403 / 9145).
        const bool bugged = isFallingBugged(s, c);
        const float factor = s.holding ? -1.0f : (bugged ? 0.8f : 1.2f);
        const float scale = (s.holding && bugged) ? 0.5f : 0.4f;
        const float signf = s.upsideDown ? -1.0f : 1.0f;
        // mini ship (0x14038c4ff / 0x14038c692): the increment and both
        // clamp bounds are divided by 0.85f (xmm9), 1.0f otherwise
        const float div = s.mini ? mini::kShipDiv : 1.0f;
        const float inc = (((signf * ship::kDtGrav) * factor) * scale) / div;
        s.yvel = roundVel(s.yvel + static_cast<double>(-inc));
        if (!s.upsideDown) {
            const float lo = static_cast<float>(ship::kMinYVel) / div;
            const float hi = static_cast<float>(ship::kMaxYVel) / div;
            s.yvel = roundVel(std::max(s.yvel, static_cast<double>(lo)));
            s.yvel = roundVel(std::min(s.yvel, static_cast<double>(hi)));
        } else {
            const float lo = static_cast<float>(ship::kMinYVelFlipped) / div;
            const float hi = static_cast<float>(ship::kMaxYVelFlipped) / div;
            s.yvel = roundVel(std::max(s.yvel, static_cast<double>(lo)));
            s.yvel = roundVel(std::min(s.yvel, static_cast<double>(hi)));
        }
    }
    // Position integration, exactly as PlayerObject::update does it: the
    // per-tick dy is computed in DOUBLE as f64(f32(dt*0.9f)) * m_yVelocity
    // (full double yvel, not pre-rounded to float), rounded to float32 once,
    // then added to the float32 position (cocos2d ccpAdd). Rounding yvel to
    // float first agrees on almost every tick but loses the occasional ulp
    // (caught by session5 ship traces at ticks 7840/8092).
    s.y = s.y + static_cast<float>(kDtSpeed * s.yvel);
    s.x = s.x + c.dx;

    // 2a. floor / band snap (checkCollisions head). The classic cube's only
    // boundary is the floor line; hitGround zeroes yvel and grounds it.
    // First the vertical bound (0x140213bfe): position.y above
    // kMaxGameplayY on two consecutive ticks is a death for every mode,
    // decided before any object is looked at.
    // The bound and the floor line move with the player's half extent
    // (mini: 2796 / 99, see mini::).
    if (s.y > kMaxGameplayY + (c.half - playerHalf(s, c))) {
        if (s.overMaxY) {
            s.dead = true;
            return;
        }
        s.overMaxY = true;
    } else {
        s.overMaxY = false;
    }
    bool skipObjects = false;
    if (!s.ship && !s.ball) {
        // m_isOnGround is NOT cleared here: checkCollisions only writes it in
        // the floor/band snaps and hitGround; the per-tick clear lives in
        // updateJump's fall path above (cubeIsFalling).
        const float groundY = floorLine(s, c);
        if (s.y < groundY) {
            if (!s.upsideDown) {
                s.y = groundY;
                s.yvel = 0.0;
                s.onGround = true;
            } else if (s.sinceFlip >= pad::kFlipFloorGraceTicks) {
                // 0x140213b1b: snap, hitGround(null, notFlipped=1),
                // updateCollide(0), then jump straight to the function's
                // end — no object collisions this tick.
                bool awayFast = -s.yvel > 5.0; // hitGround's r12 gate
                s.y = groundY;
                s.yvel = 0.0;
                s.onGround = !awayFast;
                skipObjects = true;
            }
        }
    } else if (s.ship) {
        stepShipBand(s, c);
    } else {
        stepBallBand(s, c);
    }

    if (!skipObjects) {
        // 2b. activatable objects: pads, gravity portals, mode portals.
        stepSpecials(s, c, g);
        // 2c. solids, with the rules of the (possibly just switched) mode.
        if (!s.ship && !s.ball) stepCubeSolids(s, c, g, prevY);
        else stepFlyingSolids(s, c, g, prevY,
                              s.ship ? ship::kSnapWindow : ball::kSnapWindow);
        // 2d. hazards / inner-box deaths at the resolved position.
        checkDeaths(s, c, g, prevY);
        if (s.dead) return;
    }

    // 3. input (pushButton / releaseButton). A fresh press first looks at
    // the touching rings (this tick's and last tick's contacts): if one is
    // still unused it fires INSTEAD of the ground jump; otherwise a
    // grounded cube jumps immediately. A press in the air with no ring
    // just arms m_hasJustHeld for the next ring contact. Release clears
    // both flags.
    // pushButton / releaseButton run on transitions of the PHYSICAL button
    // (inputHeld); m_isHolding (holding) can already be clear while the
    // button is still down after a ball flip or ball ring jump, and then a
    // sustained hold is a no-op until the button is released and pressed.
    if (holding && !s.inputHeld) {
        s.holding = true;      // pushButton 0x140397fbc: m_isHolding = 1
        s.hasJustHeld = true;  //                          m_hasJustHeld = 1
        const int ring = pressableRing(s);
        if (ring >= 0) {
            fireRing(s, c, g, ring);
        } else if (!s.ship && s.onGround) {
            // updateJump(0) from pushButton (0x140398036 ball / 0x14039803f
            // cube): the same jump path as the movement phase.
            s.yvel = groundJumpVel(s, c);
            s.onGround = false;
            s.hasJustHeld = false;
            if (s.ball) ballFlipAfterJump(s);
        }
    } else if (!holding && s.inputHeld) {
        s.holding = false;     // releaseButton 0x140398260: one 16-bit store
        s.hasJustHeld = false;
    }
    s.inputHeld = holding;
    if (s.sinceSwitch < (1 << 20)) ++s.sinceSwitch;
    if (s.sinceFlip < (1 << 20)) ++s.sinceFlip;
}

// collisionCheckObjects (0x214960) for the object kinds the classic levels
// use. Order of evaluation = object order within the player's x-section
// window; specials and mode portals are kept in separate lists (the obs
// contract exposes `portals`), so specials go first — no level in scope
// overlaps a pad and a mode portal on one tick.
inline void stepSpecials(PlayerState& s, const PhysicsConfig& c,
                         const LevelGeometry& g) {
    // m_touchingRings bookkeeping: this tick's contacts are collected
    // fresh; last tick's survive until after the input phase.
    for (int i = 0; i < s.nRingNow; ++i) s.ringPrev[i] = s.ringNow[i];
    s.nRingPrev = s.nRingNow;
    s.nRingNow = 0;
    // roll the one-shot activation window past everything already passed
    // (the widest box is 30 wide, so the cursor never runs past a special
    // the mini box could still touch)
    const float left = s.x - c.half;
    while (s.specialCursor < static_cast<int>(g.specials.size()) &&
           g.specials[static_cast<size_t>(s.specialCursor)].box.right < left) {
        ++s.specialCursor;
        s.specialDone >>= 1;
    }
    // The player rect is the engine's CURRENT getObjectRect(): a size
    // portal in this pass changes it for the specials tested after it
    // (collisionCheckObjects recomputes the rect per call; Clutterfunk
    // never packs two activatables into one tick, so the per-call staleness
    // is unobservable).
    for (int i = s.specialCursor; i < static_cast<int>(g.specials.size()); ++i) {
        const Special& sp = g.specials[static_cast<size_t>(i)];
        const float half = playerHalf(s, c);
        const float l = s.x - half, b = s.y - half, r = s.x + half,
                    t = s.y + half;
        if (sp.box.left > r) break;
        if (!detail::overlaps(l, b, r, t, sp.box)) continue;
        const int k = i - s.specialCursor;
        const bool done = k < 64 && ((s.specialDone >> k) & 1u);
        const double sign = s.upsideDown ? -1.0 : 1.0;
        switch (sp.kind) {
            case SpecialKind::YellowPad:
            case SpecialKind::PinkPad: {
                if (done) break;
                if (k < 64) s.specialDone |= uint64_t{1} << k;
                const float bumpMod =
                    sp.kind == SpecialKind::YellowPad ? 1.0f
                    : s.ship ? pad::kPinkBumpModShip
                    : s.ball ? pad::kPinkBumpModBall
                             : pad::kPinkBumpModCube;
                // propellPlayer 0x14039f8c9..: f32(sign) * f32(bumpMod *
                // 16) * sizeMul (0.8 for the mini)
                float v = static_cast<float>(sign) * (bumpMod * pad::kBase) *
                          sizeMul(s);
                s.yvel = roundVel(static_cast<double>(v));
                // propellPlayer 0x14039f8eb: ball / spider / swing scale the
                // result by f64(0.6f) as a raw double write.
                if (s.ball) s.yvel *= ball::kPadMul;
                s.onGround = false;
                break;
            }
            case SpecialKind::GravityPad: {
                bool target = !sp.flipped;
                if (s.upsideDown == target || done) break;
                if (k < 64) s.specialDone |= uint64_t{1} << k;
                float v = static_cast<float>(sign) *
                          (pad::kGravityBumpMod * pad::kBase) * sizeMul(s);
                s.yvel = roundVel(static_cast<double>(v));
                if (s.ball) s.yvel *= ball::kPadMul;
                s.onGround = false;
                s.upsideDown = target;
                s.yvel *= pad::kFlipDamp;
                s.sinceFlip = 0;
                break;
            }
            case SpecialKind::SizeNormal:
            case SpecialKind::SizeMini: {
                // togglePlayerScale(enable): idempotent, only the size and
                // the rect scale change (see mini::). The next specials in
                // this pass already see the new box.
                s.mini = sp.kind == SpecialKind::SizeMini;
                break;
            }
            case SpecialKind::GravityNormal:
            case SpecialKind::GravityFlip: {
                bool target = sp.kind == SpecialKind::GravityFlip;
                if (s.upsideDown == target) break;
                s.upsideDown = target;
                s.yvel *= pad::kFlipDamp;
                // flipGravity 0x14039a3cd: m_isOnGround = 0 unconditionally
                // (a grounded cube that walks into a flip portal cannot
                // jump until it lands again). Nothing else ship-specific;
                // the band snap has no post-flip grace.
                s.onGround = false;
                s.sinceFlip = 0;
                break;
            }
            case SpecialKind::YellowRing:
            case SpecialKind::PinkRing:
            case SpecialKind::GravityRing: {
                if (done) break; // hasBeenActivatedByPlayer: skipped at 0x214a8c
                // playerTouchedRing: remember the contact, then ringJump,
                // which fires only for a non-flying player holding an
                // unconsumed press (+0x989/+0x98a snapshots taken at the
                // end of this tick's update(), +0x98b once per tick).
                if (s.nRingNow < PlayerState::kMaxTouchRings) {
                    bool have = false;
                    for (int j = 0; j < s.nRingNow; ++j) have |= s.ringNow[j] == i;
                    if (!have) s.ringNow[s.nRingNow++] = i;
                }
                if (!s.ship && s.holding && s.hasJustHeld) fireRing(s, c, g, i);
                break;
            }
        }
    }

    // mode portals: rect overlap with the player's full box. Every mode
    // portal case runs toggleFlyMode(target == ship) and toggleRollMode
    // (target == ball) (0x140214e86 cube, 0x140214f61 ship via
    // switchedToMode(5), 0x1402151b5 ball); toggleFlyMode on an actual flag
    // change halves yvel (raw write, no 3-decimal rounding) and clears
    // onGround, toggleRollMode (0x39b570) changes no physics field either
    // way. So ball <-> cube keeps yvel and m_isOnGround; ship <-> anything
    // halves and ungrounds.
    const float ph = playerHalf(s, c);
    detail::forCandidates(g.portals, s.x, ph, [&](const Portal& p) {
        if (!detail::overlaps(s.x - ph, s.y - ph, s.x + ph, s.y + ph, p.box))
            return;
        const bool toShip = p.mode == PortalMode::Ship;
        const bool toBall = p.mode == PortalMode::Ball;
        // playerWillSwitchMode -> updateDualGround on every ship / ball
        // portal contact: the band is re-centered on THIS portal.
        if (toShip || toBall) s.bandFloor = p.bandFloor;
        if (s.ship != toShip) {
            s.ship = toShip;
            s.yvel *= 0.5;
            s.onGround = false;
            s.sinceSwitch = 0;
        }
        s.ball = toBall;
    }, 86.0f); // a 90-degree portal is 86 wide
}

// Cube solid resolution (collidedWithObjectInternal, cube window w = 10),
// written in gravity-relative coordinates u = sign * y so the flipped cube
// is the exact mirror: "top" faces are the ones gravity presses the player
// onto (0x140393821 lands at objTop + 15, 0x140392ec8 at objBottom - 15).
// Landing is gated on not rising (sign*yvel <= 0); eligibility is tested at
// the current and the pre-movement position.
inline void stepCubeSolids(PlayerState& s, const PhysicsConfig& c,
                           const LevelGeometry& g, float prevY) {
    const double sign = s.upsideDown ? -1.0 : 1.0;
    const double w = static_cast<double>(c.landClip);
    const float half = playerHalf(s, c); // 15, or 9 for the mini
    const double uNow = sign * static_cast<double>(s.y);
    const double uPrev = sign * static_cast<double>(prevY);
    const double bottomNow = uNow - half, bottomPrev = uPrev - half;
    auto objTopU = [&](const Box& b) {
        return sign > 0 ? static_cast<double>(b.top) : -static_cast<double>(b.bottom);
    };
    auto objBotU = [&](const Box& b) {
        return sign > 0 ? static_cast<double>(b.bottom) : -static_cast<double>(b.top);
    };
    const float l = s.x - half, b0 = s.y - half, r = s.x + half,
                t0 = s.y + half;

    if (sign * s.yvel <= 0.0) {
        // landing: highest eligible gravity-side face
        bool found = false;
        double best = 0.0;
        const Box* contacts[8];
        int nContacts = 0;
        detail::forCandidates(g.solids, s.x, half, [&](const Box& b) {
            if (!detail::overlaps(l, b0, r, t0, b)) return;
            double oTop = objTopU(b), oBot = objBotU(b);
            bool landElig = bottomNow + w >= oTop || bottomPrev + w >= oTop;
            bool overhead = oBot >= bottomNow + w || oBot >= bottomPrev + w;
            if (!(landElig && !overhead)) return;
            if (!found || oTop > best) best = oTop;
            found = true;
            if (nContacts < 8) contacts[nContacts++] = &b;
        });
        if (found) {
            // 0x14039386b: y = objTop + rect.h * 0.5 (the scaled outer box)
            s.y = static_cast<float>(sign * (best + half));
            s.yvel = 0.0;
            s.onGround = true;
            // Landing-pattern x snap (GD 2.2 jump-consistency mechanic; the
            // helper at win 2.2081 0x393cb0 runs once per bottom-contact
            // solid, in DESCENDING level-load order — the section object
            // lists append in load order and checkCollisions walks them
            // backwards; session4's 573-then-572 and session5's
            // 1060-then-1051 pairs both confirm it, and the latter rules
            // out right-to-left x order). Each call first tests whether
            // this object sits at one of the standard jump offsets from the
            // previously contacted object P (0.9x speed set: +120/+30,
            // +150/-30, +90/+60, each +-1 as float compares; the y offsets
            // are negated when upside down); on a match it pulls x toward
            // objX + the saved landing offset, clamped to +-1.0. It then
            // unconditionally replaces P with this object and recaptures the
            // offset — so a contact processed first can consume P and
            // suppress the next contact's snap. Floor contact never updates
            // P.
            for (int i = 1; i < nContacts; ++i)
                for (int j = i; j > 0 &&
                     contacts[j]->order > contacts[j - 1]->order; --j)
                    std::swap(contacts[j], contacts[j - 1]);
            const float l2 = s.x - half, b2 = s.y - half,
                        r2 = s.x + half, t2 = s.y + half;
            const float ys = static_cast<float>(sign);
            // checkSnapJumpToObject 0x140393de2: the mini's first pattern
            // is (+90, +30) instead of (+120, +30)
            const float dx1 = s.mini ? mini::kSnapDx1 : 120.0f;
            for (int i = 0; i < nContacts; ++i) {
                const Box& b = *contacts[i];
                if (objTopU(b) != best) continue;
                // overlap re-test with the current (possibly snapped) x
                if (!detail::overlaps(l2, b2, r2, t2, b)) continue;
                float ox = (b.left + b.right) * 0.5f;
                float oy = (b.bottom + b.top) * 0.5f;
                if (s.hasPrevLand &&
                    (s.prevLandX != ox || s.prevLandY != oy)) {
                    bool pat =
                        (std::fabs(ox - (s.prevLandX + dx1)) <= 1.0f &&
                         std::fabs(oy - (s.prevLandY + ys * 30.0f)) <= 1.0f) ||
                        (std::fabs(ox - (s.prevLandX + 150.0f)) <= 1.0f &&
                         std::fabs(oy - (s.prevLandY - ys * 30.0f)) <= 1.0f) ||
                        (std::fabs(ox - (s.prevLandX + 90.0f)) <= 1.0f &&
                         std::fabs(oy - (s.prevLandY + ys * 60.0f)) <= 1.0f);
                    if (pat) {
                        double target = static_cast<double>(ox) + s.landOffset;
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
    // No head hit for the classic cube: collidedWithObjectInternal's
    // bottom-face snaps are reachable only for moving objects and flying
    // modes (session4 attempt_0001 tick 5789: the cube rises 0.11 units
    // into a block bottom and keeps going). Rising cubes pass through
    // solids until the inner box intersects; the flipped cube likewise
    // passes through top faces while "falling" downward.
}

// Ship flying band (checkCollisions head: ceiling branch first, then floor;
// see the ship:: notes). hitGround runs only for motion INTO the boundary
// (ceiling: yvel > 0, floor: yvel < 0), zeroes yvel and grounds the player
// unless sign*yvel > 5 at entry — so a normal-gravity ship pinned to the
// ceiling at more than 5 up keeps whatever m_isOnGround it had. Nothing in
// flight clears the flag (only mode switches and gravity flips do), which
// is why traces show onground=1 for almost the whole flight.
inline void stepShipBand(PlayerState& s, const PhysicsConfig& c) {
    const double sign = s.upsideDown ? -1.0 : 1.0;
    // Band lines from the portal-installed floor (getMinPortalY /
    // getMaxPortalY): player center in [floor + half, floor + 300 - half]
    // (half = 15, or 9 for the mini ship; see mini::).
    const float half = playerHalf(s, c);
    const float floorY = s.bandFloor + half;
    const float ceilY = s.bandFloor + 300.0f - half;
    if (s.y > ceilY) {
        s.y = ceilY;
        if (s.yvel > 0.0) {
            if (!(sign * s.yvel > 5.0)) s.onGround = true;
            s.yvel = 0.0;
        }
    } else if (s.y < floorY) {
        s.y = floorY;
        if (s.yvel < 0.0) {
            if (!(sign * s.yvel > 5.0)) s.onGround = true;
            s.yvel = 0.0;
        }
    }
}

// Ship / ball solid phase, from collidedWithObjectInternal (see the ship::
// notes for the two blocks; the ball takes the same two blocks with the
// cube's 10-unit window, see ball::). Differences from cube:
//   - the snap window is 6 units for the ship (10 for the ball), evaluated
//     at both the current and the pre-movement position; no landing-pattern
//     x snap (checkSnapJumpToObject is called for the pure cube only).
//   - solid bottoms are symmetric ceilings: an object whose bottom is within
//     6 units of the player top (now or pre-movement) snaps y to bottom - 15
//     while yvel >= 0 and zeroes yvel.
//   - under flipped gravity the same two blocks run with the roles swapped:
//     the top snap is a head bump (yvel = 0, m_isOnGround untouched) and
//     the bottom snap is the landing (m_isOnGround = 1).
//   - deep overlaps resolve nothing; the inner-box / hazard death rules are
//     shared with cube.
// The contact list is collected once at the post-movement position
// (collisionCheckObjects) and then walked in descending load order with the
// position updated by each snap, exactly as checkCollisions does — a
// squeeze between a floor and a ceiling block therefore ends on whichever
// face the later-loaded object owns.
inline void stepFlyingSolids(PlayerState& s, const PhysicsConfig& c,
                             const LevelGeometry& g, float prevY, double w) {
    const Box* contacts[16];
    int n = 0;
    const float half = playerHalf(s, c); // 15, or 9 for the mini
    detail::forCandidates(g.solids, s.x, half, [&](const Box& b) {
        if (detail::overlaps(s.x - half, s.y - half, s.x + half, s.y + half,
                             b) && n < 16)
            contacts[n++] = &b;
    });
    for (int i = 1; i < n; ++i)
        for (int j = i; j > 0 && contacts[j]->order > contacts[j - 1]->order;
             --j)
            std::swap(contacts[j], contacts[j - 1]);
    const double bottomPrev = static_cast<double>(prevY - half);
    const double topPrev = static_cast<double>(prevY + half);
    for (int i = 0; i < n; ++i) {
        const Box& b = *contacts[i];
        const double oTop = static_cast<double>(b.top);
        const double oBot = static_cast<double>(b.bottom);
        const double bottomNow = static_cast<double>(s.y - half);
        const double topNow = static_cast<double>(s.y + half);
        // block 1: the object's top face is within the window under the
        // player's bottom (world y). A normal-gravity ship lands here; the
        // upside-down ship bumps its head.
        if (bottomNow + w >= oTop || bottomPrev + w >= oTop) {
            if (s.yvel <= 0.0) {
                s.y = b.top + half;
                // 0x140393913: bl (the block-1 window flag) picks hitHead
                // for the flipped modes, hitGround(obj) otherwise; with
                // yvel <= 0 the latter's sign*yvel > 5 gate never fires.
                s.yvel = 0.0;
                if (!s.upsideDown) s.onGround = true;
            }
            continue; // handled either way (al = 0): block 2 is skipped
        }
        // block 2: the object's bottom face is within the window above the
        // player's top. A normal-gravity ship bumps its head; the
        // upside-down ship lands (its gravity pulls it up).
        if ((oBot >= topNow - w || oBot >= topPrev - w) && s.yvel >= 0.0) {
            s.y = b.bottom - half;
            // 0x140392fcf: bl = overhead (normal gravity) -> hitHead; the
            // flipped modes land here with hitGround(obj), whose gate
            // sign*yvel > 5 is -yvel > 5, impossible with yvel >= 0.
            s.yvel = 0.0;
            if (s.upsideDown) s.onGround = true;
        }
    }
}

// Ball band (checkCollisions head, the sil path with the ball flag set):
// ceiling at getGroundHeightForMode(16) = floor + 240 (0x140211da1 ->
// .rdata 0x140623408), floor at 90. Ceiling branch (0x140213dea ->
// 0x140213da1 -> 0x140213e31): y = 315, then hitGround(null) only if
// yvel > 0. Floor branch (0x140214090 -> 0x140213fff -> 0x1402140fc): y =
// 105, then hitGround(null) unconditionally (the yvel < 0 test at
// 0x1402140f1 is reached only by the ship; the swing skips both). Neither
// snap has a post-portal grace (r12 = ceiling-height-changed is 0 in a
// plain level).
inline void stepBallBand(PlayerState& s, const PhysicsConfig& c) {
    const float half = playerHalf(s, c);
    const float floorY = s.bandFloor + half;
    const float ceilY = s.bandFloor + ball::kBandHeight - half;
    if (s.y > ceilY) {
        s.y = ceilY;
        if (s.yvel > 0.0) hitGround(s);
    } else if (s.y < floorY) {
        s.y = floorY;
        hitGround(s);
    }
}

// Deaths, evaluated at the resolved position: inner 9x9 box against solids
// (unscaled in mini mode, see mini::), the full outer box (30x30, mini
// 18x18) against rectangular hazards, playerCircleCollision against the
// circular ones (Box::radius > 0, see circleHitsPlayer). Shared by every
// mode.
//
// collidedWithObjectInternal's inner-box death (0x140393622..) has one
// escape (0x140393675..0x14039372a): within 0.1 s of a gravity flip
// (m_totalTime - [+0x800] < 0.1, .rdata 0x140622cf0) an object whose
// block-window flag bl is set is snapped instead of killing — normal
// gravity: y = objBottom - 15 (bl = overhead: objBottom >= playerBottom +
// w, now or pre-movement), flipped: y = objTop + 15 (bl = playerBottom + w
// >= objTop, now or pre-movement) — followed by hitGround(obj, 1). Only a
// deep side-on overlap right after a flip reaches it (a face within the
// window would have snapped in the blocks already); modeled here from the
// disassembly, no trace has exercised it. w is the mode's snap window.
inline void checkDeaths(PlayerState& s, const PhysicsConfig& c,
                        const LevelGeometry& g, float prevY) {
    const double w = s.ship ? ship::kSnapWindow : static_cast<double>(c.landClip);
    const bool flipGrace = s.sinceFlip < pad::kFlipFloorGraceTicks;
    const float half = playerHalf(s, c);
    detail::forCandidates(g.solids, s.x, c.inner, [&](const Box& b) {
        if (s.dead) return;
        if (!detail::overlaps(s.x - c.inner, s.y - c.inner, s.x + c.inner,
                              s.y + c.inner, b))
            return;
        if (flipGrace) {
            const double bottomNow = static_cast<double>(s.y - half);
            const double bottomPrev = static_cast<double>(prevY - half);
            const double oTop = static_cast<double>(b.top);
            const double oBot = static_cast<double>(b.bottom);
            bool bl;
            if (!s.upsideDown)
                bl = oBot >= bottomNow + w || oBot >= bottomPrev + w;
            else
                bl = bottomNow + w >= oTop || bottomPrev + w >= oTop;
            if (bl) {
                s.y = s.upsideDown ? b.top + half : b.bottom - half;
                hitGround(s);
                return;
            }
        }
        s.dead = true;
    });
    // checkCollisions' hazard loop (0x1402147c0..): every collected hazard
    // is tested against the player's outer rect, or against the circle
    // when m_objectRadius > 0 (0x1402147ed -> playerCircleCollision).
    detail::forCandidates(g.hazards, s.x, half, [&](const Box& b) {
        if (b.radius > 0.0f) {
            if (circleHitsPlayer(s.x, s.y, half, b.cx, b.cy, b.radius))
                s.dead = true;
        } else if (detail::overlaps(s.x - half, s.y - half, s.x + half,
                                    s.y + half, b)) {
            s.dead = true;
        }
    }, detail::kHazardMaxW);
}

} // namespace gdcore
