#include "core/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace gdcore {

namespace {

// Hitbox sizes as the engine builds them (GD 2.2081): GameObject::commonSetup
// takes m_width/m_height from the sprite's content size and
// GameObject::customSetup (base+0x190e20) overrides a per-id rect scale for
// hazards (spike 8: 0.2 x 0.4 of 30x30 = 6x12; pit 9: 0.3 x 0.4 of 30x27).
// Solids, portals and pads keep scale 1.0. The Stereo Madness rows were read
// back from the running game (trace-logger objects.csv); the Back On Track
// rows (35, 67, 10) from the session7 dump of level 2 (pads = their atlas
// frame size, bump_01 25x4 / gravbump_01 25x6).
//
// Gravity portal (10/11): 25x75 as the engine reports it (session7
// objects.csv: rw,rh 75x25 for the -90-degree portal at x 10455).
//
// Yellow jump ring (36, Polargeist): customSetup case at 0x191c75 sets
// m_objectType 11 and the rect scale +0x394/+0x398 = 0x3f99999a (1.2f) on
// the 30x30 ring_01 frame -> 36x36, no circle radius (+0x38c stays 0), so
// contact is the same touch-inclusive CCRect test as every other special.
// Disassembly-derived; confirm against the level-3 objects.csv dump.
//
// Slab id 68 (Dry Out, square_b_06): customSetup dispatches ids 40/62/65/68
// to the same case (byte index table 0x194b44 -> case 153 at 0x140193d0f,
// the plain-solid path with no rect scale), so its hitbox is the atlas
// frame's content size: square_b_06_001 is 30x16 in GJ_GameSheet.plist,
// exactly like 62/65 (square_b_01/04); traces/session9/objects.csv confirms
// type 0, 30x16. Ids 22/26/32/33 share case 9 (0x140193ce1): enter-effect
// / trail triggers, no collision.
//
// Pink jump pad (140, Can't Let Go): customSetup case 0x140191a67 sets
// m_objectType 9 (yellow pad 35 is the twin case 0x14019190d with type 8)
// and no rect scale, so the hitbox is its sprite frame's content size:
// traces/session13/objects.csv reports type 9, 25x5 centered on the object
// (the bump_03_001 frame; bump_02_001 at 29x7 is the red pad and fired one
// tick early at the session13 pad, x 19485). Frame sizes come from the 1x
// GJ_GameSheet.plist: every engine-reported size matches that sheet where
// the -hd/-uhd sheets disagree (pit 9 = 30x27 vs 30.25x26.5, spike 39 =
// 30x14 vs 30x13.5). Id 24 is another enter-effect trigger (case 9; the
// engine lists it as type 20, 30x30, no collision).
//
// Jumper (level 7) adds a single id, 73 (square_c_05_001, 30x30 frame). The
// customSetup byte table (0x140194b44, indexed by id - 5) sends it to the
// same case as id 5: case 0 at 0x14019371e, which sets m_objectType (+0x3a0)
// to 7 = Decoration before joining the plain-block tail — id 5 is reported
// as type 7, 30x30 in every objects.csv dump and has never collided in six
// verified levels. So 73 is decoration like 5 and is deliberately absent
// from the table below. Jumper's real novelty is gravity portals inside a
// ship section (x 7565 / 8403 / 9145), handled in physics.hpp.
//
// Time Machine (level 8) adds three ids. 141 = pink jump ring: customSetup
// case 0x140191df1 is the yellow ring's twin (m_objectType 12, +0x3d0 = 1,
// rect scale +0x394/+0x398 = 1.2f) and the engine reports it EXACTLY like
// the yellow ring: type 12, m_width/m_height 30x30, rect 36x36 centered
// (traces/session15/objects.csv, uids 2686/2692). Note the ring_02_001
// frame is 32x32 in every sheet (31.5 in -hd/-uhd) - the object size is
// NOT the frame's content size here, so the 1x-sheet rule that holds for
// pads/blocks does not extend to rings; the first implementation used
// 38.4x38.4 from the sheet and only the objects.csv dump caught it (the
// session15 completion matched either way). 45 / 46 = mirror portals: cases 0x1401928af / 0x140192973 set
// m_objectType 14 / 15 (InverseMirrorPortal / NormalMirrorPortal) with the
// same field pattern as the gravity portals (no rect scale); their
// collisionCheckObjects cases only call GJBaseGameLayer::toggleFlipped,
// which changes m_isFlipped and the camera transition and spawns effects -
// nothing on the PlayerObject (levelWillFlip 0x38b6d0 touches the streak
// only) - so they have no entry below: the sim ignores them exactly like
// the enter-effect triggers (engine: type 14/15, 44x92, 2.2081 objects.csv).
// 142 (secret coin, type 22, 40x40) is in every classic level and has never
// collided.
//
// Cycles (level 9) adds two ids: 25 (another enter-effect trigger, case 9
// like 22/24/26, no collision) and 47 = the BALL PORTAL: customSetup case
// 0x1401929d6 sets m_objectType 16 (BallPortal) with exactly the mode-portal
// field pattern of 12/13 (cases 0x1401927e9 / 0x14019284c: +0x420 = 10,
// +0x464 = 5, +0x468/+0x424 = 1, the (-5, 0) +0x334 offset, the same sprite
// helper 0x140198b20 on a same-sized frame), so it is handled with the mode
// portals (34x86) rather than in the table below. The mechanic itself -
// ball mode - lives in physics.hpp (ball::).
//
// xStep (level 10) adds 19 ids (disassembly only until the level-10
// objects.csv dump / traces exist; every case below was read from
// customSetup's THREE id switches: the main one (byte table 0x140194b44
// indexed by id-5, jump table 0x1401948dc), a second one at 0x140193d1f
// (0x140195d04 / 0x140195cfc, id-22) and a third at 0x140193dd6
// (0x140195ea8 / 0x140195e9c, id-10); a late switch can overwrite what an
// early one set):
//   - 28 / 58 / 59: enter-effect triggers, case 9 like 22/24/25/26 - no
//     collision. 48 / 49 (d_cloud) and 50 / 51 / 53 (d_ball): cases
//     0x1401922eb / 0x14019228c set m_objectType 7 = Decoration.
//   - 61 = pit_04 (1109 of them, mostly rows of ground pits at engine y
//     94): the SAME case as pit 9 (0x14019237a: type 2 Hazard, rect scale
//     +0x394 = 0.3f, +0x398 = 0.4f), 61 skips the random-frame branch
//     (only ids 9 / 135 / 1715) and nothing later touches its size. So the
//     box is 0.3 * m_width x 0.4 * m_height with m_width/m_height = the
//     sprite's TRIMMED atlas rect: commonSetup (0x14018de2f) copies
//     CCSprite::m_obRect.size (+0x1b0/+0x1b4 per the Geode cocos2d layout:
//     m_obRect at +0x1a8, m_obContentSize at +0x6c) into +0x2fc/+0x300, and
//     pit_04_001 is 30x18 in the 1x GJ_GameSheet.plist (sourceSize 30x26,
//     the visible pixels sit 4 units below the center) -> 9 x 7.2 centered
//     on the object position (getObjectRect 0x1401976c0 builds the rect
//     around the object's position, never the sprite offset). Every
//     previously verified frame is untrimmed, so this is the first id where
//     trimmed-vs-content matters; OpenGD's community table also says 9 x
//     7.2. CONFIRMED by traces/session20/objects.csv (type 2, m_width /
//     m_height 30x18, rect 9 x 7.2000003 at x-4.5 / y-3.6) and by the
//     attempt_0007 pit death at x 885, bit-exact.
//   - 66 = square_b_05 (30x16 untrimmed), 83 = square_08 (30x30): plain
//     solids (case 153), like 62/65/68 and 1-7.
//   - 69 / 70 / 71 / 72 / 74 / 75 = blockOutline_01/03/04/05/06 and
//     lightsquare_01_02, the 1.5 "outline" pieces xStep builds whole
//     structures from (73, the decoration center piece, fills them). 72 is
//     already forced to 30x30 by the main switch (case 0x140193cfb), and
//     the THIRD switch sends all of 69-75 to 0x140193f74, which writes
//     m_width = m_height = 30.0f - so the trimmed 30x2 (70), 2x30 (75) and
//     2x2 (72) atlas rects never reach the hitbox: every outline piece is a
//     30x30 solid (case 153 for the type; session20 objects.csv reports
//     30x30 for all of 69/70/71/72/74/75, and attempt_0009 runs the
//     13845-14420 structure bit-exact). 94 = lightsquare_01_05_color:
//     its own case 0x1401937a6 sets m_objectType 0 (Solid) explicitly and
//     some color flags; 30x30 frame, plain solid otherwise.
//   - 84 = gravring_01 = the BLUE GRAVITY RING (type 13): case
//     0x140191eaf is the yellow ring's twin (m_objectType 13, +0x3d0 = 1,
//     rect scale 1.2f, the same ringEffect sprite helper on the same 30x30
//     frame) -> 36x36 like 36 / 141 (session20 objects.csv: type 13,
//     36x36). The mechanic lives in physics.hpp (ring:: / fireRing);
//     traces/session21 (full completion) fires five of them with the
//     ball (x 20055 / 20133 / 20211 / 20289 / 21547), each a bit-exact
//     8.944 * 0.7 * 0.5 = 3.1304 kick with the flip. The cube path (the
//     ring at x 15075 sits under a spiked ceiling row, off the normal
//     route) is disassembly-only but shares everything except the 0.7.
//
// Clutterfunk (level 11) adds 23 ids (disassembly only until the level-11
// objects.csv dump exists; every case below was read from the three
// customSetup id switches, GameObject::createWithKey's class switch
// (0x14018b7d0: byte table 0x14018d298 / jump table 0x14018d280, id-10) and
// EnhancedGameObject::customSetup (0x1401a4f70: byte table 0x1401a639c /
// jump table 0x1401a6388, id-85)):
//   - 76 / 77 / 78 / 81 (lightsquare_04_02, a 20x2 trimmed frame with a
//     20x30 source) and 82 (lightsquare_04_sideLine, 2x20): main switch
//     case 153 (plain, type 0 Solid) and the THIRD switch sends all five
//     to 0x140193f74, m_width = m_height = 30.0f - 30x30 solids exactly
//     like xStep's 69-75. 80 (square_d_05): main case 0 (0x14019371e) =
//     Decoration like ids 5 / 73 (also forced 30x30) - no collision.
//   - 90 / 91 / 92 / 93 / 95 (blockOutline_01 / lightsquare_01_02 /
//     blockOutline_03 / _06 / _04): main switch case 23 (0x14019378c) sets
//     m_objectType 0 (Solid) + the +0x411 / +0x284 flags and jumps to
//     0x140193cfb = m_width = m_height = 30.0f; 96 (blockOutline_05): case
//     25 (0x140193779), the same without +0x284. So the 2x2 / 30x2 / 2x30
//     trimmed frames never reach the hitbox: all six are 30x30 solids (the
//     same frames as 69-75 under a second color channel).
//   - 60 (d_ball_06): main case 19 (0x14019228c) = Decoration like 50-53.
//   - 85 / 86 / 87 / 97 (d_cogwheel_01/02/03/04): createWithKey builds an
//     EnhancedGameObject (case 0x14018d197); its customSetup case
//     0x1401a51cb sets type 7 Decoration and a rotation action. No
//     collision.
//   - 88 / 89 / 98 = SAWBLADES (sawblade_01/02/03): EnhancedGameObject,
//     customSetup case 0x1401a5bd9: m_objectType = 2 (Hazard) and
//     m_objectRadius (+0x38c) = m_width * 0.3f by default, then per id:
//     88 -> the constant 0x42013333 = 32.3f (0x1401a5cd5); 89 -> m_width *
//     0.36f (0x1401a5d03, .rdata 0x140622ac0) with sawblade_02_001 = 60x60
//     untrimmed -> 21.6; 98 keeps m_width * 0.3f (0x140622aa0) with
//     sawblade_03_001 = 40x40 -> 12. The rest of the case (0x1401a5c91..)
//     is the spin animation (0x1401a7280 with 300 deg/s) - the radius
//     never changes, and GameObject::updateIsOriented (0x1401a1730)
//     returns before setting m_shouldUseOuterOb when +0x38c > 0, so a
//     sawblade is a pure circle for collision: checkCollisions' hazard loop
//     (0x1402147c0..) calls GJBaseGameLayer::playerCircleCollision
//     (0x140211df0) for any object with +0x38c > 0 instead of the rect
//     test - see gdcore::circleHitsPlayer. The table stores the bounding
//     square (2r x 2r) plus the radius. None of Clutterfunk's sawblades is
//     scaled (no key 32), so the m_scaleX/m_scaleY max() in
//     playerCircleCollision is 1.0. OpenGD's community table lists 32.3 /
//     21.6 / 12; the engine's third value is 11.775 (see kSawRadius98).
//   - 99 / 101 = the SIZE PORTALS (see SpecialKind::SizeNormal / SizeMini
//     in geometry.hpp): EffectGameObjects (case 0x14018d22a, like the
//     mode / gravity portals) typed 17 / 18. Hitbox 31x90, CONFIRMED by
//     traces/session23/objects.csv (type 17 / 18, rect 31x90).
//   - 104 (the 1.x line-color trigger, placed in a stack with 29 / 30
//     triggers at y 405..525): createWithKey takes the default path
//     (GameObject::createWithFrame with the ObjectToolbox frame for the
//     key), but ObjectToolbox::init (0x140348d70) registers keys 28 / 29 /
//     30 / 105 / ... and never 104, so the frame lookup is empty and
//     createWithKey returns null (0x14018b829): the four id-104 objects do
//     not exist in the loaded level (confirmed: no id-104 row in the
//     session23 dump).
// Theory of Everything (level 12) adds 39 ids. Sizes now come from the
// engine's own per-id table, GameObject::setupSpriteSize (0x1401a36a0:
// byte table 0x1401a3fec / jump table 0x1401a3f2c, indexed by id - 1),
// which customSetup calls at its tail (0x1401946af) for every object whose
// type is not 7 / Decoration — AFTER the id cases, so it overwrites the
// sprite-rect size commonSetup copied (that is why the dump's w/h columns
// are integers while a sawblade's radius, computed inside its case, still
// carries the -uhd trimmed rect). Decoded for every classic id, it
// reproduces each verified size (see kHitboxes), so it is the authority
// (CONFIRMED by traces/session24/objects.csv: 111 = type 19, 34x86; 135 =
// type 2, m_width/m_height 47x50, rect 14.1000004 x 20):
//   - 111 = portal_10 = the UFO PORTAL: customSetup case 0x140192ba1 sets
//     m_objectType 19 with the 12 / 13 / 47 field pattern (+0x420 = 10,
//     +0x464 = 5, +0x468 / +0x424 = 1, the same sprite helper, the same
//     (-5, 0) +0x334 offset via 0x1401930ea) and setupSpriteSize case
//     0x1401a3acc -> 34x86 exactly like the other mode portals. The
//     mechanic (UFO mode) lives in physics.hpp (ufo::).
//   - 135 = pit_b_01 (the big ground spikes): main-switch case 2
//     (0x14019237a) like pits 9 / 61 — type 2 Hazard, rect scale +0x394 =
//     0.3f (width) / +0x398 = 0.4f (height) — and it is one of the three
//     ids (9 / 135 / 1715) that pick a random display frame there
//     (pit_b_%02d_001, 1..4, 0x1401923c2..0x140192498, plain cocos
//     setDisplayFrame through vtable +0x3e0). The random frame never
//     reaches the hitbox: setupSpriteSize case 0x1401a3bde then writes
//     m_width = 47, m_height = 50 for every 135 (as it writes 30x27 for
//     every pit 9, whichever of pit_01..03 it shows — 423 identical rows in
//     the Stereo Madness dump), so the box is 0.3 * 47 x 0.4 * 50 = 14.1 x
//     20 centered on the object position. OpenGD's community table agrees.
//   - 116 / 117 / 118 / 121 / 122 = square_f_01 / 02 / 03 / 06 / 07: main
//     case 153 (plain Solid) and setupSpriteSize case 0 -> 30x30. 119 =
//     blockOutline_06 (a 2x2 trimmed frame): main case 21 (0x140193cfb,
//     m_width = m_height = 30) and setupSpriteSize 30x30 — a 30x30 solid
//     like xStep's 72. 120 = square_f_05: main case 0 (0x14019371e) =
//     type 7 Decoration like ids 5 / 73 / 80 — no collision.
//   - Decoration (type 7), no collision: 15 / 16 / 17 (rod_01..03, case 7
//     0x14019222c), 41 (chain_01), 110 (d_chain_02), 113 / 114 / 115
//     (d_brick_01..03), 129 / 130 (d_cloud_03/04) — all case 8
//     (0x1401922cc -> 0x14019266e, type 7); 54 (d_ball_05, case 19), 106 /
//     107 (d_02_chain_01/02, case 18); 123..128 (d_thorn_01..06, case 29
//     0x14019131f); 132 / 133 / 136 (d_arrow_01, d_exmark_01, d_qmark_01,
//     case 30 0x14019230a); 137 / 138 / 139 (d_wheel_01..03):
//     EnhancedGameObject (createWithKey case 0x14018d197), its customSetup
//     case 0x1401a4fe2 sets type 7 plus a rotation action.
//   - Triggers, no collision: 23 / 27 (enter-effect, createWithKey class
//     case 0x14018c7be, main case 9), 29 / 30 / 105 (color triggers,
//     EffectGameObject, main case 9 like 22 / 24 / 25 / 26 / 28 / 58 / 59).
//     104 is placed seven times and never created (see above).
struct HitboxSpec {
    int id;
    char kind; // 's' solid, 'h' hazard, 'p' yellow pad, 'g' gravity pad,
               // 'n' normal-gravity portal, 'f' flip-gravity portal,
               // 'r' yellow jump ring, 'k' pink jump pad, 'q' pink jump ring,
               // 'v' gravity ring, 'c' circular hazard (w = h = 2 * radius),
               // 'N' normal-size portal, 'M' mini-size portal
    float w, h;
};

// Sawblade radii as EnhancedGameObject::customSetup computes them (float
// multiplies of the untrimmed frame width, see above).
// CONFIRMED by traces/session23/objects.csv (radius column): 32.2999992 /
// 21.6000004 / 11.7750006. The last one is NOT 40 * 0.3: the m_width that
// customSetup multiplies is the TRIMMED rect of the sheet the game actually
// loads (GJ_GameSheet-uhd, points = pixels / 4): sawblade_03_001 is 157x158
// there -> 39.25 wide (the 1x sheet's 40x40 is the untrimmed source size),
// while sawblade_02_001 is 240x240 -> 60 either way. The dump's own w/h
// column shows 40 for id 98 (rewritten later by setupPixelScale), so the
// radius must be read from the radius column, never re-derived from w.
constexpr float kSawRadius88 = 32.3f;          // 0x42013333, explicit
constexpr float kSawRadius89 = 60.0f * 0.36f;  // sawblade_02_001 60 (uhd 240/4) * 0x140622ac0
constexpr float kSawRadius98 = 39.25f * 0.3f;  // sawblade_03_001 39.25 (uhd 157/4) * 0x140622aa0

constexpr HitboxSpec kHitboxes[] = {
    {1, 's', 30.0f, 30.0f},  {2, 's', 30.0f, 30.0f}, {3, 's', 30.0f, 30.0f},
    {4, 's', 30.0f, 30.0f},  {6, 's', 30.0f, 30.0f}, {7, 's', 30.0f, 30.0f},
    {40, 's', 30.0f, 14.0f}, {62, 's', 30.0f, 16.0f}, {65, 's', 30.0f, 16.0f},
    {68, 's', 30.0f, 16.0f},
    // xStep solids: square_b_05 (30x16), the outline pieces (forced 30x30
    // by customSetup's third switch), square_08 and lightsquare_01_05_color
    {66, 's', 30.0f, 16.0f}, {69, 's', 30.0f, 30.0f}, {70, 's', 30.0f, 30.0f},
    {71, 's', 30.0f, 30.0f}, {72, 's', 30.0f, 30.0f}, {74, 's', 30.0f, 30.0f},
    {75, 's', 30.0f, 30.0f}, {83, 's', 30.0f, 30.0f}, {94, 's', 30.0f, 30.0f},
    // Clutterfunk solids: lightsquare_04 pieces (forced 30x30 by the third
    // switch) and the second set of outline pieces (forced 30x30 by their
    // own main-switch cases), see above
    {76, 's', 30.0f, 30.0f}, {77, 's', 30.0f, 30.0f}, {78, 's', 30.0f, 30.0f},
    {81, 's', 30.0f, 30.0f}, {82, 's', 30.0f, 30.0f}, {90, 's', 30.0f, 30.0f},
    {91, 's', 30.0f, 30.0f}, {92, 's', 30.0f, 30.0f}, {93, 's', 30.0f, 30.0f},
    {95, 's', 30.0f, 30.0f}, {96, 's', 30.0f, 30.0f},
    // Theory of Everything solids: square_f blocks and blockOutline_06
    // (setupSpriteSize 30x30, see above)
    {116, 's', 30.0f, 30.0f}, {117, 's', 30.0f, 30.0f}, {118, 's', 30.0f, 30.0f},
    {119, 's', 30.0f, 30.0f}, {121, 's', 30.0f, 30.0f}, {122, 's', 30.0f, 30.0f},
    {8, 'h', 6.0f, 12.0f},   {9, 'h', 9.0f, 10.8f},  {39, 'h', 6.0f, 5.6f},
    {103, 'h', 4.0f, 7.6f},
    // xStep pit_04: 0.3 x 0.4 of the trimmed 30x18 frame (see above)
    {61, 'h', 30.0f * 0.3f, 18.0f * 0.4f},
    // Theory of Everything pit_b: 0.3 x 0.4 of setupSpriteSize's 47x50
    {135, 'h', 47.0f * 0.3f, 50.0f * 0.4f},
    // Clutterfunk sawblades: circles, the box is the bounding square
    {88, 'c', 2.0f * kSawRadius88, 2.0f * kSawRadius88},
    {89, 'c', 2.0f * kSawRadius89, 2.0f * kSawRadius89},
    {98, 'c', 2.0f * kSawRadius98, 2.0f * kSawRadius98},
    {35, 'p', 25.0f, 4.0f},  {67, 'g', 25.0f, 6.0f},
    {10, 'n', 25.0f, 75.0f}, {11, 'f', 25.0f, 75.0f},
    {36, 'r', 30.0f * 1.2f, 30.0f * 1.2f},
    {140, 'k', 25.0f, 5.0f},
    {141, 'q', 30.0f * 1.2f, 30.0f * 1.2f},
    {84, 'v', 30.0f * 1.2f, 30.0f * 1.2f},
    // Clutterfunk size portals (99 normal, 101 mini)
    {99, 'N', 31.0f, 90.0f}, {101, 'M', 31.0f, 90.0f},
};

const HitboxSpec* findSpec(int id) {
    for (const auto& s : kHitboxes)
        if (s.id == id) return &s;
    return nullptr;
}

// GameObject helper at win 2.2081 base+0x1a1910 (used by the gravity-pad
// case of collisionCheckObjects): rotation truncated to int; non-multiples
// of 90 count as flipped in (90, 270) / (-270, -90); multiples of 90 count
// as flipped only at +-180; flipY inverts the answer.
bool padIsFlipped(const gd::Object& obj) {
    int rot = static_cast<int>(obj.rotation); // cvttss2si: truncation
    bool f;
    if (rot % 90 != 0) {
        f = (rot >= 91 && rot <= 269) || (rot >= -269 && rot <= -91);
        return obj.flipY ? !f : f;
    }
    int a = std::abs(rot);
    if (obj.flipY) return a != 180;
    return a == 180;
}

} // namespace

LevelGeometry LevelGeometry::build(const gd::Level& level) {
    LevelGeometry g;
    int order = 0;
    for (const auto& obj : level.objects) {
        ++order;
        bool portal = obj.id == 12 || obj.id == 13 || obj.id == 47 ||
                      obj.id == 111;
        const HitboxSpec* spec = findSpec(obj.id);
        if (!spec && !portal) continue;
        float w = portal ? 34.0f : spec->w;
        float h = portal ? 86.0f : spec->h;
        // odd multiples of 90 degrees swap the box extents
        int rot = static_cast<int>(std::lround(obj.rotation));
        if (((rot % 180) + 180) % 180 == 90) std::swap(w, h);
        float cx = obj.x;
        float cy = obj.y + 90.0f; // level -> engine y
        Box b{cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f,
              order};
        if (portal) {
            PortalMode mode = obj.id == 13    ? PortalMode::Ship
                              : obj.id == 47  ? PortalMode::Ball
                              : obj.id == 111 ? PortalMode::Ufo
                                              : PortalMode::Cube;
            // animateInDualGroundNew: floorf((startY - H/2) / 30) * 30,
            // floored at 90 (see the Portal note in geometry.hpp).
            float bandFloor = 90.0f;
            if (mode != PortalMode::Cube) {
                const float H = mode == PortalMode::Ball ? 240.0f : 300.0f;
                float v = std::floor((cy - H * 0.5f) / 30.0f) * 30.0f;
                bandFloor = std::max(90.0f, v);
            }
            g.portals.push_back(Portal{b, mode, bandFloor});
            continue;
        }
        switch (spec->kind) {
            case 's': g.solids.push_back(b); break;
            case 'h': g.hazards.push_back(b); break;
            case 'c':
                // circle centered on the object position (playerCircleCollision
                // uses getRealPosition, never the sprite offset)
                b.radius = spec->w * 0.5f;
                b.cx = cx;
                b.cy = cy;
                g.hazards.push_back(b);
                break;
            case 'N':
                g.specials.push_back(Special{b, SpecialKind::SizeNormal, false});
                break;
            case 'M':
                g.specials.push_back(Special{b, SpecialKind::SizeMini, false});
                break;
            case 'p':
                g.specials.push_back(
                    Special{b, SpecialKind::YellowPad, padIsFlipped(obj)});
                break;
            case 'g':
                g.specials.push_back(
                    Special{b, SpecialKind::GravityPad, padIsFlipped(obj)});
                break;
            case 'n':
                g.specials.push_back(Special{b, SpecialKind::GravityNormal, false});
                break;
            case 'f':
                g.specials.push_back(Special{b, SpecialKind::GravityFlip, false});
                break;
            case 'r':
                g.specials.push_back(Special{b, SpecialKind::YellowRing, false});
                break;
            case 'k':
                g.specials.push_back(
                    Special{b, SpecialKind::PinkPad, padIsFlipped(obj)});
                break;
            case 'q':
                g.specials.push_back(Special{b, SpecialKind::PinkRing, false});
                break;
            case 'v':
                g.specials.push_back(Special{b, SpecialKind::GravityRing, false});
                break;
        }
    }
    auto byLeft = [](const Box& a, const Box& b) { return a.left < b.left; };
    std::sort(g.solids.begin(), g.solids.end(), byLeft);
    std::sort(g.hazards.begin(), g.hazards.end(), byLeft);
    std::sort(g.portals.begin(), g.portals.end(),
              [](const Portal& a, const Portal& b) { return a.box.left < b.box.left; });
    std::sort(g.specials.begin(), g.specials.end(),
              [](const Special& a, const Special& b) { return a.box.left < b.box.left; });
    return g;
}

} // namespace gdcore
