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
struct HitboxSpec {
    int id;
    char kind; // 's' solid, 'h' hazard, 'p' yellow pad, 'g' gravity pad,
               // 'n' normal-gravity portal, 'f' flip-gravity portal,
               // 'r' yellow jump ring, 'k' pink jump pad, 'q' pink jump ring
    float w, h;
};

constexpr HitboxSpec kHitboxes[] = {
    {1, 's', 30.0f, 30.0f},  {2, 's', 30.0f, 30.0f}, {3, 's', 30.0f, 30.0f},
    {4, 's', 30.0f, 30.0f},  {6, 's', 30.0f, 30.0f}, {7, 's', 30.0f, 30.0f},
    {40, 's', 30.0f, 14.0f}, {62, 's', 30.0f, 16.0f}, {65, 's', 30.0f, 16.0f},
    {68, 's', 30.0f, 16.0f},
    {8, 'h', 6.0f, 12.0f},   {9, 'h', 9.0f, 10.8f},  {39, 'h', 6.0f, 5.6f},
    {103, 'h', 4.0f, 7.6f},
    {35, 'p', 25.0f, 4.0f},  {67, 'g', 25.0f, 6.0f},
    {10, 'n', 25.0f, 75.0f}, {11, 'f', 25.0f, 75.0f},
    {36, 'r', 30.0f * 1.2f, 30.0f * 1.2f},
    {140, 'k', 25.0f, 5.0f},
    {141, 'q', 30.0f * 1.2f, 30.0f * 1.2f},
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
        bool portal = obj.id == 12 || obj.id == 13 || obj.id == 47;
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
            PortalMode mode = obj.id == 13   ? PortalMode::Ship
                              : obj.id == 47 ? PortalMode::Ball
                                             : PortalMode::Cube;
            g.portals.push_back(Portal{b, mode});
            continue;
        }
        switch (spec->kind) {
            case 's': g.solids.push_back(b); break;
            case 'h': g.hazards.push_back(b); break;
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
