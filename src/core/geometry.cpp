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
struct HitboxSpec {
    int id;
    char kind; // 's' solid, 'h' hazard, 'p' yellow pad, 'g' gravity pad,
               // 'n' normal-gravity portal, 'f' flip-gravity portal,
               // 'r' yellow jump ring
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
        bool portal = obj.id == 12 || obj.id == 13;
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
            g.portals.push_back(Portal{b, obj.id == 13});
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
