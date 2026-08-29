#include "core/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace gdcore {

namespace {

// Hitbox sizes read out of the running game (GameObject::getObjectRect via
// the trace-logger mod, GD 2.2081). Covers the Stereo Madness object set;
// everything not listed is decoration/portal/trigger with no cube collision.
struct HitboxSpec {
    int id;
    char kind; // 's' solid, 'h' hazard
    float w, h;
};

constexpr HitboxSpec kHitboxes[] = {
    {1, 's', 30.0f, 30.0f},  {2, 's', 30.0f, 30.0f}, {3, 's', 30.0f, 30.0f},
    {4, 's', 30.0f, 30.0f},  {6, 's', 30.0f, 30.0f}, {7, 's', 30.0f, 30.0f},
    {40, 's', 30.0f, 14.0f}, {62, 's', 30.0f, 16.0f}, {65, 's', 30.0f, 16.0f},
    {8, 'h', 6.0f, 12.0f},   {9, 'h', 9.0f, 10.8f},  {39, 'h', 6.0f, 5.6f},
    {103, 'h', 4.0f, 7.6f},
};

const HitboxSpec* findSpec(int id) {
    for (const auto& s : kHitboxes)
        if (s.id == id) return &s;
    return nullptr;
}

} // namespace

LevelGeometry LevelGeometry::build(const gd::Level& level) {
    LevelGeometry g;
    for (const auto& obj : level.objects) {
        const HitboxSpec* spec = findSpec(obj.id);
        if (!spec) continue;
        float w = spec->w;
        float h = spec->h;
        // odd multiples of 90 degrees swap the box extents
        int rot = static_cast<int>(std::lround(obj.rotation));
        if (((rot % 180) + 180) % 180 == 90) std::swap(w, h);
        float cx = obj.x;
        float cy = obj.y + 90.0f; // level -> engine y
        Box b{cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f};
        (spec->kind == 's' ? g.solids : g.hazards).push_back(b);
    }
    auto byLeft = [](const Box& a, const Box& b) { return a.left < b.left; };
    std::sort(g.solids.begin(), g.solids.end(), byLeft);
    std::sort(g.hazards.begin(), g.hazards.end(), byLeft);
    return g;
}

} // namespace gdcore
