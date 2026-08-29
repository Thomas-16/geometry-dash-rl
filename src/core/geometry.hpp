#pragma once

#include "core/level.hpp"

#include <vector>

namespace gdcore {

// Axis-aligned collision box in engine coordinates (engine y = level y + 90;
// the floor top is at y = 90, player center rests at 105).
struct Box {
    float left, bottom, right, top;
};

// Collision geometry for one level, built from the level string plus the
// engine-extracted hitbox table (traces/session2/objects.csv, GD 2.2081).
// Boxes are centered on the object position; rotation by an odd multiple of
// 90 degrees swaps width/height; flips never change the box.
struct LevelGeometry {
    std::vector<Box> solids;   // sorted by left
    std::vector<Box> hazards;  // sorted by left

    static LevelGeometry build(const gd::Level& level);
};

} // namespace gdcore
