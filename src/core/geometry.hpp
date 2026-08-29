#pragma once

#include "core/level.hpp"

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

struct LevelGeometry {
    std::vector<Box> solids;    // sorted by left
    std::vector<Box> hazards;   // sorted by left
    std::vector<Portal> portals; // sorted by left

    static LevelGeometry build(const gd::Level& level);
};

} // namespace gdcore
