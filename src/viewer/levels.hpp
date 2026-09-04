// Official level table (GD 2.2 main levels): id -> display name, song stem
// under <Resources>/<song>.mp3, and whether the replica sim is verified for it.
// Stereo Madness and Back On Track have engine-extracted hitbox tables
// (core/geometry.cpp) and trace-verified physics (traces/session2-7,
// including full completions of both); Polargeist's jump rings are
// trace-verified too (traces/session8, 39 orb activations, 75% of the
// level). Dry Out and Base After Base add only the slab id 68 and reachable
// flip portals; traces/session9 (Dry Out to 93%) and traces/session10 (a
// full Base After Base completion) match bit-exactly. Can't Let Go adds
// the pink jump pad id 140; traces/session13 (a full open-loop completion
// by the replay mod) matches bit-exactly. Jumper's only new id 73 is
// decoration; its upside-down ship comes from the disassembly (physics.hpp
// ship::) and traces/session14 (a full open-loop completion by the replay
// mod, 20433 ticks incl. 1381 upside-down ship ticks) matches bit-exactly.
// Time Machine adds the pink jump ring id 141 (ringJump x0.72 -> 8.05,
// 36x36 box like the yellow ring) and camera-only mirror portals 45/46;
// traces/session15 (a full open-loop completion by the replay mod, 22714
// ticks, all four gravity-portal pairs, both ship sections incl. the
// flipped one, and all six mirror portals) matches bit-exactly, and the
// replayed BFS solution (attempt_0002, 22717 ticks) fires both pink rings
// at the sim's exact ticks with yvel +8.05. Cycles adds the ball portal id
// 47 and ball mode (physics.hpp ball::); traces/session16 attempt_0002 (a
// human run through the whole first ball section, 3398 ball ticks with 18
// flips, three rings, the in-section flip portal and pad, ending in a ship
// death at x 13738) matches bit-exactly. xStep adds the blue gravity ring
// id 84 (ringJump x0.8 then flipGravity, physics.hpp fireRing), the pit
// hazard 61 and the 1.5 outline solids 69-75 (core/geometry.cpp), and it
// is the first level whose ship / ball portals sit high enough to RAISE
// the flying band (Portal::bandFloor: [180, 420] after the second ball
// portal, [150, 450] after the last ship portal); traces/session20 (10
// human attempts to 69%) and traces/session21 (a full macro-bot
// completion, 19262 ticks, five gravity rings fired, 1010 ball ticks
// above the old fixed ceiling) match bit-exactly. Clutterfunk adds the
// size portals 99 / 101 -> mini mode (physics.hpp mini::), the sawblades
// 88 / 89 / 98 as circle hazards and more 30x30 solids; traces/session23
// (runs/level11-1's policy replayed open-loop in the real game: a full
// completion, 22714 ticks bit-exact in both mods, 8942 mini-cube, 1683
// mini-ship and 3908 mini-ball ticks over nine size portals, closest
// sawblade approach 1.7 units) matches bit-exactly, and its objects.csv
// confirms 31x90 portals and the 32.3 / 21.6 / 11.775 radii. Theory of
// Everything adds the UFO portal id 111 and UFO mode (physics.hpp ufo::),
// the pit_b hazard 135 (14.1x20 from GameObject::setupSpriteSize) and
// more 30x30 solids (core/geometry.cpp); traces/session24 (a full
// completion, 19681 ticks incl. 4802 UFO ticks, both UFO sections, the
// mini UFO) matches bit-exactly after the cached-collision-rect fix
// (PlayerState::rectMini). Every other entry renders but its sim is
// unverified, which the GUI labels loudly.
#pragma once

#include <cstddef>

namespace gdv {

struct LevelInfo {
    int id;
    const char* name;
    const char* song;
    bool verified;
};

inline constexpr LevelInfo kLevels[] = {
    {1, "Stereo Madness", "StereoMadness", true},
    {2, "Back On Track", "BackOnTrack", true},
    {3, "Polargeist", "Polargeist", true},
    {4, "Dry Out", "DryOut", true},
    {5, "Base After Base", "BaseAfterBase", true},
    {6, "Can't Let Go", "CantLetGo", true},
    {7, "Jumper", "Jumper", true},
    {8, "Time Machine", "TimeMachine", true},
    {9, "Cycles", "Cycles", true},
    {10, "xStep", "xStep", true},
    {11, "Clutterfunk", "Clutterfunk", true},
    {12, "Theory of Everything", "TheoryOfEverything", true},
    {13, "Electroman Adventures", "Electroman", false},
    {14, "Clubstep", "Clubstep", false},
    {15, "Electrodynamix", "Electrodynamix", false},
    {16, "Hexagon Force", "HexagonForce", false},
    {17, "Blast Processing", "BlastProcessing", false},
    {18, "Theory of Everything 2", "TheoryOfEverything2", false},
    {19, "Geometrical Dominator", "GeometricalDominator", false},
    {20, "Deadlocked", "Deadlocked", false},
    {21, "Fingerdash", "Fingerdash", false},
    {22, "Dash", "Dash", false},
};
inline constexpr size_t kLevelCount = sizeof(kLevels) / sizeof(kLevels[0]);

inline const LevelInfo* levelInfo(int id) {
    for (const auto& l : kLevels)
        if (l.id == id) return &l;
    return nullptr;
}

}  // namespace gdv
