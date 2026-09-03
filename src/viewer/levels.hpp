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
// death at x 13738) matches bit-exactly. Every other entry
// renders but its sim is unverified, which the GUI labels loudly.
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
    {10, "xStep", "xStep", false},
    {11, "Clutterfunk", "Clutterfunk", false},
    {12, "Theory of Everything", "TheoryOfEverything", false},
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
