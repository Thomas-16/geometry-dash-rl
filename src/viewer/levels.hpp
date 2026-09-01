// Official level table (GD 2.2 main levels): id -> display name, song stem
// under <Resources>/<song>.mp3, and whether the replica sim is verified for it.
// Stereo Madness and Back On Track have engine-extracted hitbox tables
// (core/geometry.cpp) and trace-verified physics (traces/session2-7,
// including full completions of both); Polargeist's jump rings are
// trace-verified too (traces/session8, 39 orb activations, 75% of the
// level). Dry Out and Base After Base add only the slab id 68 and reachable
// flip portals; traces/session9 (Dry Out to 93%) and traces/session10 (a
// full Base After Base completion) match bit-exactly. Every other entry
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
    {6, "Can't Let Go", "CantLetGo", false},
    {7, "Jumper", "Jumper", false},
    {8, "Time Machine", "TimeMachine", false},
    {9, "Cycles", "Cycles", false},
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
