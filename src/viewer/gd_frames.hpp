// Object-id -> sprite-frame table for the Stereo Madness object set.
// Presentation-only viewer data (like the old shape table); NOT hitbox or
// collision truth. Mapping cross-checked against OpenGD's GameObject table
// and verified against the GJ_GameSheet-uhd / GJ_GameSheet02-uhd atlases.
#pragma once

namespace gdv {

struct RenderSpec {
    int id;
    const char* layers[2];  // frame names drawn back->front; nullptr = unused
    int defaultZ;           // z-layer used when the object has no key-24 prop
};

// Invisible gameplay objects (enter-effect ids 22/23/27, color triggers 29/30)
// have no layers; the viewer shows their debug markers only when toggled on.
inline const RenderSpec* renderSpec(int id) {
    static const RenderSpec specs[] = {
        {1, {"square_01_001.png"}, 1},
        {2, {"square_02_001.png"}, 1},
        {3, {"square_03_001.png"}, 1},
        {4, {"square_04_001.png"}, 1},
        {5, {"square_05_001.png"}, 1},
        {6, {"square_06_001.png"}, 1},
        {7, {"square_07_001.png"}, 1},
        {8, {"spike_01_001.png"}, 2},
        {39, {"spike_02_001.png"}, 2},
        {103, {"spike_03_001.png"}, 2},
        {9, {"pit_01_001.png"}, -1},
        {12, {"portal_03_back_001.png", "portal_03_front_001.png"}, 3},
        {13, {"portal_04_back_001.png", "portal_04_front_001.png"}, 3},
        {142, {"secretCoin_01_001.png"}, 3},
        {15, {"rod_01_001.png"}, -1},
        {16, {"rod_02_001.png"}, -1},
        {17, {"rod_03_001.png"}, -1},
        {18, {"d_spikes_01_001.png"}, -1},
        {19, {"d_spikes_02_001.png"}, -1},
        {20, {"d_spikes_03_001.png"}, -1},
        {21, {"d_spikes_04_001.png"}, -1},
        {40, {"plank_01_001.png"}, -1},
        {41, {"chain_01_001.png"}, -2},
        {54, {"d_ball_05_001.png"}, -1},
        {62, {"square_b_01_001.png"}, -2},
        {65, {"square_b_04_001.png"}, -2},
        {22, {}, 99},  // enter-effect: none (invisible)
        {23, {}, 99},  // enter-effect: fade
        {27, {}, 99},  // enter-effect: scale up
        {29, {}, 99},  // BG color trigger
        {30, {}, 99},  // ground color trigger
    };
    for (const auto& s : specs)
        if (s.id == id) return &s;
    return nullptr;
}

}  // namespace gdv
