// Object-id -> sprite-frame table for the Stereo Madness, Back On Track,
// Polargeist, Dry Out, Base After Base, Can't Let Go, Jumper and Time
// Machine object sets.
// Presentation-only viewer data (like the old shape table); NOT hitbox or
// collision truth. Mapping cross-checked against OpenGD's GameObject table
// and verified against the GJ_GameSheet-uhd / GJ_GameSheet02-uhd atlases.
#pragma once

namespace gdv {

// How the (mostly white) atlas sprite is colored in-game. Channel mapping from
// OpenGD's object.json: "black" -> channel 1010 (black), "detail" on ids whose
// primary channel is 1005/1006 -> white with additive blending (soft glow),
// everything else -> obj color (kS32, white in Stereo Madness).
enum class Tint { Obj, Black, Glow };

struct RenderSpec {
    int id;
    const char* layers[2];  // frame names drawn back->front; nullptr = unused
    int defaultZ;           // z-layer used when the object has no key-24 prop
    Tint tint;
};

// Invisible gameplay objects (enter-effect ids 22/23/27, color triggers 29/30)
// have no layers; the viewer shows their debug markers only when toggled on.
inline const RenderSpec* renderSpec(int id) {
    static const RenderSpec specs[] = {
        {1, {"square_01_001.png"}, 1, Tint::Obj},
        {2, {"square_02_001.png"}, 1, Tint::Obj},
        {3, {"square_03_001.png"}, 1, Tint::Obj},
        {4, {"square_04_001.png"}, 1, Tint::Obj},
        {5, {"square_05_001.png"}, 1, Tint::Obj},
        {6, {"square_06_001.png"}, 1, Tint::Obj},
        {7, {"square_07_001.png"}, 1, Tint::Obj},
        // Jumper: decoration block (engine type 7 like id 5, no hitbox)
        {73, {"square_c_05_001.png"}, 1, Tint::Obj},
        {8, {"spike_01_001.png"}, 2, Tint::Obj},
        {39, {"spike_02_001.png"}, 2, Tint::Obj},
        {103, {"spike_03_001.png"}, 2, Tint::Obj},
        // z 2 like spikes (OpenGD: z_layer 5) so it draws over the glow deco
        {9, {"pit_01_001.png"}, 2, Tint::Black},
        {12, {"portal_03_back_001.png", "portal_03_front_001.png"}, 3, Tint::Obj},
        {13, {"portal_04_back_001.png", "portal_04_front_001.png"}, 3, Tint::Obj},
        // Cycles: ball portal (engine type 16, 34x86 like the other mode portals)
        {47, {"portal_07_back_001.png", "portal_07_front_001.png"}, 3, Tint::Obj},
        // Back On Track: gravity portals (10 blue = normal, 11 yellow = flip),
        // yellow jump pad, blue gravity pad
        {10, {"portal_01_back_001.png", "portal_01_front_001.png"}, 3, Tint::Obj},
        {11, {"portal_02_back_001.png", "portal_02_front_001.png"}, 3, Tint::Obj},
        {35, {"bump_01_001.png"}, 2, Tint::Obj},
        {67, {"gravbump_01_001.png"}, 2, Tint::Obj},
        // Can't Let Go: pink jump pad (bump_03 is the pink 25x5 frame; bump_02
        // is the red pad)
        {140, {"bump_03_001.png"}, 2, Tint::Obj},
        // Time Machine: mirror portals (45 = mirror on, 46 = mirror off;
        // camera-only, no hitbox in the sim) and the pink jump ring
        {45, {"portal_05_back_001.png", "portal_05_front_001.png"}, 3, Tint::Obj},
        {46, {"portal_06_back_001.png", "portal_06_front_001.png"}, 3, Tint::Obj},
        {141, {"ring_02_001.png"}, 3, Tint::Obj},
        // Polargeist: yellow jump ring, glow chain
        {36, {"ring_01_001.png"}, 3, Tint::Obj},
        {110, {"d_chain_02_001.png"}, -2, Tint::Glow},
        {142, {"secretCoin_01_001.png"}, 3, Tint::Obj},
        {15, {"rod_01_001.png"}, -1, Tint::Obj},
        {16, {"rod_02_001.png"}, -1, Tint::Obj},
        {17, {"rod_03_001.png"}, -1, Tint::Obj},
        {18, {"d_spikes_01_001.png"}, -1, Tint::Glow},
        {19, {"d_spikes_02_001.png"}, -1, Tint::Glow},
        {20, {"d_spikes_03_001.png"}, -1, Tint::Glow},
        {21, {"d_spikes_04_001.png"}, -1, Tint::Glow},
        {40, {"plank_01_001.png"}, -1, Tint::Obj},
        {41, {"chain_01_001.png"}, -2, Tint::Glow},
        {54, {"d_ball_05_001.png"}, -1, Tint::Glow},
        {62, {"square_b_01_001.png"}, -2, Tint::Black},
        {65, {"square_b_04_001.png"}, -2, Tint::Black},
        {68, {"square_b_06_001.png"}, -2, Tint::Black},  // Dry Out slab step
        {22, {}, 99, Tint::Obj},  // enter-effect: none (invisible)
        {23, {}, 99, Tint::Obj},  // enter-effect: fade
        {24, {}, 99, Tint::Obj},  // enter-effect: from bottom (Can't Let Go)
        {26, {}, 99, Tint::Obj},  // enter-effect: from right (Base After Base)
        {27, {}, 99, Tint::Obj},  // enter-effect: scale up
        {29, {}, 99, Tint::Obj},  // BG color trigger
        {30, {}, 99, Tint::Obj},  // ground color trigger
        {32, {}, 99, Tint::Obj},  // line/ground toggle triggers (type 20)
        {33, {}, 99, Tint::Obj},
    };
    for (const auto& s : specs)
        if (s.id == id) return &s;
    return nullptr;
}

}  // namespace gdv
