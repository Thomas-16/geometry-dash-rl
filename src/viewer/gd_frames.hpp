// Object-id -> sprite-frame table for the Stereo Madness, Back On Track,
// Polargeist, Dry Out, Base After Base, Can't Let Go, Jumper, Time
// Machine, Cycles and xStep object sets.
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
        // xStep: blue gravity ring (engine type 13, 36x36 like 36/141),
        // ground pit hazard 61 (engine 9x7.2), the 1.5 outline pieces
        // 69-75 (all 30x30 solids in the engine, whatever their frames
        // show), square_b_05 slab, square_08 and the lit color block 94,
        // glow clouds / balls (decoration) and three more enter-effect
        // triggers
        {84, {"gravring_01_001.png"}, 3, Tint::Obj},
        {61, {"pit_04_001.png"}, 2, Tint::Black},
        {66, {"square_b_05_001.png"}, -2, Tint::Black},
        {69, {"blockOutline_01_001.png"}, 1, Tint::Obj},
        {70, {"lightsquare_01_02_001.png"}, 1, Tint::Obj},
        {71, {"blockOutline_03_001.png"}, 1, Tint::Obj},
        {72, {"blockOutline_06_001.png"}, 1, Tint::Obj},
        {74, {"blockOutline_04_001.png"}, 1, Tint::Obj},
        {75, {"blockOutline_05_001.png"}, 1, Tint::Obj},
        {83, {"square_08_001.png"}, 1, Tint::Obj},
        {94, {"lightsquare_01_05_color_001.png"}, 1, Tint::Obj},
        {48, {"d_cloud_01_001.png"}, -1, Tint::Glow},
        {49, {"d_cloud_02_001.png"}, -1, Tint::Glow},
        {50, {"d_ball_01_001.png"}, -1, Tint::Glow},
        {51, {"d_ball_02_001.png"}, -1, Tint::Glow},
        {53, {"d_ball_04_001.png"}, -1, Tint::Glow},
        // Clutterfunk: size portals (99 = back to normal, 101 = mini),
        // sawblades 88/89/98 (circular hazards, radius 32.3 / 21.6 / 11.775),
        // lightsquare_04 pieces and the second outline set (all 30x30
        // solids in the engine), the square_d_05 decoration block (engine
        // type 7 like id 5), cogwheels and d_ball_06 (decoration). Id 104
        // (the 1.x line-color trigger) is not registered in the object
        // toolbox, so the engine never creates it.
        {99, {"portal_08_back_001.png", "portal_08_front_001.png"}, 3, Tint::Obj},
        {101, {"portal_09_back_001.png", "portal_09_front_001.png"}, 3, Tint::Obj},
        {88, {"sawblade_01_001.png"}, 2, Tint::Obj},
        {89, {"sawblade_02_001.png"}, 2, Tint::Obj},
        {98, {"sawblade_03_001.png"}, 2, Tint::Obj},
        {76, {"lightsquare_04_02_001.png"}, 1, Tint::Obj},
        {77, {"lightsquare_04_02_001.png"}, 1, Tint::Obj},
        {78, {"lightsquare_04_02_001.png"}, 1, Tint::Obj},
        {81, {"lightsquare_04_02_001.png"}, 1, Tint::Obj},
        {82, {"lightsquare_04_sideLine_001.png"}, 1, Tint::Obj},
        {80, {"square_d_05_001.png"}, 1, Tint::Obj},
        {90, {"blockOutline_01_001.png"}, 1, Tint::Obj},
        {91, {"lightsquare_01_02_001.png"}, 1, Tint::Obj},
        {92, {"blockOutline_03_001.png"}, 1, Tint::Obj},
        {93, {"blockOutline_06_001.png"}, 1, Tint::Obj},
        {95, {"blockOutline_04_001.png"}, 1, Tint::Obj},
        {96, {"blockOutline_05_001.png"}, 1, Tint::Obj},
        {85, {"d_cogwheel_01_001.png"}, -1, Tint::Glow},
        {86, {"d_cogwheel_02_001.png"}, -1, Tint::Glow},
        {87, {"d_cogwheel_03_001.png"}, -1, Tint::Glow},
        {97, {"d_cogwheel_04_001.png"}, -1, Tint::Glow},
        {60, {"d_ball_06_001.png"}, -1, Tint::Glow},
        {104, {}, 99, Tint::Obj},  // line color trigger (never created)
        {28, {}, 99, Tint::Obj},  // enter-effect: scale down (xStep)
        {58, {}, 99, Tint::Obj},  // enter-effect: from right half (xStep)
        {59, {}, 99, Tint::Obj},  // enter-effect: from right half, inverted
        {110, {"d_chain_02_001.png"}, -2, Tint::Glow},
        // Theory of Everything: the UFO portal (engine type 19, 34x86 like
        // the other mode portals), the square_f blocks (116-118/121/122 =
        // 30x30 solids, 120 = decoration like id 5), blockOutline_06 as a
        // 30x30 solid (119), the big pit_b spikes 135 (engine 14.1x20; the
        // game picks pit_b_01..04 at random, presentation only), thorns,
        // bricks, chains, wheels, clouds and sign decorations, and the
        // 1.x color trigger 105 (invisible).
        {111, {"portal_10_back_001.png", "portal_10_front_001.png"}, 3, Tint::Obj},
        {116, {"square_f_01_001.png"}, 1, Tint::Obj},
        {117, {"square_f_02_001.png"}, 1, Tint::Obj},
        {118, {"square_f_03_001.png"}, 1, Tint::Obj},
        {119, {"blockOutline_06_001.png"}, 1, Tint::Obj},
        {120, {"square_f_05_001.png"}, 1, Tint::Obj},
        {121, {"square_f_06_001.png"}, 1, Tint::Obj},
        {122, {"square_f_07_001.png"}, 1, Tint::Obj},
        {135, {"pit_b_01_001.png"}, 2, Tint::Black},
        {123, {"d_thorn_01_001.png"}, -1, Tint::Glow},
        {124, {"d_thorn_02_001.png"}, -1, Tint::Glow},
        {125, {"d_thorn_03_001.png"}, -1, Tint::Glow},
        {126, {"d_thorn_04_001.png"}, -1, Tint::Glow},
        {127, {"d_thorn_05_001.png"}, -1, Tint::Glow},
        {128, {"d_thorn_06_001.png"}, -1, Tint::Glow},
        {129, {"d_cloud_03_001.png"}, -1, Tint::Glow},
        {130, {"d_cloud_04_001.png"}, -1, Tint::Glow},
        {132, {"d_arrow_01_001.png"}, -1, Tint::Glow},
        {133, {"d_exmark_01_001.png"}, -1, Tint::Glow},
        {136, {"d_qmark_01_001.png"}, -1, Tint::Glow},
        {137, {"d_wheel_01_001.png"}, -1, Tint::Glow},
        {138, {"d_wheel_02_001.png"}, -1, Tint::Glow},
        {139, {"d_wheel_03_001.png"}, -1, Tint::Glow},
        {113, {"d_brick_01_001.png"}, -1, Tint::Glow},
        {114, {"d_brick_02_001.png"}, -1, Tint::Glow},
        {115, {"d_brick_03_001.png"}, -1, Tint::Glow},
        {106, {"d_02_chain_01_001.png"}, -2, Tint::Glow},
        {107, {"d_02_chain_02_001.png"}, -2, Tint::Glow},
        {105, {}, 99, Tint::Obj},  // 1.x color trigger (invisible)
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
