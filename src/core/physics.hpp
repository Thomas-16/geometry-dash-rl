#pragma once

namespace gdcore {

// Per-tick cube physics at 240 TPS, derived empirically from real-game traces
// (see traces/ and tools/analyze_trace.py). All constants are for the 0.9x
// ("normal") speed that Stereo Madness uses throughout.
//
// Verified bit-exact against traces at float32 position level. The game's
// internal per-tick order is:
//   1. movement (PlayerObject::update): apply gravity to yvel, integrate x/y
//   2. collision resolution (ground snap, yvel reset)
//   3. input (processCommands): a jump registers here, so it takes effect
//      on the *next* tick's movement
//
// x and y are float32 accumulators in the real game; yvel is a double.
// The game's yvel doubles occasionally differ from a plain running
// subtraction by 1 ulp (arithmetic shape unknown until the Ghidra pass), but
// this has never been observed to change a float32 position.
struct PhysicsConfig {
    float dx;        // per-tick x advance
    double gravity;  // per-tick yvel decrement
    double yFactor;  // yvel -> dy factor per tick (dt 0.25 frame-units * 0.9)
    double jumpVel;  // yvel set on jump
    float groundY;   // flat-floor height (collision v0: ground only)
};

inline PhysicsConfig stereoMadnessCube() {
    return PhysicsConfig{
        static_cast<float>(5.770002 * 0.9 * 0.25),
        0.216,
        0.225,
        11.18,
        105.0f,
    };
}

struct PlayerState {
    float x = 0.0f;
    float y = 105.0f;
    double yvel = 0.0;
    bool onGround = false;
    bool holding = false; // input state after the last processCommands
};

// Advance one 1/240s tick. `holding` is the input state processCommands sees
// at the end of this tick.
//
// Jump semantics (from traces): a fresh click registers in processCommands at
// the end of the tick (yvel set, position unchanged; gravity applies on the
// following tick). A *sustained* hold re-jumps in the movement phase on the
// tick after landing, replacing gravity for that tick.
inline void step(PlayerState& s, const PhysicsConfig& c, bool holding) {
    // movement (sees input state from the previous tick's processCommands)
    if (s.onGround && s.holding) {
        s.yvel = c.jumpVel;
        s.onGround = false;
    } else {
        s.yvel -= c.gravity;
    }
    s.y = s.y + static_cast<float>(s.yvel * c.yFactor);
    s.x = s.x + c.dx;
    // collision (flat ground only for now)
    if (s.y <= c.groundY && s.yvel < 0.0) {
        s.y = c.groundY;
        s.yvel = 0.0;
        s.onGround = true;
    } else {
        s.onGround = false;
    }
    // input: fresh click jumps immediately if grounded
    if (holding && !s.holding && s.onGround) {
        s.yvel = c.jumpVel;
        s.onGround = false;
    }
    s.holding = holding;
}

} // namespace gdcore
