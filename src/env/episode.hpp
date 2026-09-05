#pragma once

#include "core/physics.hpp"
#include "env/contracts.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

// Episode semantics on top of the bit-exact tick sim: decision-point
// actions, reward, termination. Rendering- and Python-free (shared with
// tools and the Geode mod).
//
// Two action spaces (ActionMode, contracts.hpp), ONE executor
// (EnvInstance::act) used by rollouts, the planner, evaluation and replay:
//
// Macro (legacy, 2 actions). Cube input only matters on the ground (a fresh
// click mid-air does nothing; flight is ballistic), so per-tick decisions
// bury the policy gradient in ~26 no-op air decisions per hop and push the
// fatal choice ~26 decisions before the death. Instead:
//   - on the ground, action 1 = one whole hop as ONE env step (hold until
//     liftoff, release, roll ticks until landing/death/win/mode switch —
//     or until a jump ring becomes pressable, which is the one mid-air
//     moment where a click matters)
//   - action 0 = walk ticksPerAction ticks
//   - airborne cube or ship mode = hold/release for ticksPerAction ticks
//     (ship is continuous control; a held cube buffers a landing re-jump,
//     exactly like the real game; a fresh hold inside a ring fires it, a
//     hold started before the ring fires it on contact)
//   - ball mode (Cycles) is the cube's shape: on a surface, action 1 = one
//     gravity flip as ONE step (the press flips, then the ball rolls
//     through the air to the other surface; a ring on the way ends the
//     step). The flip spends m_isHolding while the button stays down, so a
//     grounded ball whose button is still down first releases for one tick
//     (the real game needs release + press to flip again). Airborne, hold
//     buffers a flip on landing exactly like the cube's re-jump.
//   - UFO mode (Theory of Everything) is tap-driven: a fresh press boosts
//     once (m_hasJustHeld is consumed) and holding does nothing more, so
//     action 1 = one TAP over ticksPerAction ticks — a release tick first
//     if the button is still down from a previous mode, then a one-tick
//     press, then releases — and action 0 = release for ticksPerAction
//     ticks. Every UFO decision therefore ends with the button up, and a
//     tap at a decision where a ring is pressable fires the ring instead
//     of boosting (pushButton's order), exactly like the real game.
//
// Precision (7 actions, docs/rl-improvement-plan.md): release or hold the
// physical button for 1, 2 or 4 ticks (R1 R2 R4 H1 H2 H4) plus the grounded
// cube / ball hop macro above (HOP, masked off unless !flying && onGround).
// The raw actions keep the button's physical semantics — a UFO that is
// already holding does NOT re-boost on another H action; it needs an R
// then an H, exactly like the real game — so every input sequence the mod
// can inject is expressible, at one-tick resolution where it matters. A
// repeated action (R2 R4 H2 H4) is interrupted after the tick in which the
// player lands, changes mode / size / gravity, or a ring becomes
// pressable, so the next decision sits on the event. The one-tick actions
// are always permitted.
//
// Steps have variable duration; StepResult.ticks feeds SMDP discounting
// (gamma^ticks) in the trainer. With EnvConfig::rewardGamma > 0 the reward
// is the discounted per-tick sum inside the step (sum_k gamma^k r_k, the
// precision-mode contract); 0 keeps the legacy flat sum. Per-tick holds and
// per-decision action ids are both recorded so an episode replays exactly
// from either (raw holds for deployment, action ids for the trainer's
// decision streams — raw holds cannot recover variable-duration choices).
namespace gdenv {

enum ActionMode : int {
    kMacro = kActionModeMacro,
    kPrecision = kActionModePrecision,
};

// Precision action ids.
enum PrecisionAction : int {
    kR1 = 0, kR2 = 1, kR4 = 2, kH1 = 3, kH2 = 4, kH4 = 5, kHop = 6,
    kNumPrecisionActions = 7,
};
constexpr int kNumMacroActions = 2;

inline int numActions(int mode) {
    return mode == kPrecision ? kNumPrecisionActions : kNumMacroActions;
}

struct EnvConfig {
    int ticksPerAction = 4; // walk/ship granularity (one 60 fps frame)
    // Safety net; x advances unconditionally, so this never fires in
    // practice (full runs: Stereo Madness ~20,354 ticks, Polargeist
    // ~21,330).
    int maxTicks = 30000;
    int maxJumpTicks = 300;             // hop macro hard cap (one hop ~104)
    float progressScale = 1.0f / 30.0f; // reward per block of x progress
    float deathPenalty = 5.0f;
    float winBonus = 50.0f;
    int actionMode = kMacro;
    // > 0: reward = sum_k gamma^k r_k over the step's ticks (r_k = per-tick
    // progress, the death / win terms at the tick they happen). 0: the
    // legacy flat sum (progress over the whole step, then death / win).
    double rewardGamma = 0.0;
};

struct StepResult {
    float reward = 0.0f;
    int ticks = 0;           // sim ticks consumed by this decision
    bool terminated = false; // death or win: bootstrap value 0
    bool truncated = false;  // tick cap: bootstrap V(final obs)
    bool won = false;
};

inline int modeOf(const gdcore::PlayerState& p) {
    return p.ship ? 1 : p.ball ? 2 : p.bird ? 3 : 0;
}

// Per-action legality for the current state, written as 0/1 into
// out[numActions(mode)].
inline void actionMask(const gdcore::PlayerState& s, int mode, uint8_t* out) {
    if (mode != kPrecision) {
        out[0] = out[1] = 1;
        return;
    }
    for (int a = 0; a < kNumPrecisionActions; ++a) out[a] = 1;
    out[kHop] = (!gdcore::flying(s) && s.onGround && !s.dead) ? 1 : 0;
}

struct EnvInstance {
    gdcore::PlayerState s;
    int tick = 0;
    std::vector<uint8_t> holds;   // per-tick input record for this episode
    std::vector<uint8_t> actions; // per-decision action ids (executor input)

    // Full default-construct is required: the landing-snap memory
    // (hasPrevLand/prevLandX/...) must clear on respawn.
    void reset() {
        s = gdcore::PlayerState{};
        tick = 0;
        holds.clear();
        actions.clear();
    }

    // Legacy entry point: the macro action space's hold bit.
    StepResult step(bool hold, const gdcore::PhysicsConfig& c,
                    const gdcore::LevelGeometry& g, const EnvConfig& cfg,
                    float endX) {
        return act(hold ? 1 : 0, c, g, cfg, endX);
    }

    // THE executor. `action` is an id of cfg.actionMode's space; an illegal
    // action (HOP while not grounded) is executed as H1 — callers mask it
    // out before sampling, so this is only a safety net for stale streams.
    StepResult act(int action, const gdcore::PhysicsConfig& c,
                   const gdcore::LevelGeometry& g, const EnvConfig& cfg,
                   float endX) {
        StepResult r;
        const float x0 = s.x;
        const bool disc = cfg.rewardGamma > 0.0;
        double acc = 0.0;    // discounted per-tick reward (rewardGamma > 0)
        double gpow = 1.0;   // gamma^k for the next tick
        actions.push_back(static_cast<uint8_t>(action));
        auto tickOnce = [&](bool h) {
            const float xb = s.x;
            gdcore::step(s, c, g, h);
            holds.push_back(h ? 1 : 0);
            ++tick;
            ++r.ticks;
            bool end = false;
            if (s.dead) {
                r.terminated = true;
                end = true;
            } else if (s.x >= endX) {
                r.won = true;
                r.terminated = true;
                end = true;
            } else if (tick >= cfg.maxTicks) {
                r.truncated = true;
                end = true;
            }
            if (disc) {
                double rk = static_cast<double>((s.x - xb) * cfg.progressScale);
                if (s.dead) rk -= cfg.deathPenalty;
                if (r.won) rk += cfg.winBonus;
                acc += gpow * rk;
                gpow *= cfg.rewardGamma;
            }
            return end;
        };

        if (cfg.actionMode == kPrecision) {
            if (action == kHop && !(!gdcore::flying(s) && s.onGround))
                action = kH1;
            if (action == kHop) {
                hopMacro(tickOnce, cfg);
            } else {
                const bool hold = action >= kH1;
                const int n = (action == kR1 || action == kH1) ? 1
                              : (action == kR2 || action == kH2) ? 2 : 4;
                const bool ground0 = s.onGround;
                const int mode0 = modeOf(s);
                const bool mini0 = s.mini;
                const bool grav0 = s.upsideDown;
                const bool ring0 = gdcore::pressableRing(s) >= 0;
                for (int i = 0; i < n; ++i) {
                    if (tickOnce(hold)) break;
                    if (i + 1 >= n) break;
                    // Interrupts: the next decision sits on the event.
                    if ((s.onGround && !ground0) || modeOf(s) != mode0 ||
                        s.mini != mini0 || s.upsideDown != grav0 ||
                        (!ring0 && gdcore::pressableRing(s) >= 0))
                        break;
                }
            }
        } else {
            const bool hold = action != 0;
            if (s.bird) {
                // UFO tap macro (see above): [release if held], press,
                // releases.
                const int startMode = modeOf(s);
                const int pressTick = s.inputHeld ? 1 : 0;
                for (int i = 0; i < cfg.ticksPerAction; ++i) {
                    if (tickOnce(hold && i == pressTick)) break;
                    if (modeOf(s) != startMode) break;
                }
            } else if (!gdcore::flying(s) && s.onGround && hold) {
                hopMacro(tickOnce, cfg);
            } else {
                // Walk / air / ship: fixed granularity; end early on a mode
                // switch so ship / UFO control starts on its own decision
                // boundary.
                const int startMode = modeOf(s);
                for (int i = 0; i < cfg.ticksPerAction; ++i) {
                    if (tickOnce(hold)) break;
                    if (modeOf(s) != startMode) break;
                }
            }
        }
        if (disc) {
            r.reward = static_cast<float>(acc);
        } else {
            r.reward = (s.x - x0) * cfg.progressScale;
            if (s.dead) r.reward -= cfg.deathPenalty;
            if (r.won) r.reward += cfg.winBonus;
        }
        return r;
    }

private:
    // Hop macro: click, then ballistic flight to the next decision point
    // (landing, mode switch, pressable ring, or end). A ball whose press
    // was already spent (button down, m_isHolding clear) releases for one
    // tick first so the click below is a fresh press.
    template <class TickFn>
    void hopMacro(TickFn& tickOnce, const EnvConfig& cfg) {
        int n = 0;
        auto t = [&](bool h) { ++n; return tickOnce(h); };
        if (s.ball && s.inputHeld && !s.holding) {
            if (t(false)) return;
        }
        bool wasFlying = false;
        while (s.onGround && !wasFlying && n < cfg.maxJumpTicks) {
            if (t(true)) return;
            wasFlying = gdcore::flying(s);
        }
        while (!s.onGround && !gdcore::flying(s) && n < cfg.maxJumpTicks &&
               gdcore::pressableRing(s) < 0) {
            if (t(false)) return;
        }
    }
};

// Replaying raw per-tick holds through the executor: the action at a
// decision boundary is the recorded bit at that tick, with one exception
// per macro that records a release tick before its press (a grounded ball
// whose press is already spent; a UFO whose button is still down from a
// previous mode): [0, 1] there is action 1. In precision mode a raw hold
// stream replays as one-tick actions (R1 / H1), which is bit-exact but a
// different decision stream than the one that produced it — decision
// streams are stored explicitly for that reason.
inline int actionForHolds(const gdcore::PlayerState& s, int mode,
                          const uint8_t* h, int64_t cursor, int64_t len) {
    const bool bit = h[cursor] != 0;
    if (mode == kPrecision) return bit ? kH1 : kR1;
    bool hold = bit;
    if (!hold && s.ball && !s.ship && s.onGround && s.inputHeld &&
        !s.holding && cursor + 1 < len && h[cursor + 1] != 0)
        hold = true;
    if (!hold && s.bird && s.inputHeld && cursor + 1 < len &&
        h[cursor + 1] != 0)
        hold = true;
    return hold ? 1 : 0;
}

} // namespace gdenv
