#pragma once

#include "core/physics.hpp"

#include <cstdint>
#include <vector>

// Episode semantics on top of the bit-exact tick sim: decision-point macro
// actions, reward, termination. Rendering- and Python-free (shared with
// tools and the future Geode mod).
//
// Cube input only matters on the ground (a fresh click mid-air does nothing;
// flight is ballistic), so per-tick decisions bury the policy gradient in
// ~26 no-op air decisions per hop and push the fatal choice ~26 decisions
// before the death. Instead:
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
// Steps have variable duration; StepResult.ticks feeds SMDP discounting
// (gamma^ticks) in the trainer. Per-tick holds are recorded so an episode
// replays exactly.
namespace gdenv {

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
};

struct StepResult {
    float reward = 0.0f;
    int ticks = 0;           // sim ticks consumed by this decision
    bool terminated = false; // death or win: bootstrap value 0
    bool truncated = false;  // tick cap: bootstrap V(final obs)
    bool won = false;
};

struct EnvInstance {
    gdcore::PlayerState s;
    int tick = 0;
    std::vector<uint8_t> holds; // per-tick input record for this episode

    // Full default-construct is required: the landing-snap memory
    // (hasPrevLand/prevLandX/prevLandY/landOffset) must clear on respawn.
    void reset() {
        s = gdcore::PlayerState{};
        tick = 0;
        holds.clear();
    }

    StepResult step(bool hold, const gdcore::PhysicsConfig& c,
                    const gdcore::LevelGeometry& g, const EnvConfig& cfg,
                    float endX) {
        StepResult r;
        float x0 = s.x;
        auto tickOnce = [&](bool h) {
            gdcore::step(s, c, g, h);
            holds.push_back(h ? 1 : 0);
            ++tick;
            ++r.ticks;
            if (s.dead) {
                r.terminated = true;
                return true;
            }
            if (s.x >= endX) {
                r.won = true;
                r.terminated = true;
                return true;
            }
            if (tick >= cfg.maxTicks) {
                r.truncated = true;
                return true;
            }
            return false;
        };

        auto modeOf = [](const gdcore::PlayerState& p) {
            return p.ship ? 1 : p.ball ? 2 : p.bird ? 3 : 0;
        };
        if (s.bird) {
            // UFO tap macro (see above): [release if held], press, releases.
            const int startMode = modeOf(s);
            const int pressTick = s.inputHeld ? 1 : 0; // release first if held
            for (int i = 0; i < cfg.ticksPerAction; ++i) {
                if (tickOnce(hold && i == pressTick)) goto done;
                if (modeOf(s) != startMode) break;
            }
        } else if (!gdcore::flying(s) && s.onGround && hold) {
            // Hop macro: click, then ballistic flight to the next decision
            // point (landing, mode switch, or end). A ball whose press was
            // already spent (button down, m_isHolding clear) releases for
            // one tick first so the click below is a fresh press.
            if (s.ball && s.inputHeld && !s.holding) {
                if (tickOnce(false)) goto done;
            }
            bool wasFlying = false;
            while (s.onGround && !wasFlying && r.ticks < cfg.maxJumpTicks) {
                if (tickOnce(true)) goto done;
                wasFlying = gdcore::flying(s);
            }
            while (!s.onGround && !gdcore::flying(s) &&
                   r.ticks < cfg.maxJumpTicks &&
                   gdcore::pressableRing(s) < 0) {
                if (tickOnce(false)) goto done;
            }
        } else {
            // Walk / air / ship: fixed granularity; end early on a mode
            // switch so ship / UFO control starts on its own decision
            // boundary.
            const int startMode = modeOf(s);
            for (int i = 0; i < cfg.ticksPerAction; ++i) {
                if (tickOnce(hold)) goto done;
                if (modeOf(s) != startMode) break;
            }
        }
    done:
        r.reward = (s.x - x0) * cfg.progressScale;
        if (s.dead) r.reward -= cfg.deathPenalty;
        if (r.won) r.reward += cfg.winBonus;
        return r;
    }
};

} // namespace gdenv
