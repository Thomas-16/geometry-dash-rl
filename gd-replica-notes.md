# Building an Exact Geometry Dash Replica for RL — Notes

## Goal (settled)

Building an exact replica of Geometry Dash's mechanics, levels(only the first one for now), movement, textures, physics, etc for an environment to train a reinforcement learning agent.
The trained agent will play the **real** Geometry Dash and beat Stereo Madness. This is sim-to-real transfer, so the replica genuinely has to be exact. Every shortcut below that would be acceptable for a replica-only project is **not** acceptable here. It will probably play the real game via a Geode mod.

## Core principle

Don't just extract data once and trust it. The pipeline is:

**extract level from game files → get physics constants from the binary → verify your sim against recorded traces from the real game**

That last step is what actually earns you "exact."

---

## Where each piece of data lives

### 1. The level (easiest)

- Official levels live in `<install dir>/Resources/LevelData.plist`
- Level strings are **base64 + zlib**
- Official levels have the **first 13 characters of the base64 stripped** — prepend `H4sIAAAAAAAAA` before decompressing
- Result is the inner level string: semicolon-separated objects, each `key,value,key,value,...`
  - Key `1` = object ID
  - Key `2` = X position
  - Key `3` = Y position

**Tools:** write it yourself (~20 lines of Python), or use `gd.py` / `gddecoder` from PyPI.

**Docs:**
- https://github.com/gd-programming/gd.docs (actively maintained fork)
- https://wyliemaster.github.io/gddocs/

**First move:** decode Stereo Madness, dump the distinct object IDs. That set is your exact minimum feature list.

### 2. Physics constants (hardest)

Not in any data file. Compiled into the binary. GD is Cocos2d-x C++.

- **Geode SDK bindings** → reverse-engineered class layouts and function signatures for `PlayerObject`, `GJBaseGameLayer`, `PlayLayer`. Tells you *what* to look at.
- **Ghidra / IDA** on the binary → the actual function bodies. Key functions: `PlayerObject::update`, jump/gravity handling, collision routines.

⚠️ **Do not trust scattered forum numbers.** There is no authoritative constants table. The one dedicated physics-doc repo (`lily-pi/GeometryPhysics`) is a solo project with ~11 commits and barely covers cube gravity.

### 3. Hitboxes

Separate from sprites, and the most commonly botched part.

- Turn on the in-game hitbox display (built into 2.2)
- Screenshot-measure against known grid coordinates
- Cross-check against the binary
- Ship and wave hitboxes differ from cube

### 4. Textures

- `Resources/*.plist` + matching `.png` are standard cocos2d sprite atlases, trivially parseable

---

## The verification loop (the real answer)

Write a Geode mod that logs, **every physics tick**: player x, y, y-velocity, gravity direction, game mode.

1. Record a fixed input sequence in the real game (e.g. "hold jump at ticks 40, 97, 155"), dump the trace
2. Feed the identical input sequence into your sim
3. Diff trajectories tick by tick

This converts "is my replica exact?" from unanswerable into a **regression test**, and surfaces things reading the binary won't (e.g. float accumulation order).

---

## Gotchas

- **Physics is coupled to framerate.** Every rendered frame is a physics tick. 60 FPS = 60 physics updates/sec; 240 FPS = 240. 2.2 caps physics at **240 TPS**.
  - → "Exact" is undefined until you pick a tick rate. Pick one (60 or 240), match it, **never use variable delta time**.
- **Determinism.** Fixed timestep. Float ops in a different order ≠ same result as the C++ original.
- **Slopes, orbs, pads, portals** each have their own quirks. Scope to what Stereo Madness actually contains.

---

## Prior art

| Project | What it is | Caveat |
|---|---|---|
| [ThePickleGawd/geometry-dash-ai](https://github.com/ThePickleGawd/geometry-dash-ai) | DQN suite + gymnasium env + C++ Geode mod controlling the **real** game | macOS only; 1x realtime, no parallelism |
| [Open-GD/OpenGD](https://github.com/Open-GD/OpenGD) | C++/axmol reimplementation of GD | README says gameplay is being rewritten and levels aren't playable — verify current state |

**Your replica's value prop:** headless at 1000x with N parallel envs for *training*. The real game is still the *deployment* target, so you need both — the replica to train in, and a Geode mod to run the trained policy. `geometry-dash-ai` is worth reading specifically for how its mod hooks and controls the game, even though its training approach (real-game-only, 1x speed) is what you're avoiding.

---

## Deployment: getting the agent into the real game

A **Geode mod is the right tool.** It's the standard GD mod loader, runs on Windows and macOS, and gives you in-process hooks into `PlayLayer` / `GJBaseGameLayer` / `PlayerObject`. That means you can read state and inject inputs at exact physics-tick boundaries instead of faking OS-level clicks. Screen-capture + `SendInput`-style automation would add nondeterministic latency and defeat the whole point of an exact sim.

The same mod does double duty: it's also your trace logger for the verification loop. Build it once, use it for both.

### Architecture

Two workable shapes:

1. **Open-loop (TAS-style).** If the replica is truly exact and deterministic, and the level is deterministic, a solved input sequence from the sim replays frame-for-frame in the real game. The mod just plays back a list of tick numbers. Simplest, and it's a hard pass/fail test of whether your replica is actually exact.
2. **Closed-loop.** Mod reads live game state each tick, sends it to the policy (in-process inference, or IPC to a Python process), gets an action back. More robust to any residual sim/real mismatch, but you must keep inference inside one tick's budget.

Start with open-loop. If it beats Stereo Madness, your replica is provably exact. If it desyncs, the tick at which it desyncs tells you exactly which mechanic you got wrong.

### Consequences of the exactness requirement

- **Tick rate must match the deployment config exactly.** Decide now whether you're targeting 60 TPS or 240 TPS and train on that. A 60-TPS-trained agent will not transfer to a 240-TPS run.
- **Input model must match.** GD reads input once per physics tick. Your sim's action space should be per-tick hold/release, not "jump events" at arbitrary times.
- **No approximations anywhere in the Stereo Madness object set.** Slopes, pads, orbs, the ship section, and speed portals all need to be bit-exact, not visually plausible.

---
