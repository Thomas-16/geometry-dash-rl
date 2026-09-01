# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Exact Geometry Dash replica for RL training (sim-to-real: train in the replica, deploy to the real game via a Geode mod). Roadmap, physics-extraction strategy, and the trace-verification loop are in @gd-replica-notes.md — read it before making design decisions.

## Build

MSYS2 UCRT64 toolchain (g++/cmake/ninja on PATH):

```
cmake -S . -B build -G Ninja && cmake --build build
```

- **Do not remove `-static` from CMakeLists.txt.** This machine's MinGW ld (binutils 2.43.1) crashes with a silent "ld returned 116 exit status" when linking C++ exception code against shared runtimes. If a new target fails with exit 116, static linking is the fix.
- raylib and zlib come from MSYS2 packages, not FetchContent.

## Viewer GUI (src/viewer)

- raylib window + a Dear ImGui panel on the right (rlImGui backend). **Deps are FetchContent'd at configure time** (`build/_deps`): ImGui `v1.92.7`, rlImGui tag `Raylib_5_5` (its `main` targets raylib 6.0 — do not bump one without the other; the 5.5 tag needs ImGui >= 1.92), nlohmann/json 3.11.3. First configure needs network; `-DVIEWER_VENDORED_DEPS=<dir>` with `imgui/`, `rlImGui/`, `json/` checkouts builds offline.
- Layout: the game viewport is `App::viewW x viewH` anchored at (0,0); every camera/background site uses `viewW(app)/viewH(app)`, never `GetScreenWidth()`. Mode input goes through `viewer/input.hpp` wrappers (gated on ImGui capture) — never raw `IsKeyPressed` in a mode. Modes draw their own controls in `Mode::drawPanel`; cross-tab requests go through `App::commands` (consumed in `main.cpp`).
- The panel launches `python -u -m gdrl.*` (cwd `python/`, absolute paths) and the tool exes via `viewer/jobs.cpp` (Win32 pipe + Job Object, children die with the viewer). `jobs.cpp`/`paths.cpp` include windows.h and must never include raylib.h.
- **Train-tab defaults mirror `PPOConfig`/`train.py`** (`Gui::TrainForm` in `gui.hpp`); change them in both places. Only non-default flags are emitted.
- **Graceful stop**: the GUI writes `runs/<name>/STOP`; `train.py` checks it once per update, saves `ckpt_latest.pt`, prints `stopped: ...` and exits 0 (also on Ctrl-C). Stale STOP files are unlinked at startup.
- Pipeline: a `| det ... WIN` line auto-launches `gdrl.eval` on `ckpt_best_det.pt` (once, plus a final eval at exit); Deploy is always a confirmed click.
- **The trainer's `upd` log line is parsed by a regex** (`Gui::pollPipelines`, `gui.cpp`) — a field added to the print in `train.py` needs the regex + `TrainStatus` updated too. `live.bin` is v2 (`python/gdrl/live.py`; header stays 40 bytes): `sps`/`samples_s` are the *last completed update's* rates, identical to the log line — never publish a mid-rollout running rate again (it excluded `t_update`/`t_sil` and read ~6x high).
- Frame pacing is driven from `main.cpp` via `Mode::targetFps()` (Play = 240, everything else 60), halved to 30 while a panel-launched python job runs and the window is unfocused. Don't call `SetTargetFPS` inside modes. The viewer costs the trainer ~7% decisions/s when open on the Training view at 60 FPS (GPU-side contention: `t_update`/`t_roll_gpu`), ~0% closed. **`App::livePreview`** (checkbox "Live preview" in the Train tab and the Training view's panel, pref `live_preview`) turns the preview off entirely: no `live.bin` polling, no world/ghost draw, 30 FPS (15 unfocused) — measured at parity with the viewer closed; the Train-tab stats/plots still update from the stdout scrape.
- Prefs persist in `<repo>/viewer_prefs.ini` (gitignored); repo root is found from the exe path. Automated captures: `viewer --shot out.png [--panel] [--tab N]` (0 View .. 4 Jobs; `--shot` never persists prefs). "Rebuild incl. viewer" renames the running `viewer.exe` to `.old` first (deleted on next start).

## Level data

- GD install: `S:\SteamLibrary\steamapps\common\Geometry Dash\`
- Official levels: `Resources\levels\<id>.txt` (Stereo Madness = `1.txt`), URL-safe base64 of gzip. The `LevelData.plist` method described in older docs/notes is obsolete as of GD 2.2.
- **Never commit level data or any extracted game assets** — they're RobTop's content. Always read from the install at runtime.
- Smoke test: `./build/dump.exe` must report 2291 objects / 31 distinct ids for Stereo Madness; `./build/dump.exe <install>/Resources/levels/2.txt` 1545 objects / 26 ids for Back On Track.
- Tools take the level from `GD_LEVEL_PATH` (default `levels/1.txt`); the trainer/eval take `--level <path>` and record `level_id` from the file stem.

## Back On Track (level 2) mechanics

Implemented from the 2.2081 disassembly (`GJBaseGameLayer::collisionCheckObjects` 0x214960, `bumpPlayer` 0x2179d0, `PlayerObject::propellPlayer` 0x39f850, `flipGravity` 0x39a1d0, `updateJump` 0x38b900, `collidedWithObjectInternal` 0x391a70) and **trace-verified bit-exact** (`traces/session7`, 6 attempts incl. a full completion of 19,269 gameplay ticks; hitboxes from `traces/session7/objects.csv`). Regression: `GD_LEVEL_PATH=<levels/2.txt> build/trace_check.exe traces/session7/attempt_*.csv` (it also checks the `upsidedown` column) — run it together with the SM traces after touching the sim.

- New ids vs SM: **35 yellow pad** (25x4, `propellPlayer(1.0)` → yvel = ±16 along the current gravity), **67 blue gravity pad** (25x6, yvel = ±12.8 then `flipGravity` → yvel halved raw → 6.4, upside down), **10 blue gravity portal** (25x75, back to normal gravity, yvel halved). Id 5 is type 7 (decoration, no collision) — engine-confirmed. Hitbox = sprite frame size × per-id scale from `customSetup` (spike 0.2×0.4); pads have no scale.
- Tick order is now the real one (`gdcore::step`): movement → floor/band snap → **specials (pads, gravity portals, mode portals) act immediately** → solids (with the new mode's rules) → hazards → input. Pads are one-shot (`PlayerState::specialCursor/specialDone` rolling window).
- **Cube gravity is `f64(f32(0.225f*0.96f)) = 0.2159999907`, not `f32(0.216)`** — the ulp only decides 3-decimal rounding ties, which appear after a mode portal halves an odd-thousandth yvel raw (session7 attempt_0002 tick 12894). Keep the expression, not a literal.
- Upside-down cube: gravity/jump/fall-clamp signs flip, "landing" is on an object's bottom face (`y = bottom − 15`), there is **no ceiling** for the classic cube (it rises until it meets a solid), and neither cube orientation has a head hit against static solids (session4 tick 5789: the cube rises 0.11 into a block bottom and keeps going). The landing-pattern x snap negates its y offsets when flipped. Ship + upside down is not modeled (no level in scope).
- The flipped section (x 10282–10600) plays out with zero input: slab → gravity pad → rise to the ceiling portal at y≈411 → fall onto the block column → yellow pad over the six spikes. `env_check trace <spec> 10200 10900` prints it.
- Obs v3 (`kObsVersion` 3, `kPlayerDim` 24): scalar 17 = gravity flag, 18–23 = next two specials (dist, dy, kind ±1 pad / ±0.5 gravity portal). Grid channels unchanged. `model.load_model_state` zero-pads older checkpoints' first input layer (v2 SM checkpoints still win deterministically); the optimizer state is reset on migration.

## Architecture rules

- `gdcore` (src/core) must stay free of rendering dependencies — it becomes the headless training sim. Rendering-only concerns (e.g., the object-id → shape table) live in the viewer.
- The viewer's shape table is presentation-only; never treat it as hitbox/collision data. Hitboxes come later from the binary (see notes).
- Future sim work: fixed timestep only, never variable delta time (determinism requirement, see notes).

## RL env contract (src/env + python/gdrl)

- **Observation versioning** (`src/env/obs.hpp`): the scalar block and grid layout are frozen; changes are additive-only and bump `kObsVersion` (checked by `python/gdrl/env.py` at DLL load; `model.load_model_state` migrates older checkpoints by zero-padding the appended scalars). v3 = scalars 0–23 (17 gravity flag, 18–23 next two pads/gravity portals); changing the grid y-band is an obs v4 decision. `obs.hpp` will be `#include`d verbatim by the Geode mod — compile-time constants only, no runtime normalization.
- **Speed portals later change an input, not the network**: per-tick `dx` is already an obs scalar and a `buildObs` parameter.
- **Env stepping is parallel and bit-deterministic**: `$GDENV_THREADS` total threads (1 = exact serial fallback), default `hardware_concurrency()-1`. After touching the env layer, run `./build/env_check det` AND `python -m gdrl.selftest` (from `python/`; serial-vs-threaded byte-identity).
- **Regression bar for RL changes**: a default `python -m gdrl.train` run must still beat Stereo Madness (deterministic `det ... WIN` line / `wins.gdtrace`), and `python -m gdrl.eval runs/<name>/ckpt_best_det.pt` must write a `solution.json`. Don't shrink `--updates` for the check — it compresses the lr/entropy anneal and changes the learning trajectory.
- Hyperparameter defaults live in `PPOConfig` (`python/gdrl/ppo.py`); the train.py CLI mirrors them — never introduce a second default (the viewer's Train tab is a third mirror, `Gui::TrainForm`; keep it in sync).
- **Performance contract**: trunks run under bf16 autocast with channels_last conv tensors (`--amp bf16`, default; `--amp none` = fp32); heads, logit bound and losses stay fp32 — keep it that way (bf16 logits inject noise into PPO ratios). The Bernoulli math is closed-form (`model.py`), not `torch.distributions`. `--compile` is opt-in and needs Triton (`pip install triton-windows`). Det eval runs off-thread (`DetEvaluator`); SIL replay is one DLL call (`gdenv_replay`). Before/after any perf change run `python -m gdrl.bench --compare <baseline.json>` (from `python/`) and check the per-phase `t_rollout/t_update/t_sil/t_det_wait/t_ckpt` columns in `log.csv`. **`ticks/s` is not a throughput metric**: ticks per decision falls ~23 → ~10 as the policy stops hopping (a ground hop is one decision of ~104 ticks) and again in POLISH (longer episodes, `target_kl` early stop no longer fires at lr ≤ 1e-4); compare `samples_s` (log.csv) / `dec/s` (log line, panel) across runs and phases instead.

## Geode mods (mod/ = trace logger, mod_replay/ = open-loop replayer)

Both build with MSVC (VS 2026), not MSYS2; Geode SDK at `$GEODE_SDK`, CLI in `.tools\geode-cli` (gitignored). The build auto-installs the `.geode` into the game:

```
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation
$env:GEODE_SDK = "C:\Users\fangj\Geode"; $env:Path = "E:\Github\geometry-dash-rl\.tools\geode-cli;$env:Path"
cmake -S mod_replay -B mod_replay/build -G Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build mod_replay/build
```

- **Sim-to-real loop**: `python -m gdrl.eval runs/<name>/ckpt_best_det.pt` → `python -m gdrl.deploy runs/<name>` (re-verifies the raw holds win via `env_check run`, then installs `<level_id>.json` into the mod's `solutions/`) → play the level in GD (normal mode, no start pos, click-between-frames off) → the mod writes `traces/attempt_NNNN.csv` + `result_NNNN.json` in its save dir → `build/trace_check.exe <that csv>` names the first divergent tick on a desync; `python -m gdrl.deploy --check-trace <csv> runs/<name>` asserts the recorded holds equal the solution.
- **Tick contract**: real tick t = t-th `GJBaseGameLayer::processCommands` after `PlayLayer::resetLevel` (trace row t). **Sim tick t = real tick t+1** — the real game's tick 0 is a pre-movement input tick (row 0 has x=0) while sim step 0 already moves the player. So the replayer injects `holds[t-1]` at real tick t, `trace_check` applies row i's hold to sim step i-1, and `deploy --check-trace` compares the same way. Getting this wrong = one-tick-early jumps that clip blocks (first real-game run died at 54.8% this way). `solution.json` ranges are `[start,end)` in sim ticks; `env_check run` ranges are inclusive sim ticks (deploy.py converts).
- Raw addresses in both mods are GD **2.2081 Windows** only.
