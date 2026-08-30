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

## Level data

- GD install: `S:\SteamLibrary\steamapps\common\Geometry Dash\`
- Official levels: `Resources\levels\<id>.txt` (Stereo Madness = `1.txt`), URL-safe base64 of gzip. The `LevelData.plist` method described in older docs/notes is obsolete as of GD 2.2.
- **Never commit level data or any extracted game assets** — they're RobTop's content. Always read from the install at runtime.
- Smoke test: `./build/dump.exe` must report 2291 objects / 31 distinct ids for Stereo Madness.

## Architecture rules

- `gdcore` (src/core) must stay free of rendering dependencies — it becomes the headless training sim. Rendering-only concerns (e.g., the object-id → shape table) live in the viewer.
- The viewer's shape table is presentation-only; never treat it as hitbox/collision data. Hitboxes come later from the binary (see notes).
- Future sim work: fixed timestep only, never variable delta time (determinism requirement, see notes).

## RL env contract (src/env + python/gdrl)

- **Observation versioning** (`src/env/obs.hpp`): the scalar block and grid layout are frozen; changes are additive-only and bump `kObsVersion` (checked by `python/gdrl/env.py` at DLL load). Scalar `[17]` is reserved as the future mode flag; changing the grid y-band is an obs v3 decision. `obs.hpp` will be `#include`d verbatim by the Geode mod — compile-time constants only, no runtime normalization.
- **Speed portals later change an input, not the network**: per-tick `dx` is already an obs scalar and a `buildObs` parameter.
- **Env stepping is parallel and bit-deterministic**: `$GDENV_THREADS` total threads (1 = exact serial fallback), default `hardware_concurrency()-1`. After touching the env layer, run `./build/env_check det` AND `python -m gdrl.selftest` (from `python/`; serial-vs-threaded byte-identity).
- **Regression bar for RL changes**: a default `python -m gdrl.train` run must still beat Stereo Madness (deterministic `det ... WIN` line / `wins.gdtrace`), and `python -m gdrl.eval runs/<name>/ckpt_best_det.pt` must write a `solution.json`. Don't shrink `--updates` for the check — it compresses the lr/entropy anneal and changes the learning trajectory.
- Hyperparameter defaults live in `PPOConfig` (`python/gdrl/ppo.py`); the train.py CLI mirrors them — never introduce a second default.
- **Performance contract**: trunks run under bf16 autocast with channels_last conv tensors (`--amp bf16`, default; `--amp none` = fp32); heads, logit bound and losses stay fp32 — keep it that way (bf16 logits inject noise into PPO ratios). The Bernoulli math is closed-form (`model.py`), not `torch.distributions`. `--compile` is opt-in and needs Triton (`pip install triton-windows`). Det eval runs off-thread (`DetEvaluator`); SIL replay is one DLL call (`gdenv_replay`). Before/after any perf change run `python -m gdrl.bench --compare <baseline.json>` (from `python/`) and check the per-phase `t_rollout/t_update/t_sil/t_det_wait/t_ckpt` columns in `log.csv`.
