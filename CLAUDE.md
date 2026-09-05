# CLAUDE.md

Exact Geometry Dash replica for RL (train in the replica, deploy to the real game via a Geode mod). Strategy in @gd-replica-notes.md. Per-level mechanics, disassembly findings and training-run history: `docs/level-and-rl-history.md`, the `src/core/physics.hpp` header comment and `src/core/geometry.cpp`.

## Build

MSYS2 UCRT64 (g++/cmake/ninja on PATH): `cmake -S . -B build -G Ninja && cmake --build build`

- **Keep `-static` in CMakeLists.txt** — this machine's ld crashes with "ld returned 116 exit status" otherwise.
- raylib/zlib from MSYS2 packages. Viewer deps are FetchContent'd: ImGui `v1.92.7` + rlImGui tag `Raylib_5_5` (pinned together, don't bump one alone), nlohmann/json 3.11.3; `-DVIEWER_VENDORED_DEPS=<dir>` for offline.

## Level data

- GD install: `S:\SteamLibrary\steamapps\common\Geometry Dash\`; levels at `Resources\levels\<id>.txt` (URL-safe base64 of gzip). Levels 1–11 are trace-verified.
- **Never commit level data or extracted game assets.** Read from the install at runtime.
- Tools take `GD_LEVEL_PATH` (default `levels/1.txt`); trainer/eval take `--level <path>`.
- Smoke test (`build/dump.exe <level>`): objects/ids = 1: 2291/31, 2: 1545/26, 3: 1531/35, 4: 1444/30, 5: 2146/35, 6: 1704/33, 7: 2813/34, 8: 3599/36, 9: 3921/40, 10: 4716/58, 11: 4863/66, 12: 4345/92.
- **Hitboxes come from an `objects.csv` dump** (the trace logger writes one on level load), never from sprite frame sizes or the viewer's shape table. Before a dump exists, the engine's own per-id size table `GameObject::setupSpriteSize` (0x1401a36a0, decode with the byte table 0x1401a3fec / jump table 0x1401a3f2c, id−1) is the authority for rect sizes — it reproduces every dumped w/h. Circular hazards: read the `radius` column (the radius is computed inside customSetup BEFORE that table runs, from the **-uhd** trimmed rect / 4, e.g. sawblade 98 = 39.25 → 11.775, not the table's 40).

## Trace regression (run after touching the sim)

Every trace must MATCH incl. death ticks, with no `[onground mismatches ...]` suffix. `GD_TRACE_DEBUG=<from>:<to>` prints sim vs real per tick.

| Level | Command |
|---|---|
| 1 | `build/trace_check.exe traces/attempt_*.csv traces/session[2-6]/attempt_*.csv` |
| 2 | `GD_LEVEL_PATH=<levels/2.txt> build/trace_check.exe traces/session7/attempt_*.csv` |
| 3 | `... 3.txt ... traces/session8/attempt_*.csv` |
| 4 | `... 4.txt ... traces/session9/attempt_*.csv traces/session11/attempt_*.csv` |
| 5 | `... 5.txt ... traces/session10/attempt_*.csv traces/session12/attempt_*.csv` |
| 6 | `... 6.txt ... traces/session13/attempt_*.csv` |
| 7 | `... 7.txt ... traces/session14/*attempt_0001.csv` |
| 8 | `... 8.txt ... traces/session15/*attempt_000[12].csv` |
| 9 | `... 9.txt ... traces/session16/attempt_*.csv traces/session17/attempt_*.csv traces/session1[789]/logger_attempt_0001.csv traces/session1[89]/attempt_0001.csv` |
| 10 | `... 10.txt ... traces/session20/attempt_*.csv traces/session21/attempt_0001.csv traces/session22/attempt_0001.csv traces/session22/logger_attempt_0001.csv` |
| 11 | `... 11.txt ... traces/session23/attempt_0001.csv traces/session23/logger_attempt_0001.csv` |
| 12 | `... 12.txt ... traces/session24/logger_attempt_0001.csv traces/session25/attempt_0001.csv traces/session25/logger_attempt_0001.csv` |

## Architecture rules

- `gdcore` (src/core) stays free of rendering deps — it is the headless sim. Fixed timestep only.
- Viewer shape table is presentation-only, never collision data.
- Tick contract: **sim tick t = real tick t+1** (real tick 0 is a pre-movement input tick). Replayer injects `holds[t-1]` at real tick t; `trace_check` applies row i to sim step i-1. `solution.json` ranges are `[start,end)` sim ticks; `env_check run` ranges are inclusive.
- Raw addresses in the mods are GD **2.2081 Windows** only.

## Viewer (src/viewer)

- Viewport is `App::viewW x viewH` at (0,0); use `viewW(app)/viewH(app)`, never `GetScreenWidth()`. Mode input goes through `viewer/input.hpp` wrappers, never raw `IsKeyPressed`. Cross-tab requests via `App::commands`. Frame pacing via `Mode::targetFps()` in `main.cpp`; don't call `SetTargetFPS` in modes.
- `jobs.cpp`/`paths.cpp` include windows.h and must never include raylib.h.
- **Train-tab defaults (`Gui::TrainForm`) mirror `PPOConfig`/`train.py`** — change all three together.
- The trainer's `upd` log line is regex-parsed in `Gui::pollPipelines` (`kUpd` = the fixed prefix through `ent`, `kDet`, `kTeach` = `| teach pX dY`); new fields go AFTER `ent` and need their own regex + `TrainStatus` field. `live.bin` v2 header stays 40 bytes; `sps`/`samples_s` are the last completed update's rates. Train-tab contract combos (obs-version / action-mode / teacher-coef) mirror `train.py`.
- Graceful stop = `runs/<name>/STOP` file. `config.json` `end_x` is the effective end; prefill from `end_x_arg`, never `end_x`.
- Prefs in `<repo>/viewer_prefs.ini` (gitignored). Captures: `viewer --shot out.png [--panel] [--tab N]`.

## RL env (src/env + python/gdrl)

- **Obs versioning** (`src/env/obs.hpp`): frozen layout, additive-only changes bump `kObsVersion` (v5 = legacy absolute-band layout, scalar 26 = mini flag). **Obs v6** (`buildObsV6`, `--obs-version 6`, dim 1693) = player-centered grid (x −30…450, y −180…180) + scalars 27–36 + 12 exact geometry rows; a separate layout selected per env handle, not a bump of v5. `obs.hpp`/`contracts.hpp` are meant to be `#include`d by the Geode mod — compile-time constants only. Old checkpoints load through `load_model_state` (zero columns for appended scalars).
- **Contracts** (`src/env/contracts.hpp`, `GdVecEnv.contracts()`): obs version, action mode (`macro` = 2 actions / `precision` = 7: R1 R2 R4 H1 H2 H4 HOP, `src/env/episode.hpp`), trajectory + snapshot versions, `kSimVersion` (bump on any physics change that alters reachable states), level + sim fingerprints. Every checkpoint/config.json records them; `--resume` and `gdrl.eval` refuse a mismatch. Snapshot blobs have a 16-byte header (`gdenv_set_state` rejects foreign blobs; legacy resumes migrate). **Defaults are obs 6 / precision** (train.py and the viewer's `TrainForm`, promoted 2026-09-05); every level 1-12 checkpoint is obs 5 / macro and needs `--obs-version 5 --action-mode macro` to resume (eval reads the checkpoint's contracts itself).
- **One executor** (`EnvInstance::act`) for rollouts, replay, the planner and eval. Precision episodes are replayed from their DECISION streams (`replay_actions`, `last_episode_actions`); raw holds stay the deployment format (`solution.json`, `.gdtrace`) and are verified raw via `run_holds` before a certificate or solution is written.
- **Env stepping is parallel and bit-deterministic** (`$GDENV_THREADS`, 1 = serial). After touching the env layer: `./build/env_check det`, `python -m gdrl.selftest` AND `python -m unittest discover -s tests` (from `python/`; needs the GD install and `runs/level12-2/solution.json` for the UFO/corridor tests).
- **Regression bar**: default `python -m gdrl.train` must still beat Stereo Madness (`det ... WIN`; reference det WIN ≈ update 155 under the default obs 6 / precision regime, ≈ 119 (79 on the 2026-09-05 code) under `--obs-version 5 --action-mode macro`) and `python -m gdrl.eval runs/<name>/ckpt_best_det.pt` writes `solution.json`. Don't shrink `--updates` — it changes the anneal.
- Hyperparameter defaults live in `PPOConfig` (`python/gdrl/ppo.py`); train.py CLI and `Gui::TrainForm` mirror them — never a second default.
- Perf contract: trunks bf16 autocast + channels_last; heads/logit bound/losses stay fp32. Compare with `python -m gdrl.bench --compare <baseline.json>`; use `samples_s`/`dec/s`, never `ticks/s`.
- A run trained before a sim fix that changes reachable states is poisoned (SIL buffer + respawn bank) — start fresh, don't `--resume`.
- **Stitched training is the default** (`--stitch 1`): the best det run's holds are the certificate (`python/gdrl/certificate.py`: immutable ids, every stitched episode attached to its exact prefix, orphans reset on a replacement); every episode and det eval starts from its state `--stitch-back` (600) before its end, a det run that gets further extends it ONLY after raw-hold verification, and the frontier follows. After 3 stalled det evals the overlap widens 1200/2400/4800 (`--stitch-widen 0` disables) and the planner's start window moves back with it. Frontier episodes carry `prefix_ticks` so traces/wins/`solution.json` are full from-x=0 holds; SIL stores (suffix, spawn blob, decisions). A det WIN writes `solution.json` from the trainer (`stitched` / `standalone_policy` fields); `gdrl.eval` starts from the checkpoint's frontier (`--from-start` to override). `--stitch 0` = the old from-x=0 regime. `--import-certificate <solution.json|.gdtrace> [--import-cut-x X]` seeds a raw-verified certificate.
- **Planner demos** (`src/env/solve.hpp`, exact-state dedup, status/pruning reported) are executor-verified and taught by the teacher loss (`--teacher-coef 0.25`, sil.py); `--demo-sil 1` is the pre-teacher baseline path. Budgets independent of the anneal: `--stop-after-secs`, `--stop-after-updates`, `--stop-on-win`; `--seed`. Corridor comparison: `python -m gdrl.corridor` (runs/corridor/results.json).

## Geode mods (mod/ = trace logger, mod_replay/ = replayer)

MSVC (VS 2026), Geode SDK at `$GEODE_SDK`, CLI in `.tools\geode-cli`. Build auto-installs into the game:

```
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation
$env:GEODE_SDK = "C:\Users\fangj\Geode"; $env:Path = "E:\Github\geometry-dash-rl\.tools\geode-cli;$env:Path"
cmake -S mod_replay -B mod_replay/build -G Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build mod_replay/build
```

Sim-to-real loop: `gdrl.eval` → `gdrl.deploy runs/<name>` → play the level in GD → `build/trace_check.exe <trace csv>` (even on a WIN — check the new mechanic was actually exercised) → `gdrl.deploy --check-trace <csv> runs/<name>`.
