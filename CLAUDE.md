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
