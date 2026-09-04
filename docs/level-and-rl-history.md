# Level, disassembly and RL history

Reference notes moved out of `CLAUDE.md` (2026-09-04). Not auto-loaded. Mechanics are implemented and documented in `src/core/physics.hpp` (header comment) and `src/core/geometry.cpp`; this file records what each level added, how it was verified, and the RL lessons.

## Per-level notes

Verification = a real-game trace replayed bit-exact by `trace_check` (positions, death tick, m_isOnGround, yvel). "Full completion" = the sim's `solution.json` replayed open-loop by the mod beat the real level.

| Lvl | New ids | Key engine facts | Verified by |
|---|---|---|---|
| 1 Stereo Madness | baseline cube/ship, spikes, blocks, portals | yvel rounds to 3 decimals; inner-box death; landing-pattern x snap | session2–6 |
| 2 Back On Track | pads 35/67, gravity portals 10/11 | specials before solids per tick; no cube head-hit; flipped cube has no ceiling | session7 (full completion) |
| 3 Polargeist | yellow ring 36 | 36x36 box, press-as-resource, one-shot, touching list lags one tick | session8 |
| 4 Dry Out | slab 68 (30x16, case 153), trigger 26 | — | session9, session11 (full completion) |
| 5 Base After Base | flip portal 11 (25x75) | onGround fall-path bug (below) | session10/12 (full completion) |
| 6 Can't Let Go | pink pad 140, trigger 24 | type 9, hitbox **25x5** (bump_03 frame, not bump_02's 29x7); bumpMod cube 0.65 → 10.4, ship 0.35, ufo 0.4, ball/spider 0.7; then yellow-pad propell path | session13 (full completion, after a one-tick pad desync) |
| 7 Jumper | 73 = decoration (case 0 like id 5) | gravity portals inside a ship section → flipped ship: sign from m_isUpsideDown in updateJump, clamps [-8, 6.4], block roles swap (top face = head bump, bottom = landing); `flipGravity` clears m_isOnGround unconditionally; m_gravity 0.95819902 for every mode → cube gravity f32(0.225f·0.9582f) | session14 (full completion; flipped clamps/band/solids still disasm-only) |
| 8 Time Machine | pink ring 141, mirror portals 45/46 | ring: type 12, engine box **36x36** (not the 32px frame), ringJump ×0.72 → ±8.05 cube / 4.137 ship (0.37), 0.42 ufo, 0.77 ball; exact m_jumpAccel 11.180031776. Mirror portals only call `toggleFlipped` (camera) — no physics | session15 (policy completion + solve_8 replay firing both rings at the sim's ticks) |
| 9 Cycles | ball portal 47 (type 16, 34x86), trigger 25 | **ball mode** (below); `kMaxGameplayY` 2790 (below) | session16 (whole first ball section), session17 (sky-exploit kill), session18 (solve_9 full completion), session19 (policy completion) |
| 10 xStep | pit 61, outline solids 69–75, blue gravity ring 84, solids 66/83/94, deco 48–53, triggers 28/58/59 | pit 61: type 2, m_width/height from the **trimmed** atlas rect (30x18) × 0.3/0.4 → **9x7.2**; 69–75 forced 30x30 by the third id switch in customSetup; ring 84: type 13, 36x36, ×0.8 → ±8.944 then `flipGravity` (yvel ×0.5, ungrounded), ball ×0.7 → 3.1304; **flying band is centered on the portal** (below) | session20, session21 (bot full completion), session22 (policy completion, five gravity rings) |
| 11 Clutterfunk | size portals 99 (normal) / 101 (mini), sawblades 88/89/98, solids 76–78/81/82/90–93/95/96, deco 60/80/85–87/97, trigger 104 (never created) | **mini mode** (below); sawblades = **circle hazards** (below), radii 32.3 / 21.6 / **11.775** (98's m_width is the uhd trimmed rect 157/4 = 39.25, not the 1x sheet's 40); 104 is not registered in ObjectToolbox::init so createWithKey returns null | session23 (2026-09-04): `runs/level11-1` policy trained from scratch on the same day's sim and deployed — **full completion first try**, 22714 ticks bit-exact in both mods, 9 size portals (8942 mini-cube / 1683 mini-ship / 3908 mini-ball ticks), closest sawblade approach 1.7 units; objects.csv confirms 31x90 portals and the radii. `runs/solve_11` = sim BFS proof |

Lessons: take every hitbox from an `objects.csv` dump (three ids were wrong from sprite frames); trace_check a WIN too and confirm the new mechanic actually fired; id 142 (secret coin) is everywhere and never collides.

### m_isOnGround fall-path clear (2026-09-01)
Byte +0x9c1 is m_isOnGround. It is cleared in `updateJump`'s fall path (helper 0x39a430), not at the head of `checkCollisions`: normal gravity when pre-gravity yvel < f32(1.92), flipped when > 1.92. So a flipped cube walking off a block stays jumpable 9 ticks. First BAB deploy died at 78% because the sim treated that press as an air no-op.

### Ball mode (Cycles)
Runs updateJump's cube branch with gravity ×0.6f, the same ±15 clamp and 2·m_gravity falling test. "Jump" = cube jump (±11.18) then `flipGravity(!upsideDown)` (yvel ×0.5 raw, ungrounded) then yvel ×0.6f raw → a 3.354 kick toward the new gravity, and it clears m_isHolding, so a held button flips once. Hence `PlayerState::inputHeld` (physical button) is split from `holding` (m_isHolding). Rings: ring velocity then ×0.7f raw, m_isHolding = 0, no flip (pink 0.77). Pads: propell then ×0.6f raw (pink bumpMod 0.7). Band: ceiling = getGroundHeightForMode(16) = 240; ceiling snap calls hitGround only if yvel > 0, floor snap always. Solids: the ship's two-block logic with the cube's 10-unit window; no landing x snap. Ball↔cube portals keep yvel/onGround; ship↔anything halves yvel and ungrounds. Also modeled, never trace-hit: the 0.1 s post-flip inner-box escape (`checkDeaths` takes prevY). Trace `ship` column is now the mode (0 cube, 1 ship, 2 ball).

### Vertical bound (2026-09-02)
`checkCollisions` 0x140213bfe flags y > `m_maxGameplayY` (+0x36a8; 2790 for classic levels, `updateMaxGameplayY` 0x23aec0) and kills on the second consecutive flagged tick. Found because `runs/level9-1` "won" by flying a flipped cube at y ≈ 25000; the real game killed it at tick 2838 (session17).

### Portal-centered flying band (2026-09-03)
`animateInDualGroundNew` 0x213370 sets center = max(90, floorf((portal.y − H/2)/30)·30) + H/2, H = 300 ship / 240 ball; band = center ∓ H/2, re-centered on every activatable ship/ball portal contact. Every level 1–9 portal yields floor 90; xStep's ball 47@18405 gives [180,420] and ship 13@23025 gives [150,450]. Now `Portal::bandFloor` / `PlayerState::bandFloor`. `runs/level10-1` sat 1000 updates at 84% on the fictional [90,330] band. Trace logger logs `bandmin,bandmax,bandcenter,bandh,...`.

### Mini mode (Clutterfunk, 2026-09-04)
Size portals: customSetup types 17 (id 99) / 18 (id 101); collisionCheckObjects slots 14/15 call `PlayerObject::togglePlayerScale(enable = type == 18)` 0x3a0700, which only writes `m_vehicleSize` +0x9f0 (0.6f / 1.0f) and the player's rect scale +0x394/+0x398 — no velocity, ground flag or position change. All 96 reads of +0x9f0 were classified; the gameplay ones: outer box = `getObjectRect()` = 30 × 0.6 = **18x18** (half 9) while the inner death box is the direct `getObjectRect(0.3, 0.3)` and **stays 9x9**; floor / band lines and the 2790 bound move by (1 − scale)·15 (float-exact: floor 99, band ±9, bound 2796); jump / ring / pad velocity × **0.8** (11.18 → 8.944); ship increment and both clamps **÷ 0.85** (−7.529 / 9.412, flipped −9.412 / 7.529); landing-pattern snap dx 120 → **90** (patterns 2/3 unchanged); snap windows, gravity, `playerIsFallingBugged` unchanged. Sim: `PlayerState::mini`, `mini::` in physics.hpp, obs v5 scalar 26.

### Circle hazards (Clutterfunk sawblades)
88/89/98 are EnhancedGameObjects; customSetup case 0x1401a5bd9 sets type 2 and `m_objectRadius` +0x38c = 32.3 (88, constant) / m_width × 0.36 (89, 60 → 21.6) / m_width × 0.3 (98, **39.25 → 11.775**: m_width at that moment is the -uhd sheet's trimmed rect 157 px / 4; the dump's later w column says 40, so always read the `radius` column). checkCollisions' hazard loop uses `playerCircleCollision` 0x211df0 for any radius > 0; with `m_fixRadiusCollision` (LevelSettingsObject +0x1cf, a 2.2 option, clear in official levels) the test is: player outer rect contains the center (inclusive) OR a rect corner is strictly inside the circle (float distance). Not a true circle–rect test. `updateIsOriented` skips the oriented box when radius > 0, so a hit is a death directly. Sim: `Box::radius/cx/cy`, `circleHitsPlayer`.

### customSetup id switches (xStep)
Three id switches: main byte table 0x140194b44 / jump table 0x1401948dc (id−5); second at 0x140193d1f (id−22); third at 0x140193dd6 (id−10, sends 69–75 to the forced-30x30 case). commonSetup 0x14018de2f copies the trimmed `m_obRect.size` (+0x1b0) into m_width/m_height.

## Sim solver
`build/solve.exe <level.txt> [solution.json]` = sim-only BFS, ~1 min/level, deploy-compatible output; existence proofs in `runs/solve_*` (discard `runs/solve_10`, solved on the old band). Its holds are per-tick and not reproducible by `GdVecEnv.replay`; use `env_check run`.

## RL history

- **Macro actions** were the original learning fix; **tanh-bounded logits** fixed saturation walls; split trunks fixed collapses; CNN + polish closed the det gap.
- **Position-dependent exploration** (2026-09-01, Polargeist det WIN 719 → 269): the 1.8% logit-bound-4 floor only at x ≥ det_x − `--explore-margin`; mastered prefix samples at bound 8 (bound stored per sample, re-applied in `evaluate`). `--burst 1000` frontier bursts, `--frontier-respawn 0.4`, `--mastered-ent 0.1`, polish on `--polish-det`.
- **Planner-in-the-loop** (2026-09-03, `runs/level9-2` stuck 500+ updates at Cycles x 19681 with p(flip) = 0.982 where 15 no-flips were needed): `src/env/solve.hpp` macro BFS in the policy's action space from the det run's own states (`WallSolver`, `--solve-back/--solve-reach/--solve-nodes 2000000`), demos into SIL (`--sil-demos 4`, 25% minibatch share) and the respawn bank. `--anchor-coef 0.1 --anchor-margin 2` hinge on the best det run's mastered-zone decisions (prefix drift had dropped det_x 12777 → 2490). `--det-rollback 0.5` restores `ckpt_best_det.pt` after 2 bad det evals once best ≥ half the level. `--solve-stall 3` keeps the planner off Stereo Madness (det WIN stays 119). Result: level9-3 from scratch beat real Cycles (session19).
- **Leaky logit bound** (`--logit-leak 0.1`): level9-3 lost 990 updates at x 12777 because raw logits were −20..−33 (gradient factor ~2.5e-5 through the tanh bound). Backward floored via a straight-through term; forward unchanged (eval/deploy use leak 0). Cycles det WIN 1759 vs 1909, but the prefix got fragile (7 rollbacks), so the planner now fires only for frontier deaths and `prune_demos` drops mastered-zone demos.
- **Poisoned runs**: any run trained before a sim fix that changed reachable states (level9-1 sky states, level10-1 fictional band) must restart, not `--resume`.
- Reference: Stereo Madness det WIN ≈ update 119 (reg-p4..p9); xStep from scratch det WIN 709 (`runs/level10-2`).
