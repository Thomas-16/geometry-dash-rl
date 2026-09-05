# RL improvements for precise, harder levels

## Findings and direction

Keep **per-level training, stitched replay certificates, and PPO**, with search serving as a strong training teacher.

The main problems are structural:

- **Missing geometry:** all 311 decisions in the level-12 UFO corridor receive an empty observation grid. The player is at y≈1011–1155; the grid covers only 90–450.
- **Restricted timing:** four-tick actions cannot express every possible input sequence.
- **Weak teaching:** planner demonstrations receive no policy-learning signal when their estimated advantage is nonpositive.
- **Fragile search:** quantized deduplication omits consequential state, and population pruning can discard useful trajectories.
- **Frontier lock-in:** moving backward within the planner does not move the actual training frontier backward.

Current environment determinism checks pass, and both session-25 completion traces match all 19,683 gameplay ticks.

## 1. Give the policy useful geometry and precise controls

**Observation v6**

- Retain the grid’s resolution and two channels, but move its window with the player: horizontal offsets −30…450 and vertical offsets −180…180.
- Preserve existing scalar meanings; append floor/ceiling clearances, applicable boundary flags, collision size, gravity-flip grace state, and landing-snap memory.
- Add an exact geometry branch containing four nearby hazards, four solids, and four interactive objects. Include relative box edges, circle center/radius, object kind, and activation/contact state. Select within the observation window by distance to the player’s box, breaking ties by object order; mask empty entries.
- Keep separate actor/value encoders and existing mixed-precision performance conventions. Geometry comes from simulator collision data.

**Precision action mode**

- Introduce seven categorical actions: hold or release for **1, 2, or 4 ticks**, plus the existing grounded cube/ball hop macro.
- Mask the hop action outside its supported grounded states. Raw actions retain physical button semantics; UFO retriggering explicitly requires release followed by press.
- Interrupt repeated actions after landing, mode/size/gravity changes, or newly pressable ring contacts. Always permit the one-tick raw actions.
- Use the same action executor in rollouts, planning, evaluation, and replay. Record masks and sampling parameters for correct PPO probability ratios.

Keep the old observation/action mode available for existing checkpoints and baseline comparisons.

## 2. Make demonstrations reliably teach the policy

- Add a dedicated supervised action loss for verified planner demonstrations, independent of the critic’s advantage estimate. Keep ordinary SIL for successful policy experience.
- Normalize teacher loss over its own batch, with initial coefficient **0.25**. Sample half the teacher batch from policy–teacher disagreements and half uniformly from verified demonstrations.
- Retain bounded logits and gradient leakage for the categorical policy. Track teacher-action probability, disagreement rate, and demonstration replay success.
- After each stalled deterministic evaluation, search from the failing policy’s actual trajectory. Requery on subsequent evaluations so teaching follows the states the policy currently visits.
- Advance certificates only when the **policy itself** produces a longer verified trajectory; planner success alone does not promote them.

The separate teaching objective draws on [demonstration-assisted RL](https://arxiv.org/abs/1704.03732); its value here must be established by the corridor comparisons.

## 3. Improve search and frontier recovery

**Planner**

- Search the new precision action space and return explicit action IDs, durations, and raw holds.
- Replace lossy deduplication with fieldwise exact transition-state equality, including tick, x, velocity, activation records, ring contacts, band state, and landing memory. Hash collisions must resolve through equality checks.
- Bound memory and expansions. Preserve trajectory diversity across height, velocity, and mode when pruning; report pruning and budget exhaustion separately from exhausted search.
- Replay every returned demonstration through the shared executor before admission. Search failure must never be reported as proof that a section is impossible.

**Stitching**

- Keep the default 600-unit overlap. After three deterministic evaluations without progress, widen it to 1200, then 2400, then 4800 units, clamped to the level start. Return to 600 after verified progress.
- Move both training and planner starts backward together so the policy can change the approach to an obstacle.
- Give certificates immutable identities and attach each episode to its exact prefix. Replacing an earlier section resets active rollout episodes and prevents mixing old prefixes with new suffixes.
- Use reachable snapshots for practice; only prefix-backed policy trajectories can extend the deployment certificate.

This extends the project’s existing return-then-explore approach, consistent with [Go-Explore](https://www.nature.com/articles/s41586-020-03157-9).

## 4. Correct contracts and improve diagnostics

- Version observation, action, trajectory, and snapshot contracts. Checkpoints record these plus level and simulator fingerprints.
- Store decision sequences explicitly: raw holds cannot uniquely recover variable-duration action choices. Preserve raw holds for unchanged `solution.json` and `.gdtrace` deployment formats.
- Evaluate using checkpoint settings automatically. Reject incompatible resumes; v6 uses fresh training state. Existing certificates may be imported only after raw replay verification.
- For precision mode, accumulate discounted per-tick rewards and use `gamma^duration` bootstrapping with duration-aware GAE: `lambda^(duration/4)`, initially retaining gamma 0.9995 and lambda 0.95.
- Add reproducible seeds and stopping budgets independent of the learning-rate schedule. Keep a learning-rate floor of 1e-4 until first completion in precision mode.
- Report frontier progress, teacher disagreement, action-duration distribution, planner cost, and wall-clock time to verified completion. Distinguish stitched completion from standalone policy completion.
- Update CLI, viewer defaults/status parsing, checkpoint handling, and project documentation together.

## Validation and acceptance

Use the selected **quick-iteration budget**:

1. Add focused tests for one-tick timing windows, UFO release/repress, ring contacts, portal transitions, action replay, truncation/GAE, search-state distinctions, and certificate replacement.
2. Verify that elevated corridor geometry appears in v6 and that one-tick sequences replay identically through policy and planner paths.
3. Run environment determinism checks and the documented trace suite. Preserve the required Stereo Madness training regression with its original annealing schedule.
4. Compare baseline, observations-only, observations-plus-precision, and the complete upgrade on the recorded level-12 corridor using seeds 0–2 and equal five-minute budgets.
5. Run one fresh full level-12 training experiment with the complete upgrade. Accept after a policy-generated stitched WIN and full raw-hold replay verification; validate real-game transfer through the existing deployment/trace loop.
6. Promote new defaults only after correctness checks pass, corridor completion improves under equal budgets, and the full-level run succeeds. Report limited seed coverage explicitly.

Future mechanics, cross-level generalization, and standalone policy distillation remain separate work. No physics randomization or simulator approximations are introduced.
