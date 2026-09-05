"""PPO trainer for the GD replica.

Headless by default. `--watch` also writes runs/<name>/live.bin, which the
viewer's training view renders live:  viewer --watch runs/<name>/live.bin

Usage (from repo root, inside the venv):
  python -m gdrl.train --run-name full --n-envs 256
  python -m gdrl.train --run-name short --end-x 2500 --updates 60
  python -m gdrl.train --run-name v6 --obs-version 6 --action-mode precision

Contracts (src/env/contracts.hpp): the observation layout, the action
space, the trajectory / snapshot formats and the level + simulator
fingerprints are recorded in every checkpoint and config.json; a resume
refuses a mismatch (a run trained under other dynamics is poisoned).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .certificate import Attachments, Certificate, widen_schedule
from .env import (ACT_HOP, GdVecEnv, OBS_VERSION, OBS_VERSION_V6,
                  PRECISION_ACTION_NAMES)
from .live import LiveWriter
from .model import (AMP_MODES, GraphedAct, autocast_ctx, configure_backend,
                    load_model_state,
                    make_model, maybe_compile)
from .ppo import PPO, PPOConfig, RolloutBuffer
from . import sil
from . import traces


_D = PPOConfig()  # hyper defaults live in PPOConfig, the CLI mirrors them


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run")
    p.add_argument("--level", default=None, help="level file path override")
    # 1024 x 64 keeps the batch at 65,536 samples/update (same learning
    # dynamics as the original 256 x 256) with 4x fewer Python<->DLL round
    # trips; the DLL steps envs on a thread pool ($GDENV_THREADS).
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--ticks-per-action", type=int, default=4)
    p.add_argument("--end-x", type=float, default=-1.0,
                   help="early win line for curriculum runs; -1 = full level")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--rollout", type=int, default=_D.rollout_steps)
    p.add_argument("--lr", type=float, default=_D.lr)
    p.add_argument("--ent-coef", type=float, default=_D.ent_coef)
    p.add_argument("--gamma", type=float, default=_D.gamma,
                   help="discount per sim tick (steps are variable-length; "
                        "gamma^ticks is applied per decision)")
    p.add_argument("--gae-lambda", type=float, default=_D.gae_lambda)
    p.add_argument("--reward-scale", type=float, default=0.1,
                   help="fixed reward multiplier; keeps returns O(1-10) so "
                        "the value-loss gradient doesn't drown the policy "
                        "gradient at the global grad clip")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--amp", choices=AMP_MODES, default=_D.amp,
                   help="trunk precision: bf16 autocast (default; heads and "
                        "losses stay fp32) or none for pure fp32")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the trunks (needs Triton: "
                        "pip install triton-windows); falls back to eager")
    p.add_argument("--no-graph", action="store_true",
                   help="don't CUDA-graph the rollout forward (debugging)")
    p.add_argument("--resume", default=None, help="checkpoint path")
    p.add_argument("--watch", action="store_true",
                   help="write live.bin for the viewer's training view")
    p.add_argument("--respawn", type=float, default=0.5,
                   help="probability a dead env respawns from a banked "
                        "mid-level snapshot instead of x=0 (frontier "
                        "practice without poisoning full-run advantages)")
    p.add_argument("--trace-every", type=int, default=20,
                   help="dump episode traces every N updates")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--arch", choices=("cnn", "mlp"), default="cnn",
                   help="policy/value network architecture (a resumed run "
                        "always keeps its checkpoint's arch)")
    p.add_argument("--det-every", type=int, default=10,
                   help="deterministic full-run eval every N updates "
                        "(0 = off); best result saved to ckpt_best_det.pt")
    # Contracts (docs/rl-improvement-plan.md). Defaults = the v6 +
    # precision regime (promoted 2026-09-05 at the user's call after the
    # corridor comparison; every level 1-12 checkpoint is obs 5 / macro
    # and must be resumed / evaluated with those flags — the trainer
    # refuses a mismatch). The viewer's TrainForm mirrors these.
    p.add_argument("--obs-version", type=int, choices=(OBS_VERSION, OBS_VERSION_V6),
                   default=OBS_VERSION_V6,
                   help="observation layout: 5 = absolute-band grid "
                        "(legacy), 6 = player-centered grid + appended "
                        "scalars + exact geometry rows")
    p.add_argument("--action-mode", choices=("macro", "precision"),
                   default="precision",
                   help="macro = 2 actions (walk/hop, 4-tick windows); "
                        "precision = 7 actions (release/hold for 1, 2, 4 "
                        "ticks + the grounded hop macro)")
    p.add_argument("--teacher-coef", type=float, default=0.25,
                   help="supervised action loss on verified planner "
                        "demonstrations (0 = off; sil.py)")
    p.add_argument("--demo-sil", type=int, default=0,
                   help="1 = ALSO feed planner demos through the "
                        "advantage-weighted SIL loss (the pre-teacher "
                        "regime, kept for baseline comparisons)")
    p.add_argument("--stitch-widen", type=int, default=1,
                   help="1 = widen the stitch overlap after 3 stalled det "
                        "evals (1200/2400/4800); 0 = fixed --stitch-back")
    p.add_argument("--seed", type=int, default=0,
                   help="numpy / torch seed (the sim is deterministic)")
    p.add_argument("--lr-floor", type=float, default=1e-4,
                   help="precision mode: the annealed lr never drops below "
                        "this until the first det completion")
    # Stopping budgets independent of the lr / entropy anneal (which spans
    # --updates): equal-budget comparisons stop here, the anneal does not
    # move.
    p.add_argument("--stop-after-secs", type=float, default=0.0,
                   help="stop after this much wall-clock (0 = off)")
    p.add_argument("--stop-after-updates", type=int, default=0,
                   help="stop after this many updates of THIS invocation "
                        "(0 = off; the anneal still spans --updates)")
    p.add_argument("--stop-on-win", action="store_true",
                   help="stop at the first verified det completion")
    # Polish phase: once the stochastic policy wins regularly, the ~1.8%
    # per-decision exploration floor (logit bound 4) is what keeps the
    # TRAINING distribution off the argmax path — raise the bound and drop
    # entropy/lr so PPO converges onto and fixes the deterministic path.
    p.add_argument("--polish-wins", type=int, default=25,
                   help="enter the polish phase after this many cumulative "
                        "from-start wins")
    p.add_argument("--polish-at", type=int, default=-1,
                   help="force the polish phase at this update (-1 = by "
                        "--polish-wins)")
    p.add_argument("--no-polish", action="store_true")
    p.add_argument("--sil-coef", type=float, default=1.0,
                   help="self-imitation loss coefficient (0 = off): after "
                        "each PPO update, replay the best recorded episodes "
                        "through the scout env (the deterministic sim IS the "
                        "experience buffer) and imitate positive-advantage "
                        "transitions")
    p.add_argument("--sil-episodes", type=int, default=8,
                   help="episodes replayed per SIL update")
    # Exploration zone. The logit bound's ~1.8% per-decision floor is what
    # cracks walls, but a full run is ~3000 decisions: applied everywhere it
    # forces ~60 random flips per run and the mastered prefix eats them
    # (Polargeist's winning policy, sampled from x=0 at bound 4, reached
    # the last third of the level 0.1% of the time; 43% at bound 8). So the
    # floor stays only where the argmax policy has not yet proven itself.
    p.add_argument("--explore-margin", type=float, default=600.0,
                   help="envs at x >= latest det_x - margin sample at the "
                        "logit-bound-4 floor; everything before that (what "
                        "the argmax policy already clears) samples at "
                        "--mastered-bound. -1 = bound 4 everywhere")
    p.add_argument("--mastered-bound", type=float, default=8.0,
                   help="logit bound in the mastered zone (8 = 0.03% floor)")
    p.add_argument("--logit-leak", type=float, default=0.1,
                   help="floor on the tanh bound's derivative (backward "
                        "only; model.py): level9-3 lost 990 updates at one "
                        "wall where the raw logit sat at -25 and the plain "
                        "bound's gradient was 2.5e-5. 0 = plain tanh bound")
    p.add_argument("--mastered-ent", type=float, default=0.1,
                   help="entropy-bonus weight in the mastered zone (1 = "
                        "same as the explore zone). Advantages there are "
                        "~0, so a full bonus erodes confidence until the "
                        "prefix breaks again")
    # Respawn share aimed at the explore zone itself. The pass-rate
    # 'learnable' filter (0.1..0.8) skips the hardest wall by construction
    # (pass rate ~0) and the wall-practice draw follows the most frequent
    # deaths, which on a long level are mid-prefix slips, not the frontier
    # (measured: 1% of rollout samples in the explore zone at the x=21000
    # Polargeist wall). This draw starts a few hundred units before the
    # det policy's death point.
    p.add_argument("--frontier-respawn", type=float, default=0.4,
                   help="fraction of respawns drawn from the 700 units "
                        "before the latest det_x (0 = off)")
    # Frontier bursts: a respawned env's job is the wall near its spawn;
    # letting it run on to a death far downstream (or to the end) spends
    # the rollout on territory other envs already cover. Truncating it a
    # burst past its spawn (value-bootstrapped, exactly like the tick cap)
    # turns tens of wall attempts per update into hundreds.
    p.add_argument("--burst", type=float, default=1000.0,
                   help="respawned episodes are truncated (bootstrapping "
                        "V) this many x units past their spawn, then "
                        "respawn again; 0 = run to death")
    p.add_argument("--polish-det", type=float, default=1.0,
                   help="also enter the polish phase once the det eval "
                        "reaches this fraction of the level (1 = det win; "
                        "0 = off). --polish-wins remains as the other gate")
    # Planner-in-the-loop. The sim is a perfect model, so a wall the sampler
    # cannot cross (Cycles x=19681: a saturated flip prior vs ~15 straight
    # no-flips, ~1e-6 per attempt, 500 updates with zero passes) is a
    # sub-second BFS in the policy's action space (env/solve.hpp). After
    # every stalled det eval the solver starts from the FAILING RUN'S OWN
    # states between the frontier and its death and searches --solve-reach
    # past it; the segments it finds are replay-verified, then taught by
    # the teacher loss (sil.py) and their states banked for respawns. It is
    # re-queried on every subsequent stalled eval so the demos follow the
    # states the policy currently visits. Pure exploration stays on.
    p.add_argument("--wall-solve", type=int, default=1,
                   help="1 = solve wall demos after each stalled det eval, "
                        "0 = off")
    p.add_argument("--solve-back", type=float, default=700.0,
                   help="planner starts span [det_x - back, det_x - 50] "
                        "along the det run (stitched: the frontier moves "
                        "it back further as the overlap widens)")
    p.add_argument("--solve-reach", type=float, default=800.0,
                   help="planner target: det_x + reach")
    p.add_argument("--solve-starts", type=int, default=7)
    p.add_argument("--solve-stall", type=int, default=3,
                   help="det evals without a new best before the planner "
                        "fires (0 = after every eval). PPO clears most "
                        "walls itself; demos everywhere slowed Stereo "
                        "Madness (det WIN 179 vs 109-119 updates)")
    p.add_argument("--solve-nodes", type=int, default=2000000,
                   help="expansion budget per planner start (a ship "
                        "section at 4-tick decisions needs ~1.3M per 800 "
                        "units; the memory budget is the same count)")
    p.add_argument("--solve-states", type=int, default=4096,
                   help="per-tick population cap in the planner (thinned "
                        "with height / velocity / mode diversity)")
    p.add_argument("--sil-demos", type=int, default=4,
                   help="demo segments replayed per teacher update")
    p.add_argument("--anchor-coef", type=float, default=0.1,
                   help="behavior-cloning weight pinning the mastered "
                        "zone (x < best det_x - margin) to the best det "
                        "run's own decisions (0 = off; non-stitched only)")
    p.add_argument("--anchor-margin", type=float, default=2.0,
                   help="hinge margin (logits) the anchor holds on the "
                        "best det run's decisions")
    # Stitched training. The deliverable is a hold sequence, and the sim is
    # deterministic, so the best det run's holds up to its death are a
    # certificate that never needs relearning. Every episode and every det
    # eval starts from the certificate's state --stitch-back units before
    # its end (the DLL replays the prefix into a snapshot); a det run that
    # gets further extends the certificate (prefix + its own suffix), and
    # the frontier moves with it. After 3 det evals without progress the
    # overlap widens (1200, 2400, 4800, clamped to the level start) so the
    # policy can change its approach to the obstacle, and returns to
    # --stitch-back on verified progress. Certificates have immutable ids
    # and every episode is attached to its exact prefix (certificate.py).
    p.add_argument("--stitch", type=int, default=1,
                   help="1 = episodes and det evals start from the "
                        "certificate frontier (best det run's state "
                        "--stitch-back before its death); 0 = from x=0")
    p.add_argument("--stitch-back", type=float, default=600.0,
                   help="frontier start = best det x - this (base overlap)")
    p.add_argument("--stitch-widen-cap", type=float, default=4800.0)
    p.add_argument("--import-certificate", default=None,
                   help="solution.json / .gdtrace whose holds seed the "
                        "certificate (raw-replay verified first)")
    p.add_argument("--import-cut-x", type=float, default=-1.0,
                   help="truncate the imported certificate at this x "
                        "(-1 = keep it whole)")
    p.add_argument("--det-rollback", type=float, default=0.5,
                   help="restore ckpt_best_det when det_x stays below this "
                        "fraction of best_det_x for 2 consecutive evals, "
                        "or below 0.3x once (0 = off); armed once best "
                        "det_x >= half the level (early det evals swing "
                        "4k-12k normally, and a rollback there discards "
                        "real learning)")
    return p.parse_args()


EXPLORE_LOGIT_BOUND = 4.0
POLISH_LOGIT_BOUND = 8.0
POLISH_ENT_COEF = 1e-4
POLISH_MAX_LR = 1e-4
POLISH_RESPAWN = 0.25

LOG_COLUMNS = ("update,global_ticks,mean_final_x,best_final_x,best_x_ever,"
               "win_rate,episodes,mean_ep_ticks,sps,policy_loss,value_loss,"
               "entropy,approx_kl,clip_frac,lr,det_x,det_win,"
               "samples_s,t_rollout,t_update,t_sil,t_det_wait,t_ckpt,"
               "t_roll_gpu,t_roll_env,t_roll_misc,frac_explore,"
               "teacher_p,teacher_disagree,front_x,cert_x,stitch_back,"
               "dur1,dur2,dur4,durhop,solve_secs,solve_nodes,elapsed_secs\n")


def load_certificate_file(path: Path) -> np.ndarray:
    """Holds of a solution.json (expanded ranges) or the best episode of a
    .gdtrace."""
    if path.suffix == ".json":
        sol = json.loads(path.read_text())
        n = int(sol["n_ticks"])
        bits = np.zeros(n, np.uint8)
        for a, b in sol["holds"]:
            bits[a:b] = 1
        return bits
    eps = traces.read(path)
    if not eps:
        raise SystemExit(f"{path}: no episodes")
    wins = [e for e in eps if e.won]
    best = wins[0] if wins else max(eps, key=lambda e: e.final_x)
    return np.asarray(best.holds, np.uint8)


class EpisodeRecorder:
    """Collects finished episodes' per-tick holds (recorded by the env)."""

    def __init__(self, env: GdVecEnv):
        self.env = env
        self.completed: list[traces.Episode] = []
        self.wins: list[traces.Episode] = []

    kMaxWins = 16  # wins are interchangeable certificates; cap the trace

    def record(self, done: np.ndarray, info: dict, holds_of=None):
        """holds_of(i) -> full from-x=0 holds of env i's finished episode
        (stitched episodes prepend their certificate prefix); default =
        the env's own record."""
        for i in np.nonzero(done)[0]:
            holds = (self.env.last_episode_holds(int(i)) if holds_of is None
                     else holds_of(int(i)))
            ep = traces.Episode(holds, float(info["final_x"][i]),
                                bool(info["won"][i]))
            self.completed.append(ep)
            if ep.won and len(self.wins) < self.kMaxWins:
                self.wins.append(ep)

    def dump_best(self, path: Path, keep: int = 5, extra: int = 3):
        if not self.completed:
            return
        eps = sorted(self.completed, key=lambda e: -e.final_x)
        chosen = eps[:keep]
        rest = eps[keep:]
        if rest:
            rng = np.random.default_rng(len(self.completed))
            chosen += [rest[j] for j in
                       rng.choice(len(rest), min(extra, len(rest)),
                                  replace=False)]
        traces.write(path, chosen)
        self.completed = []


class DetEvaluator:
    """Deterministic full-run eval of a weight snapshot on a dedicated
    1-env handle, in a background thread (DLL calls and torch kernels
    release the GIL, so ~1000 decisions x ~1 ms overlap the next rollout
    instead of blocking it). The result carries the snapshot's state dicts
    so ckpt_best_det.pt holds exactly the weights that produced the run."""

    def __init__(self, env: GdVecEnv, model, device: torch.device, amp: str):
        self.env = env
        self.device = device
        self.amp = amp
        self.snapshot = make_model(env.obs_dim, model_arch(model),
                                   model.logit_bound, n_actions=env.n_actions,
                                   obs_version=env.obs_version).to(device)
        self.thread: threading.Thread | None = None
        self.result: tuple[float, bool, dict] | None = None

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, model, optimizer, update: int, extra: dict,
              blob: np.ndarray | None = None,
              prefix: np.ndarray | None = None) -> None:
        """blob/prefix: stitched start — the run begins from this state
        snapshot and its reported holds are prefix + the run's own."""
        assert not self.busy()
        self.snapshot.load_state_dict(model.state_dict())
        self.snapshot.logit_bound = model.logit_bound
        model_state = {k: v.detach().to("cpu", copy=True)
                       for k, v in model.state_dict().items()}
        opt_state = optimizer.state_dict()
        opt_state = {"state": {i: {k: (v.detach().to("cpu", copy=True)
                                       if torch.is_tensor(v) else v)
                                   for k, v in s.items()}
                               for i, s in opt_state["state"].items()},
                     "param_groups": opt_state["param_groups"]}
        info = {"update": update, "model": model_state, "optimizer": opt_state,
                "logit_bound": model.logit_bound, **extra}
        self.result = None
        self.thread = threading.Thread(
            target=self._run,
            args=(info, blob, np.zeros(0, np.uint8) if prefix is None
                  else np.asarray(prefix, np.uint8)),
            daemon=True)
        self.thread.start()

    def _run(self, info: dict, blob, prefix: np.ndarray) -> None:
        x, won, holds, acts = self.rollout(self.snapshot, blob)
        info["holds"] = np.concatenate([prefix, holds])
        info["suffix_holds"] = holds
        info["suffix_actions"] = acts
        info["stitched"] = blob is not None
        self.result = (x, won, info)

    def rollout(self, model, blob) -> tuple[float, bool, np.ndarray, np.ndarray]:
        """Synchronous argmax run on this handle: (final x, won, suffix
        holds, suffix actions)."""
        env, dev = self.env, self.device
        env.reset()
        if blob is not None:
            env.set_state(0, blob)
        obs_d = torch.as_tensor(env._obs.copy(), device=dev)
        while True:
            mask_d = (torch.as_tensor(env.action_masks().astype(np.float32),
                                      device=dev) if model.categorical else None)
            with torch.no_grad(), autocast_ctx(dev, self.amp):
                logit, _ = model(obs_d, mask=mask_d)
            a = model.greedy(logit).to(torch.uint8).cpu().numpy()
            next_d, _, dterm, dtrunc, dinfo = env.step(a)
            if dterm[0] or dtrunc[0]:
                return (float(dinfo["final_x"][0]), bool(dinfo["won"][0]),
                        env.last_episode_holds(0), env.last_episode_actions(0))
            obs_d = torch.as_tensor(next_d, device=dev)

    def poll(self, wait: bool = False):
        """The finished (det_x, det_win, snapshot) or None; wait=True joins."""
        if self.thread is None:
            return None
        if wait:
            self.thread.join()
        if self.thread.is_alive():
            return None
        r, self.result, self.thread = self.result, None, None
        return r


class WallSolver:
    """Background exact search (gdenv_solve) from states along a known
    from-start hold sequence — the failing det run — toward a target x. Own
    env handle (the DLL releases the GIL, so the search overlaps the
    rollout). Result: verified (start blob, holds, start_x, reach_x,
    actions) demo segments plus (x, blob) snapshots every ~120 units along
    each, for the respawn bank, and the per-start planner statistics."""

    def __init__(self, env: GdVecEnv, max_nodes: int, max_states: int):
        self.env = env
        self.max_nodes = max_nodes
        self.max_states = max_states
        self.thread: threading.Thread | None = None
        self.result: dict | None = None

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, holds: np.ndarray, xs: list[float], target_x: float,
              extra_blobs: list[tuple[float, np.ndarray]] = ()) -> None:
        assert not self.busy()
        self.result = None
        self.thread = threading.Thread(
            target=self._run, args=(np.asarray(holds, np.uint8), list(xs),
                                    float(target_x), list(extra_blobs)),
            daemon=True)
        self.thread.start()

    def _run(self, holds, xs, target_x, extra) -> None:
        env = self.env
        t0 = time.perf_counter()
        starts: list[tuple[float, np.ndarray]] = []
        if len(holds):
            for x, b in zip(xs, env.trace_states(holds, xs)):
                if b is not None:
                    starts.append((x, b))
        starts += extra
        starts = starts[:env.n_envs]
        out = {"demos": [], "bank": [], "n_starts": len(starts),
               "target_x": target_x, "reach": 0.0, "secs": 0.0, "nodes": 0,
               "status": {}, "verified": 0, "unverified": 0}
        if not starts:
            self.result = out
            return
        sols = env.solve([b for _, b in starts], target_x, self.max_nodes,
                         self.max_states)
        out["reach"] = max((s["reach"] for s in sols), default=0.0)
        out["nodes"] = sum(s["nodes"] for s in sols)
        for s in sols:
            out["status"][s["status"]] = out["status"].get(s["status"], 0) + 1
        ok = []
        for (x, b), s in zip(starts, sols):
            if s["holds"] is None:
                continue
            # Replay through the shared executor before admission (the
            # DLL already did once; this is the trainer-side gate).
            good, fx = sil.verify_demo(env, b, s["holds"], s["actions"],
                                       min(target_x, env.end_x))
            if good:
                ok.append((x, b, s["holds"], s["actions"]))
                out["verified"] += 1
            else:
                out["unverified"] += 1
        if ok:
            # Snapshots along each demo: walk them in lockstep on this
            # handle (set_state + the replayed decisions).
            _, acts, _, _, n_steps, _, _, _ = sil._replay(
                env, [h for _, _, h, _ in ok], [b for _, b, _, _ in ok],
                [a for _, _, _, a in ok])
            for i, (x, b, _, _) in enumerate(ok):
                env.set_state(i, b)
            last = np.array([x for x, _, _, _ in ok])
            a = np.zeros(env.n_envs, np.uint8)
            for t in range(int(n_steps.max())):
                a[:] = 0
                for i in range(len(ok)):
                    if t < n_steps[i]:
                        a[i] = acts[i, t]
                env.step(a)
                rs = env.get_render_states()
                for i in range(len(ok)):
                    if t < n_steps[i] - 1 and rs[i, 4] == 0.0 \
                            and rs[i, 0] > last[i] + 120.0:
                        out["bank"].append((float(rs[i, 0]), env.get_state(i)))
                        last[i] = rs[i, 0]
            env.reset()
            out["demos"] = [(b, h, x, target_x, ac) for x, b, h, ac in ok]
        out["secs"] = time.perf_counter() - t0
        self.result = out

    def poll(self, wait: bool = False):
        if self.thread is None:
            return None
        if wait:
            self.thread.join()
        if self.thread.is_alive():
            return None
        r, self.result, self.thread = self.result, None, None
        return r


def model_arch(model) -> str:
    return "cnn" if hasattr(model.pi_body, "conv") else "mlp"


def check_contracts(ckpt: dict, env: GdVecEnv) -> tuple[bool, list[str]]:
    """(legacy_checkpoint, mismatches). A checkpoint without a contracts
    block predates them: acceptable only on the legacy layout / action
    space (its snapshots get migrated), never otherwise."""
    have = env.contracts()
    want = ckpt.get("contracts")
    if want is None:
        bad = []
        if env.obs_version != OBS_VERSION:
            bad.append(f"obs_version legacy(5) != {env.obs_version}")
        if env.action_mode != "macro":
            bad.append(f"action_mode macro != {env.action_mode}")
        return True, bad
    bad = []
    for key in ("obs_version", "action_mode", "n_actions", "snapshot_version",
                "sim_version", "level_fingerprint", "sim_fingerprint",
                "trajectory_version"):
        if key in want and want[key] != have[key]:
            bad.append(f"{key} {want[key]} != {have[key]}")
    return False, bad


def main():
    args = parse_args()
    configure_backend()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = Path(__file__).resolve().parents[2] / "runs" / args.run_name
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    t_wall0 = time.time()

    precision = args.action_mode == "precision"
    # Precision mode accumulates discounted per-tick rewards inside the
    # step (gamma^duration bootstrapping then follows from the buffer's
    # per-step discount); the macro mode keeps its flat step reward.
    reward_gamma = args.gamma if precision else 0.0
    env_kw = dict(obs_version=args.obs_version, action_mode=args.action_mode,
                  reward_gamma=reward_gamma)
    env = GdVecEnv(args.level, args.n_envs, args.ticks_per_action, args.end_x,
                   pinned=device.type == "cuda", **env_kw)
    contracts = env.contracts()
    cfg = PPOConfig(rollout_steps=args.rollout, lr=args.lr,
                    ent_coef=args.ent_coef, gamma=args.gamma,
                    gae_lambda=args.gae_lambda, amp=args.amp,
                    duration_lambda=precision)
    ckpt = None
    legacy_ckpt = False
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        args.arch = ckpt.get("config", {}).get("arch", "mlp")
        legacy_ckpt, bad = check_contracts(ckpt, env)
        if bad:
            raise SystemExit(
                f"resume refused: {args.resume} was trained under other "
                f"contracts ({'; '.join(bad)}). Start a fresh run — a "
                f"checkpoint's SIL buffer, respawn bank and certificate "
                f"assume its own layout, action space and dynamics.")
    model = make_model(env.obs_dim, args.arch, logit_leak=args.logit_leak,
                       n_actions=env.n_actions, obs_version=env.obs_version)
    algo = PPO(model, cfg, device)
    buf = RolloutBuffer(cfg.rollout_steps, env.n_envs, env.obs_dim, device,
                        env.n_actions)
    recorder = EpisodeRecorder(env)

    start_update = 0
    best_x = 0.0
    best_mean_x = 0.0
    best_det_x = 0.0
    total_wins = 0
    total_episodes = 0
    polish = False
    elapsed_prev = 0.0  # wall-clock of earlier invocations (resumes)
    first_win_secs = -1.0
    if ckpt is not None:
        migrated = load_model_state(model, ckpt["model"])
        for note in migrated:
            print(f"checkpoint obs migration: {note}")
        if migrated:
            # optimizer moments have the old shapes; start them fresh
            print("checkpoint obs migration: optimizer state reset")
        else:
            algo.optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt["update"]
        best_x = ckpt.get("best_x", 0.0)
        best_mean_x = ckpt.get("best_mean_x", 0.0)
        best_det_x = ckpt.get("best_det_x", 0.0)
        total_wins = ckpt.get("total_wins", 0)
        total_episodes = ckpt.get("total_episodes", 0)
        polish = ckpt.get("polish", False)
        elapsed_prev = float(ckpt.get("elapsed_secs", 0.0))
        first_win_secs = float(ckpt.get("first_win_secs", -1.0))
        model.logit_bound = ckpt.get("logit_bound", model.logit_bound)
        # ckpt config holds the *requested* --end-x (-1 = full level).
        prev_end = ckpt.get("config", {}).get("end_x", -1.0)
        prev_full = prev_end is None or prev_end < 0
        if (prev_full and args.end_x >= 0) or \
                (not prev_full and abs(prev_end - env.end_x) > 0.5):
            print(f"resume: end_x {'full' if prev_full else f'{prev_end:g}'} "
                  f"-> {env.end_x:g}; earlier det wins/ckpt_best_det.pt were "
                  f"for the old end")
        print(f"resumed from {args.resume} at update {start_update} "
              f"(arch {args.arch}{', polish' if polish else ''}"
              f"{', legacy checkpoint: snapshots migrated' if legacy_ckpt else ''})")
    model = maybe_compile(model, args.compile)

    (run_dir / "config.json").write_text(json.dumps(
        {**vars(args), "obs_dim": env.obs_dim, "end_x": env.end_x,
         # end_x above is the *effective* end (a full-level run records the
         # level's natural end); end_x_arg is what was asked for, so a
         # resume/prefill of a full-level run never inherits a hard cutoff.
         "end_x_arg": args.end_x, "contracts": contracts,
         "n_actions": env.n_actions}, indent=2))
    # Fresh (non-resume) runs truncate: appending a restarted run to the old
    # log makes the curves unreadable.
    log_path = run_dir / "log.csv"
    if not args.resume or not log_path.exists():
        log_path.write_text(LOG_COLUMNS)
    # Graceful stop for the viewer GUI: it drops runs/<name>/STOP and this
    # loop saves ckpt_latest and exits at the end of the current update. A
    # stale file from a previous run must not kill this one.
    stop_path = run_dir / "STOP"
    stop_path.unlink(missing_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run_dir / "tb"))
    except ImportError:
        tb = None

    live = LiveWriter(run_dir / "live.bin", env.n_envs) if args.watch else None

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    env.reset()
    if env.obs_tensor is None:  # CPU / no-CUDA path: plain host tensor
        env.obs_tensor = torch.from_numpy(env._obs)
        env.final_obs_tensor = torch.from_numpy(env._final_obs)
    # Host staging for the per-step action/logprob/value device->host copy.
    alv_host = torch.empty(3, env.n_envs,
                           pin_memory=device.type == "cuda")
    # Rollout forward as a CUDA graph (launch-bound at batch 1024); its
    # static input tensor is where the env obs lands each step. Off under
    # --compile (the compiled forward owns its own graphs) or --no-graph.
    gact = GraphedAct(model, env.n_envs, env.obs_dim, device, cfg.amp)
    if args.compile or args.no_graph:
        gact.enabled = False
    mask_host = torch.ones(env.n_envs, env.n_actions,
                           pin_memory=device.type == "cuda")
    mask_np = mask_host.numpy()
    global_ticks = 0
    wins_written = 0
    last_live = 0.0
    t_start = time.time()
    ticks_window = 0
    # Rates of the last completed update; what live.bin publishes.
    last_sps = 0.0
    last_samples_s = 0.0

    # Snapshot-respawn curriculum: bank live states in 500-unit x buckets;
    # dead envs respawn from a random banked state with prob --respawn.
    # Half of respawns target the walls where envs currently die (a state
    # a few hundred units before a recently observed death x — the spot
    # that actually needs practice); the rest draw uniformly over the bank
    # for retention. Episodes not started at x=0 are excluded from progress
    # stats, traces, and win certificates.
    from collections import deque
    rng = np.random.default_rng(args.seed)
    bank: dict[int, list[tuple[float, np.ndarray]]] = {}  # bucket -> (x, blob)
    kBucket = 500.0
    kBankCap = 16
    recent_deaths: deque[float] = deque(maxlen=256)
    from_start = np.ones(env.n_envs, bool)
    collapse_count = 0

    def bank_add(x: float, blob: np.ndarray):
        blobs = bank.setdefault(int(x // kBucket), [])
        blobs.append((x, blob))
        if len(blobs) > kBankCap:
            blobs.pop(0)

    def bank_pick(lo: float, hi: float):
        """A random banked (x, blob) with x in [lo, hi], or None."""
        b0, b1 = int(max(lo, 0.0) // kBucket), int(max(hi, 0.0) // kBucket)
        cands = [t for b in range(b0, b1 + 1) for t in bank.get(b, ())
                 if lo <= t[0] <= hi]
        return cands[int(rng.integers(len(cands)))] if cands else None

    # Frontier-stall scout (Go-Explore style). Ship walls need a coordinated
    # multi-decision hold pattern: measured at the x=9026 wall, the policy's
    # own sampling passed 0/10k attempts while uniform-random actions from
    # ~300 units back passed ~2% — rare for on-policy exploration, trivial
    # for a brute-force burst.
    #
    # Walls come in two shapes, needing opposite action densities: ship
    # corridors want dense holds (p≈0.5), while low overhangs (x=18645:
    # blocks at y=75 with spikes hanging under them) are passed by NOT
    # jumping for ~30 straight decisions — P(pass) ≈ 2^-30 at p=0.5, but
    # certain at p=0 from a grounded pre-wall state. Each iteration
    # therefore sweeps the hold probability, and restarts from the deepest
    # banked states so far, so mixed walls fall incrementally (explore_chain
    # below). Run once at startup for full-level coverage, and again on
    # frontier stalls if coverage was lost.
    scout = GdVecEnv(args.level, 128, args.ticks_per_action, args.end_x, **env_kw)
    frontier_x = 0.0
    stall_count = 0
    respawn_prob = args.respawn

    def random_actions(n: int, p_hold: float) -> np.ndarray:
        """Scout actions at hold density p_hold in the env's action space.
        Precision: a hold drawn on the ground is the hop macro (one whole
        hop per decision, like the macro space's action 1 — a random H4 /
        R4 stream covers ~640 ticks per 160-step burst and never leaves
        the first 1000 units), airborne / flying it is H4, a release R4."""
        hold = rng.random(n) < p_hold
        if not precision:
            return hold.astype(np.uint8)
        m = scout.action_masks()
        return np.where(hold, np.where(m[:, ACT_HOP] > 0, ACT_HOP, 5),
                        2).astype(np.uint8)  # HOP / H4 / R4

    # Deterministic full-run eval on a dedicated 1-env handle, off-thread.
    # The argmax policy is what open-loop deployment replays, so it gets its
    # own gate and its own best checkpoint (stochastic mean_x drives
    # ckpt_best).
    det_env = GdVecEnv(args.level, 1, args.ticks_per_action, args.end_x, **env_kw)
    det = DetEvaluator(det_env, model, device, args.amp)
    # Synchronous probe rollouts (widening gate, standalone check) on their
    # own handle: det_env belongs to the background thread.
    probe_env = GdVecEnv(args.level, 1, args.ticks_per_action, args.end_x, **env_kw)
    probe = DetEvaluator(probe_env, model, device, args.amp)

    def bank_deepest() -> float:
        return max((t[0] for b in bank for t in bank[b]), default=0.0)

    def explore_chain(iters: int) -> tuple[int, float]:
        """Chained Go-Explore from self-visited states only. Each iteration
        restarts from the deepest states banked SO FAR — including states
        banked by earlier iterations of this same call — so long mixed walls
        (hop onto a ledge, then a no-jump corridor) are crossed
        incrementally, ~120 banked units per hop, rather than needing one
        lucky action sequence end to end. Measured from scratch this covers
        the full level in a few seconds. Returns (n_banked, deepest_x)."""
        ns = scout.n_envs
        banked = 0
        start_x = np.zeros(ns)
        last_bank = np.zeros(ns)
        if not bank:
            # Bootstrap: random from-start rollouts (auto-reset on death).
            scout.reset()
            for _ in range(200):
                scout.step(random_actions(ns, 0.5))
                rs = scout.get_render_states()
                for i in range(ns):
                    if rs[i, 0] > 400.0 and rs[i, 4] == 0.0:
                        bank_add(float(rs[i, 0]), scout.get_state(i))
                        banked += 1
        deep_prev, stale = 0.0, 0
        for it in range(iters):
            p_hold = (0.5, 0.25, 0.1, 0.0)[it % 4]
            deep = bank_deepest()
            if deep >= env.end_x - 200.0:
                break
            # three full p-hold sweeps without gaining depth -> stuck/done
            stale = stale + 1 if deep <= deep_prev + 1.0 else 0
            if stale >= 12:
                break
            deep_prev = deep
            starts = [t for b in bank for t in bank[b]
                      if deep - 450.0 <= t[0] <= deep]
            if not starts:
                break
            for i in range(ns):
                x0, blob = starts[int(rng.integers(len(starts)))]
                scout.set_state(i, blob)
                start_x[i] = x0
            alive = np.ones(ns, bool)
            last_bank[:] = start_x
            for _ in range(160):
                _, _, sterm, strunc, _ = scout.step(random_actions(ns, p_hold))
                alive &= ~(sterm | strunc).astype(bool)
                if not alive.any():
                    break
                rs = scout.get_render_states()
                for i in np.nonzero(alive)[0]:
                    x = float(rs[i, 0])
                    if x > last_bank[i] + 120.0:
                        bank_add(x, scout.get_state(int(i)))
                        last_bank[i] = x
                        banked += 1
        return banked, bank_deepest()

    # Prioritized-respawn bookkeeping: per-spawn-bucket attempt/pass counts
    # for respawned episodes, EMA-decayed so they track the current policy.
    # "Pass" = won, or progressed a full bucket past the spawn point.
    respawn_attempts: dict[int, float] = {}
    respawn_passes: dict[int, float] = {}
    respawn_x = np.full(env.n_envs, -1.0)  # spawn x; -1 = from-start
    window_deaths: list[float] = []
    # Latest det-eval reach: the argmax policy provably clears everything
    # before it, so that prefix is the "mastered zone" (see --explore-margin).
    det_x_latest = 0.0
    # Per-env logit bound for the current step (pinned host + graph input).
    bound_host = torch.full((env.n_envs,), EXPLORE_LOGIT_BOUND,
                            pin_memory=device.type == "cuda")
    bound_np = bound_host.numpy()

    sil_buf = sil.SILBuffer()
    silu = sil.SIL(model, algo.optimizer, device, args.gamma,
                   args.reward_scale, coef=args.sil_coef, amp=args.amp,
                   anchor_coef=args.anchor_coef,
                   anchor_margin=args.anchor_margin,
                   teacher_coef=args.teacher_coef,
                   demo_in_sil=bool(args.demo_sil))
    # Replay handle sized to the replay batch (episodes + demo segments +
    # the anchor run): gdenv steps every env in a handle, so borrowing the
    # 128-env scout would waste 16x the sim work.
    sil_env = (GdVecEnv(args.level, args.sil_episodes + args.sil_demos + 1,
                        args.ticks_per_action, args.end_x, **env_kw)
               if args.sil_coef > 0 or args.teacher_coef > 0 else None)
    # Planner (see --wall-solve) on its own handle; k envs = k parallel
    # solves and the lockstep walk that snapshots the found segments.
    kSolveExtra = 3  # banked starts added to the det-path ones
    solver = (WallSolver(GdVecEnv(args.level,
                                  max(1, args.solve_starts) + kSolveExtra,
                                  args.ticks_per_action, args.end_x, **env_kw),
                         args.solve_nodes, args.solve_states)
              if args.wall_solve else None)
    solve_back = args.solve_back
    # The certificate: the best det run's from-x=0 holds and its reach
    # (certificate.py). anchor_* are kept as aliases for the non-stitched
    # BC anchor (sil.py).
    cert = Certificate()
    attach = Attachments(env.n_envs)
    det_low_count = 0  # consecutive det evals below the rollback fraction
    det_stall = 0      # consecutive det evals without a new best
    stitch_stall = 0   # consecutive det evals without certificate growth
    stitch_back_now = args.stitch_back
    demo_verified = 0
    demo_unverified = 0

    # Restore trainer state that isn't the network: exploration coverage,
    # respawn stats, SIL episodes, rng, and the anneal position (without it
    # a resume restarts lr/entropy annealing from the top).
    progress0 = 0.0
    if ckpt is not None:
        migrate = env.migrate_blob if legacy_ckpt else (lambda b: b)
        progress0 = ckpt.get("anneal_progress", 0.0)
        frontier_x = ckpt.get("frontier_x", 0.0)
        n_dropped = 0
        for bx, blob in ckpt.get("bank", []):
            mb = migrate(blob)
            if mb is None:
                n_dropped += 1
                continue
            bank_add(bx, mb)
        recent_deaths.extend(ckpt.get("recent_deaths", []))
        if "rng_state" in ckpt:
            rng.bit_generator.state = ckpt["rng_state"]
        if "sil" in ckpt:
            sil_buf.load_state_dict(ckpt["sil"], migrate)
        respawn_attempts = ckpt.get("respawn_attempts", respawn_attempts)
        respawn_passes = ckpt.get("respawn_passes", respawn_passes)
        det_x_latest = ckpt.get("det_x_latest", 0.0)
        if "certificate" in ckpt:
            cert.load_state_dict(ckpt["certificate"])
        else:
            cert.replace(np.asarray(ckpt.get("anchor_holds", []), np.uint8),
                         ckpt.get("anchor_x", 0.0))
        solve_back = ckpt.get("solve_back", solve_back)
        stitch_back_now = ckpt.get("stitch_back_now", stitch_back_now)
        stitch_stall = ckpt.get("stitch_stall", 0)
        if bank:
            print(f"restored bank: {sum(len(v) for v in bank.values())} "
                  f"states, deepest {bank_deepest():.0f}; "
                  f"anneal progress {progress0:.2f}"
                  f"{f'; {n_dropped} snapshots not migrated' if n_dropped else ''}")
        if start_update >= args.updates:
            raise SystemExit(
                f"checkpoint is at update {start_update} but --updates is "
                f"{args.updates}: nothing to do. Pass a larger --updates "
                f"(the lr/entropy anneal spans updates 0..--updates, so "
                f"extending a finished run re-opens the schedule).")
        if progress0 >= 0.999:
            print(f"note: checkpoint's anneal was complete; re-spanning it "
                  f"over --updates {args.updates} (progress "
                  f"{(start_update + 1) / args.updates:.2f} at resume)")

    def verify_holds(holds: np.ndarray) -> tuple[float, bool]:
        """Raw per-tick replay (deployment semantics): (final x, won)."""
        x, won, _ = scout.run_holds(holds)
        return x, won

    if args.import_certificate:
        # Existing certificates may be imported only after raw replay
        # verification; --import-cut-x truncates them at a decision
        # boundary (the corridor benchmarks start right before a section).
        holds = load_certificate_file(Path(args.import_certificate))
        x_raw, won_raw = verify_holds(holds)
        if args.import_cut_x > 0:
            if x_raw < args.import_cut_x:
                raise SystemExit(f"import refused: raw replay of "
                                 f"{args.import_certificate} reaches x "
                                 f"{x_raw:.0f} < cut {args.import_cut_x:.0f}")
            obs_r, _, _, ticks_r, n_r, _, _ = scout.replay([holds], None)
            xs_r = obs_r[0, :n_r[0], 5] * env.end_x
            hit = np.nonzero(xs_r >= args.import_cut_x)[0]
            if len(hit) == 0:
                raise SystemExit("import refused: cut x past the replayed run")
            tick = int(ticks_r[0, :hit[0]].sum())
            holds = holds[:tick]
            x_raw, won_raw = verify_holds(holds)
        elif not won_raw and x_raw < env.end_x:
            print(f"import: raw replay dies at x {x_raw:.0f}; the certificate "
                  f"is the holds up to that death")
        cert.replace(holds, x_raw)
        print(f"imported certificate: {len(holds)} ticks to x {x_raw:.0f}"
              f"{' (WIN)' if won_raw else ''}")

    # Startup coverage: run the chained explorer once so the respawn bank
    # spans everything randomly reachable before the first update. Pure
    # exploration — the bank never holds a state the agent didn't reach.
    # (With a restored full-coverage bank it exits immediately.)
    n_banked, deep0 = explore_chain(600)
    print(f"explore: banked {n_banked} states, deepest x {deep0:.0f} "
          f"({100 * deep0 / env.end_x:.1f}% of level)")

    # Stitched frontier (see --stitch). The frontier is the certificate's
    # state stitch_back_now before cert.x. Frontier episodes remember their
    # certificate id and prefix length (Attachments): a later certificate
    # normally extends the prefix they were cut from; if it replaces an
    # earlier section (widened overlap, import), they are reset.
    front_blob: np.ndarray | None = None
    front_tick = 0
    front_x = 0.0

    def set_frontier() -> bool:
        """Recompute the frontier from the certificate; True if it moved."""
        nonlocal front_blob, front_tick, front_x
        x0 = cert.x - stitch_back_now
        if not args.stitch or len(cert) == 0 or x0 <= 0.0:
            moved = front_blob is not None
            front_blob, front_tick, front_x = None, 0, 0.0
            return moved
        blob = scout.trace_states(cert.holds, [x0])[0]
        if blob is None:
            return False
        # Prefix length = ticks consumed before the first decision boundary
        # with x >= x0 (trace_states' rule), read off a replay rather than
        # the blob's layout.
        obs, _, _, ticks, n_steps, _, _ = scout.replay([cert.holds], None)
        xs = obs[0, :n_steps[0], 5] * env.end_x
        hit = np.nonzero(xs >= x0)[0]
        if len(hit) == 0:
            return False
        moved = front_blob is None or abs(front_x - x0) > 0.5
        front_blob, front_x = blob, x0
        front_tick = int(ticks[0, :hit[0]].sum())
        return moved

    spawn_blob: list = [None] * env.n_envs

    def spawn_frontier(i: int) -> None:
        """Start env i a new certificate-candidate episode: from the
        frontier snapshot, or x=0 without one."""
        if front_blob is not None:
            env.set_state(i, front_blob)
            attach.attach(i, cert.id, front_tick)
            spawn_blob[i] = front_blob
        else:
            env.reset_env(i)
            attach.attach(i, cert.id, 0)
            spawn_blob[i] = None
        from_start[i] = True
        respawn_x[i] = -1.0

    def install_certificate(holds: np.ndarray, x: float,
                            actions: np.ndarray | None) -> str:
        """Replace the certificate (after raw verification) and reset the
        live episodes whose prefix it no longer contains. Returns a log
        fragment."""
        nonlocal stitch_stall, stitch_back_now
        new_id, common = cert.replace(holds, x, actions)
        orphans = attach.reattach(new_id, common)
        for i in orphans:
            spawn_frontier(int(i))
        # Verified progress: the base overlap again.
        stitch_stall = 0
        stitch_back_now = args.stitch_back
        return (f" | CERT #{new_id} -> {x:.0f} ({len(holds)} ticks"
                f"{f', {len(orphans)} episodes reset' if len(orphans) else ''})")

    if set_frontier():
        for i in range(env.n_envs):
            spawn_frontier(i)
    if args.stitch:
        print(f"stitch: frontier x {front_x:.0f} (certificate #{cert.id} to "
              f"{cert.x:.0f}, prefix {front_tick} ticks, overlap "
              f"{stitch_back_now:.0f})")

    def save_ckpt(path, update_done: int, progress: float, mean_x: float,
                  model_state=None, opt_state=None, logit_bound=None):
        torch.save({"model": model_state or model.state_dict(),
                    "optimizer": opt_state or algo.optimizer.state_dict(),
                    "update": update_done, "best_x": best_x,
                    "best_mean_x": best_mean_x,
                    "best_det_x": best_det_x,
                    "total_wins": total_wins,
                    "total_episodes": total_episodes,
                    "mean_final_x": mean_x,
                    "polish": polish,
                    "logit_bound": (model.logit_bound if logit_bound is None
                                    else logit_bound),
                    "anneal_progress": progress,
                    "frontier_x": frontier_x,
                    "bank": [t for b in bank for t in bank[b]],
                    "recent_deaths": list(recent_deaths),
                    "rng_state": rng.bit_generator.state,
                    "respawn_attempts": respawn_attempts,
                    "respawn_passes": respawn_passes,
                    "det_x_latest": det_x_latest,
                    # anchor_* kept for eval.py / older readers
                    "anchor_holds": cert.holds,
                    "anchor_x": cert.x,
                    "certificate": cert.state_dict(),
                    "solve_back": solve_back,
                    "stitch_back_now": stitch_back_now,
                    "stitch_stall": stitch_stall,
                    "elapsed_secs": elapsed_prev + (time.time() - t_wall0),
                    "first_win_secs": first_win_secs,
                    "contracts": contracts,
                    "sil": sil_buf.state_dict(),
                    "config": vars(args)}, path)

    def write_solution(holds: np.ndarray, deterministic: bool, stitched: bool,
                       standalone: bool | None) -> dict:
        from .eval import holds_to_ranges, level_id_of
        sol = {"tps": traces.TPS,
               "level_id": level_id_of(args.level),
               "level_end_x": float(env.end_x),
               "n_ticks": int(len(holds)),
               "deterministic_policy": deterministic,
               "stitched": stitched,
               "standalone_policy": standalone,
               "contracts": contracts,
               "holds": holds_to_ranges(np.asarray(holds, np.uint8))}
        (run_dir / "solution.json").write_text(json.dumps(sol, indent=2))
        return sol

    stopped = False
    update = start_update
    progress = 0.0
    mean_x = 0.0
    updates_this_run = 0
    try:
        for update in range(start_update, args.updates):
            t_upd0 = time.perf_counter()
            window_final_x = []
            window_wins = 0
            window_ep_ticks = []
            window_reach = 0.0  # max final_x over ALL episodes (respawned too)
            # Rollout sub-phases: gpu = H2D + act + D2H sync, env = DLL step,
            # misc = Python bookkeeping (recorder, bank, respawn).
            t_roll_gpu = t_roll_env = t_roll_misc = 0.0
            n_explore = 0
            dur_hist = np.zeros(4)  # 1-tick, 2-tick, 4-tick, hop decisions
            t_step0 = time.perf_counter()
            buf.reset()
            # Mastered zone for this update: x < best det reach - margin.
            # The BEST run, not the latest: one bad det eval (a single
            # flipped prefix decision) must not turn the whole level into
            # the explore zone — measured: xpl 0.00 -> 0.76, prefix deaths
            # x65/update, two collapse rollbacks. The anchor repairs the
            # flip; the det rollback below is the backstop.
            # Stitched: nothing spawns before the frontier, so it is all
            # explore zone.
            explore_lo = (front_x if args.stitch else
                          max(det_x_latest, cert.x) - args.explore_margin
                          if args.explore_margin >= 0 else -1.0)
            for step_i in range(cfg.rollout_steps):
                # The env's obs buffer (pinned) is the source of truth: H2D into
                # the graph's static input, act, then keep a copy in the rollout
                # slot (device->device, async and cheap).
                gact.obs.copy_(env.obs_tensor, non_blocking=True)
                # Exploration floor only at/after the frontier; the mastered
                # prefix runs near-argmax so from-start envs actually get there.
                explore = env.player_x() >= explore_lo
                n_explore += int(explore.sum())
                bound_np[:] = np.where(explore, EXPLORE_LOGIT_BOUND,
                                       args.mastered_bound)
                ent_w_np = np.where(explore, 1.0, args.mastered_ent).astype(
                    np.float32)
                gact.bound.copy_(bound_host, non_blocking=True)
                if model.categorical:
                    mask_np[:] = env.action_masks()
                    gact.mask.copy_(mask_host, non_blocking=True)
                alv = gact()
                buf.obs[step_i].copy_(gact.obs, non_blocking=True)
                alv_host.copy_(alv)  # one D2H sync: action, logprob, value
                alv_np = alv_host.numpy()
                actions_np = alv_np[0].astype(np.uint8)
                t_step1 = time.perf_counter()
                next_obs, rew, term, trunc, info = env.step(actions_np)
                t_step2 = time.perf_counter()
                done = (term | trunc).astype(bool)
                if precision:
                    d = np.where(actions_np == ACT_HOP, 3,
                                 np.minimum(actions_np % 3, 2))
                    dur_hist += np.bincount(d, minlength=4)[:4]
                else:
                    dur_hist[2] += int((info["ticks"] <= args.ticks_per_action).sum())
                    dur_hist[3] += int((info["ticks"] > args.ticks_per_action).sum())

                rewards = rew * args.reward_scale
                discounts = (args.gamma ** info["ticks"]).astype(np.float32)
                # Episode-end fields for this step; bursts (below) are
                # trainer-side truncations the DLL knows nothing about.
                final_x_i = info["final_x"].copy()
                won_i = info["won"].copy()
                burst_i = np.zeros(env.n_envs, bool)
                trunc_i = trunc.astype(bool)
                if args.burst > 0:
                    x_now = env.player_x()
                    burst_i = ((respawn_x >= 0.0) & ~done
                               & (x_now >= respawn_x + args.burst))
                    if burst_i.any():
                        final_x_i[burst_i] = x_now[burst_i]
                        won_i[burst_i] = 0.0
                        done = done | burst_i
                # Bootstrap V(final obs) for tick-cap truncations and bursts
                # in one forward THROUGH THE ROLLOUT GRAPH: an eager forward
                # on a batch whose size varies per step makes
                # cudnn.benchmark re-autotune the convs for every new size
                # (measured: rollout misc time 0.01 s -> 0.2-1.5 s/update
                # while bursts are frequent). The graph's static input is
                # free to reuse here — it is refilled at the next step. A
                # burst env is still alive, so its current obs row is the
                # final obs. (The value head does not read the mask.)
                boot = trunc_i | burst_i
                if boot.any():
                    k = int(boot.sum())
                    fo = np.empty((k, env.obs_dim), np.float32)
                    nt = int(trunc_i.sum())
                    if nt:
                        fo[:nt] = info["final_obs"][trunc_i]
                    if k > nt:
                        fo[nt:] = env._obs[burst_i]
                    gact.obs[:k].copy_(torch.from_numpy(fo))
                    v_final = gact()[2, :k].float().cpu().numpy()
                    rewards = rewards.copy()
                    rewards[trunc_i] += discounts[trunc_i] * v_final[:nt]
                    rewards[burst_i] += discounts[burst_i] * v_final[nt:]
                buf.add(alv_np, rewards, discounts, done.astype(np.float32),
                        bound_np, ent_w_np,
                        mask_np if model.categorical else None,
                        info["ticks"])

                # Progress stats and replayable traces only from x=0 episodes.
                done_fs = done & from_start
                if done_fs.any():
                    # The env records only the suffix after its snapshot
                    # spawn; traces/wins get prefix + suffix, SIL replays
                    # the suffix from the spawn blob. A stitched episode's
                    # prefix is certificate[:prefix_ticks] of the
                    # certificate it is attached to (never a newer one:
                    # install_certificate resets orphans).
                    idx_fs = np.nonzero(done_fs)[0]
                    suffixes = {int(i): env.last_episode_holds(int(i))
                                for i in idx_fs}
                    suffix_acts = ({int(i): env.last_episode_actions(int(i))
                                    for i in idx_fs} if precision else {})
                    recorder.record(
                        done_fs, info,
                        lambda i: np.concatenate(
                            [cert.holds[:attach.prefix_ticks[i]], suffixes[i]]))
                    for i, sfx in suffixes.items():
                        sil_buf.add(sfx, float(info["final_x"][i]),
                                    bool(info["won"][i]), spawn_blob[i],
                                    suffix_acts.get(i))
                if done_fs.any():
                    fx = info["final_x"][done_fs]
                    window_final_x.extend(fx.tolist())
                    window_wins += int(info["won"][done_fs].sum())
                    window_ep_ticks.extend(
                        info["episode_ticks"][done_fs].tolist())
                    best_x = max(best_x, float(fx.max()))
                if done.any():
                    window_reach = max(window_reach,
                                       float(final_x_i[done].max()))
                total_episodes += int(done.sum())

                # Bank a few live snapshots; respawn dead envs from the bank.
                # Sample count scales with n_envs so shorter rollouts (fewer
                # sampling steps per update) still refresh the bank at the same
                # rate per env-step.
                if step_i % 4 == 0:
                    rs = env.get_render_states()
                    for i in rng.choice(env.n_envs, max(4, env.n_envs // 64),
                                        replace=False):
                        x = float(rs[i, 0])
                        if x > 400.0 and not done[i] and rs[i, 4] == 0.0:
                            bank_add(x, env.get_state(int(i)))
                done_idx = np.nonzero(done)[0]
                if len(done_idx):
                    # Per-step (not per-death) respawn candidate sets.
                    # Stitched: states behind the frontier are prefix
                    # territory — never respawn there.
                    b_lo = int(front_x // kBucket)
                    learnable = [
                        b for b in bank
                        if b >= b_lo
                        and respawn_attempts.get(b, 0.0) >= 8.0
                        and 0.1 <= (respawn_passes.get(b, 0.0)
                                    / respawn_attempts[b]) <= 0.8]
                    buckets = sorted(b for b in bank if b >= b_lo)
                for i in done_idx:
                    if respawn_x[i] >= 0.0:  # a respawned episode just ended
                        b = int(respawn_x[i] // kBucket)
                        respawn_attempts[b] = respawn_attempts.get(b, 0.0) + 1.0
                        if (won_i[i]
                                or final_x_i[i] >= respawn_x[i] + kBucket):
                            respawn_passes[b] = respawn_passes.get(b, 0.0) + 1.0
                    if not won_i[i] and not burst_i[i]:
                        recent_deaths.append(float(final_x_i[i]))
                        window_deaths.append(float(final_x_i[i]))
                    if bank and buckets and rng.random() < respawn_prob:
                        picked = None
                        u = rng.random()
                        if (det_x_latest > 700.0
                                and rng.random() < args.frontier_respawn):
                            # Frontier: just before where the argmax policy
                            # dies (see --frontier-respawn).
                            picked = bank_pick(max(det_x_latest - 700.0,
                                                   front_x),
                                               det_x_latest - 50.0)
                        if picked is None and u < 0.5 and learnable:
                            # Learnable zone: spawn buckets the policy passes
                            # sometimes but not reliably — where practice moves
                            # the needle fastest.
                            b = learnable[int(rng.integers(len(learnable)))]
                            blobs = bank[b]
                            picked = blobs[int(rng.integers(len(blobs)))]
                        if picked is None and u < 0.75 and recent_deaths:
                            # Wall practice: start a few hundred units before a
                            # recently observed death — finds brand-new walls
                            # before pass-rate stats exist for them.
                            dx = recent_deaths[int(rng.integers(
                                len(recent_deaths)))]
                            picked = bank_pick(max(dx - 450.0, front_x),
                                               dx - 50.0)
                        if picked is None:
                            # Uniform over the bank: retention practice.
                            blobs = bank[buckets[int(rng.integers(len(buckets)))]]
                            picked = blobs[int(rng.integers(len(blobs)))]
                        env.set_state(int(i), picked[1])
                        from_start[i] = False
                        attach.detach(int(i))
                        respawn_x[i] = picked[0]
                    else:
                        # Certificate-candidate episode from the frontier
                        # (or x=0). A burst env is still alive in the DLL;
                        # set_state / reset_env restart it either way.
                        spawn_frontier(int(i))

                step_ticks = int(info["ticks"].sum())
                global_ticks += step_ticks
                ticks_window += step_ticks

                if live and time.time() - last_live > 1.0 / 60.0:
                    # Publish the last completed update's rates, not a
                    # mid-rollout running rate: that one excluded
                    # t_update/t_sil and read ~6x above the log line.
                    live.write(env.get_render_states(), global_ticks, best_x,
                               last_sps, total_episodes, last_samples_s)
                    last_live = time.time()
                t_step3 = time.perf_counter()
                t_roll_gpu += t_step1 - t_step0
                t_roll_env += t_step2 - t_step1
                t_roll_misc += t_step3 - t_step2
                t_step0 = t_step3

            last_obs = torch.as_tensor(env.obs_tensor, device=device)
            sync()
            t_roll = time.perf_counter() - t_upd0

            # Anneal position is absolute: update k of --updates, also across
            # resumes. (The old progress0 + (1-progress0)*... continuation
            # pinned lr at lr_final=0 forever when resuming a finished run.)
            progress = (update + 1) / args.updates
            t0 = time.perf_counter()
            lr_floor = (args.lr_floor if precision and first_win_secs < 0
                        else 0.0)
            if polish:
                lr_now = min(cfg.lr + (cfg.lr_final - cfg.lr) * progress,
                             POLISH_MAX_LR)
                stats = algo.update(buf, last_obs, progress, lr_override=lr_now,
                                    ent_override=POLISH_ENT_COEF,
                                    lr_floor=min(lr_floor, POLISH_MAX_LR))
            else:
                stats = algo.update(buf, last_obs, progress, lr_floor=lr_floor)
            sync()
            t_update = time.perf_counter() - t0
            t0 = time.perf_counter()
            if sil_env is not None:
                stats.update(silu.update(
                    sil_env, sil_buf, args.sil_episodes, rng,
                    cfg.max_grad_norm, k_demos=args.sil_demos,
                    anchor=(cert.holds, cert.x),
                    # Stitched: the prefix is frozen holds, nothing to pin.
                    anchor_x_lo=(-1.0 if args.stitch
                                 else cert.x - args.explore_margin)))
            sync()
            t_sil = time.perf_counter() - t0
            for d in (respawn_attempts, respawn_passes):
                for b in d:
                    d[b] *= 0.99  # EMA: stats track the current policy

            sps = ticks_window / max(time.time() - t_start, 1e-9)
            t_start, ticks_window = time.time(), 0
            total_wins += window_wins
            n_eps = len(window_final_x)
            mean_x = float(np.mean(window_final_x)) if n_eps else 0.0
            max_x = float(np.max(window_final_x)) if n_eps else 0.0
            win_rate = window_wins / n_eps if n_eps else 0.0
            mean_ticks = float(np.mean(window_ep_ticks)) if n_eps else 0.0

            # Det eval: collect the previous eval's result (waiting only if a
            # new one is due or this is the last update), process it
            # (certificate, frontier, planner), THEN launch the next one
            # from the updated frontier — launching first made the next
            # run start from the frontier the result was about to move,
            # and its from-x=0-ish holds later replaced an earlier section
            # (76 episodes reset in the smoke run for nothing).
            det_x, det_win = -1.0, False
            det_due = args.det_every > 0 and (
                (update + 1) % args.det_every == 0 or update + 1 == args.updates)
            t0 = time.perf_counter()
            det_results = [r for r in (det.poll(wait=det_due),) if r is not None]
            t_det_wait = time.perf_counter() - t0
            if det_results:
                det_x, det_win, _ = det_results[-1]
                det_x_latest = det_x
            front_str = ""
            win_str = ""

            def process_det(results: list) -> None:
                nonlocal front_str, win_str, first_win_secs, stitch_stall
                nonlocal det_stall, stitch_back_now
                for dx_, dwin_, dsnap in results:
                    _process_one(dx_, dwin_, dsnap)
                if results and args.explore_margin >= 0:
                    # Demos the frontier has moved past (same reasoning).
                    sil_buf.prune_demos(max(best_det_x, cert.x)
                                        - args.explore_margin)
                if (results and args.stitch and args.stitch_widen and len(cert)
                        and cert.x < env.end_x - 0.5):
                    # (no widening once the certificate is a full run:
                    # every later det eval "stalls" at the win line, and
                    # level12-4 drifted its frontier 4800 back after
                    # winning for nothing)
                    # Widen the overlap after 3 stalled evals (1200, 2400,
                    # 4800, clamped to the level start); the planner's
                    # start window follows. Gated: the argmax policy must
                    # still carry the wider approach back to its current
                    # wall (a probe run from the candidate frontier reaches
                    # cert.x - 50), otherwise the frontier would land in
                    # territory the policy never trained on and every det
                    # eval would die there instead (corridor full-s1:
                    # 4800 back into an unlearned ship section, det 16952
                    # -> 12631 while the stochastic policy was completing).
                    want = widen_schedule(args.stitch_back, stitch_stall,
                                          cap=args.stitch_widen_cap)
                    if want != stitch_back_now:
                        ok = want < stitch_back_now
                        if not ok:
                            x0 = cert.x - want
                            blob = (scout.trace_states(cert.holds, [x0])[0]
                                    if x0 > 0 else None)
                            probe.snapshot.load_state_dict(model.state_dict())
                            probe.snapshot.logit_bound = model.logit_bound
                            px, pwon, _, _ = probe.rollout(probe.snapshot, blob)
                            ok = pwon or px >= cert.x - 50.0
                            if not ok:
                                front_str += (f" | widen -> {want:.0f} refused "
                                              f"(probe from {max(x0, 0):.0f} "
                                              f"dies at {px:.0f})")
                        if ok:
                            stitch_back_now = want
                            front_str += f" | WIDEN overlap -> {stitch_back_now:.0f}"
                if results and set_frontier():
                    # The frontier moved: new episodes and det evals start
                    # from it; live episodes keep running (their prefix is
                    # still the certificate's — install_certificate reset
                    # the ones whose prefix is gone); SIL episodes that
                    # ended behind it are prefix territory.
                    n_drop = sil_buf.drop_before(front_x)
                    front_str += (f" | FRONTIER -> {front_x:.0f} (certificate "
                                  f"#{cert.id} to {cert.x:.0f}, {len(cert)} ticks, "
                                  f"overlap {stitch_back_now:.0f}"
                                  f"{f', {n_drop} sil eps dropped' if n_drop else ''})")

            def _process_one(dx_, dwin_, dsnap) -> None:
                nonlocal front_str, win_str, first_win_secs, stitch_stall
                nonlocal det_stall
                # Certificates advance only on the POLICY'S OWN verified
                # trajectory: the det run's full holds must raw-replay
                # (deployment semantics) to the same reach.
                if (dx_ > cert.x or (dwin_ and first_win_secs < 0)) \
                        and "holds" in dsnap:
                    x_raw, won_raw = verify_holds(dsnap["holds"])
                    if x_raw >= dx_ - 0.5 or won_raw:
                        if dx_ > cert.x:
                            front_str += install_certificate(
                                np.asarray(dsnap["holds"], np.uint8), dx_,
                                dsnap.get("suffix_actions") if precision and
                                not dsnap.get("stitched") else None)
                        if dwin_ and won_raw:
                            # Stitched completion vs standalone policy
                            # completion: a from-x=0 argmax run tells
                            # them apart.
                            standalone = None
                            if dsnap.get("stitched"):
                                probe.snapshot.load_state_dict(dsnap["model"])
                                probe.snapshot.logit_bound = dsnap["logit_bound"]
                                sx, swon, _, _ = probe.rollout(probe.snapshot, None)
                                standalone = bool(swon)
                                win_str += (f" | standalone {'WIN' if swon else f'death at {sx:.0f}'}")
                            else:
                                standalone = True
                            sol = write_solution(dsnap["holds"], True,
                                                 bool(dsnap.get("stitched")),
                                                 standalone)
                            if first_win_secs < 0:
                                first_win_secs = elapsed_prev + (time.time() - t_wall0)
                                win_str += f" | FIRST VERIFIED WIN at {first_win_secs:.0f}s"
                            win_str += (f" | WROTE {run_dir / 'solution.json'} "
                                        f"({sol['n_ticks']} ticks, "
                                        f"{'stitched' if sol['stitched'] else 'from x=0'})")
                    else:
                        front_str += (f" | det run NOT raw-verified (raw "
                                      f"{x_raw:.0f} vs det {dx_:.0f}) — "
                                      f"certificate unchanged")
                else:
                    stitch_stall += 1
                det_stall = 0 if dx_ > best_det_x + 30.0 else det_stall + 1
                # Planner: demos through the wall the argmax policy just
                # died at, from ITS OWN states between the frontier and the
                # death (plus a few banked ones for diversity) — once the
                # det reach has stalled, and again on every later stalled
                # eval so the demos follow the states the policy visits.
                # Skipped while a solve is still running, and skipped when
                # the death is inside the mastered zone (prefix drift,
                # handled by the anchor + det rollback).
                frontier_lo = (max(best_det_x, cert.x) - args.explore_margin
                               if args.explore_margin >= 0 else -np.inf)
                if (solver is not None and not dwin_ and dx_ > 300.0
                        and det_stall >= args.solve_stall
                        and dx_ >= frontier_lo
                        and "holds" in dsnap and not solver.busy()):
                    k = max(1, args.solve_starts)
                    back = max(solve_back, stitch_back_now) if args.stitch else solve_back
                    lo = max(dx_ - back, front_x if args.stitch else 0.0, 50.0)
                    xs = list(np.linspace(lo, max(dx_ - 50.0, lo + 10.0), k))
                    extra = []
                    for _ in range(kSolveExtra):
                        pk = bank_pick(lo, dx_ - 50.0)
                        if pk is not None:
                            extra.append(pk)
                    solver.start(dsnap["holds"], xs,
                                 min(dx_ + args.solve_reach, env.end_x),
                                 extra)
            process_det(det_results)
            if det_due:
                t0 = time.perf_counter()
                det.start(model, algo.optimizer, update + 1,
                          {"progress": progress, "mean_x": mean_x},
                          blob=front_blob, prefix=cert.holds[:front_tick])
                if update + 1 == args.updates:  # nothing left to overlap with
                    last = det.poll(wait=True)
                    det_results.append(last)
                    det_x, det_win, _ = last
                    det_x_latest = det_x
                    process_det([last])
                t_det_wait += time.perf_counter() - t0
            sol_str = ""
            sol_res = solver.poll() if solver is not None else None
            solve_secs = 0.0
            solve_nodes = 0
            if sol_res is not None:
                n_new = 0
                for b, h, sx, tx, ac in sol_res["demos"]:
                    n_new += int(sil_buf.add_demo(b, h, sx, tx, ac))
                for bx, blob in sol_res["bank"]:
                    bank_add(bx, blob)
                demo_verified += sol_res["verified"]
                demo_unverified += sol_res["unverified"]
                if sol_res["demos"]:
                    solve_back = args.solve_back
                else:
                    solve_back = min(solve_back * 2.0, args.solve_back * 8.0)
                solve_secs = sol_res["secs"]
                solve_nodes = sol_res["nodes"]
                x0s = [sx for _, _, sx, _, _ in sol_res["demos"]]
                st = " ".join(f"{k}:{v}" for k, v in sorted(sol_res["status"].items()))
                unv_str = (f", {sol_res['unverified']} unverified"
                           if sol_res["unverified"] else "")
                sol_str = (f" | WALL {len(sol_res['demos'])}/"
                           f"{sol_res['n_starts']} demos"
                           f"{f' from {min(x0s):.0f}-{max(x0s):.0f}' if x0s else ''}"
                           f" -> {sol_res['target_x']:.0f} (reach "
                           f"{sol_res['reach']:.0f}, {n_new} new, "
                           f"{len(sol_res['bank'])} banked, {st}, "
                           f"{solve_nodes / 1e6:.2f}M nodes, {solve_secs:.2f}s"
                           f"{unv_str}"
                           f"{'' if sol_res['demos'] else f', back->{solve_back:.0f}'})")

            # Polish gate: cumulative stochastic from-start wins (the
            # original), a forced update, or the det eval reaching
            # --polish-det of the level. On long levels the stochastic
            # gate never fires (Polargeist: 0 from-start wins in 730
            # updates while the det policy won), so the det gate is the
            # one that matters there.
            det_gate = (args.polish_det > 0 and det_x_latest
                        >= args.polish_det * env.end_x - 0.5)
            if not polish and not args.no_polish and (
                    (args.polish_at >= 0 and update + 1 >= args.polish_at)
                    or (args.polish_at < 0 and (total_wins >= args.polish_wins
                                                or det_gate))):
                polish = True
                # Mastered zone already samples at --mastered-bound; the SIL
                # / eval-side bound follows the polish value as before.
                model.logit_bound = POLISH_LOGIT_BOUND
                respawn_prob = min(respawn_prob, POLISH_RESPAWN)
                print(f"upd {update:4d} | POLISH: {total_wins} wins, det "
                      f"{det_x_latest:.0f} — logit bound {POLISH_LOGIT_BOUND}, "
                      f"ent {POLISH_ENT_COEF}, lr<={POLISH_MAX_LR}, respawn "
                      f"{respawn_prob}")

            elapsed = elapsed_prev + (time.time() - t_wall0)
            row = (f"{update},{global_ticks},{mean_x:.1f},{max_x:.1f},"
                   f"{best_x:.1f},{win_rate:.4f},{n_eps},{mean_ticks:.0f},"
                   f"{sps:.0f},{stats['policy_loss']:.4f},"
                   f"{stats['value_loss']:.4f},{stats['entropy']:.4f},"
                   f"{stats['approx_kl']:.5f},{stats['clip_frac']:.4f},"
                   f"{stats['lr']:.6f},{det_x:.1f},{int(det_win)}")
            det_str = (f" | det {det_x:7.1f}{' WIN' if det_win else ''}"
                       if det_x >= 0 else "")
            die_str = ""
            if window_deaths:
                hist: dict[int, int] = {}
                for dxx in window_deaths:
                    hist[int(dxx // kBucket)] = hist.get(int(dxx // kBucket),
                                                         0) + 1
                top = sorted(hist.items(), key=lambda kv: -kv[1])[:3]
                die_str = " | die " + " ".join(
                    f"{b * kBucket:.0f}x{c}" for b, c in top)
            dur_frac = dur_hist / max(dur_hist.sum(), 1.0)
            teach_str = ""
            if stats.get("teacher_n", 0):
                teach_str = (f" | teach p{stats['teacher_p']:.2f} "
                             f"d{stats['teacher_disagree']:.2f}")
            if tb:
                for k, v in {**stats, "mean_final_x": mean_x, "best_x": best_x,
                             "win_rate": win_rate, "sps": sps,
                             "t_rollout": t_roll, "t_update": t_update,
                             "t_sil": t_sil, "t_det_wait": t_det_wait,
                             "t_roll_gpu": t_roll_gpu, "t_roll_env": t_roll_env,
                             "t_roll_misc": t_roll_misc,
                             "stitch_back": stitch_back_now,
                             "cert_x": cert.x, "solve_secs": solve_secs,
                             "dur1": dur_frac[0], "dur2": dur_frac[1],
                             "dur4": dur_frac[2], "durhop": dur_frac[3]}.items():
                    tb.add_scalar(k, v, global_ticks)
                if det_x >= 0:
                    tb.add_scalar("det_x", det_x, global_ticks)
                tb.add_scalar("front_x", front_x, global_ticks)
                if window_deaths:
                    tb.add_histogram("death_x", np.asarray(window_deaths),
                                     global_ticks)
            window_deaths.clear()

            # Frontier stall -> re-run the chained explorer, but only when the
            # bank doesn't already extend past the stuck wall (bucket FIFO churn
            # or a fresh resume can lose far coverage). If coverage exists, a
            # stall is a learning problem, not an exploration problem — respawn
            # practice already trains both sides of the wall.
            if window_reach > frontier_x + 30.0:
                frontier_x = window_reach
                stall_count = 0
            else:
                stall_count += 1
            if (stall_count >= 6 and frontier_x > 1000.0
                    and frontier_x < env.end_x - 200.0):
                if bank_deepest() < frontier_x + 500.0:
                    n_banked, deep = explore_chain(48)
                    print(f"upd {update:4d} | EXPLORE: stall at "
                          f"{frontier_x:.0f}, banked {n_banked} states, deepest "
                          f"now {deep:.0f}")
                stall_count = 0

            t0 = time.perf_counter()
            if recorder.wins and len(recorder.wins) != wins_written:
                traces.write(run_dir / "wins.gdtrace", recorder.wins)
                wins_written = len(recorder.wins)
            if (update + 1) % args.trace_every == 0:
                recorder.dump_best(run_dir / "traces" / f"upd_{update + 1:05d}.gdtrace")

            # Best-so-far by window mean: a later collapse (catastrophic
            # forgetting) never costs the best policy — resume from ckpt_best.
            if n_eps and mean_x > best_mean_x:
                best_mean_x = mean_x
                save_ckpt(run_dir / "ckpt_best.pt", update + 1, progress, mean_x)
            # Best deterministic run gets its own checkpoint: it's the policy
            # open-loop deployment actually replays — saved from the weight
            # snapshot that produced the run, not the (newer) live model.
            for dx_, _, det_snap in det_results:
                if dx_ > best_det_x:
                    best_det_x = dx_
                    save_ckpt(run_dir / "ckpt_best_det.pt", det_snap["update"],
                              det_snap["progress"], det_snap["mean_x"],
                              model_state=det_snap["model"],
                              opt_state=det_snap["optimizer"],
                              logit_bound=det_snap["logit_bound"])
            if (update + 1) % args.ckpt_every == 0 or update + 1 == args.updates:
                save_ckpt(run_dir / "ckpt_latest.pt", update + 1, progress, mean_x)
            t_ckpt = time.perf_counter() - t0

            t_total = time.perf_counter() - t_upd0
            samples_s = cfg.rollout_steps * env.n_envs / max(t_total, 1e-9)
            last_sps, last_samples_s = sps, samples_s
            frac_explore = n_explore / (cfg.rollout_steps * env.n_envs)
            with open(log_path, "a") as f:
                f.write(row + f",{samples_s:.0f},{t_roll:.3f},{t_update:.3f},"
                              f"{t_sil:.3f},{t_det_wait:.3f},{t_ckpt:.3f},"
                              f"{t_roll_gpu:.3f},{t_roll_env:.3f},"
                              f"{t_roll_misc:.3f},{frac_explore:.3f},"
                              f"{stats.get('teacher_p', 0.0):.4f},"
                              f"{stats.get('teacher_disagree', 0.0):.4f},"
                              f"{front_x:.1f},{cert.x:.1f},{stitch_back_now:.0f},"
                              f"{dur_frac[0]:.3f},{dur_frac[1]:.3f},"
                              f"{dur_frac[2]:.3f},{dur_frac[3]:.3f},"
                              f"{solve_secs:.2f},{solve_nodes},{elapsed:.0f}\n")
            dur_str = (f" | dur 1:{dur_frac[0]:.2f} 2:{dur_frac[1]:.2f} "
                       f"4:{dur_frac[2]:.2f} h:{dur_frac[3]:.2f}" if precision else "")
            print(f"upd {update:4d} | x {mean_x:7.1f} avg {max_x:7.1f} max "
                  f"(best {best_x:7.1f}, {100 * best_x / env.end_x:5.1f}%) | "
                  f"win {win_rate:.2%} | {sps / 1e6:.2f}M ticks/s | "
                  f"{samples_s / 1e3:.1f}k dec/s | "
                  f"ent {stats['entropy']:.3f}{det_str}{die_str}"
                  f" | xpl {frac_explore:.2f}{teach_str}{dur_str}"
                  f"{f' | demos {len(sil_buf.demos)}' if sil_buf.demos else ''}"
                  f"{sol_str}{front_str}{win_str}"
                  f" | {t_roll:.2f}s roll ({t_roll_gpu:.2f} gpu {t_roll_env:.2f} "
                  f"env {t_roll_misc:.2f} misc) {t_update:.2f}s upd"
                  f"{f' {t_sil:.2f}s sil' if sil_env is not None else ''}"
                  f"{f' {t_det_wait:.2f}s detwait' if t_det_wait > 0.01 else ''}"
                  f" | {elapsed:.0f}s")

            # Auto-rollback: a sustained crash of from-start progress means the
            # policy forgot the prefix — restore the best policy instead of
            # grinding back up from scratch.
            if n_eps and best_mean_x > 2000.0 and mean_x < 0.4 * best_mean_x:
                collapse_count += 1
            else:
                collapse_count = 0
            best_path = run_dir / "ckpt_best.pt"
            if collapse_count >= 6 and best_path.exists():
                ckpt = torch.load(best_path, map_location=device,
                                  weights_only=False)
                model.load_state_dict(ckpt["model"])
                algo.optimizer.load_state_dict(ckpt["optimizer"])
                collapse_count = 0
                print(f"upd {update:4d} | COLLAPSE — rolled back to ckpt_best "
                      f"(mean_x {ckpt.get('mean_final_x', 0):.0f})")

            # Det-eval rollback: the argmax policy is what ships, and its
            # reach collapsing (level9-2: 12777 -> 2490 while mean_x, over
            # a handful of from-start episodes, never tripped the gate
            # above) for 3 consecutive evals means the prefix is gone —
            # restore the best det snapshot rather than relearn it.
            if (det_results and args.det_rollback > 0
                    and best_det_x >= 0.5 * env.end_x):
                if det_x_latest < args.det_rollback * best_det_x:
                    det_low_count += 1
                else:
                    det_low_count = 0
                if det_x_latest < 0.3 * best_det_x:
                    det_low_count = max(det_low_count, 2)
                best_det_path = run_dir / "ckpt_best_det.pt"
                if det_low_count >= 2 and best_det_path.exists():
                    ckpt = torch.load(best_det_path, map_location=device,
                                      weights_only=False)
                    model.load_state_dict(ckpt["model"])
                    algo.optimizer.load_state_dict(ckpt["optimizer"])
                    det_low_count = 0
                    det_x_latest = best_det_x
                    print(f"upd {update:4d} | DET COLLAPSE — rolled back to "
                          f"ckpt_best_det (det_x {best_det_x:.0f})")

            updates_this_run += 1
            if stop_path.exists():
                stopped = True
                stop_path.unlink(missing_ok=True)
                break
            if args.stop_after_updates and updates_this_run >= args.stop_after_updates:
                print(f"budget: {updates_this_run} updates this run — stopping")
                stopped = True
                break
            if args.stop_after_secs and elapsed >= args.stop_after_secs:
                print(f"budget: {elapsed:.0f}s wall-clock — stopping")
                stopped = True
                break
            if args.stop_on_win and first_win_secs >= 0:
                print(f"first verified det WIN at {first_win_secs:.0f}s — stopping")
                stopped = True
                break
    except KeyboardInterrupt:
        stopped = True

    if stopped:
        save_ckpt(run_dir / "ckpt_latest.pt", update + 1, progress, mean_x)
        print(f"stopped: at update {update + 1}; ckpt_latest saved "
              f"(resume with --resume {run_dir / 'ckpt_latest.pt'})")

    elapsed = elapsed_prev + (time.time() - t_wall0)
    print(f"done: {total_wins} wins / {total_episodes} episodes, "
          f"best x {best_x:.1f} ({100 * best_x / env.end_x:.1f}%), best det "
          f"{best_det_x:.1f}, certificate #{cert.id} to {cert.x:.1f}, "
          f"{elapsed:.0f}s wall-clock"
          f"{f', first verified WIN at {first_win_secs:.0f}s' if first_win_secs >= 0 else ''}"
          f"{f', demos verified {demo_verified}/{demo_verified + demo_unverified}' if demo_verified + demo_unverified else ''}")
    (run_dir / "summary.json").write_text(json.dumps({
        "updates": update + 1, "best_x": best_x, "best_det_x": best_det_x,
        "cert_x": cert.x, "cert_id": cert.id, "elapsed_secs": elapsed,
        "first_win_secs": first_win_secs, "total_wins": total_wins,
        "demos_verified": demo_verified, "demos_unverified": demo_unverified,
        "contracts": contracts}, indent=2))
    if live:
        live.close()
    det.poll(wait=True)
    det_env.close()
    probe_env.close()
    if sil_env is not None:
        sil_env.close()
    if solver is not None:
        solver.poll(wait=True)
        solver.env.close()
    scout.close()
    env.close()


if __name__ == "__main__":
    main()
