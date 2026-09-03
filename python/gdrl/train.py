"""PPO trainer for the GD replica.

Headless by default. `--watch` also writes runs/<name>/live.bin, which the
viewer's training view renders live:  viewer --watch runs/<name>/live.bin

Usage (from repo root, inside the venv):
  python -m gdrl.train --run-name full --n-envs 256
  python -m gdrl.train --run-name short --end-x 2500 --updates 60
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .env import GdVecEnv
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
    # sub-second macro-action BFS (env/solve.hpp). After every det eval the
    # solver starts from the det run's OWN states --solve-back..50 units
    # before its death and searches --solve-reach past it; the segments it
    # finds go to SIL as demonstrations (fixed minibatch share) and their
    # states into the respawn bank. Pure exploration stays on: this only
    # fires where the argmax policy dies.
    p.add_argument("--wall-solve", type=int, default=1,
                   help="1 = solve wall demos after each det eval, 0 = off")
    p.add_argument("--solve-back", type=float, default=700.0,
                   help="planner starts span [det_x - back, det_x - 50] "
                        "along the det run (doubles on total failure, "
                        "up to 8x)")
    p.add_argument("--solve-reach", type=float, default=800.0,
                   help="planner target: det_x + reach")
    p.add_argument("--solve-starts", type=int, default=7)
    p.add_argument("--solve-stall", type=int, default=3,
                   help="det evals without a new best before the planner "
                        "fires (0 = after every eval). PPO clears most "
                        "walls itself; demos everywhere slowed Stereo "
                        "Madness (det WIN 179 vs 109-119 updates)")
    p.add_argument("--solve-nodes", type=int, default=2000000,
                   help="node budget per planner start (a ship section at "
                        "4-tick decisions needs ~1.3M per 800 units)")
    p.add_argument("--sil-demos", type=int, default=4,
                   help="demo segments replayed per SIL update")
    p.add_argument("--anchor-coef", type=float, default=0.1,
                   help="behavior-cloning weight pinning the mastered "
                        "zone (x < best det_x - margin) to the best det "
                        "run's own decisions (0 = off): stops the trunk "
                        "drift that broke level9-2's prefix repeatedly")
    p.add_argument("--anchor-margin", type=float, default=2.0,
                   help="hinge margin (logits) the anchor holds on the "
                        "best det run's decisions")
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
               "t_roll_gpu,t_roll_env,t_roll_misc,frac_explore\n")


class EpisodeRecorder:
    """Collects finished episodes' per-tick holds (recorded by the env)."""

    def __init__(self, env: GdVecEnv):
        self.env = env
        self.completed: list[traces.Episode] = []
        self.wins: list[traces.Episode] = []

    kMaxWins = 16  # wins are interchangeable certificates; cap the trace

    def record(self, done: np.ndarray, info: dict):
        for i in np.nonzero(done)[0]:
            ep = traces.Episode(self.env.last_episode_holds(int(i)),
                                float(info["final_x"][i]),
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
                                   model.logit_bound).to(device)
        self.thread: threading.Thread | None = None
        self.result: tuple[float, bool, dict] | None = None

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, model, optimizer, update: int, extra: dict) -> None:
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
        self.thread = threading.Thread(target=self._run, args=(info,),
                                       daemon=True)
        self.thread.start()

    def _run(self, info: dict) -> None:
        env, dev, model = self.env, self.device, self.snapshot
        obs_d = torch.as_tensor(env.reset(), device=dev)
        while True:
            with torch.no_grad(), autocast_ctx(dev, self.amp):
                logit, _ = model(obs_d)
            a = (logit > 0).to(torch.uint8).cpu().numpy()
            next_d, _, dterm, dtrunc, dinfo = env.step(a)
            if dterm[0] or dtrunc[0]:
                info["holds"] = env.last_episode_holds(0)
                self.result = (float(dinfo["final_x"][0]),
                               bool(dinfo["won"][0]), info)
                return
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
    """Background macro-action BFS (gdenv_solve) from states along a known
    from-start hold sequence — the det run — toward a target x. Own env
    handle (the DLL releases the GIL, so the search overlaps the rollout).
    Result: list of (start blob, holds, start_x, reach_x) demo segments plus
    (x, blob) snapshots every ~120 units along each, for the respawn bank."""

    def __init__(self, env: GdVecEnv, max_nodes: int):
        self.env = env
        self.max_nodes = max_nodes
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
               "target_x": target_x, "reach": 0.0, "secs": 0.0}
        if not starts:
            self.result = out
            return
        sols, reach = env.solve([b for _, b in starts], target_x,
                                self.max_nodes)
        out["reach"] = float(reach.max()) if len(reach) else 0.0
        ok = [(x, b, h) for (x, b), h in zip(starts, sols) if h is not None]
        if ok:
            # Snapshots along each demo: walk them in lockstep on this
            # handle (set_state + the replayed macro actions).
            _, acts, _, _, n_steps, _, _ = env.replay(
                [h for _, _, h in ok], [b for _, b, _ in ok])
            for i, (x, b, _) in enumerate(ok):
                env.set_state(i, b)
            last = np.array([x for x, _, _ in ok])
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
            out["demos"] = [(b, h, x, target_x) for x, b, h in ok]
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


def main():
    args = parse_args()
    configure_backend()
    device = torch.device(args.device)
    run_dir = Path(__file__).resolve().parents[2] / "runs" / args.run_name
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    env = GdVecEnv(args.level, args.n_envs, args.ticks_per_action, args.end_x,
                   pinned=device.type == "cuda")
    cfg = PPOConfig(rollout_steps=args.rollout, lr=args.lr,
                    ent_coef=args.ent_coef, gamma=args.gamma, amp=args.amp)
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        args.arch = ckpt.get("config", {}).get("arch", "mlp")
    model = make_model(env.obs_dim, args.arch)
    algo = PPO(model, cfg, device)
    buf = RolloutBuffer(cfg.rollout_steps, env.n_envs, env.obs_dim, device)
    recorder = EpisodeRecorder(env)

    start_update = 0
    best_x = 0.0
    best_mean_x = 0.0
    best_det_x = 0.0
    total_wins = 0
    total_episodes = 0
    polish = False
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
              f"(arch {args.arch}{', polish' if polish else ''})")
    model = maybe_compile(model, args.compile)

    (run_dir / "config.json").write_text(json.dumps(
        {**vars(args), "obs_dim": env.obs_dim, "end_x": env.end_x,
         # end_x above is the *effective* end (a full-level run records the
         # level's natural end); end_x_arg is what was asked for, so a
         # resume/prefill of a full-level run never inherits a hard cutoff.
         "end_x_arg": args.end_x}, indent=2))
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
    rng = np.random.default_rng(0)
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
    scout = GdVecEnv(args.level, 128, args.ticks_per_action, args.end_x)
    frontier_x = 0.0
    stall_count = 0
    respawn_prob = args.respawn

    # Deterministic full-run eval on a dedicated 1-env handle, off-thread.
    # The argmax policy is what open-loop deployment replays, so it gets its
    # own gate and its own best checkpoint (stochastic mean_x drives
    # ckpt_best).
    det_env = GdVecEnv(args.level, 1, args.ticks_per_action, args.end_x)
    det = DetEvaluator(det_env, model, device, args.amp)

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
                acts = (rng.random(ns) < 0.5).astype(np.uint8)
                scout.step(acts)
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
                acts = (rng.random(ns) < p_hold).astype(np.uint8)
                _, _, sterm, strunc, _ = scout.step(acts)
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
                   anchor_margin=args.anchor_margin)
    # Replay handle sized to the replay batch (episodes + demo segments +
    # the anchor run): gdenv steps every env in a handle, so borrowing the
    # 128-env scout would waste 16x the sim work.
    sil_env = (GdVecEnv(args.level, args.sil_episodes + args.sil_demos + 1,
                        args.ticks_per_action, args.end_x)
               if args.sil_coef > 0 else None)
    # Planner (see --wall-solve) on its own handle; k envs = k parallel
    # solves and the lockstep walk that snapshots the found segments.
    kSolveExtra = 3  # banked starts added to the det-path ones
    solver = (WallSolver(GdVecEnv(args.level,
                                  max(1, args.solve_starts) + kSolveExtra,
                                  args.ticks_per_action, args.end_x),
                         args.solve_nodes)
              if args.wall_solve else None)
    solve_back = args.solve_back
    # Anchor: the best det run's holds and its reach (sil.py).
    anchor_holds = np.zeros(0, np.uint8)
    anchor_x = 0.0
    det_low_count = 0  # consecutive det evals below the rollback fraction
    det_stall = 0      # consecutive det evals without a new best

    # Restore trainer state that isn't the network: exploration coverage,
    # respawn stats, SIL episodes, rng, and the anneal position (without it
    # a resume restarts lr/entropy annealing from the top).
    progress0 = 0.0
    if ckpt is not None:
        progress0 = ckpt.get("anneal_progress", 0.0)
        frontier_x = ckpt.get("frontier_x", 0.0)
        for bx, blob in ckpt.get("bank", []):
            bank_add(bx, blob)
        recent_deaths.extend(ckpt.get("recent_deaths", []))
        if "rng_state" in ckpt:
            rng.bit_generator.state = ckpt["rng_state"]
        if "sil" in ckpt:
            sil_buf.load_state_dict(ckpt["sil"])
        respawn_attempts = ckpt.get("respawn_attempts", respawn_attempts)
        respawn_passes = ckpt.get("respawn_passes", respawn_passes)
        det_x_latest = ckpt.get("det_x_latest", 0.0)
        anchor_holds = np.asarray(ckpt.get("anchor_holds", anchor_holds),
                                  np.uint8)
        anchor_x = ckpt.get("anchor_x", anchor_x)
        solve_back = ckpt.get("solve_back", solve_back)
        if bank:
            print(f"restored bank: {sum(len(v) for v in bank.values())} "
                  f"states, deepest {bank_deepest():.0f}; "
                  f"anneal progress {progress0:.2f}")
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

    # Startup coverage: run the chained explorer once so the respawn bank
    # spans everything randomly reachable before the first update. Pure
    # exploration — the bank never holds a state the agent didn't reach.
    # (With a restored full-coverage bank it exits immediately.)
    n_banked, deep0 = explore_chain(600)
    print(f"explore: banked {n_banked} states, deepest x {deep0:.0f} "
          f"({100 * deep0 / env.end_x:.1f}% of level)")

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
                    "anchor_holds": anchor_holds,
                    "anchor_x": anchor_x,
                    "solve_back": solve_back,
                    "sil": sil_buf.state_dict(),
                    "config": vars(args)}, path)

    stopped = False
    update = start_update
    progress = 0.0
    mean_x = 0.0
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
            t_step0 = time.perf_counter()
            buf.reset()
            # Mastered zone for this update: x < best det reach - margin.
            # The BEST run, not the latest: one bad det eval (a single
            # flipped prefix decision) must not turn the whole level into
            # the explore zone — measured: xpl 0.00 -> 0.76, prefix deaths
            # x65/update, two collapse rollbacks. The anchor repairs the
            # flip; the det rollback below is the backstop.
            explore_lo = (max(det_x_latest, anchor_x) - args.explore_margin
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
                alv = gact()
                buf.obs[step_i].copy_(gact.obs, non_blocking=True)
                alv_host.copy_(alv)  # one D2H sync: action, logprob, value
                alv_np = alv_host.numpy()
                actions_np = alv_np[0].astype(np.uint8)
                t_step1 = time.perf_counter()
                next_obs, rew, term, trunc, info = env.step(actions_np)
                t_step2 = time.perf_counter()
                done = (term | trunc).astype(bool)

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
                # final obs.
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
                        bound_np, ent_w_np)

                # Progress stats and replayable traces only from x=0 episodes.
                done_fs = done & from_start
                n_before = len(recorder.completed)
                recorder.record(done_fs, info)
                for ep in recorder.completed[n_before:]:
                    sil_buf.add(ep.holds, ep.final_x, ep.won)
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
                    learnable = [
                        b for b in bank
                        if respawn_attempts.get(b, 0.0) >= 8.0
                        and 0.1 <= (respawn_passes.get(b, 0.0)
                                    / respawn_attempts[b]) <= 0.8]
                    buckets = sorted(bank)
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
                    if bank and rng.random() < respawn_prob:
                        picked = None
                        u = rng.random()
                        if (det_x_latest > 700.0
                                and rng.random() < args.frontier_respawn):
                            # Frontier: just before where the argmax policy
                            # dies (see --frontier-respawn).
                            picked = bank_pick(det_x_latest - 700.0,
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
                            picked = bank_pick(dx - 450.0, dx - 50.0)
                        if picked is None:
                            # Uniform over the bank: retention practice.
                            blobs = bank[buckets[int(rng.integers(len(buckets)))]]
                            picked = blobs[int(rng.integers(len(blobs)))]
                        env.set_state(int(i), picked[1])
                        from_start[i] = False
                        respawn_x[i] = picked[0]
                    else:
                        if burst_i[i]:  # alive in the DLL: reset it ourselves
                            env.reset_env(int(i))
                        from_start[i] = True
                        respawn_x[i] = -1.0

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
            if polish:
                lr_now = min(cfg.lr + (cfg.lr_final - cfg.lr) * progress,
                             POLISH_MAX_LR)
                stats = algo.update(buf, last_obs, progress, lr_override=lr_now,
                                    ent_override=POLISH_ENT_COEF)
            else:
                stats = algo.update(buf, last_obs, progress)
            sync()
            t_update = time.perf_counter() - t0
            t0 = time.perf_counter()
            if sil_env is not None:
                stats.update(silu.update(
                    sil_env, sil_buf, args.sil_episodes, rng,
                    cfg.max_grad_norm, k_demos=args.sil_demos,
                    anchor=(anchor_holds, anchor_x),
                    anchor_x_lo=anchor_x - args.explore_margin))
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
            # new one is due or this is the last update), then launch the next.
            det_x, det_win = -1.0, False
            det_due = args.det_every > 0 and (
                (update + 1) % args.det_every == 0 or update + 1 == args.updates)
            t0 = time.perf_counter()
            det_results = [r for r in (det.poll(wait=det_due),) if r is not None]
            if det_due:
                det.start(model, algo.optimizer, update + 1,
                          {"progress": progress, "mean_x": mean_x})
                if update + 1 == args.updates:  # nothing left to overlap with
                    det_results.append(det.poll(wait=True))
            t_det_wait = time.perf_counter() - t0
            if det_results:
                det_x, det_win, _ = det_results[-1]
                det_x_latest = det_x
            for dx_, dwin_, dsnap in det_results:
                if dx_ > anchor_x and "holds" in dsnap:
                    anchor_holds = np.asarray(dsnap["holds"], np.uint8)
                    anchor_x = dx_
                det_stall = 0 if dx_ > best_det_x + 30.0 else det_stall + 1
                # Planner: demos through the wall the argmax policy just
                # died at, from its own states (plus a few banked ones for
                # diversity) — once the det reach has stalled. Skipped
                # while a solve is still running.
                if (solver is not None and not dwin_ and dx_ > 300.0
                        and det_stall >= args.solve_stall
                        and "holds" in dsnap and not solver.busy()):
                    k = max(1, args.solve_starts)
                    xs = list(np.linspace(max(dx_ - solve_back, 50.0),
                                          max(dx_ - 50.0, 60.0), k))
                    extra = []
                    for _ in range(kSolveExtra):
                        pk = bank_pick(dx_ - solve_back, dx_ - 50.0)
                        if pk is not None:
                            extra.append(pk)
                    solver.start(dsnap["holds"], xs,
                                 min(dx_ + args.solve_reach, env.end_x),
                                 extra)
            sol_str = ""
            sol_res = solver.poll() if solver is not None else None
            if sol_res is not None:
                n_new = 0
                for b, h, sx, tx in sol_res["demos"]:
                    n_new += int(sil_buf.add_demo(b, h, sx, tx))
                for bx, blob in sol_res["bank"]:
                    bank_add(bx, blob)
                if sol_res["demos"]:
                    solve_back = args.solve_back
                else:
                    solve_back = min(solve_back * 2.0, args.solve_back * 8.0)
                x0s = [sx for _, _, sx, _ in sol_res["demos"]]
                sol_str = (f" | WALL {len(sol_res['demos'])}/"
                           f"{sol_res['n_starts']} demos"
                           f"{f' from {min(x0s):.0f}-{max(x0s):.0f}' if x0s else ''}"
                           f" -> {sol_res['target_x']:.0f} (reach "
                           f"{sol_res['reach']:.0f}, {n_new} new, "
                           f"{len(sol_res['bank'])} banked, "
                           f"{sol_res['secs']:.2f}s"
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
            if tb:
                for k, v in {**stats, "mean_final_x": mean_x, "best_x": best_x,
                             "win_rate": win_rate, "sps": sps,
                             "t_rollout": t_roll, "t_update": t_update,
                             "t_sil": t_sil, "t_det_wait": t_det_wait,
                             "t_roll_gpu": t_roll_gpu, "t_roll_env": t_roll_env,
                             "t_roll_misc": t_roll_misc}.items():
                    tb.add_scalar(k, v, global_ticks)
                if det_x >= 0:
                    tb.add_scalar("det_x", det_x, global_ticks)
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
                              f"{t_roll_misc:.3f},{frac_explore:.3f}\n")
            print(f"upd {update:4d} | x {mean_x:7.1f} avg {max_x:7.1f} max "
                  f"(best {best_x:7.1f}, {100 * best_x / env.end_x:5.1f}%) | "
                  f"win {win_rate:.2%} | {sps / 1e6:.2f}M ticks/s | "
                  f"{samples_s / 1e3:.1f}k dec/s | "
                  f"ent {stats['entropy']:.3f}{det_str}{die_str}"
                  f" | xpl {frac_explore:.2f}"
                  f"{f' | demos {len(sil_buf.demos)}' if sil_buf.demos else ''}"
                  f"{sol_str}"
                  f" | {t_roll:.2f}s roll ({t_roll_gpu:.2f} gpu {t_roll_env:.2f} "
                  f"env {t_roll_misc:.2f} misc) {t_update:.2f}s upd"
                  f"{f' {t_sil:.2f}s sil' if sil_env is not None else ''}"
                  f"{f' {t_det_wait:.2f}s detwait' if t_det_wait > 0.01 else ''}")

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

            if stop_path.exists():
                stopped = True
                stop_path.unlink(missing_ok=True)
                break
    except KeyboardInterrupt:
        stopped = True

    if stopped:
        save_ckpt(run_dir / "ckpt_latest.pt", update + 1, progress, mean_x)
        print(f"stopped: at update {update + 1}; ckpt_latest saved "
              f"(resume with --resume {run_dir / 'ckpt_latest.pt'})")

    print(f"done: {total_wins} wins / {total_episodes} episodes, "
          f"best x {best_x:.1f} ({100 * best_x / env.end_x:.1f}%)")
    if live:
        live.close()
    det.poll(wait=True)
    det_env.close()
    if sil_env is not None:
        sil_env.close()
    if solver is not None:
        solver.poll(wait=True)
        solver.env.close()
    scout.close()
    env.close()


if __name__ == "__main__":
    main()
