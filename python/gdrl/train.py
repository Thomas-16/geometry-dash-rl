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
    return p.parse_args()


POLISH_LOGIT_BOUND = 8.0
POLISH_ENT_COEF = 1e-4
POLISH_MAX_LR = 1e-4
POLISH_RESPAWN = 0.25

LOG_COLUMNS = ("update,global_ticks,mean_final_x,best_final_x,best_x_ever,"
               "win_rate,episodes,mean_ep_ticks,sps,policy_loss,value_loss,"
               "entropy,approx_kl,clip_frac,lr,det_x,det_win,"
               "samples_s,t_rollout,t_update,t_sil,t_det_wait,t_ckpt,"
               "t_roll_gpu,t_roll_env,t_roll_misc\n")


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
        model.load_state_dict(ckpt["model"])
        algo.optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt["update"]
        best_x = ckpt.get("best_x", 0.0)
        best_mean_x = ckpt.get("best_mean_x", 0.0)
        best_det_x = ckpt.get("best_det_x", 0.0)
        total_wins = ckpt.get("total_wins", 0)
        total_episodes = ckpt.get("total_episodes", 0)
        polish = ckpt.get("polish", False)
        model.logit_bound = ckpt.get("logit_bound", model.logit_bound)
        print(f"resumed from {args.resume} at update {start_update} "
              f"(arch {args.arch}{', polish' if polish else ''})")
    model = maybe_compile(model, args.compile)

    (run_dir / "config.json").write_text(json.dumps(
        {**vars(args), "obs_dim": env.obs_dim, "end_x": env.end_x}, indent=2))
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

    sil_buf = sil.SILBuffer()
    silu = sil.SIL(model, algo.optimizer, device, args.gamma,
                   args.reward_scale, coef=args.sil_coef, amp=args.amp)
    # Replay handle sized to the replay batch: gdenv steps every env in a
    # handle, so borrowing the 128-env scout would waste 16x the sim work.
    sil_env = (GdVecEnv(args.level, args.sil_episodes, args.ticks_per_action,
                        args.end_x) if args.sil_coef > 0 else None)

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
        if bank:
            print(f"restored bank: {sum(len(v) for v in bank.values())} "
                  f"states, deepest {bank_deepest():.0f}; "
                  f"anneal progress {progress0:.2f}")
        if progress0 >= 0.999:
            print("WARNING: resumed run's lr/entropy anneal is complete "
                  "(lr ~= lr_final); pass --lr to keep learning")

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
            t_step0 = time.perf_counter()
            buf.reset()
            for step_i in range(cfg.rollout_steps):
                # The env's obs buffer (pinned) is the source of truth: H2D into
                # the graph's static input, act, then keep a copy in the rollout
                # slot (device->device, async and cheap).
                gact.obs.copy_(env.obs_tensor, non_blocking=True)
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
                if trunc.any():
                    mask = trunc.astype(bool)
                    fo = torch.as_tensor(info["final_obs"][mask], device=device)
                    with torch.no_grad(), autocast_ctx(device, cfg.amp):
                        v_final = model(fo)[1].float().cpu().numpy()
                    rewards = rewards.copy()
                    rewards[mask] += discounts[mask] * v_final
                buf.add(alv_np, rewards, discounts, done.astype(np.float32))

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
                                       float(info["final_x"][done].max()))
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
                    won_i = info["won"]
                    final_x_i = info["final_x"]
                for i in done_idx:
                    if respawn_x[i] >= 0.0:  # a respawned episode just ended
                        b = int(respawn_x[i] // kBucket)
                        respawn_attempts[b] = respawn_attempts.get(b, 0.0) + 1.0
                        if (won_i[i]
                                or final_x_i[i] >= respawn_x[i] + kBucket):
                            respawn_passes[b] = respawn_passes.get(b, 0.0) + 1.0
                    if not won_i[i]:
                        recent_deaths.append(float(final_x_i[i]))
                        window_deaths.append(float(final_x_i[i]))
                    if bank and rng.random() < respawn_prob:
                        picked = None
                        u = rng.random()
                        if u < 0.5 and learnable:
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
                        from_start[i] = True
                        respawn_x[i] = -1.0

                step_ticks = int(info["ticks"].sum())
                global_ticks += step_ticks
                ticks_window += step_ticks

                if live and time.time() - last_live > 1.0 / 60.0:
                    sps = ticks_window / max(time.time() - t_start, 1e-9)
                    live.write(env.get_render_states(), global_ticks, best_x, sps,
                               total_episodes)
                    last_live = time.time()
                t_step3 = time.perf_counter()
                t_roll_gpu += t_step1 - t_step0
                t_roll_env += t_step2 - t_step1
                t_roll_misc += t_step3 - t_step2
                t_step0 = t_step3

            last_obs = torch.as_tensor(env.obs_tensor, device=device)
            sync()
            t_roll = time.perf_counter() - t_upd0

            # Anneal position continues across resumes (progress0 from the ckpt).
            progress = progress0 + (1.0 - progress0) * (
                update + 1 - start_update) / max(args.updates - start_update, 1)
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
                stats.update(silu.update(sil_env, sil_buf, args.sil_episodes,
                                         rng, cfg.max_grad_norm))
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

            if not polish and not args.no_polish and (
                    (args.polish_at >= 0 and update + 1 >= args.polish_at)
                    or (args.polish_at < 0 and total_wins >= args.polish_wins)):
                polish = True
                model.logit_bound = POLISH_LOGIT_BOUND
                respawn_prob = min(respawn_prob, POLISH_RESPAWN)
                print(f"upd {update:4d} | POLISH: {total_wins} wins — logit "
                      f"bound {POLISH_LOGIT_BOUND}, ent {POLISH_ENT_COEF}, "
                      f"lr<={POLISH_MAX_LR}, respawn {respawn_prob}")

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
            with open(log_path, "a") as f:
                f.write(row + f",{samples_s:.0f},{t_roll:.3f},{t_update:.3f},"
                              f"{t_sil:.3f},{t_det_wait:.3f},{t_ckpt:.3f},"
                              f"{t_roll_gpu:.3f},{t_roll_env:.3f},"
                              f"{t_roll_misc:.3f}\n")
            print(f"upd {update:4d} | x {mean_x:7.1f} avg {max_x:7.1f} max "
                  f"(best {best_x:7.1f}, {100 * best_x / env.end_x:5.1f}%) | "
                  f"win {win_rate:.2%} | {sps / 1e6:.2f}M ticks/s | "
                  f"ent {stats['entropy']:.3f}{det_str}{die_str}"
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
    scout.close()
    env.close()


if __name__ == "__main__":
    main()
