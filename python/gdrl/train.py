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
import time
from pathlib import Path

import numpy as np
import torch

from .env import GdVecEnv
from .live import LiveWriter
from .model import ActorCritic
from .ppo import PPO, PPOConfig, RolloutBuffer
from . import traces


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run")
    p.add_argument("--level", default=None, help="level file path override")
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--ticks-per-action", type=int, default=4)
    p.add_argument("--end-x", type=float, default=-1.0,
                   help="early win line for curriculum runs; -1 = full level")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.9995,
                   help="discount per sim tick (steps are variable-length; "
                        "gamma^ticks is applied per decision)")
    p.add_argument("--reward-scale", type=float, default=0.1,
                   help="fixed reward multiplier; keeps returns O(1-10) so "
                        "the value-loss gradient doesn't drown the policy "
                        "gradient at the global grad clip")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
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
    return p.parse_args()


class EpisodeRecorder:
    """Collects finished episodes' per-tick holds (recorded by the env)."""

    def __init__(self, env: GdVecEnv):
        self.env = env
        self.completed: list[traces.Episode] = []
        self.wins: list[traces.Episode] = []

    def record(self, done: np.ndarray, info: dict):
        for i in np.nonzero(done)[0]:
            ep = traces.Episode(self.env.last_episode_holds(int(i)),
                                float(info["final_x"][i]),
                                bool(info["won"][i]))
            self.completed.append(ep)
            if ep.won:
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


def main():
    args = parse_args()
    device = torch.device(args.device)
    run_dir = Path(__file__).resolve().parents[2] / "runs" / args.run_name
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    env = GdVecEnv(args.level, args.n_envs, args.ticks_per_action, args.end_x)
    cfg = PPOConfig(rollout_steps=args.rollout, lr=args.lr,
                    ent_coef=args.ent_coef, gamma=args.gamma)
    model = ActorCritic(env.obs_dim)
    algo = PPO(model, cfg, device)
    buf = RolloutBuffer(cfg.rollout_steps, env.n_envs, env.obs_dim, device)
    recorder = EpisodeRecorder(env)

    start_update = 0
    best_x = 0.0
    best_mean_x = 0.0
    total_wins = 0
    total_episodes = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        algo.optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt["update"]
        best_x = ckpt.get("best_x", 0.0)
        best_mean_x = ckpt.get("best_mean_x", 0.0)
        total_wins = ckpt.get("total_wins", 0)
        total_episodes = ckpt.get("total_episodes", 0)
        print(f"resumed from {args.resume} at update {start_update}")

    (run_dir / "config.json").write_text(json.dumps(
        {**vars(args), "obs_dim": env.obs_dim, "end_x": env.end_x}, indent=2))
    # Fresh (non-resume) runs truncate: appending a restarted run to the old
    # log makes the curves unreadable.
    log_path = run_dir / "log.csv"
    if not args.resume or not log_path.exists():
        log_path.write_text(
            "update,global_ticks,mean_final_x,best_final_x,best_x_ever,"
            "win_rate,episodes,mean_ep_ticks,sps,policy_loss,value_loss,"
            "entropy,approx_kl,clip_frac,lr\n")
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run_dir / "tb"))
    except ImportError:
        tb = None

    live = LiveWriter(run_dir / "live.bin", env.n_envs) if args.watch else None

    obs_np = env.reset().copy()
    obs = torch.as_tensor(obs_np, device=device)
    global_ticks = 0
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
        """A random banked blob with x in [lo, hi], or None."""
        b0, b1 = int(max(lo, 0.0) // kBucket), int(max(hi, 0.0) // kBucket)
        cands = [t for b in range(b0, b1 + 1) for t in bank.get(b, ())
                 if lo <= t[0] <= hi]
        return cands[int(rng.integers(len(cands)))][1] if cands else None

    # Frontier-stall scout (Go-Explore style). Ship walls need a coordinated
    # multi-decision hold pattern: measured at the x=9026 wall, the policy's
    # own sampling passed 0/10k attempts while uniform-random actions from
    # ~300 units back passed ~2% — rare for on-policy exploration, trivial
    # for a brute-force burst. When the all-episode frontier stalls, run
    # random-action rollouts from banked pre-wall states and bank the states
    # BEYOND the wall, so respawn practice immediately trains the far side
    # and the crossing itself.
    scout = GdVecEnv(args.level, 128, args.ticks_per_action, args.end_x)
    frontier_x = 0.0
    stall_count = 0

    def scout_burst(frontier: float) -> tuple[int, int]:
        ns = scout.n_envs
        starts = [t for b in bank for t in bank[b]
                  if frontier - 450.0 <= t[0] <= frontier - 60.0]
        if not starts:
            return 0, 0
        banked = passes = 0
        last_bank = np.zeros(ns)
        passed = np.zeros(ns, bool)
        for _ in range(24):
            for i in range(ns):
                scout.set_state(i, starts[int(rng.integers(len(starts)))][1])
            alive = np.ones(ns, bool)
            last_bank[:] = frontier - 60.0
            passed[:] = False
            for _ in range(160):
                acts = rng.integers(0, 2, ns).astype(np.uint8)
                _, _, sterm, strunc, _ = scout.step(acts)
                alive &= ~(sterm | strunc).astype(bool)
                if not alive.any():
                    break
                rs = scout.get_render_states()
                for i in np.nonzero(alive)[0]:
                    x = float(rs[i, 0])
                    if x > frontier + 60.0 and not passed[i]:
                        passed[i] = True
                        passes += 1
                    if x > last_bank[i] + 120.0:
                        bank_add(x, scout.get_state(int(i)))
                        last_bank[i] = x
                        banked += 1
            if banked >= 48:
                break
        return banked, passes

    for update in range(start_update, args.updates):
        window_final_x = []
        window_wins = 0
        window_ep_ticks = []
        window_reach = 0.0  # max final_x over ALL episodes (respawned too)
        buf.reset()
        for step_i in range(cfg.rollout_steps):
            action, logprob, value = model.act(obs)
            actions_np = action.cpu().numpy().astype(np.uint8)
            next_obs, rew, term, trunc, info = env.step(actions_np)
            done = (term | trunc).astype(bool)

            rewards = torch.as_tensor(rew, device=device) * args.reward_scale
            discounts = torch.as_tensor(
                args.gamma ** info["ticks"], device=device,
                dtype=torch.float32)
            if trunc.any():
                mask = torch.as_tensor(trunc.astype(bool), device=device)
                fo = torch.as_tensor(
                    info["final_obs"][trunc.astype(bool)], device=device)
                with torch.no_grad():
                    _, v_final = model(fo)
                rewards[mask] += discounts[mask] * v_final
            buf.add(obs, action, logprob, rewards, value, discounts,
                    torch.as_tensor(done, dtype=torch.float32, device=device))

            # Progress stats and replayable traces only from x=0 episodes.
            done_fs = done & from_start
            recorder.record(done_fs, info)
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
            if step_i % 4 == 0:
                rs = env.get_render_states()
                for i in rng.choice(env.n_envs, 4, replace=False):
                    x = float(rs[i, 0])
                    if x > 400.0 and not done[i] and rs[i, 4] == 0.0:
                        bank_add(x, env.get_state(int(i)))
            for i in np.nonzero(done)[0]:
                if not info["won"][i]:
                    recent_deaths.append(float(info["final_x"][i]))
                if bank and rng.random() < args.respawn:
                    blob = None
                    if recent_deaths and rng.random() < 0.5:
                        # Wall practice: start a few hundred units before a
                        # recently observed death, i.e. the approach to a
                        # spot the policy currently fails.
                        dx = recent_deaths[int(rng.integers(
                            len(recent_deaths)))]
                        blob = bank_pick(dx - 450.0, dx - 50.0)
                    if blob is None:
                        buckets = sorted(bank)
                        blobs = bank[buckets[int(rng.integers(len(buckets)))]]
                        blob = blobs[int(rng.integers(len(blobs)))][1]
                    env.set_state(int(i), blob)
                    from_start[i] = False
                else:
                    from_start[i] = True

            step_ticks = int(info["ticks"].sum())
            global_ticks += step_ticks
            ticks_window += step_ticks
            obs = torch.as_tensor(next_obs, device=device).clone()

            if live and time.time() - last_live > 1.0 / 60.0:
                sps = ticks_window / max(time.time() - t_start, 1e-9)
                live.write(env.get_render_states(), global_ticks, best_x, sps,
                           total_episodes)
                last_live = time.time()

        progress = (update + 1 - start_update) / max(
            args.updates - start_update, 1)
        stats = algo.update(buf, obs, progress)

        sps = ticks_window / max(time.time() - t_start, 1e-9)
        t_start, ticks_window = time.time(), 0
        total_wins += window_wins
        n_eps = len(window_final_x)
        mean_x = float(np.mean(window_final_x)) if n_eps else 0.0
        max_x = float(np.max(window_final_x)) if n_eps else 0.0
        win_rate = window_wins / n_eps if n_eps else 0.0
        mean_ticks = float(np.mean(window_ep_ticks)) if n_eps else 0.0
        row = (f"{update},{global_ticks},{mean_x:.1f},{max_x:.1f},"
               f"{best_x:.1f},{win_rate:.4f},{n_eps},{mean_ticks:.0f},"
               f"{sps:.0f},{stats['policy_loss']:.4f},"
               f"{stats['value_loss']:.4f},{stats['entropy']:.4f},"
               f"{stats['approx_kl']:.5f},{stats['clip_frac']:.4f},"
               f"{stats['lr']:.6f}")
        with open(log_path, "a") as f:
            f.write(row + "\n")
        print(f"upd {update:4d} | x {mean_x:7.1f} avg {max_x:7.1f} max "
              f"(best {best_x:7.1f}, {100 * best_x / env.end_x:5.1f}%) | "
              f"win {win_rate:.2%} | {sps / 1e6:.2f}M ticks/s | "
              f"ent {stats['entropy']:.3f}")
        if tb:
            for k, v in {**stats, "mean_final_x": mean_x, "best_x": best_x,
                         "win_rate": win_rate, "sps": sps}.items():
                tb.add_scalar(k, v, global_ticks)

        # Frontier stall -> scout burst (see scout_burst above).
        if window_reach > frontier_x + 30.0:
            frontier_x = window_reach
            stall_count = 0
        else:
            stall_count += 1
        if stall_count >= 6 and frontier_x > 1000.0:
            n_banked, n_pass = scout_burst(frontier_x)
            if n_pass:
                print(f"upd {update:4d} | SCOUT: wall at {frontier_x:.0f} "
                      f"cracked — {n_pass} random-rollout passes, banked "
                      f"{n_banked} states beyond it")
            elif n_banked:
                print(f"upd {update:4d} | SCOUT: no pass at {frontier_x:.0f} "
                      f"yet, banked {n_banked} probe states")
            stall_count = 0

        if recorder.wins:
            traces.write(run_dir / "wins.gdtrace", recorder.wins)
        if (update + 1) % args.trace_every == 0:
            recorder.dump_best(run_dir / "traces" / f"upd_{update + 1:05d}.gdtrace")

        def save_ckpt(path):
            torch.save({"model": model.state_dict(),
                        "optimizer": algo.optimizer.state_dict(),
                        "update": update + 1, "best_x": best_x,
                        "best_mean_x": best_mean_x,
                        "total_wins": total_wins,
                        "total_episodes": total_episodes,
                        "mean_final_x": mean_x,
                        "config": vars(args)}, path)

        # Best-so-far by window mean: a later collapse (catastrophic
        # forgetting) never costs the best policy — resume from ckpt_best.
        if n_eps and mean_x > best_mean_x:
            best_mean_x = mean_x
            save_ckpt(run_dir / "ckpt_best.pt")
        if (update + 1) % args.ckpt_every == 0 or update + 1 == args.updates:
            save_ckpt(run_dir / "ckpt_latest.pt")

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

    print(f"done: {total_wins} wins / {total_episodes} episodes, "
          f"best x {best_x:.1f} ({100 * best_x / env.end_x:.1f}%)")
    if live:
        live.close()
    scout.close()
    env.close()


if __name__ == "__main__":
    main()
