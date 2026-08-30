"""Roll out a trained policy and mine a winning input sequence.

Any single winning episode is a full certificate (the sim is deterministic):
its per-tick holds replay identically in the sim, and — if the replica is
exact — in the real game via the future Geode open-loop mod.

  python -m gdrl.eval runs/full/ckpt_latest.pt --episodes 64

Writes runs/<name>/solution.json (first/best win) and eval.gdtrace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import GdVecEnv
from .model import AMP_MODES, autocast_ctx, configure_backend, make_model
from . import traces


def holds_to_ranges(holds: np.ndarray) -> list[list[int]]:
    """[start, end) tick ranges of held input — the Geode replay format."""
    ranges = []
    padded = np.diff(np.concatenate([[0], holds.astype(np.int8), [0]]))
    starts = np.nonzero(padded == 1)[0]
    ends = np.nonzero(padded == -1)[0]
    for s, e in zip(starts, ends):
        ranges.append([int(s), int(e)])
    return ranges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=64,
                   help="stochastic episodes (plus 1 deterministic)")
    p.add_argument("--level", default=None)
    p.add_argument("--end-x", type=float, default=-1.0)
    p.add_argument("--ticks-per-action", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--amp", choices=AMP_MODES, default=None,
                   help="trunk precision; default = what the checkpoint's "
                        "run trained with")
    args = p.parse_args()

    configure_backend()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    run_dir = Path(args.checkpoint).resolve().parent
    amp = args.amp or ckpt.get("config", {}).get("amp", "none")

    n = args.episodes + 1  # env 0 acts deterministically
    env = GdVecEnv(args.level, n, args.ticks_per_action, args.end_x)
    arch = ckpt.get("config", {}).get("arch", "mlp")
    model = make_model(env.obs_dim, arch,
                       ckpt.get("logit_bound", 4.0)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    obs = torch.as_tensor(env.reset(), device=device)
    episodes: list[traces.Episode | None] = [None] * n
    while any(e is None for e in episodes):
        with torch.no_grad(), autocast_ctx(device, amp):
            logit, _ = model(obs)
        act = (torch.rand_like(logit) < torch.sigmoid(logit))
        act[0] = logit[0] > 0
        actions_np = act.to(torch.uint8).cpu().numpy()
        next_obs, _, term, trunc, info = env.step(actions_np)
        done = (term | trunc).astype(bool)
        for i in np.nonzero(done)[0]:
            if episodes[i] is None:
                episodes[i] = traces.Episode(env.last_episode_holds(int(i)),
                                             float(info["final_x"][i]),
                                             bool(info["won"][i]))
        obs = torch.as_tensor(next_obs, device=device)

    eps: list[traces.Episode] = episodes  # type: ignore
    det_ep, stoch_eps = eps[0], eps[1:]
    wins = [e for e in eps if e.won]
    print(f"deterministic: {'WIN' if det_ep.won else 'death'} at "
          f"x={det_ep.final_x:.1f} ({100 * det_ep.final_x / env.end_x:.1f}%)")
    if stoch_eps:
        xs = [e.final_x for e in stoch_eps]
        print(f"stochastic ({len(stoch_eps)}): win rate "
              f"{sum(e.won for e in stoch_eps) / len(stoch_eps):.2%}, "
              f"mean x {np.mean(xs):.1f}, best {np.max(xs):.1f}")

    traces.write(run_dir / "eval.gdtrace",
                 sorted(eps, key=lambda e: -e.final_x)[:16])
    print(f"wrote {run_dir / 'eval.gdtrace'}")

    if wins:
        best = det_ep if det_ep.won else wins[0]
        sol = {
            "tps": traces.TPS,
            "level_end_x": float(env.end_x),
            "n_ticks": int(len(best.holds)),
            "deterministic_policy": bool(det_ep.won),
            "holds": holds_to_ranges(best.holds),
        }
        (run_dir / "solution.json").write_text(json.dumps(sol, indent=2))
        print(f"WROTE SOLUTION: {run_dir / 'solution.json'} "
              f"({len(sol['holds'])} hold ranges, {sol['n_ticks']} ticks)")
    else:
        print("no winning episode found")
    env.close()


if __name__ == "__main__":
    main()
