"""Roll out a trained policy and mine a winning input sequence.

Any single winning episode is a full certificate (the sim is deterministic):
its per-tick holds replay identically in the sim, and — if the replica is
exact — in the real game via the Geode open-loop mod.

  python -m gdrl.eval runs/full/ckpt_latest.pt --episodes 64

The env is built from the CHECKPOINT'S contracts (obs layout, action
space, level); a mismatch with the running DLL is refused. Writes
runs/<name>/solution.json (first/best win) and eval.gdtrace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import GdVecEnv, OBS_VERSION
from .model import (AMP_MODES, autocast_ctx, configure_backend, load_model_state,
                    make_model)
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


def level_id_of(level_path: str | None) -> int:
    """Numeric id of an official level file (Resources/levels/<id>.txt);
    None = GdVecEnv's default, Stereo Madness (1). -1 if not numeric."""
    if level_path is None:
        return 1
    stem = Path(level_path).stem
    return int(stem) if stem.isdigit() else -1


def env_kwargs_from_ckpt(ckpt: dict) -> dict:
    """obs_version / action_mode / reward_gamma the checkpoint trained under
    (legacy checkpoints: obs 5, macro, flat reward)."""
    c = ckpt.get("contracts") or {}
    cfgd = ckpt.get("config", {})
    obs_version = int(c.get("obs_version", cfgd.get("obs_version", OBS_VERSION)))
    action_mode = c.get("action_mode", cfgd.get("action_mode", "macro"))
    reward_gamma = float(c.get("reward_gamma",
                               cfgd.get("gamma", 0.0) if action_mode == "precision" else 0.0))
    return dict(obs_version=obs_version, action_mode=action_mode,
                reward_gamma=reward_gamma)


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
    p.add_argument("--from-start", action="store_true",
                   help="ignore a stitched checkpoint's certificate and "
                        "run the policy from x=0")
    p.add_argument("--amp", choices=AMP_MODES, default=None,
                   help="trunk precision; default = what the checkpoint's "
                        "run trained with")
    args = p.parse_args()

    configure_backend()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    run_dir = Path(args.checkpoint).resolve().parent
    amp = args.amp or ckpt.get("config", {}).get("amp", "none")

    if args.level is None:
        # The checkpoint records the level it was trained on; evaluating a
        # Polargeist policy on the default level (Stereo Madness) is never
        # what anyone means.
        args.level = ckpt.get("config", {}).get("level")
    trained_end = ckpt.get("config", {}).get("end_x_arg", -1.0)
    if trained_end is not None and trained_end > 0 and args.end_x < 0:
        print(f"WARNING: checkpoint was trained with --end-x {trained_end:g} "
              f"(truncated level); evaluating on the full level, so its "
              f"training 'det WIN's do not carry over")

    n = args.episodes + 1  # env 0 acts deterministically
    env_kw = env_kwargs_from_ckpt(ckpt)
    env = GdVecEnv(args.level, n, args.ticks_per_action, args.end_x, **env_kw)
    want = ckpt.get("contracts")
    if want is None:
        # Pre-contract checkpoint: the only recorded identity is the level
        # path; refuse a different level file.
        trained = ckpt.get("config", {}).get("level")
        if trained and args.level and Path(trained).name != Path(args.level).name:
            raise SystemExit(f"eval refused: legacy checkpoint trained on "
                             f"{trained}, not {args.level}")
    else:
        have = env.contracts()
        bad = [k for k in ("obs_version", "action_mode", "n_actions",
                           "level_fingerprint", "sim_fingerprint")
               if k in want and want[k] != have[k]]
        if bad:
            raise SystemExit(f"eval refused: checkpoint contracts differ from "
                             f"this DLL / level: {bad} "
                             f"({[(k, want[k], have[k]) for k in bad]})")
    print(f"contracts: obs v{env.obs_version}, {env.action_mode} actions "
          f"({env.n_actions}), level {env.contracts()['level_fingerprint']}")
    arch = ckpt.get("config", {}).get("arch", "mlp")
    model = make_model(env.obs_dim, arch, ckpt.get("logit_bound", 4.0),
                       n_actions=env.n_actions,
                       obs_version=env.obs_version).to(device)
    for note in load_model_state(model, ckpt["model"]):
        print(f"checkpoint obs migration: {note}")
    model.eval()

    env.reset()
    # Stitched checkpoints (train.py --stitch): the policy was trained from
    # the certificate frontier, so evaluate it from there and report full
    # from-x=0 holds (certificate prefix + the policy's own suffix).
    prefix = np.zeros(0, np.uint8)
    cfgd = ckpt.get("config", {})
    cert = ckpt.get("certificate")
    anchor = np.asarray(cert["holds"] if cert else ckpt.get("anchor_holds", []),
                        np.uint8)
    anchor_x = float(cert["x"] if cert else ckpt.get("anchor_x", 0.0))
    x0 = anchor_x - float(ckpt.get("stitch_back_now",
                                   cfgd.get("stitch_back", 600.0)))
    stitched = False
    if cfgd.get("stitch", 0) and len(anchor) and x0 > 0 \
            and not args.from_start:
        x_raw, won_raw, _ = env.run_holds(anchor)
        if x_raw < x0:
            print(f"WARNING: certificate does not raw-replay to its frontier "
                  f"(raw {x_raw:.0f} < {x0:.0f}); running from x=0")
        else:
            blob = env.trace_states(anchor, [x0])[0]
            if blob is not None:
                # Prefix length from a replay (same first-boundary rule as
                # trace_states), not the blob's byte layout.
                robs, _, _, rticks, rn, _, _ = env.replay([anchor], None)
                hit = np.nonzero(robs[0, :rn[0], 5] * env.end_x >= x0)[0]
                tick = int(rticks[0, :hit[0]].sum())
                prefix = anchor[:tick]
                for i in range(n):
                    env.set_state(i, blob)
                stitched = True
                print(f"stitched: starting from the certificate frontier "
                      f"x {x0:.0f} (prefix {tick} ticks, certificate to "
                      f"{anchor_x:.0f})")
    obs = torch.as_tensor(env._obs.copy(), device=device)
    episodes: list[traces.Episode | None] = [None] * n
    while any(e is None for e in episodes):
        mask = (torch.as_tensor(env.action_masks().astype(np.float32),
                                device=device) if model.categorical else None)
        with torch.no_grad(), autocast_ctx(device, amp):
            act = model.act(obs, mask=mask)[0]
            det_act = model.greedy(model(obs, mask=mask)[0])
        act[0] = det_act[0]
        actions_np = act.to(torch.uint8).cpu().numpy()
        next_obs, _, term, trunc, info = env.step(actions_np)
        done = (term | trunc).astype(bool)
        for i in np.nonzero(done)[0]:
            if episodes[i] is None:
                episodes[i] = traces.Episode(
                    np.concatenate([prefix, env.last_episode_holds(int(i))]),
                    float(info["final_x"][i]), bool(info["won"][i]))
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
        # Raw replay verification (deployment semantics) before writing.
        x_raw, won_raw, _ = env.run_holds(best.holds)
        if not won_raw:
            raise SystemExit(f"winning episode does not raw-replay to a WIN "
                             f"(raw x {x_raw:.0f}) — not writing a solution")
        sol = {
            "tps": traces.TPS,
            "level_id": level_id_of(args.level),
            "level_end_x": float(env.end_x),
            "n_ticks": int(len(best.holds)),
            "deterministic_policy": bool(det_ep.won),
            "stitched": stitched,
            "contracts": env.contracts(),
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
