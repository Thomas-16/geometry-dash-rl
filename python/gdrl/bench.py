"""Micro-benchmarks for the trainer's hot paths, so performance changes are
checked against numbers.

  python -m gdrl.bench                     # print timings
  python -m gdrl.bench --save base.json    # record a baseline
  python -m gdrl.bench --compare base.json # print deltas vs the baseline

Each figure is a wall-clock median over several repeats with the device
synchronized. Reference (RTX 4060 Ti, i7-12700KF, torch 2.13, bf16 default):
minibatch fwd+bwd ~16 ms (fp32 NCHW was 37), rollout step ~1.2 ms,
C replay of 8 x ~6k-tick episodes ~8 ms, GAE < 1 ms.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import torch

from .env import GdVecEnv
from .model import (AMP_MODES, GraphedAct, autocast_ctx, configure_backend,
                    make_model, maybe_compile)
from .ppo import PPO, PPOConfig, RolloutBuffer
from . import sil


def _timer(device: torch.device):
    def now():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()
    return now


def bench(fn, now, repeats: int = 5, inner: int = 1) -> float:
    """Median ms per call of fn (called `inner` times per repeat)."""
    fn()  # warm-up (cudnn autotune, allocator)
    ts = []
    for _ in range(repeats):
        t0 = now()
        for _ in range(inner):
            fn()
        ts.append((now() - t0) / inner * 1e3)
    return statistics.median(ts)


def run(args) -> dict[str, float]:
    configure_backend()
    device = torch.device(args.device)
    now = _timer(device)
    res: dict[str, float] = {}
    N = args.n_envs
    env = GdVecEnv(args.level, N, 4, -1.0, pinned=device.type == "cuda")
    if env.obs_tensor is None:
        env.obs_tensor = torch.from_numpy(env._obs)
    env.reset()
    rng = np.random.default_rng(0)
    acts = (rng.random(N) < 0.3).astype(np.uint8)

    res["env_step_ms"] = bench(lambda: env.step(acts), now, inner=16)
    gpu_obs = torch.empty(N, env.obs_dim, device=device)
    res["obs_h2d_ms"] = bench(
        lambda: gpu_obs.copy_(env.obs_tensor, non_blocking=True), now,
        inner=16)

    cfg = PPOConfig(amp=args.amp)
    model = maybe_compile(make_model(env.obs_dim, args.arch).to(device),
                          args.compile)
    alv_host = torch.empty(3, N, pin_memory=device.type == "cuda")

    def act_step():
        with autocast_ctx(device, cfg.amp):
            alv = model.act(gpu_obs)
        alv_host.copy_(alv)
        return alv_host[0].numpy().astype(np.uint8)
    res["act_ms"] = bench(act_step, now, inner=16)

    def rollout_step():
        gpu_obs.copy_(env.obs_tensor, non_blocking=True)
        a = act_step()
        env.step(a)
    res["rollout_step_ms"] = bench(rollout_step, now, inner=16)

    if device.type == "cuda" and not args.compile:
        gact = GraphedAct(model, N, env.obs_dim, device, cfg.amp)

        def graphed_rollout_step():
            gact.obs.copy_(env.obs_tensor, non_blocking=True)
            alv_host.copy_(gact())
            env.step(alv_host[0].numpy().astype(np.uint8))
        res["rollout_step_graphed_ms"] = bench(graphed_rollout_step, now,
                                               inner=16)

    # Minibatch fwd+bwd at the trainer's default minibatch size.
    mb = cfg.rollout_steps * N // cfg.minibatches
    b_obs = torch.rand(mb, env.obs_dim, device=device)
    b_act = (torch.rand(mb, device=device) < 0.5).float()

    def fwd_bwd():
        with autocast_ctx(device, cfg.amp):
            lp, ent, v = model.evaluate(b_obs, b_act)
        (lp.mean() + ent.mean() + v.mean()).backward()
    res[f"minibatch_fwd_bwd_{cfg.amp}_ms"] = bench(fwd_bwd, now, inner=4)
    if cfg.amp != "none":
        def fwd_bwd32():
            lp, ent, v = model.evaluate(b_obs, b_act)
            (lp.mean() + ent.mean() + v.mean()).backward()
        res["minibatch_fwd_bwd_none_ms"] = bench(fwd_bwd32, now, inner=4)

    # Full PPO update on a real rollout (target_kl disabled so every
    # minibatch runs — the trainer's early stop would hide the cost).
    cfg.target_kl = 0.0
    algo = PPO(model, cfg, device)
    buf = RolloutBuffer(cfg.rollout_steps, N, env.obs_dim, device)

    def fill():
        buf.reset()
        for t in range(cfg.rollout_steps):
            buf.obs[t].copy_(env.obs_tensor, non_blocking=True)
            with autocast_ctx(device, cfg.amp):
                alv = model.act(buf.obs[t])
            alv_host.copy_(alv)
            _, rew, term, trunc, info = env.step(
                alv_host[0].numpy().astype(np.uint8))
            buf.add(alv_host.numpy(), rew * 0.1, 0.9995 ** info["ticks"],
                    (term | trunc).astype(np.float32))
    fill()
    last_obs = torch.as_tensor(env.obs_tensor, device=device)
    lv = np.zeros(N, np.float32)
    res["gae_ms"] = bench(lambda: buf.compute_advantages(lv, 0.95), now,
                          inner=4)
    res["ppo_update_ms"] = bench(lambda: algo.update(buf, last_obs, 0.5),
                                 now, repeats=3)

    # SIL replay of k episodes generated by this (random) policy, plus the
    # SIL gradient steps.
    k = 8
    senv = GdVecEnv(args.level, k, 4, -1.0)
    senv.reset()
    got: list[np.ndarray | None] = [None] * k
    while any(g is None for g in got):
        _, _, term, trunc, _ = senv.step((rng.random(k) < 0.3).astype(np.uint8))
        for i in np.nonzero(term | trunc)[0]:
            if got[i] is None:
                got[i] = senv.last_episode_holds(int(i))
    eps = [g for g in got if g is not None]
    res["sil_replay_ms"] = bench(lambda: senv.replay(eps), now, inner=4)
    sbuf = sil.SILBuffer()
    for e in eps:
        sbuf.add(e, 1.0, False)
    silu = sil.SIL(model, algo.optimizer, device, 0.9995, 0.1, amp=cfg.amp)
    res["sil_update_ms"] = bench(lambda: silu.update(senv, sbuf, k, rng), now,
                                 repeats=3)

    # Deterministic eval: cost per decision on a 1-env handle.
    denv = GdVecEnv(args.level, 1, 4, -1.0)
    obs_d = torch.as_tensor(denv.reset(), device=device)

    def det_decision():
        with torch.no_grad(), autocast_ctx(device, cfg.amp):
            logit, _ = model(obs_d)
        a = (logit > 0).to(torch.uint8).cpu().numpy()
        denv.step(a)
    res["det_decision_ms"] = bench(det_decision, now, inner=32)

    # Accuracy spot check: closed-form Bernoulli vs torch.distributions.
    with torch.no_grad():
        logit, _ = model(b_obs[:1024])
        d = torch.distributions.Bernoulli(logits=logit)
        lp, ent, _ = model.evaluate(b_obs[:1024], b_act[:1024])
        res["bernoulli_logprob_maxdiff"] = float(
            (lp - d.log_prob(b_act[:1024])).abs().max())
        res["bernoulli_entropy_maxdiff"] = float((ent - d.entropy()).abs().max())
    for e in (env, senv, denv):
        e.close()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--arch", choices=("cnn", "mlp"), default="cnn")
    p.add_argument("--amp", choices=AMP_MODES, default=PPOConfig().amp)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the trunks (needs Triton)")
    p.add_argument("--level", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--save", default=None, help="write results JSON")
    p.add_argument("--compare", default=None, help="baseline JSON to diff")
    args = p.parse_args()
    res = run(args)
    base = json.load(open(args.compare)) if args.compare else {}
    for k, v in res.items():
        line = f"{k:34s} {v:10.3f}"
        if k in base and base[k] > 0:
            line += f"   (baseline {base[k]:10.3f}, x{base[k] / v:.2f})"
        print(line)
    if args.save:
        json.dump(res, open(args.save, "w"), indent=2)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
