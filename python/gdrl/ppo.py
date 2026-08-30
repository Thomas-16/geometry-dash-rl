"""Single-implementation PPO with GAE and correct truncation bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import ActorCritic


@dataclass
class PPOConfig:
    """Single source of hyperparameter defaults — train.py's CLI reads its
    argparse defaults from here, never the other way around."""

    rollout_steps: int = 64
    epochs: int = 4
    minibatches: int = 8
    # Discount is PER SIM TICK (steps have variable durations — a ground
    # jump is one decision spanning ~104 ticks); the buffer applies
    # gamma^ticks per step. 0.9995 = ~8.3 s reward horizon.
    gamma: float = 0.9995
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    # Entropy floor: near-zero entropy makes every trajectory identical in a
    # deterministic env, which starves PPO of signal and makes the policy
    # brittle to one bad update (observed: mean_x 16k -> 2.4k collapses).
    ent_coef_final: float = 0.002
    lr: float = 1e-3
    lr_final: float = 0.0
    max_grad_norm: float = 0.5
    # Stop the epoch loop when the update drifts this far from the rollout
    # policy — bounds the damage any single update can do.
    target_kl: float = 0.02


class RolloutBuffer:
    def __init__(self, steps: int, n_envs: int, obs_dim: int,
                 device: torch.device):
        self.steps, self.n_envs = steps, n_envs
        self.device = device
        s = (steps, n_envs)
        self.obs = torch.zeros(*s, obs_dim, device=device)
        self.actions = torch.zeros(*s, device=device)
        self.logprobs = torch.zeros(*s, device=device)
        self.rewards = torch.zeros(*s, device=device)
        self.values = torch.zeros(*s, device=device)
        # per-step discount gamma^ticks (variable-duration SMDP steps)
        self.discounts = torch.zeros(*s, device=device)
        # done = terminated (bootstrap 0); trunc_value = V(final_obs) where
        # truncated, folded into the reward before GAE.
        self.terminated = torch.zeros(*s, device=device)
        self.t = 0

    def add(self, obs, actions, logprobs, rewards, values, discounts,
            terminated):
        i = self.t
        self.obs[i] = obs
        self.actions[i] = actions
        self.logprobs[i] = logprobs
        self.rewards[i] = rewards
        self.values[i] = values
        self.discounts[i] = discounts
        self.terminated[i] = terminated
        self.t += 1

    def compute_advantages(self, last_value: torch.Tensor, lam: float):
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.steps)):
            nonterminal = 1.0 - self.terminated[t]
            next_value = last_value if t == self.steps - 1 else self.values[t + 1]
            # An episode boundary (terminated or truncated) means the next
            # row belongs to a new episode: both bootstrap and GAE carry-over
            # must stop. Truncation's V(final_obs) was already folded into
            # rewards[t] by the trainer, so nonterminal covers both here.
            g = self.discounts[t]
            delta = (self.rewards[t] + g * next_value * nonterminal
                     - self.values[t])
            gae = delta + g * lam * nonterminal * gae
            adv[t] = gae
        returns = adv + self.values
        return adv, returns

    def reset(self):
        self.t = 0


class PPO:
    def __init__(self, model: ActorCritic, cfg: PPOConfig,
                 device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                          eps=1e-5)

    def update(self, buf: RolloutBuffer, last_obs: torch.Tensor,
               progress: float, lr_override: float | None = None,
               ent_override: float | None = None) -> dict:
        """progress in [0,1] drives lr/entropy annealing; the overrides
        replace the annealed values for this call (polish phase)."""
        cfg = self.cfg
        lr = cfg.lr + (cfg.lr_final - cfg.lr) * progress
        ent_coef = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * progress
        if lr_override is not None:
            lr = lr_override
        if ent_override is not None:
            ent_coef = ent_override
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        with torch.no_grad():
            _, last_value = self.model(last_obs)
        adv, returns = buf.compute_advantages(last_value, cfg.gae_lambda)

        b_obs = buf.obs.reshape(-1, buf.obs.shape[-1])
        b_actions = buf.actions.reshape(-1)
        b_logprobs = buf.logprobs.reshape(-1)
        b_values = buf.values.reshape(-1)
        b_adv = adv.reshape(-1)
        b_returns = returns.reshape(-1)

        n = b_obs.shape[0]
        mb_size = n // cfg.minibatches
        idx = torch.arange(n, device=self.device)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        count = 0
        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            perm = idx[torch.randperm(n, device=self.device)]
            for mb in range(cfg.minibatches):
                m = perm[mb * mb_size:(mb + 1) * mb_size]
                logprob, entropy, value = self.model.evaluate(
                    b_obs[m], b_actions[m])
                ratio = (logprob - b_logprobs[m]).exp()
                madv = b_adv[m]
                # std floor 1e-2: when the policy plateaus at one obstacle,
                # trajectories converge and true advantages go to ~0 — a
                # 1e-8 eps would amplify float noise into large fake
                # gradients (the observed collapse trigger).
                madv = (madv - madv.mean()) / (madv.std() + 1e-2)
                pg1 = -madv * ratio
                pg2 = -madv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                policy_loss = torch.max(pg1, pg2).mean()
                v_clipped = b_values[m] + (value - b_values[m]).clamp(
                    -cfg.vf_clip, cfg.vf_clip)
                value_loss = 0.5 * torch.max(
                    (value - b_returns[m]) ** 2,
                    (v_clipped - b_returns[m]) ** 2).mean()
                ent = entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss - ent_coef * ent
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - ratio.log()).mean().item()
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy"] += ent.item()
                    stats["approx_kl"] += kl
                    stats["clip_frac"] += ((ratio - 1).abs()
                                           > cfg.clip).float().mean().item()
                count += 1
                if cfg.target_kl > 0 and kl > cfg.target_kl:
                    stop = True
                    break
        for k in stats:
            stats[k] /= count
        stats["lr"] = lr
        stats["ent_coef"] = ent_coef
        return stats
