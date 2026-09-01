"""Single-implementation PPO with GAE and correct truncation bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import ActorCritic, autocast_ctx


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
    # Trunk precision: "bf16" autocast (2.3x faster update on Ada, heads and
    # losses stay fp32 — see model.py) or "none" for the pure fp32 path.
    amp: str = "bf16"


class RolloutBuffer:
    """Observations live on the device (the only large field, written
    straight from the env's pinned host buffer); every per-step scalar field
    is host-side numpy, filled from one stacked device->host copy per step,
    so GAE is a cheap numpy scan and the update uploads its inputs once."""

    def __init__(self, steps: int, n_envs: int, obs_dim: int,
                 device: torch.device):
        self.steps, self.n_envs = steps, n_envs
        self.device = device
        s = (steps, n_envs)
        self.obs = torch.zeros(*s, obs_dim, device=device)
        self.actions = np.zeros(s, np.float32)
        self.logprobs = np.zeros(s, np.float32)
        self.rewards = np.zeros(s, np.float32)
        self.values = np.zeros(s, np.float32)
        # per-step discount gamma^ticks (variable-duration SMDP steps)
        self.discounts = np.zeros(s, np.float32)
        # done = terminated (bootstrap 0); trunc_value = V(final_obs) where
        # truncated, folded into the reward before GAE.
        self.terminated = np.zeros(s, np.float32)
        # Per-sample logit bound the action was sampled under (model.py):
        # the update must re-bound the logit identically or the PPO ratio
        # compares two different policies.
        self.bounds = np.full(s, 4.0, np.float32)
        # Per-sample entropy-bonus weight (1 = full ent_coef). In mastered
        # territory the policy gradient is ~0 (identical trajectories), so
        # an unweighted bonus slowly erodes confidence there until deaths
        # reappear; the trainer scales it down outside the explore zone.
        self.ent_w = np.ones(s, np.float32)
        self.t = 0

    def add(self, alv: np.ndarray, rewards: np.ndarray, discounts: np.ndarray,
            terminated: np.ndarray, bounds: np.ndarray | None = None,
            ent_w: np.ndarray | None = None):
        """alv: [3, n_envs] = action, logprob, value (model.act output).
        The obs row for this step was already copied into self.obs[t].
        bounds: per-env logit bound used for this step (None keeps the
        buffer's default 4); ent_w: per-env entropy weight (None = 1)."""
        i = self.t
        self.actions[i] = alv[0]
        self.logprobs[i] = alv[1]
        self.values[i] = alv[2]
        self.rewards[i] = rewards
        self.discounts[i] = discounts
        self.terminated[i] = terminated
        if bounds is not None:
            self.bounds[i] = bounds
        if ent_w is not None:
            self.ent_w[i] = ent_w
        self.t += 1

    def compute_advantages(self, last_value: np.ndarray, lam: float):
        adv = np.zeros_like(self.rewards)
        gae = np.zeros(self.n_envs, np.float32)
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
                                          eps=1e-5,
                                          fused=device.type == "cuda")
        self._mb_obs: torch.Tensor | None = None  # reused gather target

    def update(self, buf: RolloutBuffer, last_obs: torch.Tensor,
               progress: float, lr_override: float | None = None,
               ent_override: float | None = None) -> dict:
        """progress in [0,1] drives lr/entropy annealing; the overrides
        replace the annealed values for this call (polish phase)."""
        cfg = self.cfg
        dev = self.device
        lr = cfg.lr + (cfg.lr_final - cfg.lr) * progress
        ent_coef = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * progress
        if lr_override is not None:
            lr = lr_override
        if ent_override is not None:
            ent_coef = ent_override
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        with torch.no_grad(), autocast_ctx(dev, cfg.amp):
            last_value = self.model(last_obs)[1].float().cpu().numpy()
        adv, returns = buf.compute_advantages(last_value, cfg.gae_lambda)

        b_obs = buf.obs.reshape(-1, buf.obs.shape[-1])
        n = b_obs.shape[0]
        # One upload of every host-side field, then everything stays on the
        # device: [7, n] = actions, old logprobs, old values, adv, returns,
        # logit bounds, entropy weights.
        host = np.stack([buf.actions.reshape(-1), buf.logprobs.reshape(-1),
                         buf.values.reshape(-1), adv.reshape(-1),
                         returns.reshape(-1), buf.bounds.reshape(-1),
                         buf.ent_w.reshape(-1)])
        (b_actions, b_logprobs, b_values, b_adv, b_returns,
         b_bounds, b_ent_w) = torch.as_tensor(host, device=dev).unbind(0)

        mb_size = n // cfg.minibatches
        if self._mb_obs is None or self._mb_obs.shape[0] != mb_size:
            self._mb_obs = torch.empty(mb_size, b_obs.shape[1], device=dev)
        mb_obs = self._mb_obs
        # Stats accumulate on the device; the KL early-stop check is the
        # only per-minibatch host sync.
        stat_sum = torch.zeros(5, device=dev)
        count = 0
        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            perm = torch.randperm(n, device=dev)
            for mb in range(cfg.minibatches):
                m = perm[mb * mb_size:(mb + 1) * mb_size]
                torch.index_select(b_obs, 0, m, out=mb_obs)
                with autocast_ctx(dev, cfg.amp):
                    logprob, entropy, value = self.model.evaluate(
                        mb_obs, b_actions[m], b_bounds[m])
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
                mv = b_values[m]
                v_clipped = mv + (value - mv).clamp(-cfg.vf_clip, cfg.vf_clip)
                mr = b_returns[m]
                value_loss = 0.5 * torch.max(
                    (value - mr) ** 2, (v_clipped - mr) ** 2).mean()
                ent = entropy.mean()
                ent_term = (entropy * b_ent_w[m]).mean()
                loss = (policy_loss + cfg.vf_coef * value_loss
                        - ent_coef * ent_term)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - ratio.log()).mean()
                    stat_sum += torch.stack([
                        policy_loss.detach(), value_loss.detach(),
                        ent.detach(), kl,
                        ((ratio - 1).abs() > cfg.clip).float().mean()])
                count += 1
                if cfg.target_kl > 0 and kl.item() > cfg.target_kl:
                    stop = True
                    break
        s = (stat_sum / count).tolist()
        return {"policy_loss": s[0], "value_loss": s[1], "entropy": s[2],
                "approx_kl": s[3], "clip_frac": s[4],
                "lr": lr, "ent_coef": ent_coef}
