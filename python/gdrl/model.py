"""Actor-critic MLPs for the flat observation (12 scalars + 2 occupancy grids).
obs_dim always comes from the env (gdenv_obs_dim), never hard-coded.

Two deliberate structural choices, both learned from failed runs:

- Separate policy and value trunks. With a shared body, a frontier
  breakthrough creates large value errors whose gradients rewrite the shared
  features and destroy the already-learned level prefix (observed as mean_x
  16k -> 2.4k collapses that took 100+ updates to recover from).

- Bounded policy logit: logit = L*tanh(raw/L). An unbounded Bernoulli logit
  saturates as the policy becomes confident, and a saturated Bernoulli is
  unrecoverable: sampling yields zero trajectory diversity (measured: 20k
  stochastic rollouts at a wall produced 0 deviations while 65% of random
  action sequences passed it), so advantages are identically zero, and the
  entropy gradient vanishes exponentially in |logit| so the entropy bonus
  can't undo it. L=4 caps action certainty at sigmoid(4) ~ 98.2%, i.e. a
  permanent ~1.8% per-decision exploration floor. Deterministic eval
  (logit > 0) is unaffected — tanh is monotone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _ortho(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _trunk(obs_dim: int, h1: int, h2: int) -> nn.Sequential:
    return nn.Sequential(
        _ortho(nn.Linear(obs_dim, h1), 2 ** 0.5), nn.Tanh(),
        _ortho(nn.Linear(h1, h2), 2 ** 0.5), nn.Tanh(),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: tuple[int, int] = (512, 256),
                 logit_bound: float = 4.0):
        super().__init__()
        h1, h2 = hidden
        self.logit_bound = logit_bound
        self.pi_body = _trunk(obs_dim, h1, h2)
        self.vf_body = _trunk(obs_dim, h1, h2)
        self.policy = _ortho(nn.Linear(h2, 1), 0.01)  # Bernoulli hold logit
        self.value = _ortho(nn.Linear(h2, 1), 1.0)

    def forward(self, obs: torch.Tensor):
        raw = self.policy(self.pi_body(obs)).squeeze(-1)
        logit = self.logit_bound * torch.tanh(raw / self.logit_bound)
        value = self.value(self.vf_body(obs)).squeeze(-1)
        return logit, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logit, value = self(obs)
        if deterministic:
            action = (logit > 0).float()
            logprob = torch.zeros_like(logit)
        else:
            dist = torch.distributions.Bernoulli(logits=logit)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logit, value = self(obs)
        dist = torch.distributions.Bernoulli(logits=logit)
        return dist.log_prob(actions), dist.entropy(), value
