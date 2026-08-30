"""Actor-critic networks for the gdenv observation (18 scalars + 2 occupancy
grids, see src/env/obs.hpp). obs_dim always comes from the env
(gdenv_obs_dim), never hard-coded.

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
  (logit > 0) is unaffected — tanh is monotone. The polish phase raises L
  at runtime (plain attribute, not in state_dict) to let the training
  distribution converge toward the argmax path.

Architectures:
- "cnn" (default): per-trunk conv encoder over the (2, 24, 32) grid — the
  grid is player-anchored, so conv translation equivariance maps directly
  onto "the same obstacle pattern appearing later in the level" — with the
  scalars on a parallel branch (obs.hpp records that diffuse grid signal
  loses to O(1) scalars in a flat net; keep both paths explicit).
- "mlp": the original flat net, kept as an A/B fallback.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Grid geometry, mirroring obs.hpp (kGridRows/kGridCols, two channels:
# solid then hazard, each [row][col] row-major — so the flat obs tail
# reshapes to (2, rows, cols) as a pure view).
GRID_CH, GRID_ROWS, GRID_COLS = 2, 24, 32
GRID_DIM = GRID_CH * GRID_ROWS * GRID_COLS


def _ortho(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp_trunk(obs_dim: int, h1: int, h2: int) -> nn.Sequential:
    return nn.Sequential(
        _ortho(nn.Linear(obs_dim, h1), 2 ** 0.5), nn.Tanh(),
        _ortho(nn.Linear(h1, h2), 2 ** 0.5), nn.Tanh(),
    )


class _ConvTrunk(nn.Module):
    """Grid conv branch + scalar branch -> mixed 256-dim feature."""

    def __init__(self, scalar_dim: int):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(GRID_CH, 16, 3, stride=2, padding=1), 2 ** 0.5),
            nn.Tanh(),  # (16, 12, 16)
            _ortho(nn.Conv2d(16, 32, 3, stride=2, padding=1), 2 ** 0.5),
            nn.Tanh(),  # (32, 6, 8)
            _ortho(nn.Conv2d(32, 64, 3, stride=2, padding=1), 2 ** 0.5),
            nn.Tanh(),  # (64, 3, 4)
            nn.Flatten(),
            _ortho(nn.Linear(64 * 3 * 4, 192), 2 ** 0.5), nn.Tanh(),
        )
        self.scalars = nn.Sequential(
            _ortho(nn.Linear(scalar_dim, 64), 2 ** 0.5), nn.Tanh(),
        )
        self.mix = nn.Sequential(
            _ortho(nn.Linear(192 + 64, 256), 2 ** 0.5), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        s = obs[:, :self.scalar_dim]
        g = obs[:, self.scalar_dim:].view(-1, GRID_CH, GRID_ROWS, GRID_COLS)
        return self.mix(torch.cat([self.scalars(s), self.conv(g)], dim=-1))


class _ActorCriticBase(nn.Module):
    """Shared heads/act/evaluate over two separate trunk modules."""

    def __init__(self, pi_body: nn.Module, vf_body: nn.Module, feat_dim: int,
                 logit_bound: float):
        super().__init__()
        self.logit_bound = logit_bound
        self.pi_body = pi_body
        self.vf_body = vf_body
        self.policy = _ortho(nn.Linear(feat_dim, 1), 0.01)  # hold logit
        self.value = _ortho(nn.Linear(feat_dim, 1), 1.0)

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


class ActorCritic(_ActorCriticBase):
    """Original flat-MLP architecture ("mlp")."""

    def __init__(self, obs_dim: int, hidden: tuple[int, int] = (512, 256),
                 logit_bound: float = 4.0):
        h1, h2 = hidden
        super().__init__(_mlp_trunk(obs_dim, h1, h2),
                         _mlp_trunk(obs_dim, h1, h2), h2, logit_bound)


class ConvActorCritic(_ActorCriticBase):
    """Grid-CNN architecture ("cnn")."""

    def __init__(self, obs_dim: int, logit_bound: float = 4.0):
        scalar_dim = obs_dim - GRID_DIM
        if scalar_dim <= 0:
            raise ValueError(f"obs_dim {obs_dim} too small for grid "
                             f"{GRID_DIM} — obs.hpp/model.py mismatch")
        super().__init__(_ConvTrunk(scalar_dim), _ConvTrunk(scalar_dim), 256,
                         logit_bound)


def make_model(obs_dim: int, arch: str = "cnn",
               logit_bound: float = 4.0) -> _ActorCriticBase:
    if arch == "cnn":
        return ConvActorCritic(obs_dim, logit_bound)
    if arch == "mlp":
        return ActorCritic(obs_dim, logit_bound=logit_bound)
    raise ValueError(f"unknown arch {arch!r} (expected 'cnn' or 'mlp')")
