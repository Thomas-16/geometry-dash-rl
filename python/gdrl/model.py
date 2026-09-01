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

Precision (measured on an RTX 4060 Ti, minibatch 8192): the fp32 NCHW
fwd+bwd is 37 ms, dominated by cuDNN's weight-gradient kernels for the
tiny-channel convs; bf16 autocast with channels_last conv tensors is 16 ms.
The trunks therefore run under autocast (see `autocast_ctx`) while the heads,
the logit bound and every loss stay fp32 — bf16 logits would inject ~1e-2
noise straight into PPO's probability ratios. The Bernoulli math is written
out by hand (`torch.distributions` adds an argument-validation device sync
per call and ~10% time).
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

# Grid geometry, mirroring obs.hpp (kGridRows/kGridCols, two channels:
# solid then hazard, each [row][col] row-major — so the flat obs tail
# reshapes to (2, rows, cols) as a pure view).
GRID_CH, GRID_ROWS, GRID_COLS = 2, 24, 32
GRID_DIM = GRID_CH * GRID_ROWS * GRID_COLS

AMP_MODES = ("bf16", "none")


def configure_backend() -> None:
    """One-time global torch settings for the fixed-shape conv workload."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def autocast_ctx(device: torch.device, amp: str):
    """Trunk-precision context: bf16 autocast on CUDA, no-op otherwise."""
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if amp not in AMP_MODES:
        raise ValueError(f"unknown amp mode {amp!r} (expected {AMP_MODES})")
    return contextlib.nullcontext()


def bernoulli_logprob(logit: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return -F.binary_cross_entropy_with_logits(logit, action, reduction="none")


def bernoulli_entropy(logit: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logit)
    return p * F.softplus(-logit) + (1.0 - p) * F.softplus(logit)


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
        g = g.contiguous(memory_format=torch.channels_last)
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
        pi_feat = self.pi_body(obs)
        vf_feat = self.vf_body(obs)
        # Heads + bound in fp32 regardless of the trunk autocast state.
        with torch.autocast(obs.device.type, enabled=False):
            raw = self.policy(pi_feat.float()).squeeze(-1)
            logit = self.logit_bound * torch.tanh(raw / self.logit_bound)
            value = self.value(vf_feat.float()).squeeze(-1)
        return logit, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Returns one stacked [3, N] fp32 tensor: action, logprob, value —
        one tensor so the rollout does a single device->host copy per step."""
        logit, value = self(obs)
        if deterministic:
            action = (logit > 0).float()
            logprob = torch.zeros_like(logit)
        else:
            action = (torch.rand_like(logit) < torch.sigmoid(logit)).float()
            logprob = bernoulli_logprob(logit, action)
        return torch.stack([action, logprob, value])

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logit, value = self(obs)
        return bernoulli_logprob(logit, actions), bernoulli_entropy(logit), value


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
        # channels_last conv weights: keys/shapes unchanged, so checkpoints
        # from the NCHW era load as-is.
        self.to(memory_format=torch.channels_last)


def load_model_state(model: _ActorCriticBase, state: dict,
                     strict: bool = True) -> list[str]:
    """load_state_dict that tolerates an *appended* observation block.

    obs.hpp only ever appends scalars (v2 -> v3 added the gravity flag and
    the next-special features), so a checkpoint trained on a shorter obs is
    still meaningful: the first input layer's weight matrix is widened with
    zero columns for the new scalars, which reproduces the old policy
    exactly on any state where the new features were absent. Grid columns
    keep their positions because the grid follows the scalar block: the
    checkpoint's (rows, cols) are re-laid out over the new scalar_dim.
    Returns a list of human-readable notes about what was widened."""
    notes: list[str] = []
    own = model.state_dict()
    fixed = dict(state)
    for key, w in own.items():
        old = fixed.get(key)
        if old is None or old.shape == w.shape or old.dim() != 2 or w.dim() != 2:
            continue
        if old.shape[0] != w.shape[0] or old.shape[1] >= w.shape[1]:
            continue
        pad = w.shape[1] - old.shape[1]
        widened = torch.zeros_like(w)
        # scalar-branch layer (conv arch): plain prefix copy
        # flat-MLP first layer: scalars are a prefix, the grid a suffix
        grid = GRID_DIM if old.shape[1] > GRID_DIM else 0
        if grid:
            s_old = old.shape[1] - grid
            widened[:, :s_old] = old[:, :s_old]
            widened[:, s_old + pad:] = old[:, s_old:]
        else:
            widened[:, :old.shape[1]] = old
        fixed[key] = widened.to(old.dtype)
        notes.append(f"{key}: widened input {old.shape[1]} -> {w.shape[1]} "
                     f"(+{pad} zero columns for appended obs scalars)")
    model.load_state_dict(fixed, strict=strict)
    return notes


def make_model(obs_dim: int, arch: str = "cnn",
               logit_bound: float = 4.0) -> _ActorCriticBase:
    if arch == "cnn":
        return ConvActorCritic(obs_dim, logit_bound)
    if arch == "mlp":
        return ActorCritic(obs_dim, logit_bound=logit_bound)
    raise ValueError(f"unknown arch {arch!r} (expected 'cnn' or 'mlp')")


class GraphedAct:
    """CUDA-graph replay of `model.act` for a fixed batch: the rollout's
    per-step forward is ~30 small kernels whose launch overhead exceeds
    their GPU time. Call with the rollout batch already copied into
    `self.obs` (the graph's static input); returns the static [3, N]
    output (read it before the next call). Re-captures when the model's
    logit bound changes (polish phase) — it is baked into the graph as a
    constant. RNG inside the capture is graph-safe (fresh samples per
    replay). Falls back to eager if capture fails."""

    def __init__(self, model: _ActorCriticBase, n: int, obs_dim: int,
                 device: torch.device, amp: str):
        self.model = model
        self.amp = amp
        self.device = device
        self.obs = torch.zeros(n, obs_dim, device=device)
        self.out: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.bound: float | None = None
        self.enabled = device.type == "cuda"

    def _eager(self) -> torch.Tensor:
        with autocast_ctx(self.device, self.amp):
            return self.model.act(self.obs)

    def _capture(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._eager()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.out = self._eager()
        self.graph = g
        self.bound = self.model.logit_bound

    def __call__(self) -> torch.Tensor:
        if not self.enabled:
            return self._eager()
        if self.graph is None or self.bound != self.model.logit_bound:
            try:
                self._capture()
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: CUDA graph capture of act() failed "
                      f"({type(e).__name__}: {e}); running eager")
                self.enabled = False
                return self._eager()
        self.graph.replay()
        return self.out


def maybe_compile(model: _ActorCriticBase, enabled: bool) -> _ActorCriticBase:
    """Opt-in torch.compile of the two trunks (Inductor fuses the bias+tanh
    elementwise passes, ~20% of the update's GPU time). Needs Triton
    (`pip install triton-windows` on Windows); any failure falls back to
    eager with a warning rather than aborting a run."""
    if not enabled:
        return model
    try:
        import triton  # noqa: F401
        # Compile the bound forwards, not the modules: an OptimizedModule
        # wrapper would prefix state_dict keys with `_orig_mod.` and break
        # checkpoint compatibility.
        for body in (model.pi_body, model.vf_body):
            body.forward = torch.compile(body.forward,
                                         mode="max-autotune-no-cudagraphs")
    except Exception as e:  # noqa: BLE001 — never let compile kill a run
        print(f"WARNING: torch.compile unavailable ({type(e).__name__}: {e}); "
              f"running eager")
    return model
