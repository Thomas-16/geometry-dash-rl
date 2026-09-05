"""Actor-critic networks for the gdenv observation (scalars + optional
geometry rows + 2 occupancy grids, see src/env/obs.hpp). obs_dim always
comes from the env (gdenv_obs_dim), never hard-coded.

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
  ~1.8% per-decision exploration floor. Deterministic eval (logit > 0) is
  unaffected — tanh is monotone. The bound is a per-sample input
  (`bound` tensor [N]; a float/None falls back to `model.logit_bound`):
  measured on Polargeist, a full run is ~3300 decisions, so a blanket
  1.8% floor forces ~60 random flips per run and the winning policy,
  sampled from x=0 at L=4, reached the level's last third 0.1% of the
  time (43% at L=8). The trainer therefore keeps L=4 only near the
  frontier and uses a high bound in territory the argmax policy already
  clears; PPO stores each sample's bound so the ratios stay exact.
  The bound's own gradient d logit/d raw = 1 - tanh^2(raw/L) vanishes once
  |raw| >> L, and nothing keeps raw small: on the Cycles level9-3 run half
  of all decisions had |raw| > 8 and the corridor at x 12777 sat at raw
  -25 (gradient 2.5e-5), so PPO, SIL demonstrations and the entropy bonus
  were all dead there and the argmax policy needed 990 updates to learn
  one flip in a 19-decision window. `logit_leak` floors that derivative
  (backward only: forward values, sampling and PPO ratios are unchanged)
  so a wrong saturated decision stays trainable: d logit/d raw =
  max(1 - tanh^2, leak). 0 = the plain tanh bound.

Action spaces (env.action_mode): n_actions == 2 keeps the Bernoulli hold
head above (every level 1-12 checkpoint); n_actions > 2 (precision mode) is
a categorical head over per-action bounded logits — each logit is
L*tanh(raw/L) with the same leak, illegal actions (env.action_masks) are
masked to -inf before the softmax, so a categorical logit gap is capped at
2L (bound 4 -> the least likely legal action keeps >= e^-8 / n of the mass,
the same kind of floor the Bernoulli bound gives). The mask is a per-sample
input everywhere (act / evaluate / logprob) and PPO stores it with the
sample so the ratios compare the same masked distributions.

Architectures:
- "cnn" (default): per-trunk conv encoder over the (2, 24, 32) grid — the
  grid is player-anchored, so conv translation equivariance maps directly
  onto "the same obstacle pattern appearing later in the level" — with the
  scalars on a parallel branch (obs.hpp records that diffuse grid signal
  loses to O(1) scalar features in a flat net; keep both paths explicit).
  Obs v6 adds an exact-geometry branch: the 12 nearest-object rows go
  through a shared per-row encoder and are concatenated (they arrive
  sorted by distance, so position carries meaning).
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
# Obs v6 geometry block (obs.hpp kGeomRows x kGeomFloats), between the
# scalars and the grid.
GEOM_ROWS, GEOM_FLOATS = 12, 10
GEOM_DIM = GEOM_ROWS * GEOM_FLOATS

AMP_MODES = ("bf16", "none")
MASK_NEG = -1e4  # masked logit (finite: keeps bf16/fp32 softmax NaN-free)


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


def masked_logits(logits: torch.Tensor, mask) -> torch.Tensor:
    """logits [N, A] with illegal actions (mask 0) pushed to MASK_NEG."""
    if mask is None:
        return logits
    return torch.where(mask > 0.5, logits, torch.full_like(logits, MASK_NEG))


def categorical_logprob(logits: torch.Tensor, action: torch.Tensor,
                        mask=None) -> torch.Tensor:
    lp = F.log_softmax(masked_logits(logits, mask), dim=-1)
    return lp.gather(-1, action.long().unsqueeze(-1)).squeeze(-1)


def categorical_entropy(logits: torch.Tensor, mask=None) -> torch.Tensor:
    lp = F.log_softmax(masked_logits(logits, mask), dim=-1)
    p = lp.exp()
    return -(p * lp).sum(-1)


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
    """Grid conv branch + scalar branch [+ geometry branch] -> 256-dim."""

    def __init__(self, scalar_dim: int, geom: bool = False):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.geom = geom
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
        mix_in = 192 + 64
        if geom:
            # Shared per-row encoder, rows concatenated in distance order.
            self.geom_row = nn.Sequential(
                _ortho(nn.Linear(GEOM_FLOATS, 32), 2 ** 0.5), nn.Tanh(),
            )
            self.geom_mix = nn.Sequential(
                _ortho(nn.Linear(GEOM_ROWS * 32, 128), 2 ** 0.5), nn.Tanh(),
            )
            mix_in += 128
        self.mix = nn.Sequential(
            _ortho(nn.Linear(mix_in, 256), 2 ** 0.5), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        s = obs[:, :self.scalar_dim]
        parts = [self.scalars(s)]
        off = self.scalar_dim
        if self.geom:
            rows = obs[:, off:off + GEOM_DIM].reshape(-1, GEOM_ROWS, GEOM_FLOATS)
            enc = self.geom_row(rows).reshape(-1, GEOM_ROWS * 32)
            parts.append(self.geom_mix(enc))
            off += GEOM_DIM
        g = obs[:, off:].reshape(-1, GRID_CH, GRID_ROWS, GRID_COLS)
        g = g.contiguous(memory_format=torch.channels_last)
        parts.append(self.conv(g))
        return self.mix(torch.cat(parts, dim=-1))


class _ActorCriticBase(nn.Module):
    """Shared heads/act/evaluate over two separate trunk modules."""

    def __init__(self, pi_body: nn.Module, vf_body: nn.Module, feat_dim: int,
                 logit_bound: float, logit_leak: float = 0.0,
                 n_actions: int = 2):
        super().__init__()
        self.logit_bound = logit_bound
        self.logit_leak = logit_leak
        self.n_actions = n_actions
        self.categorical = n_actions > 2
        self.pi_body = pi_body
        self.vf_body = vf_body
        # Bernoulli: one hold logit. Categorical: one bounded logit per
        # action (state_dict key unchanged, shape differs by n_actions).
        self.policy = _ortho(nn.Linear(feat_dim, n_actions if self.categorical
                                       else 1), 0.01)
        self.value = _ortho(nn.Linear(feat_dim, 1), 1.0)

    def forward(self, obs: torch.Tensor, bound=None, mask=None):
        """bound: per-sample logit bound tensor [N] (or a float); None uses
        `self.logit_bound` for every row. mask: [N, A] legality (categorical
        only; None = all legal). Returns (logit [N] or [N, A] masked, value
        [N])."""
        pi_feat = self.pi_body(obs)
        vf_feat = self.vf_body(obs)
        if bound is None:
            bound = self.logit_bound
        # Heads + bound in fp32 regardless of the trunk autocast state.
        with torch.autocast(obs.device.type, enabled=False):
            raw = self.policy(pi_feat.float())
            if self.categorical:
                b = bound.unsqueeze(-1) if torch.is_tensor(bound) else bound
            else:
                raw = raw.squeeze(-1)
                b = bound
            y = torch.tanh(raw / b)
            logit = b * y
            if self.logit_leak > 0.0:
                # Leaky bound (see the module docstring): raw - raw.detach()
                # is exactly 0 in value, so this adds nothing to the logit
                # and adds max(leak - (1 - y^2), 0) to its derivative, i.e.
                # d logit/d raw = max(1 - y^2, leak). Plain tensor ops, so
                # torch.compile and the CUDA-graphed act path see no custom
                # autograd Function.
                slope = (1.0 - y * y).detach()
                logit = logit + (raw - raw.detach()) * torch.clamp(
                    self.logit_leak - slope, min=0.0)
            if self.categorical:
                logit = masked_logits(logit, mask)
            value = self.value(vf_feat.float()).squeeze(-1)
        return logit, value

    def logprob_entropy(self, logit: torch.Tensor, actions: torch.Tensor):
        """(logprob [N], entropy [N]) of `actions` under a forward's
        (already masked) logit."""
        if self.categorical:
            return categorical_logprob(logit, actions), categorical_entropy(logit)
        return bernoulli_logprob(logit, actions), bernoulli_entropy(logit)

    def greedy(self, logit: torch.Tensor) -> torch.Tensor:
        """Deterministic action(s) for a forward's logit, as float ids."""
        if self.categorical:
            return logit.argmax(-1).float()
        return (logit > 0).float()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False,
            bound=None, mask=None):
        """Returns one stacked [3, N] fp32 tensor: action, logprob, value —
        one tensor so the rollout does a single device->host copy per step."""
        logit, value = self(obs, bound, mask)
        if deterministic:
            action = self.greedy(logit)
            logprob = torch.zeros_like(value)
        elif self.categorical:
            # Gumbel-max sampling: graph-safe (no multinomial sync) and
            # exact for the masked softmax.
            u = torch.rand_like(logit).clamp_(1e-10, 1.0)
            action = (logit - torch.log(-torch.log(u))).argmax(-1)
            logprob = categorical_logprob(logit, action)
            action = action.float()
        else:
            action = (torch.rand_like(logit) < torch.sigmoid(logit)).float()
            logprob = bernoulli_logprob(logit, action)
        return torch.stack([action, logprob, value])

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor, bound=None,
                 mask=None):
        logit, value = self(obs, bound, mask)
        logprob, entropy = self.logprob_entropy(logit, actions)
        return logprob, entropy, value


class ActorCritic(_ActorCriticBase):
    """Original flat-MLP architecture ("mlp")."""

    def __init__(self, obs_dim: int, hidden: tuple[int, int] = (512, 256),
                 logit_bound: float = 4.0, logit_leak: float = 0.0,
                 n_actions: int = 2):
        h1, h2 = hidden
        super().__init__(_mlp_trunk(obs_dim, h1, h2),
                         _mlp_trunk(obs_dim, h1, h2), h2, logit_bound,
                         logit_leak, n_actions)


class ConvActorCritic(_ActorCriticBase):
    """Grid-CNN architecture ("cnn"), with the geometry branch for obs v6."""

    def __init__(self, obs_dim: int, logit_bound: float = 4.0,
                 logit_leak: float = 0.0, n_actions: int = 2,
                 obs_version: int = 5):
        geom = obs_version >= 6
        scalar_dim = obs_dim - GRID_DIM - (GEOM_DIM if geom else 0)
        if scalar_dim <= 0:
            raise ValueError(f"obs_dim {obs_dim} too small for grid "
                             f"{GRID_DIM} — obs.hpp/model.py mismatch")
        super().__init__(_ConvTrunk(scalar_dim, geom), _ConvTrunk(scalar_dim, geom),
                         256, logit_bound, logit_leak, n_actions)
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
    Returns a list of human-readable notes about what was widened. A
    different action head shape (Bernoulli vs categorical) is a contract
    mismatch and raises."""
    notes: list[str] = []
    own = model.state_dict()
    fixed = dict(state)
    for key in ("policy.weight", "policy.bias"):
        if key in fixed and fixed[key].shape != own[key].shape:
            raise ValueError(
                f"checkpoint action head {tuple(fixed[key].shape)} != model "
                f"{tuple(own[key].shape)}: action-space contract mismatch")
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


def make_model(obs_dim: int, arch: str = "cnn", logit_bound: float = 4.0,
               logit_leak: float = 0.0, n_actions: int = 2,
               obs_version: int = 5) -> _ActorCriticBase:
    """logit_leak matters only for training (backward); inference-only
    callers (eval, deploy, the rollout snapshot) can leave it at 0."""
    if arch == "cnn":
        return ConvActorCritic(obs_dim, logit_bound, logit_leak, n_actions,
                               obs_version)
    if arch == "mlp":
        return ActorCritic(obs_dim, logit_bound=logit_bound,
                           logit_leak=logit_leak, n_actions=n_actions)
    raise ValueError(f"unknown arch {arch!r} (expected 'cnn' or 'mlp')")


class GraphedAct:
    """CUDA-graph replay of `model.act` for a fixed batch: the rollout's
    per-step forward is ~30 small kernels whose launch overhead exceeds
    their GPU time. Call with the rollout batch already copied into
    `self.obs` (the graph's static input), the per-row logit bound in
    `self.bound` and the action mask in `self.mask` (static too); returns
    the static [3, N] output (read it before the next call). RNG inside
    the capture is graph-safe (fresh samples per replay). Falls back to
    eager if capture fails."""

    def __init__(self, model: _ActorCriticBase, n: int, obs_dim: int,
                 device: torch.device, amp: str):
        self.model = model
        self.amp = amp
        self.device = device
        self.obs = torch.zeros(n, obs_dim, device=device)
        self.bound = torch.full((n,), float(model.logit_bound), device=device)
        self.mask = torch.ones(n, model.n_actions, device=device)
        self.out: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.enabled = device.type == "cuda"

    def _eager(self) -> torch.Tensor:
        with autocast_ctx(self.device, self.amp):
            return self.model.act(self.obs, bound=self.bound,
                                  mask=self.mask if self.model.categorical
                                  else None)

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

    def __call__(self) -> torch.Tensor:
        if not self.enabled:
            return self._eager()
        if self.graph is None:
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
