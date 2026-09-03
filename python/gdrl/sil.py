"""Self-imitation learning on the sim's free deterministic replay.

The sim is deterministic, so an episode is fully described by its per-tick
holds — the trainer's EpisodeRecorder already collects them for every
finished from-start episode. This module keeps the best of those (all wins
plus the deepest deaths), regenerates their exact (obs, action, reward,
duration) decision streams by replaying the holds through a spare env
handle, and applies the SIL loss on positive-advantage transitions:

    L = -logpi(a|s) * (R - V)_+  +  0.5 * (R - V)_+^2

where R is the SMDP discounted return-to-go (gamma^ticks between decisions,
same discounting as PPO). Storage is bytes per episode; obs streams are
regenerated on demand, never cached — the sim is the experience buffer.

Two more pools ride on the same replay (both added after the Cycles run
that sat 500 updates at one wall):

- Demonstration SEGMENTS: (start state, holds) pairs the trainer's planner
  (gdenv_solve, a macro-action BFS from the deterministic policy's own
  pre-wall states) found through a wall the sampler cannot cross. They
  replay from their start blob; a segment that ends alive bootstraps
  V(final obs) into its return-to-go. Demo rows get a fixed share of every
  SIL minibatch so a few hundred wall decisions are not drowned by the
  ~40k rows of full episodes.
- The ANCHOR: the best deterministic run's holds. Its decisions inside the
  mastered zone (x < best det x - margin) get a hinge term
  anchor_coef * relu(margin - sign(a) * logit): every prefix decision is
  held at least `anchor_margin` logits on its own side. The mastered
  prefix samples near-argmax, so PPO has no gradient there and the trunk
  drifts under frontier updates until a decision's logit crosses zero and
  the prefix breaks (det_x fell 12777 -> 2490 twice in level9-2, and a
  first anchor version using plain -logpi still let a det run die at
  x 5547 after reaching 22112: at p = 0.999 the log-likelihood gradient is
  ~0, so it only repaired flips after the fact). The hinge is zero where
  the margin holds, so it never fights the frontier updates.

Replay correctness: at each decision boundary the macro layer's action is
exactly the recorded hold bit at that tick (a hop macro starts with hold=1,
walk/air/ship windows hold a constant bit), so feeding holds[cursor] back
as the action reproduces the original trajectory bit-for-bit. The one
exception - a grounded ball whose press is already spent records a release
tick before its flip's press - is disambiguated inside gdenv_replay by
looking one tick ahead. The replay itself is one DLL call (`gdenv_replay`,
thread-pooled over episodes).
"""

from __future__ import annotations

import numpy as np
import torch

from .env import GdVecEnv
from .model import autocast_ctx, bernoulli_logprob


class SILBuffer:
    """Keeps all winning episodes plus the top-K deepest non-wins, and the
    planner's demonstration segments."""

    def __init__(self, cap_wins: int = 32, cap_deaths: int = 32,
                 cap_demos: int = 64):
        self.cap_wins = cap_wins
        self.cap_deaths = cap_deaths
        self.cap_demos = cap_demos
        self.wins: list[tuple[np.ndarray, float]] = []    # (holds, final_x)
        self.deaths: list[tuple[np.ndarray, float]] = []
        # (start blob, holds, start_x, reach_x); FIFO — a wall's demos age
        # out once the policy has moved on
        self.demos: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        self._keys: set[bytes] = set()

    def __len__(self) -> int:
        return len(self.wins) + len(self.deaths)

    @staticmethod
    def _key(holds: np.ndarray) -> bytes:
        return np.asarray(holds, np.uint8).tobytes()

    def add(self, holds: np.ndarray, final_x: float, won: bool) -> None:
        if len(holds) == 0:
            return
        # Dedupe identical input sequences (deterministic sim -> identical
        # trajectories add nothing): O(1) set lookup on the raw bytes.
        key = self._key(holds)
        if key in self._keys:
            return
        pool = self.wins if won else self.deaths
        cap = self.cap_wins if won else self.cap_deaths
        pool.append((np.asarray(holds, np.uint8), float(final_x)))
        self._keys.add(key)
        if len(pool) > cap:
            # Wins are interchangeable certificates: FIFO. Deaths: keep the
            # deepest (they carry the most not-yet-mastered level suffix).
            if not won:
                pool.sort(key=lambda t: t[1])
            dropped, _ = pool.pop(0)
            self._keys.discard(self._key(dropped))

    def add_demo(self, blob: np.ndarray, holds: np.ndarray, start_x: float,
                 reach_x: float) -> bool:
        if len(holds) == 0:
            return False
        key = b"demo" + np.asarray(blob, np.uint8).tobytes() + self._key(holds)
        if key in self._keys:
            return False
        self.demos.append((np.asarray(blob, np.uint8),
                           np.asarray(holds, np.uint8), float(start_x),
                           float(reach_x)))
        self._keys.add(key)
        if len(self.demos) > self.cap_demos:
            b, h, _, _ = self.demos.pop(0)
            self._keys.discard(b"demo" + b.tobytes() + self._key(h))
        return True

    def prune_demos(self, min_reach_x: float) -> int:
        """Drop demos that end before min_reach_x (the mastered zone once
        the frontier moved on): a BFS segment there teaches a different
        route than the anchored det run, and with the leaky logit bound
        the policy can actually follow it — level9-4 lost its prefix 3x
        within ~60 updates of prefix demos. Returns the number dropped."""
        keep = [d for d in self.demos if d[3] >= min_reach_x]
        n = len(self.demos) - len(keep)
        if n:
            self.demos = keep
            self._keys = {k for k in self._keys if not k.startswith(b"demo")}
            self._keys |= {b"demo" + b.tobytes() + self._key(h)
                           for b, h, _, _ in self.demos}
        return n

    def sample(self, k: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Up to k episodes, wins weighted 3x."""
        items = [(h, 3.0) for h, _ in self.wins] + \
                [(h, 1.0) for h, _ in self.deaths]
        if not items:
            return []
        w = np.array([t[1] for t in items])
        idx = rng.choice(len(items), min(k, len(items)), replace=False,
                         p=w / w.sum())
        return [items[i][0] for i in idx]

    def sample_demos(self, k: int, rng: np.random.Generator):
        """Up to k demo segments, newest weighted up (the current wall)."""
        n = len(self.demos)
        if n == 0 or k <= 0:
            return []
        w = np.linspace(1.0, 3.0, n)
        idx = rng.choice(n, min(k, n), replace=False, p=w / w.sum())
        return [self.demos[i] for i in idx]

    def state_dict(self) -> dict:
        return {"wins": self.wins, "deaths": self.deaths, "demos": self.demos}

    def load_state_dict(self, d: dict) -> None:
        self.wins = list(d.get("wins", []))
        self.deaths = list(d.get("deaths", []))
        self.demos = list(d.get("demos", []))
        self._keys = {self._key(h) for h, _ in self.wins + self.deaths}
        self._keys |= {b"demo" + b.tobytes() + self._key(h)
                       for b, h, _, _ in self.demos}


def replay(env: GdVecEnv, episodes: list[np.ndarray]):
    """Regenerate decision streams for episodes (assigned to envs 0..k-1).

    Returns per-episode (obs [T, obs_dim], actions [T], rewards [T],
    ticks [T]) as float32 numpy arrays — a view-per-episode over the DLL's
    padded [k, T_max] batch (`GdVecEnv.replay`)."""
    obs, acts, rews, ticks, n_steps, _, _ = env.replay(episodes)
    return [(obs[i, :n], acts[i, :n].astype(np.float32), rews[i, :n],
             ticks[i, :n]) for i, n in enumerate(n_steps.tolist())]


def returns_to_go(rewards: np.ndarray, ticks: np.ndarray, gamma: float,
                  reward_scale: float, boot=None) -> np.ndarray:
    """SMDP discounted return-to-go, in the trainer's reward units.
    Accepts [T] or [k, T] (rows scanned together; pad rows with zeros).
    boot ([k] or scalar) seeds the scan: the value after the last step,
    for streams that end alive (0 = terminal). Pad rows (zero reward,
    zero ticks -> discount 1) carry it unchanged to the last valid row."""
    r = np.asarray(rewards, np.float32) * reward_scale
    g = (gamma ** np.asarray(ticks, np.float32)).astype(np.float32)
    out = np.zeros_like(r)
    acc = np.zeros(r.shape[:-1], np.float32)
    if boot is not None:
        acc = acc + np.asarray(boot, np.float32)
    for t in range(r.shape[-1] - 1, -1, -1):
        acc = r[..., t] + g[..., t] * acc
        out[..., t] = acc
    return out


class SIL:
    def __init__(self, model, optimizer, device, gamma: float,
                 reward_scale: float, coef: float = 1.0,
                 minibatch: int = 4096, steps: int = 2, amp: str = "bf16",
                 demo_share: float = 0.25, anchor_coef: float = 0.1,
                 anchor_margin: float = 2.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.coef = coef
        self.minibatch = minibatch
        self.steps = steps
        self.amp = amp
        self.demo_share = demo_share
        self.anchor_coef = anchor_coef
        self.anchor_margin = anchor_margin

    def update(self, env: GdVecEnv, buffer: SILBuffer, k: int,
               rng: np.random.Generator, max_grad_norm: float = 0.5,
               k_demos: int = 0, anchor: tuple[np.ndarray, float] | None = None,
               anchor_x_lo: float = -1.0) -> dict:
        """Replay k sampled episodes (+ k_demos planner segments + the
        anchor run) and take SIL gradient steps on their positive-advantage
        transitions; anchor rows with x < anchor_x_lo get the BC term.
        Returns stats (zeros if nothing)."""
        stats = {"sil_loss": 0.0, "sil_frac_pos": 0.0, "sil_n": 0,
                 "sil_demo_n": 0, "anchor_n": 0}
        if self.coef <= 0.0:
            return stats
        episodes = buffer.sample(k, rng) if len(buffer) else []
        demos = buffer.sample_demos(k_demos, rng)
        use_anchor = (anchor is not None and self.anchor_coef > 0
                      and anchor_x_lo > 0 and len(anchor[0]) > 0)
        holds = list(episodes) + [h for _, h, _, _ in demos]
        blobs = [None] * len(episodes) + [b for b, _, _, _ in demos]
        n_main = len(holds)
        if use_anchor:
            holds.append(anchor[0])
            blobs.append(None)
        if not holds:
            return stats
        assert len(holds) <= env.n_envs, "sil env handle too small"
        obs, acts, rews, ticks, n_steps, final_obs, ends = env.replay(
            holds, blobs if demos else None)
        # Bootstrap for streams that end alive (demo segments; a from-start
        # episode always ends terminal).
        boot = np.zeros(len(holds), np.float32)
        alive = ends == 0
        if alive.any():
            with torch.no_grad(), autocast_ctx(self.device, self.amp):
                v = self.model(torch.as_tensor(final_obs[alive],
                                               device=self.device))[1]
            boot[alive] = v.float().cpu().numpy()
        # Returns over the padded [k, T] batch (pad rows are zero reward and
        # sit after the episode end, so they never leak into valid steps),
        # then keep only the valid rows.
        rets = returns_to_go(rews, ticks, self.gamma, self.reward_scale, boot)
        T = obs.shape[1]
        valid = np.arange(T)[None, :] < n_steps[:, None]
        row_ep = np.broadcast_to(np.arange(len(holds))[:, None], valid.shape)
        is_demo = (row_ep >= len(episodes)) & (row_ep < n_main)
        is_anchor = row_ep >= n_main
        # SIL weight 1 on episode/demo rows; anchor rows get the BC term
        # inside the mastered zone only (their tail is the death path).
        sil_w = np.where(is_anchor, 0.0, 1.0).astype(np.float32)
        bc_w = np.zeros(valid.shape, np.float32)
        if use_anchor:
            x = obs[..., 5] * env.end_x
            bc_w = np.where(is_anchor & (x < anchor_x_lo), self.anchor_coef,
                            0.0).astype(np.float32)
        t_obs = torch.as_tensor(obs[valid], device=self.device)
        t_acts = torch.as_tensor(acts[valid].astype(np.float32),
                                 device=self.device)
        t_rets = torch.as_tensor(rets[valid], device=self.device)
        t_sil_w = torch.as_tensor(sil_w[valid], device=self.device)
        t_bc_w = torch.as_tensor(bc_w[valid], device=self.device)
        demo_rows = torch.as_tensor(np.nonzero(is_demo[valid])[0],
                                    device=self.device)
        n = t_obs.shape[0]
        stats["sil_n"] = n
        stats["sil_demo_n"] = int(demo_rows.numel())
        stats["anchor_n"] = int((bc_w[valid] > 0).sum())
        n_demo_mb = (int(self.minibatch * self.demo_share)
                     if demo_rows.numel() else 0)
        loss_sum = torch.zeros((), device=self.device)
        pos_sum = torch.zeros((), device=self.device)
        for _ in range(self.steps):
            # Fixed minibatch shape (sampling with replacement, so n <
            # minibatch just repeats rows): a shape that varied with the
            # replayed episode count made cudnn.benchmark re-autotune the
            # convs every update (~2 s each). Demo rows get a fixed share.
            idx = torch.randint(n, (self.minibatch - n_demo_mb,),
                                device=self.device)
            if n_demo_mb:
                pick = torch.randint(demo_rows.numel(), (n_demo_mb,),
                                     device=self.device)
                idx = torch.cat([idx, demo_rows[pick]])
            a = t_acts[idx]
            with autocast_ctx(self.device, self.amp):
                logit, value = self.model(t_obs[idx])
            logprob = bernoulli_logprob(logit, a)
            adv = (t_rets[idx] - value).clamp(min=0.0)
            w = t_sil_w[idx]
            sil_loss = (w * (-(logprob * adv.detach()) + 0.5 * adv ** 2)).mean()
            signed = (2.0 * a - 1.0) * logit
            bc_loss = (torch.relu(self.anchor_margin - signed)
                       * t_bc_w[idx]).mean()
            loss = self.coef * sil_loss + bc_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_grad_norm)
            self.optimizer.step()
            loss_sum += loss.detach()
            pos_sum += ((adv > 0) & (w > 0)).float().mean()
        stats["sil_loss"] = loss_sum.item() / self.steps
        stats["sil_frac_pos"] = pos_sum.item() / self.steps
        return stats
