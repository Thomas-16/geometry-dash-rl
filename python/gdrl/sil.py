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

Replay correctness: at each decision boundary the macro layer's action is
exactly the recorded hold bit at that tick (a hop macro starts with hold=1,
walk/air/ship windows hold a constant bit), so feeding holds[cursor] back
as the action reproduces the original trajectory bit-for-bit. The one
exception - a grounded ball whose press is already spent records a release
tick before its flip's press - is disambiguated inside gdenv_replay by
looking one tick ahead. The replay
itself is one DLL call (`gdenv_replay`, thread-pooled over episodes).
"""

from __future__ import annotations

import numpy as np
import torch

from .env import GdVecEnv
from .model import autocast_ctx


class SILBuffer:
    """Keeps all winning episodes plus the top-K deepest non-wins."""

    def __init__(self, cap_wins: int = 32, cap_deaths: int = 32):
        self.cap_wins = cap_wins
        self.cap_deaths = cap_deaths
        self.wins: list[tuple[np.ndarray, float]] = []    # (holds, final_x)
        self.deaths: list[tuple[np.ndarray, float]] = []
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

    def state_dict(self) -> dict:
        return {"wins": self.wins, "deaths": self.deaths}

    def load_state_dict(self, d: dict) -> None:
        self.wins = list(d.get("wins", []))
        self.deaths = list(d.get("deaths", []))
        self._keys = {self._key(h) for h, _ in self.wins + self.deaths}


def replay(env: GdVecEnv, episodes: list[np.ndarray]):
    """Regenerate decision streams for episodes (assigned to envs 0..k-1).

    Returns per-episode (obs [T, obs_dim], actions [T], rewards [T],
    ticks [T]) as float32 numpy arrays — a view-per-episode over the DLL's
    padded [k, T_max] batch (`GdVecEnv.replay`)."""
    obs, acts, rews, ticks, n_steps = env.replay(episodes)
    return [(obs[i, :n], acts[i, :n].astype(np.float32), rews[i, :n],
             ticks[i, :n]) for i, n in enumerate(n_steps.tolist())]


def returns_to_go(rewards: np.ndarray, ticks: np.ndarray, gamma: float,
                  reward_scale: float) -> np.ndarray:
    """SMDP discounted return-to-go, in the trainer's reward units.
    Accepts [T] or [k, T] (rows scanned together; pad rows with zeros)."""
    r = np.asarray(rewards, np.float32) * reward_scale
    g = (gamma ** np.asarray(ticks, np.float32)).astype(np.float32)
    out = np.zeros_like(r)
    acc = np.zeros(r.shape[:-1], np.float32)
    for t in range(r.shape[-1] - 1, -1, -1):
        acc = r[..., t] + g[..., t] * acc
        out[..., t] = acc
    return out


class SIL:
    def __init__(self, model, optimizer, device, gamma: float,
                 reward_scale: float, coef: float = 1.0,
                 minibatch: int = 4096, steps: int = 2, amp: str = "bf16"):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.coef = coef
        self.minibatch = minibatch
        self.steps = steps
        self.amp = amp

    def update(self, env: GdVecEnv, buffer: SILBuffer, k: int,
               rng: np.random.Generator, max_grad_norm: float = 0.5) -> dict:
        """Replay k sampled episodes and take SIL gradient steps on their
        positive-advantage transitions. Returns stats (zeros if nothing)."""
        stats = {"sil_loss": 0.0, "sil_frac_pos": 0.0, "sil_n": 0}
        if self.coef <= 0.0 or not len(buffer):
            return stats
        episodes = buffer.sample(k, rng)
        if not episodes:
            return stats
        obs, acts, rews, ticks, n_steps = env.replay(episodes)
        # Returns over the padded [k, T] batch (pad rows are zero reward and
        # sit after the episode end, so they never leak into valid steps),
        # then keep only the valid rows.
        rets = returns_to_go(rews, ticks, self.gamma, self.reward_scale)
        valid = np.arange(obs.shape[1])[None, :] < n_steps[:, None]
        t_obs = torch.as_tensor(obs[valid], device=self.device)
        t_acts = torch.as_tensor(acts[valid].astype(np.float32),
                                 device=self.device)
        t_rets = torch.as_tensor(rets[valid], device=self.device)
        n = t_obs.shape[0]
        stats["sil_n"] = n
        loss_sum = torch.zeros((), device=self.device)
        pos_sum = torch.zeros((), device=self.device)
        for _ in range(self.steps):
            # Fixed minibatch shape (sampling with replacement, so n <
            # minibatch just repeats rows): a shape that varied with the
            # replayed episode count made cudnn.benchmark re-autotune the
            # convs every update (~2 s each).
            idx = torch.randint(n, (self.minibatch,), device=self.device)
            with autocast_ctx(self.device, self.amp):
                logprob, _, value = self.model.evaluate(t_obs[idx], t_acts[idx])
            adv = (t_rets[idx] - value).clamp(min=0.0)
            loss = self.coef * (-(logprob * adv.detach()).mean()
                                + 0.5 * (adv ** 2).mean())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_grad_norm)
            self.optimizer.step()
            loss_sum += loss.detach()
            pos_sum += (adv > 0).float().mean()
        stats["sil_loss"] = loss_sum.item() / self.steps
        stats["sil_frac_pos"] = pos_sum.item() / self.steps
        return stats
