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
as the action reproduces the original trajectory bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import torch

from .env import GdVecEnv


class SILBuffer:
    """Keeps all winning episodes plus the top-K deepest non-wins."""

    def __init__(self, cap_wins: int = 32, cap_deaths: int = 32):
        self.cap_wins = cap_wins
        self.cap_deaths = cap_deaths
        self.wins: list[tuple[np.ndarray, float]] = []    # (holds, final_x)
        self.deaths: list[tuple[np.ndarray, float]] = []

    def __len__(self) -> int:
        return len(self.wins) + len(self.deaths)

    def add(self, holds: np.ndarray, final_x: float, won: bool) -> None:
        if len(holds) == 0:
            return
        pool = self.wins if won else self.deaths
        cap = self.cap_wins if won else self.cap_deaths
        # Dedupe identical input sequences (deterministic sim -> identical
        # trajectories add nothing).
        key = (len(holds), holds.tobytes())
        for h, _ in pool:
            if (len(h), h.tobytes()) == key:
                return
        pool.append((np.asarray(holds, np.uint8), float(final_x)))
        if len(pool) > cap:
            # Wins are interchangeable certificates: FIFO. Deaths: keep the
            # deepest (they carry the most not-yet-mastered level suffix).
            if won:
                pool.pop(0)
            else:
                pool.sort(key=lambda t: t[1])
                pool.pop(0)

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


def replay(env: GdVecEnv, episodes: list[np.ndarray]):
    """Regenerate decision streams for episodes (assigned to envs 0..k-1).

    Returns per-episode (obs [T, obs_dim], actions [T], rewards [T],
    ticks [T]) as float32/uint8 numpy arrays. Uses the whole handle's step
    (idle envs step too — the sim is cheap); each traced env is reset first.
    """
    k = len(episodes)
    assert k <= env.n_envs
    cursors = np.zeros(k, np.int64)
    active = np.ones(k, bool)
    out = [([], [], [], []) for _ in range(k)]
    for i in range(k):
        env.reset_env(i)
    obs_now = env._obs.copy()
    acts = np.zeros(env.n_envs, np.uint8)
    while active.any():
        for i in np.nonzero(active)[0]:
            acts[i] = episodes[i][cursors[i]]
        obs, rew, term, trunc, info = env.step(acts)
        for i in np.nonzero(active)[0]:
            o, a, r, t = out[i]
            o.append(obs_now[i].copy())
            a.append(acts[i])
            r.append(rew[i])
            nt = int(info["ticks"][i])
            t.append(nt)
            cursors[i] += nt
            if term[i] or trunc[i] or cursors[i] >= len(episodes[i]):
                active[i] = False
        obs_now = obs.copy()
    return [(np.asarray(o, np.float32), np.asarray(a, np.float32),
             np.asarray(r, np.float32), np.asarray(t, np.float32))
            for (o, a, r, t) in out]


def returns_to_go(rewards: np.ndarray, ticks: np.ndarray, gamma: float,
                  reward_scale: float) -> np.ndarray:
    """SMDP discounted return-to-go, in the trainer's reward units."""
    r = rewards * reward_scale
    g = gamma ** ticks
    out = np.zeros_like(r)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + g[t] * acc
        out[t] = acc
    return out


class SIL:
    def __init__(self, model, optimizer, device, gamma: float,
                 reward_scale: float, coef: float = 1.0,
                 minibatch: int = 4096, steps: int = 2):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.coef = coef
        self.minibatch = minibatch
        self.steps = steps

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
        streams = replay(env, episodes)
        obs = np.concatenate([s[0] for s in streams])
        acts = np.concatenate([s[1] for s in streams])
        rets = np.concatenate([
            returns_to_go(s[2], s[3], self.gamma, self.reward_scale)
            for s in streams])
        t_obs = torch.as_tensor(obs, device=self.device)
        t_acts = torch.as_tensor(acts, device=self.device)
        t_rets = torch.as_tensor(rets, device=self.device)
        n = t_obs.shape[0]
        stats["sil_n"] = n
        for _ in range(self.steps):
            idx = torch.randint(n, (min(self.minibatch, n),),
                                device=self.device)
            logprob, _, value = self.model.evaluate(t_obs[idx], t_acts[idx])
            adv = (t_rets[idx] - value).clamp(min=0.0)
            frac_pos = (adv > 0).float().mean()
            loss = self.coef * (-(logprob * adv.detach()).mean()
                                + 0.5 * (adv ** 2).mean())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_grad_norm)
            self.optimizer.step()
            stats["sil_loss"] += loss.item() / self.steps
            stats["sil_frac_pos"] += frac_pos.item() / self.steps
        return stats
