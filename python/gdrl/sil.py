"""Self-imitation learning on the sim's free deterministic replay, plus the
planner-demonstration teacher loss.

The sim is deterministic, so an episode is fully described by its decision
stream (per-tick holds in the macro action mode; per-decision action ids in
precision mode, where raw holds cannot recover variable-duration choices) —
the trainer's EpisodeRecorder already collects both for every finished
episode. This module keeps the best of those (all wins plus the deepest
deaths), regenerates their exact (obs, action, reward, duration) decision
streams by replaying through a spare env handle, and applies the SIL loss
on positive-advantage transitions:

    L = -logpi(a|s) * (R - V)_+  +  0.5 * (R - V)_+^2

where R is the SMDP discounted return-to-go (gamma^ticks between decisions,
same discounting as PPO). Storage is bytes per episode; obs streams are
regenerated on demand, never cached — the sim is the experience buffer.

Two more pools ride on the same replay:

- Demonstration SEGMENTS: (start state, decisions) pairs the trainer's
  planner (gdenv_solve, an exact BFS from the deterministic policy's own
  pre-wall states) found through a wall the sampler cannot cross. They are
  verified by replay before admission (`verify_demo`) and taught with a
  DEDICATED supervised action loss (`teacher`): -logpi(a_demo|s) averaged
  over its own batch, coefficient `teacher_coef` (0.25). It does not
  depend on the critic's advantage estimate — under plain SIL a demo whose
  return the critic already "predicts" (or over-predicts, which is what a
  saturated wrong prior looks like at a wall) received no policy signal at
  all; level12-1/12-2 sat 1300 updates at one corridor with the right
  demos in the buffer for that reason. Half of each teacher batch is drawn
  from decisions where the policy's argmax disagrees with the demo, half
  uniformly. Tracked: the mean probability the policy gives the teacher's
  action, the disagreement rate, and (trainer-side) demo replay success.
- The ANCHOR: the best deterministic run's decisions. Its rows inside the
  mastered zone (x < best det x - margin) get a hinge term
  anchor_coef * relu(margin - sign(a) * logit) (Bernoulli head; the
  categorical head uses the logit gap to the runner-up): every prefix
  decision is held at least `anchor_margin` logits on its own side. Only
  used by the non-stitched regime — stitched training freezes the prefix
  as holds, so there is nothing to pin.
"""

from __future__ import annotations

import numpy as np
import torch

from .env import GdVecEnv
from .model import autocast_ctx


def _replay(env: GdVecEnv, holds: list, blobs: list, acts: list | None):
    """Replay through the handle in its action mode: decision streams when
    every entry has one (precision), raw holds otherwise. Returns
    (obs, acts, rews, ticks, n_steps, final_obs, ends, masks)."""
    use_blobs = blobs if any(b is not None for b in blobs) else None
    if env.precision and acts is not None and all(a is not None for a in acts):
        return env.replay_actions(acts, use_blobs)
    obs, a, rews, ticks, n_steps, final_obs, ends = env.replay(holds, use_blobs)
    masks = np.ones((*a.shape, env.n_actions), np.uint8)
    return obs, a, rews, ticks, n_steps, final_obs, ends, masks


def verify_demo(env: GdVecEnv, blob: np.ndarray | None, holds: np.ndarray,
                actions: np.ndarray | None, reach_x: float) -> tuple[bool, float]:
    """Replay one demonstration segment through the shared executor and
    check it reaches reach_x alive (or wins). Returns (ok, final_x)."""
    obs, _, _, _, n_steps, final_obs, ends, _ = _replay(
        env, [holds], [blob], [actions] if actions is not None else None)
    x = float(final_obs[0, 5] * env.end_x)
    alive_or_won = ends[0] == 0 or (ends[0] == 1 and x >= env.end_x - 0.5)
    return bool(alive_or_won and x >= reach_x - 0.5), x


class SILBuffer:
    """Keeps all winning episodes plus the top-K deepest non-wins, and the
    planner's demonstration segments."""

    def __init__(self, cap_wins: int = 32, cap_deaths: int = 32,
                 cap_demos: int = 64):
        self.cap_wins = cap_wins
        self.cap_deaths = cap_deaths
        self.cap_demos = cap_demos
        # (holds, final_x, start blob or None, decision stream or None).
        # Stitched training records frontier episodes as (suffix holds,
        # frontier blob); None blob = x=0.
        self.wins: list[tuple] = []
        self.deaths: list[tuple] = []
        # (start blob, holds, start_x, reach_x, decision stream or None);
        # FIFO — a wall's demos age out once the policy has moved on
        self.demos: list[tuple] = []
        self._keys: set[bytes] = set()

    def __len__(self) -> int:
        return len(self.wins) + len(self.deaths)

    @staticmethod
    def _key(holds: np.ndarray, blob: np.ndarray | None = None) -> bytes:
        k = np.asarray(holds, np.uint8).tobytes()
        return k if blob is None else np.asarray(blob, np.uint8).tobytes() + k

    def add(self, holds: np.ndarray, final_x: float, won: bool,
            blob: np.ndarray | None = None,
            actions: np.ndarray | None = None) -> None:
        if len(holds) == 0:
            return
        # Dedupe identical input sequences (deterministic sim -> identical
        # trajectories add nothing): O(1) set lookup on the raw bytes.
        key = self._key(holds, blob)
        if key in self._keys:
            return
        pool = self.wins if won else self.deaths
        cap = self.cap_wins if won else self.cap_deaths
        pool.append((np.asarray(holds, np.uint8), float(final_x),
                     None if blob is None else np.asarray(blob, np.uint8),
                     None if actions is None else np.asarray(actions, np.uint8)))
        self._keys.add(key)
        if len(pool) > cap:
            # Wins are interchangeable certificates: FIFO. Deaths: keep the
            # deepest (they carry the most not-yet-mastered level suffix).
            if not won:
                pool.sort(key=lambda t: t[1])
            dropped = pool.pop(0)
            self._keys.discard(self._key(dropped[0], dropped[2]))

    def _rekey(self) -> None:
        self._keys = {self._key(t[0], t[2]) for t in self.wins + self.deaths}
        self._keys |= {b"demo" + d[0].tobytes() + self._key(d[1])
                       for d in self.demos}

    def drop_before(self, min_final_x: float) -> int:
        """Drop episodes that ended before min_final_x (stitched training:
        once the certificate's frontier moved past them, their decisions
        are prefix territory the policy never visits again)."""
        n0 = len(self)
        self.wins = [t for t in self.wins if t[1] >= min_final_x]
        self.deaths = [t for t in self.deaths if t[1] >= min_final_x]
        if len(self) != n0:
            self._rekey()
        return n0 - len(self)

    def clear_episodes(self) -> int:
        """Drop every episode (a certificate section was replaced: their
        suffixes hang off a prefix that no longer exists)."""
        n0 = len(self)
        self.wins, self.deaths = [], []
        self._rekey()
        return n0

    def add_demo(self, blob: np.ndarray, holds: np.ndarray, start_x: float,
                 reach_x: float, actions: np.ndarray | None = None) -> bool:
        if len(holds) == 0:
            return False
        key = b"demo" + np.asarray(blob, np.uint8).tobytes() + self._key(holds)
        if key in self._keys:
            return False
        self.demos.append((np.asarray(blob, np.uint8),
                           np.asarray(holds, np.uint8), float(start_x),
                           float(reach_x),
                           None if actions is None else np.asarray(actions, np.uint8)))
        self._keys.add(key)
        if len(self.demos) > self.cap_demos:
            d = self.demos.pop(0)
            self._keys.discard(b"demo" + d[0].tobytes() + self._key(d[1]))
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
            self._rekey()
        return n

    def sample(self, k: int, rng: np.random.Generator) -> list[tuple]:
        """Up to k (holds, start blob, actions) episodes, wins weighted 3x."""
        items = [(t[0], t[2], t[3], 3.0) for t in self.wins] + \
                [(t[0], t[2], t[3], 1.0) for t in self.deaths]
        if not items:
            return []
        w = np.array([t[3] for t in items])
        idx = rng.choice(len(items), min(k, len(items)), replace=False,
                         p=w / w.sum())
        return [items[i][:3] for i in idx]

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

    def load_state_dict(self, d: dict, migrate=None) -> None:
        """migrate: optional blob migration (GdVecEnv.migrate_blob); entries
        whose blob cannot be migrated are dropped."""
        def fix(t, blob_at):
            t = list(t)
            if migrate is not None and t[blob_at] is not None:
                t[blob_at] = migrate(t[blob_at])
                if t[blob_at] is None:
                    return None
            return tuple(t)

        def ep(t):
            t = tuple(t)
            if len(t) == 2:
                t = (t[0], t[1], None, None)
            elif len(t) == 3:
                t = (t[0], t[1], t[2], None)
            return fix(t, 2)

        def demo(t):
            t = tuple(t)
            if len(t) == 4:
                t = (t[0], t[1], t[2], t[3], None)
            return fix(t, 0)

        self.wins = [e for e in (ep(t) for t in d.get("wins", [])) if e]
        self.deaths = [e for e in (ep(t) for t in d.get("deaths", [])) if e]
        self.demos = [e for e in (demo(t) for t in d.get("demos", [])) if e]
        self._rekey()


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
                 anchor_coef: float = 0.1, anchor_margin: float = 2.0,
                 teacher_coef: float = 0.25, teacher_minibatch: int = 512,
                 demo_in_sil: bool = False, demo_share: float = 0.25):
        self.model = model
        # Pre-teacher regime (baseline comparisons): demos also ride the
        # advantage-weighted SIL loss with a fixed minibatch share.
        self.demo_in_sil = demo_in_sil
        self.demo_share = demo_share
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.coef = coef
        self.minibatch = minibatch
        self.steps = steps
        self.amp = amp
        self.anchor_coef = anchor_coef
        self.anchor_margin = anchor_margin
        self.teacher_coef = teacher_coef
        self.teacher_minibatch = teacher_minibatch

    def _logits(self, obs: torch.Tensor, masks: torch.Tensor | None):
        with autocast_ctx(self.device, self.amp):
            return self.model(obs, mask=masks)

    def update(self, env: GdVecEnv, buffer: SILBuffer, k: int,
               rng: np.random.Generator, max_grad_norm: float = 0.5,
               k_demos: int = 0, anchor: tuple[np.ndarray, float] | None = None,
               anchor_x_lo: float = -1.0) -> dict:
        """Replay k sampled episodes (+ the anchor run) and take SIL
        gradient steps on their positive-advantage transitions; anchor
        rows with x < anchor_x_lo get the BC term. Then replay k_demos
        planner segments and take teacher steps on them. Returns stats
        (zeros if nothing)."""
        stats = {"sil_loss": 0.0, "sil_frac_pos": 0.0, "sil_n": 0,
                 "sil_demo_n": 0, "anchor_n": 0, "teacher_loss": 0.0,
                 "teacher_p": 0.0, "teacher_disagree": 0.0, "teacher_n": 0}
        if self.coef > 0.0:
            self._sil_step(env, buffer, k, rng, max_grad_norm, anchor,
                           anchor_x_lo, stats,
                           k_demos if self.demo_in_sil else 0)
        if self.teacher_coef > 0.0 and k_demos > 0 and buffer.demos:
            self._teacher_step(env, buffer, k_demos, rng, max_grad_norm, stats)
        return stats

    def _sil_step(self, env, buffer, k, rng, max_grad_norm, anchor,
                  anchor_x_lo, stats, k_demos: int = 0):
        episodes = buffer.sample(k, rng) if len(buffer) else []
        demos = buffer.sample_demos(k_demos, rng)
        use_anchor = (anchor is not None and self.anchor_coef > 0
                      and anchor_x_lo > 0 and len(anchor[0]) > 0)
        holds = [h for h, _, _ in episodes] + [d[1] for d in demos]
        blobs = [b for _, b, _ in episodes] + [d[0] for d in demos]
        acts_in = [a for _, _, a in episodes] + [d[4] for d in demos]
        n_main = len(holds)
        if use_anchor:
            holds.append(anchor[0])
            blobs.append(None)
            acts_in.append(None)
        if not holds:
            return
        assert len(holds) <= env.n_envs, "sil env handle too small"
        obs, acts, rews, ticks, n_steps, final_obs, ends, masks = _replay(
            env, holds, blobs, acts_in)
        # Bootstrap for streams that end alive (a from-start episode always
        # ends terminal; a suffix cut by a burst does not).
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
        is_anchor = row_ep >= n_main
        is_demo = (row_ep >= len(episodes)) & (row_ep < n_main)
        # SIL weight 1 on episode rows; anchor rows get the BC term inside
        # the mastered zone only (their tail is the death path).
        sil_w = np.where(is_anchor, 0.0, 1.0).astype(np.float32)
        bc_w = np.zeros(valid.shape, np.float32)
        if use_anchor:
            x = obs[..., 5] * env.end_x
            bc_w = np.where(is_anchor & (x < anchor_x_lo), self.anchor_coef,
                            0.0).astype(np.float32)
        dev = self.device
        t_obs = torch.as_tensor(obs[valid], device=dev)
        t_acts = torch.as_tensor(acts[valid].astype(np.float32), device=dev)
        t_rets = torch.as_tensor(rets[valid], device=dev)
        t_sil_w = torch.as_tensor(sil_w[valid], device=dev)
        t_bc_w = torch.as_tensor(bc_w[valid], device=dev)
        t_masks = (torch.as_tensor(masks[valid].astype(np.float32), device=dev)
                   if self.model.categorical else None)
        n = t_obs.shape[0]
        stats["sil_n"] = n
        stats["anchor_n"] = int((bc_w[valid] > 0).sum())
        demo_rows = torch.as_tensor(np.nonzero(is_demo[valid])[0], device=dev)
        n_demo_mb = (int(self.minibatch * self.demo_share)
                     if demo_rows.numel() else 0)
        if n_demo_mb:
            stats["sil_demo_n"] = int(demo_rows.numel())
        loss_sum = torch.zeros((), device=dev)
        pos_sum = torch.zeros((), device=dev)
        for _ in range(self.steps):
            # Fixed minibatch shape (sampling with replacement, so n <
            # minibatch just repeats rows): a shape that varied with the
            # replayed episode count made cudnn.benchmark re-autotune the
            # convs every update (~2 s each). Demo rows (demo_in_sil) get
            # a fixed share.
            idx = torch.randint(n, (self.minibatch - n_demo_mb,), device=dev)
            if n_demo_mb:
                pick = torch.randint(demo_rows.numel(), (n_demo_mb,), device=dev)
                idx = torch.cat([idx, demo_rows[pick]])
            a = t_acts[idx]
            logit, value = self._logits(t_obs[idx],
                                        t_masks[idx] if t_masks is not None else None)
            logprob, _ = self.model.logprob_entropy(logit, a)
            adv = (t_rets[idx] - value).clamp(min=0.0)
            w = t_sil_w[idx]
            sil_loss = (w * (-(logprob * adv.detach()) + 0.5 * adv ** 2)).mean()
            if self.model.categorical:
                # Hinge on the gap between the anchor action's logit and
                # the best other action.
                own = logit.gather(-1, a.long().unsqueeze(-1)).squeeze(-1)
                other = logit.scatter(-1, a.long().unsqueeze(-1), -1e4).max(-1).values
                signed = own - other
            else:
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

    def _teacher_step(self, env, buffer, k_demos, rng, max_grad_norm, stats):
        demos = buffer.sample_demos(k_demos, rng)
        if not demos:
            return
        assert len(demos) <= env.n_envs, "sil env handle too small"
        obs, acts, _, _, n_steps, _, _, masks = _replay(
            env, [d[1] for d in demos], [d[0] for d in demos],
            [d[4] for d in demos])
        T = obs.shape[1]
        valid = np.arange(T)[None, :] < n_steps[:, None]
        dev = self.device
        t_obs = torch.as_tensor(obs[valid], device=dev)
        t_acts = torch.as_tensor(acts[valid].astype(np.float32), device=dev)
        t_masks = (torch.as_tensor(masks[valid].astype(np.float32), device=dev)
                   if self.model.categorical else None)
        n = t_obs.shape[0]
        stats["sil_demo_n"] = n
        stats["teacher_n"] = n
        # Disagreement set under the current policy (no grad), for the
        # half of the batch aimed at the decisions the policy gets wrong.
        with torch.no_grad():
            logit, _ = self._logits(t_obs, t_masks)
            greedy = self.model.greedy(logit)
            disagree = (greedy != t_acts)
            lp, _ = self.model.logprob_entropy(logit, t_acts)
            stats["teacher_p"] = float(lp.exp().mean())
            stats["teacher_disagree"] = float(disagree.float().mean())
        dis_rows = torch.nonzero(disagree).squeeze(-1)
        half = self.teacher_minibatch // 2
        loss_sum = torch.zeros((), device=dev)
        for _ in range(self.steps):
            idx = torch.randint(n, (self.teacher_minibatch - (half if dis_rows.numel() else 0),),
                                device=dev)
            if dis_rows.numel():
                pick = torch.randint(dis_rows.numel(), (half,), device=dev)
                idx = torch.cat([idx, dis_rows[pick]])
            logit, _ = self._logits(t_obs[idx],
                                    t_masks[idx] if t_masks is not None else None)
            logprob, _ = self.model.logprob_entropy(logit, t_acts[idx])
            # Normalized over its own batch: a plain mean, so the teacher's
            # gradient scale is independent of how many rows it has.
            loss = -self.teacher_coef * logprob.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_grad_norm)
            self.optimizer.step()
            loss_sum += loss.detach()
        stats["teacher_loss"] = loss_sum.item() / self.steps
