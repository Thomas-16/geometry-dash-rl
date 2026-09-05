"""ctypes wrapper over gdenv.dll — a vectorized batch of bit-exact GD sims.

The DLL owns all stepping and observation building; Python crosses the
boundary once per decision step with preallocated numpy buffers.

Contracts (src/env/contracts.hpp): an env handle is created for one
observation layout (`obs_version` 5 legacy / 6 player-centered + geometry)
and one action space (`action_mode` "macro" = 2 actions / "precision" = 7).
Every checkpoint records both plus the level and simulator fingerprints
(`GdVecEnv.contracts()`), and the trainer / evaluator refuse a mismatch.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Legacy observation contract (obs.hpp kObsVersion). A mismatch means a stale
# gdenv.dll — features would silently differ from what checkpoints assume.
OBS_VERSION = 5
OBS_VERSION_V6 = 6
ACTION_MODES = {"macro": 0, "precision": 1}
ACTION_MODE_NAMES = {v: k for k, v in ACTION_MODES.items()}
# Precision action ids (episode.hpp PrecisionAction).
ACT_R1, ACT_R2, ACT_R4, ACT_H1, ACT_H2, ACT_H4, ACT_HOP = range(7)
PRECISION_ACTION_NAMES = ("R1", "R2", "R4", "H1", "H2", "H4", "HOP")
SOLVE_STATUS = {0: "found", 1: "exhausted", 2: "node_budget", 3: "memory_budget",
                4: "verify_failed", 5: "pruned_out"}


def _find_dll() -> Path:
    override = os.environ.get("GDENV_DLL")
    if override:
        return Path(override)
    p = _REPO_ROOT / "build" / "gdenv.dll"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — build it with: cmake --build build --target gdenv"
        )
    return p


def _load(dll_path: Path) -> C.CDLL:
    d = C.CDLL(str(dll_path))
    f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    u16p = np.ctypeslib.ndpointer(np.uint16, flags="C_CONTIGUOUS")
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    d.gdenv_create.restype = C.c_void_p
    d.gdenv_create.argtypes = [C.c_char_p, C.c_int32, C.c_int32, C.c_float,
                               C.c_uint64, C.c_int32, C.c_int32, C.c_double]
    d.gdenv_destroy.argtypes = [C.c_void_p]
    d.gdenv_last_error.restype = C.c_char_p
    for name in ("gdenv_obs_dim", "gdenv_num_envs", "gdenv_ticks_per_action",
                 "gdenv_obs_version_of", "gdenv_action_mode",
                 "gdenv_num_actions"):
        getattr(d, name).restype = C.c_int32
        getattr(d, name).argtypes = [C.c_void_p]
    d.gdenv_obs_version.restype = C.c_int32
    d.gdenv_obs_version.argtypes = []
    d.gdenv_contract_versions.argtypes = [i32p]
    for name in ("gdenv_level_fingerprint", "gdenv_sim_fingerprint"):
        getattr(d, name).restype = C.c_uint64
        getattr(d, name).argtypes = [C.c_void_p]
    d.gdenv_level_end_x.restype = C.c_float
    d.gdenv_level_end_x.argtypes = [C.c_void_p]
    d.gdenv_reset_all.argtypes = [C.c_void_p, f32p]
    d.gdenv_action_masks.argtypes = [C.c_void_p, u8p]
    d.gdenv_step.argtypes = [C.c_void_p, u8p, f32p, f32p, f32p, u8p, u8p,
                             f32p, f32p]
    for name in ("gdenv_last_episode_holds", "gdenv_last_episode_actions"):
        getattr(d, name).restype = C.c_int32
        getattr(d, name).argtypes = [C.c_void_p, C.c_int32, u8p, C.c_int32]
    d.gdenv_get_render_states.argtypes = [C.c_void_p, f32p]
    d.gdenv_state_size.restype = C.c_int32
    d.gdenv_state_size.argtypes = []
    d.gdenv_get_state.argtypes = [C.c_void_p, C.c_int32, u8p]
    d.gdenv_set_state.restype = C.c_int32
    d.gdenv_set_state.argtypes = [C.c_void_p, C.c_int32, u8p, C.c_void_p]
    d.gdenv_reset_env.argtypes = [C.c_void_p, C.c_int32, C.c_void_p]
    d.gdenv_state_hash.restype = C.c_uint64
    d.gdenv_state_hash.argtypes = [u8p]
    for name in ("gdenv_replay", "gdenv_replay_actions"):
        getattr(d, name).restype = C.c_int32
        getattr(d, name).argtypes = [
            C.c_void_p, C.c_int32, u8p, i32p, C.c_int32, C.c_void_p, u8p,
            f32p, f32p, i32p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
    d.gdenv_solve.restype = C.c_int32
    d.gdenv_solve.argtypes = [C.c_void_p, C.c_int32, u8p, f32p, C.c_int32,
                              C.c_int32, u8p, C.c_int32, i32p, u8p, u16p,
                              C.c_int32, i32p, i32p, f32p]
    d.gdenv_trace_states.restype = C.c_int32
    d.gdenv_trace_states.argtypes = [C.c_void_p, u8p, C.c_int32, C.c_int32,
                                     f32p, u8p, u8p]
    d.gdenv_run_holds.restype = C.c_int32
    d.gdenv_run_holds.argtypes = [C.c_void_p, u8p, C.c_int32, f32p]
    return d


def _pinned_f32(shape) -> tuple[np.ndarray, object]:
    """Page-locked float32 buffer: (numpy view for the DLL, torch view for
    non_blocking host->device copies). Falls back to plain numpy when torch
    has no CUDA (pinning needs a CUDA context)."""
    import torch
    if not torch.cuda.is_available():
        return np.zeros(shape, np.float32), None
    t = torch.zeros(*shape, dtype=torch.float32, pin_memory=True)
    return t.numpy(), t


def action_mode_id(mode) -> int:
    if isinstance(mode, str):
        return ACTION_MODES[mode]
    return int(mode)


class GdVecEnv:
    """n_envs independent episodes with auto-reset.

    step(actions) -> (obs, rewards, terminated, truncated, info) where info
    holds `final_obs`, `final_x`, `won`, `episode_ticks`, each valid only
    where `terminated | truncated`. All returned arrays are views into
    preallocated buffers — copy them if you keep references across steps.

    pinned=True allocates the obs buffers in page-locked memory so the
    trainer can `buf.copy_(env.obs_tensor, non_blocking=True)` straight from
    the DLL's output (measured: halves the per-step host->device cost).

    obs_version / action_mode / reward_gamma select the contracts (module
    docstring). reward_gamma > 0 makes each step's reward the discounted
    per-tick sum inside the step (precision-mode contract); 0 = flat.
    """

    def __init__(self, level_path: str | None = None, n_envs: int = 256,
                 ticks_per_action: int = 4, end_x: float = -1.0,
                 pinned: bool = False, obs_version: int = OBS_VERSION,
                 action_mode="macro", reward_gamma: float = 0.0):
        self._dll = _load(_find_dll())
        try:
            v = int(self._dll.gdenv_obs_version())
        except AttributeError:
            v = 1  # DLL predates the version export
        if v != OBS_VERSION:
            raise RuntimeError(
                f"gdenv.dll obs version {v} != expected {OBS_VERSION} — "
                f"rebuild the DLL (cmake --build build) or update gdrl")
        path = level_path.encode() if level_path else None
        self.level_path = level_path
        self._env = self._dll.gdenv_create(path, n_envs, ticks_per_action,
                                           end_x, 0, int(obs_version),
                                           action_mode_id(action_mode),
                                           float(reward_gamma))
        if not self._env:
            raise RuntimeError(
                self._dll.gdenv_last_error().decode(errors="replace"))
        self.n_envs = n_envs
        self.obs_dim = self._dll.gdenv_obs_dim(self._env)
        self.obs_version = int(self._dll.gdenv_obs_version_of(self._env))
        self.action_mode = ACTION_MODE_NAMES[
            int(self._dll.gdenv_action_mode(self._env))]
        self.n_actions = int(self._dll.gdenv_num_actions(self._env))
        self.reward_gamma = float(reward_gamma)
        self.end_x = self._dll.gdenv_level_end_x(self._env)
        self.ticks_per_action = self._dll.gdenv_ticks_per_action(self._env)
        self.state_size = int(self._dll.gdenv_state_size())
        n, od = n_envs, self.obs_dim
        if pinned:
            self._obs, self.obs_tensor = _pinned_f32((n, od))
            self._final_obs, self.final_obs_tensor = _pinned_f32((n, od))
        else:
            self._obs = np.zeros((n, od), np.float32)
            self._final_obs = np.zeros((n, od), np.float32)
            self.obs_tensor = self.final_obs_tensor = None
        self._rew = np.zeros(n, np.float32)
        self._dur = np.zeros(n, np.float32)
        self._term = np.zeros(n, np.uint8)
        self._trunc = np.zeros(n, np.uint8)
        self._final_info = np.zeros((n, 3), np.float32)
        self._render = np.zeros((n, 5), np.float32)
        self._masks = np.ones((n, self.n_actions), np.uint8)
        # Obs-row pointers for the per-env set_state/reset_env calls (a
        # ctypes cast per call was measurable in the respawn loop).
        base = self._obs.ctypes.data
        self._row_ptr = [C.c_void_p(base + i * od * 4) for i in range(n)]
        self._holds_buf = np.zeros(1 << 16, np.uint8)  # > maxTicks (30000)
        self._info = {
            "ticks": self._dur,
            "final_obs": self._final_obs,
            "final_x": self._final_info[:, 0],
            "won": self._final_info[:, 1],
            "episode_ticks": self._final_info[:, 2],
        }

    # ---------------------------------------------------------------- meta
    def contracts(self) -> dict:
        """Everything a checkpoint records to be refused on mismatch."""
        v = np.zeros(6, np.int32)
        self._dll.gdenv_contract_versions(v)
        return {
            "obs_version": self.obs_version,
            "action_mode": self.action_mode,
            "n_actions": self.n_actions,
            "reward_gamma": self.reward_gamma,
            "action_version": int(v[2]),
            "trajectory_version": int(v[3]),
            "snapshot_version": int(v[4]),
            "sim_version": int(v[5]),
            "level_fingerprint": f"{int(self._dll.gdenv_level_fingerprint(self._env)):016x}",
            "sim_fingerprint": f"{int(self._dll.gdenv_sim_fingerprint(self._env)):016x}",
        }

    @property
    def precision(self) -> bool:
        return self.action_mode == "precision"

    # ---------------------------------------------------------------- core
    def reset(self) -> np.ndarray:
        self._dll.gdenv_reset_all(self._env, self._obs)
        return self._obs

    def step(self, actions: np.ndarray):
        actions = np.ascontiguousarray(actions, np.uint8)
        self._dll.gdenv_step(self._env, actions, self._obs, self._rew,
                             self._dur, self._term, self._trunc,
                             self._final_obs, self._final_info)
        return self._obs, self._rew, self._term, self._trunc, self._info

    def action_masks(self) -> np.ndarray:
        """[n, n_actions] u8 legality for the CURRENT states (after the last
        step / reset / set_state). A view into a reused buffer."""
        self._dll.gdenv_action_masks(self._env, self._masks)
        return self._masks

    def last_episode_holds(self, i: int) -> np.ndarray:
        """Per-tick 0/1 holds of env i's most recently finished episode."""
        buf = self._holds_buf
        n = self._dll.gdenv_last_episode_holds(self._env, i, buf, len(buf))
        if n > len(buf):  # never for a valid episode; keep it correct anyway
            buf = np.zeros(n, np.uint8)
            self._dll.gdenv_last_episode_holds(self._env, i, buf, n)
        return buf[:n].copy()

    def last_episode_actions(self, i: int) -> np.ndarray:
        """Per-decision action ids of env i's most recently finished
        episode (the decision stream; replay it with replay_actions)."""
        buf = self._holds_buf
        n = self._dll.gdenv_last_episode_actions(self._env, i, buf, len(buf))
        if n > len(buf):
            buf = np.zeros(n, np.uint8)
            self._dll.gdenv_last_episode_actions(self._env, i, buf, n)
        return buf[:n].copy()

    def player_x(self) -> np.ndarray:
        """Per-env player x as of the last step/reset — read straight from
        the obs buffer's progress scalar (obs.hpp out[5] = x / end_x, in
        both layouts), so it costs nothing and needs no DLL call."""
        return self._obs[:, 5] * self.end_x

    def get_render_states(self) -> np.ndarray:
        """[n, 5] = x, y, yvel, mode, dead."""
        self._dll.gdenv_get_render_states(self._env, self._render)
        return self._render

    def reset_env(self, i: int) -> None:
        """Reset env i to x=0; refreshes its row in the obs buffer."""
        self._dll.gdenv_reset_env(self._env, i, self._row_ptr[i])

    def get_state(self, i: int) -> np.ndarray:
        buf = np.empty(self.state_size, np.uint8)
        self._dll.gdenv_get_state(self._env, i, buf)
        return buf

    def set_state(self, i: int, blob: np.ndarray) -> None:
        """Restore env i from a snapshot; refreshes its row in the obs
        buffer, so call between step() and consuming that step's obs.
        Raises on a foreign blob (older layout / other sim version)."""
        blob = np.ascontiguousarray(blob, np.uint8)
        if len(blob) != self.state_size:
            raise ValueError(f"snapshot blob is {len(blob)} bytes, this DLL "
                             f"uses {self.state_size} (see migrate_blob)")
        if self._dll.gdenv_set_state(self._env, i, blob, self._row_ptr[i]):
            raise RuntimeError(
                self._dll.gdenv_last_error().decode(errors="replace"))

    def state_hash(self, blob: np.ndarray) -> int:
        """Exact identity hash of a snapshot (the planner's dedup key)."""
        return int(self._dll.gdenv_state_hash(
            np.ascontiguousarray(blob, np.uint8)))

    _fresh: np.ndarray | None = None

    def fresh_blob(self) -> np.ndarray:
        """State blob of a fresh x=0 episode (cached)."""
        if self._fresh is None:
            tmp = np.empty(self.state_size, np.uint8)
            # Any env's reset state is the same; borrow the last one's
            # snapshot without disturbing it: reset_env writes the row obs.
            i = self.n_envs - 1
            keep = self.get_state(i)
            self.reset_env(i)
            self._dll.gdenv_get_state(self._env, i, tmp)
            self.set_state(i, keep)
            self._fresh = tmp
        return self._fresh

    def migrate_blob(self, blob: np.ndarray) -> np.ndarray | None:
        """Bring a snapshot from a checkpoint into this DLL's blob layout.
        Blobs of the current size pass through; a pre-contract blob
        (PlayerState + tick, no header) gets the header of a fresh blob
        prepended — valid because the sim version has not changed since
        those runs. Returns None when the blob cannot be migrated."""
        blob = np.asarray(blob, np.uint8)
        if len(blob) == self.state_size:
            return blob
        header = 16
        if len(blob) == self.state_size - header:
            # old layout: PlayerState then int32 tick (+ alignment pad),
            # exactly the new blob minus its 16-byte header
            fresh = self.fresh_blob()
            return np.concatenate([fresh[:header], blob])
        return None

    # -------------------------------------------------------------- replay
    def _replay(self, fn, seqs, start_blobs, want_masks: bool):
        k = len(seqs)
        assert 0 < k <= self.n_envs
        data = np.ascontiguousarray(np.concatenate(
            [np.asarray(e, np.uint8).reshape(-1) for e in seqs]))
        lengths = np.array([len(e) for e in seqs], np.int32)
        n_steps = np.zeros(k, np.int32)
        blobs_ptr = None
        if start_blobs is not None and any(b is not None for b in start_blobs):
            blobs = np.zeros((k, self.state_size), np.uint8)
            for i, b in enumerate(start_blobs):
                blobs[i] = b if b is not None else self.fresh_blob()
            blobs_ptr = blobs.ctypes.data_as(C.c_void_p)
        empty_u8 = np.zeros(1, np.uint8)
        empty_f32 = np.zeros(1, np.float32)
        if fn(self._env, k, data, lengths, 0, None, empty_u8, empty_f32,
              empty_f32, n_steps, blobs_ptr, None, None, None):
            raise RuntimeError(
                self._dll.gdenv_last_error().decode(errors="replace"))
        T = int(n_steps.max())
        obs = np.zeros((k, T, self.obs_dim), np.float32)
        acts = np.zeros((k, T), np.uint8)
        rews = np.zeros((k, T), np.float32)
        ticks = np.zeros((k, T), np.float32)
        final_obs = np.zeros((k, self.obs_dim), np.float32)
        ends = np.zeros(k, np.uint8)
        masks = np.ones((k, T, self.n_actions), np.uint8)
        if fn(self._env, k, data, lengths, T,
              obs.ctypes.data_as(C.c_void_p), acts, rews, ticks, n_steps,
              blobs_ptr, final_obs.ctypes.data_as(C.c_void_p),
              ends.ctypes.data_as(C.c_void_p),
              masks.ctypes.data_as(C.c_void_p) if want_masks else None):
            raise RuntimeError(
                self._dll.gdenv_last_error().decode(errors="replace"))
        return obs, acts, rews, ticks, n_steps, final_obs, ends, masks

    def replay(self, episodes: list[np.ndarray],
               start_blobs: list[np.ndarray | None] | None = None):
        """Regenerate the decision streams of recorded episodes (per-tick
        hold bits) on envs 0..k-1 in one DLL call each way (count pass, then
        an exactly-sized fill pass). start_blobs[i] (a get_state blob, or
        None for x=0) makes episode i a segment replayed from that state.
        Returns (obs [k, T, obs_dim], acts [k, T] u8, rews [k, T],
        ticks [k, T], n_steps [k], final_obs [k, obs_dim], ends [k] u8)
        with rows valid for t < n_steps[i]; ends: 0 = holds exhausted alive
        (bootstrap V(final_obs)), 1 = terminated, 2 = tick cap. The envs end
        reset; the obs buffer for those rows is stale until the next
        step/reset. In precision mode raw holds replay as one-tick actions
        (a bit-exact trajectory, but not the original decision stream —
        use replay_actions for that)."""
        return self._replay(self._dll.gdenv_replay, episodes, start_blobs,
                            False)[:7]

    def replay_actions(self, streams: list[np.ndarray],
                       start_blobs: list[np.ndarray | None] | None = None):
        """Like replay, for explicit decision streams (per-decision action
        ids). Returns (obs, acts, rews, ticks, n_steps, final_obs, ends,
        masks [k, T, n_actions] u8)."""
        return self._replay(self._dll.gdenv_replay_actions, streams,
                            start_blobs, True)

    def solve(self, start_blobs: list[np.ndarray], target_x,
              max_nodes: int = 0, max_states: int = 0, cap: int = 1 << 15,
              cap_acts: int = 1 << 14) -> list[dict]:
        """Exact BFS in this handle's action space from each start blob to
        x >= target_x (a scalar or per-start array) on the DLL's thread
        pool. Returns one dict per start: holds (per-tick u8 or None),
        actions (u8 ids or None), durations (u16 or None), status (name),
        reach, nodes, stored, pruned, stage (precision mode: 1 = found in
        the macro space and translated, 2 = found in the precision space).
        A found path was replayed through the executor inside the DLL
        before being returned."""
        k = len(start_blobs)
        blobs = np.ascontiguousarray(np.stack(start_blobs).astype(np.uint8))
        tx = np.ascontiguousarray(
            np.broadcast_to(np.asarray(target_x, np.float32), (k,)).copy())
        out = np.zeros((k, cap), np.uint8)
        lens = np.zeros(k, np.int32)
        acts = np.zeros((k, cap_acts), np.uint8)
        durs = np.zeros((k, cap_acts), np.uint16)
        n_acts = np.zeros(k, np.int32)
        status = np.zeros(k, np.int32)
        stats = np.zeros((k, 5), np.float32)
        if self._dll.gdenv_solve(self._env, k, blobs, tx, max_nodes, max_states,
                                 out, cap, lens, acts, durs, cap_acts, n_acts,
                                 status, stats):
            raise RuntimeError(
                self._dll.gdenv_last_error().decode(errors="replace"))
        res = []
        for i in range(k):
            n = int(lens[i])
            res.append({
                "holds": out[i, :n].copy() if n > 0 else None,
                "actions": acts[i, :n_acts[i]].copy() if n > 0 else None,
                "durations": durs[i, :n_acts[i]].copy() if n > 0 else None,
                "status": SOLVE_STATUS.get(int(status[i]), str(status[i])),
                "reach": float(stats[i, 0]),
                "nodes": int(stats[i, 1]),
                "stored": int(stats[i, 2]),
                "pruned": int(stats[i, 3]),
                "stage": int(stats[i, 4]),
            })
        return res

    def trace_states(self, holds: np.ndarray, xs) -> list[np.ndarray | None]:
        """State blobs along a from-start hold sequence at the first decision
        boundary with x >= xs[j] (xs ascending); None where the episode ended
        first. Uses env slot 0 (left reset)."""
        h = np.ascontiguousarray(np.asarray(holds, np.uint8).reshape(-1))
        xs = np.ascontiguousarray(np.asarray(xs, np.float32).reshape(-1))
        n = len(xs)
        blobs = np.zeros((n, self.state_size), np.uint8)
        found = np.zeros(n, np.uint8)
        if self._dll.gdenv_trace_states(self._env, h, len(h), n, xs, blobs,
                                        found):
            raise RuntimeError(
                self._dll.gdenv_last_error().decode(errors="replace"))
        return [blobs[j].copy() if found[j] else None for j in range(n)]

    def run_holds(self, holds: np.ndarray) -> tuple[float, bool, int]:
        """Raw per-tick verification of a from-x=0 hold sequence (what the
        replay mod injects, no decision boundaries): (final_x, won,
        ticks)."""
        h = np.ascontiguousarray(np.asarray(holds, np.uint8).reshape(-1))
        out = np.zeros(3, np.float32)
        self._dll.gdenv_run_holds(self._env, h, len(h), out)
        return float(out[0]), bool(out[1] > 0.5), int(out[2])

    def close(self) -> None:
        if self._env:
            self._dll.gdenv_destroy(self._env)
            self._env = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
