"""ctypes wrapper over gdenv.dll — a vectorized batch of bit-exact GD sims.

The DLL owns all stepping and observation building; Python crosses the
boundary once per decision step with preallocated numpy buffers.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Observation contract this code was written against (obs.hpp kObsVersion).
# A mismatch means a stale gdenv.dll — features would silently differ from
# what checkpoints/models assume.
OBS_VERSION = 4


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
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    d.gdenv_create.restype = C.c_void_p
    d.gdenv_create.argtypes = [C.c_char_p, C.c_int32, C.c_int32, C.c_float,
                               C.c_uint64]
    d.gdenv_destroy.argtypes = [C.c_void_p]
    d.gdenv_last_error.restype = C.c_char_p
    for name in ("gdenv_obs_dim", "gdenv_num_envs", "gdenv_ticks_per_action"):
        getattr(d, name).restype = C.c_int32
        getattr(d, name).argtypes = [C.c_void_p]
    d.gdenv_level_end_x.restype = C.c_float
    d.gdenv_level_end_x.argtypes = [C.c_void_p]
    d.gdenv_reset_all.argtypes = [C.c_void_p, f32p]
    d.gdenv_step.argtypes = [C.c_void_p, u8p, f32p, f32p, f32p, u8p, u8p,
                             f32p, f32p]
    d.gdenv_last_episode_holds.restype = C.c_int32
    d.gdenv_last_episode_holds.argtypes = [C.c_void_p, C.c_int32, u8p,
                                           C.c_int32]
    d.gdenv_get_render_states.argtypes = [C.c_void_p, f32p]
    d.gdenv_state_size.restype = C.c_int32
    d.gdenv_state_size.argtypes = []
    d.gdenv_get_state.argtypes = [C.c_void_p, C.c_int32, u8p]
    d.gdenv_set_state.argtypes = [C.c_void_p, C.c_int32, u8p, C.c_void_p]
    d.gdenv_reset_env.argtypes = [C.c_void_p, C.c_int32, C.c_void_p]
    d.gdenv_replay.restype = C.c_int32
    d.gdenv_replay.argtypes = [C.c_void_p, C.c_int32, u8p, i32p, C.c_int32,
                               C.c_void_p, u8p, f32p, f32p, i32p]
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


class GdVecEnv:
    """n_envs independent Stereo Madness episodes with auto-reset.

    step(actions) -> (obs, rewards, terminated, truncated, info) where info
    holds `final_obs`, `final_x`, `won`, `episode_ticks`, each valid only
    where `terminated | truncated`. All returned arrays are views into
    preallocated buffers — copy them if you keep references across steps.

    pinned=True allocates the obs buffers in page-locked memory so the
    trainer can `buf.copy_(env.obs_tensor, non_blocking=True)` straight from
    the DLL's output (measured: halves the per-step host->device cost).
    """

    def __init__(self, level_path: str | None = None, n_envs: int = 256,
                 ticks_per_action: int = 4, end_x: float = -1.0,
                 pinned: bool = False):
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
        self._env = self._dll.gdenv_create(path, n_envs, ticks_per_action,
                                           end_x, 0)
        if not self._env:
            raise RuntimeError(
                self._dll.gdenv_last_error().decode(errors="replace"))
        self.n_envs = n_envs
        self.obs_dim = self._dll.gdenv_obs_dim(self._env)
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
        # Obs-row pointers for the per-env set_state/reset_env calls (a
        # ctypes cast per call was measurable in the respawn loop).
        base = self._obs.ctypes.data
        self._row_ptr = [C.c_void_p(base + i * od * 4) for i in range(n)]
        self._holds_buf = np.zeros(1 << 16, np.uint8)  # > maxTicks (21600)
        self._info = {
            "ticks": self._dur,
            "final_obs": self._final_obs,
            "final_x": self._final_info[:, 0],
            "won": self._final_info[:, 1],
            "episode_ticks": self._final_info[:, 2],
        }

    def reset(self) -> np.ndarray:
        self._dll.gdenv_reset_all(self._env, self._obs)
        return self._obs

    def step(self, actions: np.ndarray):
        actions = np.ascontiguousarray(actions, np.uint8)
        self._dll.gdenv_step(self._env, actions, self._obs, self._rew,
                             self._dur, self._term, self._trunc,
                             self._final_obs, self._final_info)
        return self._obs, self._rew, self._term, self._trunc, self._info

    def last_episode_holds(self, i: int) -> np.ndarray:
        """Per-tick 0/1 holds of env i's most recently finished episode."""
        buf = self._holds_buf
        n = self._dll.gdenv_last_episode_holds(self._env, i, buf, len(buf))
        if n > len(buf):  # never for a valid episode; keep it correct anyway
            buf = np.zeros(n, np.uint8)
            self._dll.gdenv_last_episode_holds(self._env, i, buf, n)
        return buf[:n].copy()

    def get_render_states(self) -> np.ndarray:
        """[n, 5] = x, y, yvel, ship, dead."""
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
        buffer, so call between step() and consuming that step's obs."""
        self._dll.gdenv_set_state(self._env, i, blob, self._row_ptr[i])

    def replay(self, episodes: list[np.ndarray]):
        """Regenerate the decision streams of recorded episodes (per-tick
        hold bits) on envs 0..k-1 in one DLL call each way (count pass, then
        an exactly-sized fill pass). Returns
        (obs [k, T, obs_dim], acts [k, T] u8, rews [k, T], ticks [k, T],
        n_steps [k]) with rows valid for t < n_steps[i]. The envs end reset;
        the obs buffer for those rows is stale until the next step/reset."""
        k = len(episodes)
        assert 0 < k <= self.n_envs
        holds = np.ascontiguousarray(np.concatenate(
            [np.asarray(e, np.uint8).reshape(-1) for e in episodes]))
        lengths = np.array([len(e) for e in episodes], np.int32)
        n_steps = np.zeros(k, np.int32)
        empty_u8 = np.zeros(1, np.uint8)
        empty_f32 = np.zeros(1, np.float32)
        if self._dll.gdenv_replay(self._env, k, holds, lengths, 0, None,
                                  empty_u8, empty_f32, empty_f32, n_steps):
            raise RuntimeError(
                self._dll.gdenv_last_error().decode(errors="replace"))
        T = int(n_steps.max())
        obs = np.zeros((k, T, self.obs_dim), np.float32)
        acts = np.zeros((k, T), np.uint8)
        rews = np.zeros((k, T), np.float32)
        ticks = np.zeros((k, T), np.float32)
        if self._dll.gdenv_replay(self._env, k, holds, lengths, T,
                                  obs.ctypes.data_as(C.c_void_p), acts, rews,
                                  ticks, n_steps):
            raise RuntimeError(
                self._dll.gdenv_last_error().decode(errors="replace"))
        return obs, acts, rews, ticks, n_steps

    def close(self) -> None:
        if self._env:
            self._dll.gdenv_destroy(self._env)
            self._env = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
