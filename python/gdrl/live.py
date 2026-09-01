"""live.bin: seqlock'd snapshot of all env players for the viewer's
TrainingViewMode. Fixed-size mmap'd file; the writer bumps `seq` to odd,
writes the payload, bumps to even. The viewer re-reads if it sees an odd or
changed seq. Same-machine page-cache coherence makes 60 Hz polling free.
`sps` (sim ticks/s) and `samples_s` (decisions/s) are the rates of the last
*completed* update, identical to the trainer's `upd` log line.

Layout (little-endian):
  u32 magic 'GDLV', u32 version=2, u32 seq, u32 n_envs,
  u64 global_ticks, f32 best_x_ever, f32 sps, u32 total_episodes, f32 samples_s
  then n_envs * {f32 x, f32 y, f32 yvel, u8 ship, u8 dead, u16 pad}
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x564C4447  # 'GDLV'
VERSION = 2
HEADER = struct.Struct("<IIIIQffIf")
REC = 20  # f32 x, y, yvel + u8 ship, u8 dead + u16 pad


class LiveWriter:
    def __init__(self, path: str | Path, n_envs: int):
        self.n_envs = n_envs
        size = HEADER.size + n_envs * REC
        self._f = open(path, "w+b")
        self._f.truncate(size)
        self._mm = mmap.mmap(self._f.fileno(), size)
        self._seq = 0
        self._rec = np.zeros((n_envs, REC // 4), np.float32)  # staging

    def write(self, render_states: np.ndarray, global_ticks: int,
              best_x: float, sps: float, episodes: int,
              samples_s: float = 0.0) -> None:
        """render_states: [n, 5] = x, y, yvel, ship, dead from the env."""
        self._seq += 1  # odd: write in progress
        self._mm[8:12] = struct.pack("<I", self._seq)
        rec = self._rec
        rec[:, :3] = render_states[:, :3]
        # pack ship/dead bytes into the fourth f32 slot
        flags = (render_states[:, 3] > 0).astype(np.uint32) | \
                ((render_states[:, 4] > 0).astype(np.uint32) << 8)
        rec[:, 3] = flags.view(np.float32)
        self._mm[HEADER.size:] = rec.tobytes()
        self._seq += 1  # even: consistent
        self._mm[:HEADER.size] = HEADER.pack(
            MAGIC, VERSION, self._seq, self.n_envs, global_ticks,
            float(best_x), float(sps), episodes, float(samples_s))

    def close(self) -> None:
        self._mm.close()
        self._f.close()
