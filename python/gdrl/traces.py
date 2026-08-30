""".gdtrace: episodes as per-tick hold bits.

The sim is deterministic, so hold bits fully define an episode; the viewer
re-simulates them for exact playback. Layout (little-endian):

  header: u32 magic 'GDTR', u32 version=1, u32 tps=240, u32 n_episodes
  per episode: u32 n_ticks, f32 final_x, u8 won, u8 pad[3],
               holds bitpacked LSB-first, ceil(n_ticks/8) bytes
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x52544447  # 'GDTR' little-endian
VERSION = 1
TPS = 240


@dataclass
class Episode:
    holds: np.ndarray  # [n_ticks] u8 of 0/1, one per physics tick
    final_x: float
    won: bool


def write(path: str | Path, episodes: list[Episode]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, VERSION, TPS, len(episodes)))
        for ep in episodes:
            holds = np.asarray(ep.holds, np.uint8)
            f.write(struct.pack("<IfBxxx", len(holds), float(ep.final_x),
                                int(ep.won)))
            f.write(np.packbits(holds, bitorder="little").tobytes())


def read(path: str | Path) -> list[Episode]:
    data = Path(path).read_bytes()
    magic, version, tps, n = struct.unpack_from("<IIII", data, 0)
    if magic != MAGIC or version != VERSION or tps != TPS:
        raise ValueError(f"bad gdtrace header in {path}")
    off = 16
    episodes = []
    for _ in range(n):
        n_ticks, final_x, won = struct.unpack_from("<IfB", data, off)
        off += 12
        nbytes = (n_ticks + 7) // 8
        bits = np.frombuffer(data, np.uint8, nbytes, off)
        off += nbytes
        holds = np.unpackbits(bits, bitorder="little")[:n_ticks]
        episodes.append(Episode(holds, final_x, bool(won)))
    return episodes
