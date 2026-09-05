"""Certificates: the best deterministic run's from-x=0 input record, with an
immutable identity, plus the bookkeeping that ties every stitched episode
to the exact prefix it was cut from.

Stitched training (train.py --stitch) starts episodes and det evals from a
snapshot `stitch_back` units before the certificate's end and reports each
episode as certificate[:prefix_ticks] + the episode's own suffix. That
concatenation is only meaningful while the certificate's first
prefix_ticks ticks are the ones the episode was cut from. A certificate is
replaced whenever the POLICY'S OWN det run gets further; the new one
normally extends the old (same prefix up to the frontier), but a widened
frontier or an imported certificate can replace an earlier section. Every
replacement therefore gets a new id and reports the common prefix length,
and the trainer resets every live episode whose prefix is longer than
that — old prefixes never mix with new suffixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def common_prefix_ticks(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common prefix of two hold sequences."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.nonzero(np.asarray(a[:n], np.uint8) != np.asarray(b[:n], np.uint8))[0]
    return int(diff[0]) if len(diff) else n


@dataclass
class Certificate:
    id: int = 0
    holds: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))
    x: float = 0.0
    # Decision stream of the run that produced the holds (precision mode
    # keeps it so the certificate can be replayed at its own decision
    # boundaries); None in macro mode.
    actions: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.holds)

    def replace(self, holds: np.ndarray, x: float,
                actions: np.ndarray | None = None) -> tuple[int, int]:
        """Install a new certificate. Returns (new id, common prefix ticks
        with the previous certificate). An identical certificate keeps its
        id."""
        holds = np.asarray(holds, np.uint8)
        if len(holds) == len(self.holds) and np.array_equal(holds, self.holds):
            self.x = float(x)
            return self.id, len(self.holds)
        common = common_prefix_ticks(self.holds, holds)
        self.id += 1
        self.holds = holds
        self.x = float(x)
        self.actions = None if actions is None else np.asarray(actions, np.uint8)
        return self.id, common

    def state_dict(self) -> dict:
        return {"id": self.id, "holds": self.holds, "x": self.x,
                "actions": self.actions}

    def load_state_dict(self, d: dict) -> None:
        self.id = int(d.get("id", 0))
        self.holds = np.asarray(d.get("holds", np.zeros(0, np.uint8)), np.uint8)
        self.x = float(d.get("x", 0.0))
        a = d.get("actions")
        self.actions = None if a is None else np.asarray(a, np.uint8)


class Attachments:
    """Per-env (certificate id, prefix ticks) of the live stitched episodes.
    `orphans(new_id, common)` lists envs whose prefix no longer exists after
    a replacement; the trainer resets them before mixing anything."""

    def __init__(self, n: int):
        self.cert_id = np.zeros(n, np.int64)
        self.prefix_ticks = np.zeros(n, np.int64)
        self.stitched = np.zeros(n, bool)  # started from a certificate snapshot

    def attach(self, i: int, cert_id: int, prefix_ticks: int) -> None:
        self.cert_id[i] = cert_id
        self.prefix_ticks[i] = prefix_ticks
        self.stitched[i] = prefix_ticks > 0

    def detach(self, i: int) -> None:
        self.cert_id[i] = -1
        self.prefix_ticks[i] = 0
        self.stitched[i] = False

    def orphans(self, new_id: int, common: int) -> np.ndarray:
        """Indices of stitched envs whose prefix exceeds the common prefix
        of the replaced certificate (they cannot attach to `new_id`)."""
        return np.nonzero(self.stitched & (self.cert_id != new_id)
                          & (self.prefix_ticks > common))[0]

    def reattach(self, new_id: int, common: int) -> np.ndarray:
        """Move every compatible stitched env onto the new certificate and
        return the orphans (left as they are for the caller to reset)."""
        orphans = self.orphans(new_id, common)
        ok = self.stitched & (self.prefix_ticks <= common)
        self.cert_id[ok] = new_id
        return orphans


def widen_schedule(base: float, stalls: int, patience: int = 3,
                   cap: float = 4800.0) -> float:
    """Stitch-back overlap after `stalls` consecutive det evals without
    progress: base (600) for the first `patience` stalls, then doubling
    every `patience` more (1200, 2400, 4800), capped. Verified progress
    resets stalls to 0."""
    steps = stalls // patience
    return float(min(base * (2.0 ** steps), cap))
