"""Bit-determinism selftest: the DLL must produce byte-identical streams at
any thread count. Runs the same scripted episode batch (including set_state
round trips) under GDENV_THREADS=1 and GDENV_THREADS=16 in separate
subprocesses and compares stream hashes.

Usage (from repo root):  python -m gdrl.selftest
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

N_ENVS = 64
STEPS = 600


def _child() -> str:
    import numpy as np

    from .env import GdVecEnv

    env = GdVecEnv(None, N_ENVS)
    rng = np.random.default_rng(1234)
    h = hashlib.sha256()
    obs = env.reset()
    h.update(obs.tobytes())
    snapshots: list[np.ndarray] = []
    for step in range(STEPS):
        acts = (rng.random(N_ENVS) < 0.4).astype(np.uint8)
        obs, rew, term, trunc, info = env.step(acts)
        # Snapshot/restore round trips exercise set_state under load.
        if step % 40 == 20:
            snapshots.append(env.get_state(int(rng.integers(N_ENVS))))
        if snapshots and step % 40 == 39:
            env.set_state(int(rng.integers(N_ENVS)),
                          snapshots[int(rng.integers(len(snapshots)))])
        for buf in (obs, rew, term, trunc, info["ticks"], info["final_obs"],
                    info["final_x"], info["won"], info["episode_ticks"]):
            h.update(np.ascontiguousarray(buf).tobytes())
    # Batched replay (gdenv_replay) of the last finished episodes must be
    # bit-identical across thread counts too.
    eps = [env.last_episode_holds(i) for i in range(8)]
    eps = [e for e in eps if len(e)] or [np.ones(64, np.uint8)]
    for buf in env.replay(eps):
        h.update(np.ascontiguousarray(buf).tobytes())
    env.close()
    return h.hexdigest()


def main() -> int:
    if "--child" in sys.argv:
        print(_child())
        return 0
    digests = {}
    for threads in ("1", "16"):
        child_env = {**os.environ, "GDENV_THREADS": threads}
        out = subprocess.run(
            [sys.executable, "-m", "gdrl.selftest", "--child"],
            capture_output=True, text=True, env=child_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if out.returncode != 0:
            print(out.stdout, out.stderr, sep="\n")
            print(f"SELFTEST FAIL: child exited {out.returncode} "
                  f"(GDENV_THREADS={threads})")
            return 1
        digests[threads] = out.stdout.strip().splitlines()[-1]
        print(f"GDENV_THREADS={threads}: {digests[threads]}")
    if digests["1"] != digests["16"]:
        print("SELFTEST FAIL: serial and threaded streams differ")
        return 1
    print(f"selftest OK: {N_ENVS} envs x {STEPS} steps bit-identical "
          f"across thread counts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
