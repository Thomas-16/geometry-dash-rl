"""Ship a solved input sequence to the real game (fang.rl-replay Geode mod).

  python -m gdrl.deploy runs/<name>            # verify in the sim, then install
  python -m gdrl.deploy runs/<name> --no-verify
  python -m gdrl.deploy --check-trace <attempt_NNNN.csv> runs/<name>

Verification re-steps the raw per-tick holds through gdcore via
build/env_check.exe (no macro-action layer — exactly what the mod injects) and
requires a WIN. Install copies solution.json to
<solutions-dir>/<level_id>.json, which the mod loads on level reset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MOD_ID = "fang.rl-replay"
REPO = Path(__file__).resolve().parents[2]


def default_dest() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        sys.exit("LOCALAPPDATA not set; pass --dest")
    return Path(base) / "GeometryDash" / "geode" / "mods" / MOD_ID / "solutions"


def load_solution(run: Path) -> tuple[Path, dict]:
    path = run if run.suffix == ".json" else run / "solution.json"
    if not path.exists():
        sys.exit(f"no solution at {path} — run python -m gdrl.eval first")
    sol = json.loads(path.read_text())
    if sol.get("tps", 240) != 240:
        sys.exit(f"tps {sol['tps']} != 240")
    return path, sol


def expand(sol: dict):
    n = int(sol["n_ticks"])
    bits = bytearray(n)
    for a, b in sol["holds"]:
        if not (0 <= a < b <= n):
            sys.exit(f"bad hold range [{a},{b}) for n_ticks {n}")
        for t in range(a, b):
            bits[t] = 1
    return bits


def verify(sol: dict, level_path: str | None) -> str:
    exe = REPO / "build" / "env_check.exe"
    if not exe.exists():
        sys.exit(f"{exe} missing — build the sim first (cmake --build build)")
    # env_check ranges are INCLUSIVE; solution ranges are [start, end).
    # Both index sim ticks, so no shift here.
    spec = ",".join(f"{a}-{b - 1}" if b - 1 > a else f"{a}"
                    for a, b in sol["holds"])
    env = dict(os.environ)
    if level_path:
        env["GD_LEVEL_PATH"] = level_path
    out = subprocess.run([str(exe), "run", spec], capture_output=True,
                         text=True, env=env, cwd=REPO).stdout.strip()
    print(f"sim: {out}")
    if not out.startswith("WIN"):
        sys.exit("solution does not win in the sim (full level end) - refusing to deploy")
    return out


def check_trace(csv_path: Path, sol: dict) -> None:
    """Assert the mod's recorded trace holds == the solution bits (proves
    human input was blocked and injection landed on the right ticks).
    Real tick t carries sim hold t-1 (sim step 0 already moves the player;
    the real game's tick 0 is a pre-movement input tick)."""
    bits = expand(sol)
    with open(csv_path, newline="") as f:
        fmt = f.readline().strip()
        assert fmt == "format=3", fmt
        rows = [r for r in csv.DictReader(f) if r["src"] == "0"]
    bad = []
    for r in rows:
        t = int(r["tick"])
        if t - 1 >= len(bits):
            break
        want = bits[t - 1] if t >= 1 else 0
        if int(r["holding"]) != want:
            bad.append(t)
    print(f"trace {csv_path.name}: {len(rows)} ticks, solution {len(bits)} ticks, "
          f"{len(bad)} hold mismatches" + (f" (first at tick {bad[0]})" if bad else ""))
    last = rows[-1]
    print(f"last row: tick {last['tick']} x={last['realx']} dead={last['dead']}")
    if bad:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", help="runs/<name> or a solution.json path")
    p.add_argument("--dest", type=Path, default=None,
                   help="solutions dir (default: the mod's save dir)")
    p.add_argument("--level", default=None,
                   help="level file for sim verification (default SM)")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--check-trace", type=Path, default=None,
                   help="compare a mod-recorded attempt_NNNN.csv against the "
                        "solution instead of deploying")
    args = p.parse_args()

    src, sol = load_solution(Path(args.run))
    level_id = int(sol.get("level_id", 1))
    print(f"{src}: level {level_id}, {sol['n_ticks']} ticks, "
          f"{len(sol['holds'])} hold ranges, det={sol.get('deterministic_policy')}")

    if args.check_trace:
        check_trace(args.check_trace, sol)
        return

    if not args.no_verify:
        verify(sol, args.level)

    dest = (args.dest or default_dest())
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{level_id}.json"
    shutil.copyfile(src, out)
    print(f"installed -> {out}")
    print("in game: disable click-between-frames / physics bypass, open the level "
          "in normal mode (no start pos), press play. F8 toggles the replayer.")
    if args.dest is None:
        print(f"results land in {dest.parent / 'traces'}")


if __name__ == "__main__":
    main()
