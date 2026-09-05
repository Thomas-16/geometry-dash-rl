"""Equal-budget corridor comparison (docs/rl-improvement-plan.md, validation
step 4): the recorded Theory of Everything UFO spike-column corridor
(x 16950-18570, where level12-1 / level12-2 stalled for 1300 updates),
entered from the level12-2 certificate cut at x 16400, win line 18700.

Four configurations x seeds, each a fresh train.py run under the same
wall-clock budget, stopping at the first verified det completion:

  baseline   obs 5, macro actions, demos through SIL, fixed overlap
  obs        obs 6, macro actions, demos through SIL, fixed overlap
  obsprec    obs 6, precision actions, demos through SIL, fixed overlap
  full       obs 6, precision actions, teacher loss, widening overlap

  python -m gdrl.corridor --seeds 0 1 2 --budget-secs 300
  python -m gdrl.corridor --configs full --seeds 0 --budget-secs 90  # smoke

Writes runs/corridor/<config>-s<seed>/ and runs/corridor/results.json,
and prints the table (first verified WIN time, best det x, certificate x).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEVEL12 = "S:/SteamLibrary/steamapps/common/Geometry Dash/Resources/levels/12.txt"

CONFIGS = {
    "baseline": ["--obs-version", "5", "--action-mode", "macro",
                 "--teacher-coef", "0", "--demo-sil", "1", "--stitch-widen", "0"],
    "obs": ["--obs-version", "6", "--action-mode", "macro",
            "--teacher-coef", "0", "--demo-sil", "1", "--stitch-widen", "0"],
    "obsprec": ["--obs-version", "6", "--action-mode", "precision",
                "--teacher-coef", "0", "--demo-sil", "1", "--stitch-widen", "0"],
    "full": ["--obs-version", "6", "--action-mode", "precision",
             "--teacher-coef", "0.25", "--demo-sil", "0", "--stitch-widen", "1"],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=list(CONFIGS),
                   choices=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--budget-secs", type=float, default=300.0)
    p.add_argument("--level", default=LEVEL12)
    p.add_argument("--certificate", default=str(REPO / "runs" / "level12-2" / "solution.json"))
    p.add_argument("--cut-x", type=float, default=16400.0)
    p.add_argument("--end-x", type=float, default=18700.0)
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--updates", type=int, default=3000,
                   help="anneal span (never reached; the budget stops runs)")
    p.add_argument("--extra", default="",
                   help="extra train.py flags for every run, as one string "
                        "(e.g. --extra=\"--wall-solve 0\")")
    p.add_argument("--prefix", default="corridor")
    args = p.parse_args()

    out_dir = REPO / "runs" / args.prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for cfg in args.configs:
        for seed in args.seeds:
            name = f"{args.prefix}/{cfg}-s{seed}"
            if name in results and results[name].get("done"):
                print(f"skip {name} (done)")
                continue
            cmd = [sys.executable, "-u", "-m", "gdrl.train", "--run-name", name,
                   "--level", args.level, "--n-envs", str(args.n_envs),
                   "--updates", str(args.updates), "--seed", str(seed),
                   "--import-certificate", args.certificate,
                   "--import-cut-x", str(args.cut_x), "--end-x", str(args.end_x),
                   "--stop-after-secs", str(args.budget_secs), "--stop-on-win",
                   "--det-every", "5", "--solve-stall", "1",
                   *CONFIGS[cfg], *shlex.split(args.extra)]
            print(f"=== {name}: {' '.join(cmd[3:])}", flush=True)
            t0 = time.time()
            log = out_dir / f"{cfg}-s{seed}.log"
            with open(log, "w", encoding="utf-8") as f:
                rc = subprocess.run(cmd, cwd=REPO / "python", stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            summary = {}
            sp = REPO / "runs" / name / "summary.json"
            if sp.exists():
                summary = json.loads(sp.read_text())
            results[name] = {"config": cfg, "seed": seed, "rc": rc,
                             "secs": time.time() - t0, "done": rc == 0,
                             **{k: summary.get(k) for k in
                                ("updates", "best_det_x", "cert_x",
                                 "first_win_secs", "elapsed_secs",
                                 "demos_verified", "demos_unverified")}}
            results_path.write_text(json.dumps(results, indent=2))
            print(f"    rc {rc}, {results[name]}", flush=True)

    print("\nconfig     seed  updates  best_det_x   cert_x  first_win_secs")
    for name, r in sorted(results.items()):
        fw = r.get("first_win_secs")
        fw = "-" if fw is None or fw < 0 else f"{fw:.0f}"
        print(f"{r['config']:<10} {r['seed']:>4}  {r.get('updates') or 0:>7}  "
              f"{(r.get('best_det_x') or 0):>10.0f}  {(r.get('cert_x') or 0):>7.0f}  {fw:>14}")


if __name__ == "__main__":
    main()
