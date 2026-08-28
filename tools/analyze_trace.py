"""Analyze a trace CSV from the fang.trace-logger Geode mod (format=1).

Derives empirical physics constants: tick rate, x-speed per tick, gravity
per tick (freefall yvel delta), jump impulse (yvel right after a jump).
"""
import csv
import sys
from collections import Counter


def load(path):
    rows = []
    with open(path, newline="") as f:
        header = f.readline().strip()
        assert header.startswith(("format=1", "format=2", "format=3")), header
        for r in csv.DictReader(f):
            rows.append({
                "src": int(r.get("src", 0)),
                "nodex": float(r.get("nodex", 0)),
                "realx": float(r.get("realx", 0)),
                "realy": float(r.get("realy", 0)),
                "totaltime": float(r.get("totaltime", 0)),
                "tick": int(r["tick"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "yvel": float(r["yvel"]),
                "holding": int(r["holding"]),
                "onground": int(r["onground"]),
                "upsidedown": int(r["upsidedown"]),
                "ship": int(r["ship"]),
                "dead": int(r["dead"]),
                "speed": float(r["speed"]),
                "dt": float(r["dt"]),
                "halftick": int(r["halftick"]),
                "lasttick": int(r["lasttick"]),
            })
    return rows


def analyze(path):
    all_rows = load(path)
    for src in sorted({r["src"] for r in all_rows}):
        name = {0: "processCommands", 1: "PlayerObject::update"}[src]
        analyze_stream(path, [r for r in all_rows if r["src"] == src], name)


def analyze_stream(path, rows, name):
    print(f"== {path} [{name}]: {len(rows)} ticks ==")
    print(f"dt values: {Counter(r['dt'] for r in rows)}")
    print(f"halftick:  {Counter(r['halftick'] for r in rows)}")
    print(f"lasttick:  {Counter(r['lasttick'] for r in rows)}")
    print(f"speed values: {Counter(r['speed'] for r in rows)}")
    print(f"modes: ship={sum(r['ship'] for r in rows)} ticks, "
          f"dead={sum(r['dead'] for r in rows)}, "
          f"holding={sum(r['holding'] for r in rows)}")

    dx = Counter()
    for a, b in zip(rows, rows[1:]):
        dx[round(b["x"] - a["x"], 6)] += 1
    print(f"x deltas: {dx.most_common(8)}")

    # gravity: yvel delta while airborne (cube, not dead, no hold transition)
    dyv = Counter()
    for a, b in zip(rows, rows[1:]):
        if not a["onground"] and not b["onground"] and not a["ship"] \
                and not b["dead"] and a["yvel"] != 0:
            dyv[round(b["yvel"] - a["yvel"], 9)] += 1
    print(f"airborne cube yvel deltas (gravity/tick): {dyv.most_common(8)}")

    # jump impulse: yvel on the first tick it becomes positive from ground
    jumps = Counter()
    for a, b in zip(rows, rows[1:]):
        if a["onground"] and not a["ship"] and b["yvel"] > 0 and a["yvel"] <= 0:
            jumps[round(b["yvel"], 9)] += 1
    if jumps:
        print(f"cube jump initial yvel: {jumps.most_common(5)}")

    # ship section, if present
    dyv_ship = Counter()
    for a, b in zip(rows, rows[1:]):
        if a["ship"] and b["ship"] and not b["dead"]:
            dyv_ship[(a["holding"], round(b["yvel"] - a["yvel"], 9))] += 1
    if dyv_ship:
        print(f"ship (holding, yvel delta): {dyv_ship.most_common(10)}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
