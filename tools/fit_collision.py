"""Derive landing-snap and death rules from paired trace + collision logs.

For each attempt:
  - landing events: ticks where src0 onground goes 0->1 with y > floor.
    Prints pre-collision y (src=1 row, post-movement) vs snapped y (src=0)
    vs the top of the solid under the player -> snap rule + clip depth.
  - death event: player position at last ticks vs the killer object's hitbox
    -> bounds on the player's inner hazard hitbox.
"""
import csv
import glob
import os
import sys


def load_fmt(path):
    with open(path, newline="") as f:
        f.readline()
        return list(csv.DictReader(f))


def load_objects(base):
    objs = {}
    for r in load_fmt(os.path.join(base, "objects.csv")):
        uid = int(r["uniqueid"])
        objs[uid] = {
            "objid": int(r["objid"]), "type": int(r["type"]),
            "x": float(r["x"]), "y": float(r["y"]),
            "rx": float(r["rx"]), "ry": float(r["ry"]),
            "rw": float(r["rw"]), "rh": float(r["rh"]),
        }
    return objs


def solids_near(objs, x):
    return [o for o in objs.values()
            if o["type"] == 0 and o["rx"] - 40 <= x <= o["rx"] + o["rw"] + 40]


def analyze(base, apath):
    n = apath[-8:-4]
    rows = load_fmt(apath)
    src0 = [r for r in rows if r["src"] == "0"]
    src1 = [r for r in rows if r["src"] == "1"]
    s1_by_tick = {}
    for r in src1:
        s1_by_tick.setdefault(int(r["tick"]), r)  # first update of that tick
    objs = load_objects(base)
    cols = load_fmt(os.path.join(base, f"collisions_{n}.csv"))
    deaths = [c for c in cols if c["kind"] == "1" and c["uniqueid"] != "2410"]

    print(f"\n=== attempt_{n} ({len(src0)} ticks) ===")

    # landings above the floor
    prev_g = None
    shown = 0
    for i, r in enumerate(src0):
        g = r["onground"] == "1"
        y = float(r["realy"])
        if g and prev_g is False and y > 105.5 and shown < 12:
            t = int(r["tick"])
            x = float(r["realx"])
            s1 = s1_by_tick.get(t)
            pre_y = float(s1["realy"]) if s1 else float("nan")
            cands = solids_near(objs, x)
            tops = sorted({o["ry"] + o["rh"] for o in cands
                           if abs(o["ry"] + o["rh"] + 15 - y) < 8},
                          key=lambda tp: abs(tp + 15 - y))
            top = tops[0] if tops else float("nan")
            print(f"  land t={t} x={x:.4f} pre_y={pre_y!r} -> y={y!r} "
                  f"blocktop={top} snap_off={y - top:.6g} clipdepth={top + 15 - pre_y:.6g}")
            shown += 1
        prev_g = g

    for d in deaths:
        t = int(d["tick"])
        uid = int(d["uniqueid"])
        near = [r for r in src0 if abs(int(r["tick"]) - t) <= 2]
        for r in near:
            print(f"  death-ctx t={r['tick']} x={r['realx']} y={r['realy']} "
                  f"yvel={r['yvel']} g={r['onground']}")
        if uid in objs:
            o = objs[uid]
            print(f"  DEATH t={t} killer id={o['objid']} type={o['type']} "
                  f"hazrect=[{o['rx']:.4g},{o['ry']:.4g} {o['rw']}x{o['rh']}] "
                  f"center=({o['x']},{o['y']})")
            if near:
                r = near[-1]
                px, py = float(r["realx"]), float(r["realy"])
                # overlap margins if player inner half-extents were hx,hy:
                # need px+hx > rx  and px-hx < rx+rw  etc.
                print(f"    player last pos ({px:.4f},{py:.4f}); "
                      f"dx_left={o['rx'] - px:.4f} dx_right={px - (o['rx']+o['rw']):.4f} "
                      f"dy_bot={o['ry'] - py:.4f} dy_top={py - (o['ry']+o['rh']):.4f}")
        else:
            print(f"  DEATH t={t} killer uniqueid={uid} (null/unknown)")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "traces/session2"
    picks = sys.argv[2:] or ["0005", "0002", "0007", "0012", "0004"]
    for n in picks:
        analyze(base, os.path.join(base, f"attempt_{n}.csv"))
