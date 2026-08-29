"""Digest the mod's objects.csv + collisions_*.csv dumps.

Prints, per object id: GameObjectType, hitbox size, and hitbox-center offset
relative to the object position, with every distinct variant listed (so
rotation/flip effects are visible). Also summarizes the player rects seen in
collision tests and all death events.
"""
import collections
import csv
import glob
import sys


def load_rows(path):
    with open(path, newline="") as f:
        fmt = f.readline().strip()
        assert fmt.startswith("format="), f"{path}: bad header {fmt!r}"
        return list(csv.DictReader(f))


def analyze_objects(path):
    rows = load_rows(path)
    player = [r for r in rows if r["uniqueid"] == "-1"]
    objs = [r for r in rows if r["uniqueid"] != "-1"]
    by_id = collections.defaultdict(list)
    for r in objs:
        by_id[int(r["objid"])].append(r)

    print(f"{path}: {len(objs)} objects, {len(by_id)} distinct ids\n")
    print(f"{'id':>5} {'type':>5} {'count':>6}  variants (w x h @ offset ox,oy | rot flips)")
    for oid in sorted(by_id):
        rows_ = by_id[oid]
        variants = collections.Counter()
        for r in rows_:
            w, h = float(r["rw"]), float(r["rh"])
            # hitbox center offset relative to object position
            ox = float(r["rx"]) + w / 2 - float(r["x"])
            oy = float(r["ry"]) + h / 2 - float(r["y"])
            key = (round(w, 4), round(h, 4), round(ox, 4), round(oy, 4),
                   float(r["rot"]), r["flipx"], r["flipy"])
            variants[key] += 1
        t = rows_[0]["type"]
        first = True
        for (w, h, ox, oy, rot, fx, fy), n in sorted(variants.items()):
            head = f"{oid:>5} {t:>5} {len(rows_):>6}" if first else " " * 18
            print(f"{head}  {w:g} x {h:g} @ {ox:g},{oy:g} | rot={rot:g} "
                  f"flip={fx}{fy} (x{n})")
            first = False
    if player:
        r = player[0]
        print(f"\nplayer: rect {r['rw']} x {r['rh']} at rx={r['rx']} ry={r['ry']} "
              f"pos {r['x']},{r['y']} w={r['w']} h={r['h']}")


def analyze_collisions(paths):
    for path in paths:
        rows = load_rows(path)
        tests = [r for r in rows if r["kind"] == "0"]
        deaths = [r for r in rows if r["kind"] == "1"]
        rects = collections.Counter(
            (r["rw"], r["rh"]) for r in tests)
        print(f"\n{path}: {len(tests)} collision tests, {len(deaths)} destroyPlayer calls")
        for (w, h), n in rects.most_common():
            print(f"  player rect {w} x {h}: x{n}")
        by_type = collections.Counter(
            (r["objtype"], r["objid"], r["result"]) for r in tests)
        for (t, oid, res), n in sorted(by_type.items()):
            print(f"  tests vs type={t} id={oid} result={res}: x{n}")
        for r in deaths:
            print(f"  DEATH tick={r['tick']} killer objid={r['objid']} "
                  f"uniqueid={r['uniqueid']} type={r['objtype']}")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "traces"
    analyze_objects(f"{base}/objects.csv")
    analyze_collisions(sorted(glob.glob(f"{base}/collisions_*.csv")))
