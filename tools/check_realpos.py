"""Verify integration rules using the authoritative realx/realy fields.

Checks, per substep (src=0 end-of-tick samples):
  dx == speed_const * dt-ish (constant per tick)
  during freefall: d(realy) vs yvel (integration factor)
  jump: yvel sequence around the jump tick
"""
import sys
from collections import Counter

from analyze_trace import load


def main(path):
    rows = [r for r in load(path) if r["src"] == 0]
    print(f"{path}: {len(rows)} end-of-tick samples")

    dxs = Counter(round(b["realx"] - a["realx"], 6) for a, b in zip(rows, rows[1:]))
    print("realx deltas:", dxs.most_common(5))

    # freefall: relate dy to yvel of the *new* tick (post-update velocity)
    rel = Counter()
    for a, b in zip(rows, rows[1:]):
        if not a["onground"] and not b["onground"] and not b["dead"] \
                and b["yvel"] != 0:
            dy = b["realy"] - a["realy"]
            rel[round(dy / b["yvel"], 6)] += 1
    print("dy / yvel_new (airborne):", rel.most_common(5))

    # first jump: print 6 ticks around first yvel > 0
    for i, r in enumerate(rows):
        if r["yvel"] > 0:
            for rr in rows[max(0, i - 2):i + 6]:
                print(f"  t={rr['tick']} yvel={rr['yvel']:.6f} "
                      f"realy={rr['realy']:.6f} og={rr['onground']} "
                      f"hold={rr['holding']}")
            break

    # totaltime advance per tick
    dts = Counter(round(b["totaltime"] - a["totaltime"], 12)
                  for a, b in zip(rows, rows[1:]))
    print("totaltime deltas:", dts.most_common(3))


if __name__ == "__main__":
    main(sys.argv[1])
