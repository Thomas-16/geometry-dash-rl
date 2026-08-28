"""Correlate the processCommands (src=0) and PlayerObject::update (src=1)
streams of a format=2 trace by tick index to determine per-tick call order
and where position actually changes."""
import sys
from collections import Counter

from analyze_trace import load


def main(path):
    rows = load(path)
    per_tick = Counter()
    for r in rows:
        per_tick[r["tick"]] += 0  # ensure key
    counts = Counter()
    by_tick = {}
    for r in rows:
        by_tick.setdefault(r["tick"], []).append(r)
    for t, rs in by_tick.items():
        counts[tuple(r["src"] for r in rs)] += 1
    print("src sequence per tick:", counts.most_common(10))

    # where does x change: between src0->src1 of same tick, or across ticks?
    intra = Counter()
    inter = Counter()
    ticks = sorted(by_tick)
    for t in ticks:
        rs = by_tick[t]
        s0 = [r for r in rs if r["src"] == 0]
        s1 = [r for r in rs if r["src"] == 1]
        if s0 and s1:
            intra[round(s1[-1]["x"] - s0[0]["x"], 6) != 0] += 1
    for a, b in zip(ticks, ticks[1:]):
        last_a = by_tick[a][-1]
        first_b = by_tick[b][0]
        inter[round(first_b["x"] - last_a["x"], 6) != 0] += 1
    print("x changed within tick (src0 -> src1):", dict(intra))
    print("x changed across tick boundary:", dict(inter))

    # sample of first 12 ticks: (tick, src, x, y, yvel)
    for t in ticks[:12]:
        for r in by_tick[t]:
            print(f"  t={t} src={r['src']} x={r['x']:.6f} y={r['y']:.4f} "
                  f"yvel={r['yvel']:.4f} og={r['onground']}")


if __name__ == "__main__":
    main(sys.argv[1])
