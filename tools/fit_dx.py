"""Fit the exact double per-tick x increment from float32 trace samples.

The game accumulates x in double and exposes float32. We search candidate
doubles and check float32(i * dx) == sample for all alive ticks.
"""
import struct
import sys

from analyze_trace import load


def f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


def main(path):
    rows = [r for r in load(path) if r["src"] == 0 and not r["dead"]]
    samples = [(r["tick"], r["realx"]) for r in rows]
    n = samples[-1][0]

    candidates = {
        "311.58/240": 311.58 / 240,
        "5.77*0.9*0.25": 5.77 * 0.9 * 0.25,
        "5.770002*0.9*0.25": 5.770002 * 0.9 * 0.25,
        "0.9*0.25*346.2/60": 0.9 * 0.25 * 346.2 / 60,
    }
    # brute refine around 1.29825 at double ulp scale via last-sample match
    for name, dx in candidates.items():
        # simple model: x_i = i * dx (start x=0); compare float32
        bad = sum(1 for i, x in samples if abs(f32(i * dx) - x) > 1e-6 * max(1, x))
        exact = sum(1 for i, x in samples if f32(i * dx) == f32(x))
        print(f"{name}: dx={dx!r}  exact={exact}/{len(samples)}  bad={bad}")

    # accumulate model: x += dx each tick (double), matches float32 sample?
    for name, dx in candidates.items():
        acc = 0.0
        exact = 0
        first_bad = None
        prev_t = samples[0][0]
        for i, x in samples:
            acc += dx * (i - prev_t)
            prev_t = i
            if f32(acc) == f32(x):
                exact += 1
            elif first_bad is None:
                first_bad = (i, acc, x)
        print(f"accum {name}: exact={exact}/{len(samples)} first_bad={first_bad}")


if __name__ == "__main__":
    main(sys.argv[1])
