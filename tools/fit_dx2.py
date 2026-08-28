"""Test float32-accumulation models for x."""
import struct
import sys

from analyze_trace import load


def f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


def main(path):
    rows = [r for r in load(path) if r["src"] == 0 and not r["dead"]]
    samples = [(r["tick"], r["realx"]) for r in rows]

    SPEED = 0.899999976158142  # float32 m_playerSpeed as double
    models = {
        # float32 accumulator, various per-tick increments
        "f32acc dx=f32(5.770002*0.9*0.25)": f32(5.770002 * 0.9 * 0.25),
        "f32acc dx=f32(5.77*0.9*0.25)": f32(5.77 * 0.9 * 0.25),
        "f32acc dx=f32(5.770002*S*0.25)": f32(5.770002 * SPEED * 0.25),
        "f32acc dx=f32(f32(5.770002)*f32(S)*0.25)":
            f32(f32(f32(5.770002) * f32(SPEED)) * 0.25),
    }
    for name, dx in models.items():
        acc = 0.0
        exact = 0
        first_bad = None
        prev_t = samples[0][0]
        for i, x in samples:
            for _ in range(i - prev_t):
                acc = f32(acc + dx)
            prev_t = i
            if acc == f32(x):
                exact += 1
            elif first_bad is None:
                first_bad = (i, acc, x)
        print(f"{name}: dx={dx!r} exact={exact}/{len(samples)} first_bad={first_bad}")

    # double accumulator with dx exactly float32(1.29825044...)
    for name, dx in models.items():
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
        print(f"dbl acc {name}: exact={exact}/{len(samples)} first_bad={first_bad}")


if __name__ == "__main__":
    main(sys.argv[1])
