"""Fit the y integration model over the first jump arc."""
import itertools
import struct
import sys

from analyze_trace import load


def f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


def main(path):
    rows = [r for r in load(path) if r["src"] == 0]
    # first airborne segment started by a jump
    i0 = next(i for i, r in enumerate(rows) if r["yvel"] > 0)
    seg = list(itertools.takewhile(lambda r: not r["onground"], rows[i0:]))
    print(f"jump arc: ticks {seg[0]['tick']}..{seg[-1]['tick']} ({len(seg)} ticks)")

    G_opts = {"0.216": 0.216, "f32(0.216)": f32(0.216), "0.9*0.24": 0.9 * 0.24}
    K_opts = {"0.225": 0.225, "f32(0.225)": f32(0.225), "0.9*0.25": 0.9 * 0.25}
    J_opts = {"11.18": 11.18, "f32(11.18)": f32(11.18)}

    y0 = 105.0
    for (gn, G), (kn, K), (jn, J) in itertools.product(
            G_opts.items(), K_opts.items(), J_opts.items()):
        for acc_name, acc_f32 in (("f32acc", True), ("dblacc", False)):
            yvel = J
            y = y0
            ok_y = ok_v = 0
            first_bad = None
            for r in seg[1:]:  # seg[0] is the tick yvel was set, y not moved yet
                yvel = yvel - G
                step = yvel * K
                y = f32(y + f32(step)) if acc_f32 else y + step
                if f32(y) == f32(r["realy"]):
                    ok_y += 1
                elif first_bad is None:
                    first_bad = (r["tick"], y, r["realy"])
                if yvel == r["yvel"]:
                    ok_v += 1
            n = len(seg) - 1
            if ok_y == n and ok_v == n:
                print(f"EXACT: G={gn} K={kn} J={jn} {acc_name}")
            elif ok_y > n * 0.9:
                print(f"close: G={gn} K={kn} J={jn} {acc_name} "
                      f"y={ok_y}/{n} v={ok_v}/{n} first_bad={first_bad}")


if __name__ == "__main__":
    main(sys.argv[1])
