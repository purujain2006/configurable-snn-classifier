"""Add the final spatial size and flatten fan-in to a leaderboard CSV.

WHY THIS EXISTS

`fc_in_features` is the number that decides most feasibility violations, and it
is not in the leaderboard. It is computed inside `NetPlan` and thrown away:

    fc_in = out_c * out_hw[0] * out_hw[1]

`out_hw` is the feature map leaving the LAST conv block, after that block's
pool. It is not the input resolution, and it is not a sampled knob. It falls
out of resolution, depth, kernel, stride and downsample mode together, so no
single column in the file predicts it.

This walks the same arithmetic `plan_encoder` walks and writes three columns:

    out_hw          final spatial size, e.g. "12x12"
    fc_in_features  out_hw squared times final channels, or channels under gap
    fan_in_headroom 8159 minus fc_in_features. Negative means it violates.

    python tools/add_spatial.py <csv> [-o <csv>]

The arithmetic is duplicated here rather than imported because this has to run
against CSVs from older code versions whose configs no longer build. If
`planning.py` changes, change this too. `verify` below is the guard: it
recomputes the violation string the gate recorded and reports any row where the
two disagree.
"""
import argparse
import csv
import os
import re
import sys

FAN_IN_LIMIT = 8159   # NEURON_LIMITS["fan_in"]
POOL_STRIDE = 2       # DownsampleSpec.pool_stride, a fixed default


def conv_out_size(size, k, stride, padding, dilation=1):
    """Mirrors snnsearch.planning.conv_out_size."""
    return (size + 2 * padding - dilation * (k - 1) - 1) // stride + 1


def per_layer(row, family, depth, fallback):
    """ch_0/ch_1/... when the search sampled per layer, else the uniform key.

    A per-layer run leaves `channels` and `kernel_size` empty and a uniform run
    leaves `ch_0` and `k_0` empty, so reading only one of the two silently
    yields a column of blanks.
    """
    out = []
    for i in range(depth):
        v = row.get(f"{family}_{i}", "")
        out.append(v if str(v).strip() != "" else fallback)
    return out


def num(v, default=None):
    s = str(v).strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def final_hw(row):
    """(H_out, W_out, channels_out) or None when the row lacks the geometry."""
    size = num(row.get("resize_to_actual") or row.get("resize_to"))
    depth = num(row.get("depth"))
    if not size or not depth:
        return None

    mode = (row.get("downsample_mode") or "stride").strip()
    ks = per_layer(row, "k", depth, row.get("kernel_size", ""))
    ss = per_layer(row, "stride", depth, row.get("stride", ""))
    chs = per_layer(row, "ch", depth, row.get("channels", ""))
    pools = per_layer(row, "pool", depth, "")

    cur = size
    last_c = None
    for i in range(depth):
        k = num(ks[i])
        s = num(ss[i], 1) or 1
        c = num(chs[i])
        if not k or not c:
            return None
        last_c = c

        # plan_encoder: a map at 3x3 or smaller stops downsampling entirely.
        can_shrink = cur > 3
        want_pool = str(pools[i]).strip().lower() == "true" or mode == "pool"

        if can_shrink and not want_pool:
            stride, padding = s, 0
        else:
            stride, padding = 1, (k - 1) // 2   # size-preserving

        cur = conv_out_size(cur, k, stride, padding)
        if cur < 1:
            return None
        if can_shrink and want_pool:
            cur = cur // POOL_STRIDE
            if cur < 1:
                return None

    return cur, cur, last_c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    path = os.path.expanduser(args.csv_path)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} is empty")

    fields = list(rows[0].keys())
    added = ["out_hw", "final_channels", "fc_in_features", "fan_in_headroom"]
    fields += [f for f in added if f not in fields]

    n_done = n_skip = n_over = 0
    disagree = []
    for row in rows:
        got = final_hw(row)
        if got is None:
            for f in added:
                row[f] = ""
            n_skip += 1
            continue
        h, w, c = got
        reduction = (row.get("final_reduction") or "flatten").strip()
        fc_in = c if reduction == "gap" else c * h * w

        row["out_hw"] = f"{h}x{w}"
        row["final_channels"] = c
        row["fc_in_features"] = fc_in
        row["fan_in_headroom"] = FAN_IN_LIMIT - fc_in
        n_done += 1
        n_over += fc_in > FAN_IN_LIMIT

        # The gate wrote the offending number into `violations`. If it recorded
        # one and it is not the number computed here, this arithmetic has
        # drifted from planning.py and every column above is wrong.
        recorded = row.get("violations_actual") or row.get("violations") or ""
        m = re.search(r"(?:classifier|fc\d+): neuron_fan_in (\d+)", recorded)
        if m and int(m.group(1)) != fc_in:
            disagree.append((row.get("trial_id", "?"), int(m.group(1)), fc_in))

    dest = args.out or path.replace(".csv", "_spatial.csv")
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w_.writeheader()
        w_.writerows(rows)

    print(f"{n_done} rows computed, {n_skip} skipped for missing geometry")
    print(f"{n_over} exceed the {FAN_IN_LIMIT} fan-in limit")
    print(f"-> {dest}")

    if disagree:
        print(f"\nWARNING: {len(disagree)} rows disagree with the violation the "
              f"gate recorded. This arithmetic no longer matches planning.py.")
        for tid, was, now in disagree[:8]:
            print(f"  {tid}: recorded {was}, computed {now}")
    else:
        print("\nEvery row carrying a recorded fan-in violation matches. "
              "The arithmetic agrees with the gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
