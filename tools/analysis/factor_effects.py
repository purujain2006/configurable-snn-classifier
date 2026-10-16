"""How each on/off choice did across a search, level by level.

    python tools/analysis/factor_effects.py results/dvs128_search03

Reads trials.jsonl and, for each choice the balanced block covers (pooling,
bias, head, conv dropout, firing-rate penalty), prints per level: how many
trials used it, how many were rejected by the chip check, stopped early, or
finished, and the median and best validation accuracy of the finished ones
(deployed network, and on the simulator).

The balanced block gives every level the same number of trials, so these
counts start equal; differences in how many survive are part of the answer.
Everything else about each trial was sampled at random, so a gap of a point
or two between levels is within noise. Retrain the winner with the choice
switched on and off, with a few seeds, to settle a close call.
"""
import argparse
import json
import os
import statistics
import sys


def levels(cfg):
    depth = int(cfg.get("depth", 0) or 0)
    pools = any(cfg.get(f"ds_{i}") == "pool" for i in range(depth)) or \
        cfg.get("downsample_mode") == "pool"
    return {
        "pooling": cfg.get("pool_type", "avg") if pools else "none",
        "bias_mode": cfg.get("bias_mode", "threshold"),
        "final_reduction": cfg.get("final_reduction", "flatten"),
        "conv_dropout": "on" if cfg.get("use_conv_dropout") else "off",
        "rate_penalty": "on" if cfg.get("use_rate_penalty") else "off",
    }


def fmt(xs):
    if not xs:
        return "      -       -"
    return f"{statistics.median(xs):7.4f} {max(xs):7.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="search results directory holding trials.jsonl")
    args = ap.parse_args()
    path = os.path.join(args.run_dir, "trials.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"{len(rows)} trials in {path}\n")

    table = {}
    for r in rows:
        cfg = r.get("config") or r.get("flat_config") or {}
        if not cfg:
            continue
        for factor, level in levels(cfg).items():
            t = table.setdefault(factor, {}).setdefault(level, {
                "n": 0, "rejected": 0, "stopped": 0, "finished": 0, "hw": [], "sim": []})
            t["n"] += 1
            if r.get("feasible") is False:
                t["rejected"] += 1
            elif r.get("hw_val_accuracy") is None:
                t["stopped"] += 1
            else:
                t["finished"] += 1
                t["hw"].append(r["hw_val_accuracy"])
                if r.get("sim_val_accuracy") is not None:
                    t["sim"].append(r["sim_val_accuracy"])

    for factor, by_level in table.items():
        print(f"{factor}")
        print(f"  {'level':10s} {'trials':>6s} {'rejected':>8s} {'stopped':>7s} {'finished':>8s}"
              f"   {'val median   best':>17s}   {'simulator median  best':>22s}")
        for level, t in sorted(by_level.items()):
            print(f"  {level:10s} {t['n']:6d} {t['rejected']:8d} {t['stopped']:7d} {t['finished']:8d}"
                  f"   {fmt(t['hw'])}   {fmt(t['sim']):>22s}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
