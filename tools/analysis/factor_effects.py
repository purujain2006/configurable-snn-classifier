"""How each on/off choice did across a search, level by level.

    python tools/analysis/factor_effects.py results/dvs128_search04
    python tools/analysis/factor_effects.py results/dvs128_search04 --block 96

Reads trials.jsonl and trial_progress.jsonl and, for each choice the balanced
block covers, prints per level: how many trials used it, how many were
rejected by the chip check, stopped early, or finished; the median and best
validation accuracy of the finished ones (deployed network, and on the
simulator); and the median validation accuracy every trial had at one fixed
epoch (--epoch, default 19, the last epoch before the first early-stopping
checkpoint).

Read the last column first. Two things bias the others:

  * Early stopping. A trial is cut at its first checkpoint if it trails the
    others then, so options that learn slowly at first (augmentation, TET,
    dropout) finish less often and their "finished" medians rest on a few
    survivors. At the fixed epoch nobody has been stopped yet.
  * Steering. After the balanced block, Optuna mostly picks the options that
    have scored well, with better-tuned settings around them, so those
    options' numbers rise for reasons that are not the option. --block N
    keeps only the first N trials to start, the balanced block, where every
    option had the same number of trials and everything else was random.

A trial the chip check rejected stopped before it sampled any training
setting, so it counts only toward the shape choices (pooling, bias, head,
signed spikes, input bits, T).

T is spread evenly in the block from Search 06 on. Earlier blocks drew it
at random per trial, so their T counts are uneven (Search 05: 15, 33, 15,
33 at T = 16, 24, 32, 48).
"""
import argparse
import json
import os
import statistics
import sys

SHAPE = ("pooling", "bias_mode", "final_reduction", "signed_spikes", "input_bits", "T")


def levels(cfg):
    depth = int(cfg.get("depth", 0) or 0)
    pools = any(cfg.get(f"ds_{i}") == "pool" for i in range(depth)) or \
        cfg.get("downsample_mode") == "pool"
    bits = int(cfg.get("input_bits", 1) or 1)
    tau_mode = cfg.get("tau_mode") or ("layer" if cfg.get("trainable_tau") else "fixed")
    return {
        "pooling": cfg.get("pool_type", "avg") if pools else "none",
        "bias_mode": cfg.get("bias_mode", "threshold"),
        "final_reduction": cfg.get("final_reduction", "flatten"),
        "signed_spikes": "on" if cfg.get("signed_spikes") else "off",
        "input_bits": "1" if bits == 1 else "multi",
        "tau_mode": tau_mode,
        "loss": cfg.get("loss", "rate"),
        "augment": "on" if cfg.get("use_augment") else "off",
        "conv_dropout": "on" if cfg.get("use_conv_dropout") else "off",
        "rate_penalty": "on" if cfg.get("use_rate_penalty") else "off",
        "T": str(cfg.get("T", "")),
    }


def fmt(xs):
    if not xs:
        return "      -       -"
    return f"{statistics.median(xs):7.4f} {max(xs):7.4f}"


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="search results directory holding trials.jsonl")
    ap.add_argument("--block", type=int, default=0,
                    help="keep only the first N trials to start (the balanced block)")
    ap.add_argument("--epoch", type=int, default=19,
                    help="epoch at which every trial is compared (default 19)")
    args = ap.parse_args()
    rows = read_jsonl(os.path.join(args.run_dir, "trials.jsonl"))
    progress = read_jsonl(os.path.join(args.run_dir, "trial_progress.jsonl"))

    # start order: when each trial first reported anything
    order, at_epoch = {}, {}
    for p in progress:
        order.setdefault(p["trial_id"], len(order))
        if p.get("epoch") == args.epoch and p.get("phase") in ("float", "grid") \
                and p.get("val_accuracy") is not None:
            at_epoch[p["trial_id"]] = p["val_accuracy"]

    if args.block:
        rows = [r for r in rows if order.get(r["trial_id"], 10 ** 9) < args.block]
        print(f"{len(rows)} finished or stopped trials among the first {args.block} to start\n")
    else:
        print(f"{len(rows)} trials in {args.run_dir}\n")

    table = {}
    for r in rows:
        cfg = r.get("config") or r.get("flat_config") or {}
        if not cfg:
            continue
        rejected = r.get("feasible") is False or cfg.get("infeasible_geometry")
        for factor, level in levels(cfg).items():
            if rejected and factor not in SHAPE:
                continue          # never sampled
            t = table.setdefault(factor, {}).setdefault(level, {
                "n": 0, "rejected": 0, "stopped": 0, "finished": 0,
                "hw": [], "sim": [], "ep": []})
            t["n"] += 1
            if rejected:
                t["rejected"] += 1
                continue
            if r["trial_id"] in at_epoch:
                t["ep"].append(at_epoch[r["trial_id"]])
            if r.get("hw_val_accuracy") is None:
                t["stopped"] += 1
            else:
                t["finished"] += 1
                t["hw"].append(r["hw_val_accuracy"])
                if r.get("sim_val_accuracy") is not None:
                    t["sim"].append(r["sim_val_accuracy"])

    ep_head = f"at epoch {args.epoch}"
    for factor in levels({}):
        by_level = table.get(factor)
        if not by_level:
            continue
        print(f"{factor}")
        print(f"  {'level':10s} {'trials':>6s} {'rejected':>8s} {'stopped':>7s} {'finished':>8s}"
              f"   {'finished: median   best':>23s}   {'simulator: median  best':>23s}"
              f"   {ep_head + ': n  median':>22s}")
        # numbers by value, so T = 8 comes before 16
        for level, t in sorted(by_level.items(),
                               key=lambda kv: (0, int(kv[0]), "") if kv[0].isdigit()
                               else (1, 0, kv[0])):
            ep = f"{len(t['ep']):4d}  {statistics.median(t['ep']):.4f}" if t["ep"] else "   -       -"
            print(f"  {level:10s} {t['n']:6d} {t['rejected']:8d} {t['stopped']:7d} {t['finished']:8d}"
                  f"   {fmt(t['hw']):>23s}   {fmt(t['sim']):>23s}   {ep:>22s}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
