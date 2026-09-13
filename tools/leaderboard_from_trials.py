"""Rebuild a complete leaderboard from trials.jsonl, one column per key.

WHY THE STREAMED LEADERBOARD IS LOSSY

`append_csv` writes its header when the first row arrives and then appends. A
key that only shows up on trial 200 has no column to go in, so `DictWriter` is
configured to drop it rather than crash mid-search. That is the right trade for
a file whose job is to survive a crash at trial 380 of 400.

It is the wrong trade for analysis. A per-layer search samples `ch_0`, `k_1`,
`ds_2` and so on, none of which are in the fixed header, so all of them were
discarded. The columns that did survive, `channels` and `kernel_size`, are only
sampled in uniform mode and sat empty. A per-layer leaderboard therefore carried
no architecture at all beyond depth, which is why neither knob could be tested.

Nothing was lost, because trials.jsonl records each trial's whole config. This
reads that file, takes the union of every key that appears anywhere, and writes
a CSV where each one has its own column. Runs already on disk can be rebuilt
without retraining.

    python tools/leaderboard_from_trials.py <run-dir> [-o leaderboard_full.csv]
"""
import argparse
import csv
import json
import os
import re
import sys

# Mirrors LEADERBOARD_COLS in snnsearch/streaming.py, so the two files read the
# same way left to right and a reader who knows one knows the other. Sorting
# alphabetically instead put ch_0 beside conv_dropout and k_0 beside
# label_smoothing, which is correct and unreadable.
#
# "@ch" expands to every ch_0, ch_1, ... that appears, in numeric order, at that
# position. That keeps a per-layer family together and next to the uniform knob
# it replaces, rather than scattered through the alphabet.
ORDER = [
    "trial_id", "status",
    # outcome
    "val_accuracy", "hw_val_accuracy", "pre_export_val_accuracy",
    "float_val_accuracy", "quant_gap", "end_to_end_gain",
    # hardware
    "deployable", "weight_clip_frac", "max_abs_weight", "min_threshold",
    "feasible", "synops_per_sample", "firing_rate",
    "neurons", "connections", "params",
    # architecture: depth, then the per-layer families, then the uniform keys
    # they stand in for
    "depth", "@ch", "@k", "@ds", "@stride", "@pool",
    "channels", "kernel_size", "stride", "downsample_mode", "pool_type",
    "resize_to", "T", "N",
    # neuron
    "tau", "trainable_tau", "trainable_threshold",
    # head
    "fc_layers", "@fc_width", "final_reduction",
    # regularization
    "dropout_rate", "conv_dropout", "use_conv_dropout",
    "rate_penalty", "use_rate_penalty",
    "norm", "tdbn_alpha", "label_smoothing", "grad_clip",
    # optimization
    "optimizer", "lr", "weight_decay", "scheduler", "warmup_epochs",
    # bookkeeping, last because nobody sorts on it
    "epochs", "epochs_requested", "epochs_run", "stopped_early",
    "best_val_accuracy", "wall_time", "deploy_reasons", "violations",
]


def natural(key):
    """ch_0, ch_1, ..., ch_10 in that order rather than ch_0, ch_1, ch_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key)]


def load(run_dir):
    path = os.path.join(os.path.expanduser(run_dir), "trials.jsonl")
    if not os.path.isfile(path):
        sys.exit(f"no trials.jsonl in {run_dir}\n"
                 "  This rebuilds from the per-trial record, which is the only\n"
                 "  file that keeps each trial's whole config.")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def flatten(rec):
    """One flat dict per trial: outcome fields plus every sampled key."""
    row = {k: v for k, v in rec.items() if k not in ("config", "_run")}
    for key, val in (rec.get("config") or {}).items():
        # Absolute paths say nothing about the configuration and make records
        # from different machines look different.
        if key in ("data_dir", "data_dir_abs", "results_dir"):
            continue
        row[key] = val
    if rec.get("_run"):
        row["_run"] = rec["_run"]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    rows = [flatten(r) for r in load(args.run_dir)]
    if not rows:
        sys.exit("trials.jsonl is empty")

    keys = set()
    for r in rows:
        keys.update(r)

    head = []
    for slot in ORDER:
        if slot.startswith("@"):
            fam = slot[1:]
            head += sorted((k for k in keys
                            if re.fullmatch(rf"{re.escape(fam)}_\d+", k)),
                           key=natural)
        elif slot in keys:
            head.append(slot)
    # Anything the template does not name still gets a column. A key silently
    # dropped is the failure this whole file exists to undo.
    head += sorted(keys - set(head), key=natural)

    dest = args.out or os.path.join(os.path.expanduser(args.run_dir),
                                    "leaderboard_full.csv")
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=head, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    streamed = os.path.join(os.path.expanduser(args.run_dir), "leaderboard.csv")
    n_before = 0
    if os.path.isfile(streamed):
        with open(streamed, newline="", encoding="utf-8") as fh:
            n_before = len(next(csv.reader(fh), []))

    print(f"{len(rows)} trials, {len(head)} columns -> {dest}")
    if n_before:
        gained = [k for k in head if k not in
                  set(csv.DictReader(open(streamed, newline="", encoding="utf-8")).fieldnames or [])]
        print(f"streamed leaderboard.csv had {n_before} columns; "
              f"{len(gained)} recovered here")
        if gained:
            print("  " + ", ".join(gained[:24])
                  + (" ..." if len(gained) > 24 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
