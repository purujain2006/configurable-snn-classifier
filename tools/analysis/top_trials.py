"""The best trials of a search so far, with the settings that matter.

    python tools/analysis/top_trials.py results/dvs128_search05
    python tools/analysis/top_trials.py results/dvs128_search05 --n 25

Reads leaderboard.csv, which grows while the search runs, so this works at any
point. Ranked by the deployed network's validation accuracy. `sim` is the same
validation clips on the simulator and `dis` the clips where the two disagree,
which should always be 0. With about 200 validation clips one score moves by
roughly 3 points by chance, so a few points between neighbours is a tie.
"""
import argparse
import csv
import os
import sys


def pooling(row):
    if "pool" in (row.get("downsample_per_layer") or "").split(",") or \
            row.get("downsample_mode") == "pool":
        return row.get("pool_type") or "avg"
    return "none"


def yes(v):
    return "on" if str(v).lower() in ("true", "1") else "off"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    path = os.path.join(args.run_dir, "leaderboard.csv")
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("val_accuracy")]
    rows.sort(key=lambda r: float(r["val_accuracy"] or 0), reverse=True)
    print(f"{len(rows)} deployed trials in {path}\n")
    head = (f"{'trial':9s} {'val':>6s} {'sim':>6s} {'dis':>3s} {'T':>3s} {'size':>4s} "
            f"{'pool':5s} {'head':8s} {'bias':9s} {'signed':6s} {'bits':>4s} "
            f"{'tau':8s} {'loss':4s} {'aug':4s} {'epochs':>6s}")
    print(head)
    print("-" * len(head))
    for r in rows[:args.n]:
        sim = r.get("sim_val_accuracy") or ""
        print(f"{r['trial_id'][:9]:9s} {float(r['val_accuracy']):6.3f} "
              f"{(float(sim) if sim else float('nan')):6.3f} {r.get('sim_disagreements') or '-':>3s} "
              f"{r.get('T', ''):>3s} {r.get('resize_to', ''):>4s} {pooling(r):5s} "
              f"{r.get('final_reduction', ''):8s} {r.get('bias_mode', ''):9s} "
              f"{yes(r.get('signed_spikes')):6s} {r.get('input_bits') or '1':>4s} "
              f"{(r.get('tau_mode') or 'fixed'):8s} {(r.get('loss') or 'rate'):4s} "
              f"{yes(r.get('use_augment')):4s} {r.get('epochs_run', ''):>6s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
