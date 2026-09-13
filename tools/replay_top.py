"""Retrain the best N configurations of a search, long, and tabulate the result.

WHY THIS EXISTS

One test number cannot tell you whether a method works or whether one
configuration got a good draw. The search ranks on validation over a short
horizon; what you want to report is test accuracy over a long one, on several
survivors, under whatever training change you are arguing for.

Doing that by hand means copying flat configs out of trials.jsonl, writing them
into best.json-shaped files, remembering the same flags five times, and then
collecting five single.json files into a table. Every one of those steps is a
chance to run configuration A and record it as configuration B, which is the
mistake this project already made once.

    python tools/replay_top.py --run ~/resultsSNNConfigurable/dvs128_v2 \
        --n 5 --epochs 300 --train weight_decay=1e-3 --out results/top5

Writes, per rank, a replayable trial file and a full run directory, then
summary.csv across all of them. Runs are sequential on purpose: each gets the
whole GPU, and a long run is not worth sharing a card for.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Carried from the search record so a row says where it came from, plus what
# the replay measured. search_hw_val next to replay_hw_val is the interesting
# pair: it shows how much of the search's ranking survived a longer run.
SUMMARY_COLS = [
    "rank", "trial_id", "search_hw_val",
    "hw_val_accuracy", "hw_test_accuracy",
    "float_val_accuracy", "quant_gap",
    "synops_per_sample", "synops_over_dense", "weight_clip_frac",
    "max_abs_weight", "deployable", "wall_minutes", "status",
]


def load_trials(run_dir):
    """Every completed trial that actually deployed, newest file wins."""
    path = os.path.join(os.path.expanduser(run_dir), "trials.jsonl")
    if not os.path.isfile(path):
        sys.exit(f"no trials.jsonl under {run_dir}\n"
                 "  This reads the search's own record, not leaderboard.csv,\n"
                 "  because only the jsonl carries each trial's full config.")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            acc = rec.get("hw_val_accuracy")
            # A trial with no hardware number never reached the deploy phase,
            # so there is nothing to rank it on. Ranking it as zero would put
            # pruned trials below infeasible ones, which is meaningless.
            if acc is None or not rec.get("config"):
                continue
            if rec.get("deployable") is False:
                continue
            out.append(rec)
    return out


def pick_top(trials, n):
    """Top n by hardware validation accuracy, identical configs collapsed."""
    seen, picked = set(), []
    for rec in sorted(trials, key=lambda r: -float(r["hw_val_accuracy"])):
        key = json.dumps(rec["config"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        picked.append(rec)
        if len(picked) >= n:
            break
    return picked


def write_trial_file(rec, path):
    """best.json shape, which is what --from-best reads."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"trial_id": rec.get("trial_id"),
                   "hw_val_accuracy": rec.get("hw_val_accuracy"),
                   "flat_config": rec["config"]}, fh, indent=2, default=str)


def run_one(args, trial_path, out_dir, rank):
    cmd = [sys.executable, os.path.join(ROOT, "main.py"), "single",
           "-c", args.config,
           "--from-best", trial_path,
           "--results-dir", out_dir,
           "--ckpt", f"rank{rank}.pth"]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.train:
        cmd += ["--train"] + list(args.train)
    print("\n" + "=" * 78)
    print(f"rank {rank}: " + " ".join(cmd))
    print("=" * 78, flush=True)
    # Inherit stdout so the per-epoch curve is visible live. A four-hour job
    # that prints nothing until it finishes is a job you cannot tell from a
    # hung one.
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def collect(out_dir, rec, rank, rc):
    row = {c: "" for c in SUMMARY_COLS}
    row.update(rank=rank, trial_id=rec.get("trial_id"),
               search_hw_val=rec.get("hw_val_accuracy"),
               status="ok" if rc == 0 else f"exit {rc}")
    path = os.path.join(out_dir, "single.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        for key in SUMMARY_COLS:
            if key in res:
                row[key] = res[key]
    elif rc == 0:
        row["status"] = "no single.json"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="search results directory")
    ap.add_argument("--out", required=True, help="where to put the replay runs")
    ap.add_argument("-c", "--config", default="configs/dvs128_long.yaml")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--train", nargs="*", default=[], metavar="KEY=VALUE",
                    help="passed through, e.g. --train weight_decay=1e-3")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the trial files and print the commands only")
    args = ap.parse_args()

    trials = load_trials(args.run)
    picked = pick_top(trials, args.n)
    if not picked:
        sys.exit(f"no deployable trials with a hardware accuracy in {args.run}")

    out_root = os.path.expanduser(args.out)
    os.makedirs(out_root, exist_ok=True)
    print(f"{len(trials)} deployable trials, taking top {len(picked)}:")
    for i, rec in enumerate(picked, 1):
        print(f"  rank {i}  {rec.get('trial_id')}  "
              f"search hw_val {float(rec['hw_val_accuracy']):.4f}")

    rows, t0 = [], time.time()
    for i, rec in enumerate(picked, 1):
        out_dir = os.path.join(out_root, f"rank{i}")
        trial_path = os.path.join(out_dir, "trial.json")
        write_trial_file(rec, trial_path)
        if args.dry_run:
            continue
        rc = run_one(args, trial_path, out_dir, i)
        rows.append(collect(out_dir, rec, i, rc))
        # Written after EVERY run, not at the end. A five-run sweep that dies
        # on run four should not lose runs one through three.
        with open(os.path.join(out_root, "summary.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
            w.writeheader()
            w.writerows(rows)

    if args.dry_run:
        print("\ndry run: trial files written, nothing trained")
        return 0

    print("\n" + "=" * 78)
    print(f"{len(rows)} runs in {(time.time() - t0) / 60:.1f} min")
    print(f"{os.path.join(out_root, 'summary.csv')}\n")
    hdr = ("rank", "trial_id", "search_hw_val", "hw_val_accuracy",
           "hw_test_accuracy", "synops_over_dense")
    print("  ".join(f"{h:>16}" for h in hdr))
    for row in rows:
        print("  ".join(
            f"{row.get(h, '') if not isinstance(row.get(h), float) else f'{row[h]:.4f}':>16}"
            for h in hdr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
