"""Re-annotate a search whose records name an input size it never trained at.

WHAT WENT WRONG

Every trial sampled an input resolution and a frame count, wrote them to its
record, and then built its dataloaders from the config file instead. So the
leaderboard names one input and the run used another, and the feasibility
column was computed against the named one.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT DO

It does not overwrite the recorded values. Those are the evidence that the
search wrote down something different from what it ran, and that discrepancy is
the finding. Deleting it would leave a tidy file and no way to show anyone why
the numbers moved.

Instead every affected column is split in two:

    resize_to  ->  resize_to_recorded  +  resize_to_actual
    T          ->  T_recorded          +  T_actual
    feasible   ->  feasible_recorded   +  feasible_actual

and `feasibility_changed` marks the rows where the answer flips. Feasibility is
recomputed by calling the search's own check_feasibility, not by reimplementing
the arithmetic here, so the two cannot drift apart.

    python tools/correct_broken_run.py <run-dir> --resize 64 --T 16 -o <out-dir>

Only shape decides feasibility, so T is carried for the record and changes
nothing. Resolution changes exactly two quantities: the total axon count, and
the fan-in of the first fully-connected layer when the head flattens rather
than pooling. Convolution fan-in and fan-out are kernel times channels and do
not move.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# config_to_specs needs these and a leaderboard row does not carry them. None
# affects feasibility: batch size, epoch count and the data path change nothing
# about how many connections a neuron has.
FILLERS = {"N": 16, "epochs": 60, "data_dir": ".", "pool_type": "avg"}


def load_configs(run_dir):
    """trial_id -> flat config. Prefers trials.jsonl, which carries the whole
    dict; the CSV drops per-layer keys like fc_width_0 that the fan-out check
    reads."""
    out, src = {}, None
    path = os.path.join(run_dir, "trials.jsonl")
    if os.path.isfile(path):
        src = "trials.jsonl"
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("trial_id") and rec.get("config"):
                    out[rec["trial_id"]] = dict(rec["config"])
    return out, src


def row_to_config(row):
    """Fall back to the leaderboard row when trials.jsonl has no entry."""
    cfg = {}
    for key, val in row.items():
        if val in (None, "") or key.startswith("_"):
            continue
        try:
            cfg[key] = int(val) if str(val).lstrip("-").isdigit() else float(val)
        except ValueError:
            cfg[key] = {"True": True, "False": False}.get(val, val)
    return cfg


def feasibility_at(cfg, resize, T):
    """(feasible, violations) for this config at the given input size."""
    from snnsearch.spaces import config_to_specs
    from snnsearch.hardware import check_feasibility
    flat = {**FILLERS, **cfg, "resize_to": int(resize), "T": int(T)}
    # A row that never sampled a hidden-FC width cannot have its last conv
    # block's fan-out checked. Say so rather than guessing a width and
    # reporting a number nobody can trace.
    unknown_width = int(flat.get("fc_layers", 0) or 0) > 0 and "fc_width_0" not in flat
    try:
        spec = config_to_specs(flat)
    except Exception as exc:
        return False, [f"could not build: {type(exc).__name__}: {exc}"], unknown_width
    try:
        ok, why = check_feasibility(spec["input"], spec["encoder"],
                                    spec["downsample"], spec["head"], spec["output"])
    except Exception as exc:
        return False, [f"check failed: {type(exc).__name__}: {exc}"], unknown_width
    return ok, why, unknown_width


def recorded_feasible(v):
    """True, False, or None when the record never said.

    Older runs only wrote `feasible` when a trial was REJECTED, so a trial that
    trained left the column blank. Reading blank as False makes every trained
    trial look like it was rejected and then "became feasible", which is how the
    first version of this report claimed 129 trials flipped in a direction that
    is physically impossible: raising the resolution can only add violations.
    """
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--resize", type=int, required=True,
                    help="the resolution every trial ACTUALLY trained at")
    ap.add_argument("--T", type=int, required=True,
                    help="the frame count every trial ACTUALLY trained at")
    args = ap.parse_args()

    run_dir = os.path.expanduser(args.run_dir)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    lb = os.path.join(run_dir, "leaderboard.csv")
    if not os.path.isfile(lb):
        sys.exit(f"no leaderboard.csv in {run_dir}")

    configs, src = load_configs(run_dir)
    with open(lb, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("leaderboard.csv is empty")

    fields = list(rows[0].keys())
    for old, new in (("resize_to", "resize_to_recorded"), ("T", "T_recorded"),
                     ("feasible", "feasible_recorded")):
        if old in fields:
            fields[fields.index(old)] = new
    added = ["resize_to_actual", "T_actual", "feasible_actual",
             "feasibility_changed", "violations_actual", "fc_width_unknown"]
    fields += [f for f in added if f not in fields]

    n_flip_bad = n_flip_good = n_unknown = n_unrecorded = 0
    n_ok = n_bad = 0
    out_rows = []
    for row in rows:
        cfg = configs.get(row.get("trial_id")) or row_to_config(row)
        ok, why, unknown = feasibility_at(cfg, args.resize, args.T)
        was = recorded_feasible(row.get("feasible"))
        new = dict(row)
        new["resize_to_recorded"] = row.get("resize_to", "")
        new["T_recorded"] = row.get("T", "")
        new["feasible_recorded"] = row.get("feasible", "")
        new.pop("resize_to", None)
        new.pop("T", None)
        new.pop("feasible", None)
        new["resize_to_actual"] = args.resize
        new["T_actual"] = args.T
        new["feasible_actual"] = ok
        new["feasibility_changed"] = "" if was is None else (was != ok)
        new["violations_actual"] = "; ".join(why)[:300]
        new["fc_width_unknown"] = unknown
        n_ok += bool(ok)
        n_bad += not ok
        if was is None:
            n_unrecorded += 1
        elif was and not ok:
            n_flip_bad += 1
        elif ok and not was:
            n_flip_good += 1
        if unknown:
            n_unknown += 1
        out_rows.append(new)

    dest = os.path.join(out_dir, "leaderboard_corrected.csv")
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    # Which violation actually fires, counted. Resolution can only break two
    # things, so if anything else shows up here the assumption is wrong.
    kinds = {}
    for r in out_rows:
        for v in (r["violations_actual"] or "").split(";"):
            v = v.strip()
            if v:
                kinds[v.split()[0].rstrip(":")] = kinds.get(v.split()[0].rstrip(":"), 0) + 1

    lines = [
        f"# Corrected feasibility for {os.path.basename(run_dir)}",
        "",
        f"Every trial in this run trained at **{args.resize}x{args.resize}, "
        f"T={args.T}**, whatever its record says.",
        "",
        f"- trials: {len(out_rows)}",
        f"- **infeasible at {args.resize}px: {n_bad}** "
        f"({n_bad / max(1, len(out_rows)):.0%})",
        f"- feasible at {args.resize}px: {n_ok}",
        "",
        "Against what the record claimed:",
        "",
        f"- recorded feasible, infeasible at {args.resize}px: {n_flip_bad}",
        f"- recorded infeasible, feasible at {args.resize}px: {n_flip_good}",
        f"- record said nothing: {n_unrecorded}",
        "",]
    if n_unrecorded:
        lines += [
            f"That last count is large because this run only wrote `feasible` "
            f"when it REJECTED a trial. A trial that trained left the column "
            f"blank, so most rows have nothing to compare against and the flip "
            f"counts above cover only the {len(out_rows) - n_unrecorded} rows "
            f"that do. The absolute count at the top is the number to use.",
            "",]
    lines += [
        "Recorded values are preserved in `*_recorded` columns. They are the",
        "evidence that the search wrote down an input it did not use.",
        "",
        "## Which limit fires at the actual resolution",
        "",
        "| violation | trials |",
        "|---|---|",
    ]
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {n} |")
    if n_unknown:
        lines += ["", f"{n_unknown} rows sampled a hidden FC layer whose width the "
                      "record does not carry, so their last conv block's fan-out "
                      "could not be checked. Marked `fc_width_unknown`."]
    if src:
        lines += ["", f"Configurations read from `{src}`."]

    with open(os.path.join(out_dir, "CORRECTION.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {dest}\n      {os.path.join(out_dir, 'CORRECTION.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
