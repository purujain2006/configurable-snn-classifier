"""K-fold cross-subject validation for a chosen network: `single --folds k`.

One fixed set of validation people scores every network on the same exam,
which is fair for ranking but reads high or low depending on who was drawn:
on DVS128 Gesture, three of the four people run.seed 1 holds out score below
the test average for every network. With search.val_folds = k the training
people are split into k groups (data.loaders.subject_folds) and the network
trains k times, each time with one group as validation and the rest as
training, so every training person is held out exactly once.

    RESULTS/fold0 ... fold<k-1>   one `single` run each (single.json, .pth)
    RESULTS/folds.json            validation per fold, mean and standard error

It costs k trainings per network, so it is for finalists and chosen networks,
not for search trials. Fold runs never score the test set.
"""

import copy
import glob
import json
import math
import os
import re

_FOLD_DIR = re.compile(r"fold(\d+)$")


def fold_config(cfg, fold):
    """`cfg` for one fold: val_fold set, results in <results_dir>/fold<i>."""
    from .pipeline import results_dir
    out = copy.deepcopy(cfg)
    out["search"]["val_fold"] = int(fold)
    out["run"]["results_dir"] = os.path.join(results_dir(cfg), f"fold{int(fold)}")
    return out


def run_folds(cfg, folds, **single_kwargs):
    """Train each fold in `folds` in turn with run_single, rewriting the
    summary after each, so a stopped run still has one for the folds done."""
    from .pipeline import results_dir, run_single
    k = int(cfg["search"]["val_folds"])
    out = results_dir(cfg)
    folds = [int(i) for i in folds]
    for n, fold in enumerate(folds, 1):
        print(f"\n=== fold {fold} of {k} (0 to {k - 1}), run {n} of {len(folds)} ===")
        code = run_single(fold_config(cfg, fold), **single_kwargs)
        if code:
            return code
        summary = write_summary(out)
    print()
    print("\n".join(summary_lines(summary)))
    return 0


def fold_runs(path):
    """[(fold, folder)] for the fold<i> folders under `path`, in fold order."""
    found = []
    for d in glob.glob(os.path.join(path, "fold*")):
        m = _FOLD_DIR.search(os.path.basename(d))
        if m and os.path.isdir(d):
            found.append((int(m.group(1)), d))
    return sorted(found)


def _final_train_accuracy(folder):
    """The last epoch's training accuracy in progress.jsonl, or None."""
    last = None
    try:
        with open(os.path.join(folder, "progress.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line) if line.strip() else {}
                if rec.get("train_acc") is not None:
                    last = rec["train_acc"]
    except OSError:
        pass
    return last


def summarize(path):
    """The summary of the fold runs under `path`, from their single.json
    files. Validation only: a test score in a single.json is never read."""
    rows, configs, ks = [], [], set()
    for fold, folder in fold_runs(path):
        single = os.path.join(folder, "single.json")
        if not os.path.isfile(single):
            continue
        with open(single, encoding="utf-8") as fh:
            rec = json.load(fh)
        split = rec.get("split") or {}
        ks.add(split.get("val_folds"))
        configs.append(rec.get("config"))
        rows.append({"fold": fold, "val_people": split.get("val_people"),
                     "val_clips": split.get("val_clips"),
                     "train_clips": split.get("train_clips"),
                     "hw_val_accuracy": rec.get("hw_val_accuracy"),
                     "final_train_accuracy": _final_train_accuracy(folder),
                     "wall_minutes": rec.get("wall_minutes"),
                     "replayed_from": rec.get("replayed_from")})
    k = next(iter(ks)) if len(ks) == 1 else None
    vals = [r["hw_val_accuracy"] for r in rows if r["hw_val_accuracy"] is not None]
    mean = sum(vals) / len(vals) if vals else None
    se = (math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
          if len(vals) > 1 else None)
    scored = [r for r in rows if r["hw_val_accuracy"] is not None and r["val_clips"]]
    clips = sum(r["val_clips"] for r in scored)
    pooled = (sum(r["hw_val_accuracy"] * r["val_clips"] for r in scored) / clips
              if clips else None)
    problems = []
    if len(ks) > 1:
        problems.append(f"the runs name different fold counts: {sorted(ks, key=str)}")
    if any(c != configs[0] for c in configs[1:]):
        problems.append("the runs did not all train the same network and recipe")
    return {"folder": os.path.abspath(path), "val_folds": k,
            "folds_done": [r["fold"] for r in rows],
            "folds_missing": [i for i in range(k or 0) if i not in {r["fold"] for r in rows}],
            "mean_val_accuracy": mean, "se_val_accuracy": se,
            "pooled_val_accuracy": pooled, "val_clips": clips,
            "problems": problems, "per_fold": rows}


def write_summary(path):
    """summarize(path), written to <path>/folds.json. Returns the summary."""
    summary = summarize(path)
    target = os.path.join(path, "folds.json")
    with open(target + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    os.replace(target + ".tmp", target)
    return summary


def _pct(v):
    return "   -  " if v is None else f"{v:.4f}"


def summary_lines(summary):
    """The printed table: one row per fold, then the mean and its standard error."""
    lines = [f"folds     : {summary['folder']}",
             f"{'fold':>4s}  {'val people':24s} {'clips':>5s}  {'val':>6s}  {'train':>6s}"]
    for r in summary["per_fold"]:
        people = ", ".join(map(str, r["val_people"] or []))
        lines.append(f"{r['fold']:>4d}  {people:24s} {r['val_clips'] or 0:>5d}  "
                     f"{_pct(r['hw_val_accuracy'])}  {_pct(r['final_train_accuracy'])}")
    mean, se = summary["mean_val_accuracy"], summary["se_val_accuracy"]
    lines.append(f"mean over {len(summary['per_fold'])} folds: {_pct(mean)}"
                 + (f" +/- {se:.4f} (standard error)" if se is not None else ""))
    lines.append(f"pooled over {summary['val_clips']} clips: "
                 f"{_pct(summary['pooled_val_accuracy'])}")
    if summary["folds_missing"]:
        lines.append(f"not run yet: folds {', '.join(map(str, summary['folds_missing']))}")
    for p in summary["problems"]:
        lines.append(f"WARNING: {p}")
    lines.append("val is the deployed network on its fold's people, and train is the last "
                 "epoch's training accuracy. The test set is not scored.")
    return lines
