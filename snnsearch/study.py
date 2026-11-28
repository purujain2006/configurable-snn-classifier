"""Does each option help? Flip one at a time on one network and train every
variant in full, with several seeds.

    python main.py study -c configs/dvs128_search04.yaml --from-best results/dvs128_search04/best_finalist.json

A search cannot answer this cleanly. Its trials differ in everything at once,
Optuna crowds the options that started well, and early stopping judges a
trial at one checkpoint, which favors options that learn fast over options
that end well. Here every variant is the same network with one option
changed, every variant trains for the full epoch count with the same seeds,
and nothing is stopped early. An option's effect is the variant's score
minus the baseline's for the same seed, averaged over seeds, so the luck of
a seed cancels.

A variant the chip cannot hold (signed spikes doubling a fan-in past the
limit, say) is listed with the reason and not trained.

Besides every option the search covers, it studies options outside the
search, which improve a chosen network without searching again: more
timesteps (T), random temporal deletion, label-aware mirroring, the
optimizer, and starting the firing-rate penalty halfway.

With folds (`study --folds 5`) the replicates are subject folds instead of
seeds: each variant and the baseline train once per fold, each time with one
group of training people as validation, and the paired difference is taken
fold by fold. Every training person is then held out once, so a variant
cannot win by suiting the four people of the fixed validation split.

Writes into <results_dir>/study/:
    study.csv           one row per variant and seed (or fold)
    study_summary.csv   per variant: mean, spread, worst, a score, the paired
                        difference from the baseline with its standard error,
                        and a verdict
"""

import csv
import math
import os
import statistics

from .spaces import BALANCED_FACTORS, geometry_fits, config_to_specs
from .hardware import check_spec_feasibility


# When the study switches on something the network had off, it uses the
# middle of the search's range for the amount.
ON_AMOUNTS = {
    "augment": {"aug_shift": 0.1, "aug_cutout": 0.2, "aug_drop": 0.15},
    "conv_dropout": {"conv_dropout": 0.15},
    "rate_penalty": {"rate_penalty": 3e-3},
}

# Options outside the search space, each with its levels
STUDY_T = (48, 64, 80, 96)
TIME_KEEP_ON = 0.75            # SEW ResNet's 12 of 16 frames
SGD_LR = 0.03                  # SGD wants a far larger step than Adam's 1e-3
DELAYED_PENALTY_START = 0.5    # the rate penalty starts halfway through
EXTRA_FACTORS = {
    "T": [str(t) for t in STUDY_T],
    "time_drop": ["off", "on"],
    "mirror": ["off", "on"],
    "optimizer": ["adam", "adamw", "sgd"],
    "rate_penalty_timing": ["from_start", "delayed"],
}
ALL_OPTIONS = list(BALANCED_FACTORS) + list(EXTRA_FACTORS)


def current_levels(flat):
    """The level of every balanced-block choice this network uses."""
    depth = int(flat.get("depth", 0) or 0)
    pools = any(flat.get(f"ds_{i}") == "pool" for i in range(depth)) or \
        flat.get("downsample_mode") == "pool"
    tau_mode = flat.get("tau_mode") or ("layer" if flat.get("trainable_tau") else "fixed")
    return {
        "pooling": flat.get("pool_type", "avg") if pools else "none",
        "bias_mode": flat.get("bias_mode", "threshold"),
        "final_reduction": flat.get("final_reduction", "flatten"),
        "signed_spikes": "on" if flat.get("signed_spikes") else "off",
        "input_bits": "1" if int(flat.get("input_bits", 1) or 1) == 1 else "multi",
        "tau_mode": tau_mode,
        "loss": flat.get("loss", "rate"),
        # shift, cutout and drop. use_augment alone also switches on a mirror or
        # temporal deletion, which are their own options below
        "augment": "on" if flat.get("use_augment") and any(
            float(flat.get(k) or 0) > 0 for k in ("aug_shift", "aug_cutout", "aug_drop"))
            else "off",
        "conv_dropout": "on" if flat.get("use_conv_dropout") else "off",
        "rate_penalty": "on" if flat.get("use_rate_penalty") else "off",
        "T": str(int(flat.get("T") or 0)),
        "time_drop": "on" if flat.get("use_augment") and
                     0 < float(flat.get("aug_time_keep") or 0) < 1 else "off",
        "mirror": "on" if flat.get("use_augment") and flat.get("aug_flip") else "off",
        "optimizer": flat.get("optimizer") or "adam",
        "rate_penalty_timing": "delayed" if float(flat.get("rate_penalty_start") or 0) > 0
                               else "from_start",
    }


def _set_pooling(flat, level):
    f = dict(flat)
    depth = int(f.get("depth", 0) or 0)
    pooled = [i for i in range(depth) if f.get(f"ds_{i}") == "pool"]
    if level == "none":
        # each pool becomes a stride-2 convolution: the same downsampling
        for i in pooled:
            f[f"ds_{i}"] = "stride"
            f[f"stride_{i}"] = 2
            f.pop(f"pool_{i}", None)
        f.pop("pool_type", None)
        return f
    if not pooled:
        strided = [i for i in range(depth) if f.get(f"ds_{i}") == "stride"]
        if not strided:
            return None
        for i in strided:
            f[f"ds_{i}"] = "pool"
            f[f"pool_{i}"] = 1
            f.pop(f"stride_{i}", None)
    f["pool_type"] = level
    return f


def apply_option(flat, option, level, max_input_bits=4):
    """`flat` with one option set to `level`, or None if it cannot apply."""
    f = dict(flat)
    if option == "pooling":
        return _set_pooling(f, level)
    if option in ("bias_mode", "final_reduction", "tau_mode", "loss"):
        f[option] = level
        if option == "tau_mode":
            f.pop("trainable_tau", None)
        return f
    if option == "signed_spikes":
        f["signed_spikes"] = level == "on"
        return f
    if option == "input_bits":
        if level == "1":
            f["input_bits"] = 1
        else:
            if max_input_bits <= 1:
                return None
            f["input_bits"] = min(2, max_input_bits)
        return f
    if option in ON_AMOUNTS:
        flag = {"augment": "use_augment", "conv_dropout": "use_conv_dropout",
                "rate_penalty": "use_rate_penalty"}[option]
        amounts = dict(ON_AMOUNTS[option])
        if option == "augment" and f.get("hflip_ok"):
            amounts["aug_flip"] = True
        if level == "on":
            f[flag] = True
            for k, v in amounts.items():
                f.setdefault(k, v)
        else:
            f[flag] = False
            for k in list(amounts) + (["aug_flip"] if option == "augment" else []):
                f.pop(k, None)
        return f
    if option == "T":
        f["T"] = int(level)
        return f
    if option == "time_drop":
        if level == "on":
            f["use_augment"] = True
            f["aug_time_keep"] = TIME_KEEP_ON
        else:
            f.pop("aug_time_keep", None)
        return f
    if option == "mirror":
        if level == "on":
            if not (f.get("hflip_labels") or f.get("hflip_ok")):
                return None                     # the dataset has no valid mirror
            f["use_augment"] = True
            f["aug_flip"] = True
        else:
            f["aug_flip"] = False
        return f
    if option == "optimizer":
        f["optimizer"] = level
        if level == "sgd":
            f["lr"] = SGD_LR
        elif (flat.get("optimizer") or "adam") == "sgd":
            f["lr"] = 1e-3
        return f
    if option == "rate_penalty_timing":
        if not f.get("use_rate_penalty"):
            return None                         # no penalty to time
        f["rate_penalty_start"] = DELAYED_PENALTY_START if level == "delayed" else 0.0
        return f
    raise ValueError(f"unknown option {option!r}")


def option_variants(flat, max_input_bits=4, only=None):
    """The baseline, then one variant per other level of each choice.

    Returns dicts with name, option, level, flat (None when skipped) and
    reason (why it was skipped, else "")."""
    now = current_levels(flat)
    out = [{"name": "baseline", "option": None, "level": None, "flat": dict(flat),
            "reason": ""}]
    unknown = sorted(set(only or ()) - set(ALL_OPTIONS))
    if unknown:
        raise SystemExit(f"study --only: unknown option(s) {', '.join(unknown)}; "
                         f"known: {', '.join(ALL_OPTIONS)}")
    for option, levels in {**BALANCED_FACTORS, **EXTRA_FACTORS}.items():
        if only and option not in only:
            continue
        if option == "input_bits" and max_input_bits <= 1:
            continue
        for level in levels:
            if level == now[option]:
                continue
            name = f"{option}={level}"
            v = apply_option(flat, option, level, max_input_bits)
            if v is None:
                out.append({"name": name, "option": option, "level": level, "flat": None,
                            "reason": "does not apply to this network"})
                continue
            if not geometry_fits(v):
                try:
                    _ok, why = check_spec_feasibility(config_to_specs({"tau": 2, **v}))
                    reason = "does not fit the chip: " + "; ".join(why)[:200]
                except Exception as exc:
                    reason = f"does not build: {exc}"[:200]
                out.append({"name": name, "option": option, "level": level, "flat": None,
                            "reason": reason})
                continue
            out.append({"name": name, "option": option, "level": level, "flat": v,
                        "reason": ""})
    return out


def _rep(r):
    """What pairs a variant's run with the baseline's: the seed, and the fold
    when the study runs over folds."""
    return (r["seed"], r.get("fold"))


def summarize_study(variants, rows):
    """Per variant: mean, spread and worst over seeds (or folds), the score
    mean - 0.5 * spread, and the paired difference from the baseline (same
    seed and fold) with its standard error and a verdict."""
    base = {_rep(r): float(r["val_accuracy"] or 0.0) for r in rows
            if r["variant"] == "baseline"}
    out = []
    for v in variants:
        mine = [r for r in rows if r["variant"] == v["name"]]
        vals = [float(r["val_accuracy"] or 0.0) for r in mine]
        diffs = [float(r["val_accuracy"] or 0.0) - base[_rep(r)] for r in mine
                 if _rep(r) in base]
        std = statistics.stdev(vals) if len(vals) > 1 else None
        row = {"variant": v["name"], "runs": len(vals),
               "mean_val_accuracy": round(statistics.mean(vals), 4) if vals else None,
               "std_val_accuracy": round(std, 4) if std is not None else None,
               "worst_val_accuracy": round(min(vals), 4) if vals else None,
               # rewards a variant that holds up on every fold, not only on average
               "score": round(statistics.mean(vals) - 0.5 * std, 4) if std is not None else None,
               "diff_vs_baseline": None, "diff_se": None, "verdict": v["reason"] or ""}
        if v["name"] != "baseline" and diffs:
            d = statistics.mean(diffs)
            se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else None
            row["diff_vs_baseline"] = round(d, 4)
            row["diff_se"] = round(se, 4) if se is not None else None
            if se is None:
                row["verdict"] = "one run: no verdict"
            elif d - 2 * se > 0:
                row["verdict"] = "helps"
            elif d + 2 * se < 0:
                row["verdict"] = "hurts"
            else:
                row["verdict"] = "no clear difference"
        out.append(row)
    return out


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def replicates(run_seed, seeds=3, folds=0):
    """[(seed, fold)] each variant trains on: `seeds` seeds on the run's
    validation split, or, with folds, one seed on each of `folds` subject
    folds."""
    if folds:
        return [(run_seed + 1000, f) for f in range(int(folds))]
    return [(run_seed + 1000 * (j + 1), None) for j in range(int(seeds))]


def job_flats(trained, reps, folds=0):
    """One flat config per (variant, seed, fold), with the fold in keys the
    trial function moves into the run config (search.fold_override)."""
    jobs, flats = [], []
    for v in trained:
        for seed, fold in reps:
            f = {**v["flat"], "seed": seed}
            if fold is not None:
                f.update(_val_folds=int(folds), _val_fold=int(fold))
            jobs.append((v, seed, fold))
            flats.append(f)
    return jobs, flats


def _check_folds(cfg, bundle, folds):
    """Fail before any training if the folds cannot be built."""
    from .data.loaders import split_settings, subject_folds, subject_ids
    s = split_settings(cfg)
    if s["val_split"] != "subject":
        raise SystemExit("study --folds holds people out, so it needs "
                         "search.val_split: subject")
    ids = subject_ids(bundle.train)
    if ids is None:
        raise SystemExit("study --folds needs a dataset that names each clip's person")
    subject_folds(ids, int(folds), s["seed"])          # raises on a bad count


def run_option_study(cfg, flat, seeds=3, epochs=None, only=None, folds=0):
    """Build the variants of `flat`, train each with every seed (or on every
    fold), write the tables and print the summary. Returns the summary rows."""
    import types
    from .pipeline import results_dir
    from .search import run_jobs, _warm_frame_caches
    from .spaces import data_facts, MULTI_BIT_CODINGS, MAX_INPUT_BITS
    from .data import build_dataset

    out_dir = os.path.join(results_dir(cfg), "study")
    os.makedirs(out_dir, exist_ok=True)
    flat = {k: v for k, v in flat.items() if not k.startswith("_")}
    T = int(flat.get("T") or cfg["encoding"].get("T", 16))
    bundle = build_dataset({**cfg["dataset"], "T": T})
    flat.update(data_facts(bundle))           # the dataset's facts win, as in replays
    coding = cfg["encoding"].get("coding") or ("passthrough" if bundle.is_event else "levels")
    max_bits = int(cfg["search"].get("max_input_bits") or MAX_INPUT_BITS) \
        if coding in MULTI_BIT_CODINGS else 1
    if epochs:
        flat["epochs"] = int(epochs)
    if folds:
        _check_folds(cfg, bundle, folds)
    reps = replicates(int(cfg["run"].get("seed", 1)), seeds, folds)

    variants = option_variants(flat, max_bits, only)
    trained = [v for v in variants if v["flat"] is not None]
    each = (f"{folds} subject folds (every training person held out once)" if folds
            else f"{len(reps)} seeds")
    print(f"option study: {len(trained)} variants x {each} = {len(trained) * len(reps)} "
          f"runs of {flat.get('epochs')} epochs, no early stopping", flush=True)
    for v in variants:
        if v["flat"] is None:
            print(f"  skipped {v['name']}: {v['reason']}", flush=True)
    # every frame cache a variant needs, before parallel trials race to build one
    _warm_frame_caches(cfg, sorted({int(v["flat"]["T"]) for v in trained}),
                       types.SimpleNamespace(log=print))

    jobs, flats = job_flats(trained, reps, folds)
    results = run_jobs(cfg, out_dir, flats, cfg["run"].get("name", "search") + "_study")
    rows = [{"variant": v["name"], "option": v["option"], "level": v["level"], "seed": seed,
             "fold": fold, **m} for (v, seed, fold), m in zip(jobs, results)]
    _write_csv(os.path.join(out_dir, "study.csv"), rows)
    summary = summarize_study(variants, rows)
    _write_csv(os.path.join(out_dir, "study_summary.csv"), summary)

    print(f"\n{'variant':26s} {'mean':>7s} {'worst':>7s} {'vs baseline':>12s}   verdict")
    for r in summary:
        diff = "" if r["diff_vs_baseline"] is None else \
            f"{r['diff_vs_baseline']:+.4f}" + (f" +-{r['diff_se']:.3f}" if r["diff_se"] else "")
        mean = "" if r["mean_val_accuracy"] is None else f"{r['mean_val_accuracy']:.4f}"
        worst = "" if r["worst_val_accuracy"] is None else f"{r['worst_val_accuracy']:.4f}"
        print(f"{r['variant']:26s} {mean:>7s} {worst:>7s} {diff:>12s}   {r['verdict']}")
    print(f"\nwritten to {out_dir}")
    return summary
