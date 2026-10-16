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

Writes into <results_dir>/study/:
    study.csv           one row per variant and seed
    study_summary.csv   per variant: mean, spread, the paired difference from
                        the baseline with its standard error, and a verdict
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
        "augment": "on" if flat.get("use_augment") else "off",
        "conv_dropout": "on" if flat.get("use_conv_dropout") else "off",
        "rate_penalty": "on" if flat.get("use_rate_penalty") else "off",
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
    raise ValueError(f"unknown option {option!r}")


def option_variants(flat, max_input_bits=4, only=None):
    """The baseline, then one variant per other level of each choice.

    Returns dicts with name, option, level, flat (None when skipped) and
    reason (why it was skipped, else "")."""
    now = current_levels(flat)
    out = [{"name": "baseline", "option": None, "level": None, "flat": dict(flat),
            "reason": ""}]
    for option, levels in BALANCED_FACTORS.items():
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


def summarize_study(variants, rows):
    """Per variant: mean and spread over seeds, and the paired difference from
    the baseline (same seed) with its standard error and a verdict."""
    base = {r["seed"]: float(r["val_accuracy"] or 0.0) for r in rows if r["variant"] == "baseline"}
    out = []
    for v in variants:
        mine = [r for r in rows if r["variant"] == v["name"]]
        vals = [float(r["val_accuracy"] or 0.0) for r in mine]
        diffs = [float(r["val_accuracy"] or 0.0) - base[r["seed"]] for r in mine
                 if r["seed"] in base]
        row = {"variant": v["name"], "seeds": len(vals),
               "mean_val_accuracy": round(statistics.mean(vals), 4) if vals else None,
               "std_val_accuracy": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
               "diff_vs_baseline": None, "diff_se": None, "verdict": v["reason"] or ""}
        if v["name"] != "baseline" and diffs:
            d = statistics.mean(diffs)
            se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else None
            row["diff_vs_baseline"] = round(d, 4)
            row["diff_se"] = round(se, 4) if se is not None else None
            if se is None:
                row["verdict"] = "one seed: no verdict"
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


def run_option_study(cfg, flat, seeds=3, epochs=None, only=None):
    """Build the variants of `flat`, train each with every seed, write the
    tables and print the summary. Returns the summary rows."""
    from .pipeline import results_dir
    from .search import run_jobs
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
    run_seed = int(cfg["run"].get("seed", 1))
    seed_list = [run_seed + 1000 * (j + 1) for j in range(int(seeds))]

    variants = option_variants(flat, max_bits, only)
    trained = [v for v in variants if v["flat"] is not None]
    print(f"option study: {len(trained)} variants x {len(seed_list)} seeds = "
          f"{len(trained) * len(seed_list)} runs of {flat.get('epochs')} epochs, "
          f"no early stopping", flush=True)
    for v in variants:
        if v["flat"] is None:
            print(f"  skipped {v['name']}: {v['reason']}", flush=True)

    jobs = [(v, seed) for v in trained for seed in seed_list]
    results = run_jobs(cfg, out_dir, [{**v["flat"], "seed": seed} for v, seed in jobs],
                       cfg["run"].get("name", "search") + "_study")
    rows = [{"variant": v["name"], "option": v["option"], "level": v["level"], "seed": seed,
             **m} for (v, seed), m in zip(jobs, results)]
    _write_csv(os.path.join(out_dir, "study.csv"), rows)
    summary = summarize_study(variants, rows)
    _write_csv(os.path.join(out_dir, "study_summary.csv"), summary)

    print(f"\n{'variant':26s} {'mean':>7s} {'vs baseline':>12s}   verdict")
    for r in summary:
        diff = "" if r["diff_vs_baseline"] is None else \
            f"{r['diff_vs_baseline']:+.4f}" + (f" +-{r['diff_se']:.3f}" if r["diff_se"] else "")
        mean = "" if r["mean_val_accuracy"] is None else f"{r['mean_val_accuracy']:.4f}"
        print(f"{r['variant']:26s} {mean:>7s} {diff:>12s}   {r['verdict']}")
    print(f"\nwritten to {out_dir}")
    return summary
