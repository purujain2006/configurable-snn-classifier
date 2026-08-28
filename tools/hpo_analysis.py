"""What the recorded trials support, and what they do not.

Hyperparameter searches produce a leaderboard, and a leaderboard invites you to
read the top row as a finding. It usually is not one. This computes what the
data can actually carry:

  NOISE FLOOR      How much a score moves when nothing about the model changes.
                   Measured from the final epochs of a settled run, where the
                   learning rate is near zero and the weights barely move.
                   Everything else is judged against this.

  SIGNAL SHARE     How much of the spread between trials is real configuration
                   difference rather than that noise. If the answer is small,
                   the ranking is mostly luck and no amount of analysis fixes it.

  PER-KNOB EFFECT  For each hyperparameter, whether accuracy depends on it at
                   all, by permutation rather than by an asymptotic test, since
                   the sample is small and the distribution is not normal.
                   Corrected for testing many knobs at once.

  FEASIBILITY      Which settings get rejected by the connection limits. This
                   is arithmetic rather than training, so it is exact and needs
                   no statistics beyond counting.

  SELECTION BIAS   How much the best-of-N score overstates the truth, and what
                   the winner is expected to be worth on a fresh evaluation.

  POWER            The difference a search this size could have detected. Mostly
                   this explains why an inconclusive result is inconclusive.

    python tools/hpo_analysis.py <dir-with-trial-data> [-o outdir]
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

RNG = random.Random(0)
PERM = 20000

# Hyperparameters, split by how they must be tested. Anything not listed is
# an outcome or bookkeeping rather than a knob.
KNOBS = ["downsample_mode", "norm", "optimizer", "scheduler", "final_reduction",
         "trainable_tau", "trainable_threshold", "depth", "channels",
         "kernel_size", "stride", "resize_to", "T", "tau", "fc_layers",
         "dropout_rate", "lr", "weight_decay", "label_smoothing", "grad_clip",
         "tdbn_alpha"]

# Whether a knob is categorical is a property of the DATA, not a list written
# once. tau was sampled as a float in the early searches and snapped to the
# hardware's eight legal values later; hard-coding it either way makes one of
# those two datasets nonsense -- 59 "levels" with one trial each, or a rank
# correlation over eight ties.
MAX_LEVELS_FOR_CATEGORICAL = 8


def classify(rows, knob):
    """categorical, continuous, or None when there is nothing to test."""
    vals = [r.get(knob) for r in rows if r.get(knob) not in (None, "")]
    if len(vals) < 6:
        return None
    uniq = set(map(str, vals))
    if len(uniq) < 2:
        return None
    numeric = [num(v) for v in vals]
    if all(v is not None for v in numeric) and len(uniq) > MAX_LEVELS_FOR_CATEGORICAL:
        return "continuous"
    return "categorical"


# ----------------------------------------------------------------- loading
def load_leaderboards(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "leaderboard.csv"),
                                 recursive=True)):
        run = os.path.relpath(os.path.dirname(path), root)
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_run"] = run
                rows.append(r)
    return rows


def load_jsonl(root, name):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "**", name), recursive=True)):
        run = os.path.relpath(os.path.dirname(path), root)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec["_run"] = run
                    out.append(rec)
    return out


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------- statistics
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def var(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def kruskal_stat(groups):
    allv = [x for g in groups for x in g]
    n = len(allv)
    r = ranks(allv)
    out, i = 0.0, 0
    for g in groups:
        s = sum(r[i:i + len(g)])
        out += s * s / len(g)
        i += len(g)
    return 12.0 / (n * (n + 1)) * out - 3 * (n + 1)


def perm_p_groups(values, labels, n_perm=PERM):
    """Permutation p for 'accuracy depends on this categorical knob'."""
    by = defaultdict(list)
    for v, l in zip(values, labels):
        by[l].append(v)
    groups = [g for g in by.values() if len(g) >= 2]
    if len(groups) < 2:
        return None, None, None
    obs = kruskal_stat(groups)
    sizes = [len(g) for g in groups]
    pool = [x for g in groups for x in g]
    hits = 0
    for _ in range(n_perm):
        RNG.shuffle(pool)
        i, sh = 0, []
        for s in sizes:
            sh.append(pool[i:i + s])
            i += s
        if kruskal_stat(sh) >= obs:
            hits += 1
    n = sum(sizes)
    # epsilon-squared: the share of rank variance the grouping explains
    eps2 = max(0.0, (obs - len(groups) + 1) / (n - len(groups)))
    return obs, (hits + 1) / (n_perm + 1), eps2


def spearman(x, y, n_perm=PERM // 4):
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    def corr(a, b):
        ma, mb = mean(a), mean(b)
        num_ = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
        return num_ / den if den else 0.0
    obs = corr(rx, ry)
    hits = 0
    shuf = list(rx)
    for _ in range(n_perm):
        RNG.shuffle(shuf)
        if abs(corr(shuf, ry)) >= abs(obs):
            hits += 1
    return obs, (hits + 1) / (n_perm + 1)


def benjamini_hochberg(pairs):
    """[(key, p)] -> {key: q}. Controls the false discovery rate."""
    items = sorted([(p, k) for k, p in pairs if p is not None])
    m, q, prev = len(items), {}, 1.0
    for i in range(m - 1, -1, -1):
        p, k = items[i]
        prev = min(prev, p * m / (i + 1))
        q[k] = min(1.0, prev)
    return q


def expected_max_of_n(n):
    """E[max] of n standard normals, in sd units. Blom's approximation."""
    if n < 2:
        return 0.0
    from math import sqrt, log, pi
    # accurate enough over the range a search covers
    return sqrt(2 * log(n)) - (log(log(n)) + log(4 * pi)) / (2 * sqrt(2 * log(n)))


# --------------------------------------------------------------- analyses
def noise_floor(root):
    """Score spread with the model effectively frozen.

    The point is to measure the score moving while the MODEL does not, so the
    epochs used must be ones where the learning rate has actually decayed. An
    earlier version took the last 12 epochs of the longest trial regardless,
    picked a run that was still climbing, and reported its training progress as
    noise -- inflating the floor and making every real effect look marginal.

    So: only epochs whose learning rate is below a small fraction of that
    trial's starting rate, and only trials that have several such epochs.
    Averaged across trials, because one trial's tail is a small sample.
    """
    rows = load_jsonl(root, "trial_progress.jsonl")
    by = defaultdict(list)
    for r in rows:
        acc, ep, lr = num(r.get("val_accuracy")), r.get("epoch"), num(r.get("lr"))
        if acc is not None and ep is not None:
            by[(r["_run"], r.get("trial_id"))].append((ep, acc, lr))

    sds, used, tails = [], 0, []
    for series in by.values():
        series.sort()
        lrs = [lr for _, _, lr in series if lr is not None]
        if len(series) < 8 or not lrs:
            continue
        cut = max(lrs) * 0.05          # "settled" = under 5% of the peak rate
        tail = [a for _, a, lr in series if lr is not None and lr <= cut]
        if len(tail) < 4:
            continue
        sds.append(math.sqrt(var(tail)))
        tails.extend(tail)
        used += 1
    if not sds:
        return None
    return {"trials": used, "n": len(tails), "sd": mean(sds),
            "mean": mean(tails), "min": min(tails), "max": max(tails)}


def signal_share(scores, noise_sd):
    """How much of the between-trial spread is real, not noise."""
    total = var(scores)
    if total <= 0:
        return None
    noise = noise_sd ** 2
    return {"total_sd": math.sqrt(total), "noise_sd": noise_sd,
            "signal_sd": math.sqrt(max(0.0, total - noise)),
            "share": max(0.0, (total - noise) / total)}


def trained_only(rows, target):
    """Trials that actually trained.

    A configuration rejected by the connection limits is recorded with a score
    of 0. Those zeros are a fact about the ARITHMETIC, not about training, and
    leaving them in makes every knob that correlates with infeasibility look
    like it ruins accuracy. depth=5 scoring 0.03 is "usually rejected", not
    "trains badly". Feasibility gets its own section; this one is about the
    trials that reached a GPU.
    """
    out = []
    for r in rows:
        v = num(r.get(target))
        if v is None or v <= 0.0:
            continue
        if str(r.get("feasible", "true")).lower() in ("false", "0"):
            continue
        out.append(r)
    return out


def knob_effects(rows, target):
    """One test per hyperparameter, then a false-discovery correction."""
    scored = trained_only(rows, target)
    out, pvals = [], []
    for knob in KNOBS:
        kind = classify(scored, knob)
        if kind is None:
            continue
        pairs = [(num(r[target]), r.get(knob)) for r in scored
                 if r.get(knob) not in (None, "")]
        y = [v for v, _ in pairs]
        if kind == "categorical":
            g = [str(l) for _, l in pairs]
            _, p, eps2 = perm_p_groups(y, g)
            if p is None:
                continue
            by = defaultdict(list)
            for v, l in zip(y, g):
                by[l].append(v)
            by = {k: v for k, v in by.items() if len(v) >= 2}
            if len(by) < 2:
                continue
            bl = max(by, key=lambda k: mean(by[k]))
            wl = min(by, key=lambda k: mean(by[k]))
            out.append({"knob": knob, "kind": "categorical", "n": len(y),
                        "levels": len(by), "p": p, "effect": eps2,
                        "best_level": f"{bl} (n={len(by[bl])})",
                        "best_mean": mean(by[bl]),
                        "worst_level": f"{wl} (n={len(by[wl])})",
                        "worst_mean": mean(by[wl]),
                        "spread": mean(by[bl]) - mean(by[wl])})
        else:
            x = [num(l) for _, l in pairs]
            keep = [(a, b) for a, b in zip(y, x) if b is not None]
            if len(keep) < 8:
                continue
            y = [a for a, _ in keep]
            x = [b for _, b in keep]
            rho, p = spearman(x, y)
            out.append({"knob": knob, "kind": "continuous", "n": len(y),
                        "levels": len(set(x)), "p": p, "effect": abs(rho),
                        "best_level": f"rho={rho:+.3f}", "best_mean": "",
                        "worst_level": "", "worst_mean": "", "spread": ""})
        pvals.append((knob, out[-1]["p"]))
    q = benjamini_hochberg(pvals)
    for r in out:
        r["q"] = q.get(r["knob"])
    out.sort(key=lambda r: (r["q"] if r["q"] is not None else 1.0, -r["effect"]))
    return out


def feasibility(rows):
    """Which settings the connection limits reject. Counting, not inference."""
    out = []
    for knob in KNOBS:
        if classify(rows, knob) != "categorical":
            continue
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            lvl = r.get(knob)
            if lvl in (None, ""):
                continue
            ok = str(r.get("feasible", "")).lower() in ("true", "1")
            by[str(lvl)][0 if ok else 1] += 1
        if len(by) < 2:
            continue
        for lvl, (ok, bad) in sorted(by.items()):
            n = ok + bad
            if n:
                out.append({"knob": knob, "level": lvl, "n": n,
                            "rejected": bad, "reject_rate": bad / n})
    return out


def convergence(rows, target):
    """Best-so-far against trial index: did the search still have room?"""
    seq = [num(r.get(target)) for r in rows if num(r.get(target)) is not None]
    best, curve = -1.0, []
    for i, v in enumerate(seq, 1):
        best = max(best, v)
        curve.append((i, best))
    if not curve:
        return None
    last_gain = max((i for i, b in curve if b == curve[-1][1]), default=len(curve))
    return {"n": len(curve), "final_best": curve[-1][1],
            "reached_at": last_gain, "idle_since": len(curve) - last_gain,
            "curve": curve}


def selection_bias(scores, noise_sd):
    """How much the best-of-N overstates, and what to expect on a rerun."""
    n = len(scores)
    if n < 2 or noise_sd <= 0:
        return None
    z = expected_max_of_n(n)
    return {"n": n, "observed_max": max(scores), "noise_sd": noise_sd,
            "expected_optimism": z * noise_sd,
            "shrunk_estimate": max(scores) - z * noise_sd}


def power(noise_sd, per_group):
    """Smallest difference two groups of this size could separate."""
    if per_group < 2 or noise_sd <= 0:
        return None
    se = noise_sd * math.sqrt(2.0 / per_group)
    return {"per_group": per_group, "se_of_difference": se,
            "detectable_80pct": 2.8 * se}


def knob_collinearity(rows, target, effects, top=6):
    """How much the leading knobs move together.

    Optuna samples adaptively: once it favours a region it draws many similar
    configurations, so knobs co-vary. A univariate test then credits the SAME
    variance to every knob in the correlated block, which is why a table like
    the one above can show a dozen knobs all clearing correction. Reporting the
    pairwise agreement is the cheap honest version of the multivariate analysis
    that this sample is too small to support.
    """
    trained = trained_only(rows, target)
    names = [e["knob"] for e in effects[:top]]
    cols = {}
    for k in names:
        vals = []
        for r in trained:
            v = num(r.get(k))
            if v is None:
                v = None if r.get(k) in (None, "") else float(hash(str(r.get(k))) % 997)
            vals.append(v)
        if all(v is not None for v in vals) and len(set(vals)) > 1:
            cols[k] = vals
    out = []
    keys = list(cols)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = ranks(cols[keys[i]]), ranks(cols[keys[j]])
            ma, mb = mean(a), mean(b)
            nu = sum((x - ma) * (y - mb) for x, y in zip(a, b))
            de = math.sqrt(sum((x - ma) ** 2 for x in a)
                           * sum((y - mb) ** 2 for y in b))
            if de:
                out.append((keys[i], keys[j], nu / de))
    out.sort(key=lambda t: -abs(t[2]))
    return out


# ----------------------------------------------------------------- report
def fmt_p(p):
    if p is None:
        return "     -"
    return "<1e-4 " if p < 1e-4 else f"{p:.4f}"


def write_report(root, out_dir, replicates=()):
    rows = load_leaderboards(root)
    if not rows:
        sys.exit(f"no leaderboard.csv found under {root}")
    target = "val_accuracy"
    trained = trained_only(rows, target)
    scored = [num(r[target]) for r in trained]
    all_scored = [num(r.get(target)) for r in rows
                  if num(r.get(target)) is not None]

    nf = noise_floor(root)
    within_sd = nf["sd"] if nf else 0.0
    rep_sd = math.sqrt(var(list(replicates))) if len(replicates) >= 2 else None
    # Comparing two CONFIGURATIONS means comparing two separate runs, so the
    # uncertainty that matters is run-to-run, not epoch-to-epoch within one run.
    # They are not the same quantity and the second is much smaller.
    sd = rep_sd if rep_sd else within_sd
    lines = []
    w = lines.append

    w("# What the search data supports")
    w("")
    w(f"Source: `{root}`  ")
    w(f"Trials recorded: **{len(rows)}** across "
      f"{len(set(r['_run'] for r in rows))} searches, "
      f"**{len(all_scored)}** scored, of which **{len(scored)}** actually "
      f"trained. The rest were rejected by the connection limits and "
      f"recorded as 0, so they are excluded from every effect below.")
    w("")

    # --- noise floor
    w("## 1. The noise floor")
    w("")
    if nf:
        w(f"From {nf['n']} epochs across {nf['trials']} trials, keeping only "
          f"epochs whose learning rate had fallen below 5% of its peak. The "
          f"model is barely moving there, so whatever the score does is not "
          f"the model doing it.")
        w("")
        w("| | |")
        w("|---|---|")
        w("| mean | {nf['mean']:.4f} |")
        w("| min to max | {nf['min']:.4f} to {nf['max']:.4f} |")
        w("| **sd** | **{nf['sd']:.4f}** |")
        w("")
        w(f"So a score moves about **{nf['sd']*176:.1f} clips** with the model "
          f"effectively frozen. Two trials closer together than roughly "
          f"{2*nf['sd']:.3f} are indistinguishable.")
    else:
        w("Not enough per-epoch data to measure it.")
    w("")

    if rep_sd:
        w("### Between runs, which is the number that matters")
        w("")
        w("Repeated runs of ONE configuration scored: "
          + ", ".join(f"{r:.4f}" for r in replicates))
        w("")
        w("| | |")
        w("|---|---|")
        w(f"| within a run, model frozen (sd) | {within_sd:.4f} "
          f"({within_sd*176:.1f} clips) |")
        w(f"| **between runs, same config (sd)** | **{rep_sd:.4f}** "
          f"({rep_sd*176:.1f} clips) |")
        w(f"| ratio | **{rep_sd/within_sd:.0f}x** |" if within_sd else "")
        w("")
        w("The second is the one to judge configurations by. Two trials are "
          "different runs, so the epoch-to-epoch figure is the wrong yardstick "
          f"and understates the uncertainty by about {rep_sd/within_sd:.0f} times."
          if within_sd else "")
        w("")
        w(f"Everything below uses **{sd:.4f}**.")
        w("")
    elif within_sd:
        w("> No replicate runs supplied, so the only measurable noise is "
          "epoch-to-epoch within a single run. That is a FLOOR on the "
          "uncertainty and probably well below the truth: two trials are two "
          "separate runs, and training varies run to run by more than a frozen "
          "model's score does. Pass `--replicates` with several runs of one "
          "configuration to measure it properly.")
        w("")

    # --- signal share
    w("## 2. How much of the spread is real")
    w("")
    ss = signal_share(scored, sd) if sd else None
    if ss:
        w("| | |")
        w("|---|---|")
        w("| spread between trials (sd) | {ss['total_sd']:.4f} |")
        w("| noise (sd) | {ss['noise_sd']:.4f} |")
        w("| real configuration effect (sd) | {ss['signal_sd']:.4f} |")
        w("| **share of variance that is real** | **{ss['share']:.1%}** |")
        w("")
        if ss["share"] > 0.9:
            w("Configurations genuinely differ. Ranking them is meaningful, "
              "even though the exact value of any single score is not.")
        else:
            w("A large part of the ranking is noise. Treat ordering with care.")
    w("")

    # --- knobs
    w("## 3. Which hyperparameters matter")
    w("")
    w("Permutation tests, since the sample is small and the distribution is not "
      "normal. `q` is the false-discovery-corrected p-value across all knobs "
      "tested together, so it already accounts for looking at many at once.")
    w("")
    eff = knob_effects(rows, target)
    w("| knob | kind | n | effect | p | q | best | worst | gap |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in eff:
        bl = f"{r['best_level']}" + (f" ({r['best_mean']:.3f})"
                                     if isinstance(r["best_mean"], float) else "")
        wl = f"{r['worst_level']}" + (f" ({r['worst_mean']:.3f})"
                                      if isinstance(r["worst_mean"], float) else "")
        gap = f"{r['spread']:+.3f}" if isinstance(r["spread"], float) else ""
        w(f"| `{r['knob']}` | {r['kind'][:4]} | {r['n']} | {r['effect']:.3f} | "
          f"{fmt_p(r['p'])} | {fmt_p(r['q'])} | {bl} | {wl} | {gap} |")
    w("")
    sig = [r for r in eff if r["q"] is not None and r["q"] < 0.10]
    if sig:
        w("**Survives correction (q < 0.10):** "
          + ", ".join(f"`{r['knob']}`" for r in sig))
    else:
        w("**Nothing survives correction.** With this many trials that is the "
          "expected outcome rather than evidence the knobs do not matter. "
          "See the power section.")
    w("")
    coll = knob_collinearity(rows, target, eff)
    strong = [c for c in coll if abs(c[2]) > 0.4]
    if strong:
        w("### These knobs move together")
        w("")
        w("Optuna samples adaptively, so once it favours a region it draws many "
          "similar configurations. When two knobs co-vary, a one-at-a-time test "
          "credits both with the same shared variance, and the table above "
          "overstates how many knobs independently matter.")
        w("")
        w("| knob A | knob B | rank correlation |")
        w("|---|---|---|")
        for a, b, r in strong[:8]:
            w("| `{a}` | `{b}` | {r:+.2f} |")
        w("")
    w("")

    # --- feasibility
    w("## 4. What the connection limits reject")
    w("")
    w("Arithmetic on the plan, not training, so these counts are exact.")
    w("")
    fe = [f for f in feasibility(rows) if f["rejected"]]
    if fe:
        fe.sort(key=lambda f: -f["reject_rate"])
        w("| knob | setting | trials | rejected | rate |")
        w("|---|---|---|---|---|")
        for f in fe[:15]:
            w(f"| `{f['knob']}` | {f['level']} | {f['n']} | {f['rejected']} | "
              f"{f['reject_rate']:.0%} |")
    else:
        w("No infeasible trials recorded.")
    w("")

    # --- convergence
    w("## 5. Had the search finished?")
    w("")
    cv = convergence(rows, target)
    if cv:
        w(f"Best score reached **{cv['final_best']:.4f}** at trial "
          f"**{cv['reached_at']}** of {cv['n']}, then {cv['idle_since']} trials "
          f"with no improvement.")
        w("")
        if cv["idle_since"] < cv["n"] * 0.3:
            w("The best arrived late. A longer search would probably still be "
              "finding better configurations, so this budget was the binding "
              "constraint rather than the space.")
        else:
            w("The best arrived early and a long tail found nothing better, "
              "which suggests the space rather than the budget is the limit.")
    w("")

    # --- selection bias
    w("## 6. What the winner is actually worth")
    w("")
    sb = selection_bias(scored, sd) if sd else None
    if sb:
        w(f"Taking the best of {sb['n']} noisy scores overstates by about "
          f"`E[max of n] x sd`.")
        w("")
        w("| | |")
        w("|---|---|")
        w("| reported best | {sb['observed_max']:.4f} |")
        w(f"| expected optimism | {sb['expected_optimism']:.4f} "
          f"({sb['expected_optimism']*176:.1f} clips) |")
        w("| **expected on a rerun** | **{sb['shrunk_estimate']:.4f}** |")
        w("")
        w("This is the floor on the correction, not the whole of it. It counts "
          "only evaluation noise. Run-to-run variation in training adds more.")
    w("")

    # --- power
    w("## 7. What a search this size could detect")
    w("")
    for per in (5, 10, 20, 50):
        pw = power(sd, per) if sd else None
        if pw:
            w(f"- {per:>2} trials per group: can separate differences above "
              f"**{pw['detectable_80pct']:.4f}** "
              f"({pw['detectable_80pct']*176:.1f} clips)")
    w("")
    w("Anything smaller than that needs more trials, not more analysis.")
    w("")

    os.makedirs(out_dir, exist_ok=True)
    rp = os.path.join(out_dir, "hpo_report.md")
    with open(rp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    cp = os.path.join(out_dir, "hpo_knob_effects.csv")
    with open(cp, "w", newline="", encoding="utf-8") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=[
            "knob", "kind", "n", "levels", "effect", "p", "q",
            "best_level", "best_mean", "worst_level", "worst_mean", "spread"])
        wcsv.writeheader()
        for r in eff:
            wcsv.writerow(r)

    fp = os.path.join(out_dir, "hpo_feasibility.csv")
    with open(fp, "w", newline="", encoding="utf-8") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=["knob", "level", "n", "rejected",
                                              "reject_rate"])
        wcsv.writeheader()
        for f in feasibility(rows):
            wcsv.writerow(f)

    print("\n".join(lines))
    print(f"\nwrote {rp}\n      {cp}\n      {fp}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory containing leaderboard.csv files")
    ap.add_argument("-o", "--out", default=".", help="where to write the report")
    ap.add_argument("--replicates", default="",
                    help="comma-separated scores from REPEATED runs of ONE "
                         "configuration. Without these the report can only "
                         "measure within-run noise, which understates how much "
                         "a rerun moves.")
    a = ap.parse_args()
    reps = [float(x) for x in a.replicates.split(',') if x.strip()]
    return write_report(a.root, a.out, reps)


if __name__ == "__main__":
    sys.exit(main())
