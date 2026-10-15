"""Reproduce the statistics recorded in Practice2.py's search-space comments.

Those comments carry claims like "chi-square p=6e-4, Cramer's V=0.59" that
narrowed the search space. They were computed once, by hand, and never checked
back in. This reads the raw trial records and recomputes them, so each claim is
either confirmed or corrected against the data it came from.

No scipy: the tests are implemented here, and p-values come from permutation
rather than an asymptotic approximation, which is the safer choice at n=30.
"""
import json, math, os, sys
from collections import Counter
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
PERM = 20000


# ---------------------------------------------------------------- statistics
def chi2_contingency(table):
    """Pearson chi-square on a 2-D count table. Returns (chi2, dof, cramers_v)."""
    t = np.asarray(table, dtype=float)
    n = t.sum()
    if n == 0:
        return 0.0, 0, 0.0
    expected = np.outer(t.sum(1), t.sum(0)) / n
    mask = expected > 0
    chi2 = float(((t[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    dof = (t.shape[0] - 1) * (t.shape[1] - 1)
    v = math.sqrt(chi2 / (n * min(t.shape[0] - 1, t.shape[1] - 1))) if n and min(t.shape) > 1 else 0.0
    return chi2, dof, v


def perm_p_chi2(labels, groups, n_perm=PERM):
    """Permutation p-value for association between two categorical vectors."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    lv, gv = np.unique(labels), np.unique(groups)
    def stat(lab):
        tab = np.array([[np.sum((lab == a) & (groups == b)) for b in gv] for a in lv])
        return chi2_contingency(tab)[0]
    obs = stat(labels)
    perm = np.array([stat(RNG.permutation(labels)) for _ in range(n_perm)])
    return obs, float((np.sum(perm >= obs) + 1) / (n_perm + 1))


def rank(x):
    """Average ranks, ties shared."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y, n_perm=PERM):
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perm = np.array([np.corrcoef(RNG.permutation(rx), ry)[0, 1] for _ in range(n_perm // 4)])
    p = float((np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1))
    return rho, p


def mannwhitney(a, b, n_perm=PERM):
    """Two-sided permutation test on the rank-sum. Also returns the effect size
    (common-language: probability a random a exceeds a random b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    combined = np.concatenate([a, b])
    na = len(a)
    def stat(v):
        return rank(v)[:na].sum()
    obs = stat(combined)
    perm = np.array([stat(RNG.permutation(combined)) for _ in range(n_perm // 4)])
    centre = perm.mean()
    p = float((np.sum(np.abs(perm - centre) >= abs(obs - centre)) + 1) / (len(perm) + 1))
    greater = sum(1 for x in a for y in b if x > y)
    ties = sum(1 for x in a for y in b if x == y)
    cles = (greater + 0.5 * ties) / (len(a) * len(b)) if len(a) and len(b) else float("nan")
    return p, cles


def kruskal(groups, n_perm=PERM):
    groups = [np.asarray(g, float) for g in groups if len(g)]
    if len(groups) < 2:
        return float("nan"), float("nan")
    allv = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    n = len(allv)
    def stat(v):
        r, out, i = rank(v), 0.0, 0
        for s in sizes:
            out += (r[i:i + s].sum() ** 2) / s
            i += s
        return 12.0 / (n * (n + 1)) * out - 3 * (n + 1)
    obs = stat(allv)
    perm = np.array([stat(RNG.permutation(allv)) for _ in range(n_perm // 4)])
    return obs, float((np.sum(perm >= obs) + 1) / (len(perm) + 1))


# ---------------------------------------------------------------- data load
DEPLOY = []


def load(root):
    """Every trial from every search: feasible ones from the leaderboards,
    rejected ones from infeasible.jsonl, tagged with which run they came from."""
    rows = []
    for dirpath, _dirnames, filenames in os.walk(root):
        run = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if "leaderboard.csv" in filenames:
            df = pd.read_csv(os.path.join(dirpath, "leaderboard.csv"))
            df["run"], df["source"] = run, "leaderboard"
            rows.append(df)
        if "trials.jsonl" in filenames:
            recs = []
            with open(os.path.join(dirpath, "trials.jsonl"), encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
            if recs:
                dep = pd.DataFrame(recs)
                keep = [c for c in ("trial_id", "hw_val_accuracy", "float_val_accuracy",
                                    "quant_gap", "weight_clip_frac", "min_threshold",
                                    "deployable", "deploy_reasons", "epochs_run",
                                    "stopped_early") if c in dep.columns]
                DEPLOY.append(dep[keep].assign(run=run))
        if "infeasible.jsonl" in filenames:
            recs = []
            with open(os.path.join(dirpath, "infeasible.jsonl"), encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    cfg = dict(r.get("config", {}))
                    # per-layer spaces use ch_0/ch_1..., uniform uses channels
                    chans = [v for k, v in cfg.items() if k.startswith("ch_")]
                    if chans and "channels" not in cfg:
                        cfg["channels"] = max(chans)
                    ks = [v for k, v in cfg.items() if k.startswith("k_")]
                    if ks and "kernel_size" not in cfg:
                        cfg["kernel_size"] = max(ks)
                    cfg.update(trial_id=r.get("trial_id"), val_accuracy=0.0, feasible=False,
                               violations=r.get("violations", ""), run=run, source="infeasible")
                    recs.append(cfg)
            if recs:
                rows.append(pd.DataFrame(recs))
    if not rows:
        sys.exit(f"no trial data under {root}")
    df = pd.concat(rows, ignore_index=True)
    df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1"])
    return df


def hdr(s):
    print(f"\n{'=' * 76}\n{s}\n{'=' * 76}")


def verdict(claim, holds, detail):
    mark = "CONFIRMED" if holds else "DOES NOT HOLD"
    print(f"  [{mark}] {claim}\n      {detail}")


def main(root):
    df = load(root)
    hdr("SAMPLE")
    print(f"  {len(df)} trials across {df['run'].nunique()} runs")
    for run, g in df.groupby("run"):
        print(f"    {run:<28} {len(g):>3} trials  "
              f"({int(g['feasible'].sum())} feasible, {int((~g['feasible']).sum())} rejected)")
    # a pruned trial has no deploy report, so its val_accuracy is blank.
    # Those rows count for feasibility but cannot enter an accuracy test.
    df["scored"] = df["feasible"] & df["val_accuracy"].notna()
    feas = df[df["scored"]]
    print(f"  feasible: {int(df['feasible'].sum())};  of those, scored: {len(feas)} "
          f"({int(df['feasible'].sum()) - len(feas)} pruned before any score landed)")
    print(f"  best recorded accuracy: {feas['val_accuracy'].max():.4f}")

    # ---- claim 1: channel count drives infeasibility ----
    hdr("CLAIM 1  channels drive infeasibility (recorded: chi-square p=6e-4, Cramer's V=0.59)")
    d = df.dropna(subset=["channels"]).copy()
    d["ch"] = d["channels"].astype(float).astype(int)
    tab, rowlab = [], []
    print(f"  {'channels':>9} {'n':>4} {'feasible':>9} {'rate':>7}")
    for ch in sorted(d["ch"].unique()):
        g = d[d["ch"] == ch]
        tab.append([int(g["feasible"].sum()), int((~g["feasible"]).sum())])
        rowlab.append(ch)
        print(f"  {ch:>9} {len(g):>4} {int(g['feasible'].sum()):>9} {g['feasible'].mean():>6.0%}")
    chi2, dof, v = chi2_contingency(tab)
    _, p = perm_p_chi2(d["feasible"].values.astype(int), d["ch"].values)
    print(f"\n  chi2={chi2:.2f} dof={dof}  Cramer's V={v:.2f}  permutation p={p:.4f}")
    verdict("wider channels are less feasible", v > 0.3 and p < 0.05,
            f"V={v:.2f} (recorded 0.59), p={p:.4f} (recorded 6e-4)")

    # ---- claim 2: shallow depth wins ----
    hdr("CLAIM 2  shallow networks score better (recorded: Kruskal-Wallis q=0.08)")
    f = feas.dropna(subset=["depth"]).copy()
    f["depth"] = f["depth"].astype(float).astype(int)
    groups, labels = [], []
    print(f"  {'depth':>6} {'n':>4} {'median':>8} {'best':>8}")
    for dep in sorted(f["depth"].unique()):
        g = f[f["depth"] == dep]["val_accuracy"].values
        groups.append(g); labels.append(dep)
        print(f"  {dep:>6} {len(g):>4} {np.median(g):>8.4f} {g.max():>8.4f}")
    H, p = kruskal(groups)
    verdict("depth affects accuracy", p < 0.10, f"Kruskal-Wallis H={H:.2f}, p={p:.4f} (recorded q=0.08)")

    # ---- claim 3: hidden FC layers are waste ----
    hdr("CLAIM 3  top models use 0 hidden FC layers (recorded: Kruskal-Wallis q=0.04)")
    f = feas.dropna(subset=["fc_layers"]).copy()
    f["fc_layers"] = f["fc_layers"].astype(float).astype(int)
    print(f"  {'fc_layers':>10} {'n':>4} {'median':>8} {'best':>8}")
    groups = []
    for k in sorted(f["fc_layers"].unique()):
        g = f[f["fc_layers"] == k]["val_accuracy"].values
        groups.append(g)
        print(f"  {k:>10} {len(g):>4} {np.median(g):>8.4f} {g.max():>8.4f}")
    H, p = kruskal(groups)
    top10 = f.nlargest(min(10, len(f)), "val_accuracy")
    n0 = int((top10["fc_layers"] == 0).sum())
    print(f"\n  top {len(top10)} by accuracy: {n0} of {len(top10)} used 0 hidden FC layers")
    verdict("hidden FC layers do not help", n0 >= len(top10) * 0.8,
            f"{n0}/{len(top10)} of the best used none; Kruskal-Wallis p={p:.4f} (recorded q=0.04)")

    # ---- claim 4: no normalization means the network does not learn ----
    hdr("CLAIM 4  removing normalization kills learning (recorded: Mann-Whitney p=0.0015, OR 21.8)")
    f = feas.dropna(subset=["norm"]).copy()
    f["norm"] = f["norm"].astype(str)
    print(f"  {'norm':>8} {'n':>4} {'median':>8} {'best':>8} {'dead (<0.15)':>13}")
    for nm in sorted(f["norm"].unique()):
        g = f[f["norm"] == nm]["val_accuracy"].values
        print(f"  {nm:>8} {len(g):>4} {np.median(g):>8.4f} {g.max():>8.4f} {int((g < 0.15).sum()):>13}")
    none = f[f["norm"] == "none"]["val_accuracy"].values
    withn = f[f["norm"] != "none"]["val_accuracy"].values
    if len(none) and len(withn):
        p, cles = mannwhitney(withn, none)
        verdict("normalization is required", p < 0.05,
                f"p={p:.4f} (recorded 0.0015), P(normalized > unnormalized)={cles:.2f}")
    else:
        print(f"  [NOT TESTABLE] norm='none' appears {len(none)} times in this data; "
              f"the claim came from an earlier study whose records are not here.")

    # ---- claim 5: learning rate correlates with accuracy ----
    hdr("CLAIM 5  learning rate correlates positively (recorded: Spearman q=0.04)")
    f = feas.dropna(subset=["lr"]).copy()
    rho, p = spearman(f["lr"].astype(float).values, f["val_accuracy"].values)
    top10 = f.nlargest(min(10, len(f)), "val_accuracy")
    print(f"  n={len(f)}  Spearman rho={rho:+.3f}  permutation p={p:.4f}")
    print(f"  top {len(top10)} lr range: {top10['lr'].min():.2e} to {top10['lr'].max():.2e}")
    verdict("higher lr scores better", rho > 0 and p < 0.10,
            f"rho={rho:+.3f}, p={p:.4f} (recorded q=0.04)")

    # ---- claim 6: tau's correlation is an artefact of rounding ----
    hdr("CLAIM 6  tau's correlation does not survive the integer register (recorded: rho=-0.60)")
    f = feas.dropna(subset=["tau"]).copy()
    tau = f["tau"].astype(float).values
    rho_c, p_c = spearman(tau, f["val_accuracy"].values)
    deployed = np.clip(np.round(tau), 2, 128)
    rho_d, p_d = spearman(deployed, f["val_accuracy"].values)
    print(f"  sampled tau range: {tau.min():.3f} to {tau.max():.3f}")
    print(f"  distinct sampled values : {len(np.unique(np.round(tau,6)))}")
    print(f"  distinct DEPLOYED values: {len(np.unique(deployed))}  -> {sorted(set(deployed.astype(int)))}")
    print(f"\n  continuous tau vs accuracy: rho={rho_c:+.3f}  p={p_c:.4f}")
    print(f"  deployed  tau vs accuracy: rho={rho_d:+.3f}  p={p_d:.4f}")
    verdict("the continuous correlation is not something the chip can use",
            len(np.unique(deployed)) < len(np.unique(np.round(tau, 6))),
            f"{len(np.unique(np.round(tau,6)))} sampled values collapse to "
            f"{len(np.unique(deployed))} on chip")

    # ---- claim 7: kernel sizes 5 and 7 ----
    hdr("CLAIM 7  every top-10 configuration used kernel 5 or 7")
    f = feas.dropna(subset=["kernel_size"]).copy()
    f["kernel_size"] = f["kernel_size"].astype(float).astype(int)
    top10 = f.nlargest(min(10, len(f)), "val_accuracy")
    hits = int(top10["kernel_size"].isin([5, 7]).sum())
    print(f"  kernel counts overall  : {dict(Counter(f['kernel_size']))}")
    print(f"  kernel counts in top {len(top10)}: {dict(Counter(top10['kernel_size']))}")
    verdict("the best models use kernel 5 or 7", hits == len(top10),
            f"{hits}/{len(top10)} of the best used 5 or 7")

    # ---- what actually rejected configurations ----
    hdr("WHY CONFIGURATIONS WERE REJECTED")
    infeas = df[df["source"] == "infeasible"]
    if len(infeas):
        kinds = Counter()
        for v in infeas["violations"].fillna(""):
            for part in str(v).split(";"):
                part = part.strip()
                if part:
                    kinds[part.split()[0].rstrip(":") + " " + " ".join(part.split()[1:3])] += 1
        for k, c in kinds.most_common():
            print(f"  {c:>3}x  {k}")
    else:
        print("  no infeasible records in this data")

    # ---- what happened at the deployment stage ----
    hdr("DEPLOYMENT OUTCOMES  (only runs that report hw_val_accuracy reached this stage)")
    if DEPLOY:
        dep = pd.concat(DEPLOY, ignore_index=True)
        if "hw_val_accuracy" in dep.columns:
            reached = dep[dep["hw_val_accuracy"].notna()]
            print(f"  trials that produced a deployment report: {len(reached)} of {len(dep)}")
            print(f"  {'trial':>10} {'run':>26} {'float':>7} {'hw':>7} {'gap':>7} {'clip':>6} {'deployable':>11}")
            for _, r in reached.iterrows():
                print(f"  {str(r['trial_id']):>10} {r['run'][-24:]:>26} "
                      f"{r.get('float_val_accuracy', float('nan')):>7.4f} "
                      f"{r['hw_val_accuracy']:>7.4f} "
                      f"{r.get('quant_gap', float('nan')):>+7.4f} "
                      f"{r.get('weight_clip_frac', float('nan')):>6.3f} "
                      f"{str(r.get('deployable')):>11}")
            for _, r in reached.iterrows():
                if r.get("deploy_reasons"):
                    print(f"      {r['trial_id']}: {r['deploy_reasons']}")
            # Runs that predate the two-metric rework have no hw_val_accuracy key
            # at all, so their blanks are expected and say nothing. Only runs that
            # DO report it can be asked why a report is missing.
            hw_runs = set(dep[dep["hw_val_accuracy"].notna()]["run"])
            era = dep[dep["run"].isin(hw_runs)]
            missing = era[era["hw_val_accuracy"].isna()]
            pruned = missing[missing["stopped_early"] == True] if "stopped_early" in missing else missing
            ran_full = missing[missing["stopped_early"] == False] if "stopped_early" in missing else missing.iloc[0:0]
            print(f"\n  in the {len(hw_runs)} runs that report it at all ({len(era)} trials):")
            print(f"    produced a deployment report            : {len(era) - len(missing)}")
            print(f"    pruned by ASHA before the deploy phase  : {len(pruned)}  (expected)")
            print(f"    ran to completion yet produced no report: {len(ran_full)}")
            for _, r in ran_full.iterrows():
                print(f"        {r['trial_id']}  epochs_run={r.get('epochs_run')}")
            older = dep[~dep["run"].isin(hw_runs)]
            print(f"  the other {len(older)} trials predate the hardware metric entirely")
        else:
            print("  no run in this data reports hw_val_accuracy")
    else:
        print("  no trials.jsonl found")

    # ---- the ceiling ----
    hdr("THE CEILING REACHED")
    print("  target                 : 0.975 (search_config)")
    print(f"  best float accuracy    : {feas['val_accuracy'].max():.4f}")
    best = feas.nlargest(1, "val_accuracy").iloc[0]
    print(f"  best trial             : {best['trial_id']}  ({best['run']})")
    keys = ["depth", "channels", "kernel_size", "resize_to", "T", "tau",
            "optimizer", "lr", "weight_decay", "scheduler", "norm", "fc_layers"]
    print("  its configuration      : " +
          ", ".join(f"{k}={best[k]}" for k in keys if k in best.index and pd.notna(best[k])))
    early = feas[feas.get("stopped_early", False).astype(str).str.lower() == "true"] \
        if "stopped_early" in feas.columns else feas.iloc[0:0]
    print(f"  stopped early by ASHA  : {len(early)}/{len(feas)}")
    if "epochs_run" in feas.columns:
        print(f"  median epochs run      : {feas['epochs_run'].median():.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
