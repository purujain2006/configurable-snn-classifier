"""Recount SynOps for saved networks with the current counter, without retraining.

WHY THIS EXISTS

A run measures SynOps once, right after training, and writes the number into
single.json. If the counter is fixed later, that number stays wrong: the
search05_top5 replays were measured before SynOps counted signed spikes by
magnitude (ab250eb), so a -1 spike could cancel a +1 spike and every signed
network reported too little work. Their checkpoints hold the deployed network,
so the fix only needs the validation clips run through it again.

    python tools/analysis/recount_synops.py -c configs/dvs128_search05.yaml results/search05_top5

Takes the config the checkpoints trained with (for the dataset, run.seed and
search.val_split) and checkpoint files or folders (every *.pth below them). For
each one it rebuilds the deployed network exactly as it was scored
(hs_export.load_deployed), rebuilds the validation clips from the checkpoint's
own T, frame size and input bits, counts with snnsearch/synops.py, and prints the new figures next to the
old ones from the single.json beside the checkpoint. --out writes the table as
CSV.

Validation is the split a run measures on, so new and old cover the same clips.
"""
import argparse
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

COLUMNS = ["checkpoint", "trial", "clips", "old_synops_per_sample", "synops_per_sample",
           "new_over_old", "spikes_per_sample", "firing_rate", "synops_ceiling_per_sample",
           "dense_macs_per_inference", "synops_over_dense", "old_synops_over_dense", "reason"]


def find_checkpoints(paths):
    """Checkpoint files named directly, plus every *.pth under folders, sorted."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "**", "*.pth"), recursive=True)
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SystemExit(f"no such file or folder: {p}")
    if not out:
        raise SystemExit("no .pth checkpoints found under " + ", ".join(paths))
    return sorted(set(os.path.normpath(p) for p in out))


def old_numbers(ckpt_path, ckpt=None):
    """What the run reported when it was trained: single.json beside the
    checkpoint, else the checkpoint's own metrics. Empty when neither has it."""
    side = os.path.join(os.path.dirname(ckpt_path), "single.json")
    for source in ([side] if os.path.isfile(side) else []):
        with open(source, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("synops_per_sample") is not None:
            return {"old_synops_per_sample": rec.get("synops_per_sample"),
                    "old_synops_over_dense": rec.get("synops_over_dense")}
    m = (ckpt or {}).get("metrics") or {}
    return {"old_synops_per_sample": m.get("synops_per_sample"),
            "old_synops_over_dense": m.get("synops_over_dense")}


def trial_name(ckpt_path):
    """The trial id from the trial.json beside the checkpoint, else the file name."""
    side = os.path.join(os.path.dirname(ckpt_path), "trial.json")
    if os.path.isfile(side):
        try:
            with open(side, encoding="utf-8") as fh:
                return json.load(fh).get("trial_id") or ""
        except (OSError, ValueError):
            pass
    return os.path.splitext(os.path.basename(ckpt_path))[0]


def validation_loader(spec, cfg, batch_size=16):
    """The validation clips a run measures on, coded the way this network reads them.
    `cfg` is the loaded run config: its dataset block, run.seed and search.val_split."""
    from snnsearch.data.base import build_dataset
    from snnsearch.data.loaders import build_dataloaders
    from snnsearch.encoders import build_encoder

    inp = spec["input"]
    bundle = build_dataset({**cfg["dataset"], "T": inp.T})
    coding = "passthrough" if bundle.is_event else "levels"
    encoder = build_encoder(coding, T=inp.T, resize_to=inp.resize_to or None,
                            input_bits=getattr(inp, "input_bits", 1) or 1)
    _train, val, _test = build_dataloaders(bundle, batch_size=batch_size, encoder=encoder,
                                           seed=cfg["run"].get("seed", 1),
                                           val_split=cfg["search"].get("val_split", "subject"))
    return val


def recount(ckpt_path, loader_for, device):
    """One row: the deployed network in `ckpt_path`, counted on loader_for(spec)."""
    from snnsearch.hs_export import load_deployed
    from snnsearch.synops import measure_synops
    from snnsearch.train import exact_arithmetic

    net, spec, ckpt = load_deployed(ckpt_path, device)
    loader = loader_for(spec)
    with exact_arithmetic():
        s = measure_synops(net, loader, device, spec=spec)
    row = {"checkpoint": ckpt_path, "trial": trial_name(ckpt_path), **old_numbers(ckpt_path, ckpt)}
    for k in ("synops_per_sample", "spikes_per_sample", "firing_rate", "synops_ceiling_per_sample",
              "dense_macs_per_inference", "synops_over_dense"):
        row[k] = s.get(k)
    row["reason"] = s.get("reason", "")
    try:
        row["clips"] = len(loader.dataset)
    except (AttributeError, TypeError):
        row["clips"] = ""
    new, old = row.get("synops_per_sample"), row.get("old_synops_per_sample")
    row["new_over_old"] = (new / old) if (new and old) else None
    return row


def _fmt(v, digits=3):
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{digits}f}"
    return str(v)


def print_table(rows):
    print(f"\n{'trial':<12} {'old SynOps':>14} {'new SynOps':>14} {'new/old':>8} "
          f"{'firing':>7} {'over dense':>10}  reason")
    for r in rows:
        print(f"{r['trial']:<12} {_fmt(r.get('old_synops_per_sample')):>14} "
              f"{_fmt(r.get('synops_per_sample')):>14} {_fmt(r.get('new_over_old'), 2):>8} "
              f"{_fmt(r.get('firing_rate')):>7} {_fmt(r.get('synops_over_dense'), 2):>10}  "
              f"{r.get('reason') or ''}")


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-c", "--config", required=True,
                    help="the config the checkpoints trained with (dataset, run.seed, "
                         "search.val_split)")
    ap.add_argument("paths", nargs="+", help="checkpoint files or folders holding *.pth")
    ap.add_argument("--data-root", help="dataset root, if not the config's")
    ap.add_argument("--out", help="also write the table to this CSV file")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from snnsearch import runconfig
    overrides = {"dataset": {"root": args.data_root}} if args.data_root else None
    cfg = runconfig.load(args.config, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpts = find_checkpoints(args.paths)
    print(f"{len(ckpts)} checkpoint(s), counting on {device}")

    cache = {}

    def loader_for(spec):
        inp = spec["input"]
        key = (inp.T, inp.resize_to, getattr(inp, "input_bits", 1))
        if key not in cache:
            cache[key] = validation_loader(spec, cfg)
        return cache[key]

    rows = []
    for p in ckpts:
        print(f"\n{p}")
        try:
            rows.append(recount(p, loader_for, device))
        except Exception as exc:          # one bad checkpoint should not lose the rest
            rows.append({"checkpoint": p, "trial": trial_name(p),
                         "reason": f"{type(exc).__name__}: {exc}"})
    print_table(rows)
    if args.out:
        write_csv(rows, args.out)
        print(f"\nwrote {args.out}")
    return rows


if __name__ == "__main__":
    main()
