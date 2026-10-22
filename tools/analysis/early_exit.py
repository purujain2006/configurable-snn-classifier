"""Accuracy and SynOps when the chip is fed only the first k of a clip's T frames.

    python tools/analysis/early_exit.py -c configs/dvs128_search05.yaml --test results/search05_top5/rank1 results/search05_top5/rank2

SynOps grow with the frames processed, and on HiAER-Spike the host can stop
sending input whenever it likes. DVS128 frames split each clip into T groups
of equal event count, so the first k frames are the first k/T of the gesture's
events. This feeds each deployed network the first k frames and then the flush
steps (the zero-input steps that drain the pipeline, as `single` scores it),
in the chip's arithmetic (hs_export.exact_copy), and reports accuracy and
SynOps per clip at each k. No training: it answers how much of the T frames
matter at inference for a network trained on all of them.

Each argument is a run folder holding one .pth, as `single` writes it, or a
.pth file. Validation is rebuilt from the config's run.seed and
search.val_split, the way per_subject.py does it; --test adds the test set.
The k = T row is exactly what `single` measured, so when single.json is beside
the checkpoint it must equal hw_val_accuracy (and hw_test_accuracy), which
proves the clips and the network are the ones the run used.

It also reports, from the k = T run, the frame after which each clip's running
prediction (the class with the most output spikes so far) never changed: a
cheap early-exit signal. "settled by k" is the share of clips whose answer was
already final once frame k's evidence reached the output. SynOps are counted
on the float64 network that is scored, so the k = T figure can differ a little
from single.json's, which counted the float32 one.

--k picks the frame counts (default every --every frames up to T, plus T).
--csv writes one row per checkpoint, split and k.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from per_subject import find_checkpoint, labels  # noqa: E402

SPLIT_NAMES = {"val": "validation", "test": "test"}
EXPECTED = {"val": "hw_val_accuracy", "test": "hw_test_accuracy"}


def frame_counts(T, ks=None, every=8):
    """The k to try for a T-frame network: the given ones up to T, else every
    `every` frames, and always T itself."""
    chosen = ks if ks else range(every, T + 1, every)
    return sorted({k for k in chosen if 1 <= k <= T} | {T})


class _ClipSets:
    """Validation and test clips for one input format, built once and shared
    by every checkpoint that reads that format."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cache = {}

    def get(self, spec):
        inp = spec["input"]
        key = (inp.T, inp.resize_to, inp.input_bits)
        if key not in self.cache:
            self.cache[key] = self._build(inp)
        return self.cache[key]

    def _build(self, inp):
        from torch.utils.data import Subset
        from snnsearch.data.base import build_dataset
        from snnsearch.data.loaders import split_indices, _make_collate
        from snnsearch.encoders import build_encoder
        cfg = self.cfg
        bundle = build_dataset({**cfg["dataset"], "T": inp.T})
        # the same coding rule as pipeline.resolve_specs
        coding = cfg["encoding"].get("coding") or \
            ("passthrough" if bundle.is_event else "levels")
        encoder = build_encoder(coding, T=inp.T, resize_to=inp.resize_to or None,
                                input_bits=inp.input_bits)
        # the same split as pipeline.make_loaders, from the same settings
        _train_idx, val_idx, held = split_indices(
            bundle.train, seed=cfg["run"].get("seed", 1),
            val_split=cfg["search"].get("val_split", "subject"))
        return {"sets": {"val": Subset(bundle.train, val_idx), "test": bundle.test},
                "held": held, "collate": _make_collate(bundle, encoder)}


def score_split(net, dataset, collate, device, ks, batch=16):
    """Accuracy and SynOps per clip at each k, and the frame each clip's
    running prediction settled at in the k = T run.

    Batch by batch, every k runs on the same clips, so the data is read once.
    Each k has its own counter, attached only while its k runs.
    """
    import torch
    from torch.utils.data import DataLoader
    from snnsearch.synops import SynOpsCounter
    from snnsearch.train import forward_over_time, hardware_flush_steps
    T, flush = ks[-1], hardware_flush_steps(net)       # frame_counts ends at T
    counters = {k: SynOpsCounter() for k in ks}
    pred = {k: [] for k in ks}
    true, settled = [], []
    outs = []
    hook = net.register_forward_hook(lambda _m, _i, o: outs.append(o.detach()))
    net.eval()
    try:
        with torch.no_grad():
            for x, y, _lengths in DataLoader(dataset, batch_size=batch, shuffle=False,
                                             collate_fn=collate):
                if x.shape[1] != T:
                    raise SystemExit(f"clips have {x.shape[1]} frames, the network was "
                                     f"trained on {T}")
                x, y = x.to(device), y.to(device)
                for k in ks:
                    outs.clear()
                    with counters[k].attach(net):
                        out = forward_over_time(net, x[:, :k])
                    counters[k].add_samples(x.shape[0])
                    pred[k] += out.argmax(1).tolist()
                true += y.tolist()
                # outs now holds the k = T run, the last one
                seq = outs[0] if getattr(net, "step_mode", "s") == "m" else torch.stack(outs)
                settled += settle_frames(seq, flush, T).tolist()
    finally:
        hook.remove()
    synops = {k: counters[k].summary() for k in ks}
    n = max(1, len(true))
    return {"clips": len(true), "true": true, "pred": pred, "settled": settled,
            "acc": {k: sum(p == t for p, t in zip(pred[k], true)) / n for k in ks},
            "synops": {k: synops[k].get("synops_per_sample") for k in ks},
            "synops_reason": {k: synops[k]["reason"] for k in ks if synops[k].get("reason")}}


def settle_frames(seq, flush, T):
    """Per clip, the number of input frames whose evidence had reached the
    output when the running prediction last changed.

    seq is the output at every step, (T + flush, N, classes). The running
    prediction after step s is the class with the most output spikes in
    steps 0..s. Output step s carries input frames 0..s - flush, since every
    spiking layer delays by one step, so a change at step s needed
    s - flush + 1 frames. 0 means the first answer was already the final one.
    """
    import torch
    running = seq.cumsum(0).argmax(-1)                     # (steps, N)
    steps = running.shape[0]
    changed = running[1:] != running[:-1]                  # change at step j + 1
    last = steps - 1 - changed.flip(0).int().argmax(0)     # last changed step
    last = torch.where(changed.any(0), last, torch.zeros_like(last))
    return (last - flush + 1).clamp(0, T)


def score(cfg, targets, device, ks=None, every=8, splits=("val",), batch=16):
    """One record per checkpoint: per split, accuracy and SynOps at each k and
    the settle frames, plus what single.json says the k = T row must be."""
    from snnsearch.hs_export import load_deployed, exact_copy
    from snnsearch.train import exact_arithmetic, hardware_flush_steps
    clip_sets = _ClipSets(cfg)
    out = []
    for target in targets:
        pth, single = find_checkpoint(target)
        net, spec, _ckpt = load_deployed(pth, device)
        T = spec["input"].T
        clips = clip_sets.get(spec)
        exact = exact_copy(net)
        rec = {"target": target, "pth": pth, "T": T, "ks": frame_counts(T, ks, every),
               "flush": hardware_flush_steps(exact), "held": clips["held"], "splits": {}}
        with exact_arithmetic():
            for split in splits:
                rec["splits"][split] = score_split(exact, clips["sets"][split],
                                                   clips["collate"], device, rec["ks"], batch)
        rec["expected"] = None
        if single:
            with open(single, encoding="utf-8") as fh:
                saved = json.load(fh)
            rec["expected"] = {s: saved.get(EXPECTED[s]) for s in splits}
            rec["expected"]["synops"] = saved.get("synops_per_sample")
        out.append(rec)
    return out


def _settled_by(frames, k):
    return sum(f <= k for f in frames) / max(1, len(frames))


def _pct(frames, q):
    s = sorted(frames)
    return s[min(len(s) - 1, int(q * len(s)))] if s else None


def _grid(records, rows, cell, fmt, width, many):
    """Table lines: one row per (label, key), one column per checkpoint, and
    the mean of the columns that have a value."""
    lines = []
    for label, key in rows:
        vals = [cell(r, key) for r in records]
        line = f"  {label:13s} " + " ".join(
            f"{'-':>{width}s}" if v is None else f"{v:>{width}{fmt}}" for v in vals)
        have = [v for v in vals if v is not None]
        if many and have:
            line += f" {sum(have) / len(have):>8{fmt}}"
        lines.append(line)
    return lines


def report(records):
    """The printed report, and whether every k = T row matched single.json."""
    names = labels([r["target"] for r in records])
    width = max([8] + [len(n) for n in names])
    many = len(records) > 1
    head = f"  {'':13s} " + " ".join(f"{n:>{width}s}" for n in names) + \
        (f" {'mean':>8s}" if many else "")
    splits = list(records[0]["splits"])
    first = records[0]
    t_flush = ["%d, %d" % (r["T"], r["flush"]) for r in records]
    lines = [head, f"  {'T, flush':13s} " + " ".join(f"{s:>{width}s}" for s in t_flush)]
    for split in splits:
        part = first["splits"][split]
        held = "" if split != "val" else (
            f", people {', '.join(map(str, first['held']))} held out" if first["held"]
            else ", NOT held out of training (clip split)")
        lines += ["", f"{SPLIT_NAMES[split]} ({part['clips']} clips{held}): first k frames, "
                  "then the flush steps"]
        ks = sorted({k for r in records for k in r["ks"]})
        krows = [(f"k = {k}", k) for k in ks]

        def at(r, k, field):
            return r["splits"][split][field].get(k) if k in r["ks"] else None
        lines += ["accuracy", head]
        lines += _grid(records, krows, lambda r, k: at(r, k, "acc"), ".3f", width, many)
        lines += ["SynOps per clip, millions", head]
        lines += _grid(records, krows,
                       lambda r, k: None if at(r, k, "synops") is None
                       else at(r, k, "synops") / 1e6, ".2f", width, many)
        lines += ["SynOps per clip as a share of k = T", head]
        lines += _grid(records, krows, lambda r, k: _share(r, split, k), ".3f", width, many)
        lines += ["running prediction in the k = T run: settled by frame", head]
        lines += _grid(records, [(f"<= {k}", k) for k in ks],
                       lambda r, k: _settled_by(r["splits"][split]["settled"], k)
                       if k in r["ks"] else None, ".3f", width, many)
        lines += _grid(records, [("median frame", 0.5), ("90th pct", 0.9)],
                       lambda r, q: _pct(r["splits"][split]["settled"], q), ".0f", width, many)
        for name, r in zip(names, records):
            for k, why in r["splits"][split]["synops_reason"].items():
                lines.append(f"  {name} k = {k}: no SynOps, {why}")
    lines.append("")
    ok = True
    for name, r in zip(names, records):
        exp = r["expected"]
        if not exp:
            lines.append(f"{name}: no single.json beside it, nothing to check against")
            continue
        T, got = r["T"], {s: r["splits"][s]["acc"][r["T"]] for s in splits}
        same = all(exp[s] is None or abs(exp[s] - got[s]) < 1e-9 for s in splits)
        ok &= same
        parts = ", ".join(f"{s} {got[s]:.4f} vs {exp[s]}" for s in splits)
        syn = r["splits"]["val"]["synops"].get(T) if "val" in splits else None
        if syn is not None and exp.get("synops"):
            parts += f"; val SynOps {syn / 1e6:.2f}M vs {exp['synops'] / 1e6:.2f}M"
        lines.append(f"{name}: k = T {'matches' if same else 'DOES NOT MATCH'} "
                     f"single.json ({parts})")
    return lines, ok


def _share(r, split, k):
    syn = r["splits"][split]["synops"]
    if k not in r["ks"] or syn.get(k) is None or not syn.get(r["T"]):
        return None
    return syn[k] / syn[r["T"]]


def write_csv(records, path):
    names = labels([r["target"] for r in records])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "split", "T", "k", "clips", "accuracy",
                    "synops_per_clip", "settled_by_k"])
        for name, r in zip(names, records):
            for split, part in r["splits"].items():
                for k in r["ks"]:
                    w.writerow([name, split, r["T"], k, part["clips"], part["acc"][k],
                                part["synops"][k], _settled_by(part["settled"], k)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True,
                    help="the config the checkpoints trained with (dataset, run.seed, "
                         "search.val_split)")
    ap.add_argument("targets", nargs="+", help="run folders or .pth files")
    ap.add_argument("--k", help="frame counts to try, comma separated, e.g. 4,8,16 "
                                "(T is always added)")
    ap.add_argument("--every", type=int, default=8,
                    help="without --k, try every this many frames (default 8)")
    ap.add_argument("--test", action="store_true", help="score the test set too")
    ap.add_argument("--data-root", help="dataset root, if not the config's")
    ap.add_argument("--csv", help="write one row per checkpoint, split and k here")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    import torch
    from snnsearch import runconfig
    overrides = {"dataset": {"root": args.data_root}} if args.data_root else None
    cfg = runconfig.load(args.config, overrides)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ks = [int(k) for k in args.k.split(",")] if args.k else None
    splits = ("val", "test") if args.test else ("val",)
    records = score(cfg, args.targets, device, ks=ks, every=args.every, splits=splits,
                    batch=args.batch)
    lines, ok = report(records)
    print("\n".join(lines))
    if args.csv:
        write_csv(records, args.csv)
        print(f"\none row per checkpoint, split and k in {args.csv}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
