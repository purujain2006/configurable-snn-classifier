"""Accuracy per person and per lighting, on validation and on test.

    python tools/analysis/per_subject.py -c configs/dvs128_search05.yaml results/search05_top5/rank1 results/search05_top5/rank2

Validation holds out whole people from the training set (search.val_split
"subject", chosen by run.seed) and test is the dataset's own held-out people,
so the two can differ only because some people are harder to read than
others. This scores each checkpoint clip by clip, the way `single` scored it
(the deployed network, in the chip's arithmetic), and groups the clips by the
person and the lighting named in their file names, e.g.
user07_fluorescent_led_3.npz.

Each argument is a run folder holding one .pth, as `single` writes it, or a
.pth file. When the folder also holds single.json, the overall scores must
equal the ones in it, which proves the clips and the split are the ones the
run used. A person has about 50 clips, so one person's score moves by about
6 points by chance; a pattern every checkpoint shares means more than one
checkpoint's number.

--csv writes one row per clip and checkpoint, for looking at classes too.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

_NAME = re.compile(r"user(\d+)_(.+?)(?:_\d+)?\.\w+$", re.IGNORECASE)
SPLITS = ("val", "test")


def recording_of(path):
    """(subject, lighting) from a DVS128 Gesture file name, or None."""
    m = _NAME.search(os.path.basename(str(path)))
    return (int(m.group(1)), m.group(2)) if m else None


def _paths(dataset):
    samples = getattr(dataset, "samples", None)
    if not samples:
        return None
    return [s[0] if isinstance(s, (tuple, list)) else s for s in samples]


def find_checkpoint(target):
    """(pth path, single.json path or None) for a run folder or a .pth file."""
    if os.path.isfile(target):
        single = os.path.join(os.path.dirname(target), "single.json")
        return target, (single if os.path.isfile(single) else None)
    found = sorted(glob.glob(os.path.join(target, "*.pth")))
    if len(found) != 1:
        raise SystemExit(f"{target}: expected one .pth, found {len(found)}")
    single = os.path.join(target, "single.json")
    return found[0], (single if os.path.isfile(single) else None)


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
        train_paths, test_paths = _paths(bundle.train), _paths(bundle.test)
        if not train_paths or not test_paths:
            raise SystemExit("this dataset does not name its clip files, so there is "
                             "no person to group by")
        # the same split as pipeline.make_loaders, from the same settings
        _train_idx, val_idx, held = split_indices(
            bundle.train, seed=cfg["run"].get("seed", 1),
            val_split=cfg["search"].get("val_split", "subject"))
        names = {"val": [train_paths[i] for i in val_idx], "test": test_paths}
        for split, paths in names.items():
            if any(recording_of(p) is None for p in paths):
                raise SystemExit(f"a {split} clip has no user number in its file name "
                                 f"({next(p for p in paths if recording_of(p) is None)})")
        return {"sets": {"val": Subset(bundle.train, val_idx), "test": bundle.test},
                "names": names, "held": held, "collate": _make_collate(bundle, encoder)}


def predictions(net, dataset, collate, device, batch=16):
    """(predicted, true) class per clip, in the dataset's order."""
    import torch
    from torch.utils.data import DataLoader
    from snnsearch.train import forward_over_time
    pred, true = [], []
    net.eval()
    with torch.no_grad():
        for x, y, _lengths in DataLoader(dataset, batch_size=batch, shuffle=False,
                                         collate_fn=collate):
            pred += forward_over_time(net, x.to(device)).argmax(1).tolist()
            true += y.tolist()
    return pred, true


def score(cfg, targets, device):
    """One record per checkpoint: its clips with person, lighting and result,
    and the comparison with its single.json."""
    from snnsearch.hs_export import load_deployed, exact_copy
    from snnsearch.train import exact_arithmetic
    clip_sets = _ClipSets(cfg)
    out = []
    for target in targets:
        pth, single = find_checkpoint(target)
        net, spec, _ckpt = load_deployed(pth, device)
        clips = clip_sets.get(spec)
        exact = exact_copy(net)
        rec = {"target": target, "pth": pth, "held": clips["held"], "clips": {}}
        with exact_arithmetic():
            for split in SPLITS:
                pred, true = predictions(exact, clips["sets"][split], clips["collate"], device)
                rec["clips"][split] = [
                    {"path": p, "subject": recording_of(p)[0], "lighting": recording_of(p)[1],
                     "label": t, "pred": q}
                    for p, t, q in zip(clips["names"][split], true, pred)]
        rec["overall"] = {s: _acc(rec["clips"][s]) for s in SPLITS}
        rec["expected"] = None
        if single:
            with open(single, encoding="utf-8") as fh:
                saved = json.load(fh)
            rec["expected"] = {"val": saved.get("hw_val_accuracy"),
                               "test": saved.get("hw_test_accuracy")}
        out.append(rec)
    return out


def _acc(clips):
    return sum(c["label"] == c["pred"] for c in clips) / max(1, len(clips))


def labels(targets):
    """Short column names: the folder name, with its parent when two collide."""
    base = [os.path.basename(os.path.normpath(t)) for t in targets]
    long = [os.path.join(os.path.basename(os.path.dirname(os.path.normpath(t))), b)
            for t, b in zip(targets, base)]
    return [lg if base.count(b) > 1 else b for b, lg in zip(base, long)]


def table(records, key):
    """Lines of a table: one row per split and value of `key`, one column per
    checkpoint, plus the mean of the columns when there are several."""
    names = labels([r["target"] for r in records])
    width = max([6] + [len(n) for n in names])
    many = len(records) > 1
    head = f"{'split':5s} {key:16s} {'clips':>5s} " + " ".join(f"{n:>{width}s}" for n in names)
    if many:
        head += f" {'mean':>6s}"
    lines = [head, "-" * len(head)]
    for split in SPLITS:
        groups = sorted({c[key] for c in records[0]["clips"][split]})
        for g in groups + ["all"]:
            accs = []
            for r in records:
                clips = [c for c in r["clips"][split] if g == "all" or c[key] == g]
                accs.append(_acc(clips))
            n = sum(1 for c in records[0]["clips"][split] if g == "all" or c[key] == g)
            row = f"{split:5s} {str(g):16s} {n:5d} " + " ".join(f"{a:>{width}.3f}" for a in accs)
            if many:
                row += f" {sum(accs) / len(accs):6.3f}"
            lines.append(row)
        lines.append("")
    return lines


def report(records):
    """The printed report, and whether every overall score matched single.json."""
    ok = True
    first = records[0]
    val_people = sorted({c["subject"] for c in first["clips"]["val"]})
    test_people = sorted({c["subject"] for c in first["clips"]["test"]})
    lines = [f"validation: people {', '.join(map(str, val_people))} "
             f"({len(first['clips']['val'])} clips)"
             + ("" if first["held"] else ", NOT held out of training (clip split)"),
             f"test:       people {', '.join(map(str, test_people))} "
             f"({len(first['clips']['test'])} clips)", ""]
    lines += ["by person"] + table(records, "subject")
    lines += ["by lighting"] + table(records, "lighting")
    for name, r in zip(labels([r["target"] for r in records]), records):
        exp = r["expected"]
        if not exp:
            lines.append(f"{name}: no single.json beside it, nothing to check against")
            continue
        same = all(exp[s] is None or abs(exp[s] - r["overall"][s]) < 1e-9 for s in SPLITS)
        ok &= same
        lines.append(f"{name}: {'matches' if same else 'DOES NOT MATCH'} single.json "
                     f"(val {r['overall']['val']:.4f} vs {exp['val']}, "
                     f"test {r['overall']['test']:.4f} vs {exp['test']})")
    return lines, ok


def write_csv(records, path):
    names = labels([r["target"] for r in records])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "split", "subject", "lighting", "label", "pred", "file"])
        for name, r in zip(names, records):
            for split in SPLITS:
                for c in r["clips"][split]:
                    w.writerow([name, split, c["subject"], c["lighting"], c["label"],
                                c["pred"], os.path.basename(c["path"])])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True,
                    help="the config the checkpoints trained with (dataset, run.seed, "
                         "search.val_split)")
    ap.add_argument("targets", nargs="+", help="run folders or .pth files")
    ap.add_argument("--data-root", help="dataset root, if not the config's")
    ap.add_argument("--csv", help="write one row per clip and checkpoint here")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    import torch
    from snnsearch import runconfig
    overrides = {"dataset": {"root": args.data_root}} if args.data_root else None
    cfg = runconfig.load(args.config, overrides)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    records = score(cfg, args.targets, device)
    lines, ok = report(records)
    print("\n".join(lines))
    if args.csv:
        write_csv(records, args.csv)
        print(f"\nper-clip results in {args.csv}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
