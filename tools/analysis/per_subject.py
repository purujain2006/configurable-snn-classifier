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
6 points by chance. A pattern every checkpoint shares means more than one
checkpoint's number.

A folder of fold runs, as `single --folds k` writes it (fold0, fold1, ...),
counts as one checkpoint: each fold's network is scored on its own
validation people, which together are every training person once, so every
training person gets an accuracy from a network that never trained on them.
Each fold must match its own single.json. Fold runs are validation only: no
test column. Score fold folders and single-split runs in separate calls.

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


def fold_runs(target):
    """[(fold, run folder)] when `target` is a folder of fold runs (fold0,
    fold1, ... and no .pth of its own), else None."""
    from snnsearch.folds import fold_runs as runs
    if not os.path.isdir(target) or glob.glob(os.path.join(target, "*.pth")):
        return None
    return runs(target) or None


class _ClipSets:
    """Validation and test clips for one input format and split, the dataset
    built once per input format and shared by every checkpoint that reads it."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.data = {}
        self.cache = {}

    def get(self, spec, folds=0, fold=0):
        inp = spec["input"]
        key = (inp.T, inp.resize_to, inp.input_bits)
        if key not in self.data:
            self.data[key] = self._data(inp)
        if key + (folds, fold) not in self.cache:
            self.cache[key + (folds, fold)] = self._split(self.data[key], folds, fold)
        return self.cache[key + (folds, fold)]

    def _data(self, inp):
        from snnsearch.data.base import build_dataset
        from snnsearch.data.loaders import _make_collate
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
        return {"bundle": bundle, "train_paths": train_paths, "test_paths": test_paths,
                "collate": _make_collate(bundle, encoder)}

    def _split(self, data, folds, fold):
        from torch.utils.data import Subset
        from snnsearch.data.loaders import split_indices, split_settings
        bundle = data["bundle"]
        # the same split as pipeline.make_loaders, from the same settings. A
        # fold run's fold comes from its single.json
        settings = {**split_settings(self.cfg), "folds": folds, "fold": fold}
        _train_idx, val_idx, held = split_indices(bundle.train, **settings)
        names = {"val": [data["train_paths"][i] for i in val_idx],
                 "test": data["test_paths"]}
        for split, paths in names.items():
            if any(recording_of(p) is None for p in paths):
                raise SystemExit(f"a {split} clip has no user number in its file name "
                                 f"({next(p for p in paths if recording_of(p) is None)})")
        return {"sets": {"val": Subset(bundle.train, val_idx), "test": bundle.test},
                "names": names, "held": held, "collate": data["collate"]}


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


def _score_one(clip_sets, target, device, splits=SPLITS, folds=0, fold=0):
    """One checkpoint's clips on `splits`, with person, lighting and result,
    and what its single.json says they scored."""
    from snnsearch.hs_export import load_deployed, exact_copy
    from snnsearch.train import exact_arithmetic
    pth, single = find_checkpoint(target)
    net, spec, _ckpt = load_deployed(pth, device)
    clips = clip_sets.get(spec, folds, fold)
    exact = exact_copy(net)
    rec = {"target": target, "pth": pth, "held": clips["held"], "clips": {}}
    with exact_arithmetic():
        for split in splits:
            pred, true = predictions(exact, clips["sets"][split], clips["collate"], device)
            rec["clips"][split] = [
                {"path": p, "subject": recording_of(p)[0], "lighting": recording_of(p)[1],
                 "label": t, "pred": q}
                for p, t, q in zip(clips["names"][split], true, pred)]
    rec["overall"] = {s: _acc(rec["clips"][s]) for s in splits}
    rec["expected"] = None
    if single:
        with open(single, encoding="utf-8") as fh:
            saved = json.load(fh)
        rec["expected"] = {s: saved.get(f"hw_{s}_accuracy") for s in splits}
    return rec


def recorded_split(target):
    """The split a run's single.json records (`single` writes it), or {}."""
    _pth, single = find_checkpoint(target)
    if single is None:
        return {}
    with open(single, encoding="utf-8") as fh:
        return json.load(fh).get("split") or {}


def _score_run(clip_sets, target, device):
    """One checkpoint. A fold run is scored on its own fold's validation
    people, named in its single.json, and never on test."""
    split = recorded_split(target)
    if not split.get("val_folds"):
        return _score_one(clip_sets, target, device)
    rec = _score_one(clip_sets, target, device, ("val",), split["val_folds"],
                     split["val_fold"])
    rec.update(fold=split["val_fold"], recorded_people=split.get("val_people"))
    return rec


def _score_folds(clip_sets, target, runs, device):
    """A folder of fold runs as one record: each fold's network on its own
    validation people, pooled, so the clips are every training person once."""
    folds = []
    for fold, run in runs:
        split = recorded_split(run)
        if not split.get("val_folds") or split.get("val_fold") != fold:
            raise SystemExit(f"{run}: its single.json does not say it is fold {fold} "
                             f"(its split: {split or 'none recorded'})")
        folds.append(_score_run(clip_sets, run, device))
    val = [dict(c, fold=f["fold"]) for f in folds for c in f["clips"]["val"]]
    return {"target": target, "folds": folds, "held": True,
            "clips": {"val": val}, "overall": {"val": _acc(val)}, "expected": None}


def score(cfg, targets, device):
    """One record per checkpoint (or folder of fold runs): its clips with
    person, lighting and result, and the comparison with its single.json."""
    clip_sets = _ClipSets(cfg)
    runs = [fold_runs(t) for t in targets]
    if any(runs) and not all(runs):
        raise SystemExit("fold runs and single-split runs hold out different people; "
                         "score them in separate calls")
    records = [_score_folds(clip_sets, t, r, device) if r else _score_run(clip_sets, t, device)
               for t, r in zip(targets, runs)]
    # one table needs one set of people: a column for other people would read 0
    people = [sorted({c["subject"] for c in r["clips"]["val"]}) for r in records]
    if any(p != people[0] for p in people):
        raise SystemExit("these runs were validated on different people "
                         + ", ".join(f"{t}: {p}" for t, p in zip(targets, people))
                         + ". Score them in separate calls")
    return records


def _acc(clips):
    return sum(c["label"] == c["pred"] for c in clips) / max(1, len(clips))


def labels(targets):
    """Short column names: the folder name, with its parent when two collide."""
    base = [os.path.basename(os.path.normpath(t)) for t in targets]
    long = [os.path.join(os.path.basename(os.path.dirname(os.path.normpath(t))), b)
            for t, b in zip(targets, base)]
    return [lg if base.count(b) > 1 else b for b, lg in zip(base, long)]


def splits_of(records):
    """The splits the records were scored on, in SPLITS order."""
    return tuple(s for s in SPLITS if s in records[0]["clips"])


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
    for split in splits_of(records):
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
    if "folds" in first:
        lines = [f"validation: {len(first['folds'])} folds, each person scored by the "
                 f"network that held them out: people {', '.join(map(str, val_people))} "
                 f"({len(first['clips']['val'])} clips). Test is not scored."]
    else:
        lines = [f"validation: people {', '.join(map(str, val_people))} "
                 f"({len(first['clips']['val'])} clips)"
                 + ("" if first["held"] else ", NOT held out of training (clip split)")]
    if "test" in first["clips"]:
        test_people = sorted({c["subject"] for c in first["clips"]["test"]})
        lines.append(f"test:       people {', '.join(map(str, test_people))} "
                     f"({len(first['clips']['test'])} clips)")
    lines += ["", "by person"] + table(records, "subject")
    lines += ["by lighting"] + table(records, "lighting")
    for name, r in zip(labels([r["target"] for r in records]), records):
        parts = ([(f"{name}/fold{f['fold']}", f) for f in r["folds"]]
                 if "folds" in r else [(name, r)])
        for part_name, part in parts:
            same, line = _check(part_name, part)
            ok &= same
            lines.append(line)
    return lines, ok


def _check(name, r):
    """(matches, line) for one checkpoint against its single.json."""
    exp = r["expected"]
    if not exp:
        return True, f"{name}: no single.json beside it, nothing to check against"
    same = all(exp[s] is None or abs(exp[s] - r["overall"][s]) < 1e-9 for s in exp)
    people = r.get("recorded_people")
    moved = people is not None and people != r["held"]
    return same and not moved, (
        f"{name}: {'matches' if same and not moved else 'DOES NOT MATCH'} single.json ("
        + ", ".join(f"{s} {r['overall'][s]:.4f} vs {exp[s]}" for s in exp)
        + (f"; held out {r['held']}, single.json says {people}" if moved else "") + ")")


def write_csv(records, path):
    names = labels([r["target"] for r in records])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "split", "subject", "lighting", "label", "pred", "file"])
        for name, r in zip(names, records):
            for split in splits_of([r]):
                for c in r["clips"][split]:
                    w.writerow([f"{name}/fold{c['fold']}" if "fold" in c else name,
                                split, c["subject"], c["lighting"], c["label"],
                                c["pred"], os.path.basename(c["path"])])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True,
                    help="the config the checkpoints trained with (dataset, run.seed, "
                         "search.val_split)")
    ap.add_argument("targets", nargs="+",
                    help="run folders, .pth files, or folders of fold runs (fold0, ...)")
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
