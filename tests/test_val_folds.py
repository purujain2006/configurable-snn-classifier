"""K-fold cross-subject validation (search.val_folds, `single --folds`).

    python tests/test_val_folds.py

The folds are checked on a training set shaped like DVS128 Gesture's: 23
people with file names like user07_led_3.npz and 36 to 60 clips each, 1176 in
all. Every fold must hold out whole people that are absent from its training
indices, the folds must cover every person exactly once with near-equal clip
counts, and val_folds 0 must give the split every run so far used (run.seed 1
holds out people 6, 10, 17 and 21, 216 clips). Then the summary `single
--folds` writes, from fold folders made by hand, and two folds of a real
tiny run. Runs on a CPU in about a minute. No dataset needed.
"""
import copy
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snnsearch import runconfig  # noqa: E402
from snnsearch.cli import _overrides, build_parser, single_folds  # noqa: E402
from snnsearch.data.loaders import (split_indices, split_settings,  # noqa: E402
                                    subject_folds, subject_ids)
from snnsearch import folds as fold_mod  # noqa: E402

# clips per person: the four people seed 1 holds out total 216, and no three
# of them reach 15% of 1176, so the old split stops at exactly those four
CLIPS = {6: 60, 21: 60, 10: 48, 17: 48}
for _u, _c in zip([u for u in range(1, 24) if u not in CLIPS], [36] * 2 + [60] * 6 + [48] * 11):
    CLIPS[_u] = _c
LIGHTS = ("fluorescent", "fluorescent_led", "lab", "led", "natural")


class Clips:
    """A dataset as split_indices sees one: a length and `.samples` paths."""

    def __init__(self, clips=CLIPS, name="user{u:02d}_{light}_{i}.npz"):
        self.samples = [(f"/data/{i % 11}/" + name.format(u=u, light=LIGHTS[i % 5], i=i // 5),
                         i % 11)
                        for u in sorted(clips) for i in range(clips[u])]

    def __len__(self):
        return len(self.samples)


def old_split(dataset, val_fraction=0.15, seed=1):
    """split_indices(val_split="subject") as it was before folds, frozen here."""
    ids = subject_ids(dataset)
    n = len(dataset)
    subjects = sorted(set(ids))
    random.Random(seed).shuffle(subjects)
    held, count = [], 0
    for s in subjects:
        if count >= val_fraction * n:
            break
        held.append(s)
        count += ids.count(s)
    held_set = set(held)
    return ([i for i, s in enumerate(ids) if s not in held_set],
            [i for i, s in enumerate(ids) if s in held_set], sorted(held))


class SingleSplitUnchanged(unittest.TestCase):
    def test_seed_1_holds_out_6_10_17_21(self):
        train, val, held = split_indices(Clips(), seed=1)
        self.assertEqual(held, [6, 10, 17, 21])
        self.assertEqual((len(val), len(train)), (216, 960))

    def test_val_folds_0_is_the_old_split(self):
        data = Clips()
        for seed in range(1, 8):
            with self.subTest(seed=seed):
                self.assertEqual(split_indices(data, seed=seed, folds=0, fold=0),
                                 old_split(data, seed=seed))

    def test_settings_default_to_the_old_split(self):
        cfg = runconfig.load()
        self.assertEqual((cfg["search"]["val_folds"], cfg["search"]["val_fold"]), (0, 0))
        self.assertEqual(split_settings(cfg),
                         {"seed": 1, "val_split": "subject", "folds": 0, "fold": 0})


class Folds(unittest.TestCase):
    def setUp(self):
        self.data = Clips()
        self.ids = subject_ids(self.data)

    def test_folds_hold_out_every_person_once(self):
        everyone = sorted(CLIPS)
        for k in (2, 3, 4, 5, 6, 8, 23):
            for seed in (1, 2, 1001):
                with self.subTest(k=k, seed=seed):
                    groups = subject_folds(self.ids, k, seed)
                    self.assertEqual(len(groups), k)
                    self.assertTrue(all(groups))
                    self.assertEqual(sorted(p for g in groups for p in g), everyone)
                    seen = []
                    for fold in range(k):
                        train, val, held = split_indices(self.data, seed=seed,
                                                         folds=k, fold=fold)
                        self.assertEqual(held, groups[fold])
                        # no validation person anywhere in this fold's training
                        self.assertFalse({self.ids[i] for i in train} & set(held))
                        self.assertEqual({self.ids[i] for i in val}, set(held))
                        self.assertEqual(sorted(train + val), list(range(len(self.data))))
                        seen += val
                    # the validation clips of the k folds: disjoint, all of them
                    self.assertEqual(sorted(seen), list(range(len(self.data))))

    def test_fold_sizes_are_near_equal(self):
        for k in (2, 3, 4, 5, 6, 8):
            for seed in (1, 2, 3):
                with self.subTest(k=k, seed=seed):
                    sizes = [sum(CLIPS[p] for p in g)
                             for g in subject_folds(self.ids, k, seed)]
                    # within one person's clips of each other
                    self.assertLess(max(sizes) - min(sizes), max(CLIPS.values()))
                    if k <= 5:
                        self.assertLessEqual(max(sizes) - min(sizes), 12, sizes)

    def test_sizes_even_out_for_any_people(self):
        for trial in range(100):
            r = random.Random(trial)
            clips = {p: r.randint(5, 70) for p in range(1, r.randint(4, 30))}
            ids = [p for p in clips for _ in range(clips[p])]
            k = r.randint(2, len(clips))
            with self.subTest(trial=trial, k=k):
                groups = subject_folds(ids, k, trial)
                self.assertEqual(sorted(p for g in groups for p in g), sorted(clips))
                sizes = [sum(clips[p] for p in g) for g in groups]
                self.assertLess(max(sizes) - min(sizes), max(clips.values()))

    def test_run_seed_decides_the_folds(self):
        self.assertEqual(subject_folds(self.ids, 5, 1), subject_folds(self.ids, 5, 1))
        self.assertNotEqual(subject_folds(self.ids, 5, 1), subject_folds(self.ids, 5, 2))

    def test_refusals_say_why(self):
        cases = [
            ("person id", lambda: split_indices(Clips(name="clip_{u}_{i}.npz"), folds=5)),
            ("person id", lambda: split_indices(type("NoNames", (), {"__len__": lambda s: 9})(),
                                                folds=3)),
            ("2 to 23", lambda: split_indices(self.data, folds=24)),
            ("2 to 23", lambda: subject_folds(self.ids, 1)),
            ("0 to 4", lambda: split_indices(self.data, folds=5, fold=5)),
            ("val_split 'subject'", lambda: split_indices(self.data, val_split="clip",
                                                          folds=5)),
        ]
        for words, call in cases:
            with self.subTest(words=words):
                with self.assertRaises(SystemExit) as ctx:
                    call()
                self.assertIn(words, str(ctx.exception))


class Settings(unittest.TestCase):
    def test_config_checks(self):
        for search in ({"val_folds": 1}, {"val_folds": -2}, {"val_folds": 5, "val_fold": 5},
                       {"val_folds": 3, "val_fold": -1}):
            with self.subTest(search=search):
                with self.assertRaises(SystemExit):
                    runconfig.load(overrides={"search": search})
        cfg = runconfig.load(overrides={"search": {"val_folds": 5, "val_fold": 4}})
        self.assertEqual(split_settings(cfg)["folds"], 5)
        self.assertEqual(split_settings(cfg)["fold"], 4)

    def test_command_line(self):
        def parse(*argv):
            args = build_parser().parse_args(["single", *argv])
            o = _overrides(args)
            cfg = runconfig.load(overrides=o)
            return o.get("search", {}), single_folds(cfg, args)
        self.assertEqual(parse(), ({}, None))
        self.assertEqual(parse("--folds", "5"), ({"val_folds": 5}, [0, 1, 2, 3, 4]))
        # fold 0 is a value, not "unset"
        self.assertEqual(parse("--folds", "5", "--fold", "0"),
                         ({"val_folds": 5, "val_fold": 0}, [0]))
        with self.assertRaises(SystemExit) as ctx:
            parse("--fold", "2")          # no fold count anywhere
        self.assertIn("--folds", str(ctx.exception))
        args = build_parser().parse_args(["single", "--fold", "3"])
        o = _overrides(args)
        o["search"]["val_folds"] = 4      # as a config file would set it
        self.assertEqual(single_folds(runconfig.load(overrides=o), args), [3])
        args = build_parser().parse_args(["single"])
        cfg = runconfig.load(overrides={"search": {"val_folds": 4, "val_fold": 2}})
        self.assertEqual(single_folds(cfg, args), [2])


def write_fold(root, fold, k, val, clips, people, train_acc=0.9, config=None):
    d = os.path.join(root, f"fold{fold}")
    os.makedirs(d)
    with open(os.path.join(d, "single.json"), "w", encoding="utf-8") as fh:
        json.dump({"hw_val_accuracy": val, "hw_test_accuracy": 0.123456,
                   "wall_minutes": 1.0, "config": config or {"train": {"seed": 1}},
                   "split": {"val_split": "subject", "seed": 1, "val_folds": k,
                             "val_fold": fold, "val_people": people,
                             "val_clips": clips, "train_clips": 1176 - clips}}, fh)
    with open(os.path.join(d, "progress.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"epoch": 1, "train_acc": 0.5}) + "\n")
        fh.write(json.dumps({"epoch": 2, "train_acc": train_acc}) + "\n")


def keys_naming_test(node):
    """Every key anywhere in `node` that names the test set."""
    if isinstance(node, dict):
        return ([k for k in node if "test" in k.lower()]
                + [k for v in node.values() for k in keys_naming_test(v)])
    if isinstance(node, list):
        return [k for v in node for k in keys_naming_test(v)]
    return []


class Summary(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="snn_folds_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_mean_standard_error_and_no_test(self):
        vals, clips = [0.70, 0.80, 0.90], [240, 228, 240]
        for fold, (v, n) in enumerate(zip(vals, clips)):
            write_fold(self.root, fold, 5, v, n, [fold + 1], train_acc=0.95 + fold / 100)
        s = fold_mod.write_summary(self.root)
        self.assertEqual(s["val_folds"], 5)
        self.assertEqual(s["folds_done"], [0, 1, 2])
        self.assertEqual(s["folds_missing"], [3, 4])
        self.assertAlmostEqual(s["mean_val_accuracy"], 0.80)
        self.assertAlmostEqual(s["se_val_accuracy"], 0.1 / math.sqrt(3))
        self.assertAlmostEqual(s["pooled_val_accuracy"],
                               sum(v * n for v, n in zip(vals, clips)) / sum(clips))
        self.assertEqual([r["final_train_accuracy"] for r in s["per_fold"]],
                         [0.95, 0.96, 0.97])
        self.assertEqual(s["problems"], [])
        with open(os.path.join(self.root, "folds.json"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertEqual(keys_naming_test(json.loads(text)), [])
        self.assertNotIn("0.123456", text)
        printed = "\n".join(fold_mod.summary_lines(s))
        self.assertIn("+/- 0.0577", printed)
        self.assertIn("not run yet: folds 3, 4", printed)
        self.assertNotIn("0.1234", printed)

    def test_different_networks_are_flagged(self):
        write_fold(self.root, 0, 2, 0.7, 100, [1])
        write_fold(self.root, 1, 2, 0.8, 100, [2], config={"train": {"seed": 1001}})
        s = fold_mod.summarize(self.root)
        self.assertTrue(any("same network" in p for p in s["problems"]), s["problems"])
        self.assertIn("WARNING", "\n".join(fold_mod.summary_lines(s)))


def named_datasets(root=None, **kw):
    """tests/test_per_subject.py's synthetic clips named after people."""
    spec = importlib.util.spec_from_file_location(
        "_named", os.path.join(ROOT, "tests", "test_per_subject.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.named_datasets(root, **kw)


class TwoFoldsEndToEnd(unittest.TestCase):
    """`single --folds 2` on tiny synthetic data: two short trainings."""

    @classmethod
    def setUpClass(cls):
        from snnsearch.cli import main
        cls.out = tempfile.mkdtemp(prefix="snn_val_folds_")
        cls.cfg_path = os.path.join(cls.out, "tiny.json")
        cls.cfg = {
            "run": {"name": "folds", "seed": 1},
            "dataset": {"name": None, "module": os.path.abspath(__file__),
                        "factory": "named_datasets"},
            "encoding": {"coding": "levels", "T": 2, "resize_to": None},
            "search": {"epochs": 2, "batch_size": 8, "num_workers": 0},
            "objective": {"mode": "accuracy"},
            "report": {"html": False},
        }
        with open(cls.cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cls.cfg, fh)
        cls.results = os.path.join(cls.out, "results")
        assert main(["single", "-c", cls.cfg_path, "--results-dir", cls.results,
                     "--ckpt", "net.pth", "--folds", "2"]) == 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_each_fold_is_a_run_without_test(self):
        people = []
        for fold in (0, 1):
            with open(os.path.join(self.results, f"fold{fold}", "single.json"),
                      encoding="utf-8") as fh:
                rec = json.load(fh)
            self.assertEqual((rec["split"]["val_folds"], rec["split"]["val_fold"]), (2, fold))
            self.assertNotIn("hw_test_accuracy", rec)
            self.assertIn("test_skipped", rec)
            self.assertIsNotNone(rec["hw_val_accuracy"])
            people.append(rec["split"]["val_people"])
        # named_datasets has people 1 to 8 in train
        self.assertFalse(set(people[0]) & set(people[1]))
        self.assertEqual(sorted(people[0] + people[1]), list(range(1, 9)))

    def test_summary_covers_both_folds(self):
        with open(os.path.join(self.results, "folds.json"), encoding="utf-8") as fh:
            s = json.load(fh)
        self.assertEqual((s["val_folds"], s["folds_done"], s["folds_missing"]), (2, [0, 1], []))
        self.assertEqual(s["problems"], [])
        self.assertEqual(s["val_clips"], 64)
        self.assertEqual(keys_naming_test(s), [])

    def test_one_fold_alone_rewrites_the_summary(self):
        from snnsearch.cli import main
        other = os.path.join(self.out, "one")
        self.assertEqual(main(["single", "-c", self.cfg_path, "--results-dir", other,
                               "--ckpt", "net.pth", "--folds", "2", "--fold", "1"]), 0)
        self.assertFalse(os.path.exists(os.path.join(other, "fold0")))
        s = fold_mod.summarize(other)
        self.assertEqual((s["folds_done"], s["folds_missing"]), ([1], [0]))

    def test_per_subject_scores_every_person_once(self):
        """tools/analysis/per_subject.py on the folder of folds: each fold's
        network on its own people, matching each fold's single.json."""
        import torch
        sys.path.insert(0, os.path.join(ROOT, "tools", "analysis"))
        import per_subject
        cfg = runconfig.load(self.cfg_path)
        records = per_subject.score(cfg, [self.results], torch.device("cpu"))
        lines, ok = per_subject.report(records)
        text = "\n".join(lines)
        self.assertTrue(ok, text)
        r = records[0]
        self.assertEqual(list(r["clips"]), ["val"])
        self.assertEqual(sorted({c["subject"] for c in r["clips"]["val"]}), list(range(1, 9)))
        self.assertEqual(len(r["clips"]["val"]), 64)
        self.assertEqual(text.count("matches single.json"), 2)
        self.assertNotIn("test:", text)
        # a fold folder and a single-split run hold out different people
        with self.assertRaises(SystemExit):
            per_subject.score(cfg, [self.results, os.path.join(self.results, "fold0")],
                              torch.device("cpu"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
