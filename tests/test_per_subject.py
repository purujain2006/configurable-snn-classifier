"""Accuracy per person (tools/analysis/per_subject.py).

    python tests/test_per_subject.py

Trains a tiny network on synthetic clips named like DVS128 Gesture's files,
then requires the tool's overall validation and test scores to equal the
ones `single` wrote. Anything else means it rebuilt a different split or
scored a different network, and its per-person table would describe neither.
Runs on a CPU in under a minute.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "analysis"))

import per_subject  # noqa: E402

LIGHTS = ("fluorescent", "fluorescent_led", "lab", "led", "natural")


def named_datasets(root=None, **kw):
    """examples/synthetic_data.py, with every clip named after a person and a
    lighting: eight people in train, three in test."""
    spec = importlib.util.spec_from_file_location(
        "_synthetic", os.path.join(ROOT, "examples", "synthetic_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bundle = mod.make_datasets(n_train=64, n_test=24, **kw)
    for ds, first, people in ((bundle.train, 1, 8), (bundle.test, 24, 3)):
        n = len(ds)
        ds.samples = [(f"/data/{i % 2}/user{first + i * people // n:02d}_"
                       f"{LIGHTS[i % len(LIGHTS)]}_{i % 3}.npz", i % 2) for i in range(n)]
        # Wrong labels on about a third of the clips, so no network scores
        # 1.0 and the overall score depends on exactly which clips are in a
        # split: a tool that rebuilt the wrong one would not match.
        y = ds.tensors[1]
        flip = (torch.arange(n) * 7) % 10 < 3
        y[flip] = (y[flip] + 1) % bundle.num_classes
    return bundle


class Names(unittest.TestCase):
    def test_recording_of(self):
        r = per_subject.recording_of
        self.assertEqual(r("user07_fluorescent_led_3.npz"), (7, "fluorescent_led"))
        self.assertEqual(r("/x/frames/test/4/user24_lab_0.npz"), (24, "lab"))
        self.assertEqual(r("user01_led.npz"), (1, "led"))
        self.assertIsNone(r("clip_0003.npz"))

    def test_labels_keep_colliding_folder_names_apart(self):
        self.assertEqual(per_subject.labels(["a/rank1", "a/rank2"]), ["rank1", "rank2"])
        self.assertEqual(per_subject.labels(["top5/rank1", "aug/rank1", "aug/rank2"]),
                         [os.path.join("top5", "rank1"), os.path.join("aug", "rank1"),
                          "rank2"])


class MatchesSingle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from snnsearch import runconfig
        from snnsearch.pipeline import run_single
        cls.out = tempfile.mkdtemp(prefix="snn_per_subject_")
        cls.cfg = runconfig.load(overrides={
            "run": {"name": "per_subject", "results_dir": cls.out, "seed": 1},
            "dataset": {"name": None, "module": os.path.abspath(__file__),
                        "factory": "named_datasets"},
            "encoding": {"coding": "levels", "T": 2, "resize_to": None},
            "search": {"epochs": 2, "batch_size": 8, "num_workers": 0},
            "objective": {"mode": "accuracy"},
            "report": {"html": False},
        })
        assert run_single(cls.cfg, ckpt="net.pth") == 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_overall_scores_equal_single_json(self):
        import torch
        records = per_subject.score(self.cfg, [self.out], torch.device("cpu"))
        lines, ok = per_subject.report(records)
        self.assertTrue(ok, "\n".join(lines))
        with open(os.path.join(self.out, "single.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        r = records[0]
        self.assertEqual(r["overall"]["val"], saved["hw_val_accuracy"])
        self.assertEqual(r["overall"]["test"], saved["hw_test_accuracy"])
        for split in per_subject.SPLITS:
            self.assertTrue(0 < r["overall"][split] < 1, (split, r["overall"]))
        # validation is whole people, none of whom are in test
        val_people = {c["subject"] for c in r["clips"]["val"]}
        test_people = {c["subject"] for c in r["clips"]["test"]}
        self.assertTrue(val_people and test_people)
        self.assertFalse(val_people & test_people)
        self.assertEqual(sorted(val_people), r["held"])

    def test_a_different_split_does_not_match(self):
        """run.seed 2 holds out other people than the run did, and the check
        has to say so rather than print a table for the wrong people."""
        import copy
        cfg = copy.deepcopy(self.cfg)
        cfg["run"]["seed"] = 2
        records = per_subject.score(cfg, [self.out], torch.device("cpu"))
        self.assertNotEqual(records[0]["held"], [4, 7])
        lines, ok = per_subject.report(records)
        self.assertFalse(ok)
        self.assertIn("DOES NOT MATCH", "\n".join(lines))

    def test_tables_and_csv(self):
        import csv
        import torch
        records = per_subject.score(self.cfg, [self.out, self.out], torch.device("cpu"))
        lines, ok = per_subject.report(records)
        text = "\n".join(lines)
        self.assertTrue(ok, text)
        self.assertIn("by person", text)
        self.assertIn("by lighting", text)
        self.assertIn("mean", text)
        path = os.path.join(self.out, "clips.csv")
        per_subject.write_csv(records, path)
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        n = sum(len(records[0]["clips"][s]) for s in per_subject.SPLITS)
        self.assertEqual(len(rows), 2 * n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
