"""Feeding the chip fewer frames (tools/analysis/early_exit.py).

    python tests/test_early_exit.py

Trains a tiny network on synthetic event clips whose class shows only in the
second half of the frames, then requires the tool's k = T row to equal the
scores `single` wrote, and the rows with fewer frames to differ from it.
Anything else means it rebuilt a different split, scored a different network,
or did not actually cut the input. Runs on a CPU in under a minute.
"""
import csv
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

import early_exit  # noqa: E402

T = 4


def late_clips(root=None, T=T, **_ignored):
    """Two classes of event clips, H = W = 16, alternating. The first T // 2
    frames are empty and the rest hold sparse noise plus the class's bright
    quadrant, so fed only the empty frames every clip looks the same and gets
    the same answer. Clips are named after people as DVS128 Gesture's are, so
    validation holds out whole people."""
    from torch.utils.data import TensorDataset
    from snnsearch.data.base import DatasetBundle
    g = torch.Generator().manual_seed(0)

    def build(n, first, people):
        y = torch.arange(n) % 2
        x = torch.zeros(n, T, 2, 16, 16)
        x[:, T // 2:] = (torch.rand(n, T - T // 2, 2, 16, 16, generator=g) < 0.05).float()
        for i, cls in enumerate(y.tolist()):
            c = cls * 8
            x[i, T // 2:, :, :8, c:c + 8] = (
                torch.rand(T - T // 2, 2, 8, 8, generator=g) < 0.8).float()
        # Wrong labels on every tenth clip, so no network scores 1.0 and the
        # score depends on exactly which clips are in a split.
        flip = torch.arange(n) % 10 == 0
        y[flip] = 1 - y[flip]
        ds = TensorDataset(x, y)
        ds.samples = [(f"user{first + i * people // n:02d}_lab_{i % 3}.npz", int(y[i]))
                      for i in range(n)]
        return ds

    return DatasetBundle(train=build(64, 1, 8), test=build(24, 24, 3), C=2, H=16, W=16,
                         num_classes=2, name="late_clips", is_event=True)


class Pieces(unittest.TestCase):
    def test_frame_counts(self):
        fc = early_exit.frame_counts
        self.assertEqual(fc(48), [8, 16, 24, 32, 40, 48])
        self.assertEqual(fc(20), [8, 16, 20])
        self.assertEqual(fc(4), [4])
        self.assertEqual(fc(48, ks=[4, 12, 100]), [4, 12, 48])
        self.assertEqual(fc(48, every=16), [16, 32, 48])

    def test_settle_frames(self):
        """flush 2, T 4: output step s carries frames 0..s - 2."""
        steps = torch.zeros(6, 4, 3)
        steps[0, 0, 1] = 1            # clip 0: class 1 from the first step on
        steps[3, 1, 2] = 1            # clip 1: 0 until step 3, then 2
        steps[2, 2, 1] = 1            # clip 2: 1 at step 2, 2 wins at step 5
        steps[4, 2, 2] = 1
        steps[5, 2, 2] = 1
        steps[1, 3, 2] = 1            # clip 3: 2, then back to 0 at step 4
        steps[4, 3, 0] = 2
        got = early_exit.settle_frames(steps, flush=2, T=4).tolist()
        self.assertEqual(got, [0, 2, 4, 3])


class MatchesSingle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from snnsearch import runconfig
        from snnsearch.pipeline import run_single
        cls.out = tempfile.mkdtemp(prefix="snn_early_exit_")
        cls.cfg = runconfig.load(overrides={
            "run": {"name": "early_exit", "results_dir": cls.out, "seed": 1},
            "dataset": {"name": None, "module": os.path.abspath(__file__),
                        "factory": "late_clips"},
            "encoding": {"coding": "passthrough", "T": T, "resize_to": None},
            "search": {"epochs": 8, "batch_size": 8, "num_workers": 0},
            "objective": {"mode": "accuracy"},
            "report": {"html": False},
        })
        assert run_single(cls.cfg, ckpt="net.pth") == 0
        cls.records = early_exit.score(cls.cfg, [cls.out], torch.device("cpu"),
                                       ks=[1, 2, 3], splits=("val", "test"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_full_k_equals_single_json(self):
        lines, ok = early_exit.report(self.records)
        text = "\n".join(lines)
        self.assertTrue(ok, text)
        self.assertIn("k = T matches single.json", text)
        with open(os.path.join(self.out, "single.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        r = self.records[0]
        self.assertEqual(r["ks"], [1, 2, 3, T])
        self.assertEqual(r["splits"]["val"]["acc"][T], saved["hw_val_accuracy"])
        self.assertEqual(r["splits"]["test"]["acc"][T], saved["hw_test_accuracy"])

    def test_fewer_frames_change_accuracy_and_cost_less(self):
        r = self.records[0]
        for split in ("val", "test"):
            part = r["splits"][split]
            # the first T // 2 frames are empty: fed only those, every clip
            # looks the same, so the input really was cut
            for k in range(1, T // 2 + 1):
                self.assertEqual(len(set(part["pred"][k])), 1, (split, k))
            self.assertGreater(len(set(part["pred"][T])), 1, split)
            self.assertTrue(any(part["acc"][k] != part["acc"][T] for k in (1, 2, 3)),
                            (split, part["acc"]))
            syn = [part["synops"][k] for k in r["ks"]]
            self.assertNotIn(None, syn, part["synops_reason"])
            self.assertEqual(syn, sorted(syn), (split, syn))
            self.assertLess(syn[0], syn[-1])
            self.assertEqual(len(part["settled"]), part["clips"])
            self.assertTrue(all(0 <= f <= T for f in part["settled"]))

    def test_a_different_split_does_not_match(self):
        """The old clip split holds out 9 clips where the run held out 16, and
        the check has to say so rather than print a table for the wrong clips.
        9 and 16 share no factor, so the two scores can only agree at 0 or 1."""
        import copy
        right = self.records[0]["splits"]["val"]
        self.assertEqual(right["clips"], 16)
        self.assertTrue(0 < right["acc"][T] < 1, right["acc"])
        cfg = copy.deepcopy(self.cfg)
        cfg["search"]["val_split"] = "clip"
        records = early_exit.score(cfg, [self.out], torch.device("cpu"))
        self.assertIsNone(records[0]["held"])
        self.assertEqual(records[0]["splits"]["val"]["clips"], 9)
        lines, ok = early_exit.report(records)
        self.assertFalse(ok)
        self.assertIn("DOES NOT MATCH", "\n".join(lines))

    def test_two_columns_and_csv(self):
        records = self.records + self.records
        lines, ok = early_exit.report(records)
        text = "\n".join(lines)
        self.assertTrue(ok, text)
        for part in ("accuracy", "SynOps per clip", "settled by frame", "mean", "test ("):
            self.assertIn(part, text)
        path = os.path.join(self.out, "early_exit.csv")
        early_exit.write_csv(records, path)
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2 * 2 * 4)
        full = [row for row in rows if row["k"] == str(T)]
        self.assertTrue(all(float(row["settled_by_k"]) == 1.0 for row in full))


if __name__ == "__main__":
    unittest.main(verbosity=2)
