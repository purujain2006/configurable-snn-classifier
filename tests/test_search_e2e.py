"""A real search and a real option study, end to end, on synthetic data.

    python tests/test_search_e2e.py        # about 5 minutes; needs ray and optuna

Everything a search does after its trials finish (finalists, the report) and
everything the study does had only ever run at the end of a multi-hour
search, so a bug there cost the whole run. This drives them through Ray in
minutes: a balanced block that early stopping must leave alone, trials after
it that it may cut, finalists, the report, and an option study.

`main.py preflight` runs the same kind of mini search on the real dataset and
GPU before every search; this test is for changes to the code itself.
"""
import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HAVE_RAY = bool(importlib.util.find_spec("ray") and importlib.util.find_spec("optuna"))


def config(out):
    from snnsearch import runconfig
    return runconfig.load(overrides={
        "run": {"name": "e2e", "results_dir": out, "seed": 1},
        "dataset": {"module": os.path.join(ROOT, "examples", "synthetic_data.py"),
                    "n_train": 64, "n_test": 16},
        "encoding": {"coding": "levels", "T": 2},
        "search": {"trials": 9, "epochs": 4, "grace_period": 2, "reduction_factor": 3,
                   "space": "per_layer", "balanced_trials": 6, "startup_trials": 0,
                   "finalists": 1, "finalist_seeds": 2, "gpu_fraction": 0.5,
                   "num_workers": 0, "cpu_per_trial": 1, "batch_size": 8,
                   "T_choices": [2], "skip_preflight": True},
        "report": {"html": True},
    })


@unittest.skipUnless(HAVE_RAY, "needs ray and optuna")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform == "win32":
            # Some Windows shells and IDE terminals run everything inside one
            # job object, which refuses the nested one Ray creates for its
            # children. Only Ray's clean-up on a crash depends on it.
            import ray._private.utils as ru
            ru.set_kill_child_on_death_win32 = lambda proc: None
        import ray
        from snnsearch.search import run_search
        cls.out = tempfile.mkdtemp(prefix="snn_e2e_")
        cls.cfg = config(cls.out)
        try:
            run_search(cls.cfg, cls.out)
        finally:
            if ray.is_initialized():
                ray.shutdown()

    def rows(self):
        with open(os.path.join(self.out, "trials.jsonl"), encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def test_every_trial_recorded_and_exact(self):
        rows = self.rows()
        self.assertEqual(len(rows), 9)
        for r in rows:
            if r.get("feasible") and r.get("hw_val_accuracy") is not None:
                self.assertEqual(r["sim_disagreements"], 0, r["trial_id"])
        with open(os.path.join(self.out, "leaderboard.csv"), encoding="utf-8") as fh:
            self.assertEqual(sum(1 for _ in csv.DictReader(fh)), 9)

    def test_balanced_block_ran_to_the_end(self):
        with open(os.path.join(self.out, "trial_progress.jsonl"), encoding="utf-8") as fh:
            progress = [json.loads(line) for line in fh if line.strip()]
        order = []
        for p in progress:
            if p["trial_id"] not in order:
                order.append(p["trial_id"])
        by_id = {r["trial_id"]: r for r in self.rows()}
        for tid in order[:6]:
            self.assertFalse(by_id[tid]["stopped_early"], tid)
            self.assertEqual(by_id[tid]["epochs_run"], 4, tid)

    def test_finalists_and_report(self):
        with open(os.path.join(self.out, "finalists.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual(sorted(int(r["seed"]) for r in rows), [1001, 2001])
        with open(os.path.join(self.out, "best_finalist.json"), encoding="utf-8") as fh:
            best = json.load(fh)
        self.assertEqual(best["best"]["seeds"], 2)
        from snnsearch import report
        self.assertTrue(os.path.isfile(report.build(self.out, cfg=self.cfg)))

    def test_option_study(self):
        import ray
        from snnsearch.pipeline import load_flat_config
        from snnsearch.study import run_option_study
        flat = load_flat_config(os.path.join(self.out, "best_finalist.json"))
        try:
            summary = run_option_study(self.cfg, flat, seeds=2, epochs=2, only=["loss"])
        finally:
            if ray.is_initialized():
                ray.shutdown()
        names = [r["variant"] for r in summary]
        self.assertEqual(names[0], "baseline")
        self.assertEqual(len(names), 2)
        self.assertTrue(all(r["seeds"] == 2 for r in summary))
        self.assertTrue(os.path.isfile(os.path.join(self.out, "study", "study_summary.csv")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
