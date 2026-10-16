"""The checks that stop a broken search before it starts.

    python tests/test_preflight.py

Every rule here was learned from a search that ran for hours first: an
early-stopping cut inside the float warm-up, a package file that never
reached the cluster, a config changed after it was checked.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snnsearch import runconfig
from snnsearch.preflight import (check_search_config, fingerprint, preflight_gate,
                                 warmup_epochs, _missing_package_modules)


def cfg(**search):
    c = runconfig.load(overrides={
        "dataset": {"module": os.path.join(ROOT, "examples", "synthetic_data.py")},
        "search": {"trials": 100, "epochs": 80, "grace_period": 30, "space": "per_layer",
                   "balanced_trials": 48, **search}})
    c["_source"] = "test.yaml"
    return c


class ConfigRules(unittest.TestCase):
    def test_first_cut_must_come_after_the_float_warmup(self):
        self.assertEqual(warmup_epochs(80), 20)
        errors, _ = check_search_config(cfg(grace_period=20))
        self.assertEqual(len(errors), 1)
        self.assertIn("float warm-up", errors[0])
        self.assertIn("at least 21", errors[0])
        errors, warnings = check_search_config(cfg(grace_period=22))
        self.assertEqual(errors, [])
        self.assertTrue(any("dips" in w for w in warnings))
        self.assertEqual(check_search_config(cfg()), ([], []))

    def test_other_rules(self):
        self.assertTrue(check_search_config(cfg(balanced_trials=200))[0])
        self.assertTrue(check_search_config(cfg(epochs=1))[0])
        self.assertTrue(check_search_config(cfg(T_choices=[8, "16"]))[0])
        self.assertTrue(check_search_config(cfg(space="uniform"))[1])
        self.assertTrue(check_search_config(cfg(protect_balanced=False))[1])
        self.assertTrue(check_search_config(cfg(finalists=3, finalist_seeds=1))[1])
        no_root = runconfig.load(overrides={"dataset": {"name": "dvs128", "root": "/nowhere"},
                                            "search": {"epochs": 80, "grace_period": 30}})
        self.assertTrue(any("dataset.root" in e for e in check_search_config(no_root)[0]))

    def test_shipped_configs(self):
        """Every config in configs/ passes, except Search 04's, which is kept
        as the record of the search that had the warm-up problem."""
        here = os.path.join(ROOT, "configs")
        for name in sorted(os.listdir(here)):
            if not name.endswith(".yaml"):
                continue
            with self.subTest(name=name):
                c = runconfig.load(os.path.join(here, name), {"dataset": {"root": ROOT}})
                errors, _ = check_search_config(c)
                if name == "dvs128_search04.yaml":
                    self.assertTrue(errors)
                else:
                    self.assertEqual(errors, [])


class Limits(unittest.TestCase):
    def test_a_config_can_change_the_chip_limits_and_the_next_one_resets_them(self):
        from snnsearch.hardware import AXON_LIMITS, NEURON_LIMITS
        runconfig.load(overrides={"limits": {"total_axons": 20000, "neuron_fan_in": 9000}})
        self.assertEqual((AXON_LIMITS["total_axons"], NEURON_LIMITS["fan_in"]), (20000, 9000))
        runconfig.load()
        self.assertEqual((AXON_LIMITS["total_axons"], NEURON_LIMITS["fan_in"]), (16000, 8159))
        with self.assertRaises(SystemExit):
            runconfig.load(overrides={"limits": {"axons": 1}})


class Gate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cfg = cfg()

    def write(self, **rec):
        base = {"passed": True, "quick": False, "fingerprint": fingerprint(self.cfg),
                "when": "now"}
        with open(os.path.join(self.dir, "preflight.json"), "w") as fh:
            json.dump({**base, **rec}, fh)

    def test_search_waits_for_a_full_matching_preflight(self):
        self.assertIn("No preflight", preflight_gate(self.cfg, self.dir))
        self.write(passed=False)
        self.assertIn("failed", preflight_gate(self.cfg, self.dir))
        self.write(quick=True)
        self.assertIn("quick", preflight_gate(self.cfg, self.dir))
        self.write()
        self.assertIsNone(preflight_gate(self.cfg, self.dir))
        self.cfg["search"]["skip_preflight"] = True
        self.write(passed=False)
        self.assertIsNone(preflight_gate(self.cfg, self.dir))

    def test_what_invalidates_a_preflight(self):
        self.write()
        self.cfg["search"]["trials"] = 999          # only how long it runs
        self.assertIsNone(preflight_gate(self.cfg, self.dir))
        self.cfg["search"]["epochs"] = 60           # what it does
        self.assertIn("changed", preflight_gate(self.cfg, self.dir))

    def test_fingerprint_is_stable(self):
        self.assertEqual(fingerprint(cfg()), fingerprint(cfg()))
        self.assertNotEqual(fingerprint(cfg()), fingerprint(cfg(grace_period=31)))


class Environment(unittest.TestCase):
    def test_every_package_module_that_is_imported_exists(self):
        """A file that never reached the cluster (git commit -a skips new
        files) shows up here instead of hours into a run."""
        self.assertEqual(_missing_package_modules(), [])


@unittest.skipUnless(importlib.util.find_spec("ray"), "needs ray")
class Scheduler(unittest.TestCase):
    def test_balanced_block_is_never_cut(self):
        from types import SimpleNamespace
        from ray.tune.schedulers import TrialScheduler
        from snnsearch.scheduling import BalancedBlockASHA
        s = BalancedBlockASHA(protect=3, time_attr="t", metric="acc", mode="max",
                              max_t=10, grace_period=1, reduction_factor=2)
        trials = [SimpleNamespace(trial_id=f"t{i}") for i in range(8)]

        def run(scheduler):
            for tr in trials:
                scheduler.on_trial_add(None, tr)
            # the five after the block score well first, then the block scores
            # worst, so plain ASHA cuts the block
            order = trials[3:] + trials[:3]
            return {tr.trial_id: scheduler.on_trial_result(None, tr, {"t": 1, "acc": 1 - k / 10})
                    for k, tr in enumerate(order)}

        plain = run(BalancedBlockASHA(protect=0, time_attr="t", metric="acc", mode="max",
                                      max_t=10, grace_period=1, reduction_factor=2))
        self.assertTrue(all(plain[f"t{i}"] == TrialScheduler.STOP for i in range(3)), plain)
        s = BalancedBlockASHA(protect=3, time_attr="t", metric="acc", mode="max",
                              max_t=10, grace_period=1, reduction_factor=2)
        kept = run(s)
        self.assertTrue(all(kept[f"t{i}"] == TrialScheduler.CONTINUE for i in range(3)), kept)
        self.assertTrue(all(s.is_protected(f"t{i}") for i in range(3)))
        self.assertFalse(s.is_protected("t3"))
        # at max_t a protected trial still finishes
        self.assertEqual(s.on_trial_result(None, trials[0], {"t": 10, "acc": 0.0}),
                         TrialScheduler.STOP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
