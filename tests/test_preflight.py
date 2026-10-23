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


GiB = 2**30
L40S = 44.42 * GiB                  # mahowald's card, as torch reports it


class Memory(unittest.TestCase):
    """Search 05's first preflight measured 8.12 GiB for its heaviest trial and
    suggested 0.333, while 0.25 passed its own check: the check left out the
    CUDA context that the suggestion counted. It had also measured a block
    trial chosen by chip neurons, and missed a heavier one."""

    def test_ray_packing(self):
        from snnsearch.preflight import ray_trials_per_gpu
        for frac, n in ((1.0, 1), (0.5, 2), (0.333, 3), (0.25, 4), (0.167, 5), (0.166, 6)):
            self.assertEqual(ray_trials_per_gpu(frac), n, frac)

    def test_search05_numbers(self):
        from snnsearch.preflight import memory_verdict
        v = {f: memory_verdict(8.12 * GiB, L40S, f, 384, 5) for f in (0.167, 0.25, 0.333)}
        self.assertFalse(v[0.167]["fits_cap"])
        # its own cap holds it, but four at once plus their contexts overflow
        self.assertTrue(v[0.25]["fits_cap"])
        self.assertFalse(v[0.25]["fits_card"])
        self.assertTrue(v[0.333]["fits_cap"] and v[0.333]["fits_card"])
        self.assertEqual((v[0.25]["suggest"], v[0.25]["fit"]), (0.333, 3))

    def test_the_suggestion_always_passes(self):
        from snnsearch.preflight import memory_verdict
        for total in (8 * GiB, 24 * GiB, L40S, 80 * GiB):
            for tenth in range(1, 400):
                peak = tenth / 10 * GiB
                v = memory_verdict(peak, total, 1.0, 384, 5)
                if v["fit"] < 1:
                    self.assertIsNone(v["suggest"])
                    continue
                w = memory_verdict(peak, total, v["suggest"], 384, 5)
                self.assertTrue(w["fits_cap"] and w["fits_card"], (total, peak))
                self.assertEqual(w["resident"], v["fit"], (total, peak))

    def test_cpu_cores_can_limit_the_suggestion(self):
        from snnsearch.preflight import memory_verdict
        v = memory_verdict(1 * GiB, L40S, 1.0, 8, 4)
        self.assertEqual((v["fit"], v["suggest"], v["cpu_limited"]), (2, 0.5, True))


class HeaviestTrial(unittest.TestCase):
    """The memory check measures the heaviest trial, and random proposals
    never find it: on DVS128 Gesture the climb lands at 2.7 times the heaviest
    of 1,000 proposals that fit the chip."""

    @classmethod
    def setUpClass(cls):
        import random
        from snnsearch.spaces import (make_define_by_run, sample_config, balanced_points,
                                      _LEGACY_DATA)
        cls.data = dict(_LEGACY_DATA)
        cls.space = make_define_by_run(16, 80, ".", [16, 48], True, data=cls.data,
                                       max_input_bits=4, seed=1)
        rng = random.Random(0)
        block = balanced_points(24, seed=1, t_choices=[16, 48], batch_size=16, epochs=80,
                                data=cls.data, max_input_bits=4, run_seed=1)
        cls.block = [sample_config(cls.space, rng, fixed=p) for p in block]
        cls.random = [f for f in (sample_config(cls.space, rng) for _ in range(2000))
                      if not f.get("infeasible_geometry")]

    def test_climb_beats_every_random_proposal_and_fits_the_chip(self):
        from snnsearch.preflight import heaviest_shapes, activation_load
        from snnsearch.spaces import geometry_fits
        shapes = heaviest_shapes(self.space, self.block + self.random[:50], keep=2)
        self.assertEqual(len(shapes), 2)
        best = max(map(activation_load, self.random))
        self.assertGreater(activation_load(shapes[0]), best)
        self.assertGreaterEqual(activation_load(shapes[0]), activation_load(shapes[1]))
        for f in shapes:
            self.assertFalse(f.get("infeasible_geometry"))
            self.assertTrue(geometry_fits(f))
            self.assertEqual(f["T"], 48)

    def test_heavy_variants(self):
        from snnsearch.preflight import distinct_heaviest, heavy_variants, HEAVY_TRAINING
        shape = distinct_heaviest(self.block, 1)[0]
        variants = heavy_variants(self.space, shape)
        self.assertEqual(sorted(v["norm"] for v in variants), ["bn", "tdbn"])
        for v in variants:
            for k, want in HEAVY_TRAINING.items():
                self.assertEqual(v[k], want, k)
            self.assertEqual(v["resize_to"], shape["resize_to"])
            self.assertEqual(v["T"], shape["T"])

    def test_an_option_the_space_no_longer_offers_is_drawn_instead(self):
        import random
        from snnsearch.preflight import _prefer_sample
        f = _prefer_sample(self.space, random.Random(0),
                           {**self.block[0], "loss": "gone", "conv_dropout": 9.0})
        self.assertIn(f["loss"], ("rate", "tet"))
        self.assertLessEqual(f.get("conv_dropout", 0.0), 0.3)

    def test_uniform_space(self):
        import random
        from snnsearch.preflight import heaviest_shapes
        from snnsearch.spaces import make_define_by_run, sample_config
        space = make_define_by_run(16, 80, ".", [16, 48], False, data=self.data,
                                   max_input_bits=1, seed=1)
        rng = random.Random(0)
        fits = [f for f in (sample_config(space, rng) for _ in range(300))
                if not f.get("infeasible_geometry")]
        shapes = heaviest_shapes(space, fits, keep=1)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["T"], 48)


def _has_cuda():
    if not importlib.util.find_spec("torch"):
        return False
    import torch
    return torch.cuda.is_available()


def _gib_block_point():
    """The heaviest balanced-block trial under about 2 GiB at batch 16, light
    enough for a laptop GPU and heavy enough that rounding in the allocator
    does not decide the measurement."""
    import random
    from snnsearch.preflight import activation_load
    from snnsearch.spaces import (make_define_by_run, sample_config, balanced_points,
                                  _LEGACY_DATA)
    data = dict(_LEGACY_DATA)
    t = [16, 24, 32]
    space = make_define_by_run(16, 80, ".", t, True, data=data, max_input_bits=4, seed=1)
    rng = random.Random(0)
    pts = [sample_config(space, rng, fixed=p)
           for p in balanced_points(48, seed=1, t_choices=t, batch_size=16, epochs=80,
                                    data=data, max_input_bits=4, run_seed=1)]
    return max((p for p in pts if activation_load(p) <= 4e6), key=activation_load)


@unittest.skipUnless(_has_cuda(), "needs a CUDA GPU")
class MeasuredPeak(unittest.TestCase):
    def test_a_trial_that_fits_is_measured_directly(self):
        import torch
        from snnsearch.preflight import (estimate_training_peak, extrapolated_peak,
                                         measure_training_peak)
        from snnsearch.spaces import config_to_specs
        flat = _gib_block_point()
        dev = torch.device("cuda")
        direct = measure_training_peak(config_to_specs(flat), dev)
        self.assertGreater(direct, 1 * GiB)
        est, exact = estimate_training_peak(flat, dev)
        self.assertTrue(exact)
        self.assertLess(abs(est - direct) / direct, 0.02)
        # the heaviest trial of the space may not fit the GPU at its batch
        # size. Then it is extrapolated from batches 8 and 4
        line = extrapolated_peak(flat, dev, 8)
        self.assertLess(abs(line - direct) / direct, 0.10, (line / GiB, direct / GiB))


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
