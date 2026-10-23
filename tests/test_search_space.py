"""Search-space and generation regressions, without Optuna, Ray, or a GPU.

    python tests/test_search_space.py

Covers three promises: the search samples every option it claims to (and
only where it applies), the balanced block gives every option the same number
of trials with shapes that fit the chip, and nothing is tied to one dataset.
"""
from collections import Counter
import itertools
from pathlib import Path
import os
import pickle
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snnsearch.hardware import check_feasibility, check_spec_feasibility
from snnsearch.planning import plan_network
from snnsearch.spaces import (config_to_specs, make_define_by_run, balanced_points,
                              balanced_design, geometry_fits, resize_range,
                              BALANCED_FACTORS, TAU_MODES)
from snnsearch.streaming import LEADERBOARD_COLS

DVS = {"data_C": 2, "data_H": 128, "data_W": 128, "num_classes": 11, "hflip_ok": False}
CIFAR = {"data_C": 3, "data_H": 32, "data_W": 32, "num_classes": 10, "hflip_ok": True}
MNIST = {"data_C": 1, "data_H": 28, "data_W": 28, "num_classes": 10, "hflip_ok": False}

# every knob the search must try, and what it builds
NEW_KNOBS = ("input_bits", "signed_spikes", "tau_mode", "loss", "use_augment",
             "aug_shift", "aug_cutout", "aug_drop", "aug_flip")


class RecordingTrial:
    """Exercise conditional sampling while retaining the sampler API used."""

    def __init__(self, values=None):
        self.values = values or {}
        self.params = {}
        self.calls = {}

    def suggest_int(self, name, low, high, step=1, log=False):
        value = self.values.get(name, low)
        assert low <= value <= high and (value - low) % step == 0
        self.calls[name] = ("int", low, high, step, log)
        self.params[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = self.values.get(name, choices[0])
        assert value in choices
        self.calls[name] = ("categorical", list(choices))
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, log=False):
        value = self.values.get(name, low)
        assert low <= value <= high
        self.calls[name] = ("float", low, high, log)
        self.params[name] = value
        return value


def sample(per_layer=True, values=None, data=DVS, max_input_bits=4):
    space = make_define_by_run(4, 3, ".", [8, 16], per_layer=per_layer,
                               data=data, max_input_bits=max_input_bits, seed=7)
    # Ray checkpoints the callable before it samples another trial.
    space = pickle.loads(pickle.dumps(space))
    trial = RecordingTrial(values)
    constants = space(trial)
    return trial, {**trial.params, **constants}


# a small shape that fits every dataset here, so the training knobs get sampled
SMALL = {"depth": 2, "ch_0": 8, "ch_1": 8, "k_0": 3, "k_1": 3, "stride_0": 2, "stride_1": 2,
         "ds_0": "stride", "ds_1": "stride", "fc_layers": 0, "final_reduction": "gap"}


class SearchSpaceTests(unittest.TestCase):
    def test_ordered_knobs_and_conditional_layers(self):
        for per_layer in (False, True):
            for resize in (24, 88):
                with self.subTest(per_layer=per_layer, resize=resize):
                    trial, flat = sample(per_layer, {"resize_to": resize})
                    self.assertNotIn("final_reduction", flat.keys() - trial.params.keys())
                    self.assertEqual(trial.calls["resize_to"],
                                     ("int", 24, 88, 8, False))
                    names = [("channels", "kernel_size")]
                    if per_layer:
                        names = [(f"ch_{i}", f"k_{i}") for i in range(2)]
                        self.assertNotIn("ch_2", trial.params)
                        self.assertNotIn("channels", trial.params)
                    for channel, kernel in names:
                        self.assertEqual(trial.calls[channel],
                                         ("int", 8, 128, 1, True))
                        self.assertEqual(trial.calls[kernel],
                                         ("int", 3, 9, 2, False))
                    self.assertEqual(trial.calls["final_reduction"],
                                     ("categorical", ["flatten", "gap"]))
                    self.assertEqual(config_to_specs(flat)["input"].resize_to,
                                     resize)

    def test_gap_and_flatten_have_the_same_fan_in_on_chip(self):
        """GAP shrinks the head's parameters, not its fan-in.

        A global average pool has no neurons on chip. It folds into the head's
        synapses, so each head neuron still connects to every neuron of the
        last conv layer, exactly as a flatten head does. The plan's
        fc_in_features (32 for GAP) is the parameter count, and the
        feasibility check must not mistake it for the fan-in.
        """
        for per_layer in (False, True):
            outcomes = {}
            for reduction in ("flatten", "gap"):
                with self.subTest(per_layer=per_layer, reduction=reduction):
                    _, flat = sample(per_layer, {
                        "resize_to": 88, "channels": 32,
                        "ch_0": 32, "ch_1": 32,
                        "fc_layers": 1, "final_reduction": reduction,
                    })
                    spec = config_to_specs(flat)
                    self.assertEqual(spec["head"].final_reduction, reduction)
                    plan = plan_network(spec["input"], spec["encoder"],
                                        spec["output"], spec["downsample"],
                                        spec["head"])
                    feasible, violations = check_feasibility(
                        spec["input"], spec["encoder"], spec["downsample"],
                        spec["head"], spec["output"])
                    self.assertEqual(plan.fc_in_features,
                                     32 if reduction == "gap" else 32 * 21 * 21)
                    self.assertFalse(feasible)
                    fan_in = [v for v in violations if v.startswith("fc0: neuron_fan_in")]
                    self.assertEqual(len(fan_in), 1, violations)
                    outcomes[reduction] = fan_in[0]
            self.assertEqual(outcomes["flatten"], outcomes["gap"])
        self.assertIn("final_reduction", LEADERBOARD_COLS)

    def test_every_new_option_is_sampled(self):
        """input bits, signed spikes, tau mode, loss and augmentation are all
        dimensions of the search, with the choices the docs promise."""
        trial, flat = sample(True, {**SMALL, "resize_to": 32, "use_augment": True},
                             data=CIFAR)
        self.assertNotIn("infeasible_geometry", flat)
        self.assertEqual(trial.calls["input_bits"], ("int", 1, 4, 1, False))
        self.assertEqual(trial.calls["signed_spikes"], ("categorical", [False, True]))
        self.assertEqual(trial.calls["tau_mode"], ("categorical", TAU_MODES))
        self.assertEqual(trial.calls["loss"], ("categorical", ["rate", "tet"]))
        self.assertEqual(trial.calls["use_augment"], ("categorical", [False, True]))
        for name in ("aug_shift", "aug_cutout", "aug_drop"):
            self.assertEqual(trial.calls[name][0], "float", name)
        self.assertEqual(trial.calls["aug_flip"], ("categorical", [False, True]))
        for name in NEW_KNOBS:
            self.assertIn(name, LEADERBOARD_COLS, name)

    def test_options_only_where_they_apply(self):
        # no flips where a mirror changes the label, no augmentation amounts
        # when augmentation is off, one bit when the coding sends one bit
        trial, _ = sample(True, {**SMALL, "resize_to": 64, "use_augment": True}, data=DVS)
        self.assertNotIn("aug_flip", trial.calls)
        trial, _ = sample(True, {**SMALL, "resize_to": 64, "use_augment": False})
        self.assertFalse(any(k.startswith("aug_") for k in trial.calls))
        trial, flat = sample(True, {**SMALL, "resize_to": 64}, max_input_bits=1)
        self.assertNotIn("input_bits", trial.calls)
        self.assertEqual(config_to_specs(flat)["input"].input_bits, 1)

    def test_sampled_options_reach_the_specs(self):
        values = {**SMALL, "resize_to": 32, "input_bits": 3, "signed_spikes": True,
                  "tau_mode": "channel", "loss": "tet", "use_augment": True,
                  "aug_shift": 0.1, "aug_cutout": 0.2, "aug_drop": 0.05, "aug_flip": True}
        trial, flat = sample(True, values, data=CIFAR)
        spec = config_to_specs(flat)
        self.assertEqual((spec["input"].C, spec["input"].H, spec["input"].W), (3, 32, 32))
        self.assertEqual(spec["output"].num_classes, 10)
        self.assertEqual(spec["input"].input_bits, 3)
        self.assertTrue(spec["input"].hflip_ok)
        n = spec["neuron"]
        self.assertTrue(n.signed_spikes and n.trainable_tau and n.tau_per_channel)
        t = spec["train"]
        self.assertEqual((t.loss, t.augment, t.aug_shift, t.aug_cutout, t.aug_drop, t.aug_flip),
                         ("tet", True, 0.1, 0.2, 0.05, True))
        self.assertEqual(t.seed, 7)
        for mode, (trainable, per_channel) in {"fixed": (False, False),
                                               "layer": (True, False),
                                               "channel": (True, True)}.items():
            n = config_to_specs({**flat, "tau_mode": mode})["neuron"]
            self.assertEqual((n.trainable_tau, n.tau_per_channel), (trainable, per_channel))

    def test_signed_spikes_and_input_bits_are_checked_against_the_chip(self):
        """Both are geometry: each multiplies a fan-in or an axon count."""
        base = {**SMALL, "N": 4, "epochs": 3, "data_dir": ".", "T": 8, "tau": 2, **DVS}
        # 88 x 88 x 2 = 15,488 axons fits; two bits per value do not
        self.assertTrue(geometry_fits({**base, "resize_to": 88, "input_bits": 1}))
        self.assertFalse(geometry_fits({**base, "resize_to": 88, "input_bits": 2}))
        # 9 x 9 x 56 = 4,536 inputs per neuron fits; signed, 9,072 does not
        wide = {**base, "resize_to": 32, "ch_0": 56, "k_1": 9}
        ok_unsigned, _ = check_spec_feasibility(config_to_specs(wide))
        ok_signed, why = check_spec_feasibility(config_to_specs({**wide, "signed_spikes": True}))
        self.assertTrue(ok_unsigned)
        self.assertFalse(ok_signed)
        self.assertTrue(any("fan_in" in v for v in why), why)


class SampleConfigTests(unittest.TestCase):
    def test_random_configs_come_from_the_real_space(self):
        import random
        from snnsearch.spaces import sample_config
        space = make_define_by_run(4, 3, ".", [8, 16], data=CIFAR, max_input_bits=4, seed=7)
        rng = random.Random(0)
        flats = [sample_config(space, rng) for _ in range(200)]
        fits = [f for f in flats if not f.get("infeasible_geometry")]
        self.assertTrue(fits)
        for f in fits:
            self.assertTrue(16 <= f["resize_to"] <= 32 and f["T"] in (8, 16))
            self.assertIn(f["tau_mode"], TAU_MODES)
            config_to_specs(f)
        # a fixed value the space cannot take is refused, as Optuna would
        with self.assertRaises(ValueError):
            sample_config(space, rng, fixed={"depth": 9})


class GenericDatasetTests(unittest.TestCase):
    def test_resize_range_follows_the_dataset(self):
        self.assertEqual(resize_range(DVS), (24, 88, 8))       # 89^2 x 2 = 15,842 axons
        self.assertEqual(resize_range(CIFAR), (16, 32, 4))     # native fits
        self.assertEqual(resize_range(MNIST), (16, 28, 4))
        lo, hi, step = resize_range({**CIFAR, "data_H": 30, "data_W": 30})
        self.assertEqual(hi, 30)                               # native kept off-grid
        self.assertEqual((hi - lo) % step, 0)

    def test_space_uses_the_dataset_not_dvs(self):
        for data in (CIFAR, MNIST):
            with self.subTest(data=data):
                lo, hi, step = resize_range(data)
                trial, flat = sample(True, {**SMALL, "resize_to": hi}, data=data)
                self.assertEqual(trial.calls["resize_to"], ("int", lo, hi, step, False))
                spec = config_to_specs(flat)
                self.assertEqual((spec["input"].C, spec["input"].H, spec["input"].W,
                                  spec["output"].num_classes),
                                 (data["data_C"], data["data_H"], data["data_W"],
                                  data["num_classes"]))
                self.assertNotIn("infeasible_geometry", flat)

    def test_old_records_rebuild_as_dvs128(self):
        """Trial records from before the dataset facts were recorded all came
        from DVS128 Gesture."""
        spec = config_to_specs({**SMALL, "N": 4, "epochs": 3, "T": 8, "tau": 2,
                                "resize_to": 64, "trainable_tau": True})
        self.assertEqual((spec["input"].C, spec["input"].H, spec["output"].num_classes),
                         (2, 128, 11))
        self.assertTrue(spec["neuron"].trainable_tau)
        self.assertFalse(spec["neuron"].tau_per_channel)


class BalancedBlockTests(unittest.TestCase):
    def check_block(self, data, max_bits, n):
        points = balanced_points(n, seed=1, t_choices=[8, 16], epochs=80,
                                 data=data, max_input_bits=max_bits, run_seed=1)
        self.assertEqual(len(points) % 6, 0)
        self.assertGreaterEqual(len(points), n)

        def pooling(p):
            pools = any(p.get(f"ds_{i}") == "pool" for i in range(p["depth"]))
            return p.get("pool_type", "avg") if pools else "none"

        def bits(p):
            return "1" if p.get("input_bits", 1) == 1 else "multi"

        per_factor = {
            "pooling": pooling, "bias_mode": lambda p: p["bias_mode"],
            "final_reduction": lambda p: p["final_reduction"],
            "signed_spikes": lambda p: p["signed_spikes"],
            "tau_mode": lambda p: p["tau_mode"], "loss": lambda p: p["loss"],
            "augment": lambda p: p["use_augment"],
            "conv_dropout": lambda p: p["use_conv_dropout"],
            "rate_penalty": lambda p: p["use_rate_penalty"]}
        if max_bits > 1:
            per_factor["input_bits"] = bits
        for name, level_of in per_factor.items():
            counts = Counter(level_of(p) for p in points)
            self.assertEqual(len(counts), len(BALANCED_FACTORS[name]), name)
            self.assertEqual(len(set(counts.values())), 1, (name, counts))
        # pairs of two-level choices land near n/4 each
        twos = [k for k in per_factor if len(BALANCED_FACTORS[k]) == 2]
        for a, b in itertools.combinations(twos, 2):
            pairs = Counter((per_factor[a](p), per_factor[b](p)) for p in points)
            self.assertEqual(len(pairs), 4, (a, b))
            self.assertLessEqual(max(abs(c - len(points) / 4) for c in pairs.values()),
                                 len(points) / 8, (a, b, pairs))
        consts = {"N": 16, "epochs": 80, "data_dir": ".", **data}
        self.assertTrue(all(geometry_fits({**consts, **p}) for p in points))
        return points

    def test_every_level_equally_often_and_every_shape_fits(self):
        points = self.check_block(DVS, 4, 48)
        self.assertTrue(any(p.get("input_bits", 1) > 1 for p in points))

    def test_other_datasets(self):
        self.check_block(CIFAR, 4, 24)
        points = self.check_block(MNIST, 1, 24)
        self.assertFalse(any("input_bits" in p for p in points))

    def test_first_layer_takes_every_choice_the_space_offers(self):
        """The space lets layer 0 downsample by stride, by pooling or not at
        all, so the block tries all three. A first layer at full resolution
        is also the heaviest shape the space can propose, and the preflight's
        memory check measures the block's heaviest trial."""
        trial, _ = sample(True, {**SMALL, "resize_to": 64})
        offered = set(trial.calls["ds_0"][1])
        self.assertIn("none", offered)
        for data in (DVS, CIFAR):
            with self.subTest(data=data["data_H"]):
                points = balanced_points(48, seed=1, t_choices=[16, 24, 32, 48],
                                         data=data, max_input_bits=4)
                self.assertEqual({p["ds_0"] for p in points}, offered)
                for p in points:
                    pools = any(p[f"ds_{i}"] == "pool" for i in range(p["depth"]))
                    self.assertEqual(pools, "pool_type" in p, p)

    def test_every_T_equally_often(self):
        """Search 05 drew T at random in the block and got 15, 33, 15 and 33
        trials, then put every winner at the largest T. A fair T curve needs
        each T equally often, crossed evenly with the other choices."""
        from snnsearch.spaces import block_factors
        self.assertNotIn("T", block_factors([16], 4))
        self.assertEqual(list(block_factors([96, 32, 64, 48], 4)["T"]),
                         ["32", "48", "64", "96"])
        ts = [32, 48, 64, 96]
        points = balanced_points(96, seed=1, t_choices=ts, epochs=80, data=DVS,
                                 max_input_bits=4, run_seed=1)
        self.assertEqual(len(points), 96)
        self.assertEqual(Counter(p["T"] for p in points), {t: 24 for t in ts})
        heads = Counter((p["T"], p["final_reduction"]) for p in points)
        self.assertEqual(len(heads), 8)
        self.assertLessEqual(max(abs(c - 12) for c in heads.values()), 4, heads)
        # one T leaves the block as it was. Five values round 96 up to 120
        self.assertEqual({p["T"] for p in balanced_points(24, seed=1, t_choices=[16],
                                                          data=DVS, max_input_bits=4)}, {16})
        self.assertEqual(len(balanced_points(96, seed=1, t_choices=[16, 24, 32, 48, 64],
                                             data=DVS, max_input_bits=4)), 120)

    def test_points_are_values_the_space_can_take(self):
        """Optuna replays each point through the space, so every value must
        be one the space would suggest."""
        for p in balanced_points(12, seed=3, t_choices=[8, 16], data=CIFAR, max_input_bits=4):
            trial, flat = sample(True, p, data=CIFAR)
            self.assertNotIn("infeasible_geometry", flat)
            for k, v in p.items():
                if k in trial.params:
                    self.assertEqual(trial.params[k], v, k)

    def test_design_rounds_up_and_balances(self):
        import random
        rows = balanced_design({"a": {"x": 0, "y": 0}, "b": {"p": 0, "q": 0, "r": 0}},
                               10, random.Random(0))
        self.assertEqual(len(rows), 12)
        self.assertEqual(Counter(r["b"] for r in rows), {"p": 4, "q": 4, "r": 4})
        self.assertEqual(Counter((r["a"], r["b"]) for r in rows),
                         {(a, b): 2 for a in "xy" for b in "pqr"})


class FinalistTests(unittest.TestCase):
    def test_finalists_from_a_stopped_search_s_records(self):
        """A search stopped before its end still has trials.jsonl. The best
        finished, deployable trials come from there."""
        import json as _json
        import tempfile
        from snnsearch.search import finalists_from_records
        rows = [
            {"trial_id": "a", "status": "complete", "feasible": True, "deployable": True,
             "hw_val_accuracy": 0.70, "val_accuracy": 0.70, "config": {"depth": 2}},
            {"trial_id": "b", "status": "complete", "feasible": True, "deployable": True,
             "hw_val_accuracy": 0.78, "val_accuracy": 0.78, "config": {"depth": 3}},
            {"trial_id": "c", "status": "complete", "feasible": True, "deployable": False,
             "hw_val_accuracy": 0.90, "val_accuracy": 0.0, "config": {}},       # not deployable
            {"trial_id": "d", "status": "complete", "feasible": True, "deployable": None,
             "hw_val_accuracy": None, "val_accuracy": 0.95, "config": {}},      # cut early
            {"trial_id": "e", "status": "complete", "feasible": False, "config": {}},
            {"trial_id": "f", "status": "error", "config": {}},
        ]
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "trials.jsonl"), "w") as fh:
            for r in rows:
                fh.write(_json.dumps(r) + "\n")
        top = finalists_from_records(d, 3)
        self.assertEqual([t[1] for t in top], ["b", "a"])
        self.assertEqual(top[0], (0.78, "b", {"depth": 3}))


    def test_jobs_and_selection_by_mean(self):
        from snnsearch.search import finalist_jobs, summarize_finalists
        top = [(0.90, "a", {"depth": 2, "seed": 1, "_private": 1}),
               (0.88, "b", {"depth": 3, "seed": 1})]
        jobs = finalist_jobs(top, [1001, 2001], epochs=120)
        self.assertEqual(len(jobs), 4)
        self.assertEqual([j["config"]["seed"] for j in jobs], [1001, 2001, 1001, 2001])
        self.assertTrue(all(j["config"]["epochs"] == 120 for j in jobs))
        self.assertNotIn("_private", jobs[0]["config"])
        # the search's top trial got lucky. The second is better on average
        rows = [{"rank": 0, "val_accuracy": 0.80}, {"rank": 0, "val_accuracy": 0.84},
                {"rank": 1, "val_accuracy": 0.86}, {"rank": 1, "val_accuracy": 0.85}]
        summary, best = summarize_finalists(top, rows)
        self.assertEqual(best["search_trial_id"], "b")
        self.assertAlmostEqual(summary[0]["mean_val_accuracy"], 0.82)
        self.assertEqual(best["seeds"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
