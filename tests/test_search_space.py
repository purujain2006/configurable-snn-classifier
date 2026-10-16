"""Search-space and generation regressions, without Optuna, Ray, or a GPU.

    python tests/test_search_space.py
"""
import contextlib
import io
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snnsearch.hardware import check_feasibility
from snnsearch.planning import plan_network
from snnsearch.spaces import config_to_specs, make_define_by_run
from snnsearch.streaming import LEADERBOARD_COLS


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
        self.params[name] = value
        return value


class SearchSpaceTests(unittest.TestCase):
    def sample(self, per_layer, values=None):
        space = make_define_by_run(4, 3, ".", [8, 16], per_layer=per_layer)
        # Ray checkpoints the callable before it samples another trial.
        space = pickle.loads(pickle.dumps(space))
        trial = RecordingTrial(values)
        constants = space(trial)
        self.assertNotIn("final_reduction", constants)
        return trial, {**trial.params, **constants}

    def test_ordered_knobs_and_conditional_layers(self):
        for per_layer in (False, True):
            for resize in (24, 88):
                with self.subTest(per_layer=per_layer, resize=resize):
                    trial, flat = self.sample(per_layer, {"resize_to": resize})
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

        A global average pool has no neurons on chip; it folds into the head's
        synapses, so each head neuron still connects to every neuron of the
        last conv layer, exactly as a flatten head does. The plan's
        fc_in_features (32 for GAP) is the parameter count, and the
        feasibility check must not mistake it for the fan-in.
        """
        for per_layer in (False, True):
            outcomes = {}
            for reduction in ("flatten", "gap"):
                with self.subTest(per_layer=per_layer, reduction=reduction):
                    _, flat = self.sample(per_layer, {
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


class BalancedBlockTests(unittest.TestCase):
    def test_every_level_equally_often_and_every_shape_fits(self):
        """The balanced block gives each option the same number of trials,
        and every one of them gets past the chip check, so none is lost."""
        from collections import Counter
        from snnsearch.spaces import balanced_points, geometry_fits
        points = balanced_points(2, seed=1, t_choices=[8, 16], epochs=80)
        self.assertEqual(len(points), 96)
        def pooling(p):
            pools = any(p.get(f"ds_{i}") == "pool" for i in range(p["depth"]))
            return p.get("pool_type", "avg") if pools else "none"
        self.assertEqual(Counter(pooling(p) for p in points), {"none": 32, "avg": 32, "max": 32})
        for key in ("bias_mode", "final_reduction", "use_conv_dropout", "use_rate_penalty"):
            self.assertEqual(sorted(Counter(p[key] for p in points).values()), [48, 48], key)
        consts = {"N": 16, "epochs": 80, "data_dir": "."}
        self.assertTrue(all(geometry_fits({**consts, **p}) for p in points))


if __name__ == "__main__":
    unittest.main(verbosity=2)
