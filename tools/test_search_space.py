"""Search-space and generation regressions, without Optuna, Ray, or a GPU.

    python tools/test_search_space.py
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

import tools.Misc.build_from_practice2 as generator
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

    def test_gap_makes_high_resolution_head_feasible(self):
        for per_layer in (False, True):
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
                    if reduction == "gap":
                        self.assertEqual(plan.fc_in_features, 32)
                        self.assertTrue(feasible, violations)
                    else:
                        self.assertEqual(plan.fc_in_features, 32 * 21 * 21)
                        self.assertFalse(feasible)
                        self.assertTrue(any("fc0: neuron_fan_in" in reason
                                            for reason in violations))
        self.assertIn("final_reduction", LEADERBOARD_COLS)


class GenerationTests(unittest.TestCase):
    def test_generated_modules_match_checked_in_copies(self):
        source = ROOT / "Practice2.py"
        lines = generator.read(source)
        decorators = generator.decorator_starts(source)
        names = ("config.py", "neuron.py", "model.py", "folding.py",
                 "quantize.py", "train.py", "spaces.py")
        # Never regenerate over the workspace: other modules are hand-edited.
        with tempfile.TemporaryDirectory(prefix="snnsearch-generation-") as tmp:
            with patch.object(generator, "PKG", tmp):
                for name in names:
                    with self.subTest(module=name), contextlib.redirect_stdout(io.StringIO()):
                        doc, imports, spans = generator.PLAN[name]
                        generator.write_module(name, doc, imports,
                                               generator.snap(spans, decorators), lines)
                        self.assertEqual(
                            Path(tmp, name).read_text(encoding="utf-8"),
                            (ROOT / "snnsearch" / name).read_text(encoding="utf-8"))

    def test_patch_rejects_missing_and_ambiguous_matches(self):
        with patch.dict(generator.PATCHES, {"test.py": [("target", "replacement")]}):
            for text, hits in (("absent", 0), ("target target", 2)):
                with self.subTest(hits=hits):
                    with self.assertRaisesRegex(SystemExit, f"matched {hits} times"):
                        generator.apply_patches("test.py", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
