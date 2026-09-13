"""Regression checks for sampled inputs, replay overrides and search ordering.

    python tools/test_input_pipeline.py

No DVS data, GPU, Ray or Optuna needed. Tensor checks require the training stack;
the config and orchestration checks also run without it.
"""
import copy
import io
import json
import os
import pickle
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snnsearch import cli, pipeline, runconfig, search
from snnsearch._torch import _HAS_TORCH, _HAS_SPIKINGJELLY
from snnsearch.data.base import DatasetBundle


def trial_config(**overrides):
    return dict(depth=2, channels=8, kernel_size=3, stride=2, tau=2,
                T=8, resize_to=32, N=99, epochs=3, fc_layers=0,
                final_reduction="gap", **overrides)


def run_config():
    return runconfig.load(overrides={
        "dataset": {"name": "dvs128", "root": ".", "T": 99},
        "encoding": {"T": 16, "resize_to": 64},
        "search": {"batch_size": 4, "num_workers": 0, "gpu_fraction": 0.125},
        "report": {"html": False},
    })


class InputPipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = run_config()
        self.bundle = DatasetBundle([], [], C=3, H=48, W=40, num_classes=7,
                                    is_event=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(pipeline, "_require_torch"))
        self.dataset = self.stack.enter_context(
            patch.object(pipeline, "build_dataset", return_value=self.bundle))

    def test_sampled_inputs_reach_dataset_encoder_spec_and_loader(self):
        flat = trial_config()
        before = copy.deepcopy((self.cfg, flat))
        output = io.StringIO()
        with redirect_stdout(output), patch.object(pipeline, "build_dataloaders",
                                                  return_value=([], [], [])) as build:
            bundle, encoder, spec = pipeline.resolve_specs(self.cfg, flat, quiet=True)
            pipeline.make_loaders(self.cfg, bundle, encoder, spec, quiet=True)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(self.dataset.call_args.args[0]["T"], 8)
        self.assertEqual((encoder.T, encoder.size), (8, (32, 32)))
        self.assertEqual(asdict(spec["input"]),
                         dict(N=4, C=3, H=48, W=40, T=8, resize_to=32))
        self.assertEqual(spec["output"].num_classes, 7)
        self.assertEqual(spec["train"].epochs, 3)
        build.assert_called_once_with(bundle, batch_size=4, encoder=encoder,
                                      num_workers=0, seed=1)
        self.assertEqual((self.cfg, flat), before)

    def test_missing_or_null_input_knobs_inherit_config(self):
        for null in (False, True):
            for keys in (("T",), ("resize_to",), ("T", "resize_to")):
                with self.subTest(null=null, keys=keys):
                    flat = trial_config()
                    for key in keys:
                        if null:
                            flat[key] = None
                        else:
                            flat.pop(key)
                    _, encoder, spec = pipeline.resolve_specs(self.cfg, flat, quiet=True)
                    T = 16 if "T" in keys else 8
                    size = 64 if "resize_to" in keys else 32
                    self.assertEqual((spec["input"].T, spec["input"].resize_to), (T, size))
                    self.assertEqual((encoder.T, encoder.size), (T, (size, size)))
                    self.assertEqual(self.dataset.call_args.args[0]["T"], T)
                    search._assert_records_what_ran(flat, spec, encoder)

    def test_zero_resize_preserves_native_shape(self):
        flat = trial_config()
        flat["resize_to"] = 0
        _, encoder, spec = pipeline.resolve_specs(self.cfg, flat, quiet=True)
        self.assertEqual(spec["input"].resize_to, 0)
        self.assertIsNone(encoder.size)

    def test_prepare_keeps_four_value_contract(self):
        with redirect_stdout(io.StringIO()), patch.object(
                pipeline, "build_dataloaders", return_value=([], [], [])):
            bundle, encoder, loaders, spec = pipeline.prepare(self.cfg)
        self.assertIs(bundle, self.bundle)
        self.assertEqual(len(loaders), 3)
        self.assertEqual((encoder.T, spec["input"].T), (16, 16))

    def test_record_guard_detects_mismatch_and_is_picklable(self):
        _, encoder, spec = pipeline.resolve_specs(self.cfg, trial_config(), quiet=True)
        self.assertIs(pickle.loads(pickle.dumps(search._assert_records_what_ran)),
                      search._assert_records_what_ran)
        search._assert_records_what_ran({}, spec, encoder)
        for key, value in (("T", 16), ("resize_to", 64)):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                search._assert_records_what_ran({key: value}, spec, encoder)
        encoder.T = 2
        with self.assertRaisesRegex(RuntimeError, "encoder has 2"):
            search._assert_records_what_ran(trial_config(), spec, encoder)

    def test_trial_checks_final_input_before_building_loaders(self):
        ray = ModuleType("ray")
        ray.tune = SimpleNamespace(report=Mock())
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"ray": ray, "torch": fake_torch}), \
                patch("snnsearch.hardware.check_feasibility", return_value=(False, ["limit"])) as check, \
                patch.object(pipeline, "make_loaders") as loaders, \
                patch("snnsearch.train.run_training") as train:
            search._make_trainable(self.cfg, ".")(trial_config())
        self.assertEqual((check.call_args.args[0].T, check.call_args.args[0].resize_to),
                         (8, 32))
        loaders.assert_not_called()
        train.assert_not_called()
        self.assertFalse(ray.tune.report.call_args.args[0]["feasible"])

    def test_cli_replay_flags_override_record_and_summary(self):
        self.bundle.C, self.bundle.H, self.bundle.W, self.bundle.num_classes = 2, 128, 128, 11
        with tempfile.TemporaryDirectory(prefix="snnsearch_inputs_") as out:
            path = os.path.join(out, "best.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"config": trial_config()}, fh)
            for flags, expected in (([], (8, 32)),
                                    (["--T", "16", "--resize-to", "64"], (16, 64)),
                                    (["--resize-to", "0"], (8, 0))):
                with self.subTest(flags=flags), redirect_stdout(io.StringIO()) as output, \
                        patch("snnsearch._torch._require_torch"), \
                        patch.object(pipeline, "build_dataloaders", return_value=([], [], [])), \
                        patch("snnsearch.train.run_training", return_value={}) as train, \
                        patch.object(pipeline, "_maybe_report"):
                    args = ["-c", os.path.join(ROOT, "configs", "dvs128.yaml"),
                            "--from-best", path, "--results-dir", out] + flags
                    self.assertEqual(cli.main(["single"] + args), 0)
                    spec = train.call_args.args[0]
                    self.assertEqual((spec["input"].T, spec["input"].resize_to), expected)
                    if flags:
                        self.assertIn("from the command line", output.getvalue())
                    with patch("snnsearch.cost.format_summary", return_value="summary") as summary:
                        self.assertEqual(cli.main(["summary"] + args), 0)
                    self.assertEqual(summary.call_args.args[0], spec)
            self.assertEqual(pipeline.load_flat_config(path), trial_config())


class CacheOrderingTests(unittest.TestCase):
    def test_cache_warmup_is_dvs_only(self):
        writer = SimpleNamespace(log=Mock())
        cfg = run_config()
        with patch("snnsearch.data.builtin.warmup_frame_cache") as warm:
            search._warm_frame_caches(cfg, [8, 16], writer)
            warm.assert_called_once_with(".", [8, 16], log=writer.log)
            warm.reset_mock()
            cfg["dataset"]["name"] = "cifar10"
            search._warm_frame_caches(cfg, [8, 16], writer)
            warm.assert_not_called()
            # Loading a custom module keeps the default name='dvs128', but
            # build_dataset gives the module precedence over that name.
            cfg["dataset"]["name"] = "dvs128"
            cfg["dataset"]["module"] = "examples/synthetic_data.py"
            search._warm_frame_caches(cfg, [8, 16], writer)
            warm.assert_not_called()

    def test_warmup_precedes_ray_and_uses_same_choices_as_space(self):
        modules = {name: ModuleType(name) for name in (
            "ray", "ray.tune", "ray.tune.schedulers", "ray.tune.search",
            "ray.tune.search.optuna")}
        ray = modules["ray"]
        ray.tune = modules["ray.tune"]
        ray.is_initialized = lambda: False
        events = []
        ray.init = lambda **kw: events.append("ray")
        modules["ray.tune.schedulers"].ASHAScheduler = Mock()
        modules["ray.tune.search.optuna"].OptunaSearch = Mock()
        for choices in (None, [8, 16]):
            cfg = run_config()
            cfg["search"]["T_choices"] = choices
            events.clear()
            def warm(_cfg, values, _writer):
                events.append(("warm", values))
            def space(**kwargs):
                events.append(("space", kwargs["t_choices"]))
                raise RuntimeError("stop before scheduling")
            with self.subTest(choices=choices), patch.dict(sys.modules, modules), \
                    patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
                    patch("snnsearch.results.ResultsWriter"), \
                    patch.object(search, "_ray_temp_dir", return_value=None), \
                    patch.object(search, "_warm_frame_caches", side_effect=warm), \
                    patch("snnsearch.spaces.make_define_by_run", side_effect=space), \
                    self.assertRaisesRegex(RuntimeError, "stop before scheduling"):
                search.run_search(cfg, ".")
            self.assertEqual([e[0] if isinstance(e, tuple) else e for e in events],
                             ["warm", "ray", "space"])
            self.assertIs(events[0][1], events[2][1])
            self.assertEqual(events[0][1], choices or [16])


@unittest.skipUnless(_HAS_TORCH and _HAS_SPIKINGJELLY, "needs torch and spikingjelly")
class TensorPipelineTests(unittest.TestCase):
    def test_actual_encoded_batch_has_trial_time_and_resolution(self):
        cfg = run_config()
        cfg["dataset"] = {"module": os.path.join(ROOT, "examples", "synthetic_data.py")}
        cfg["encoding"]["coding"] = "direct"
        with redirect_stdout(io.StringIO()):
            bundle, encoder, loaders, spec = pipeline.prepare(cfg, trial_config())
        x, _, lengths = next(iter(loaders[0]))
        self.assertEqual(tuple(x.shape), (4, 8, 2, 32, 32))
        self.assertTrue((lengths == 8).all())
        self.assertEqual((spec["input"].C, spec["output"].num_classes),
                         (bundle.C, bundle.num_classes))
        search._assert_records_what_ran(trial_config(), spec, encoder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
