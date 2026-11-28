"""The option study's variants and verdicts, without training anything.

    python tests/test_study.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snnsearch.spaces import BALANCED_FACTORS, config_to_specs, geometry_fits
from snnsearch.study import (option_variants, apply_option, current_levels,
                             summarize_study, EXTRA_FACTORS, replicates, job_flats)
from snnsearch.search import fold_override

DVS = {"data_C": 2, "data_H": 128, "data_W": 128, "num_classes": 11, "hflip_ok": False}
MIRROR = "0,2,1,6,5,4,3,7,8,-1,10"
# a small network like Search 04's leaders: average pooling, signed, 2 bits
NET = {"N": 16, "epochs": 80, "T": 32, "resize_to": 32, "depth": 2,
       "k_0": 3, "ch_0": 16, "ds_0": "pool", "pool_0": 1,
       "k_1": 3, "ch_1": 24, "ds_1": "stride", "stride_1": 2,
       "fc_layers": 0, "final_reduction": "flatten", "bias_mode": "threshold",
       "signed_spikes": True, "input_bits": 2, "pool_type": "avg", "tau": 4,
       "tau_mode": "layer", "loss": "rate", "use_augment": False,
       "use_conv_dropout": False, "use_rate_penalty": True, "rate_penalty": 0.01,
       "seed": 1, **DVS}


class Variants(unittest.TestCase):
    def test_one_variant_per_other_level_of_every_choice(self):
        for net in (NET, {**NET, "hflip_labels": MIRROR}):
            self._one_variant_per_other_level(net)

    def _one_variant_per_other_level(self, NET):
        vs = option_variants(NET, max_input_bits=4)
        self.assertEqual(vs[0]["name"], "baseline")
        now = current_levels(NET)
        expected = sum(sum(1 for lv in levels if lv != now[o])
                       for o, levels in {**BALANCED_FACTORS, **EXTRA_FACTORS}.items())
        self.assertEqual(len(vs) - 1, expected)
        for v in vs[1:]:
            self.assertNotEqual(v["level"], now[v["option"]], v["name"])
            if v["flat"] is not None:
                # exactly that option changed
                self.assertEqual(current_levels(v["flat"])[v["option"]], v["level"], v["name"])
                others = {k: l for k, l in current_levels(v["flat"]).items() if k != v["option"]}
                self.assertEqual(others, {k: l for k, l in now.items() if k != v["option"]})
                self.assertTrue(geometry_fits(v["flat"]), v["name"])
                config_to_specs(v["flat"])
            else:
                self.assertTrue(v["reason"])

    def test_pooling_changes_keep_the_downsampling(self):
        none = apply_option(NET, "pooling", "none")
        self.assertEqual((none["ds_0"], none["stride_0"]), ("stride", 2))
        self.assertNotIn("pool_0", none)
        self.assertNotIn("pool_type", none)
        mx = apply_option(NET, "pooling", "max")
        self.assertEqual((mx["ds_0"], mx["pool_type"]), ("pool", "max"))
        back = apply_option(none, "pooling", "avg")
        self.assertEqual((back["ds_0"], back["pool_0"], back["pool_type"]), ("pool", 1, "avg"))

    def test_switching_regularizers_on_and_off(self):
        on = apply_option(NET, "augment", "on")
        self.assertTrue(on["use_augment"])
        self.assertEqual((on["aug_shift"], on["aug_cutout"], on["aug_drop"]), (0.1, 0.2, 0.15))
        self.assertNotIn("aug_flip", on)                 # DVS gestures cannot be mirrored
        self.assertTrue(apply_option({**NET, "hflip_ok": True}, "augment", "on")["aug_flip"])
        off = apply_option(NET, "rate_penalty", "off")
        self.assertFalse(off["use_rate_penalty"])
        self.assertNotIn("rate_penalty", off)
        self.assertEqual(config_to_specs(off)["train"].rate_penalty, 0.0)
        self.assertEqual(apply_option(NET, "input_bits", "1")["input_bits"], 1)
        self.assertEqual(apply_option({**NET, "input_bits": 1}, "input_bits", "multi")["input_bits"], 2)

    def test_a_variant_that_does_not_fit_is_skipped_with_the_reason(self):
        # 9x9x56 = 4,536 inputs per neuron fits; signed it is 9,072
        wide = {**NET, "signed_spikes": False, "ds_0": "stride", "stride_0": 2,
                "ch_0": 56, "k_1": 9}
        wide.pop("pool_0")
        wide.pop("pool_type")
        signed = [v for v in option_variants(wide, 4, only=["signed_spikes"])
                  if v["name"] == "signed_spikes=on"][0]
        self.assertIsNone(signed["flat"])
        self.assertIn("fan_in", signed["reason"])

    def test_only(self):
        vs = option_variants(NET, 4, only=["loss", "augment"])
        self.assertEqual([v["name"] for v in vs], ["baseline", "loss=tet", "augment=on"])
        self.assertNotIn("input_bits", [v["option"] for v in option_variants(NET, 1)])
        with self.assertRaises(SystemExit):
            option_variants(NET, 4, only=["timesteps"])     # a typo fails before training

    def test_options_outside_the_search(self):
        """More timesteps, temporal deletion, a label-aware mirror, the
        optimizer and a late rate penalty, each reaching the training spec."""
        net = {**NET, "hflip_labels": MIRROR}
        vs = {v["name"]: v for v in option_variants(net, 4, only=list(EXTRA_FACTORS))}
        self.assertEqual(sorted(vs), sorted(["baseline", "T=48", "T=64", "T=80", "T=96",
                                             "time_drop=on", "mirror=on", "optimizer=adamw",
                                             "optimizer=sgd", "rate_penalty_timing=delayed"]))
        spec = config_to_specs(vs["T=96"]["flat"])
        self.assertEqual(spec["input"].T, 96)
        tr = config_to_specs(vs["time_drop=on"]["flat"])["train"]
        self.assertEqual((tr.augment, tr.aug_time_keep, tr.aug_shift, tr.aug_flip),
                         (True, 0.75, 0.0, False))
        spec = config_to_specs(vs["mirror=on"]["flat"])
        self.assertEqual((spec["train"].augment, spec["train"].aug_flip, spec["train"].aug_time_keep),
                         (True, True, 0.0))
        self.assertEqual(spec["input"].hflip_labels, MIRROR)
        tr = config_to_specs(vs["optimizer=sgd"]["flat"])["train"]
        self.assertEqual((tr.optimizer, tr.lr, tr.momentum), ("sgd", 0.03, 0.9))
        tr = config_to_specs(vs["rate_penalty_timing=delayed"]["flat"])["train"]
        self.assertEqual((tr.rate_penalty, tr.rate_penalty_start), (0.01, 0.5))
        # a dataset with no valid mirror, and no penalty to delay: skipped with a reason
        plain = {v["name"]: v for v in option_variants(
            {**NET, "use_rate_penalty": False}, 4, only=["mirror", "rate_penalty_timing"])}
        self.assertIsNone(plain["mirror=on"]["flat"])
        self.assertIsNone(plain["rate_penalty_timing=delayed"]["flat"])
        self.assertTrue(plain["mirror=on"]["reason"])

    def test_folds_as_replicates(self):
        """--folds 5: every variant trains once on each subject fold with one
        seed; the fold travels in the job's flat config, and the trial moves it
        into the run config without touching the shared one."""
        self.assertEqual(replicates(1, seeds=3), [(1001, None), (2001, None), (3001, None)])
        reps = replicates(1, folds=5)
        self.assertEqual(reps, [(1001, f) for f in range(5)])
        trained = [{"name": "baseline", "flat": dict(NET)}, {"name": "T=64", "flat": {**NET, "T": 64}}]
        jobs, flats = job_flats(trained, reps, folds=5)
        self.assertEqual(len(flats), 10)
        self.assertEqual([(f["_val_folds"], f["_val_fold"], f["seed"]) for f in flats[:5]],
                         [(5, f, 1001) for f in range(5)])
        run_cfg = {"search": {"val_split": "subject"}, "run": {"seed": 1}}
        trial, job_cfg = fold_override(run_cfg, flats[3])
        self.assertNotIn("_val_fold", trial)
        self.assertEqual(trial["T"], NET["T"])
        self.assertEqual(job_cfg["search"], {"val_split": "subject", "val_folds": 5, "val_fold": 3})
        self.assertEqual(run_cfg["search"], {"val_split": "subject"})      # untouched
        same, cfg = fold_override(run_cfg, {"T": 48})
        self.assertIs(cfg, run_cfg)
        _seed_jobs, seed_flats = job_flats(trained, replicates(1, seeds=2))
        self.assertTrue(all("_val_fold" not in f for f in seed_flats))


class Verdicts(unittest.TestCase):
    def test_paired_by_seed(self):
        variants = [{"name": "baseline", "reason": ""}, {"name": "a", "reason": ""},
                    {"name": "b", "reason": ""}, {"name": "c", "reason": "does not fit"}]
        rows = []
        for seed, base in ((1, 0.70), (2, 0.80), (3, 0.75)):
            rows.append({"variant": "baseline", "seed": seed, "val_accuracy": base})
            rows.append({"variant": "a", "seed": seed, "val_accuracy": base + 0.05})   # helps
            rows.append({"variant": "b", "seed": seed,
                         "val_accuracy": base + (0.02 if seed == 2 else -0.02)})     # noise
        out = {r["variant"]: r for r in summarize_study(variants, rows)}
        self.assertEqual(out["a"]["verdict"], "helps")
        self.assertAlmostEqual(out["a"]["diff_vs_baseline"], 0.05)
        self.assertEqual(out["b"]["verdict"], "no clear difference")
        self.assertEqual(out["c"]["verdict"], "does not fit")
        self.assertIsNone(out["c"]["mean_val_accuracy"])
        # a spread of 0.05 across seeds would have hidden the +0.05 effect
        # without pairing. Pairing cancels it
        self.assertEqual(out["a"]["diff_se"], 0.0)

    def test_paired_by_fold(self):
        """Over folds, the seed is the same everywhere and the fold pairs the
        runs. The worst fold and mean - 0.5 * spread are reported too."""
        variants = [{"name": "baseline", "reason": ""}, {"name": "a", "reason": ""}]
        rows = []
        for fold, base in enumerate((0.70, 0.90, 0.80)):
            rows.append({"variant": "baseline", "seed": 1001, "fold": fold, "val_accuracy": base})
            rows.append({"variant": "a", "seed": 1001, "fold": fold, "val_accuracy": base + 0.03})
        out = {r["variant"]: r for r in summarize_study(variants, rows)}
        self.assertEqual(out["a"]["verdict"], "helps")
        self.assertAlmostEqual(out["a"]["diff_vs_baseline"], 0.03)
        self.assertEqual(out["a"]["runs"], 3)
        self.assertAlmostEqual(out["a"]["worst_val_accuracy"], 0.73)
        self.assertAlmostEqual(out["baseline"]["score"], 0.80 - 0.5 * 0.10, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
