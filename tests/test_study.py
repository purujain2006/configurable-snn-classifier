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
                             summarize_study)

DVS = {"data_C": 2, "data_H": 128, "data_W": 128, "num_classes": 11, "hflip_ok": False}
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
        vs = option_variants(NET, max_input_bits=4)
        self.assertEqual(vs[0]["name"], "baseline")
        expected = sum(len(levels) - 1 for levels in BALANCED_FACTORS.values())
        self.assertEqual(len(vs) - 1, expected)
        now = current_levels(NET)
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
        # without pairing; pairing cancels it
        self.assertEqual(out["a"]["diff_se"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
