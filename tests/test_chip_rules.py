"""The rules a network must follow to run on HiAER-Spike, each as one test.

    python tests/test_chip_rules.py

Each rule below was violated by a real run before it was enforced:
  * tau is a power of two (rank1 trained to tau 3)
  * weight decay does not move tau (it drove tau 8 -> 3 -> 0 as decay rose)
  * fan-in counts the neurons in front of an average pool (rank1's classifier
    needs 32,256 inputs per neuron; the limit is 8,159)
  * a bias delivered by bias neurons splits exactly across INT16 synapses and
    the 4095 fan-out limit

The simulator itself is checked against the real hs_api simpleSim by
`python tools/match_simplesim.py --selftest`, which needs hs_api installed.
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snnsearch.config import (InputSpec, EncoderSpec, DownsampleSpec, HeadSpec,
                              OutputSpec)
from snnsearch.hardware import check_feasibility, HW_TAU_CHOICES, INT16_MAX


def encoder(*layers):
    return EncoderSpec(layers_json=json.dumps([
        {"out_channels": c, "kernel_size": k, "stride": s, "pool": p}
        for c, k, s, p in layers]))


def feasible(enc, reduction="flatten", resize=48, bias_neurons=False):
    return check_feasibility(InputSpec(C=2, H=128, W=128, T=16, resize_to=resize), enc,
                             DownsampleSpec(mode="stride", pool_type="avg"),
                             HeadSpec(final_reduction=reduction, fc_widths=""),
                             OutputSpec(num_classes=11), bias_neurons=bias_neurons)


class FanIn(unittest.TestCase):
    def test_rank1_is_rejected_because_pooling_quadruples_fan_in(self):
        rank1 = encoder((81, 3, 1, True), (56, 5, 1, True))
        ok, why = feasible(rank1)
        self.assertFalse(ok)
        self.assertEqual(why, ["classifier: neuron_fan_in 32256 > 8159"])

    def test_gap_does_not_reduce_fan_in(self):
        rank1 = encoder((81, 3, 1, True), (56, 5, 1, True))
        self.assertEqual(feasible(rank1, "gap")[1], feasible(rank1, "flatten")[1])

    def test_strided_conv_does_not_multiply_fan_in(self):
        # b71e0bdc from Search 02: 0.9034 val, and it fits the chip
        self.assertEqual(feasible(encoder((79, 3, 1, True), (71, 5, 2, False))), (True, []))

    def test_conv_behind_a_pool_counts_four_inputs_per_pooled_value(self):
        # 5x5 kernel over 82 pooled channels: 25 * 82 * 4 = 8200 > 8159
        ok, why = feasible(encoder((82, 3, 1, True), (8, 5, 2, False)))
        self.assertFalse(ok)
        self.assertIn("block1: neuron_fan_in 8200 > 8159", why)

    def test_max_pool_runs_as_or_neurons_without_fan_in_cost(self):
        # rank1's shape with max pooling: one OR neuron per window, so the
        # classifier reads the pooled 56 x 12 x 12 = 8,064 values and fits
        rank1 = encoder((81, 3, 1, True), (56, 5, 1, True))
        ok, why = check_feasibility(InputSpec(C=2, H=128, W=128, T=16, resize_to=48), rank1,
                                    DownsampleSpec(mode="stride", pool_type="max"),
                                    HeadSpec(final_reduction="flatten", fc_widths=""),
                                    OutputSpec(num_classes=11))
        self.assertEqual((ok, why), (True, []))

    def test_bias_neurons_take_fan_in(self):
        # 81 channels: 25 * 81 * 4 = 8100 fits, plus up to 4 bias neurons still fits
        enc = encoder((81, 3, 1, True), (8, 5, 2, False))
        self.assertTrue(feasible(enc)[0])
        self.assertTrue(feasible(enc, bias_neurons=True)[0])


class Tau(unittest.TestCase):
    def test_search_only_offers_powers_of_two(self):
        self.assertTrue(all(t & (t - 1) == 0 for t in HW_TAU_CHOICES))

    def test_trainable_tau_can_only_land_on_powers_of_two(self):
        import torch
        from snnsearch.neuron import HardwareLIFNode
        node = HardwareLIFNode(tau=8, learn_tau=True)
        reached = set()
        for v in torch.linspace(-2.0, 10.0, 241):
            node.raw_log2_tau.data.fill_(float(v))
            reached.add(node.hw_leak())
        self.assertEqual(sorted(reached), [2, 4, 8, 16, 32, 64, 128])

    def test_weight_decay_leaves_tau_and_threshold_alone(self):
        import torch
        from snnsearch.neuron import HardwareLIFNode
        from snnsearch.train import param_groups
        net = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3), torch.nn.BatchNorm2d(4),
                                  HardwareLIFNode(tau=8, learn_tau=True, learn_threshold=True))
        decayed, kept = param_groups(net, 1e-2)
        self.assertEqual(len(decayed["params"]), 1)            # the conv weight only
        opt = torch.optim.Adam(param_groups(net, 1e-2), lr=1e-2)
        before = float(net[2].raw_log2_tau)
        for _ in range(200):
            opt.zero_grad()
            for p in net.parameters():
                p.grad = torch.zeros_like(p)
            opt.step()
        self.assertEqual(float(net[2].raw_log2_tau), before)


class BiasNeurons(unittest.TestCase):
    def test_split_is_exact_and_within_limits(self):
        from snnsearch.hs_export import split_bias
        for value, n in [(5, 10), (-32767, 100), (100000, 576), (-131068, 4096), (40000, 9000)]:
            parts = split_bias(value, n)
            self.assertTrue(all(abs(w) <= INT16_MAX for w, _ in parts))
            self.assertTrue(all(sl.stop - sl.start <= 4095 for _, sl in parts))
            # every target neuron receives the whole bias, exactly once
            per_target = [0] * n
            for w, sl in parts:
                for t in range(sl.start, sl.stop):
                    per_target[t] += w
            self.assertEqual(set(per_target), {value})


if __name__ == "__main__":
    unittest.main(verbosity=2)
