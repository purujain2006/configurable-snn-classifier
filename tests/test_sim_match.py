"""The PyTorch model and IntegerSim fire the same spikes, for every feature.

    python tests/test_sim_match.py

Training uses the PyTorch model; deployment is judged on IntegerSim, which
`tools/match_simplesim.py --selftest` proves equal to hs_api's simpleSim. This
closes the loop: small networks combining every feature the search can pick
(average or OR pooling, flatten or GAP, threshold or bias-neuron biases,
signed spikes, per-channel tau, multi-bit input) must produce identical spikes
in every layer at every timestep, in float32 as training runs.

Thresholds are set low after export so every layer fires; a match on silent
layers would prove nothing.
"""
import itertools
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from snnsearch.config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                              HeadSpec, NeuronSpec, TrainSpec)
from snnsearch.model import build_model
from snnsearch.neuron import HardwareLIFNode, OrPool
from snnsearch.hs_export import integer_layers, IntegerSim
from snnsearch.train import hardware_flush_steps

torch.backends.cudnn.allow_tf32 = False


def build(pool_type, head, bias_mode, signed, per_channel, bits, seed):
    torch.manual_seed(seed)
    layers = [{"out_channels": 6, "kernel_size": 3, "stride": 1, "pool": True},
              {"out_channels": 5, "kernel_size": 5, "stride": 1, "pool": True}]
    spec = {"input": InputSpec(C=2, H=16, W=16, T=6, resize_to=16, N=2, input_bits=bits),
            "encoder": EncoderSpec(layers_json=json.dumps(layers), norm="bn"),
            "output": OutputSpec(num_classes=4),
            "downsample": DownsampleSpec(mode="stride", pool_type=pool_type),
            "head": HeadSpec(final_reduction=head, fc_widths="", dropout_rate=0.0),
            "neuron": NeuronSpec(tau=4, trainable_tau=True, tau_per_channel=per_channel,
                                 signed_spikes=signed),
            "train": TrainSpec(fold_bias_mode=bias_mode)}
    net = build_model(spec).eval()
    net.to_qat_folded(bias_mode=bias_mode)
    for m in net.modules():
        if hasattr(m, "bn"):
            m.bn.running_mean.uniform_(-0.2, 0.2)
            m.bn.running_var.uniform_(0.5, 1.5)
            m.bn.bias.data.uniform_(-0.1, 0.1)
            m.conv.weight.data.mul_(4)
    for m in net.conv_fc:
        if isinstance(m, torch.nn.Linear):
            m.parametrizations.weight.original.data.mul_(6)
        if isinstance(m, HardwareLIFNode) and per_channel:
            m.raw_log2_tau.data.copy_(torch.randint(1, 4, m.raw_log2_tau.shape).float())
    net = net.export_deployed().eval()
    # low thresholds so every layer fires
    for m in net.conv_fc:
        if isinstance(m, HardwareLIFNode):
            m.learn_threshold = False
            m._v_threshold_const = float(torch.empty(1).uniform_(0.01, 0.05))
            if m.signed:
                m._v_threshold_neg_const = float(torch.empty(1).uniform_(0.01, 0.05))
    return net


def compare(net, bits, clips=3, T=6):
    from spikingjelly.activation_based import functional
    flush = hardware_flush_steps(net)
    sim = IntegerSim(integer_layers(net), (2, 16, 16))
    mods = [m for m in net.conv_fc if isinstance(m, (HardwareLIFNode, OrPool))]
    mismatches, fired = 0, [0] * len(mods)
    for _ in range(clips):
        top = 2 ** bits - 1
        x = ((torch.rand(T, 2, 16, 16) < 0.3) * torch.randint(1, top + 1, (T, 2, 16, 16))).float()
        rec = [[] for _ in mods]
        hooks = [m.register_forward_hook(lambda mod, i, o, k=k: rec[k].append(o.detach().clone()))
                 for k, m in enumerate(mods)]
        with torch.no_grad():
            for t in range(T + flush):
                net(x[t].unsqueeze(0) if t < T else torch.zeros(1, 2, 16, 16))
        for h in hooks:
            h.remove()
        functional.reset_net(net)
        for t, (_groups, outputs) in enumerate(sim.steps(x.double(), flush)):
            for li in range(len(mods)):
                a = rec[li][t].reshape(-1).double()
                b = outputs[li].reshape(-1)
                mismatches += int((a != b).sum())
                fired[li] += int(b.abs().sum())
    return mismatches, fired


class SimMatch(unittest.TestCase):
    def test_every_feature_combination_matches_spike_for_spike(self):
        combos = list(itertools.product(("avg", "max"), ("flatten", "gap"),
                                        ("threshold", "neuron"), (False, True)))
        extra = [("avg", "flatten", "threshold", False, True, 3),     # per-channel tau, 3-bit input
                 ("max", "gap", "neuron", True, True, 2)]             # everything at once
        cases = [(p, h, b, s, False, 1) for p, h, b, s in combos] + extra
        for i, (pool, head, bias, signed, per_channel, bits) in enumerate(cases):
            with self.subTest(pool=pool, head=head, bias=bias, signed=signed,
                              per_channel=per_channel, bits=bits):
                net = build(pool, head, bias, signed, per_channel, bits, seed=i)
                mismatches, fired = compare(net, bits)
                self.assertTrue(all(f > 0 for f in fired), f"a layer never fired: {fired}")
                self.assertEqual(mismatches, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
