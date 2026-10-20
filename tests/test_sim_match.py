"""The PyTorch model and IntegerSim fire the same spikes, for every feature.

    python tests/test_sim_match.py

Training uses the PyTorch model; deployment is judged on IntegerSim, which
`tools/match_simplesim.py --selftest` proves equal to hs_api's simpleSim. This
closes the loop: small networks combining every feature the search can pick
(average or OR pooling, flatten or GAP, threshold or bias-neuron biases,
signed spikes, per-channel tau, multi-bit input) must produce identical spikes
in every layer at every timestep, in float32 as training runs, and in the
float64 copy (hs_export.exact_copy) that deployed accuracy is measured on.

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

import numpy as np
import torch

from snnsearch.config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                              HeadSpec, NeuronSpec, TrainSpec)
from snnsearch.model import build_model
from snnsearch.neuron import HardwareLIFNode, OrPool
from snnsearch.hs_export import integer_layers, IntegerSim, exact_copy
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
    dt = next(net.parameters()).dtype
    mismatches, fired = 0, [0] * len(mods)
    for _ in range(clips):
        top = 2 ** bits - 1
        x = ((torch.rand(T, 2, 16, 16) < 0.3) * torch.randint(1, top + 1, (T, 2, 16, 16))).float()
        rec = [[] for _ in mods]
        hooks = [m.register_forward_hook(lambda mod, i, o, k=k: rec[k].append(o.detach().clone()))
                 for k, m in enumerate(mods)]
        with torch.no_grad():
            for t in range(T + flush):
                net((x[t].unsqueeze(0) if t < T else torch.zeros(1, 2, 16, 16)).to(dt))
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
                # the float64 copy every measurement of a deployed network uses
                torch.manual_seed(100 + i)
                mismatches, _ = compare(exact_copy(net), bits)
                self.assertEqual(mismatches, 0, "exact_copy")


class FpgaTarget(unittest.TestCase):
    """tools/match_simplesim.py --hs-target CRI runs clips on the FPGA through
    hs_api. Without hardware, stand-in hs_api and hs_bridge modules check the
    plumbing: the core reaches CRI_network, the chip is cleared before every
    clip, step()'s (spikes, latency, hbm_access) is read, and output spikes
    become predictions."""

    def run_stub(self, target):
        import importlib.util
        import types
        from unittest import mock
        spec = importlib.util.spec_from_file_location(
            "match_simplesim", os.path.join(ROOT, "tools", "match_simplesim.py"))
        ms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ms)
        calls = {"clear": [], "kwargs": None, "inits": 0}

        class FakeNet:
            def __init__(self, axons, connections, outputs, target, **kw):
                calls["kwargs"] = {"target": target, **kw}
                self.outputs = outputs
                self.n_cls = len(outputs) // (2 if any("-" in str(k) for k in outputs) else 1)
                self.simpleSim = types.SimpleNamespace(
                    numNeurons=len(connections),
                    initialize_sim_vars=lambda n: calls.__setitem__("inits", calls["inits"] + 1))

            def step(self, inputs):
                spikes = [self.outputs[2]]       # class 2 fires every step
                calls["counter"] = calls.get("counter", 0) + 7     # runs on, like the FPGA's
                return (spikes, calls["counter"], 0) if target == "CRI" else spikes

        api = types.ModuleType("hs_api.api")
        api.CRI_network = FakeNet
        models = types.ModuleType("hs_api.neuron_models")
        models.LIF_neuron = models.ANN_neuron = lambda *a: ("neuron", a)
        ctrl = types.ModuleType("hs_bridge.FPGA_Execution.fpga_controller")
        ctrl.clear = lambda n, dump, core: calls["clear"].append((n, dump, core))
        fe = types.ModuleType("hs_bridge.FPGA_Execution")
        fe.fpga_controller = ctrl
        mods = {"hs_api": types.ModuleType("hs_api"), "hs_api.api": api,
                "hs_api.neuron_models": models,
                "hs_bridge": types.ModuleType("hs_bridge"),
                "hs_bridge.FPGA_Execution": fe,
                "hs_bridge.FPGA_Execution.fpga_controller": ctrl}
        net = build("avg", "flatten", "threshold", False, False, 1, seed=0)
        layers = integer_layers(net)
        clips = [(torch.zeros(4, 2, 16, 16, dtype=torch.uint8), 0) for _ in range(3)]
        with mock.patch.dict(sys.modules, mods):
            out = ms._real_simplesim(layers, (2, 16, 16), clips, flush=2,
                                     target=target, core=5)
        return out, calls

    def test_fpga_path(self):
        out, calls = self.run_stub("CRI")
        self.assertEqual(calls["kwargs"], {"target": "CRI", "coreID": 5})
        self.assertEqual(len(calls["clear"]), 3)
        self.assertTrue(all(core == 5 and not dump for _n, dump, core in calls["clear"]))
        self.assertEqual([p for p, _, _ in out], [2, 2, 2])
        # the counter's increase within a clip, not its running value
        self.assertEqual([lat for _, _, lat in out], [7.0, 7.0, 7.0])
        self.assertEqual(len(out[0][1]), 4 + 2)          # T steps plus the flush

    def test_simplesim_path_is_unchanged(self):
        out, calls = self.run_stub("simpleSim")
        self.assertEqual(calls["kwargs"], {"target": "simpleSim"})
        self.assertEqual((calls["clear"], calls["inits"]), ([], 3))
        self.assertEqual([(p, lat) for p, _, lat in out], [(2, None)] * 3)


def planted_bug_hs_api(spike_cap=1000):
    """Stand-in hs_api and hs_bridge modules: a small LIF simulator whose
    "simpleSim" target is correct and whose "CRI" target has two planted
    bugs, a logical-shift leak for negative membranes and a cap on spikes per
    step. tools/fpga_diff.py has to find both."""
    import types
    WRAP = 2 ** 36
    state = {}

    class Model:
        def __init__(self, kind, theta, _nu, leak):
            self.kind, self.theta, self.leak = kind, int(theta), int(leak)

    class FakeNet:
        def __init__(self, axons, connections, outputs, target, **kw):
            self.axons, self.conns, self.outputs = axons, connections, set(outputs)
            self.target = target
            if target == "CRI":
                # as on crisdsc0: hs_api pads each model object's neurons to a
                # multiple of 32, and hs_bridge sends the total in 17 bits
                per_model = {}
                for _syn, m in connections.values():
                    per_model[id(m)] = per_model.get(id(m), 0) + 1
                padded = sum(-(-n // 32) * 32 for n in per_model.values())
                if padded >= 2 ** 17:
                    raise ValueError(f"Insufficient bit width=17: {padded} neurons")
            self.reset()
            self.simpleSim = types.SimpleNamespace(
                numNeurons=len(connections), initialize_sim_vars=lambda n: self.reset())
            state["net"] = self

        def reset(self):
            self.v = {k: 0 for k in self.conns}
            self.counter = 0
            self.fired = []          # every neuron that fired, per step, before the cap
            self.events = 0          # synapses applied

        def leak(self, v, s):
            if self.target == "CRI" and v < 0:
                u = (v - ((v % WRAP) >> s)) % WRAP
                return u - WRAP if u >= WRAP // 2 else u
            return v - (v >> s)

        def step(self, inputs, membranePotential=False):
            # simpleSim's order: phase one fires on the membrane carried in,
            # resets what fired and leaks the rest (memoryless neurons reset);
            # phase two adds the input axons and the spikes just fired
            spikes = []
            for k, (_syn, m) in self.conns.items():
                v = self.v[k]
                if v > m.theta:
                    spikes.append(k)
                    v = 0
                self.v[k] = 0 if m.kind == "ANN" else self.leak(v, m.leak)
            self.fired.append(set(spikes))
            if self.target == "CRI":
                spikes = spikes[:spike_cap]
            for a in inputs:
                self.events += len(self.axons[a])
                for post, w in self.axons[a]:
                    self.v[post] += w
            for n in spikes:
                self.events += len(self.conns[n][0])
                for post, w in self.conns[n][0]:
                    self.v[post] += w
            self.counter += 100
            out = [k for k in spikes if k in self.outputs]
            mem = list(self.v.items())
            if self.target == "CRI":
                return mem, (out, self.counter, 0)
            return mem, out

    api = types.ModuleType("hs_api.api")
    api.CRI_network = FakeNet
    api.last_net = lambda: state["net"]
    models = types.ModuleType("hs_api.neuron_models")
    models.LIF_neuron = lambda *a: Model("LIF", *a)
    models.ANN_neuron = lambda *a: Model("ANN", *a)
    ctrl = types.ModuleType("hs_bridge.FPGA_Execution.fpga_controller")
    ctrl.clear = lambda n, dump, core: state["net"].reset()
    fe = types.ModuleType("hs_bridge.FPGA_Execution")
    fe.fpga_controller = ctrl
    return {"hs_api": types.ModuleType("hs_api"), "hs_api.api": api,
            "hs_api.neuron_models": models, "hs_bridge": types.ModuleType("hs_bridge"),
            "hs_bridge.FPGA_Execution": fe, "hs_bridge.FPGA_Execution.fpga_controller": ctrl}


def load_fpga_diff():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fpga_diff", os.path.join(ROOT, "tools", "fpga_diff.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FpgaDiff(unittest.TestCase):
    """tools/fpga_diff.py against a stand-in with planted bugs: compare has to
    name the negative-leak bug where it starts, and micro has to fail exactly
    the tests the planted bugs break, with the right smallest failing case."""

    def test_compare_finds_the_planted_negative_leak(self):
        import tempfile
        from unittest import mock
        fd = load_fpga_diff()
        net = build("avg", "flatten", "threshold", True, False, 1, seed=3)
        layers = integer_layers(net)
        torch.manual_seed(0)
        frames = (torch.rand(6, 2, 16, 16) < 0.3).to(torch.uint8)
        flush = hardware_flush_steps(net)
        with mock.patch.dict(sys.modules, planted_bug_hs_api(spike_cap=10 ** 9)):
            ref = fd.record_network(layers, (2, 16, 16), frames, flush, "simpleSim")
            hw = fd.record_network(layers, (2, 16, 16), frames, flush, "CRI")
            same = fd.record_network(layers, (2, 16, 16), frames, flush, "simpleSim")
        self.assertIsNone(fd.analyze(ref, same)["first"])
        sp = fd.analyze(ref, same)["spikes"]
        self.assertEqual(sum(sp["missed"]) + sum(sp["extra"]), 0)
        res = fd.analyze(ref, hw)
        sp = res["spikes"]
        self.assertGreater(sum(sp["missed"]) + sum(sp["extra"]), 0)
        self.assertEqual((sp["out_check_ref"], sp["out_check_hw"]), (0, 0))
        # an output spike lost on its way to the host: the membranes still call for it
        cut = dict(hw, out_spikes=hw["out_spikes"].copy())
        t, j = (int(x) for x in np.argwhere(cut["out_spikes"])[0])
        cut["out_spikes"][t, j] = False
        self.assertEqual(fd.analyze(ref, cut)["spikes"]["out_check_hw"], 1)
        self.assertIsNotNone(res["first"])
        self.assertGreater(res["origin_logical_shift"], 0, res["rows"][:3])
        self.assertTrue(all(r["before"] < 0 or r["sim"] < 0 for r in res["rows"]
                            if r["why"] == "negative leak (logical shift)"))
        with tempfile.TemporaryDirectory() as d:
            for name, rec in (("ref.npz", ref), ("hw.npz", hw)):
                fd.save_record(os.path.join(d, name), rec)
            back = fd.analyze(fd.load_record(os.path.join(d, "ref.npz")),
                              fd.load_record(os.path.join(d, "hw.npz")))
        self.assertEqual(back["first"], res["first"])
        page = fd.report_compare(res, ref, hw)
        for part in ("Where the membranes differ", "Spikes the FPGA missed and added",
                     "Output spikes per class",
                     "Membranes over the clip", "prefers-color-scheme: dark"):
            self.assertIn(part, page)

    def test_show_flags_the_neurons_the_planted_leak_hits(self):
        """show reads one simulator record: the neurons it flags as negative
        and leaking have to include where compare later finds the bug."""
        import tempfile
        from unittest import mock
        fd = load_fpga_diff()
        net = build("avg", "flatten", "threshold", True, False, 1, seed=3)
        layers = integer_layers(net)
        torch.manual_seed(0)
        frames = (torch.rand(6, 2, 16, 16) < 0.3).to(torch.uint8)
        flush = hardware_flush_steps(net)
        with mock.patch.dict(sys.modules, planted_bug_hs_api(spike_cap=10 ** 9)):
            ref = fd.record_network(layers, (2, 16, 16), frames, flush, "simpleSim")
            sim_net = sys.modules["hs_api.api"].last_net()
            hw = fd.record_network(layers, (2, 16, 16), frames, flush, "CRI")
        ref.update(clip=0, label=1, split="val")
        F = fd.spikes_from_membranes(ref["membrane"], fd.thresholds(ref["table"], ref["keys"]))
        keys = set(ref["keys"])
        self.assertEqual([{k for k, f in zip(ref["keys"], row) if f} for row in F],
                         [fired & keys for fired in sim_net.fired])
        cmp = fd.analyze(ref, hw)
        with tempfile.TemporaryDirectory() as d:
            path, out = os.path.join(d, "ref.npz"), os.path.join(d, "show.html")
            fd.save_record(path, ref)
            self.assertEqual(fd.main(["show", path, "-o", out]), 0)
            with open(out, encoding="utf-8") as f:
                page = f.read()
            res = fd.analyze_one(fd.load_record(path))
        self.assertGreater(res["n_hit"], 0)
        self.assertEqual(res["out_check"], 0)
        self.assertGreater(res["spikes"], 0)
        self.assertEqual(sum(r["spikes"] for r in res["layers"]), res["spikes"])
        self.assertEqual(res["events"]["total"], sim_net.events)
        self.assertIsNone(fd.analyze_one(dict(ref, fan_out=None))["events"])  # older records
        self.assertLessEqual(res["first_step"], cmp["first"])
        struck = {r["key"] for r in cmp["rows"] if r["why"] == "negative leak (logical shift)"}
        self.assertTrue(struck)
        self.assertLessEqual(struck, set(res["hit_keys"]))
        for r in res["rows"]:
            self.assertLess(r["membrane"], 0)
            self.assertLess(r["bug_next"], r["sim_next"])
        self.assertEqual(res["pred"], cmp["pred_ref"])
        for part in ("Spikes per step", "Activity by layer", "Headroom", "Decision over time",
                     "Negative membranes, by layer and step", "What the bug would do",
                     "Output spikes per class", "prefers-color-scheme: dark"):
            self.assertIn(part, page)

    def test_neuron_lookup_by_index_is_a_dictionary(self):
        """connectome_utils finds a neuron by index by scanning every entry,
        and hs_api does that for every neuron at every step when membranes
        are read; build() swaps in a dictionary with the same answers."""
        import types
        fd = load_fpga_diff()

        class Entry:
            def __init__(self, kind, idx):
                self.kind, self.idx = kind, idx

            def get_neuron_type(self):
                return self.kind

            def get_coreTypeIdx(self):
                return self.idx

        entries = {"A0": Entry("axon", 0), "n0": Entry("neuron", 0), "n1": Entry("neuron", 1)}

        def scan(idx):
            raise AssertionError("the slow scan was called")
        conn = types.SimpleNamespace(connectomeDict=entries, get_neuron_by_idx=scan)
        net = types.SimpleNamespace(connectome=conn)
        fd.fast_index_lookup(net)
        self.assertIs(net.connectome.get_neuron_by_idx(1), entries["n1"])
        self.assertIs(net.connectome.get_neuron_by_idx(0), entries["n0"])     # not the axon
        self.assertIsNone(net.connectome.get_neuron_by_idx(7))

    def test_micro_fails_exactly_what_the_planted_bugs_break(self):
        from unittest import mock
        fd = load_fpga_diff()
        with mock.patch.dict(sys.modules, planted_bug_hs_api(spike_cap=1000)):
            results = {t["name"]: t for t in fd.run_micro(("simpleSim", "CRI"))}
        self.assertFalse(results["negative_leak"]["pass"])
        for row in results["negative_leak"]["rows"]:
            if row["case"].startswith("positive"):
                self.assertTrue(row["pass"], row["case"])
            else:
                self.assertTrue(row["matches_logical"], row["case"])
        self.assertEqual(results["output_burst"]["first_fail"], "1024 spikes in one step")
        self.assertEqual(results["internal_burst"]["first_fail"], "1024 spikes in one step")
        self.assertEqual(results["fan_out"]["first_fail"], "axon to 2048 neurons")
        self.assertTrue(results["thresholds"]["pass"])
        self.assertTrue(results["bias_neuron"]["pass"])
        page = fd.report_micro(list(results.values()))
        self.assertIn("FPGA micro tests", page)
        self.assertIn("1024 spikes in one step", page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
