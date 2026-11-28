"""The PyTorch model and IntegerSim fire the same spikes, for every feature.

    python tests/test_sim_match.py

Training uses the PyTorch model. Deployment is judged on IntegerSim, which
`tools/match_simplesim.py --selftest` proves equal to hs_api's simpleSim. This
closes the loop: small networks combining every feature the search can pick
(average or OR pooling, flatten or GAP, threshold or bias-neuron biases,
signed spikes, per-channel tau, multi-bit input) must produce identical spikes
in every layer at every timestep, in float32 as training runs, and in the
float64 copy (hs_export.exact_copy) that deployed accuracy is measured on.

Thresholds are set low after export so every layer fires. A match on silent
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
from snnsearch.hs_export import integer_layers, IntegerSim, exact_copy, as_if_neurons
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


class UnrollMatch(unittest.TestCase):
    """unroll_to_hs_api builds each layer's synapses with numpy. It must
    write exactly what the one-synapse-at-a-time walk over _synapses wrote:
    the same keys, synapses, weights and order in every list."""

    def test_synapse_arrays_are_the_generator_in_order(self):
        from snnsearch.hs_export import _synapses, _synapse_arrays
        rng = np.random.default_rng(0)

        def check(L, prev, out):
            ref = list(_synapses(L, prev, out))
            pre, post, w = _synapse_arrays(L, prev, out)
            self.assertEqual(list(zip(pre.tolist(), post.tolist(), w.tolist())),
                             [(int(a), int(b), int(c)) for a, b, c in ref])
            return len(ref)

        for (c_in, h, w_, c_out, kh, kw, pad, stride, pool) in [
                (3, 9, 7, 4, 3, 5, (1, 2), (2, 1), 2),        # padding, stride, odd sizes behind a pool
                (2, 8, 8, 3, 3, 3, (0, 0), (1, 1), 1),
                (1, 11, 6, 2, 5, 2, (2, 0), (3, 2), 1),
                (2, 5, 5, 2, 7, 7, (0, 0), (1, 1), 1)]:          # a kernel larger than the map
            hp, wp = h // pool, w_ // pool
            out = (c_out, (hp + 2 * pad[0] - kh) // stride[0] + 1,
                   (wp + 2 * pad[1] - kw) // stride[1] + 1)
            W = rng.integers(-3, 4, (c_out, c_in, kh, kw))
            L = {"kind": "conv", "w": W, "pool": pool, "padding": pad, "stride": stride}
            if out[1] > 0 and out[2] > 0:
                with self.subTest(conv=(c_in, h, w_, kh, kw, pad, stride, pool)):
                    check(L, (c_in, h, w_), out)
        with self.subTest(kind="orpool"):
            self.assertEqual(check({"kind": "orpool", "pool": 2}, (3, 7, 9), (3, 3, 4)), 3 * 6 * 8)
        with self.subTest(kind="fc behind a pool"):
            check({"kind": "fc", "pool": 2, "w": rng.integers(-3, 4, (5, 3 * 3 * 4))},
                  (3, 7, 9), (5,))
        with self.subTest(kind="fc"):
            check({"kind": "fc", "pool": 1, "w": rng.integers(-3, 4, (4, 30))}, (30,), (4,))

    def test_unroll_writes_what_the_walk_wrote(self):
        from collections import defaultdict
        from unittest import mock
        from snnsearch.hs_export import (unroll_to_hs_api, _synapses, _normalize,
                                         _layer_shapes)

        def walked(layers, in_shape, groups):
            """Every synapse list as the old loop appended it."""
            layers = [_normalize(L) for L in layers]
            shapes = _layer_shapes(layers, in_shape)
            bits = int(layers[0].get("input_bits", 1) or 1)
            n_in = int(np.prod(in_shape))
            lists, prev = defaultdict(list), tuple(in_shape)
            for li, L in enumerate(layers):
                if li == 0:
                    sources = [([f"A{i * bits + k}" for i in range(n_in)], 1, 2 ** k)
                               for k in range(bits)]
                else:
                    sources = [(keys, sign, 1) for keys, sign in groups[li - 1]]
                for pre, post, weight in _synapses(L, prev, shapes[li]):
                    for src_keys, src_sign, mult in sources:
                        for tgt_keys, tgt_sign in groups[li]:
                            if L["kind"] == "orpool":
                                if src_sign == tgt_sign:
                                    lists[src_keys[pre]].append((tgt_keys[post], 1))
                            else:
                                w = int(weight * src_sign * tgt_sign * mult)
                                if w:
                                    lists[src_keys[pre]].append((tgt_keys[post], w))
                prev = shapes[li]
            return lists

        combos = list(itertools.product(("avg", "max"), ("flatten", "gap"),
                                        ("threshold", "neuron"), (False, True)))
        cases = [(p, h, b, s, False, 1) for p, h, b, s in combos] + [
            ("avg", "flatten", "threshold", False, True, 3), ("max", "gap", "neuron", True, True, 2)]
        for i, (pool, head, bias, signed, per_channel, bits) in enumerate(cases):
            with self.subTest(pool=pool, head=head, bias=bias, signed=signed, bits=bits):
                layers = integer_layers(build(pool, head, bias, signed, per_channel, bits, seed=i))
                with mock.patch.dict(sys.modules, planted_bug_hs_api()):
                    axons, conns, groups = unroll_to_hs_api(layers, (2, 16, 16))
                ref = walked(layers, (2, 16, 16), groups)
                n = 0
                for k, syn in axons.items():
                    self.assertEqual(syn, ref.get(k, []), k)
                    n += len(syn)
                for k, (syn, _model) in conns.items():
                    if not k.startswith("BiasN"):
                        self.assertEqual(syn, ref.get(k, []), k)
                        n += len(syn)
                self.assertEqual(n, sum(len(v) for v in ref.values()))
                self.assertGreater(n, 0)


class FpgaTarget(unittest.TestCase):
    """tools/match_simplesim.py --hs-target CRI runs clips on the FPGA through
    hs_api. Without hardware, stand-in hs_api and hs_bridge modules check the
    plumbing: the core reaches CRI_network, the chip is cleared before every
    clip, step()'s (spikes, latency, hbm_access) is read, and output spikes
    become predictions."""

    def run_stub(self, target, cores=1, parts_fn=None):
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
                calls["hbm"] = calls.get("hbm", 0) + 3
                return (spikes, calls["counter"], calls["hbm"]) if target == "CRI" else spikes

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
            out, info = ms._real_simplesim(layers, (2, 16, 16), clips, flush=2,
                                           target=target, core=5, cores=cores,
                                           parts_fn=parts_fn)
        calls["mods"] = mods
        return out, calls, info

    def test_fpga_path(self):
        out, calls, info = self.run_stub("CRI")
        self.assertEqual(calls["kwargs"], {"target": "CRI", "coreID": 5})
        self.assertEqual(len(calls["clear"]), 3)
        self.assertTrue(all(core == 5 and not dump for _n, dump, core in calls["clear"]))
        self.assertEqual([p for p, _, _, _ in out], [2, 2, 2])
        # the counters' increase within a clip, not their running values
        self.assertEqual([(c, h) for _, _, c, h in out], [(7.0, 3.0)] * 3)
        self.assertEqual(info["counter_mode"], "cumulative")
        self.assertEqual(len(out[0][1]), 4 + 2)          # T steps plus the flush
        self.assertEqual(len(info["step_s"]), 3 * (4 + 2))
        self.assertEqual(len(info["clip_s"]), 3)

    def test_simplesim_path_is_unchanged(self):
        out, calls, info = self.run_stub("simpleSim")
        self.assertEqual(calls["kwargs"], {"target": "simpleSim"})
        self.assertEqual((calls["clear"], calls["inits"]), ([], 3))
        self.assertEqual([(p, c, h) for p, _, c, h in out], [(2, None, None)] * 3)

    def test_cores_place_every_axon_and_neuron_and_clear_every_core(self):
        """--cores 4: a membership naming every axon and neuron goes to
        CRI_network with n_cores, and all four cores are cleared before
        every clip."""
        def parts_fn(n_cores, xadj, adjncy):
            return [i % n_cores for i in range(len(xadj) - 1)]
        out, calls, info = self.run_stub("CRI", cores=4, parts_fn=parts_fn)
        kw = calls["kwargs"]
        self.assertEqual((kw["target"], kw["n_cores"]), ("CRI", 4))
        self.assertNotIn("coreID", kw)
        from unittest import mock
        from snnsearch.hs_export import unroll_to_hs_api
        net = build("avg", "flatten", "threshold", False, False, 1, seed=0)
        with mock.patch.dict(sys.modules, calls["mods"]):
            axons, conns, _groups = unroll_to_hs_api(integer_layers(net), (2, 16, 16))
        self.assertEqual(set(kw["membership"]), set(axons) | set(conns))
        self.assertEqual(set(kw["membership"].values()), {0, 1, 2, 3})
        self.assertEqual(sum(info["placement"]["neurons"]), len(conns))
        self.assertEqual(len(calls["clear"]), 3 * 4)
        self.assertEqual({core for _n, _d, core in calls["clear"]}, {0, 1, 2, 3})
        self.assertEqual([p for p, _, _, _ in out], [2, 2, 2])

    def test_graph_and_placement_counts(self):
        """The METIS graph is undirected with no repeats or self loops, and a
        core's relays count each outside neuron once, its host axons each
        axon once."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "match_simplesim", os.path.join(ROOT, "tools", "match_simplesim.py"))
        ms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ms)
        axons = {"A0": [("n0", 1), ("n1", 2)], "A1": [("n2", 1)]}
        conns = {"n0": ([("n2", 1), ("n3", 1)], None), "n1": ([("n3", 1), ("n1", 5)], None),
                 "n2": ([("n0", 1)], None), "n3": ([], None)}
        names, src, dst = ms.network_edges(axons, conns)
        self.assertEqual(names, ["A0", "A1", "n0", "n1", "n2", "n3"])
        xadj, adjncy = ms.undirected_csr(len(names), src, dst)
        nbrs = {names[i]: sorted(names[j] for j in adjncy[xadj[i]:xadj[i + 1]])
                for i in range(len(names))}
        self.assertEqual(nbrs["n0"], ["A0", "n2", "n3"])   # n0-n2 both ways, kept once
        self.assertEqual(nbrs["n1"], ["A0", "n3"])         # its self loop dropped
        for a, bs in nbrs.items():
            for b in bs:
                self.assertIn(a, nbrs[b])
        # cores: A0, A1, n0, n1 on 0; n2, n3 on 1
        st = ms.placement_stats(names, src, dst, [0, 0, 0, 0, 1, 1], 2, 2)
        self.assertEqual(st["neurons"], [2, 2])
        self.assertEqual(st["host_axons_in"], [1, 1])        # A0 feeds core 0, A1 core 1
        self.assertEqual(st["relays_in"], [1, 2])            # n2 into 0; n0 and n1 into 1
        self.assertEqual(st["synapses_cut"], 4)              # n0-n2, n0-n3, n1-n3, n2-n0

    def test_fast_input_packets_send_hs_bridge_bytes(self):
        """input_packet builds the bytes hs_bridge v14's input_user sends
        (its loop is copied below, minus the DMA call), for any axon count,
        core, repeats, negatives and axons past the last packet. The swap
        checks that against the installed input_user itself, sorts the list
        in place as input_user does, and keeps the original when they differ."""
        import importlib.util
        import math
        import types
        spec = importlib.util.spec_from_file_location(
            "match_simplesim", os.path.join(ROOT, "tools", "match_simplesim.py"))
        ms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ms)
        sent = []

        def hs_bridge_input_user(inputs, numAxons, simDump=False, coreID=0, reserve=True,
                                 cont_flag=False):
            currInput = inputs
            currInput.sort()
            coreBits = np.binary_repr(coreID, 5) + 3 * '0'
            coreByte = int(coreBits, 2)
            commandList = [] if cont_flag else [0] * 62 + [coreByte] + [1]
            for count in range(math.ceil(numAxons / 256)):
                one_hot_bin = ["0"] * 256
                inputSegment = [i for i in currInput if (256 * count) <= i and i < (256 * count + 256)]
                for axon in inputSegment:
                    one_hot_bin[axon % 256] = "1"
                while one_hot_bin:
                    curr_byte = "".join(one_hot_bin[:8][::-1])
                    commandList = commandList + [int(curr_byte, 2)]
                    one_hot_bin = one_hot_bin[8:]
                commandList = commandList + 30 * [0] + [coreByte, 0]
            command = np.array(commandList, dtype=np.uint64)
            if simDump:
                return ["".join('{:0{width}x}'.format(i, width=2) for i in p[::-1])
                        for p in np.split(command, len(command) // 64)]
            sent.append(command)

        rng = np.random.default_rng(1)
        for n_axons in (0, 1, 255, 256, 257, 1000, 4100):
            for core in (0, 1, 15):
                ins = rng.integers(-3, n_axons + 300, size=int(rng.integers(0, 400))).tolist()
                hs_bridge_input_user(list(ins), n_axons, coreID=core)
                np.testing.assert_array_equal(ms.input_packet(ins, n_axons, core), sent.pop())

        dma = types.SimpleNamespace(
            DmaMethodNormal="normal",
            dma_dump_write=lambda cmd, n, *rest: sent.append((np.array(cmd), n, rest)))
        ctrl = types.SimpleNamespace(input_user=hs_bridge_input_user, dmadump=dma)
        network = types.SimpleNamespace(input_user=hs_bridge_input_user)
        self.assertTrue(ms.use_fast_input_packets(ctrl, network))
        self.assertIs(network.input_user, ctrl.input_user)
        self.assertIsNot(ctrl.input_user, hs_bridge_input_user)
        self.assertTrue(ms.use_fast_input_packets(ctrl, network))     # twice is fine
        ins = [700, 3, 255, 3]
        network.input_user(ins, 1000, coreID=9)
        self.assertEqual(ins, [3, 3, 255, 700])
        cmd, n, rest = sent.pop()
        self.assertEqual((n, rest), (len(cmd), (1, 0, 0, 0, "normal")))
        hs_bridge_input_user(list(ins), 1000, coreID=9)
        np.testing.assert_array_equal(cmd, sent.pop())

        def different(inputs, numAxons, simDump=False, **kw):
            return ["00" * 64] if simDump else None
        ctrl = types.SimpleNamespace(input_user=different, dmadump=dma)
        self.assertFalse(ms.use_fast_input_packets(ctrl))
        self.assertIs(ctrl.input_user, different)

    def test_host_meter_counts_dma_calls_and_what_they_read(self):
        """HostMeter wraps the dmadump module's read and write and hs_bridge's
        spike decoder: per step it reports the calls, their time, and the
        packets read by kind, and it follows a reset."""
        import importlib.util
        import types
        spec = importlib.util.spec_from_file_location(
            "match_simplesim", os.path.join(ROOT, "tools", "match_simplesim.py"))
        ms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ms)

        def packet(b62, b63):
            return [0] * 62 + [b62, b63]

        stream = [packet(255, 255), packet(238, 238), packet(238, 238), packet(205, 171),
                  packet(0xBA, 0xBA), packet(0xBA, 0xCA), packet(7, 7)]
        dd = types.SimpleNamespace(
            dma_dump_read=lambda *a: (0, stream.pop(0)),
            dma_dump_write=lambda *a: 0)
        ctrl = types.SimpleNamespace(read_spikes=lambda p: (0, [(0, 1, 0)]))
        meter = ms.HostMeter(dd, ctrl)
        dd.dma_dump_write([0] * 64, 64, 1, 0, 0, 0, 0)
        for _ in range(7):
            _code, p = dd.dma_dump_read(1, 0, 0, 0, 0, 64)
            if p[62] == 238:
                self.assertEqual(ctrl.read_spikes(p), (0, [(0, 1, 0)]))
        h = meter.per_step(steps=1, step_s=1.0)
        self.assertEqual((h["dma_reads"], h["dma_writes"]), (7, 1))
        self.assertEqual(h["packets"], {"count": 2, "done": 1, "empty": 1, "other": 1,
                                        "spikes": 2})
        self.assertGreaterEqual(h["decode_s"], 0.0)
        self.assertAlmostEqual(h["other_s"], 1.0 - h["dma_read_s"] - h["dma_write_s"]
                               - h["decode_s"])
        line = ms.host_line(h)
        self.assertIn("7 DMA reads", line)
        self.assertIn("2 spikes", line)
        load = ms.load_split(meter, 120.0)
        self.assertEqual((load["dma_writes"], load["dma_reads"]), (1, 7))
        self.assertAlmostEqual(load["python_s"], 120.0 - load["dma_write_s"] - load["dma_read_s"])
        self.assertIn("1 DMA writes", ms.load_line(load))
        self.assertIn("building it in Python 2.0 min", ms.load_line(load))
        meter.reset()
        self.assertEqual(meter.per_step(1, 0.0)["dma_reads"], 0)

    def test_card_option_sends_every_dma_call_to_that_card(self):
        """--card N: hs_bridge passes card 0 to every dmadump call. After
        use_card, every write, read and full dump goes to card N, by position
        or by keyword, also under a wrapper added afterwards (as hs_bridge's
        write check is). card_files lists one card's device files only."""
        import importlib.util
        import tempfile
        import types
        spec = importlib.util.spec_from_file_location(
            "match_simplesim", os.path.join(ROOT, "tools", "match_simplesim.py"))
        ms = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ms)
        seen = []

        def dma_dump_write(data, nWriteData, wordSize, index, engineIndex, address, dmaMethod):
            seen.append(("write", index))
            return 0

        def dma_dump_read(wordSize, index, engineIndex, address, dmaMethod, readCount):
            seen.append(("read", index))
            return 0, [0] * 64

        def dma_dump(writeNotRead, nWriteData, wordSize, data, index, engineIndex, address,
                     byteCount, dmaMethod, readCount):
            seen.append(("dump", index))

        dd = types.SimpleNamespace(dma_dump_write=dma_dump_write, dma_dump_read=dma_dump_read,
                                   dma_dump=dma_dump, DmaMethodNormal=0)
        ms.use_card(dd, 2)
        dd.dma_dump_write([1], 1, 1, 0, 0, 0, 0)
        dd.dma_dump_read(1, 0, 0, 0, 0, 64)
        dd.dma_dump(1, 1, 1, [1], 0, 0, 0, 8, 0, 1)
        dd.dma_dump_write([1], 1, 1, index=0, engineIndex=0, address=0, dmaMethod=0)
        checked = dd.dma_dump_write
        dd.dma_dump_write = lambda *a, **k: checked(*a, **k)
        dd.dma_dump_write([1], 1, 1, 0, 0, 0, 0)
        self.assertEqual(seen, [("write", 2), ("read", 2), ("dump", 2), ("write", 2),
                                ("write", 2)])
        ms.use_card(dd, 2)                           # the same card again is fine
        with self.assertRaises(RuntimeError):
            ms.use_card(dd, 3)

        with tempfile.TemporaryDirectory() as dev:
            for name in ["adxdma0", "adxdma0ch0", "adxdma2", "adxdma2ch0", "adxdma2hc0",
                         "adxdma2ui", "adxdma2w0", "adxdma21", "adxdma21ch0", "null"]:
                open(os.path.join(dev, name), "w").close()
            self.assertEqual([os.path.basename(p) for p in ms.card_files(2, dev)],
                             ["adxdma2", "adxdma2ch0", "adxdma2hc0", "adxdma2ui", "adxdma2w0"])
            self.assertEqual(ms.adxdma_cards(dev, os.path.join(dev, "no_sysfs")),
                             {0: None, 2: None, 21: None})


def planted_bug_hs_api(spike_cap=1000):
    """Stand-in hs_api and hs_bridge modules: a small LIF simulator whose
    "simpleSim" target is correct and whose "CRI" target has two planted
    bugs, a logical-shift leak for negative membranes and a cap on spikes per
    step. tools/fpga_diff.py has to find both."""
    import types
    WRAP = 2 ** 32                      # the membrane, bits 31:0 of the word
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
                self.v[k] = (0 if m.kind == "ANN" else v if m.kind == "IF"
                             else self.leak(v, m.leak))
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
            mem = list(self.v.items())     # signed, as hs_bridge decodes them
            if self.target == "CRI":
                return mem, (out, self.counter, 0)
            return mem, out

    api = types.ModuleType("hs_api.api")
    api.CRI_network = FakeNet
    api.last_net = lambda: state["net"]
    models = types.ModuleType("hs_api.neuron_models")
    models.LIF_neuron = lambda *a: Model("LIF", *a)
    models.ANN_neuron = lambda *a: Model("ANN", *a)
    models.IF_neuron = lambda theta, nu=-17: Model("IF", theta, nu, 0)
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

    def test_if_neurons_dodge_the_planted_leak(self):
        """The same network deployed as IF neurons: hs_api gets IF models for
        every leaking layer, IntegerSim matches its simpleSim membrane for
        membrane, and the planted negative-leak bug has nothing to act on."""
        from unittest import mock
        fd = load_fpga_diff()
        net = build("avg", "flatten", "threshold", True, False, 1, seed=3)
        layers = as_if_neurons(integer_layers(net))
        self.assertTrue(all(L["if_neuron"] != L["memoryless"] for L in layers))
        torch.manual_seed(0)
        frames = (torch.rand(6, 2, 16, 16) < 0.3).to(torch.uint8)
        flush = hardware_flush_steps(net)
        with mock.patch.dict(sys.modules, planted_bug_hs_api(spike_cap=10 ** 9)):
            ref = fd.record_network(layers, (2, 16, 16), frames, flush, "simpleSim")
            kinds = {m.kind for _syn, m in sys.modules["hs_api.api"].last_net().conns.values()}
            hw = fd.record_network(layers, (2, 16, 16), frames, flush, "CRI")
        self.assertIn("IF", kinds)
        self.assertNotIn("LIF", kinds)
        self.assertTrue((np.asarray(ref["membrane"]) < 0).any())    # negatives to leak
        self.assertIsNone(fd.analyze(ref, hw)["first"])
        sim = IntegerSim(layers, (2, 16, 16))
        rows = [np.concatenate([v.flatten().numpy() for v, _s in groups])
                for groups, _out in sim.steps(frames, flush)]
        np.testing.assert_array_equal(np.array(rows, np.int64), ref["membrane"])
        self.assertEqual(fd.analyze_one(ref)["n_hit"], 0)       # nothing leaks

    def test_neuron_lookup_by_index_is_a_dictionary(self):
        """connectome_utils finds a neuron by index by scanning every entry,
        and hs_api does that for every neuron at every step when membranes
        are read (by core index on simpleSim, by HBM index on the FPGA);
        build() swaps in dictionaries with the same answers."""
        import types
        fd = load_fpga_diff()

        class Entry:
            def __init__(self, kind, idx):
                self.kind, self.idx = kind, idx

            def get_neuron_type(self):
                return self.kind

            def get_coreTypeIdx(self):
                return self.idx

            def get_hbmIdx(self):
                return 100 + self.idx

        entries = {"A0": Entry("axon", 0), "n0": Entry("neuron", 0), "n1": Entry("neuron", 1),
                   "n1b": Entry("neuron", 1)}

        def scan(idx):
            raise AssertionError("the slow scan was called")
        conn = types.SimpleNamespace(connectomeDict=entries, get_neuron_by_idx=scan,
                                     get_neuron_by_hbmIdx=scan)
        net = types.SimpleNamespace(connectome=conn)
        fd.fast_index_lookup(net)
        for lookup, base in (("get_neuron_by_idx", 0), ("get_neuron_by_hbmIdx", 100)):
            find = getattr(net.connectome, lookup)
            self.assertIs(find(base + 1), entries["n1"])      # the first match, as the scan
            self.assertIs(find(base), entries["n0"])          # not the axon
            self.assertIsNone(find(base + 7))
        # an older connectome_utils with no HBM lookup gets none added
        old = types.SimpleNamespace(connectomeDict=entries, get_neuron_by_idx=scan)
        fd.fast_index_lookup(types.SimpleNamespace(connectome=old))
        self.assertFalse(hasattr(old, "get_neuron_by_hbmIdx"))

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
        self.assertTrue(results["if_neuron"]["pass"])
        page = fd.report_micro(list(results.values()))
        self.assertIn("FPGA micro tests", page)
        self.assertIn("1024 spikes in one step", page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
