"""Check the SynOps arithmetic against cases with a known answer.

A search reported 73,228,244 SynOps per sample for a network whose spiking
layers hold 3,741,184 outgoing connections driven over 12 passes. The ceiling
is 44,894,208. The measurement was 1.63x an amount that cannot be exceeded, and
it went to a leaderboard as a fact.

Nothing caught it because the number looked plausible and nothing knew what
plausible was. These tests know: for hand-built cases the exact answer is
computable, and the ceiling is checkable for any case at all.

Runs with torch when torch is present, and falls back to checking the pure
arithmetic when it is not, so it is useful on a laptop and on the cluster.

    python tests/test_synops.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

passed = failed = 0


def check(name, got, want, tol=1e-6):
    global passed, failed
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    print(f"[{'  ok  ' if ok else ' FAIL '}] {name:<48} got {got:,.1f}  want {want:,.1f}")
    passed, failed = passed + ok, failed + (not ok)
    return ok


def check_true(name, cond, detail=""):
    global passed, failed
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name:<48} {detail}")
    passed, failed = passed + bool(cond), failed + (not cond)
    return cond


def dense_macs_arithmetic():
    """The formulas, independent of torch."""
    print("\n-- dense MAC formulas --")
    # Linear(1024 -> 512), batch 1: every output touches every input.
    check("Linear 1024->512 dense MACs", 512 * 1024, 524_288)
    # Conv2d(64 -> 64, k=7) producing 4x4: each output element reads 7*7*64.
    check("Conv 64->64 k7 out 4x4 dense MACs", 64 * 4 * 4 * 64 * 7 * 7, 3_211_264)


def summary_guards():
    """Exercise the real summary guard even without a tensor runtime."""
    from snnsearch.synops import SynOpsCounter

    print("\n-- the ceiling guard --")
    c = SynOpsCounter()
    check_true("empty measurement has a reason",
               c.summary()["synops_per_sample"] is None and bool(c.summary().get("reason")))
    c.add_samples(3)
    # Two temporal passes for batches of two and one samples: six total
    # sample-passes, with eight dense MACs per sample-pass.
    c.calls = {"lin": 4}
    c.ceiling = {"lin": 48}
    c.ops = {"lin": 48.0}
    rows = [{"layer": "unrelated_cost_name", "macs": 8}]
    s = c.summary(cost_rows=rows)
    check("a total at the ceiling is accepted", s["synops_per_sample"], 16)
    check("ceiling normalizes all batches once", s["synops_ceiling_per_sample"], 16)
    check("ANN comparison retains temporal work", s["synops_over_dense"], 2)

    # This is below the old (incorrect) ceiling of 8 * 4 calls per sample.
    c.ops["lin"] = 60.0
    for cost_rows in (rows, None):
        s = c.summary(cost_rows=cost_rows)
        check_true("excess rejected with or without cost table",
                   s["synops_per_sample"] is None and "ceiling" in s.get("reason", ""))
        check_true("invalid derived metrics are not reported",
                   s.get("synops_over_dense") is None
                   and s["per_layer"]["lin"]["synops_per_sample"] is None)

    # A large silent layer must not conceal an impossible smaller layer.
    c.ops["silent"] = 0.0
    c.ceiling["silent"] = 1000
    check_true("silent layers cannot hide an impossible layer",
               c.summary()["synops_per_sample"] is None)

    for bad in (float("nan"), float("inf"), -1.0):
        c.ops["lin"] = bad
        s = c.summary()
        check_true("non-finite/negative activity is invalid",
                   s["synops_per_sample"] is None and bool(s.get("reason")), str(bad))

    c.spikes["lif"] = 2
    c.spike_elements["lif"] = 4
    c.reset()
    check_true("reset clears all measurement state",
               not any((c.ops, c.ceiling, c.calls, c.spikes,
                        c.spike_elements, c.samples)))


def with_torch():
    import torch
    from torch import nn
    from snnsearch.synops import SynOpsCounter

    print("\n-- measured against a known-sparse input --")

    # A Linear layer driven by an input with exactly 100 ones out of 1024.
    lin = nn.Linear(1024, 512, bias=False)
    x = torch.zeros(1, 1024)
    x[0, :100] = 1.0

    c = SynOpsCounter()
    with c.attach(lin):
        with torch.no_grad():
            lin(x)
        c.add_samples(1)
    s = c.summary()
    # 100 input spikes, each driving 512 accumulates.
    check("Linear, 100 of 1024 active", s["synops_per_sample"], 100 * 512)

    # Fully dense input must reproduce the dense MAC count exactly.
    c = SynOpsCounter()
    with c.attach(lin):
        with torch.no_grad():
            lin(torch.ones(1, 1024))
        c.add_samples(1)
    check("Linear, fully dense == dense MACs",
          c.summary()["synops_per_sample"], 1024 * 512)

    # Silent input must cost nothing. This is the property the whole
    # event-driven argument rests on.
    c = SynOpsCounter()
    with c.attach(lin):
        with torch.no_grad():
            lin(torch.zeros(1, 1024))
        c.add_samples(1)
    check("Linear, no spikes == no ops", c.summary()["synops_per_sample"], 0.0)

    # Conv2d, half the input active.
    conv = nn.Conv2d(64, 64, kernel_size=7, stride=2, bias=False)
    xi = torch.zeros(1, 64, 13, 13)
    xi.view(-1)[: xi.numel() // 2] = 1.0
    c = SynOpsCounter()
    with c.attach(conv):
        with torch.no_grad():
            out = conv(xi)
        c.add_samples(1)
    dense = out.numel() * 64 * 7 * 7
    check("Conv2d, half active == half dense", c.summary()["synops_per_sample"],
          dense * 0.5)

    # Grouped convolution: each output reads only in_channels / groups.
    conv = nn.Conv2d(4, 6, kernel_size=3, groups=2, bias=False)
    c = SynOpsCounter()
    with c.attach(conv), torch.no_grad():
        conv(torch.ones(2, 4, 5, 5))
        c.add_samples(2)
    check("grouped convolution dense arithmetic",
          c.summary()["synops_per_sample"], 6 * 3 * 3 * 2 * 3 * 3)
    c.reset()
    with c.attach(conv), torch.no_grad():
        conv(torch.zeros(1, 4, 5, 5))
        c.add_samples(1)
    check("Conv2d, no spikes == no ops", c.summary()["synops_per_sample"], 0)

    print("\n-- batches, timesteps, and firing reports --")
    lin = nn.Linear(4, 2, bias=False)
    rows = [{"layer": "lin", "macs": 8}]
    c = SynOpsCounter()
    with c.attach(lin), torch.no_grad():
        for batch_size in (2, 1):
            for _ in range(3):
                lin(torch.ones(batch_size, 4))
            c.add_samples(batch_size)
    s = c.summary(cost_rows=rows)
    check("uneven batches have the same per-sample cost", s["synops_per_sample"], 24)
    check("uneven batches do not loosen the ceiling", s["synops_ceiling_per_sample"], 24)
    c.ops[""] *= 1.25
    check_true("multiple batches still reject excess activity",
               c.summary(cost_rows=rows)["synops_per_sample"] is None)

    c = SynOpsCounter()
    with c.attach(lin), torch.no_grad():
        lin(torch.ones(3, 2, 4))  # T, N, features in one module call
        c.add_samples(2)
    s = c.summary(cost_rows=rows)
    check("multi-step Linear accounts for every timestep", s["synops_per_sample"], 24)
    check("multi-step Linear has the same dense ceiling", s["synops_ceiling_per_sample"], 24)

    from snnsearch.neuron import HardwareLIFNode
    from spikingjelly.activation_based import layer

    conv = layer.Conv2d(1, 2, kernel_size=1, bias=False, step_mode="m")
    c = SynOpsCounter()
    with c.attach(conv), torch.no_grad():
        conv(torch.ones(3, 2, 1, 2, 2))
        c.add_samples(2)
    check("multi-step Conv2d has the same dense ceiling",
          c.summary(cost_rows=rows)["synops_per_sample"], 24)

    net = nn.Sequential(nn.Linear(4, 2, bias=False), HardwareLIFNode(v_threshold=0.5),
                        nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        net[0].weight.fill_(0.5)
    c = SynOpsCounter()
    with c.attach(net), torch.no_grad():
        net(torch.ones(1, 4))  # the hardware neuron has one timestep of latency
        net(torch.ones(1, 4))
        c.add_samples(1)
    s = c.summary()
    check("LIF spikes feed reporting, not the operation sum", s["synops_per_sample"], 18)
    check("the fully connected head counts its own input", s["per_layer"]["2"]["synops_per_sample"], 2)
    check("LIF firing rate uses all observed elements", s["firing_rate"], 0.5)
    check_true("LIF layer has no SynOps entry", "synops_per_sample" not in s["per_layer"]["1"])
    check_true("context manager removes all hooks",
               all(not m._forward_hooks for m in net.modules()))

    # Running half precision must not overflow the spike sum at 65,504.
    lin = nn.Linear(4, 2, bias=False).half()
    c = SynOpsCounter()
    with c.attach(lin), torch.no_grad():
        lin(torch.ones(20_000, 4, dtype=torch.float16))
        c.add_samples(20_000)
    check("half-precision activity sum stays finite", c.summary()["synops_per_sample"], 8)

    print("\n-- public measurement API --")
    from snnsearch.synops import measure_synops

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_fc = nn.Sequential(nn.Flatten(), nn.Linear(4, 2, bias=False))

        def forward(self, x):
            return self.conv_fc(x)

    net = TinyNet().train()
    loader = [(torch.ones(n, 3, 1, 2, 2), torch.zeros(n, dtype=torch.long)) for n in (2, 1)]
    s = measure_synops(net, loader, torch.device("cpu"), None, 1)
    check("max_batches counts only measured samples", s["samples"], 2)
    check("public API preserves temporal arithmetic", s["synops_per_sample"], 24)
    check_true("measurement restores training mode and hooks",
               net.training and all(not m._forward_hooks for m in net.modules()))

    # Analog values above one cannot yield a valid event count.
    bad_loader = [(torch.full((1, 2, 1, 2, 2), 2.0), torch.zeros(1, dtype=torch.long))]
    s = measure_synops(net, bad_loader, torch.device("cpu"), None, 1)
    check_true("public API refuses impossible measurements without spec",
               s["synops_per_sample"] is None and "ceiling" in s.get("reason", ""))


def check_num(name, got, want):
    """check(), but a missing (None) measurement fails instead of raising."""
    if got is None or want is None:
        return check_true(name, False, f"got {got}  want {want}")
    return check(name, got, want)


def signed_spikes():
    """A signed spike is +1 or -1, and either is one event.

    Summing spikes let a -1 cancel a +1. Search 05's best trial, signed, got
    no SynOps at all: the sum came out negative and was rejected as invalid.
    Behind a pool the cancelling is worse, because the average of +1 and -1
    is 0, and so is OR(+) - OR(-) when both fire.

    The reference is the same network with every spike positive: weights |w|
    on an input whose two channels never overlap make each unit's drive
    |w . x|, so it fires exactly where the signed unit fired, sign dropped.
    The two fire the same spikes, but the chip does more for the signed one:
    each spike reaches both neurons of every signed unit it feeds, and an OR
    pool has one OR neuron per sign.
    """
    import torch
    from torch import nn
    from spikingjelly.activation_based import functional, layer
    from snnsearch._torch import set_step_mode
    from snnsearch.neuron import HardwareLIFNode, OrPool
    from snnsearch.synops import SynOpsCounter

    print("\n-- signed spikes --")

    lin = nn.Linear(1024, 512, bias=False)
    x = torch.zeros(1, 1024)
    x[0, :60] = 1.0
    x[0, 60:100] = -1.0
    c = SynOpsCounter()
    with c.attach(lin), torch.no_grad():
        lin(x)
        c.add_samples(1)
    check_num("Linear, 60 of +1 and 40 of -1", c.summary()["synops_per_sample"], 100 * 512)
    c.reset()
    with c.attach(lin), torch.no_grad():
        lin(-torch.ones(1, 1024))
        c.add_samples(1)
    check_num("Linear, all -1 == dense MACs", c.summary()["synops_per_sample"], 1024 * 512)

    # Channel 0 on a checkerboard, channel 1 on the other squares. The signed
    # weights drive +1 on one color and -1 on the other, so every 2x2 window
    # holds two of each and averages to zero.
    board = (torch.arange(4).reshape(4, 1) + torch.arange(4)) % 2
    x = torch.stack([board, 1 - board]).float().unsqueeze(0)      # 1, 2, 4, 4
    w = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).reshape(2, 2, 1, 1)
    T = 4

    def build(signed, pool, step_mode):
        conv = layer.Conv2d(2, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            conv.weight.copy_(w if signed else w.abs())
        pooling = {"avg": layer.AvgPool2d(2), "gap": layer.AdaptiveAvgPool2d(1),
                   "or": OrPool(2, 2)}[pool]
        net = nn.Sequential(conv, HardwareLIFNode(v_threshold=0.5, signed=signed), pooling,
                            layer.Flatten(), layer.Linear(2 if pool == "gap" else 8, 3, bias=False))
        set_step_mode(net, step_mode)
        return net.eval()

    def measure(net, step_mode):
        """Two samples with a reset between, as forward_over_time runs them."""
        spikes = []
        probe = net[1].register_forward_hook(lambda m, i, o: spikes.append(o.reshape(-1).clone()))
        c = SynOpsCounter()
        with c.attach(net), torch.no_grad():
            for _sample in range(2):
                if step_mode == "m":
                    net(x.unsqueeze(0).expand(T, *x.shape).contiguous())
                else:
                    for _t in range(T):
                        net(x)
                functional.reset_net(net)
                c.add_samples(1)
        probe.remove()
        return c.summary(), torch.cat(spikes)

    # By hand, per sample: the conv reads 16 of 32 inputs on each of 4 steps,
    # each reaching its 2 output channels: 16 x 2 x 4 = 128, and 256 into a
    # signed layer, where every unit is two neurons. All 32 units fire on
    # steps 2 to 4 (the hardware neuron fires on the membrane carried in).
    # The head (no neurons of its own here) reads them through the pool:
    #   avg, GAP  folded into the head on chip, so each of the 32 units keeps
    #             its own 3 synapses: 3 steps x 32 x 3 = 288, signed or not
    #   OR        the OR neurons read 32 |spikes| on 3 steps (96) and emit one
    #             step late, on steps 3 and 4: 8 windows x 3 x 2 = 48. Signed,
    #             every window holds both signs and emits two spikes: 96.
    # OR's count also proves the reset: carrying sample 1's last step into
    # sample 2 would add to sample 2.
    want = {False: {"avg": 128 + 288, "gap": 128 + 288, "or": 128 + 96 + 48},
            True: {"avg": 256 + 288, "gap": 256 + 288, "or": 256 + 96 + 96}}
    head = {False: {"avg": 288, "gap": 288, "or": 48},
            True: {"avg": 288, "gap": 288, "or": 96}}
    kinds = {"avg": ["conv", "linear"], "gap": ["conv", "linear"],
             "or": ["conv", "orpool", "linear"]}
    for step_mode in ("s", "m"):
        for pool in ("avg", "gap", "or"):
            tag = f"{pool} pool, step mode {step_mode}:"
            s_signed, spk_signed = measure(build(True, pool, step_mode), step_mode)
            s_plain, spk_plain = measure(build(False, pool, step_mode), step_mode)
            check_true(f"{tag} signed net fires -1 spikes", bool((spk_signed < 0).any()))
            check_true(f"{tag} |spikes| equal the unsigned net's",
                       torch.equal(spk_signed.abs(), spk_plain))
            for signed, summ in ((False, s_plain), (True, s_signed)):
                name = "signed" if signed else "unsigned"
                check_num(f"{tag} {name} SynOps by hand", summ["synops_per_sample"],
                          want[signed][pool])
                check_num(f"{tag} {name} head by hand",
                          summ["per_layer"]["4"]["synops_per_sample"], head[signed][pool])
                check_true(f"{tag} {name} layers in order",
                           [r["kind"] for r in summ["layers"]] == kinds[pool])
            check_num(f"{tag} signed firing rate == unsigned",
                      s_signed["firing_rate"], s_plain["firing_rate"])
    check_num("firing rate counts every |spike|", s_signed["firing_rate"], 0.75)

    print("\n-- signed spikes in the deployed model --")
    deployed_signed_model()


def deployed_model(pool_type, head, norm, signed, bits=1):
    """A small converted network, as a search trial measures it, with low
    thresholds so every layer fires. Input is 2 x 16 x 16, T = 6."""
    import json
    import torch
    from snnsearch.config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                                  HeadSpec, NeuronSpec, TrainSpec)
    from snnsearch.model import build_model
    from snnsearch.neuron import HardwareLIFNode

    torch.manual_seed(0)
    layers = [{"out_channels": 6, "kernel_size": 3, "stride": 1, "pool": True},
              {"out_channels": 5, "kernel_size": 5, "stride": 1, "pool": True}]
    spec = {"input": InputSpec(C=2, H=16, W=16, T=6, resize_to=16, N=2, input_bits=bits),
            "encoder": EncoderSpec(layers_json=json.dumps(layers), norm=norm),
            "output": OutputSpec(num_classes=4),
            "downsample": DownsampleSpec(mode="stride", pool_type=pool_type),
            "head": HeadSpec(final_reduction=head, fc_widths="", dropout_rate=0.0),
            "neuron": NeuronSpec(tau=4, signed_spikes=signed),
            "train": TrainSpec(fold_bias_mode="threshold")}
    net = build_model(spec).eval()
    net.to_qat_folded(bias_mode="threshold")
    for m in net.modules():
        if hasattr(m, "bn"):
            m.conv.weight.data.mul_(4)
    net = net.export_deployed().eval()
    for m in net.conv_fc:
        if isinstance(m, HardwareLIFNode):
            m.learn_threshold = False
            m._v_threshold_const = 0.03
            m._v_threshold_neg_const = 0.03
    return net, spec


def deployed_signed_model():
    """The real converted network, signed, through measure_synops: the module
    types, flush steps and step modes a search trial measures."""
    import torch
    from snnsearch.synops import measure_synops

    for pool_type, head, norm in (("avg", "gap", "bn"), ("max", "flatten", "bn"),
                                  ("avg", "flatten", "tdbn")):
        net, spec = deployed_model(pool_type, head, norm, signed=True)
        loader = [((torch.rand(2, 6, 2, 16, 16) < 0.3).float(), torch.zeros(2, dtype=torch.long))
                  for _ in range(2)]
        s = measure_synops(net, loader, torch.device("cpu"), spec=spec)
        tag = f"{pool_type} pool, {head}, {norm}:"
        syn = s["synops_per_sample"]
        check_true(f"{tag} SynOps measured", syn is not None and syn > 0, s.get("reason", ""))
        check_true(f"{tag} within the dense ceiling",
                   syn is not None and syn <= s["synops_ceiling_per_sample"])
        check_true(f"{tag} firing rate positive", (s.get("firing_rate") or 0) > 0)


def multi_bit_input():
    """B-bit input: levels 0 to 2**B - 1, carried on B axons per value.

    Search 05 rejected multi-bit trials with "exceeds the dense ceiling", by
    up to 4x: the first layer counted a level of 7 as seven events against a
    ceiling of one per value. On chip a 7 is three axon spikes (bits 0, 1
    and 2), and a value has B synapses per weight, so the count is the set
    bits and the ceiling is B times the layer's dense MACs.
    """
    import torch
    from torch import nn
    from snnsearch.encoders import LevelsEncoder, PassthroughEncoder
    from snnsearch.synops import SynOpsCounter
    from snnsearch.train import forward_over_time

    print("\n-- multi-bit input --")

    net = nn.Sequential(nn.Conv2d(1, 4, kernel_size=1, bias=False))
    net.input_bits = 3

    def count(x):
        c = SynOpsCounter()
        with c.attach(net), torch.no_grad():
            net(x)
            c.add_samples(1)
        return c.summary()

    # every level 0..7 once: 0+1+1+2+1+2+2+3 = 12 axon spikes, each reaching
    # the 4 outputs of a 1x1 conv; 8 values x 3 axons x 4 outputs at most
    s = count(torch.arange(8.0).reshape(1, 1, 2, 4))
    check_num("levels 0..7 count one event per set bit", s["synops_per_sample"], 12 * 4)
    check_num("ceiling is B synapses per value", s["synops_ceiling_per_sample"], 8 * 3 * 4)
    s = count(torch.full((1, 1, 2, 4), 7.0))
    check_num("every bit set reaches the ceiling, no more", s["synops_per_sample"], 8 * 3 * 4)
    s = count(torch.full((1, 1, 2, 4), 2.5))
    check_true("a value no axon can carry is no measurement",
               s["synops_per_sample"] is None and "whole levels" in s.get("reason", ""),
               s.get("reason", ""))

    # The encoders a search uses, into the converted network. Both inputs
    # average above one level, where the old count overflowed the ceiling.
    torch.manual_seed(1)
    cases = (("passthrough", 3, PassthroughEncoder(T=6, size=(16, 16), input_bits=3),
              torch.randint(0, 4, (2, 6, 2, 32, 32)).float()),      # event counts
             ("levels", 2, LevelsEncoder(T=6, input_bits=2),
              torch.rand(2, 2, 16, 16)))                             # an image
    for coding, B, encoder, raw in cases:
        tag = f"{coding}, {B} bits:"
        x = encoder(raw)
        check_true(f"{tag} whole levels, mean above 1",
                   bool(torch.all(x == x.round()) and x.max() == 2 ** B - 1 and x.mean() > 1),
                   f"mean {float(x.mean()):.2f}")
        net, _spec = deployed_model("avg", "gap", "bn", signed=False, bits=B)
        c = SynOpsCounter()
        with c.attach(net), torch.no_grad():
            forward_over_time(net, x)
            c.add_samples(x.shape[0])
        s = c.summary()
        check_true(f"{tag} SynOps measured", (s["synops_per_sample"] or 0) > 0,
                   s.get("reason", ""))
        # Each set bit is one axon spike, reaching that value's own synapses
        # (fewer at the border, where the window holds padding). Flush steps
        # read zeros. The input alone decides it, so the chip reference is
        # exact even for this float32 network.
        ref = chip_events(net, list(x), B)
        check_num(f"{tag} first layer counts the set bits",
                  s["layers"][0]["synops_per_sample"], ref["events"][0] / x.shape[0])


def folded_pools_by_hand():
    """Behind an average pool or GAP, every neuron in front of the pool fires
    its own synapses.

    The chip has no pooling unit. hs_export folds an average pool into the
    next layer (integer_layers "pool" and "scale"), so after a 2x2 pool that
    layer reads all four neurons of each window, each through its own
    synapse, and a spike in front of the pool reaches every synapse of its
    pooled position. The counter used to read the pooled mean, a quarter of
    an event per spike: rank2's conv1 counted 4x low. Each case is counted by
    hand and against the synapses hs_export writes (_synapses).
    """
    import numpy as np
    import torch
    from torch import nn
    from snnsearch.hs_export import _synapses
    from snnsearch.synops import SynOpsCounter

    print("\n-- behind an average pool and GAP, by hand --")

    def count(net, x):
        c = SynOpsCounter()
        with c.attach(net), torch.no_grad():
            net(x)
            c.add_samples(1)
        return c.summary()

    def exported(L, x, out_shape):
        """Spikes in x times the synapses hs_export gives each neuron."""
        prev = tuple(x.shape[1:])
        deg = np.zeros(int(np.prod(prev)))
        for pre, _post, w in _synapses(L, prev, out_shape):
            deg[pre] += int(w) != 0
        return float((x.reshape(-1).numpy() * deg).sum())

    # five spikes on a 4x4 map: three in the top-left window
    x4 = torch.zeros(1, 1, 4, 4)
    for y, xx in ((0, 0), (0, 1), (1, 1), (2, 3), (3, 0)):
        x4[0, 0, y, xx] = 1.0
    # four spikes on an 8x8 map: pooled to (0, 0) twice, (1, 1), (1, 3)
    x8 = torch.zeros(1, 1, 8, 8)
    for y, xx in ((0, 0), (1, 0), (3, 3), (2, 7)):
        x8[0, 0, y, xx] = 1.0
    ones = lambda *shape: np.ones(shape, dtype=np.int64)

    # Linear behind 2x2: each spike reaches the 3 outputs. The pooled mean
    # gave 5/4 x 3 = 3.75. The ceiling is all 16 neurons, 16 x 3.
    s = count(nn.Sequential(nn.AvgPool2d(2), nn.Flatten(), nn.Linear(4, 3, bias=False)), x4)
    want = exported({"kind": "linear", "w": ones(3, 4), "pool": 2}, x4, (3,))
    check_num("Linear behind 2x2 avg, by hand", s["synops_per_sample"], 5 * 3)
    check_num("Linear behind 2x2 avg == hs_export synapses", s["synops_per_sample"], want)
    check_num("Linear behind 2x2 avg ceiling", s["synops_ceiling_per_sample"], 16 * 3)
    check_num("Linear behind 2x2 avg pool area", s["layers"][0]["input_pool_area"], 4)

    # Linear behind GAP: the same 5 spikes, 3 outputs each, not 5/16 x 3.
    s = count(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 3, bias=False)), x4)
    want = exported({"kind": "linear", "w": ones(3, 1), "pool": 4}, x4, (3,))
    check_num("Linear behind GAP, by hand", s["synops_per_sample"], 5 * 3)
    check_num("Linear behind GAP == hs_export synapses", s["synops_per_sample"], want)
    check_num("Linear behind GAP ceiling", s["synops_ceiling_per_sample"], 16 * 3)
    check_num("Linear behind GAP pool area", s["layers"][0]["input_pool_area"], 16)

    # 2x2 pool then GAP, as a pooled last block before a GAP head: 8x8 behind
    # each value, so 4 spikes x 3 outputs.
    s = count(nn.Sequential(nn.AvgPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                            nn.Linear(1, 3, bias=False)), x8)
    want = exported({"kind": "linear", "w": ones(3, 1), "pool": 8}, x8, (3,))
    check_num("Linear behind 2x2 avg and GAP, by hand", s["synops_per_sample"], 4 * 3)
    check_num("Linear behind 2x2 avg and GAP == hs_export", s["synops_per_sample"], want)

    # Conv k3 p1 behind 2x2, on the 4x4 pooled map, 2 output channels. A
    # pooled corner is in 4 output windows, an edge in 6, the inside in 9:
    # (4 + 4 + 9 + 6) x 2. A mean over the map would miss the border.
    conv = nn.Conv2d(1, 2, 3, padding=1, bias=False)
    s = count(nn.Sequential(nn.AvgPool2d(2), conv), x8)
    want = exported({"kind": "conv", "w": ones(2, 1, 3, 3), "pool": 2, "stride": (1, 1),
                     "padding": (1, 1)}, x8, (2, 4, 4))
    check_num("Conv behind 2x2 avg, by hand", s["synops_per_sample"], (4 + 4 + 9 + 6) * 2)
    check_num("Conv behind 2x2 avg == hs_export synapses", s["synops_per_sample"], want)

    # A zero weight is no synapse (unroll_to_hs_api skips it), so pruning
    # cuts SynOps: output 0 loses pooled input 0, which 3 spikes feed.
    lin = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        lin.weight[0, 0] = 0.0
    s = count(nn.Sequential(nn.AvgPool2d(2), nn.Flatten(), lin), x4)
    w = ones(3, 4)
    w[0, 0] = 0
    want = exported({"kind": "linear", "w": w, "pool": 2}, x4, (3,))
    check_num("a pruned weight fires no event, by hand", s["synops_per_sample"], 5 * 3 - 3)
    check_num("a pruned weight == hs_export synapses", s["synops_per_sample"], want)


def chip_events(net, clips, bits=1):
    """What HiAER-Spike does for these clips, independent of SynOpsCounter.

    IntegerSim (simpleSim's arithmetic) fires every chip neuron. Each spike,
    and each input axon spike (one per set bit), is multiplied by the
    synapses hs_export gives that neuron, counted as unroll_to_hs_api stores
    them (a zero weight is not stored), and by the neurons it reaches per
    target unit: both of a signed unit, or the OR neuron of its own sign.

    Returns per chip layer, summed over the clips: "events" its synapses
    receive, "spikes" its neurons fire, and "synapses" it holds with every
    weight nonzero (cost.py's count).
    """
    import numpy as np
    import torch
    from snnsearch.hs_export import IntegerSim, integer_layers, _layer_shapes, _normalize, _synapses
    from snnsearch.train import hardware_flush_steps

    layers = [_normalize(L) for L in integer_layers(net)]
    in_shape = tuple(clips[0].shape[1:])
    shapes = _layer_shapes(layers, in_shape)
    groups = [2 if L["signed"] else 1 for L in layers]
    outdeg, synapses, prev = [], [], in_shape
    for i, (L, shp) in enumerate(zip(layers, shapes)):
        deg = np.zeros(int(np.prod(prev)), dtype=np.int64)
        n = 0
        for pre, _post, w in _synapses(L, prev, shp):
            n += 1
            deg[pre] += int(w) != 0
        outdeg.append(torch.as_tensor(deg, dtype=torch.float64))
        src = bits if i == 0 else groups[i - 1]
        synapses.append(n * groups[i] if L["kind"] == "orpool" else n * src * groups[i])
        prev = shp
    first = [sum(groups[:i]) for i in range(len(layers))]

    events, spikes = [0.0] * len(layers), [0.0] * len(layers)
    flush = hardware_flush_steps(net)
    sim = IntegerSim(layers, in_shape)
    for frames in clips:
        T = frames.shape[0]
        for t, (fired, _outputs) in enumerate(sim.steps(frames.double(), flush)):
            v = (frames[t] if t < T else torch.zeros_like(frames[0])).reshape(-1).to(torch.int64)
            axons = sum(((v >> k) & 1) for k in range(bits)).double()
            for i, L in enumerate(layers):
                reach = 1 if L["kind"] == "orpool" else groups[i]
                src = ([axons] if i == 0 else
                       [fired[first[i - 1] + g][1].reshape(-1) for g in range(groups[i - 1])])
                events[i] += float(sum(float((sp * outdeg[i]).sum()) for sp in src)) * reach
                spikes[i] += float(sum(float(fired[first[i] + g][1].sum())
                                       for g in range(groups[i])))
    return {"events": events, "spikes": spikes, "synapses": synapses,
            "kinds": [L["kind"] for L in layers]}


def matches_the_chip():
    """End to end: the counter on the converted network equals chip_events,
    layer by layer, and cost.py's synapses equal hs_export's.

    A conv behind a 2x2 average pool, a head behind a pool and GAP or
    flatten, OR pools, signed spikes, multi-bit input and pruned weights, on
    the float64 copy every deployed measurement uses (hs_export.exact_copy).
    """
    import torch
    from snnsearch.cost import count_neurons_and_synapses
    from snnsearch.hs_export import exact_copy
    from snnsearch.synops import SynOpsCounter
    from snnsearch.train import forward_over_time

    print("\n-- against the chip: IntegerSim spikes x hs_export synapses --")
    cases = [("avg", "gap", False, 1, False), ("avg", "gap", True, 1, False),
             ("avg", "flatten", False, 1, False), ("avg", "flatten", True, 1, False),
             ("max", "gap", True, 1, False), ("max", "flatten", False, 1, False),
             ("max", "flatten", True, 1, False), ("avg", "gap", False, 3, False),
             ("avg", "gap", True, 1, True)]
    for pool, head, signed, bits, prune in cases:
        tag = (f"{pool}, {head}{', signed' if signed else ''}"
               f"{f', {bits}-bit' if bits > 1 else ''}{', pruned' if prune else ''}:")
        net, spec = deployed_model(pool, head, "bn", signed=signed, bits=bits)
        net = exact_copy(net)
        if prune:
            # half of every weight zero, as magnitude pruning would leave it
            with torch.no_grad():
                for m in net.modules():
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                        m.weight[torch.rand_like(m.weight) < 0.5] = 0.0
        torch.manual_seed(7)
        top = 2 ** bits - 1
        clips = [((torch.rand(6, 2, 16, 16) < 0.3) * torch.randint(1, top + 1, (6, 2, 16, 16))).double()
                 for _ in range(2)]
        c = SynOpsCounter()
        with c.attach(net), torch.no_grad():
            for frames in clips:
                forward_over_time(net, frames.unsqueeze(0))
                c.add_samples(1)
        rows = count_neurons_and_synapses(spec)["rows"]
        s = c.summary(cost_rows=rows)
        ref = chip_events(net, clips, bits)
        n = len(clips)
        layers = s["layers"]
        if not check_true(f"{tag} chip layers in order", [r["kind"] for r in layers] == ref["kinds"],
                          f"{[r['kind'] for r in layers]} vs {ref['kinds']}"):
            continue
        check_true(f"{tag} every layer receives events", all(e > 0 for e in ref["events"]),
                   str(ref["events"]))
        for i, r in enumerate(layers):
            check_num(f"{tag} layer {i} {r['kind']} SynOps == chip",
                      r["synops_per_sample"], ref["events"][i] / n)
            check_num(f"{tag} layer {i} spikes == IntegerSim",
                      r["spikes_per_sample"], ref["spikes"][i] / n)
            check_num(f"{tag} layer {i} cost.py synapses == hs_export",
                      r.get("chip_synapses"), ref["synapses"][i])
        check_num(f"{tag} total SynOps == chip", s["synops_per_sample"], sum(ref["events"]) / n)
        check_num(f"{tag} chip_synapses == hs_export", s.get("chip_synapses"), sum(ref["synapses"]))
        check_true(f"{tag} within the ceiling",
                   s["synops_per_sample"] <= s["synops_ceiling_per_sample"])


def rank2_synapses():
    """Search 05 rank2's chip synapses, from its geometry alone.

    conv0 11 channels at 40x40 (k9 p4), a 2x2 average pool folded into conv1
    (k7, 10 channels at 14x14), conv2 and conv3, signed, flatten head. conv1
    as the chip holds it: 3,920 neurons x 4,312 fan-in (7 x 7 x 11 x 4 pooled
    x 2 signs), 4x the 4,225,760 the cost table used to show. conv0's padded
    taps hold no synapse. hs_export.unroll_to_hs_api on rank2.pth gives the
    same 28,855,640.
    """
    import json
    from snnsearch.config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                                  HeadSpec, NeuronSpec)
    from snnsearch.cost import count_neurons_and_synapses

    print("\n-- Search 05 rank2, as the chip holds it --")
    layers = [{"out_channels": 11, "kernel_size": 9, "stride": 1, "pool": True},
              {"out_channels": 10, "kernel_size": 7, "stride": 1, "pool": False},
              {"out_channels": 121, "kernel_size": 5, "stride": 2, "pool": False},
              {"out_channels": 97, "kernel_size": 3, "stride": 1, "pool": False}]
    spec = {"input": InputSpec(C=2, H=128, W=128, T=48, resize_to=40),
            "encoder": EncoderSpec(layers_json=json.dumps(layers), norm="tdbn"),
            "output": OutputSpec(num_classes=11),
            "downsample": DownsampleSpec(mode="stride", pool_type="avg"),
            "head": HeadSpec(final_reduction="flatten", fc_widths=""),
            "neuron": NeuronSpec(signed_spikes=True)}
    counts = count_neurons_and_synapses(spec)
    rows = {r["layer"]: r for r in counts["rows"]}
    check_num("rank2 conv1 synapses == 3,920 x 4,312", rows["conv1"]["connections"], 3_920 * 4_312)
    check_num("rank2 conv0 synapses, padding excluded", rows["conv0"]["connections"],
              11 * 2 * 340 * 340 * 2)
    check_num("rank2 avg pool row holds no synapse", rows["pool0"]["connections"], 0)
    check_num("rank2 chip synapses == hs_export unrolled", counts["totals"]["connections"],
              28_855_640)
    check_num("rank2 ANN MACs, one pass", counts["totals"]["macs"],
              35_200 // 2 * 2 * 81 + 1_960 * 539 + 3_025 * 250 + 873 * 1_089 + 873 * 11)


def main():
    print("=" * 74)
    print("SynOps arithmetic")
    print("=" * 74)
    dense_macs_arithmetic()
    summary_guards()
    rank2_synapses()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\n  torch absent: skipping the measured cases.")
        print("  Run this on the cluster to exercise the hooks.")
    else:
        with_torch()
        folded_pools_by_hand()
        signed_spikes()
        multi_bit_input()
        matches_the_chip()
    print("\n" + "=" * 74)
    print(f"{passed}/{passed + failed} passed")
    print("SYNOPS TESTS PASSED" if not failed else "SYNOPS TESTS FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
