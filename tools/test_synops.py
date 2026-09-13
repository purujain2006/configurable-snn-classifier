"""Check the SynOps arithmetic against cases with a known answer.

WHY THIS EXISTS

A search reported 73,228,244 SynOps per sample for a network whose spiking
layers hold 3,741,184 outgoing connections driven over 12 passes. The ceiling
is 44,894,208. The measurement was 1.63x an amount that cannot be exceeded, and
it went to a leaderboard as a fact.

Nothing caught it because the number looked plausible and nothing knew what
plausible was. These tests know: for hand-built cases the exact answer is
computable, and the ceiling is checkable for any case at all.

Runs with torch when torch is present, and falls back to checking the pure
arithmetic when it is not, so it is useful on a laptop and on the cluster.

    python tools/test_synops.py
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
    c.dense_macs = {"lin": 48}
    c.ops = {"lin": 48.0}
    rows = [{"layer": "unrelated_cost_name", "connections": 8}]
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
    c.dense_macs["silent"] = 1000
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
               not any((c.ops, c.dense_macs, c.calls, c.spikes,
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
    rows = [{"layer": "lin", "connections": 8}]
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


def main():
    print("=" * 74)
    print("SynOps arithmetic")
    print("=" * 74)
    dense_macs_arithmetic()
    summary_guards()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\n  torch absent: skipping the measured cases.")
        print("  Run this on the cluster to exercise the hooks.")
    else:
        with_torch()
    print("\n" + "=" * 74)
    print(f"{passed}/{passed + failed} passed")
    print("SYNOPS TESTS PASSED" if not failed else "SYNOPS TESTS FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
