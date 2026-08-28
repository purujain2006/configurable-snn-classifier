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

    print("\n-- the ceiling guard --")
    rows = [{"layer": "lin", "neurons": 512, "connections": 524_288}]
    c = SynOpsCounter()
    c.samples = 1
    c.ops = {"lin": 524_288.0 * 5}      # five passes' worth
    c.calls = {"lin": 2}                # but only two passes happened
    s = c.summary(cost_rows=rows)
    check_true("an impossible total is refused, not reported",
               s["synops_per_sample"] is None, s.get("reason", "")[:70])

    c.ops = {"lin": 524_288.0 * 2}      # exactly two passes: legal
    c.calls = {"lin": 2}
    s = c.summary(cost_rows=rows)
    check_true("a total at the ceiling is accepted",
               s["synops_per_sample"] is not None,
               f"{s.get('synops_per_sample', 0):,.0f} SynOps/sample")


def main():
    print("=" * 74)
    print("SynOps arithmetic")
    print("=" * 74)
    dense_macs_arithmetic()
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
