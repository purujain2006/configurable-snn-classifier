"""Run the whole pipeline on synthetic data, in under a minute, on a CPU.

WHY THIS EXISTS

`summary` and `check` import no deep-learning library, so they pass on a
machine where every torch-dependent path is broken. That happened: splitting
Practice2.py into modules left 28 names undefined and four calls with the wrong
signature, and none of it was visible until a training run reached the line.

Static checks (tools/check_static.py) catch a missing name. They cannot catch a
tensor of the wrong shape, a device mismatch, or a fold that silently produces
a different network. Only running it does that. So this drives every stage:

    build model -> train -> fold BN -> quantize -> audit -> export -> report

on 48 synthetic samples with T=2 and one epoch. It asserts on structure rather
than accuracy, because a model that saw 48 samples has not learned anything and
a threshold on accuracy would only produce a flaky test.

    python tools/smoke_test.py
    python tools/smoke_test.py --keep     # leave the output directory behind
"""
import os
import shutil
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL = "  ok  ", " FAIL "
results = []


def check(name, fn):
    """Run one stage. A failure is recorded and reported, never raised."""
    try:
        detail = fn()
        results.append(True)
        print(f"[{PASS}] {name:<34} {detail or ''}")
        return True
    except Exception as exc:
        results.append(False)
        print(f"[{FAIL}] {name:<34} {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=6)
        return False


def main():
    keep = "--keep" in sys.argv
    out = tempfile.mkdtemp(prefix="snnsearch_smoke_")

    print("=" * 74)
    print("snnsearch smoke test: the whole pipeline on synthetic data")
    print("=" * 74)
    print(f"output: {out}\n")

    try:
        import torch
    except ImportError:
        sys.exit("torch is required for the smoke test.\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cu121")

    from snnsearch import runconfig
    from snnsearch.pipeline import prepare, run_single

    cfg = runconfig.load(overrides={
        "run": {"name": "smoke", "results_dir": out, "seed": 0},
        "dataset": {"name": None,
                    "module": os.path.join(ROOT, "examples", "synthetic_data.py"),
                    "factory": "make_datasets"},
        # T=2 keeps the time loop honest (a single step would hide the pipeline
        # flush) while staying fast.
        "encoding": {"coding": "direct", "T": 2, "resize_to": None},
        "search": {"epochs": 1, "batch_size": 8},
        "objective": {"mode": "accuracy"},
        "report": {"html": True},
    })

    state = {}

    def build():
        bundle, encoder, loaders, spec = prepare(cfg)
        state.update(bundle=bundle, encoder=encoder, loaders=loaders, spec=spec)
        tr, va, te = loaders
        return (f"{bundle.name} {bundle.C}x{bundle.H}x{bundle.W} "
                f"-> {len(tr)}/{len(va)}/{len(te)} batches")

    def one_batch():
        # Three values, always. forward_over_time documents its input as
        # (N, T, C, H, W) and transposes internally, so time sits at dim 1.
        x, y, lengths = next(iter(state["loaders"][0]))
        assert x.ndim == 5, f"expected (N,T,C,H,W), got {tuple(x.shape)}"
        assert x.shape[1] == 2, f"time axis should be T=2, got {x.shape[1]}"
        assert len(lengths) == x.shape[0], "lengths must have one entry per sample"
        return f"batch {tuple(x.shape)}  labels {tuple(y.shape)}"

    def model():
        from snnsearch.model import build_model
        net = build_model(state["spec"])
        n = sum(p.numel() for p in net.parameters())
        state["net"] = net
        return f"{n:,} parameters"

    def forward():
        import torch
        from snnsearch.train import forward_over_time
        from snnsearch._torch import functional
        x, y, _lengths = next(iter(state["loaders"][0]))
        net = state["net"].eval()
        functional.reset_net(net)
        with torch.no_grad():
            o = forward_over_time(net, x)
        # x is (N, T, C, H, W), so N is dim 0. Time is summed away.
        assert o.shape == (x.shape[0], state["bundle"].num_classes), \
            f"logits should be (N, classes), got {tuple(o.shape)}"
        assert torch.isfinite(o).all(), "logits contain NaN or inf"
        return f"logits {tuple(o.shape)}"

    def full_run():
        rc = run_single(cfg, ckpt="smoke.pth")
        assert rc == 0, f"run_single returned {rc}"
        return "train -> fold -> quantize -> audit finished"

    def payload():
        import json
        with open(os.path.join(out, "single.json"), encoding="utf-8") as fh:
            p = json.load(fh)
        for k in ("float_val_accuracy", "hw_val_accuracy", "deployable",
                  "synops_per_sample", "hw_test_accuracy", "quant_gap",
                  "end_to_end_gain"):
            assert k in p, f"single.json is missing {k}"
        # quant_gap compares the same weights before and after export, so it
        # cannot be negative by more than float noise. A large negative value
        # means it has drifted back to comparing different training phases.
        if p["quant_gap"] is not None:
            assert p["quant_gap"] > -0.05, \
                f"quant_gap {p['quant_gap']} is negative; it is comparing the wrong pair"
        state["payload"] = p
        return (f"hw={p['hw_val_accuracy']:.3f} "
                f"test={p['hw_test_accuracy']:.3f} "
                f"export cost={p['quant_gap']} "
                f"synops={p['synops_per_sample']:,.0f}")

    def report():
        path = os.path.join(out, "report.html")
        assert os.path.isfile(path), "report.html was not written"
        size = os.path.getsize(path)
        assert size > 2000, f"report.html is suspiciously small ({size} bytes)"
        return f"report.html {size:,} bytes"

    check("dataset + loaders", build)
    check("one batch has a time axis", one_batch)
    check("model builds", model)
    check("forward over time", forward)
    check("full single run", full_run)
    check("result payload is complete", payload)
    check("html report written", report)

    print("\n" + "=" * 74)
    ok = all(results)
    print(f"{sum(results)}/{len(results)} stages passed")
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    if keep:
        print(f"\noutput kept at {out}")
    else:
        shutil.rmtree(out, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
