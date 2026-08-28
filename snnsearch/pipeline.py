"""Wiring: config -> dataset -> encoder -> model -> train -> report.

This is the layer that used to be implicit in Practice2.py's `main`, where the
DVS dataset was constructed inline. Keeping it separate means `single` and
`search` share one path from config to loaders, so they cannot drift.
"""

import json
import os
import time

from . import runconfig
from .config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                     HeadSpec, NeuronSpec, TrainSpec)
from .data import build_dataset
from .data.loaders import build_dataloaders
from .encoders import build_encoder
from ._torch import _require_torch, torch


def prepare(cfg):
    """Resolve a run config into (bundle, encoder, loaders, spec dict)."""
    _require_torch()
    enc_cfg = cfg["encoding"]

    bundle = build_dataset(cfg["dataset"])
    print(f"dataset   : {json.dumps(bundle.describe(), default=str)}")

    # Event data passes through; static data needs a coding to gain a time axis.
    coding = enc_cfg.get("coding") or ("passthrough" if bundle.is_event else "direct")
    encoder = build_encoder(coding, T=enc_cfg.get("T", 16),
                            resize_to=enc_cfg.get("resize_to"))
    print(f"encoding  : {json.dumps(encoder.describe(), default=str)}")

    resize = enc_cfg.get("resize_to")
    H = W = resize if resize else None
    spec = {
        "input": InputSpec(C=bundle.C, H=bundle.H, W=bundle.W,
                           T=enc_cfg.get("T", 16),
                           resize_to=resize or 0,
                           N=cfg["search"].get("batch_size", 16)),
        "output": OutputSpec(num_classes=bundle.num_classes),
        "encoder": EncoderSpec(),
        "downsample": DownsampleSpec(),
        "head": HeadSpec(),
        "neuron": NeuronSpec(),
        "train": TrainSpec(),
    }
    loaders = build_dataloaders(bundle, batch_size=spec["input"].N, encoder=encoder,
                                num_workers=cfg["search"].get("num_workers", 0),
                                seed=cfg["run"].get("seed", 1))
    return bundle, encoder, loaders, spec


def measure_run_synops(res, loaders, spec, max_batches=None):
    """SynOps for a finished run, from the converted network it produced.

    `single` and each search trial both need this, and both already hold the
    same three things, so it lives here rather than in either caller. Returns a
    dict to merge into the result, empty when there is no converted network to
    measure.
    """
    from .synops import measure_synops

    hw_net = res.get("hw_net")
    if hw_net is None:
        return {}
    device = next(hw_net.parameters()).device
    summary = measure_synops(hw_net, loaders[1], device, spec=spec,
                             max_batches=max_batches)
    out = {"synops_per_sample": summary.get("synops_per_sample"),
           "spikes_per_sample": summary.get("spikes_per_sample")}
    # The dense comparison is the argument for spiking at all, so keep it when
    # the plan could be costed.
    for k in ("dense_macs_per_inference", "synops_over_dense"):
        if k in summary:
            out[k] = summary[k]
    if summary.get("reason"):
        out["synops_reason"] = summary["reason"]
    return out


def results_dir(cfg):
    d = cfg["run"].get("results_dir") or os.path.join(
        "results", cfg["run"].get("name", "run"))
    os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)


def run_single(cfg, ckpt="best.pth"):
    """Train one configuration, then fold, quantize and audit it."""
    from .train import run_training

    bundle, encoder, loaders, spec = prepare(cfg)
    out = results_dir(cfg)
    runconfig.apply_train_overrides(cfg, spec["train"])

    t0 = time.time()
    res = run_training(spec, loaders=loaders, ckpt_path=os.path.join(out, ckpt))
    mins = (time.time() - t0) / 60

    res.update(measure_run_synops(res, loaders, spec))

    payload = {k: v for k, v in res.items() if k != "hw_net"}
    payload.update(wall_minutes=round(mins, 2), config=runconfig.describe(cfg))
    with open(os.path.join(out, "single.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"\nfinished in {mins:.1f} min")
    for k in ("float_val_accuracy", "hw_val_accuracy", "quant_gap",
              "synops_per_sample", "deployable", "deploy_reasons"):
        if k in payload:
            print(f"  {k:<20} {payload[k]}")

    _maybe_report(cfg, out)
    return 0


def run_search_mode(cfg):
    """The automated search, then the report."""
    from .search import run_search

    out = results_dir(cfg)
    run_search(cfg, out)
    _maybe_report(cfg, out)
    return 0


def _maybe_report(cfg, out):
    if not cfg.get("report", {}).get("html", True):
        return
    try:
        from . import report
        path = report.build(out, cfg=cfg)
        print(f"\nreport: {path}")
    except Exception as exc:                 # a broken report must not lose a run
        print(f"\n[warn] report generation failed: {type(exc).__name__}: {exc}")
        print("       the raw result files are intact in", out)
