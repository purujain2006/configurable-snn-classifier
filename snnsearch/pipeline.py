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


def apply_overrides(obj, values, section):
    """Set dataclass fields from a config dict, rejecting unknown names.

    An unrecognised key is an error rather than a shrug. Silently ignoring
    `chanels: 64` would train the default 128 and report success, and the
    resulting number would be attributed to a configuration that never ran.
    """
    from dataclasses import fields
    if not values:
        return obj
    known = {f.name for f in fields(obj)}
    for key, val in values.items():
        if key not in known:
            raise SystemExit(
                f"architecture.{section}: unknown field {key!r}\n"
                f"  known fields: {', '.join(sorted(known))}")
        setattr(obj, key, val)
    return obj


def specs_from_flat(flat, batch_size=None):
    """Rebuild the full spec dict from a flat trial config.

    best.json stores exactly what Optuna sampled, so reproducing a winning
    trial means replaying that dict rather than transcribing twenty fields into
    YAML by hand and getting one of them wrong.
    """
    from .spaces import config_to_specs
    flat = dict(flat)
    if batch_size:
        flat["N"] = batch_size
    flat.setdefault("epochs", 40)
    flat.setdefault("data_dir", ".")
    return config_to_specs(flat)


def prepare(cfg, flat_config=None):
    """Resolve a run config into (bundle, encoder, loaders, spec dict).

    `flat_config`, when given, supplies the architecture: it is a trial config
    as recorded in best.json, replayed through the same code the search used.
    """
    _require_torch()
    enc_cfg = cfg["encoding"]

    bundle = build_dataset(cfg["dataset"])
    print(f"dataset   : {json.dumps(bundle.describe(), default=str)}")

    # T and resize_to belong to the trial when one is being replayed. The
    # search samples them, so reading them from the config file instead would
    # rebuild the winning architecture at the wrong input size and time depth,
    # which is a different network wearing the winner's name.
    T = int(flat_config["T"]) if flat_config and "T" in flat_config \
        else enc_cfg.get("T", 16)
    resize = (flat_config.get("resize_to") if flat_config
              else enc_cfg.get("resize_to"))
    batch = cfg["search"].get("batch_size", 16)

    # Event data passes through; static data needs a coding to gain a time axis.
    coding = enc_cfg.get("coding") or ("passthrough" if bundle.is_event else "direct")
    encoder = build_encoder(coding, T=T, resize_to=resize)
    print(f"encoding  : {json.dumps(encoder.describe(), default=str)}")

    if flat_config:
        # Replaying a trial: the flat config already fixes the architecture,
        # every neuron parameter and the whole training schedule.
        spec = specs_from_flat(flat_config, batch_size=batch)
        print(f"architecture: replayed from a trial config "
              f"({len(flat_config)} fields)")
    else:
        arch = cfg.get("architecture") or {}
        spec = {
            "encoder": apply_overrides(EncoderSpec(), arch.get("encoder"), "encoder"),
            "downsample": apply_overrides(DownsampleSpec(), arch.get("downsample"),
                                          "downsample"),
            "head": apply_overrides(HeadSpec(), arch.get("head"), "head"),
            "neuron": apply_overrides(NeuronSpec(), arch.get("neuron"), "neuron"),
            "train": apply_overrides(TrainSpec(), cfg.get("train"), "train"),
        }

    # The dataset decides the input shape and the class count, whatever the
    # architecture says, because those are facts about the data.
    spec["input"] = InputSpec(C=bundle.C, H=bundle.H, W=bundle.W,
                              T=T, resize_to=resize or 0, N=batch)
    spec["output"] = OutputSpec(num_classes=bundle.num_classes)
    loaders = build_dataloaders(bundle, batch_size=spec["input"].N, encoder=encoder,
                                num_workers=cfg["search"].get("num_workers", 0),
                                seed=cfg["run"].get("seed", 1))
    return bundle, encoder, loaders, spec


def evaluate_on_test(res, loaders):
    """Score the converted network on the held-out test set.

    DELIBERATELY NOT CALLED FROM THE SEARCH. The search selects on validation,
    and a metric the selection process can see is no longer held out: run it
    per trial and the best trial is partly chosen for fitting the test set,
    which is how a search reports a number it cannot reproduce.

    So this runs once, in `single`, on a configuration already chosen. That is
    also the only number comparable to published results, which quote test
    accuracy. Validation accuracy on a split carved from train is not the same
    quantity and cannot be set beside it.
    """
    from .train import evaluate

    hw_net = res.get("hw_net")
    if hw_net is None or len(loaders) < 3 or loaders[2] is None:
        return {}
    device = next(hw_net.parameters()).device
    was_training = hw_net.training
    hw_net.eval()
    try:
        return {"hw_test_accuracy": evaluate(hw_net, loaders[2], device)}
    except Exception as exc:
        return {"hw_test_accuracy": None, "test_reason": f"{type(exc).__name__}: {exc}"}
    finally:
        hw_net.train(was_training)


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


def load_flat_config(path):
    """Read a trial config out of best.json, or out of a bare flat dict."""
    with open(os.path.abspath(os.path.expanduser(path)), encoding="utf-8") as fh:
        data = json.load(fh)
    flat = data.get("flat_config") or data.get("config") or data
    if not isinstance(flat, dict) or "depth" not in flat:
        raise SystemExit(
            f"{path} does not look like a trial config.\n"
            "  Expected best.json from a search, which carries a 'flat_config'\n"
            "  key holding the sampled hyperparameters.")
    return flat


def run_single(cfg, ckpt="best.pth", from_best=None, epochs=None):
    """Train one configuration, then fold, quantize and audit it.

    `epochs` overrides whatever the config or the replayed trial says. The
    search runs a short budget so that hundreds of trials fit in an evening;
    that budget is a property of the search, not of the network, and the final
    run of a chosen configuration usually wants a longer one.
    """
    from .train import run_training

    flat = load_flat_config(from_best) if from_best else None
    bundle, encoder, loaders, spec = prepare(cfg, flat_config=flat)
    out = results_dir(cfg)
    # A replayed trial brings its own epoch count; an explicit flag still wins.
    if flat is None:
        runconfig.apply_train_overrides(cfg, spec["train"])
    if epochs:
        was = spec["train"].epochs
        spec["train"].epochs = int(epochs)
        # Both the LR schedule and the float/grid split are defined as
        # fractions of the budget, so this lengthens the schedule rather than
        # appending to it. cosine now decays to zero at the new end, and the
        # quantization-aware phase grows with it.
        warm = max(1, round(epochs * spec["train"].qat_warmup_frac))
        print(f"epochs    : {was} -> {epochs}  "
              f"({warm} float warmup, {epochs - warm} on the quantized grid)")

    # Record the trajectory. The search streams this per trial; `single` did
    # not, so a run that ended lower than expected offered two endpoints and no
    # way to tell a plateau from an oscillation from a collapse. `phase` marks
    # which side of the fold each epoch is on, which is the comparison that
    # matters when the float and quantized halves behave differently.
    curve_path = os.path.join(out, "progress.jsonl")
    open(curve_path, "w").close()

    def report_fn(**kw):
        rec = {"epoch": kw.get("epoch"), "phase": kw.get("phase", "float"),
               "val_acc": kw.get("val_acc", kw.get("hw_val_acc")),
               "train_acc": kw.get("train_acc"), "train_loss": kw.get("train_loss"),
               "lr": kw.get("lr"), "best_val_acc": kw.get("best_val_acc")}
        with open(curve_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if rec["epoch"] is not None and rec["val_acc"] is not None:
            print(f"  epoch {rec['epoch']:>3} [{rec['phase']:<5}] "
                  f"val {rec['val_acc']:.4f}  best {rec['best_val_acc'] or 0:.4f}  "
                  f"lr {rec['lr'] or 0:.2e}")

    t0 = time.time()
    res = run_training(spec, loaders=loaders, report_fn=report_fn,
                       ckpt_path=os.path.join(out, ckpt))
    mins = (time.time() - t0) / 60

    # Training is done and the checkpoint is on disk. Everything below is
    # measurement and reporting, and none of it is worth losing an hour of
    # training over: a missing helper in one of these once raised ImportError
    # after a 100-epoch run, discarding every metric at the moment they were
    # about to be written. Each stage now records its own failure and the
    # results file is written regardless.
    for label, fn in (("synops", lambda: measure_run_synops(res, loaders, spec)),
                      ("test", lambda: evaluate_on_test(res, loaders))):
        try:
            res.update(fn())
        except Exception as exc:
            res[f"{label}_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [warn] {label} measurement failed: {type(exc).__name__}: {exc}")
            print("         training is unaffected; the checkpoint is written.")

    payload = {k: v for k, v in res.items() if k != "hw_net"}
    payload.update(wall_minutes=round(mins, 2), config=runconfig.describe(cfg))
    with open(os.path.join(out, "single.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"\nfinished in {mins:.1f} min")
    for k in ("float_val_accuracy", "pre_export_val_accuracy", "hw_val_accuracy",
              "quant_gap", "end_to_end_gain", "synops_per_sample",
              "deployable", "deploy_reasons"):
        if k in payload:
            print(f"  {k:<24} {payload[k]}")
    if payload.get("hw_test_accuracy") is not None:
        print(f"\n  hw_test_accuracy         {payload['hw_test_accuracy']}")
        print("    ^ the held-out test set. This is the number comparable to")
        print("      published results; the validation figures above are not.")

    try:
        _maybe_report(cfg, out)
    except Exception as exc:
        print(f"  [warn] report generation failed: {type(exc).__name__}: {exc}")
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
