"""Wiring: config -> dataset -> encoder -> model -> train -> report.

This is the layer that used to be implicit in Practice2.py's `main`, where the
DVS dataset was constructed inline. Keeping it separate means `single` and
`search` share one path from config to loaders, so they cannot drift.
"""

import json
import os
import time
from dataclasses import asdict

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


# TrainSpec fields that spaces.config_to_specs fills from a sampled trial
# config. A config file must not be able to overwrite these, or a leaderboard
# row would describe a learning rate the run did not use, which is the same
# class of failure as recording an input size a trial did not train at.
#
# Everything else in TrainSpec (momentum, step_gamma, qat_schedule_epochs,
# qat_scheduler) is unsampled and therefore settable from `train:`.
_SEARCH_OWNED_TRAIN_FIELDS = frozenset({
    "epochs", "optimizer", "lr", "weight_decay", "scheduler", "warmup_epochs",
    "label_smoothing", "grad_clip", "rate_penalty", "qat_mode",
    "qat_warmup_frac", "qat_epochs", "qat_lr_scale",
    "fold_bias_mode", "fold_bias_margin",
})


def coerce_to_field(obj, key, raw):
    """Turn a command-line string into the type the dataclass field declares.

    Reading the annotation beats guessing: `scheduler=none` must stay the
    string "none" while `weight_decay=1e-3` must become a float, and no
    heuristic over the text alone gets both right.
    """
    from dataclasses import fields
    known = {f.name: f for f in fields(obj)}
    if key not in known:
        raise SystemExit(
            f"--train: unknown field {key!r}\n"
            f"  known fields: {', '.join(sorted(known))}")
    if not isinstance(raw, str):
        return raw
    ann = known[key].type
    ann = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
    try:
        if "bool" in ann:
            return raw.lower() in ("1", "true", "yes", "on")
        if "int" in ann and "float" not in ann:
            return int(float(raw))          # so 40.0 and 40 both work
        if "float" in ann:
            return float(raw)
    except ValueError:
        raise SystemExit(f"--train: {key}={raw!r} is not a valid {ann}")
    return raw


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


def resolve_specs(cfg, flat_config=None, quiet=False):
    """Resolve a run config into (bundle, encoder, spec dict). No loaders yet.

    `flat_config`, when given, supplies the architecture AND the input shape: it
    is a trial config, either one recorded in best.json or the one Optuna just
    sampled. Both callers go through here, which is the point.

    SEPARATE FROM LOADER CONSTRUCTION because the connection-limit check needs
    the final input spec and should run before anything expensive. Checking it
    against a T and resize_to that the loaders then contradict validates a
    network nobody trains.
    """
    _require_torch()
    enc_cfg = cfg["encoding"]

    # T and resize_to belong to the trial whenever there is one. The search
    # samples them, so reading them from the config file instead builds the
    # architecture at the wrong input size and time depth, which is a different
    # network wearing the trial's name.
    # A trial that did not sample one of them falls back to the file, rather
    # than to nothing: an absent resize_to means native resolution, which on
    # DVS128 is 128x128 and twice the axon limit.
    flat = flat_config or {}
    T = int(flat["T"]) if flat.get("T") else int(enc_cfg.get("T", 16))
    resize = flat["resize_to"] if flat.get("resize_to") is not None \
        else enc_cfg.get("resize_to")
    batch = cfg["search"].get("batch_size", 16)

    # T reaches the DATASET, not only the encoder. Event data is cached as a
    # fixed number of frames per clip, so the frame count is decided when the
    # dataset is built. Passing it only to the encoder leaves every run on
    # whichever cache the dataset defaults to, and the sampled T becomes a label
    # on a run that ignored it.
    bundle = build_dataset({**cfg["dataset"], "T": T})
    if not quiet:
        print(f"dataset   : {json.dumps(bundle.describe(), default=str)}")

    # Event data passes through; static data needs a coding to gain a time axis.
    coding = enc_cfg.get("coding") or ("passthrough" if bundle.is_event else "direct")
    encoder = build_encoder(coding, T=T, resize_to=resize or None)
    if not quiet:
        print(f"encoding  : {json.dumps(encoder.describe(), default=str)}")

    if flat_config:
        # The flat config already fixes the architecture, every neuron
        # parameter and the whole training schedule.
        # The flat builder requires both input keys even when this trial did
        # not sample them (or a replay removed them for a CLI override).
        spec = specs_from_flat({**flat_config, "T": T, "resize_to": resize or 0},
                               batch_size=batch)
        # ...except the TrainSpec fields the search never samples. Those would
        # otherwise sit at their dataclass defaults with no way to reach them,
        # so a `train:` block in the config file is silently inert for every
        # trial and every replay. qat_schedule_epochs is the one that matters:
        # without it the quantized phase's cosine stretches across whatever the
        # epoch budget leaves over, which is what cost 14 points between a
        # 40-epoch run and a 100-epoch one. A long run needs it set.
        #
        # Only fields the sampler did not choose, so a config file can never
        # overwrite a trial's own decision and quietly change what it means.
        for key, val in (cfg.get("train") or {}).items():
            if key in _SEARCH_OWNED_TRAIN_FIELDS:
                if not quiet:
                    print(f"train     : ignoring {key}={val!r} from the config "
                          "file; the trial sampled it")
                continue
            apply_overrides(spec["train"], {key: val}, "train")
        if not quiet:
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

    # The dataset decides the channel count and the class count, because those
    # are facts about the data. T and resize_to are not: they are choices, and
    # they belong to whoever made them.
    spec["input"] = InputSpec(C=bundle.C, H=bundle.H, W=bundle.W,
                              T=T, resize_to=resize or 0, N=batch)
    spec["output"] = OutputSpec(num_classes=bundle.num_classes)
    return bundle, encoder, spec


def make_loaders(cfg, bundle, encoder, spec, quiet=False):
    """Build the three dataloaders for an already-resolved spec."""
    workers = cfg["search"].get("num_workers", 0)
    loaders = build_dataloaders(bundle, batch_size=spec["input"].N, encoder=encoder,
                                num_workers=workers,
                                seed=cfg["run"].get("seed", 1))
    # A trial and its replay must see the same number of gradient steps per
    # epoch. If these differ the two are not the same experiment, whatever the
    # configs say, and comparing their curves is meaningless.
    if not quiet:
        print(f"loaders   : {len(loaders[0])} train / {len(loaders[1])} val / "
              f"{len(loaders[2])} test batches, batch={spec['input'].N}, "
              f"T={spec['input'].T}, resize={spec['input'].resize_to or 'none'}, "
              f"workers={workers}")
    return loaders


def prepare(cfg, flat_config=None):
    """Resolve a run config into (bundle, encoder, loaders, spec dict)."""
    bundle, encoder, spec = resolve_specs(cfg, flat_config)
    loaders = make_loaders(cfg, bundle, encoder, spec)
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


def run_single(cfg, ckpt="best.pth", from_best=None, epochs=None,
               input_overrides=None, train_overrides=None):
    """Train one configuration, then fold, quantize and audit it.

    `epochs` overrides whatever the config or the replayed trial says. The
    search runs a short budget so that hundreds of trials fit in an evening;
    that budget is a property of the search, not of the network, and the final
    run of a chosen configuration usually wants a longer one.

    `input_overrides` holds T and resize_to when they were typed on the command
    line, and they beat the replayed trial. A record can be wrong about what its
    run actually did, and reproducing the run then means contradicting the
    record on purpose, so that has to be expressible.

    `train_overrides` is the same idea for the optimization recipe, and it
    outranks everything: the config file, and the replayed trial's own sampled
    values. A deliberate experiment is exactly a replay with one thing changed,
    so blocking that would leave no way to ask whether weight decay is what the
    result depends on.
    """
    from .train import run_training

    flat = load_flat_config(from_best) if from_best else None
    for key, val in (input_overrides or {}).items():
        if val is None or not flat:
            continue
        if flat.pop(key, None) is not None:
            # cfg["encoding"] already carries the typed value, so dropping the
            # recorded one lets it through.
            print(f"override  : {key}={val} from the command line, "
                  f"replacing the value recorded in {os.path.basename(from_best)}")
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

    # Last word, after the config file and after the replayed trial. Applied
    # here rather than inside prepare so the ordering is visible: whatever a
    # record says, this is what the run used, and it says so on stdout.
    for key, raw in (train_overrides or {}).items():
        was = getattr(spec["train"], key, None)
        val = coerce_to_field(spec["train"], key, raw)
        setattr(spec["train"], key, val)
        print(f"train     : {key} {was!r} -> {val!r}  (command line)")

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
               "lr": kw.get("lr"), "best_val_acc": kw.get("best_val_acc"),
               "firing_rate": kw.get("firing_rate")}
        with open(curve_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if rec["epoch"] is not None and rec["val_acc"] is not None:
            # train_acc and loss are printed because the search records them and
            # `single` did not. Comparing a trial against its replay on val
            # alone cannot separate "trained differently" from "scored
            # differently", and that was exactly the question.
            print(f"  epoch {rec['epoch']:>3} [{rec['phase']:<5}] "
                  f"val {rec['val_acc']:.4f}  best {rec['best_val_acc'] or 0:.4f}  "
                  f"train {rec['train_acc'] or 0:.4f}  "
                  f"loss {rec['train_loss'] or 0:.4f}  lr {rec['lr'] or 0:.2e}"
                  + (f"  rate {rec['firing_rate']:.4f}"
                     if rec["firing_rate"] is not None else ""))

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
    # `describe(cfg)` summarizes the CONFIG FILE, which on a --from-best run is
    # not what trained: the trial supplies the architecture, the input size and
    # the schedule. It reported resize_to=64 and space=uniform for runs that
    # used 48 and per-layer geometry. Nothing was mis-trained, the real spec
    # was always in the checkpoint, but single.json is what a person reads, and
    # a results file that names the wrong input size is how the last bug
    # travelled.
    payload.update(
        wall_minutes=round(mins, 2),
        config_file=runconfig.describe(cfg),
        replayed_from=from_best,
        config={name: asdict(s) for name, s in spec.items()},
    )
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
