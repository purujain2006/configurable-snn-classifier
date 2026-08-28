"""The entry point. `python main.py <mode> [--config file] [overrides]`

Five modes, ordered by what they cost:

  check     what this machine is missing. No torch needed.
  summary   architecture table, connection limits, cost. No torch, no dataset.
  single    train one configuration end to end, then fold and export.
  search    the automated search.
  report    rebuild report.html from a finished run directory.

The config file supplies values; command-line flags override it, so a saved
config stays the record of intent while a flag can vary one thing.
"""

import argparse
import json
import os
import sys

from . import runconfig
from ._torch import _HAS_TORCH, _HAS_SPIKINGJELLY, _HAS_HS_API


def _overrides(args):
    """Turn the handful of shared flags into a config-shaped dict."""
    o = {"run": {}, "dataset": {}, "encoding": {}, "search": {}, "objective": {}}
    if getattr(args, "data_root", None):     o["dataset"]["root"] = args.data_root
    if getattr(args, "dataset", None):       o["dataset"]["name"] = args.dataset
    if getattr(args, "coding", None):        o["encoding"]["coding"] = args.coding
    if getattr(args, "T", None):             o["encoding"]["T"] = args.T
    if getattr(args, "resize_to", None) is not None:
        o["encoding"]["resize_to"] = args.resize_to
    if getattr(args, "trials", None):        o["search"]["trials"] = args.trials
    if getattr(args, "epochs", None):        o["search"]["epochs"] = args.epochs
    if getattr(args, "gpu_fraction", None):  o["search"]["gpu_fraction"] = args.gpu_fraction
    if getattr(args, "results_dir", None):   o["run"]["results_dir"] = args.results_dir
    if getattr(args, "objective", None):     o["objective"]["mode"] = args.objective
    if getattr(args, "synops_budget", None): o["objective"]["synops_budget"] = args.synops_budget
    return {k: v for k, v in o.items() if v}


def _add_common(p):
    p.add_argument("--config", "-c", help="YAML config file")
    p.add_argument("--data-root", help="override dataset.root")
    p.add_argument("--dataset", help="override dataset.name")
    p.add_argument("--coding", choices=["direct", "poisson", "temporal", "passthrough"])
    p.add_argument("--T", type=int, help="timesteps")
    p.add_argument("--resize-to", type=int, dest="resize_to")
    p.add_argument("--results-dir")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="snnsearch",
        description="Hyperparameter search for spiking networks under hardware limits.")
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("check", help="report what this machine is missing")
    c.add_argument("--data-root", nargs="?")

    s = sub.add_parser("summary", help="architecture, limits and cost. No torch needed.")
    _add_common(s)
    s.add_argument("--depth", type=int, default=2)
    s.add_argument("--channels", type=int, default=32)
    s.add_argument("--kernel-size", type=int, default=7)
    s.add_argument("--stride", type=int, default=2)

    o = sub.add_parser("single", help="train one configuration end to end")
    _add_common(o)
    o.add_argument("--epochs", type=int)
    o.add_argument("--ckpt", default="best.pth")

    r = sub.add_parser("search", help="the automated search")
    _add_common(r)
    r.add_argument("--trials", type=int)
    r.add_argument("--epochs", type=int)
    r.add_argument("--gpu-fraction", type=float, dest="gpu_fraction",
                   help="GPU per trial; 0.25 runs four trials on one card")
    r.add_argument("--objective", choices=["accuracy", "constrained", "weighted", "pareto"])
    r.add_argument("--synops-budget", type=float, dest="synops_budget")

    p = sub.add_parser("report", help="rebuild report.html from a run directory")
    p.add_argument("run_dir")

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.mode == "check":
        return _check(getattr(args, "data_root", None))

    if args.mode == "report":
        from . import report
        path = report.build(args.run_dir)
        print(f"wrote {path}")
        return 0

    cfg = runconfig.load(getattr(args, "config", None), _overrides(args))
    print(runconfig.describe(cfg), "\n")

    if args.mode == "summary":
        return _summary(cfg, args)

    # everything below needs the full stack
    from ._torch import _require_torch
    _require_torch()

    if args.mode == "single":
        from .pipeline import run_single
        return run_single(cfg, ckpt=args.ckpt)

    if args.mode == "search":
        from .pipeline import run_search_mode
        return run_search_mode(cfg)

    return 1


def _summary(cfg, args):
    """Plan and cost a configuration without importing torch."""
    from .config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                         HeadSpec, NeuronSpec, TrainSpec)
    from .cost import format_summary
    from .data.base import _REGISTRY

    enc = cfg["encoding"]
    ds = cfg["dataset"]
    # Shape comes from the dataset when it is known without loading it, so
    # `summary` still works with no data on disk.
    C, H, W, ncls = _shape_hint(ds.get("name"))

    spec = {
        "input": InputSpec(C=C, H=H, W=W, T=enc.get("T", 16),
                           resize_to=enc.get("resize_to") or 0),
        "encoder": EncoderSpec(depth=args.depth, channels=args.channels,
                               kernel_size=args.kernel_size, stride=args.stride),
        "output": OutputSpec(num_classes=ncls),
        "downsample": DownsampleSpec(),
        "head": HeadSpec(fc_widths=""),
        "neuron": NeuronSpec(),
        "train": TrainSpec(),
    }
    print(format_summary(spec))
    return 0


def _shape_hint(name):
    """Input shape for the built-in datasets, without importing torchvision."""
    table = {
        "dvs128": (2, 128, 128, 11),
        "cifar10": (3, 32, 32, 10),
        "cifar100": (3, 32, 32, 100),
        "mnist": (1, 28, 28, 10),
        "fashion_mnist": (1, 28, 28, 10),
    }
    return table.get(name, (2, 128, 128, 11))


def _check(data_root=None):
    print("=" * 70)
    print("snnsearch environment")
    print("=" * 70)
    v = sys.version_info
    print(f"[{'ok  ' if v >= (3, 9) else 'MISS'}] python        {v.major}.{v.minor}.{v.micro} (needs >= 3.9)")
    print(f"        {sys.executable}")
    for label, present, hint in [
        ("torch", _HAS_TORCH, "pip install torch --index-url https://download.pytorch.org/whl/cu121"),
        ("spikingjelly", _HAS_SPIKINGJELLY, "pip install spikingjelly   (AFTER torch)"),
        ("hs_api", _HAS_HS_API, "optional; the arithmetic is mirrored without it"),
    ]:
        mark = "ok  " if present else ("warn" if label == "hs_api" else "MISS")
        print(f"[{mark}] {label:<13} {'' if present else hint}")
    for mod in ("ray", "optuna", "yaml"):
        try:
            __import__(mod)
            print(f"[ok  ] {mod:<13}")
        except ImportError:
            need = "search mode" if mod in ("ray", "optuna") else "richer configs (a fallback parser is built in)"
            print(f"[{'MISS' if mod != 'yaml' else 'warn'}] {mod:<13} needed for {need}")
    if _HAS_TORCH:
        import torch
        if torch.cuda.is_available():
            print(f"[ok  ] CUDA          {torch.cuda.get_device_name(0)}")
        else:
            print("[warn] CUDA          not available; training runs T steps per sample, so CPU is slow")
    if data_root:
        ok = os.path.isdir(os.path.expanduser(data_root))
        print(f"[{'ok  ' if ok else 'MISS'}] dataset root  {data_root}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
