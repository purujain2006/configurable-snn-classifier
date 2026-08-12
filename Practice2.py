"""
Configurable SNN classifier for DVS Gesture, targeting HiAER-Spike.

Goal: depth, width, downsample strategy, head shape, neuron type, and the
optimization recipe are all parameters, so the same code is reusable across
experiments -- and a hyperparameter search (Optuna, orchestrated by Ray Tune)
can explore the space of configs that are both HARDWARE-FEASIBLE (fit
HiAER-Spike's axon/fan-in/fan-out limits) and hit a validation-accuracy
target (>97.5%).

Run modes:
  python Practice.py summary --input ... --encoder ...              (architecture table + feasibility + neuron/synapse counts; no torch needed)
  python Practice.py single  --data-dir ... --encoder ...           (one training run, laptop or single GPU)
  python Practice.py search  --compute local  --trials 50 ...       (Optuna search, single machine)
  python Practice.py search  --compute cluster --trials 200 ...     (Optuna search, Ray cluster, multi-GPU/node)

Layering note: sections 1-4 (config, shape planning, feasibility, cost model)
are deliberately free of torch/spikingjelly imports, so `summary` runs anywhere
and the shape math can be unit-tested without a GPU stack installed.
"""

import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import sys
import threading
import time
import warnings
from copy import deepcopy
from typing import Callable, Optional
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    from spikingjelly.datasets import pad_sequence_collate
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from spikingjelly.activation_based import neuron, functional, surrogate, layer

    _HAS_TORCH = True
    _TORCH_IMPORT_ERROR = None
except ImportError as _err:                      # summary/feasibility still work
    _HAS_TORCH = False
    _TORCH_IMPORT_ERROR = _err
    torch = None

    class _NoTorchModule:
        """Placeholder base class so this module still imports without torch."""

    class nn:                                    # noqa: N801 - shim, not a real class
        Module = _NoTorchModule


try:
    from hs_api.custom_neurons import Custom_LIFNode, Custom_IFNode
    _HAS_HS_API = True
except ImportError:
    _HAS_HS_API = False


def _no_grad(fn):
    """`@torch.no_grad()` that survives being imported without torch.

    DVSGesturePuru is defined at module level, so a bare `@torch.no_grad()` on
    one of its methods is evaluated at import time and raises when torch is the
    None placeholder above. That broke `summary`, which the module docstring
    promises runs without torch. The methods themselves only ever run with torch
    present, so degrading to an identity decorator is safe.
    """
    return torch.no_grad()(fn) if _HAS_TORCH else fn


def _require_torch():
    if not _HAS_TORCH:
        raise RuntimeError(
            "This action needs torch + spikingjelly, which failed to import "
            f"({_TORCH_IMPORT_ERROR}). The `summary` subcommand works without them."
        )


# =============================================================================
# 1. Config dataclasses
# =============================================================================

MAX_FC_HIDDEN = 3          # search never proposes more hidden FC layers than this


@dataclass
class InputSpec:
    N: int = 16
    C: int = 2
    H: int = 128
    W: int = 128
    T: int = 16
    resize_to: int = 0   # 0 = no resize, feed native H,W straight through


def effective_hw(input_cfg: "InputSpec") -> tuple[int, int]:
    if input_cfg.resize_to:
        return input_cfg.resize_to, input_cfg.resize_to
    return input_cfg.H, input_cfg.W


@dataclass
class ConvLayerSpec:
    """One convolutional layer. Every geometric knob is per-layer, so the search
    (and a user hand-designing a net) can vary kernel size, stride, channel count
    and pooling independently across depth."""
    out_channels: int = 128
    kernel_size: int = 3
    stride: int = 2
    padding: int = 0
    dilation: int = 1
    pool: bool = False          # per-layer downsample-by-pooling flag
    # per-layer neuron overrides; None -> fall back to the global NeuronSpec.
    # per-LAYER is supported (and often helps); per-CHANNEL is deliberately not
    # offered -- it is usually too many degrees of freedom for this dataset.
    tau: "float | None" = None
    v_threshold: "float | None" = None


@dataclass
class EncoderSpec:
    """
    Two ways to specify the encoder:

      * UNIFORM (back-compat): set depth/channels/kernel_size/stride and every
        layer is built identically. This is what the old flat config did.
      * PER-LAYER: set layers_json to a JSON list of ConvLayerSpec dicts, e.g.
        '[{"out_channels":64,"kernel_size":3,"stride":2},
          {"out_channels":128,"kernel_size":3,"stride":1,"pool":true}]'
        When present, layers_json WINS and the uniform fields are ignored for
        shape. This is what the per-layer search emits.

    `resolve_conv_layers` collapses either form into a list[ConvLayerSpec].
    """
    # uniform fallback
    depth: int = 3
    channels: int = 128
    kernel_size: int = 3
    stride: int = 2
    padding: int = 0
    dilation: int = 1
    # per-layer override (JSON list of dicts); empty string -> use uniform
    layers_json: str = ""
    # encoder-wide (not per-layer)
    bias: bool = False
    norm: str = "bn"          # "none" | "bn" | "tdbn"
    tdbn_alpha: float = 1.0   # tdbn only: gamma is initialised to alpha * v_threshold


def resolve_conv_layers(encoder_cfg: EncoderSpec) -> list[ConvLayerSpec]:
    """EncoderSpec -> concrete per-layer list, from layers_json if given else uniform."""
    import json
    if encoder_cfg.layers_json:
        raw = json.loads(encoder_cfg.layers_json)
        if not isinstance(raw, list) or not raw:
            raise ValueError("layers_json must be a non-empty JSON list of layer dicts")
        valid = {f.name for f in fields(ConvLayerSpec)}
        layers = []
        for i, d in enumerate(raw):
            unknown = set(d) - valid
            if unknown:
                raise ValueError(f"layer {i}: unknown keys {unknown}. Valid: {sorted(valid)}")
            layers.append(ConvLayerSpec(**d))
        return layers
    # uniform broadcast
    return [ConvLayerSpec(out_channels=encoder_cfg.channels, kernel_size=encoder_cfg.kernel_size,
                          stride=encoder_cfg.stride, padding=encoder_cfg.padding,
                          dilation=encoder_cfg.dilation, pool=False)
            for _ in range(encoder_cfg.depth)]


@dataclass
class OutputSpec:
    num_classes: int = 11


@dataclass
class DownsampleSpec:
    mode: str = "stride"        # "stride" or "pool"
    pool_type: str = "avg"      # "max" or "avg"
    pool_kernel_size: int = 2
    pool_stride: int = 2


@dataclass
class HeadSpec:
    final_reduction: str = "flatten"   # "flatten" or "gap"
    fc_widths: str = "512"             # comma-separated HIDDEN widths, e.g. "512" or "512,256"
                                       # "" means go straight from the encoder to the classifier
    dropout_rate: float = 0.5


@dataclass
class NeuronSpec:
    neuron_type: str = "LIF"
    # tau is an INTEGER. HiAER-Spike's LIF_neuron takes an integer `leak`
    # register, and the converter's convention is leak == round(tau). A
    # continuous tau in [1.5, 2.5] therefore deploys as leak=2 for EVERY value
    # in the range: the dimension exists in training and vanishes on chip. See
    # HW_TAU_CHOICES.
    tau: int = 2
    v_threshold: float = 1.0
    v_reset: float = 0.0
    # Q2/Q3: promote tau / threshold from fixed values to learned parameters.
    # Granularity is PER-LAYER (one learned scalar per conv/fc neuron group),
    # never per-channel -- see ConvLayerSpec note. Both are deployable:
    # hs_api reads a neuron model PER NEURON KEY, so heterogeneous thresholds
    # and leaks across layers cost nothing on hardware.
    trainable_tau: bool = False
    trainable_threshold: bool = False


@dataclass
class TrainSpec:
    """Optimization recipe -- everything that affects training but not shape."""
    epochs: int = 20
    optimizer: str = "adam"        # "adam" | "adamw" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9          # sgd only
    scheduler: str = "cosine"      # "cosine" | "step" | "onecycle" | "none"
    step_gamma: float = 0.2        # step only
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    grad_clip: float = 0.0         # 0 = off
    # --- deployment / quantization-aware training (section 5c) -------------
    # qat_mode selects how the model reaches the deployable INT16 grid:
    #   "inline"  (Option A, default): after a short float warmup, every
    #             (conv, BN) pair folds INLINE and the REST of the run trains on
    #             the folded + quantized grid. The neuron sees the true deployed
    #             weight every step, so the network never builds the precise-
    #             weight balances that post-hoc quantization would shatter.
    #   "tail"    (Option B / older): full float run, then a fixed qat_epochs of
    #             fold+quantize fine-tuning bolted on the end.
    #   "ptq"     : no QAT at all -- fold + quantize once and measure (baseline).
    qat_mode: str = "inline"
    # inline mode: fraction of the TOTAL epoch budget spent in float warmup
    # before folding. BN needs a few real epochs before running_var is worth
    # folding from; the rest of the budget is on-grid training.
    qat_warmup_frac: float = 0.25
    qat_epochs: int = 4            # tail mode only
    qat_lr_scale: float = 0.5      # grid-phase lr = qat_lr_scale * lr
    # "threshold" is the only mode the converter can currently deploy: it maps
    # conv/linear WEIGHTS only, with no bias, so a folded bias must live in the
    # per-channel threshold. "conv" is exact but needs on-chip conv bias.
    fold_bias_mode: str = "threshold"
    # threshold mode only. The fold sets theta' = theta - b', and the chip can
    # only store theta' in (0, w_alpha]. Nothing in the float model stops b'
    # from leaving that band, and measured runs show it does: with theta capped
    # at 1.0 the observed b' reached ~1.59, so theta' went to -0.59 and the
    # config was rejected as undeployable after a full training run.
    #
    # constrain_fold_bias keeps b' inside the band DURING inline QAT, with a
    # straight-through gradient, so the network trains against the constraint
    # instead of meeting it at export. Set to 0 to disable and recover the old
    # behaviour. The margin is expressed as a fraction of theta: the constraint
    # is theta' >= fold_bias_margin * theta.
    fold_bias_margin: float = 0.05


def parse_fc_widths(spec: str) -> list[int]:
    """'512,256' -> [512, 256];  '' / '0' / 'none' -> []."""
    if spec is None:
        return []
    spec = str(spec).strip()
    if spec == "" or spec.lower() in ("none", "0"):
        return []
    widths = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError(f"FC width must be positive, got {value}")
        widths.append(value)
    return widths


def _norm_alias(raw: str):
    """--encoder batchnorm=true|false still works; maps onto the new norm field."""
    return "norm", ("bn" if raw.lower() in ("1", "true", "yes", "y") else "none")


_FIELD_ALIASES = {"EncoderSpec": {"batchnorm": _norm_alias}}


def _cast(value: str, caster):
    if caster is bool:
        return value.lower() in ("1", "true", "yes", "y")
    return caster(value)


def fill_from_kv(schema_cls, values: list[str]):
    obj = schema_cls()
    field_types = {f.name: f.type for f in fields(schema_cls)}
    aliases = _FIELD_ALIASES.get(schema_cls.__name__, {})
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected key=value (e.g. 'N=16'), got: {item!r}")
        key, raw_value = item.split("=", 1)
        if key in aliases:
            key, raw_value = aliases[key](raw_value)
        if key not in field_types:
            raise ValueError(f"Unknown field {key!r}. Valid: {list(field_types)}")
        setattr(obj, key, _cast(raw_value, field_types[key]))
    return obj


def build_config_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", nargs="*", default=[])
    parser.add_argument("--encoder", nargs="*", default=[])
    parser.add_argument("--output", nargs="*", default=[])
    parser.add_argument("--downsample", nargs="*", default=[])
    parser.add_argument("--head", nargs="*", default=[])
    parser.add_argument("--neuron", nargs="*", default=[])
    parser.add_argument("--train", nargs="*", default=[])
    return parser


def parse_config(argv=None):
    parser = build_config_parser()
    args, _ = parser.parse_known_args(argv)
    return {
        "input": fill_from_kv(InputSpec, args.input),
        "encoder": fill_from_kv(EncoderSpec, args.encoder),
        "output": fill_from_kv(OutputSpec, args.output),
        "downsample": fill_from_kv(DownsampleSpec, args.downsample),
        "head": fill_from_kv(HeadSpec, args.head),
        "neuron": fill_from_kv(NeuronSpec, args.neuron),
        "train": fill_from_kv(TrainSpec, args.train),
    }


# =============================================================================
# 2. Shape planning -- ONE source of truth
#
# The model, the feasibility checker, and the cost model all walk the same
# plan. Previously the model and the checker each re-derived shapes with
# duplicated loops, which is exactly how the head came to be unchecked.
# =============================================================================

class InfeasibleConfig(ValueError):
    """Raised when a config cannot be built at all (e.g. spatial size collapses)."""


def conv_out_size(size: int, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> int:
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@dataclass
class ConvBlockPlan:
    index: int
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    in_hw: tuple[int, int]
    conv_out_hw: tuple[int, int]     # after conv, where the LIF neurons live
    pool: bool
    pool_kernel: int
    pool_stride: int
    out_hw: tuple[int, int]          # after optional pool -> next block's input
    tau: "float | None" = None       # per-layer neuron overrides (None -> global)
    v_threshold: "float | None" = None


@dataclass
class LinearPlan:
    index: int
    in_features: int
    out_features: int
    is_classifier: bool


@dataclass
class NetPlan:
    blocks: list[ConvBlockPlan] = field(default_factory=list)
    linears: list[LinearPlan] = field(default_factory=list)
    reduction: str = "flatten"
    encoder_out_c: int = 0
    encoder_out_hw: tuple[int, int] = (0, 0)
    fc_in_features: int = 0


def plan_encoder(input_cfg: InputSpec, encoder_cfg: EncoderSpec,
                 downsample_cfg: DownsampleSpec) -> list[ConvBlockPlan]:
    cur_H, cur_W = effective_hw(input_cfg)
    in_channels = input_cfg.C
    layers = resolve_conv_layers(encoder_cfg)
    plans: list[ConvBlockPlan] = []

    for i, L in enumerate(layers):
        # "still big enough to shrink?" -- guards against the feature map
        # collapsing below the kernel. Applies per-layer now.
        can_shrink = cur_H > 3 and cur_W > 3

        # Where does this layer's downsampling come from? Priority:
        #   1. per-layer L.pool=True  -> stride-1 conv, then a pool
        #   2. else global stride mode -> use L.stride on the conv
        #   3. else (pool mode global, layer didn't ask) -> stride-1 conv
        # If the map is already too small to shrink, force stride 1 regardless.
        want_pool = L.pool or (downsample_cfg.mode == "pool")
        if can_shrink and not want_pool:
            conv_stride = L.stride
            conv_padding = L.padding
        else:
            conv_stride = 1
            conv_padding = (L.dilation * (L.kernel_size - 1)) // 2  # size-preserving

        oh = conv_out_size(cur_H, L.kernel_size, conv_stride, conv_padding, L.dilation)
        ow = conv_out_size(cur_W, L.kernel_size, conv_stride, conv_padding, L.dilation)
        if oh < 1 or ow < 1:
            raise InfeasibleConfig(
                f"block{i}: spatial size collapses to {oh}x{ow} "
                f"(in {cur_H}x{cur_W}, k={L.kernel_size}, s={conv_stride}, p={conv_padding}). "
                "Kernel is larger than the feature map."
            )

        do_pool = can_shrink and want_pool
        if do_pool:
            ph = oh // downsample_cfg.pool_stride
            pw = ow // downsample_cfg.pool_stride
            if ph < 1 or pw < 1:
                raise InfeasibleConfig(
                    f"block{i}: pool collapses {oh}x{ow} to {ph}x{pw} "
                    f"(pool_stride={downsample_cfg.pool_stride})."
                )
        else:
            ph, pw = oh, ow

        plans.append(ConvBlockPlan(
            index=i,
            in_channels=in_channels,
            out_channels=L.out_channels,
            kernel_size=L.kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=L.dilation,
            in_hw=(cur_H, cur_W),
            conv_out_hw=(oh, ow),
            pool=do_pool,
            pool_kernel=downsample_cfg.pool_kernel_size if do_pool else 0,
            pool_stride=downsample_cfg.pool_stride if do_pool else 0,
            out_hw=(ph, pw),
            tau=L.tau,
            v_threshold=L.v_threshold,
        ))

        cur_H, cur_W = ph, pw
        in_channels = L.out_channels

    return plans


def plan_network(input_cfg: InputSpec, encoder_cfg: EncoderSpec, output_cfg: OutputSpec,
                 downsample_cfg: DownsampleSpec, head_cfg: HeadSpec) -> NetPlan:
    blocks = plan_encoder(input_cfg, encoder_cfg, downsample_cfg)

    if blocks:
        out_c = blocks[-1].out_channels
        out_hw = blocks[-1].out_hw
    else:
        out_c = input_cfg.C
        out_hw = effective_hw(input_cfg)

    if head_cfg.final_reduction == "gap":
        fc_in = out_c
    elif head_cfg.final_reduction == "flatten":
        fc_in = out_c * out_hw[0] * out_hw[1]
    else:
        raise ValueError(f"Unknown final_reduction {head_cfg.final_reduction!r}")

    widths = parse_fc_widths(head_cfg.fc_widths)
    dims = [fc_in] + widths + [output_cfg.num_classes]
    linears = [
        LinearPlan(index=i, in_features=dims[i], out_features=dims[i + 1],
                   is_classifier=(i == len(dims) - 2))
        for i in range(len(dims) - 1)
    ]

    return NetPlan(blocks=blocks, linears=linears, reduction=head_cfg.final_reduction,
                   encoder_out_c=out_c, encoder_out_hw=out_hw, fc_in_features=fc_in)


# =============================================================================
# 3. HiAER-Spike hardware feasibility check -- now covers the FC head too
# =============================================================================

AXON_LIMITS = {
    "total_axons": 16_383,
    "fan_out": 4_096,   # axonal fan-out: how many neurons one input axon feeds
    "fan_in": 8_191,    # axonal fan-in: how many axons feed one neuron
}
NEURON_LIMITS = {
    "fan_out": 4_095,   # neuron-to-neuron fan-out
    "fan_in": 8_159,    # neuron-to-neuron fan-in
}


def check_feasibility(input_cfg: InputSpec, encoder_cfg: EncoderSpec,
                      downsample_cfg: DownsampleSpec,
                      head_cfg: HeadSpec = None, output_cfg: OutputSpec = None):
    """
    Walk the whole network -- conv encoder AND fully-connected head -- checking
    the hardware formulas at every layer. Block 0 consumes raw input axons;
    everything after that is neuron-to-neuron.

    Returns (is_feasible: bool, violations: list[str]).

    Two corrections vs. the original encoder-only version:
      * the LAST conv block's fan-out is the first FC layer's width, not the
        conv fan-out formula -- there is no next conv for it to feed;
      * the FC layers are checked at all, which they previously were not.
    """
    if head_cfg is None:
        head_cfg = HeadSpec()
    if output_cfg is None:
        output_cfg = OutputSpec()

    violations: list[str] = []
    try:
        plan = plan_network(input_cfg, encoder_cfg, output_cfg, downsample_cfg, head_cfg)
    except InfeasibleConfig as exc:
        return False, [str(exc)]

    first_fc_width = plan.linears[0].out_features if plan.linears else 0

    for b in plan.blocks:
        k, s = b.kernel_size, b.stride
        fan_in = k * k * b.in_channels

        is_last = (b.index == len(plan.blocks) - 1)
        if is_last:
            # feeds the head, not another conv
            fan_out = first_fc_width
            fan_out_label = "fan_out(->fc)"
        else:
            fan_out = math.ceil(k / s) ** 2 * b.out_channels
            fan_out_label = "fan_out(->conv)"

        if b.index == 0:
            total_axons = b.in_hw[0] * b.in_hw[1] * b.in_channels
            if total_axons > AXON_LIMITS["total_axons"]:
                violations.append(f"block0: total_axons {total_axons} > {AXON_LIMITS['total_axons']}")
            if fan_out > AXON_LIMITS["fan_out"]:
                violations.append(f"block0: axonal_{fan_out_label} {fan_out} > {AXON_LIMITS['fan_out']}")
            if fan_in > AXON_LIMITS["fan_in"]:
                violations.append(f"block0: axonal_fan_in {fan_in} > {AXON_LIMITS['fan_in']}")
        else:
            if fan_out > NEURON_LIMITS["fan_out"]:
                violations.append(f"block{b.index}: neuron_{fan_out_label} {fan_out} > {NEURON_LIMITS['fan_out']}")
            if fan_in > NEURON_LIMITS["fan_in"]:
                violations.append(f"block{b.index}: neuron_fan_in {fan_in} > {NEURON_LIMITS['fan_in']}")

    for lin in plan.linears:
        tag = "classifier" if lin.is_classifier else f"fc{lin.index}"
        if lin.in_features > NEURON_LIMITS["fan_in"]:
            violations.append(f"{tag}: neuron_fan_in {lin.in_features} > {NEURON_LIMITS['fan_in']}")
        # the LAST linear's neurons are the outputs -- they feed nothing on-chip
        if not lin.is_classifier and lin.out_features > NEURON_LIMITS["fan_out"]:
            violations.append(f"{tag}: neuron_fan_out {lin.out_features} > {NEURON_LIMITS['fan_out']}")

    return len(violations) == 0, violations


# =============================================================================
# 4. Cost model: neurons, synaptic connections, trainable parameters
#
# "Neurons"     = spiking units (one per LIF/IF site), counted once, not per T.
# "Connections" = synapses actually realized on hardware. For a conv this is
#                 out_C * H_out * W_out * (k*k*in_C) -- weight SHARING means
#                 this is much larger than the parameter count.
# "Parameters"  = trainable floats in PyTorch.
# =============================================================================

def count_neurons_and_synapses(cfg: dict) -> dict:
    """cfg is the same dict used everywhere else: input/encoder/output/downsample/head/neuron."""
    plan = plan_network(cfg["input"], cfg["encoder"], cfg["output"], cfg["downsample"], cfg["head"])
    encoder_cfg, input_cfg = cfg["encoder"], cfg["input"]

    rows = []
    for b in plan.blocks:
        oh, ow = b.conv_out_hw
        neurons = b.out_channels * oh * ow
        weights_per_neuron = b.kernel_size * b.kernel_size * b.in_channels
        rows.append({
            "layer": f"conv{b.index}",
            "detail": (f"{b.in_channels}x{b.in_hw[0]}x{b.in_hw[1]} -> "
                       f"{b.out_channels}x{oh}x{ow}  k={b.kernel_size} s={b.stride} p={b.padding}"),
            "neurons": neurons,
            "connections": neurons * weights_per_neuron,
            "params": weights_per_neuron * b.out_channels + (b.out_channels if encoder_cfg.bias else 0),
        })
        if encoder_cfg.norm != "none":
            # tdBN has the SAME parameter count as ordinary BN -- one gamma and
            # one beta per channel. (It is BNTT that would scale as C*T.)
            label = "tdbn" if encoder_cfg.norm == "tdbn" else "bn"
            rows.append({
                "layer": f"{label}{b.index}",
                "detail": (f"tdBN({b.out_channels}) over (T,N,H,W)" if encoder_cfg.norm == "tdbn"
                           else f"BatchNorm2d({b.out_channels})"),
                "neurons": 0, "connections": 0, "params": 2 * b.out_channels,
            })
        if b.pool:
            ph, pw = b.out_hw
            rows.append({
                "layer": f"pool{b.index}",
                "detail": f"avg k={b.pool_kernel} s={b.pool_stride} -> {b.out_channels}x{ph}x{pw}",
                "neurons": 0,
                "connections": b.out_channels * ph * pw * b.pool_kernel * b.pool_kernel,
                "params": 0,
            })

    if plan.reduction == "gap":
        rows.append({
            "layer": "gap",
            "detail": f"{plan.encoder_out_c}x{plan.encoder_out_hw[0]}x{plan.encoder_out_hw[1]} -> {plan.encoder_out_c}",
            "neurons": 0,
            "connections": plan.encoder_out_c * plan.encoder_out_hw[0] * plan.encoder_out_hw[1],
            "params": 0,
        })

    for lin in plan.linears:
        tag = "classifier" if lin.is_classifier else f"fc{lin.index}"
        rows.append({
            "layer": tag,
            "detail": f"{lin.in_features} -> {lin.out_features}",
            "neurons": lin.out_features,
            "connections": lin.in_features * lin.out_features,
            "params": lin.in_features * lin.out_features,
        })

    totals = {
        "neurons": sum(r["neurons"] for r in rows),
        "connections": sum(r["connections"] for r in rows),
        "params": sum(r["params"] for r in rows),
    }
    in_h, in_w = effective_hw(input_cfg)
    return {
        "rows": rows,
        "totals": totals,
        "input_axons": in_h * in_w * input_cfg.C,
        "timesteps": input_cfg.T,
        "neuron_updates_per_sample": totals["neurons"] * input_cfg.T,
        "plan": plan,
    }


def format_summary(cfg: dict) -> str:
    """Human-readable architecture + feasibility + cost table. This is the thing
    to paste into a write-up."""
    counts = count_neurons_and_synapses(cfg)
    feasible, violations = check_feasibility(
        cfg["input"], cfg["encoder"], cfg["downsample"], cfg["head"], cfg["output"]
    )
    in_h, in_w = effective_hw(cfg["input"])

    lines = []
    lines.append("=" * 78)
    lines.append("DVS Gesture SNN -- architecture summary")
    lines.append("=" * 78)
    ec = cfg["encoder"]
    nc = cfg["neuron"]
    layers = resolve_conv_layers(ec)
    mode = "per-layer" if ec.layers_json else "uniform"
    lines.append(f"input        : {cfg['input'].C}x{in_h}x{in_w}  (native 128x128, resize_to={cfg['input'].resize_to}), T={cfg['input'].T}, batch={cfg['input'].N}")
    lines.append(f"encoder      : {mode}, depth={len(layers)} bias={ec.bias} norm={ec.norm}"
                 + (f" tdbn_alpha={ec.tdbn_alpha}" if ec.norm == "tdbn" else ""))
    if ec.layers_json:
        for i, L in enumerate(layers):
            ov = ("".join(f" {k}={getattr(L, k)}" for k in ("tau", "v_threshold") if getattr(L, k) is not None))
            lines.append(f"  layer {i}    : ch={L.out_channels} k={L.kernel_size} stride={L.stride} pool={L.pool}{ov}")
    lines.append(f"downsample   : mode={cfg['downsample'].mode} pool={cfg['downsample'].pool_type} "
                 f"k={cfg['downsample'].pool_kernel_size} s={cfg['downsample'].pool_stride}")
    lines.append(f"head         : reduction={cfg['head'].final_reduction} fc_widths=[{cfg['head'].fc_widths}] "
                 f"dropout={cfg['head'].dropout_rate}")
    leak = int(round(float(nc.tau)))
    tau_tag = f"learned int, init {leak}" if nc.trainable_tau else f"{leak}"
    th_tag = "learned in (0,1]" if nc.trainable_threshold else f"{nc.v_threshold}"
    lines.append(f"neuron       : {nc.neuron_type} leak={tau_tag} v_th={th_tag} v_reset={nc.v_reset}")
    th_int = int(min(max(round(float(nc.v_threshold) / W_DELTA), 1), INT16_MAX))
    lines.append(f"  -> hardware: LIF_neuron(threshold={th_int}, shift=0, leak={leak})"
                 f"   [w_alpha={W_ALPHA:g}, w_bits={W_BITS}, weights clamped to +/-1]")
    if "train" in cfg:
        t = cfg["train"]
        lines.append(f"training     : {t.optimizer} lr={t.lr:g} wd={t.weight_decay:g} sched={t.scheduler} "
                     f"epochs={t.epochs} label_smoothing={t.label_smoothing:g} grad_clip={t.grad_clip:g}")
    lines.append("")

    header = f"{'layer':<12} {'shape / detail':<46} {'neurons':>10} {'connections':>14} {'params':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in counts["rows"]:
        lines.append(f"{r['layer']:<12} {r['detail']:<46} {r['neurons']:>10,} {r['connections']:>14,} {r['params']:>12,}")
    lines.append("-" * len(header))
    t = counts["totals"]
    lines.append(f"{'TOTAL':<12} {'':<46} {t['neurons']:>10,} {t['connections']:>14,} {t['params']:>12,}")
    lines.append("")
    lines.append(f"input axons             : {counts['input_axons']:,}  (limit {AXON_LIMITS['total_axons']:,})")
    lines.append(f"spiking neurons         : {t['neurons']:,}")
    lines.append(f"synaptic connections    : {t['connections']:,}")
    lines.append(f"trainable parameters    : {t['params']:,}")
    lines.append(f"neuron updates / sample : {counts['neuron_updates_per_sample']:,}  ({t['neurons']:,} x T={counts['timesteps']})")
    n_spiking = len(counts["plan"].blocks) + len(counts["plan"].linears)
    lines.append(f"pipeline flush steps    : {n_spiking}  (1 per spiking layer; the converter's "
                 f"post-input zero steps)")
    lines.append("")
    if feasible:
        lines.append("HiAER-Spike feasibility : PASS")
    else:
        lines.append("HiAER-Spike feasibility : FAIL")
        for v in violations:
            lines.append(f"   - {v}")
    lines.append("=" * 78)
    return "\n".join(lines)


# =============================================================================
# 4b. HiAER-Spike deployment semantics -- the numbers the chip actually runs
#
# Every constant here is read off hs_api, not chosen. Changing one means the
# trained model and the converted model stop agreeing.
#
#   quantizer.Quantize_Network(w_alpha=1, dynamic_alpha=False, w_bits=16)
#     w_delta = w_alpha / (2**(w_bits-1) - 1) = 1/32767
#     weight_quantization._pq.forward:  w/alpha -> clamp(-1, 1) -> 15-bit round
#     _quantize_LIF:                    v_threshold -> int(v_threshold/w_delta)
#
# Three consequences that the float model never sees:
#
#   1. WEIGHTS ARE CLAMPED TO [-1, 1]. With w_alpha fixed at 1 there is no
#      rescaling: any |w| > 1 is silently truncated. BN folding multiplies
#      conv weights by gamma/sqrt(var+eps), so folding is precisely the step
#      that pushes weights out of range. This is the dominant source of
#      float-to-chip accuracy loss and nothing in the stack warns about it.
#   2. THRESHOLDS QUANTIZE TO int(theta * 32767), so theta must stay in
#      (0, 1]. theta = 1.0 lands exactly on INT16_MAX; above that the integer
#      threshold overflows the field. theta <= 0 fires every timestep.
#   3. LEAK IS AN INTEGER. tau deploys as round(tau), so tau is a discrete
#      knob with very few usable settings in the small-tau regime.
# =============================================================================

W_BITS = 16
W_ALPHA = 1.0
INT16_MAX = 2 ** (W_BITS - 1) - 1            # 32767
W_DELTA = W_ALPHA / INT16_MAX                # 1/32767

# The leak values worth searching. Small-tau spacing is coarse *because the
# hardware register is coarse there* -- tau 2 and tau 3 are genuinely different
# networks, tau 2.1 and tau 2.4 are the same network. The large-tau end matches
# the working DVS conversion (Custom_IFNode's default tau=63, leak_lif=63).
HW_TAU_CHOICES = [2, 3, 4, 6, 8, 16, 32, 63]
HW_TAU_MIN, HW_TAU_MAX = 2, 128


if _HAS_TORCH:
    def _ste(quantized, original):
        """Straight-through estimator: forward uses `quantized`, backward passes
        the gradient to `original` unchanged."""
        return original + (quantized - original).detach()

    def fake_quantize_weight(w):
        """
        Bit-exact mirror of hs_api.quantizer.weight_quantization with
        w_alpha=1, w_bits=16, in a differentiable wrapper.

        The clamp is the part that matters: it is what makes an out-of-range
        folded weight visible to the loss instead of showing up as a surprise
        after conversion.
        """
        b = W_BITS - 1                                  # quantizer uses w_bit - 1
        levels = 2 ** b - 1
        wc = torch.clamp(w / W_ALPHA, min=-1.0, max=1.0)
        q = torch.round(wc.abs() * levels) / levels * torch.sign(wc) * W_ALPHA
        return _ste(q, w)

    def fold_bias_band(v_th: float, margin: float = 0.05):
        """The interval b' must lie in for theta' = theta - b' to be storable.

        Two-sided, because both ends fail silently:
          * theta' <= 0     -> the neuron fires on every timestep regardless of
                               input. deployment_report treats this as blocking.
          * theta' > w_alpha -> quantized_threshold_int clamps to INT16_MAX, so
                               the deployed threshold is not the one trained.
        Returns (lo, hi) with lo <= 0 <= hi for any theta in (0, w_alpha].
        """
        lo = float(v_th) - W_ALPHA            # keeps theta' <= w_alpha
        hi = float(v_th) * (1.0 - float(margin))   # keeps theta' >= margin*theta
        return lo, hi

    def constrain_fold_bias(b, v_th, margin: float = 0.05):
        """Clamp the folded bias into the band above, straight-through.

        Forward uses the clamped value, so the neuron sees the b' that will
        actually deploy and the loss responds to the constraint. Backward passes
        the gradient to bn.bias unchanged, so the optimizer can still move beta
        toward a legal value on its own rather than being pinned at the bound.
        """
        lo, hi = fold_bias_band(v_th, margin)
        return _ste(torch.clamp(b, min=lo, max=hi), b)

    def fake_quantize_threshold(th):
        """theta -> the float the chip's integer threshold represents.
        Clamped into (0, 1] so int(theta/W_DELTA) is a legal INT16 threshold."""
        thc = torch.clamp(th, min=W_DELTA, max=W_ALPHA)
        return _ste(torch.round(thc / W_DELTA) * W_DELTA, th)

    def quantized_threshold_int(th):
        """The integer written into LIF_neuron(threshold=...)."""
        if torch.is_tensor(th):
            return torch.clamp(torch.round(th / W_DELTA), 1, INT16_MAX).to(torch.int64)
        return int(min(max(round(float(th) / W_DELTA), 1), INT16_MAX))

    def weight_clip_fraction(net) -> float:
        """Fraction of conv/linear weights outside the representable [-1, 1].
        Call on the FOLDED net -- pre-fold numbers are meaningless."""
        out_of_range = total = 0
        for m in net.modules():
            if isinstance(m, (layer.Conv2d, nn.Conv2d, layer.Linear, nn.Linear)):
                w = m.weight.data
                out_of_range += (w.abs() > W_ALPHA).sum().item()
                total += w.numel()
        return out_of_range / max(1, total)
else:
    fake_quantize_weight = fake_quantize_threshold = None
    quantized_threshold_int = weight_clip_fraction = None
    fold_bias_band = constrain_fold_bias = None


# =============================================================================
# 5. Model -- built directly from the plan, so it cannot drift from the checker
# =============================================================================

if _HAS_TORCH:
    class TdBatchNorm2d(layer.BatchNorm2d):
        """
        tdBN (Zheng et al., "Going Deeper with Directly-Trained Larger SNNs").

        Two differences from ordinary BatchNorm2d:
          * statistics are taken over (T, N, H, W) rather than (N, H, W), so a
            single normalization is shared by the whole sequence instead of each
            timestep being normalized by its own batch stats at train time and
            by pooled running stats at eval time;
          * gamma is initialised to alpha * v_threshold, so pre-activations are
            scaled relative to the firing threshold rather than to unit variance.

        The (T, N, H, W) reduction comes for free from spikingjelly's multi-step
        mode: with step_mode='m' the layer flattens (T, N) into one batch axis
        before calling into BatchNorm2d. That is exactly tdBN's reduction, which
        is why this class only has to fix the initialisation.

        Foldable at inference like ordinary BN -- one gamma, one sigma, no
        per-timestep weights.
        """
        def __init__(self, num_features, alpha: float = 1.0, v_threshold: float = 1.0, **kwargs):
            kwargs.setdefault("step_mode", "m")
            super().__init__(num_features, **kwargs)
            self.alpha = alpha
            self.v_threshold = v_threshold
            nn.init.constant_(self.weight, alpha * v_threshold)
else:
    TdBatchNorm2d = None


if _HAS_TORCH:
    class HardwareLIFNode(neuron.BaseNode):
        """
        The neuron HiAER-Spike actually runs, trained directly.

        This replaces LIFNode / ParametricLIFNode / LearnableThresholdLIF on
        every path. Those are all fine SNN neurons; none of them is the neuron
        the converter emits, and three of the differences change the answer:

        1. OPERATION ORDER / LATENCY. SpikingJelly charges then fires within a
           timestep. hs_api's Custom_IFNode -- whose docstring says its order
           was "changed to match the converter's order" -- fires on the membrane
           carried in from the PREVIOUS step, then resets, leaks, and only then
           integrates the new input:

               spike = v > theta ;  v = reset ;  v -= (v - v_reset)/tau ;  v += x

           So every spiking layer adds one timestep of latency, which is exactly
           why the conversion script runs `num_layers` extra zero-input steps to
           flush the pipeline. A network trained with the SpikingJelly order has
           a different latency structure than the one deployed, and the deeper
           the net the worse the mismatch. See forward_over_time().

        2. INTEGER LEAK. tau deploys as round(tau) into an integer register, so
           tau is not continuous and never was. Learning it as a float and
           rounding at conversion silently moves the network. When learn_tau is
           on, the rounding happens in the forward pass with a straight-through
           gradient, so the loss sees the leak the chip will use.

        3. DECAYED INPUT. SpikingJelly's decay_input=True (which build_neuron
           previously used everywhere) folds x into the decay: v += (x - (v -
           v_reset))/tau. The hardware adds the raw integer weight, undecayed.
           That is decay_input=False, and it is baked into the order above.

        The threshold is likewise held on the chip's grid. It is parametrised
        through a sigmoid so it lives in (0, 1]: theta = 1.0 quantizes to
        exactly INT16_MAX, and anything larger overflows the threshold field
        while anything <= 0 fires unconditionally. Per-channel thresholds (what
        BN folding produces in bias_mode='threshold') are supported and
        broadcast, because hs_api reads a neuron model per neuron key.
        """

        def __init__(self, tau: int = 2, v_threshold: float = 1.0, v_reset: float = 0.0,
                     learn_tau: bool = False, learn_threshold: bool = False,
                     surrogate_function: Callable = None, detach_reset: bool = True,
                     step_mode: str = "s", store_v_seq: bool = False):
            super().__init__(v_threshold=float(v_threshold), v_reset=v_reset,
                             surrogate_function=surrogate_function or surrogate.ATan(),
                             detach_reset=detach_reset, step_mode=step_mode,
                             backend="torch", store_v_seq=store_v_seq)
            self.learn_tau = bool(learn_tau)
            self.learn_threshold = bool(learn_threshold)

            tau0 = float(min(max(int(round(float(tau))), HW_TAU_MIN), HW_TAU_MAX))
            if self.learn_tau:
                self.raw_tau = nn.Parameter(torch.tensor(tau0))
            else:
                self.register_buffer("_tau_const", torch.tensor(tau0))

            if self.learn_threshold:
                th0 = float(min(max(float(v_threshold), 1e-3), 1.0 - 1e-4))
                self.raw_threshold = nn.Parameter(torch.tensor(math.log(th0 / (1.0 - th0))))

        # ---- hardware parameters, always reported on the chip's grid --------
        @property
        def hw_tau(self):
            if self.learn_tau:
                t = torch.clamp(self.raw_tau, HW_TAU_MIN, HW_TAU_MAX)
                return _ste(torch.round(t), t)          # integer leak, STE gradient
            return self._tau_const

        @property
        def v_threshold(self):
            if getattr(self, "learn_threshold", False):
                return fake_quantize_threshold(torch.sigmoid(self.raw_threshold))
            v = self._v_threshold_const
            if torch.is_tensor(v):
                return fake_quantize_threshold(v)       # per-channel, post-fold
            return float(quantized_threshold_int(v)) * W_DELTA

        @v_threshold.setter
        def v_threshold(self, value):
            # BaseNode.__init__ assigns here before raw_threshold exists
            self._v_threshold_const = value

        def raw_v_threshold(self):
            """The threshold BEFORE the legality clamp.

            v_threshold clamps into (0, 1] so the forward pass cannot divide by
            a nonsense threshold. That clamp must never be what the deployment
            check looks at, or a fold that drove theta negative would be
            reported as fine and then fire unconditionally on chip.
            """
            if getattr(self, "learn_threshold", False):
                return torch.sigmoid(self.raw_threshold)
            return self._v_threshold_const

        def hw_leak(self) -> int:
            return int(round(float(torch.as_tensor(self.hw_tau).detach().item())))

        def hw_threshold_int(self):
            """Integer threshold(s) written into LIF_neuron(threshold=...)."""
            return quantized_threshold_int(torch.as_tensor(self.v_threshold).detach())

        # ---- dynamics -------------------------------------------------------
        def single_step_forward(self, x: "torch.Tensor"):
            self.v_float_to_tensor(x)
            v_reset = 0.0 if self.v_reset is None else float(self.v_reset)
            v_th = self.v_threshold

            # During inline-fold QAT (ConvBNFoldQuant, threshold mode) the folded
            # bias b' is delivered here each step instead of into the conv.
            # Subtracting b' from the input is identical to comparing against a
            # per-channel threshold (theta - b'), which is what the chip does --
            # so the neuron trains against the exact deployed pre-activation.
            fb = getattr(self, "_fold_bias", None)
            if fb is not None:
                shape = [1, fb.shape[0]] + [1] * (x.dim() - 2)   # broadcast over (N,C,...)
                x = x + fb.reshape(*shape)

            # 1. fire on the membrane carried in from the previous timestep.
            #    The chip compares v > theta (strictly greater); one LSB of the
            #    integer grid is W_DELTA, so subtracting it reproduces `>` from
            #    the surrogate's `>=` without breaking the gradient.
            spike = self.surrogate_function(self.v - v_th - W_DELTA)
            # 2. hard reset
            spike_d = spike.detach() if self.detach_reset else spike
            self.v = v_reset * spike_d + (1.0 - spike_d) * self.v
            # 3. integer leak toward v_reset
            self.v = self.v - (self.v - v_reset) / self.hw_tau
            # 4. integrate this timestep's input, undecayed
            self.v = self.v + x
            return spike

        def extra_repr(self):
            th = self.v_threshold
            th_s = "per-channel" if torch.is_tensor(th) and th.numel() > 1 else f"{float(th):.5f}"
            return (f"tau={self.hw_leak()} (integer leak), v_threshold={th_s}, "
                    f"learn_tau={self.learn_tau}, learn_threshold={self.learn_threshold}")


    class _WeightFakeQuant(nn.Module):
        """Parametrization that makes `module.weight` return the INT16 value the
        chip stores, while gradients flow to the underlying float parameter."""
        def forward(self, w):
            return fake_quantize_weight(w)


    class ConvBNFoldQuant(nn.Module):
        """
        True quantization-aware training with INLINE BN folding (Option A).

        A (Conv2d, BatchNorm2d) pair collapses into ONE module whose forward, on
        EVERY step of EVERY batch, does what the deployed chip does:

            w' = conv_w * gamma / sqrt(running_var + eps)      # fold, per channel
            b' = beta + (conv_b - running_mean) * scale
            y  = conv(x, fake_quantize(w'))  (+ b' as bias, or into threshold)

        So the neuron downstream always sees the folded, INT16-rounded weight --
        not the raw float conv weight that only gets folded and quantized later.
        The whole run trains on the grid, so the network never gets to build the
        precise-weight balances that ordinary float training leans on and that
        post-hoc quantization then shatters.

        Why fold from RUNNING stats, not batch stats: the chip folds the running
        (inference) statistics -- that is the number that deploys. Folding batch
        stats during training would mean the quantized weight the neuron sees
        flickers with each mini-batch and does not match deployment. BN's affine
        params (gamma, beta) and its running stats still update normally from a
        cheap statistics-only pass over the raw conv output, so the fold tracks
        the data as training proceeds; only the *weight used in the conv* is the
        folded/quantized one. (This is the standard freeze-BN-stats QAT fold,
        e.g. Jacob et al. 2018 / PyTorch's fused ConvBn.)

        bias_mode mirrors fold_bn:
          "conv"      -> b' added as a conv bias (exact; runs in SpikingJelly).
          "threshold" -> b' handed to the following neuron's per-channel
                         threshold, matching the HiAER-Spike conversion. The
                         conv stays bias-free, exactly as deployed.

        At the end of training, `export_folded_conv()` emits a plain bias-free
        (or biased) Conv2d holding the final quantized weight -- so the deploy
        path and converter see an ordinary folded conv with nothing left to do.
        """

        def __init__(self, conv, bn, bias_mode="threshold", quantize=True,
                     fold_bias_margin: float = 0.05):
            super().__init__()
            self.conv = conv                     # keep as trainable float params
            self.bn = bn                         # keep for gamma/beta + running stats
            self.bias_mode = bias_mode
            self.quantize = quantize
            self.fold_bias_margin = float(fold_bias_margin)
            self.stride = conv.stride
            self.padding = conv.padding
            self.dilation = conv.dilation
            self.groups = conv.groups
            self.out_channels = conv.out_channels
            self.in_channels = conv.in_channels
            self.kernel_size = conv.kernel_size
            # neuron whose threshold absorbs b' in threshold mode; wired by the
            # model builder right after construction (needs the sibling module).
            self._bias_sink = None
            self.step_mode = getattr(conv, "step_mode", "s")

        def _sink_threshold(self) -> float:
            """The base theta of the neuron this block's bias is folded into."""
            node = self._bias_sink
            if node is None:
                return W_ALPHA
            th = node.raw_v_threshold()          # UNclamped: the real value
            th = torch.as_tensor(th).detach().float()
            return float(th.min())               # scalar before export

        # --- the fold, from CURRENT bn params/stats --------------------------
        def _folded_weight_bias(self):
            bn = self.bn
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)     # [O]
            w = self.conv.weight * scale.reshape(-1, *([1] * (self.conv.weight.ndim - 1)))
            cb = self.conv.bias if self.conv.bias is not None else 0.0
            b = bn.bias + (cb - bn.running_mean) * scale               # [O]
            if self.quantize:
                w = fake_quantize_weight(w)      # STE: forward rounds, backward passes
            # In threshold mode b' becomes theta - b' on chip, so it has to stay
            # inside the storable band. Applied here rather than at export so the
            # forward pass, and therefore the loss, sees the deployed value.
            # conv mode needs no constraint: b' stays on the conv, untouched.
            if self.bias_mode == "threshold" and self.fold_bias_margin > 0:
                b = constrain_fold_bias(b, self._sink_threshold(), self.fold_bias_margin)
            return w, b

        @torch.no_grad()
        def fold_bias_violation(self):
            """Fraction of channels whose unconstrained b' sits outside the band,
            with the worst offender. Reports pressure the clamp is absorbing."""
            bn = self.bn
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            cb = self.conv.bias if self.conv.bias is not None else 0.0
            b = bn.bias + (cb - bn.running_mean) * scale
            lo, hi = fold_bias_band(self._sink_threshold(), self.fold_bias_margin)
            out = ((b < lo) | (b > hi))
            return {"frac_out_of_band": out.float().mean().item(),
                    "min_b": b.min().item(), "max_b": b.max().item(),
                    "band": (lo, hi)}

        def _update_bn_stats(self, x):
            """Run BN in train mode purely to update running_mean/var (and let
            gamma/beta receive gradient) WITHOUT letting it also renormalise the
            path -- the fold already applied the affine transform. We do this by
            pushing the raw conv output through bn and discarding the result."""
            if self.training:
                raw = self._conv_raw(x)
                _ = self.bn(raw)                 # updates running stats; output dropped

        def _conv_raw(self, x):
            # step_mode 'm' inputs are (T, N, C, H, W): fold T into N for conv2d
            if self.step_mode == "m" and x.dim() == 5:
                T, N = x.shape[0], x.shape[1]
                y = F.conv2d(x.flatten(0, 1), self.conv.weight, self.conv.bias,
                             self.stride, self.padding, self.dilation, self.groups)
                return y.reshape(T, N, *y.shape[1:])
            return F.conv2d(x, self.conv.weight, self.conv.bias,
                            self.stride, self.padding, self.dilation, self.groups)

        def forward(self, x):
            # keep BN statistics current from the true conv output
            self._update_bn_stats(x)
            w, b = self._folded_weight_bias()
            use_bias = b if self.bias_mode == "conv" else None
            if self.step_mode == "m" and x.dim() == 5:
                T, N = x.shape[0], x.shape[1]
                y = F.conv2d(x.flatten(0, 1), w, use_bias,
                             self.stride, self.padding, self.dilation, self.groups)
                y = y.reshape(T, N, *y.shape[1:])
            else:
                y = F.conv2d(x, w, use_bias,
                             self.stride, self.padding, self.dilation, self.groups)
            # in threshold mode b' lives in the neuron; push the CURRENT b' onto
            # it every step. Keep it attached to the graph so beta (bn.bias)
            # still receives gradient through the neuron's input during grid
            # training -- freezing beta would throw away capacity the chip can
            # represent (b' becomes the per-channel threshold at export).
            if self.bias_mode == "threshold" and self._bias_sink is not None:
                self._bias_sink._fold_bias = b
            return y

        @torch.no_grad()
        def export_folded_conv(self):
            """Freeze into a plain Conv2d holding the final folded+quantized
            weight -- what fold_bn would have produced, but already grid-trained."""
            w, b = self._folded_weight_bias()
            has_bias = self.bias_mode == "conv"
            conv = layer.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups, bias=has_bias)
            conv.weight.data.copy_(w)
            if has_bias:
                conv.bias.data.copy_(b)
            conv.step_mode = self.step_mode
            return conv, b


    class _LegacyLearnableThresholdLIF(neuron.ParametricLIFNode):
        """
        The previous trainable-threshold neuron, kept only so old checkpoints
        still load. NOT deployable: continuous tau (= 1/sigmoid(w)) rounds to an
        integer leak at conversion, and its charge-then-fire order does not match
        the converter. New runs use HardwareLIFNode.

        Q3's "special challenges", and how this handles them:

          * Non-differentiable fire. The threshold sits inside a Heaviside.
            Gradient reaches it only through the surrogate: d/d(v_th) of
            surrogate(v - v_th) = -surrogate'(v - v_th). Inherited unchanged --
            we only swap the constant v_th for a parameter, the surrogate does
            the rest.
          * Positivity. A threshold <= 0 makes the neuron fire every step and
            kills learning. We parametrise v_th = softplus(raw), so it is always
            > 0 and can never cross zero no matter what the optimiser does.
          * Scale coupling. Scaling a layer's input weights and its threshold
            together leaves the spike train unchanged (for v_reset=0), so weight
            and threshold are jointly under-determined. BatchNorm already fixes
            the input scale, which pins the threshold -- so trainable_threshold
            is most stable WITH norm on. With norm='none' it can drift; the raw
            parameter gets weight decay to damp that.
          * Per-LAYER only. One scalar per neuron group. Per-channel thresholds
            were considered and rejected as too many DOF for this dataset.

        Deployment: at conversion the LEARNED tau and threshold are read out as
        floats and baked into a fixed Custom_LIFNode -- the hardware neuron is
        never itself "trainable"; training only discovers good constants.
        """
        def __init__(self, init_threshold: float = 1.0, learn_threshold: bool = True, **kwargs):
            kwargs["v_threshold"] = init_threshold      # base stores the initial value
            super().__init__(**kwargs)
            self.learn_threshold = learn_threshold
            if learn_threshold:
                inv = math.log(math.expm1(max(init_threshold, 1e-3)))  # softplus^{-1}
                self.raw_threshold = nn.Parameter(torch.tensor(float(inv)))

        @property
        def v_threshold(self):
            if getattr(self, "learn_threshold", False):
                return F.softplus(self.raw_threshold)
            return self._v_threshold_const

        @v_threshold.setter
        def v_threshold(self, value):
            # BaseNode.__init__ assigns a float here before raw_threshold exists
            self._v_threshold_const = value
else:
    HardwareLIFNode = _WeightFakeQuant = _LegacyLearnableThresholdLIF = None

LearnableThresholdLIF = _LegacyLearnableThresholdLIF   # back-compat alias


def build_neuron(neuron_cfg: NeuronSpec, tau, v_threshold: float):
    """
    One neuron group, always the deployable neuron.

    There is deliberately no "train with a nice neuron, convert to the hardware
    one later" path any more. Every substitution of that kind -- SpikingJelly's
    charge-then-fire order, decay_input=True, a continuous tau -- is a silent
    change to the network that only shows up after conversion, which is exactly
    the gap this search exists to close. Trainability is a flag ON the hardware
    neuron, not a different neuron.
    """
    _require_torch()
    if neuron_cfg.neuron_type != "LIF":
        raise NotImplementedError(
            f"neuron_type={neuron_cfg.neuron_type!r} is not wired to the HiAER-Spike "
            "conversion path. The converter emits LIF_neuron(threshold, shift, leak); "
            "add an ANN_neuron/IF_neuron export before searching over this."
        )
    return HardwareLIFNode(
        tau=tau, v_threshold=v_threshold, v_reset=neuron_cfg.v_reset,
        learn_tau=neuron_cfg.trainable_tau,
        learn_threshold=neuron_cfg.trainable_threshold,
        surrogate_function=surrogate.ATan(), detach_reset=True,
    )


class DVSGesturePuru(nn.Module):
    def __init__(self, input_cfg: InputSpec, encoder_cfg: EncoderSpec, output_cfg: OutputSpec,
                 downsample_cfg: DownsampleSpec, head_cfg: HeadSpec, neuron_cfg: NeuronSpec):
        _require_torch()
        super().__init__()

        plan = plan_network(input_cfg, encoder_cfg, output_cfg, downsample_cfg, head_cfg)

        def resolved(over, glob):
            return glob if over is None else over

        def mk_neuron(tau, v_th):
            return build_neuron(neuron_cfg, tau=tau, v_threshold=v_th)

        # a conv with no normalization after it needs its own bias, or every
        # output channel is locked to zero mean
        norm_kind = encoder_cfg.norm
        if norm_kind not in ("none", "bn", "tdbn"):
            raise ValueError(f"Unknown norm {norm_kind!r}. Valid: none, bn, tdbn")
        use_bias = encoder_cfg.bias or norm_kind == "none"

        # tdBN's statistics span the time axis, so the whole net must run in
        # multi-step mode. For conv / linear / dropout / LIF this is numerically
        # identical to the manual single-step loop -- the ONLY thing that changes
        # is where BatchNorm takes its statistics from. That keeps bn vs tdbn a
        # clean one-variable comparison.
        self.step_mode = "m" if norm_kind == "tdbn" else "s"

        modules = []
        for b in plan.blocks:
            modules.append(layer.Conv2d(
                b.in_channels, b.out_channels,
                kernel_size=b.kernel_size, stride=b.stride,
                padding=b.padding, dilation=b.dilation, bias=use_bias,
            ))
            if norm_kind == "tdbn":
                modules.append(TdBatchNorm2d(b.out_channels, alpha=encoder_cfg.tdbn_alpha,
                                             v_threshold=resolved(b.v_threshold, neuron_cfg.v_threshold)))
            elif norm_kind == "bn":
                modules.append(layer.BatchNorm2d(b.out_channels))
            modules.append(mk_neuron(resolved(b.tau, neuron_cfg.tau),
                                     resolved(b.v_threshold, neuron_cfg.v_threshold)))
            if b.pool:
                pool_cls = layer.MaxPool2d if downsample_cfg.pool_type == "max" else layer.AvgPool2d
                modules.append(pool_cls(kernel_size=b.pool_kernel, stride=b.pool_stride))

        # Q4: GAP replaces the flatten. It collapses HxW to 1x1 BEFORE the FC
        # head, so the classifier's fan-in is just `channels` -- no giant first
        # Linear, far fewer params, less overfitting. Deployable on HiAER-Spike:
        # average-pooling over the spatial grid is a fixed uniform-weight linear
        # map (weight 1/(H*W)), which folds into the following Linear exactly the
        # way AvgPool folds into the next conv. (Per Christopher's CIFAR-10 GAP.)
        if plan.reduction == "gap":
            modules += [layer.AdaptiveAvgPool2d(1), layer.Flatten()]
        else:
            modules += [layer.Flatten()]

        for lin in plan.linears:
            if head_cfg.dropout_rate > 0:
                modules.append(layer.Dropout(head_cfg.dropout_rate))
            modules.append(layer.Linear(lin.in_features, lin.out_features, bias=False))
            modules.append(mk_neuron(neuron_cfg.tau, neuron_cfg.v_threshold))

        self.conv_fc = nn.Sequential(*modules)
        if self.step_mode == "m":
            functional.set_step_mode(self.conv_fc, "m")
            if _HAS_HS_API:
                # hs_api's custom neurons are built for their conversion flow and
                # may be single-step only. Fail loudly rather than silently
                # running a model whose time semantics are wrong.
                for m in self.conv_fc.modules():
                    if isinstance(m, (Custom_LIFNode, Custom_IFNode)) and getattr(m, "step_mode", "s") != "m":
                        raise RuntimeError(
                            "norm='tdbn' needs multi-step neurons, but hs_api's "
                            f"{type(m).__name__} did not accept step_mode='m'. "
                            "Use norm='bn' with hs_api neurons, or verify multi-step support."
                        )
        self.plan = plan
        self.final_H, self.final_W = plan.encoder_out_hw
        self.in_features = plan.fc_in_features

    def forward(self, x: "torch.Tensor"):
        return self.conv_fc(x)

    def to_qat_folded(self, bias_mode: str = "threshold", fold_bias_margin: float = 0.05):
        """
        Swap every (Conv2d, BatchNorm2d) pair for a single ConvBNFoldQuant, in
        place, so the whole subsequent run trains on the folded+quantized grid
        (Option A / true QAT). The neuron after each pair becomes the sink for
        the folded bias in threshold mode, and fold_bias_margin keeps that bias
        inside the band where theta - b' remains a storable threshold.

        Call AFTER a short float warmup (BN needs a few epochs of real statistics
        before its running_var is meaningful enough to fold from). Idempotent:
        a model already converted is left unchanged.
        """
        _require_torch()
        if getattr(self, "_qat_folded", False):
            return self
        mods = list(self.conv_fc)
        new_mods, i = [], 0
        while i < len(mods):
            m = mods[i]
            nxt = mods[i + 1] if i + 1 < len(mods) else None
            nxt2 = mods[i + 2] if i + 2 < len(mods) else None
            if isinstance(m, (layer.Conv2d, nn.Conv2d)) and isinstance(nxt, (layer.BatchNorm2d, nn.BatchNorm2d)):
                block = ConvBNFoldQuant(m, nxt, bias_mode=bias_mode, quantize=True,
                                        fold_bias_margin=fold_bias_margin)
                block.step_mode = self.step_mode
                new_mods.append(block)
                # link the bias sink to the following neuron (threshold mode).
                # Wired BEFORE any forward pass, because _folded_weight_bias
                # reads the sink's threshold to size the constraint band.
                if bias_mode == "threshold" and isinstance(nxt2, HardwareLIFNode):
                    block._bias_sink = nxt2
                i += 2                       # consumed conv + bn
            else:
                new_mods.append(m)
                i += 1
        self.conv_fc = nn.Sequential(*new_mods)
        if self.step_mode == "m":
            functional.set_step_mode(self.conv_fc, "m")
        # bias-free Linears are already on the grid; parametrize them too so the
        # FC weights train quantized alongside the folded convs.
        enable_weight_fake_quant(self)
        self._qat_folded = True
        self._qat_bias_mode = bias_mode
        return self

    @_no_grad
    def export_deployed(self):
        """
        After QAT, produce the plain BN-free model the converter consumes: each
        ConvBNFoldQuant becomes an ordinary folded Conv2d holding the final
        quantized weight, and threshold-mode biases are baked into per-channel
        neuron thresholds. Equivalent to what fold_bn produces, but the weights
        were trained ON the grid rather than snapped onto it afterwards.
        """
        _require_torch()
        if not getattr(self, "_qat_folded", False):
            return self
        bake_weight_fake_quant(self)          # collapse Linear fake-quant params
        mods = list(self.conv_fc)
        new_mods = []
        for idx, m in enumerate(mods):
            if isinstance(m, ConvBNFoldQuant):
                conv, b_prime = m.export_folded_conv()
                new_mods.append(conv)
                if m.bias_mode == "threshold" and isinstance(m._bias_sink, HardwareLIFNode):
                    node = m._bias_sink
                    th = node.raw_v_threshold()
                    base = th.detach() if torch.is_tensor(th) else float(th)
                    node.learn_threshold = False
                    if hasattr(node, "raw_threshold"):
                        node.raw_threshold.requires_grad_(False)
                    node._v_threshold_const = base - b_prime.reshape(-1, 1, 1)
                    node._fold_bias = None     # bias now lives in the threshold
            else:
                new_mods.append(m)
        self.conv_fc = nn.Sequential(*new_mods)
        if self.step_mode == "m":
            functional.set_step_mode(self.conv_fc, "m")
        self._qat_folded = False
        return self


def build_model(cfg: dict) -> "DVSGesturePuru":
    return DVSGesturePuru(
        input_cfg=cfg["input"], encoder_cfg=cfg["encoder"], output_cfg=cfg["output"],
        downsample_cfg=cfg["downsample"], head_cfg=cfg["head"], neuron_cfg=cfg["neuron"],
    )


# =============================================================================
# 5b. BN folding for HiAER-Spike deployment  (answers Q1)
#
# HiAER-Spike has no BN layer, so BN must be folded into the preceding conv
# after training. In float this is EXACT -- conv and BN are both affine with
# nothing between them, so the folded net computes the identical pre-neuron
# current and therefore the identical spike train. The whole problem is where
# the fold's manufactured bias goes, because Custom_LIFNode has no bias field.
#
#   W'_c = W_c * gamma_c / sqrt(var_c + eps)
#   b'_c = beta_c - mu_c * gamma_c / sqrt(var_c + eps)      (nonzero even if the
#                                                            conv had bias=False)
#
# Two deployment routes, both provided:
#   * bias_mode="conv"      : keep b' on the conv. Exact. Needs hs_api's conv to
#                             support a per-output-channel bias.
#   * bias_mode="threshold" : fold b' into a per-channel threshold theta'_c =
#                             theta_c - b'_c. No conv bias needed, but only exact
#                             at the membrane steady state -- see the transient
#                             note below. Custom_LIFNode's scalar v_threshold is
#                             widened to a per-channel tensor, which broadcasts.
# =============================================================================

def _fold_conv_bn_params(conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, eps):
    """Pure-array fold, usable from numpy (reference/tests) or torch. Returns
    (W', b'). Shapes: conv_w [O,I,kh,kw], the rest [O]."""
    scale = bn_gamma / (bn_var + eps) ** 0.5           # [O]
    w_prime = conv_w * scale.reshape(-1, *([1] * (conv_w.ndim - 1)))
    cb = conv_b if conv_b is not None else 0.0
    b_prime = bn_beta + (cb - bn_mean) * scale
    return w_prime, b_prime


def fold_bn(net: "DVSGesturePuru", bias_mode: str = "conv") -> "DVSGesturePuru":
    """
    Return a BN-free deep copy of `net`. `net` must be in eval mode so BN uses
    running stats (folding in train mode silently uses batch stats -- the single
    most common cause of a "folding lost accuracy" report).

    bias_mode="conv"      -> folded conv carries b' (exact). This path runs and
                             verifies exactly inside SpikingJelly.
    bias_mode="threshold" -> b' moved into a PER-CHANNEL neuron threshold; conv
                             stays bias-free. This targets the HiAER-Spike
                             conversion, where per-neuron thresholds are native.
                             Note: Custom_LIFNode's jit eval expects a scalar
                             float threshold, so a per-channel tensor will not
                             run through its jit path in-framework -- verify this
                             mode on hardware (or with the numpy reference), and
                             use bias_mode="conv" for in-SpikingJelly checks. Its
                             approximation error is set by |b'|/theta
                             (fold_bias_report); it is only exact at the membrane
                             steady state.
    """
    _require_torch()
    import copy
    if net.training:
        raise RuntimeError("fold_bn requires net.eval() -- fold from running stats, not batch stats.")
    if bias_mode not in ("conv", "threshold"):
        raise ValueError("bias_mode must be 'conv' or 'threshold'")

    folded = copy.deepcopy(net)
    seq = folded.conv_fc
    mods = list(seq)
    new_mods = []
    i = 0
    while i < len(mods):
        m = mods[i]
        is_conv = isinstance(m, (layer.Conv2d, nn.Conv2d))
        nxt = mods[i + 1] if i + 1 < len(mods) else None
        is_bn = isinstance(nxt, (layer.BatchNorm2d, nn.BatchNorm2d))
        if is_conv and is_bn:
            bn = nxt
            eps = bn.eps
            w_prime, b_prime = _fold_conv_bn_params(
                m.weight.data, m.bias.data if m.bias is not None else None,
                bn.weight.data, bn.bias.data, bn.running_mean.data, bn.running_var.data, eps)

            if bias_mode == "conv":
                fused = layer.Conv2d(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride,
                                     padding=m.padding, dilation=m.dilation, groups=m.groups, bias=True)
                fused.weight.data.copy_(w_prime)
                fused.bias.data.copy_(b_prime)
                new_mods.append(fused)
                # the neuron that follows (skip index i+2) is untouched
            else:  # threshold
                fused = layer.Conv2d(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride,
                                     padding=m.padding, dilation=m.dilation, groups=m.groups, bias=False)
                fused.weight.data.copy_(w_prime)
                new_mods.append(fused)
                # push b' into the following neuron's threshold as theta - b'
                node = mods[i + 2]
                th = node.v_threshold                       # property: learned or constant
                base_th = th.detach() if torch.is_tensor(th) else float(th)
                new_th = base_th - b_prime.reshape(-1, 1, 1)   # broadcast over (C,H,W)
                # A learned threshold is now FIXED: after folding, the per-channel
                # constant IS the threshold, so the sigmoid parameter must stop
                # overriding it (otherwise the fold is silently discarded).
                if getattr(node, "learn_threshold", False):
                    node.learn_threshold = False
                    if hasattr(node, "raw_threshold"):
                        node.raw_threshold.requires_grad_(False)
                # store on the node so it moves with .to(device)
                if hasattr(node, "_v_threshold_const"):
                    node._v_threshold_const = new_th
                else:
                    node.v_threshold = new_th
            i += 2  # consumed conv + bn; neuron handled in place
        else:
            new_mods.append(m)
            i += 1

    folded.conv_fc = nn.Sequential(*new_mods)
    if folded.step_mode == "m":
        functional.set_step_mode(folded.conv_fc, "m")

    # The fused Conv2d modules were freshly constructed, so they default to CPU
    # even when `net` is on CUDA -- feeding a GPU batch then raises
    # "Input type (cuda.FloatTensor) and weight type (FloatTensor) differ".
    # Realign the whole folded net to the SOURCE net's device.
    try:
        src_device = next(net.parameters()).device
        folded.to(src_device)
        # threshold-mode stores per-channel theta as a plain attribute, not a
        # registered buffer, so .to() skips it -- move those tensors by hand.
        for mod in folded.conv_fc:
            for attr in ("_v_threshold_const", "v_threshold"):
                val = getattr(mod, attr, None)
                if torch.is_tensor(val):
                    setattr(mod, attr, val.to(src_device))
    except StopIteration:
        pass

    folded._folded = True
    folded._fold_bias_mode = bias_mode
    return folded


def fold_bias_report(net: "DVSGesturePuru") -> list[dict]:
    """
    Per-conv distribution of |b'|/theta -- the number that decides whether
    bias_mode='threshold' is safe. (|b'| ~ 0.1*theta folds cleanly; ~0.5*theta
    does not.) Call on the trained eval-mode net BEFORE folding.
    """
    _require_torch()
    rows = []
    mods = list(net.conv_fc)
    for i, m in enumerate(mods):
        if isinstance(m, (layer.Conv2d, nn.Conv2d)) and i + 1 < len(mods) \
                and isinstance(mods[i + 1], (layer.BatchNorm2d, nn.BatchNorm2d)):
            bn = mods[i + 1]
            _, b_prime = _fold_conv_bn_params(
                m.weight.data, m.bias.data if m.bias is not None else None,
                bn.weight.data, bn.bias.data, bn.running_mean.data, bn.running_var.data, bn.eps)
            node = mods[i + 2] if i + 2 < len(mods) else None
            th = float(node.v_threshold) if node is not None and not torch.is_tensor(node.v_threshold) else 1.0
            ab = b_prime.abs()
            # theta' = theta - b' must land in (0, w_alpha] to be storable.
            # Report how far outside that band the raw fold would fall, which is
            # what TrainSpec.fold_bias_margin exists to prevent during QAT.
            lo, hi = fold_bias_band(th, 0.0)
            below = (b_prime < lo)          # theta' > w_alpha, saturates at INT16_MAX
            above = (b_prime > hi)          # theta' <= 0, fires unconditionally
            rows.append({"conv_index": i, "theta": th,
                         "mean_abs_b_over_theta": (ab.mean() / th).item(),
                         "max_abs_b_over_theta": (ab.max() / th).item(),
                         "theta_prime_min": (th - b_prime.max()).item(),
                         "theta_prime_max": (th - b_prime.min()).item(),
                         "frac_theta_prime_le_0": above.float().mean().item(),
                         "frac_theta_prime_over_alpha": below.float().mean().item()})
    return rows


@torch.no_grad() if _HAS_TORCH else (lambda f: f)
def verify_fold(bn_net, folded_net, loader, device, max_batches: int = None) -> dict:
    """
    The comparison Christopher asked for, in the order that actually localises
    bugs. Test accuracy alone is too coarse to catch a subtly wrong fold.

      1. spike/output agreement: fraction of argmax predictions that match.
         For bias_mode='conv' this should be 1.0 exactly (fp noise aside).
      2. max |logit| difference: ~1e-5 in fp32 for an exact fold; large => bug.
      3. both test accuracies, as the final headline number.
    """
    _require_torch()
    bn_net.eval(); folded_net.eval()
    n = agree = 0
    correct_bn = correct_fold = 0
    max_abs = 0.0
    for bi, (x, y, _l) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        o_bn = forward_over_time(bn_net, x)
        o_fd = forward_over_time(folded_net, x)
        max_abs = max(max_abs, (o_bn - o_fd).abs().max().item())
        p_bn, p_fd = o_bn.argmax(1), o_fd.argmax(1)
        agree += (p_bn == p_fd).sum().item()
        correct_bn += (p_bn == y).sum().item()
        correct_fold += (p_fd == y).sum().item()
        n += y.size(0)
    return {"n": n, "pred_agreement": agree / max(1, n),
            "max_abs_logit_diff": max_abs,
            "acc_bn": correct_bn / max(1, n), "acc_folded": correct_fold / max(1, n),
            "acc_delta": (correct_fold - correct_bn) / max(1, n)}


# =============================================================================
# 5c. Deployment: quantization-aware fine-tune + the report that gates a config
#
# Order matters and is not negotiable: BN must be folded BEFORE weights are
# quantized, because folding is what scales weights out of the representable
# [-1, 1] band. Quantizing pre-fold weights measures a range the chip never
# sees.
#
#   train (float, BN)  ->  fold BN  ->  fake-quantize weights  ->  fine-tune
#
# The fine-tune is what turns "how much does quantization hurt?" into "train a
# network that is already quantized". Clipping stops being damage and becomes a
# constraint the optimizer works inside.
# =============================================================================

DEPLOY_LIMITS = {
    # A config is not deployable if the fold drives any threshold to <= 0: that
    # neuron fires every timestep regardless of input, on chip and in software.
    "min_threshold": 0.0,
    # Soft budget. Above this the fold is pushing real signal outside [-1, 1]
    # and the INT16 grid is throwing it away.
    "max_weight_clip_frac": 0.02,
}


def _quantizable_modules(net):
    for m in net.modules():
        if isinstance(m, (layer.Conv2d, nn.Conv2d, layer.Linear, nn.Linear)):
            yield m


def enable_weight_fake_quant(net):
    """Make every conv/linear weight read back as its INT16 value (STE backward)."""
    _require_torch()
    from torch.nn.utils import parametrize
    for m in _quantizable_modules(net):
        if not parametrize.is_parametrized(m, "weight"):
            parametrize.register_parametrization(m, "weight", _WeightFakeQuant())
    return net


def bake_weight_fake_quant(net):
    """Collapse the parametrization so state_dict holds the quantized weights."""
    _require_torch()
    from torch.nn.utils import parametrize
    for m in _quantizable_modules(net):
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    return net


def deployment_report(net) -> dict:
    """
    What the chip will be handed, and whether it is legal. Call on the FOLDED
    (and ideally quantized) net.
    """
    _require_torch()
    rows, min_th, max_th = [], float("inf"), 0.0
    for i, m in enumerate(net.conv_fc):
        if isinstance(m, HardwareLIFNode):
            # unclamped: the clamp inside v_threshold exists to keep the forward
            # pass sane, not to make an illegal fold look legal
            th = torch.as_tensor(m.raw_v_threshold()).detach().float()
            th_int = m.hw_threshold_int()
            th_int = th_int.float() if torch.is_tensor(th_int) else torch.tensor(float(th_int))
            rows.append({
                "module_index": i,
                "leak": m.hw_leak(),
                "threshold_min": th.min().item(),
                "threshold_max": th.max().item(),
                "threshold_int_min": int(th_int.min().item()),
                "threshold_int_max": int(th_int.max().item()),
                "per_channel": bool(th.numel() > 1),
            })
            min_th = min(min_th, th.min().item())
            max_th = max(max_th, th.max().item())

    clip = weight_clip_fraction(net)
    max_abs_w = 0.0
    for m in _quantizable_modules(net):
        max_abs_w = max(max_abs_w, m.weight.data.abs().max().item())

    # BLOCKING: the network is not representable, so its measured accuracy is
    # meaningless -- score it 0 and move on.
    blocking = []
    if not math.isfinite(min_th):
        blocking.append("no spiking layers found")
    elif min_th <= DEPLOY_LIMITS["min_threshold"]:
        blocking.append(f"folded threshold {min_th:.4f} <= 0 -- neuron fires unconditionally")

    # WARNING: representable but lossy. Deliberately NOT a rejection: the QAT
    # phase trains against the clipping, and hw_val_accuracy already prices in
    # whatever damage survives. Rejecting here would discard configs that clip a
    # little and still deploy well. Surfaced so a high clip rate is diagnosable
    # when a config's quant_gap is large.
    warnings_ = []
    if clip > DEPLOY_LIMITS["max_weight_clip_frac"]:
        warnings_.append(f"{clip*100:.2f}% of folded weights outside [-1, 1] "
                         f"(max |w| = {max_abs_w:.3f}); INT16 grid discards them")
    # The other end of the threshold field. quantized_threshold_int clamps at
    # INT16_MAX, so a theta above w_alpha deploys as w_alpha with no error
    # raised anywhere. Not blocking, since the network still runs, but the
    # deployed threshold is not the trained one.
    if max_th > W_ALPHA:
        warnings_.append(f"max folded threshold {max_th:.4f} > w_alpha "
                         f"({W_ALPHA:g}); it saturates at INT16_MAX on chip")

    return {"neurons": rows, "weight_clip_frac": clip, "max_abs_weight": max_abs_w,
            "min_threshold": (min_th if math.isfinite(min_th) else None),
            "max_threshold": max_th, "deployable": not blocking,
            "blocking_reasons": blocking, "warnings": warnings_,
            "reasons": blocking + warnings_}


def hardware_export(net) -> dict:
    """
    Everything the converter needs, with nothing left to hardcode.

    The conversion script currently pastes `threshold = 32767` and
    `leak_lif = 63` by hand after reading them off a quantization printout.
    Those two lines are why per-layer neurons were impossible. This emits the
    integer threshold and integer leak PER SPIKING LAYER, so the converter can
    build one LIF_neuron per layer -- which hs_api already supports, since
    CRI_network reads a neuron model per neuron key
    (`self.userConnections[neuronKey][modelIdx]`).
    """
    _require_torch()
    layers, li = [], 0
    for m in net.conv_fc:
        if isinstance(m, HardwareLIFNode):
            th_int = m.hw_threshold_int()
            layers.append({
                "layer_index": li,
                "threshold_int": (th_int.reshape(-1).tolist() if torch.is_tensor(th_int)
                                  and th_int.numel() > 1 else int(torch.as_tensor(th_int).reshape(-1)[0])),
                "per_channel_threshold": bool(torch.is_tensor(th_int) and th_int.numel() > 1),
                "leak": m.hw_leak(),
                "shift": 0,
            })
            li += 1
    return {"w_alpha": W_ALPHA, "w_bits": W_BITS, "w_delta": W_DELTA,
            "int16_max": INT16_MAX, "flush_steps": hardware_flush_steps(net),
            "neuron_layers": layers}


def deploy_and_measure(net, cfg, train_loader, val_loader, device, report_fn=None):
    """
    Turn a trained float model into the deployable one and measure IT.

    Returns (hw_net, metrics). `metrics["hw_val_accuracy"]` is the number the
    search optimizes -- post-fold, post-INT16, with the converter's neuron
    dynamics and pipeline latency.
    """
    _require_torch()
    train_cfg: TrainSpec = cfg["train"]
    net.eval()

    folded = fold_bn(net, bias_mode=train_cfg.fold_bias_mode)
    folded.eval()
    acc_folded = evaluate(folded, val_loader, device)

    enable_weight_fake_quant(folded)
    acc_ptq = evaluate(folded, val_loader, device)      # post-training quantization

    # quantization-aware fine-tune: the weights being trained are already the
    # clipped/rounded ones, so the optimizer can route around the grid
    acc_qat = acc_ptq
    if train_cfg.qat_epochs > 0 and train_loader is not None:
        qat_cfg = deepcopy(train_cfg)
        qat_cfg.lr = train_cfg.lr * train_cfg.qat_lr_scale
        qat_cfg.scheduler = "cosine"
        qat_cfg.warmup_epochs = 0
        qat_cfg.epochs = train_cfg.qat_epochs
        opt = build_optimizer(folded, qat_cfg)
        sched, per_batch = build_scheduler(opt, qat_cfg, len(train_loader))
        crit = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
        best_state = None
        for ep in range(train_cfg.qat_epochs):
            folded.train()
            train_one_epoch(folded, train_loader, opt, device, crit,
                            grad_clip=train_cfg.grad_clip,
                            batch_scheduler=sched if per_batch else None)
            if sched is not None and not per_batch:
                sched.step()
            a = evaluate(folded, val_loader, device)
            if a > acc_qat:
                acc_qat = a
                best_state = deepcopy(folded.state_dict())
            if report_fn is not None:
                report_fn(phase="qat", epoch=ep, hw_val_acc=a)
        if best_state is not None:
            folded.load_state_dict(best_state)

    bake_weight_fake_quant(folded)
    folded.eval()
    hw_acc = evaluate(folded, val_loader, device)

    rep = deployment_report(folded)
    metrics = {
        "hw_val_accuracy": hw_acc,
        "folded_val_accuracy": acc_folded,
        "ptq_val_accuracy": acc_ptq,
        "qat_val_accuracy": acc_qat,
        "weight_clip_frac": rep["weight_clip_frac"],
        "max_abs_weight": rep["max_abs_weight"],
        "min_threshold": rep["min_threshold"],
        "deployable": rep["deployable"],
        "deploy_reasons": "; ".join(rep["blocking_reasons"])[:300],
        "deploy_warnings": "; ".join(rep["warnings"])[:300],
        "flush_steps": hardware_flush_steps(folded),
    }
    return folded, metrics


# =============================================================================
# 6. Data: resize + binarize, dataset/loader construction
# =============================================================================

class DVSResizeAndBinarize:
    """
    Resizes each frame (if size is given) and binarizes to {0,1}.

    Note this is NOT average-downsampling: bilinear interpolation of a sparse
    binary frame produces small fractional values, and `> 0` promotes any
    nonzero contribution back to a full spike. Effectively OR-pooling -- no
    events are lost, but event density per pixel rises at low resolutions.
    """
    def __init__(self, size=None):
        self.size = size

    def __call__(self, frames):
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        T, C, H, W = frames.shape
        if self.size is not None:
            out = torch.zeros((T, C, *self.size), dtype=frames.dtype)
            for t in range(T):
                out[t] = F.interpolate(frames[t].unsqueeze(0).float(), size=self.size,
                                        mode="bilinear", align_corners=False).squeeze(0)
        else:
            out = frames.float()
        return (out > 0).float()


def build_dataloaders(cfg: dict, data_dir: str, val_fraction: float = 0.15, num_workers: int = 0):
    # num_workers=0 on purpose: each Ray trial is already its own subprocess
    # (only given 1 CPU via tune.with_resources). DataLoader workers spawn
    # ANOTHER layer of subprocesses on top -- nested multiprocessing like that
    # is broken on Windows and would oversubscribe CPUs across concurrent trials.
    _require_torch()
    input_cfg = cfg["input"]
    resize_size = (input_cfg.resize_to, input_cfg.resize_to) if input_cfg.resize_to else None
    transform = DVSResizeAndBinarize(size=resize_size)

    full_train = DVS128Gesture(root=data_dir, frames_number=input_cfg.T, split_by="number",
                                train=True, data_type="frame", transform=transform)
    test_set = DVS128Gesture(root=data_dir, frames_number=input_cfg.T, split_by="number",
                              train=False, data_type="frame", transform=transform)

    n = len(full_train)
    n_val = int(val_fraction * n)
    torch.manual_seed(1)
    indices = torch.randperm(n)
    train_set = Subset(full_train, indices[n_val:])
    val_set = Subset(full_train, indices[:n_val])

    kwargs = dict(batch_size=input_cfg.N, pin_memory=True,
                  collate_fn=pad_sequence_collate, num_workers=num_workers)
    # drop_last ONLY on train. On val/test it silently discards the remainder
    # (up to N-1 samples) and biases every accuracy number reported.
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader, test_loader


# =============================================================================
# 7. Optimizer / scheduler construction
# =============================================================================

def build_optimizer(net, train_cfg: TrainSpec):
    name = train_cfg.optimizer.lower()
    if name == "adam":
        # NB: weight_decay in Adam is classic L2 added to the gradient, which
        # interacts with the adaptive step. Use adamw for decoupled decay.
        return torch.optim.Adam(net.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(net.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(net.parameters(), lr=train_cfg.lr, momentum=train_cfg.momentum,
                               weight_decay=train_cfg.weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer {train_cfg.optimizer!r}")


def build_scheduler(optimizer, train_cfg: TrainSpec, steps_per_epoch: int):
    """Returns (scheduler, step_per_batch). `None` scheduler = constant LR."""
    name = train_cfg.scheduler.lower()
    epochs = train_cfg.epochs

    if name == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=train_cfg.lr, epochs=epochs,
            steps_per_epoch=max(1, steps_per_epoch),
        )
        return sched, True

    if name == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - train_cfg.warmup_epochs))
    elif name == "step":
        main = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=train_cfg.step_gamma)
    elif name == "none":
        main = None
    else:
        raise ValueError(f"Unknown scheduler {train_cfg.scheduler!r}")

    if train_cfg.warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=train_cfg.warmup_epochs)
        if main is None:
            return warm, False
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, main], milestones=[train_cfg.warmup_epochs]), False

    return main, False


# =============================================================================
# 8. Train / evaluate -- loop over T manually; filters never see T, only the
#    neuron's internal state does.
# =============================================================================

def hardware_flush_steps(net) -> int:
    """
    Extra zero-input timesteps needed to drain the pipeline, = the number of
    spiking layers.

    HardwareLIFNode fires on the membrane carried in from the previous step
    (the converter's order), so each spiking layer delays its output by one
    timestep. The conversion script does exactly this: T input frames, then
    `num_layers` steps with an empty input list, accumulating output spikes
    throughout and dividing by T. Without the flush, the last frames' evidence
    never reaches the classifier and the deployed accuracy is lower than
    anything measured in training -- for no reason visible in the config.
    """
    return sum(1 for m in net.conv_fc if isinstance(m, HardwareLIFNode))


def forward_over_time(net, x, flush_steps: int = None):
    """
    x: (N, T, C, H, W) -> output spike rate, neuron state reset afterwards.

    Mirrors the conversion script's measurement exactly: accumulate output over
    T input steps PLUS `flush_steps` zero-input steps, then divide by T (not by
    T + flush) so the rate stays comparable across depths.

    Single-step nets are driven by an explicit loop; multi-step nets (tdbn) get
    the whole (T, N, ...) tensor at once because tdBN's statistics span T.
    """
    x = x.transpose(0, 1)  # (T, N, C, H, W)
    T = x.shape[0]
    if flush_steps is None:
        flush_steps = hardware_flush_steps(net)

    if getattr(net, "step_mode", "s") == "m":
        if flush_steps > 0:
            pad = torch.zeros_like(x[:1]).expand(flush_steps, *x.shape[1:])
            x = torch.cat([x, pad], dim=0)
        out = net(x).sum(0) / T
    else:
        out_sum = 0.0
        for t in range(T):
            out_sum = out_sum + net(x[t])
        if flush_steps > 0:
            zeros = torch.zeros_like(x[0])
            for _ in range(flush_steps):
                out_sum = out_sum + net(zeros)
        out = out_sum / T
    functional.reset_net(net)
    return out


def train_one_epoch(net, loader, optimizer, device, criterion=None,
                    grad_clip: float = 0.0, batch_scheduler=None):
    net.train()
    criterion = criterion or (lambda o, y: F.cross_entropy(o, y))
    total, correct, loss_sum = 0, 0, 0.0
    for x, y, _lengths in loader:  # pad_sequence_collate returns (data, labels, lengths);
                                    # lengths is always == T here since split_by="number"
                                    # gives every sample a fixed frame count -- safe to ignore.
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = forward_over_time(net, x)
        loss = criterion(out, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


def evaluate(net, loader, device):
    net.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y, _lengths in loader:
            x, y = x.to(device), y.to(device)
            out = forward_over_time(net, x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def run_training(cfg: dict, data_dir: str, device=None, report_fn=None, ckpt_path: str = None) -> float:
    """Shared training loop used by both `single` mode and each Ray trial.
    If ckpt_path is given, the best-val model weights are saved there for later
    folding/deployment."""
    _require_torch()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg: TrainSpec = cfg["train"]

    net = build_model(cfg).to(device)
    train_loader, val_loader, _ = build_dataloaders(cfg, data_dir=data_dir)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

    mode = getattr(train_cfg, "qat_mode", "inline")

    # ---- how the epoch budget is split -----------------------------------
    if mode == "inline":
        warmup = max(1, round(train_cfg.epochs * train_cfg.qat_warmup_frac))
        warmup = min(warmup, train_cfg.epochs - 1) if train_cfg.epochs > 1 else 1
        grid_epochs = train_cfg.epochs - warmup
    else:
        warmup, grid_epochs = train_cfg.epochs, 0   # tail/ptq handled below

    def run_phase(net, optimizer, scheduler, step_per_batch, n_epochs, epoch0,
                  best, best_state, tag):
        """One training phase; reports each epoch into the shared trajectory so
        ASHA still sees a continuous per-epoch curve across warmup + grid."""
        for e in range(n_epochs):
            train_loss, train_acc = train_one_epoch(
                net, train_loader, optimizer, device, criterion,
                grad_clip=train_cfg.grad_clip,
                batch_scheduler=scheduler if step_per_batch else None)
            if scheduler is not None and not step_per_batch:
                scheduler.step()
            val_acc = evaluate(net, val_loader, device)
            if val_acc > best:
                best = val_acc
                best_state = deepcopy(net.state_dict())
            if report_fn is not None:
                report_fn(epoch=epoch0 + e, train_loss=train_loss, train_acc=train_acc,
                          val_acc=val_acc, best_val_acc=best,
                          lr=optimizer.param_groups[0]["lr"], phase=tag)
        return best, best_state

    # ---- phase 1: float warmup (BN active, precise weights) --------------
    optimizer = build_optimizer(net, train_cfg)
    scheduler, step_per_batch = build_scheduler(optimizer, train_cfg, len(train_loader))
    best, best_state = 0.0, None
    best, best_state = run_phase(net, optimizer, scheduler, step_per_batch,
                                 warmup, 0, best, best_state, tag="float")
    float_best = best

    if mode == "inline":
        # ---- phase 2: fold inline, train the rest ON the grid ------------
        if best_state is not None:
            net.load_state_dict(best_state)          # grid-train from best warmup
        net.eval()                                   # fold from running stats
        net.to_qat_folded(bias_mode=train_cfg.fold_bias_mode,
                          fold_bias_margin=getattr(train_cfg, "fold_bias_margin", 0.05))
        net.to(device)

        grid_cfg = deepcopy(train_cfg)
        grid_cfg.lr = train_cfg.lr * train_cfg.qat_lr_scale
        grid_cfg.warmup_epochs = 0
        grid_cfg.scheduler = "cosine"
        grid_cfg.epochs = max(1, grid_epochs)
        gopt = build_optimizer(net, grid_cfg)
        gsched, gstep = build_scheduler(gopt, grid_cfg, len(train_loader))

        grid_best, grid_state = 0.0, None
        if grid_epochs > 0:
            grid_best, grid_state = run_phase(net, gopt, gsched, gstep,
                                              grid_epochs, warmup, 0.0, None, tag="grid")
            if grid_state is not None:
                net.load_state_dict(grid_state)

        # ---- freeze into the deployable BN-free model & measure IT -------
        net.eval()
        hw_net = deepcopy(net).export_deployed().to(device)
        hw_acc = evaluate(hw_net, val_loader, device)
        rep = deployment_report(hw_net)
        hw = {
            "hw_val_accuracy": hw_acc,
            "grid_val_accuracy": grid_best,
            "weight_clip_frac": rep["weight_clip_frac"],
            "max_abs_weight": rep["max_abs_weight"],
            "min_threshold": rep["min_threshold"],
            "deployable": rep["deployable"],
            "deploy_reasons": "; ".join(rep["blocking_reasons"])[:300],
            "deploy_warnings": "; ".join(rep["warnings"])[:300],
            "flush_steps": hardware_flush_steps(hw_net),
        }
    else:
        # ---- tail / ptq: old behaviour -----------------------------------
        if best_state is not None:
            net.load_state_dict(best_state)
        hw_net, hw = deploy_and_measure(net, cfg, train_loader, val_loader, device)

    hw["float_val_accuracy"] = float_best
    hw["quant_gap"] = float_best - hw["hw_val_accuracy"]

    if ckpt_path is not None:
        torch.save({
            "state_dict": net.state_dict(),
            "hw_state_dict": hw_net.state_dict(),
            "hardware_export": hardware_export(hw_net),
            "encoder_layers_json": cfg["encoder"].layers_json,
            "config": {n: asdict(s) for n, s in cfg.items()},
            "metrics": hw,
            "folded": True, "bias_mode": train_cfg.fold_bias_mode,
            "qat_mode": mode,
        }, ckpt_path)
    return {"float_val_accuracy": float_best, "hw_net": hw_net, **hw}


# =============================================================================
# 9. Dataset cache warmup (unchanged behaviour)
# =============================================================================

@contextlib.contextmanager
def _quiet_with_heartbeat(label: str, every_s: int = 20):
    """
    spikingjelly prints one line per saved .npz file during extraction --
    hundreds of lines that look like a hang, not progress. This CAPTURES
    stdout into a buffer for the duration and prints a single calm heartbeat
    line every `every_s` seconds instead. On failure the buffer is dumped in
    full before the exception propagates, so diagnostics are never lost.
    """
    import io

    stop = threading.Event()
    start = time.time()
    real_stdout = sys.stdout   # captured BEFORE redirecting
    buf = io.StringIO()

    def heartbeat():
        while not stop.wait(every_s):
            print(f"[warmup] {label} -- still working ({int(time.time() - start)}s elapsed)", file=real_stdout)
            real_stdout.flush()

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    sys.stdout = buf
    failed = False
    try:
        yield
    except Exception:
        failed = True
        raise
    finally:
        sys.stdout = real_stdout
        stop.set()
        t.join()
        elapsed = int(time.time() - start)
        if failed:
            print(f"[warmup] {label} -- FAILED after {elapsed}s. Captured output:")
            print(buf.getvalue())
        else:
            print(f"[warmup] {label} -- done ({elapsed}s)")


def _class_dirs_have_npz(split_dir: str, num_classes: int = 11) -> bool:
    """True only if every class subfolder 0..num_classes-1 exists and has at
    least one .npz sample. Tells a genuinely finished cache apart from one
    that is merely present-but-half-built."""
    if not os.path.isdir(split_dir):
        return False
    for c in range(num_classes):
        class_dir = os.path.join(split_dir, str(c))
        if not os.path.isdir(class_dir):
            return False
        if not any(f.endswith(".npz") for f in os.listdir(class_dir)):
            return False
    return True


def _extract_is_complete(data_dir: str) -> bool:
    return os.path.isfile(os.path.join(data_dir, "extract", "DvsGesture", "trials_to_train.txt"))


def _events_np_is_complete(data_dir: str) -> bool:
    base = os.path.join(data_dir, "events_np")
    return _class_dirs_have_npz(os.path.join(base, "train")) and _class_dirs_have_npz(os.path.join(base, "test"))


def _frame_cache_is_complete(data_dir: str, T: int) -> bool:
    base = os.path.join(data_dir, f"frames_number_{T}_split_by_number")
    return _class_dirs_have_npz(os.path.join(base, "train")) and _class_dirs_have_npz(os.path.join(base, "test"))


def warmup_dataset_cache(data_dir: str, T_values: list):
    """
    Build spikingjelly's frame-conversion cache for every T value in the search,
    sequentially, BEFORE any parallel trial starts. Caching depends only on
    (frames_number, split_by) -- transforms are applied lazily per-sample -- so
    warming up T alone is sufficient. Without this, parallel trials sampling the
    same T race to build the same shared folder and can leave it corrupted.
    """
    _require_torch()
    extract_ok = _extract_is_complete(data_dir)
    events_np_ok = extract_ok and _events_np_is_complete(data_dir)

    if events_np_ok:
        print("[warmup] extract/ and events_np/ already complete -- skipping rebuild")
    else:
        for name in ("extract", "events_np"):
            stale = os.path.join(data_dir, name)
            if os.path.isdir(stale):
                print(f"[warmup] {name}/ missing or incomplete -- removing {stale}")
                shutil.rmtree(stale, ignore_errors=True)

    for T in T_values:
        if _frame_cache_is_complete(data_dir, T):
            print(f"[warmup] frame cache for T={T} already complete -- skipping")
            continue
        stale = os.path.join(data_dir, f"frames_number_{T}_split_by_number")
        if os.path.isdir(stale):
            print(f"[warmup] frame cache for T={T} incomplete -- removing {stale}")
            shutil.rmtree(stale, ignore_errors=True)
        with _quiet_with_heartbeat(f"building frame cache for T={T} (train split)"):
            DVS128Gesture(root=data_dir, frames_number=T, split_by="number", train=True, data_type="frame")
        with _quiet_with_heartbeat(f"building frame cache for T={T} (test split)"):
            DVS128Gesture(root=data_dir, frames_number=T, split_by="number", train=False, data_type="frame")

    print("[warmup] dataset cache ready for all T values:", T_values)


# =============================================================================
# 10. Ray + Optuna search
# =============================================================================

# NARROWED after the 65-trial analysis (see DVS_Search_Statistical_Analysis).
# 512 was the only hidden-FC width that showed up in a strong trial; the head is
# rarely used at all now (fc_layers capped at 1), so keep a small set.
FC_WIDTH_CHOICES = [128, 256, 512]


def config_to_specs(config: dict) -> dict:
    """Flat Ray/Optuna config dict -> the structured cfg every other function takes.

    Encoder is assembled per-layer when per-layer keys (k_0, ch_0, ...) are
    present, otherwise from the uniform depth/channels/kernel_size fields."""
    import json
    n_fc = int(config.get("fc_layers", 1))
    widths = [str(config[f"fc_width_{i}"]) for i in range(n_fc) if f"fc_width_{i}" in config]
    norm = config.get("norm", "bn")
    depth = int(config["depth"])

    if "ch_0" in config:  # per-layer search emitted individual layer keys
        layers = []
        for i in range(depth):
            layers.append({
                "out_channels": config[f"ch_{i}"],
                "kernel_size": config[f"k_{i}"],
                "stride": config.get(f"stride_{i}", 1),
                "pool": bool(config.get(f"pool_{i}", False)),
            })
        encoder = EncoderSpec(layers_json=json.dumps(layers), bias=(norm == "none"),
                              norm=norm, tdbn_alpha=config.get("tdbn_alpha", 1.0))
    else:                 # uniform
        encoder = EncoderSpec(depth=depth, channels=config["channels"],
                              kernel_size=config["kernel_size"], stride=config.get("stride", 1),
                              padding=0, dilation=1, bias=(norm == "none"),
                              norm=norm, tdbn_alpha=config.get("tdbn_alpha", 1.0))

    return {
        "input": InputSpec(N=config["N"], C=2, H=128, W=128,
                           T=config["T"], resize_to=config["resize_to"]),
        "encoder": encoder,
        "output": OutputSpec(num_classes=11),
        "downsample": DownsampleSpec(mode=config.get("downsample_mode", "stride"),
                                     pool_type=config.get("pool_type", "avg"),
                                     pool_kernel_size=2, pool_stride=2),
        "head": HeadSpec(final_reduction=config.get("final_reduction", "flatten"),
                         fc_widths=",".join(widths),
                         dropout_rate=config.get("dropout_rate", 0.5)),
        # v_threshold starts at 1.0 = exactly INT16_MAX after quantization; the
        # sigmoid parametrization keeps it inside (0, 1] if it is learned.
        "neuron": NeuronSpec(neuron_type="LIF", tau=int(round(float(config["tau"]))),
                             v_threshold=1.0, v_reset=0.0,
                             trainable_tau=bool(config.get("trainable_tau", False)),
                             trainable_threshold=bool(config.get("trainable_threshold", False))),
        "train": TrainSpec(epochs=config["epochs"],
                           optimizer=config.get("optimizer", "adam"),
                           lr=config.get("lr", 1e-3),
                           weight_decay=config.get("weight_decay", 0.0),
                           scheduler=config.get("scheduler", "cosine"),
                           warmup_epochs=config.get("warmup_epochs", 0),
                           label_smoothing=config.get("label_smoothing", 0.0),
                           grad_clip=config.get("grad_clip", 0.0),
                           qat_mode=config.get("qat_mode", "inline"),
                           qat_warmup_frac=config.get("qat_warmup_frac", 0.25),
                           qat_epochs=config.get("qat_epochs", 4),
                           qat_lr_scale=config.get("qat_lr_scale", 0.5),
                           fold_bias_mode=config.get("fold_bias_mode", "threshold"),
                           fold_bias_margin=config.get("fold_bias_margin", 0.05)),
    }


# NARROWED to the regions the statistics said matter:
#   channels: 128/256 were the dominant INFEASIBILITY driver (chi-square p=6e-4,
#     Cramer's V=0.59: 32ch 88% feasible, 128ch 14%) and never beat 32/64 on
#     accuracy. Dropped.
#   kernel: every top-10 model used k5 or k7; k3 dropped.
CHANNEL_CHOICES = [32, 64]
KERNEL_CHOICES = [5, 7]


class DefineByRunSpace:
    """
    Optuna define-by-run space, as a MODULE-LEVEL CALLABLE OBJECT rather than a
    closure.

    Why a class: Ray Tune periodically checkpoints the search algorithm, and
    OptunaSearch.save() pickles the space object. A function defined inside
    another function is a local object that pickle cannot serialise
    ("Can't get local object 'make_define_by_run.<locals>.define_by_run'"), which
    aborts the whole study at the first checkpoint. An instance of a top-level
    class pickles fine -- its state is just the plain ints/strs/lists below.

    Define-by-run (vs a flat dict) is what lets the space be honestly
    conditional: a per-layer kernel size k_3 is only suggested on trials whose
    depth >= 4, and fc_width_1 only when there is a 2nd FC layer -- so Optuna's
    surrogate never sees a dimension that had no effect on that trial's score.

    per_layer=True  -> kernel/stride/channels/pool are sampled INDEPENDENTLY per
                       conv layer (answers "is kernel size independently
                       optimised?" -- yes).
    per_layer=False -> one kernel/channels/stride shared by all layers (the
                       older uniform space), useful as a cheaper baseline.
    """

    def __init__(self, batch_size: int, epochs: int, data_dir_abs: str,
                 t_choices: list, per_layer: bool = True):
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.data_dir_abs = str(data_dir_abs)
        self.t_choices = list(t_choices)
        self.per_layer = bool(per_layer)

    def __call__(self, trial):
        batch_size = self.batch_size
        epochs = self.epochs
        data_dir_abs = self.data_dir_abs
        t_choices = self.t_choices
        per_layer = self.per_layer
        # ---- encoder depth ----
        # winners are depth 2-3 (Kruskal-Wallis favoured shallow, q=0.08); depth
        # 5 only ever reached 0.77. Keep 2-4 so depth-3 branches stay in play.
        depth = trial.suggest_int("depth", 2, 4)

        if per_layer:
            # independent geometry per layer
            for i in range(depth):
                trial.suggest_categorical(f"k_{i}", KERNEL_CHOICES)
                trial.suggest_categorical(f"ch_{i}", CHANNEL_CHOICES)
                # each layer independently: downsample by stride-2, by pooling,
                # or not at all (stride 1, no pool -> size-preserving).
                # each layer independently: downsample by stride-2, by pooling,
                # or not at all. The chosen branch sets the flat key
                # config_to_specs reads; the others stay absent and default off.
                ds = trial.suggest_categorical(f"ds_{i}", ["stride", "pool", "none"])
                if ds == "stride":
                    trial.suggest_int(f"stride_{i}", 2, 2)
                elif ds == "pool":
                    trial.suggest_int(f"pool_{i}", 1, 1)
        else:
            trial.suggest_categorical("channels", CHANNEL_CHOICES)
            trial.suggest_categorical("kernel_size", KERNEL_CHOICES)
            mode = trial.suggest_categorical("downsample_mode", ["stride", "pool"])
            if mode == "stride":
                # stride=1 never downsamples -> the flatten explodes and the
                # config is infeasible every time (all the fc_in fan-in busts in
                # the data came from here). Force stride-2.
                trial.suggest_categorical("stride", [2])

        # resize_to=0 (native 128x128) is omitted: 128*128*2 = 32,768 axons is
        # over the 16,383 limit, so it could never pass feasibility.
        trial.suggest_categorical("resize_to", [32, 64])
        trial.suggest_categorical("T", t_choices)

        # ---- head: GAP vs flatten, then variable-depth FC ----
        # GAP collapses HxW before the head (Q4): far fewer params, less
        # overfitting, and deployable. When GAP is chosen the huge first-FC
        # fan-in disappears, so many more configs pass feasibility.
        # final_reduction fixed to flatten (set in the constants below): every
        # top model used it, and GAP historically cost ~40 pts on this task.
        # fc_layers 0-1 only: every top-10 model had 0 hidden FC (Kruskal-Wallis
        # q=0.04), so 2-3 hidden layers are pure waste. Keep 1 as a branch.
        n_fc = trial.suggest_int("fc_layers", 0, 1)
        for i in range(n_fc):
            trial.suggest_categorical(f"fc_width_{i}", FC_WIDTH_CHOICES)

        # ---- neuron: INTEGER leak, per-layer (Q2, Q3) ------------------------
        # tau was suggest_float(1.5, 2.5). On chip the leak register is an
        # integer and the converter's convention is leak = round(tau), so EVERY
        # value in that interval deployed as leak=2: the search was resolving a
        # difference that does not survive conversion, and the float metric was
        # rewarding it. The old Spearman rho=-0.60 for tau is therefore a
        # correlation with a coordinate the hardware cannot represent -- it does
        # not carry over, which is why the range is reopened rather than
        # narrowed around the old winners.
        trial.suggest_categorical("tau", HW_TAU_CHOICES)
        # Now genuinely A/B-able: both are quantized in the forward pass, and
        # hs_api reads a neuron model per neuron key, so per-layer leaks and
        # thresholds deploy as-is.
        trial.suggest_categorical("trainable_tau", [False, True])
        trial.suggest_categorical("trainable_threshold", [False, True])

        # ---- regularization ----
        # dropout: not significant, but winners live in 0.1-0.45; trim the tails.
        trial.suggest_float("dropout_rate", 0.1, 0.45)
        # norm=none is GONE: it produced every dead (below-chance) network
        # (pooled Mann-Whitney p=0.0015; learned-vs-dead odds ratio 21.8). Only
        # the two foldable, hardware-legal options remain -- and this lets tdBN
        # get a fair test against plain BN on the good backbone.
        norm = trial.suggest_categorical("norm", ["bn", "tdbn"])
        if norm == "tdbn":
            trial.suggest_float("tdbn_alpha", 0.5, 2.0)
        trial.suggest_float("label_smoothing", 0.0, 0.2)
        trial.suggest_categorical("grad_clip", [0.0, 1.0, 5.0])

        # ---- optimizer / LR schedule ----
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        # lr: positive correlation (Spearman q=0.04); winners are 1.6e-3..3.5e-3,
        # so drop the under-converging low end (was 1e-4).
        trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        # weight_decay is the main lever on whether FOLDED weights fit inside the
        # representable [-1, 1] band (w_alpha=1). The old ranges were fitted
        # against a float objective that could not see clipping, so their floors
        # (1e-8) are almost certainly too low now that clipping is scored --
        # negligible decay lets the folded distribution sprawl past the grid.
        # Floors raised; the search re-decides the rest against hw_val_accuracy.
        if optimizer == "adamw":
            trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)    # decoupled
        else:
            trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        sched = trial.suggest_categorical("scheduler", ["cosine", "onecycle", "step", "none"])
        if sched in ("cosine", "step"):
            trial.suggest_int("warmup_epochs", 0, 3)

        # final_reduction pinned here (not searched) -- see the head note above.
        return {"N": batch_size, "epochs": epochs, "data_dir": data_dir_abs,
                "pool_type": "avg", "final_reduction": "flatten"}


def make_define_by_run(batch_size: int, epochs: int, data_dir_abs: str, t_choices: list,
                       per_layer: bool = True) -> "DefineByRunSpace":
    """Factory kept for API compatibility -- returns a picklable space object."""
    return DefineByRunSpace(batch_size, epochs, data_dir_abs, t_choices, per_layer)


# Known-good configs to evaluate FIRST, so the search branches out from the
# winners instead of rediscovering them. Each dict must match the UNIFORM space
# exactly -- only the params that space suggests on that config's path, no extras
# (e.g. norm="bn" => no tdbn_alpha; fc_layers=0 => no fc_width_i; scheduler
# "none" => no warmup_epochs). All values sit inside the narrowed ranges above.
#
# CAVEAT, and it is a big one: these accuracies were measured against the OLD
# objective -- float, SpikingJelly neuron order, decay_input=True, continuous
# tau, no fold, no INT16. Their BACKBONE geometry (depth 2-3, 32ch, k5/k7,
# resize 64, no hidden FC) is still a sensible prior and is why they are kept.
# Their neuron and regularization settings are not: tau 1.69 was never a real
# setting, and weight decay ~1e-8 was chosen under a metric that could not see
# weight clipping. Those fields are moved onto the legal grid below, so treat
# these as geometry seeds whose scores will not reproduce.
SEED_CONFIGS = [
    # depth-2, k7 backbone (87.5% under the old float objective)
    dict(depth=2, channels=32, kernel_size=7, downsample_mode="stride", stride=2,
         resize_to=64, T=16, fc_layers=0, tau=2, trainable_tau=False,
         trainable_threshold=False, dropout_rate=0.28, norm="bn",
         label_smoothing=0.15, grad_clip=0.0, optimizer="adam", lr=0.0029,
         weight_decay=1e-5, scheduler="none"),
    # same backbone, k5 branch, large-leak end of the grid (matches the working
    # DVS conversion, where leak=63 made rounding error negligible)
    dict(depth=2, channels=32, kernel_size=5, downsample_mode="stride", stride=2,
         resize_to=64, T=16, fc_layers=0, tau=63, trainable_tau=False,
         trainable_threshold=False, dropout_rate=0.29, norm="bn",
         label_smoothing=0.16, grad_clip=0.0, optimizer="adam", lr=0.0035,
         weight_decay=1e-5, scheduler="none"),
    # depth-3 branch (does one more conv block survive the extra flush step?)
    dict(depth=3, channels=32, kernel_size=7, downsample_mode="stride", stride=2,
         resize_to=64, T=16, fc_layers=0, tau=4, trainable_tau=False,
         trainable_threshold=False, dropout_rate=0.35, norm="bn",
         label_smoothing=0.15, grad_clip=0.0, optimizer="adam", lr=0.0021,
         weight_decay=1e-4, scheduler="none"),
]


# =============================================================================
# 10b. Incremental results writer  (re-added -- the rewrite had dropped this)
#
# export_trial_records() below only runs AFTER tuner.fit() returns, so a crash
# or Ctrl-C 30 hours into a search leaves you nothing. ResultsWriter streams to
# disk as the run happens: flush + fsync after every record, so the file is
# always complete between writes and `tail`-able live. The two coexist --
# streaming for safety/live-watching, export_trial_records for the final flat
# table.
#
# Ray trials are SEPARATE PROCESSES, so per-trial streaming is written by a
# DRIVER-side Tune callback (single process, no interleaving), never from inside
# the trainable.
# =============================================================================

class ResultsWriter:
    """Append-only, flush-on-every-write result sink rooted at a directory."""

    def __init__(self, root: str, echo: bool = True):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.echo = echo
        self.started = time.time()

    def path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def append_jsonl(self, name: str, record: dict) -> dict:
        record = {"wall_time": round(time.time() - self.started, 3), **record}
        with open(self.path(name), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    def write_json(self, name: str, obj):
        tmp = self.path(name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path(name))     # atomic: never a half-written best.json

    def write_text(self, name: str, text: str):
        tmp = self.path(name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path(name))

    def append_csv(self, name: str, row: dict, header_order=None):
        p = self.path(name)
        new = not os.path.exists(p)
        with open(p, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header_order or list(row), extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
            fh.flush()

    def log(self, msg: str):
        if self.echo:
            print(msg, flush=True)


def default_results_dir(mode: str, explicit: str = None) -> str:
    if explicit:
        return explicit
    return os.path.join("results", f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def validate_data_dir(data_dir: str) -> str:
    """
    Fail early and legibly on a bad --data-dir.

    spikingjelly calls os.mkdir(root/'download') without creating parents, so a
    root that does not exist surfaces as a bare `FileNotFoundError: [WinError 3]
    ... '<root>\\download'` from deep in the library, which does not point at the
    path you typed. Check it here instead. Returns the absolute path.
    """
    if not data_dir or data_dir.strip(".") == "":
        raise SystemExit(f"--data-dir is a placeholder, not a path: {data_dir!r}\n"
                         "  Pass the real DVS128 Gesture root folder.")
    abs_dir = os.path.abspath(os.path.expanduser(data_dir))
    if not os.path.isdir(abs_dir):
        raise SystemExit(
            f"--data-dir does not exist:\n    {abs_dir}\n\n"
            "  It must be the DVS128 Gesture ROOT folder, containing:\n"
            "      <root>/download/DvsGesture.tar.gz\n"
            "      <root>/download/gesture_mapping.csv\n"
            "  spikingjelly builds extract/, events_np/ and frames_number_*/ beside them.")
    # already-built caches mean the raw archive is no longer needed
    if os.path.isdir(os.path.join(abs_dir, "extract")) or os.path.isdir(os.path.join(abs_dir, "events_np")):
        return abs_dir
    tarball = os.path.join(abs_dir, "download", "DvsGesture.tar.gz")
    if not os.path.isfile(tarball):
        try:
            found = ", ".join(sorted(os.listdir(abs_dir))[:12]) or "(empty)"
        except OSError:
            found = "(unreadable)"
        raise SystemExit(
            f"--data-dir exists but has no dataset in it:\n    {abs_dir}\n\n"
            f"  Expected:\n      {tarball}\n  Found instead: {found}\n\n"
            "  If your root is one level deeper (a folder of the same name inside "
            "itself), point --data-dir at the inner one.")
    return abs_dir


def export_trial_records(results, out_dir: str) -> str:
    """
    Flatten every trial into one self-contained table for statistical analysis
    (variance across seeds, which knobs move accuracy, feasibility rate, the
    accuracy/neuron-count Pareto front). Ray keeps its own per-trial logs, but a
    single flat file with {full config} x {final metrics} x {arch counts} is what
    you actually load into pandas.

    Writes both trial_records.jsonl (loss-less) and trial_records.csv (convenient).
    Returns the CSV path.
    """
    import csv, json as _json
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "trial_records.jsonl")
    csv_path = os.path.join(out_dir, "trial_records.csv")

    rows = []
    for r in results:
        cfg_flat = dict(r.config)
        metrics = dict(r.metrics) if r.metrics else {}
        # derive architecture counts even for trials that never trained, so
        # feasibility analysis covers the whole sample, not just trained ones.
        arch = {}
        try:
            specs = config_to_specs(cfg_flat)
            feasible, violations = check_feasibility(
                specs["input"], specs["encoder"], specs["downsample"], specs["head"], specs["output"])
            arch["feasible"] = feasible
            arch["violations"] = "; ".join(violations)[:300]
            if feasible:
                c = count_neurons_and_synapses(specs)["totals"]
                arch.update(neurons=c["neurons"], connections=c["connections"], params=c["params"])
        except Exception as e:
            arch["arch_error"] = str(e)[:200]
        rows.append({"trial_id": getattr(r, "trial_id", None),
                     **{f"cfg.{k}": v for k, v in cfg_flat.items()},
                     **{f"metric.{k}": v for k, v in metrics.items()},
                     **arch})

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(_json.dumps(row, default=str) + "\n")
    keys = sorted({k for row in rows for k in row})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[records] wrote {len(rows)} trials -> {csv_path}")
    return csv_path


def trainable(config: dict):
    """Ray Tune trainable. `config` is a flat dict of sampled hyperparameters."""
    from ray import tune

    cfg = config_to_specs(config)

    # tier 1: feasibility -- free, before touching a GPU. Also catches configs
    # whose feature map collapses, which used to crash the trial instead.
    feasible, violations = check_feasibility(
        cfg["input"], cfg["encoder"], cfg["downsample"], cfg["head"], cfg["output"])
    if not feasible:
        # ASHA prunes on float_val_accuracy and Ray strictly requires that key on
        # EVERY report, including this instant-reject one. An infeasible config is
        # the worst possible outcome, so report 0.0 for both the float trajectory
        # and the selection metric. phase="deploy" keeps it out of the "pruned
        # before deploy" bucket -- it never trained, it was rejected.
        tune.report({"val_accuracy": 0.0, "float_val_accuracy": 0.0,
                     "feasible": False, "deployable": False, "phase": "deploy",
                     "violations": "; ".join(violations)[:400]})
        return

    counts = count_neurons_and_synapses(cfg)
    static = {
        "feasible": True,
        "neurons": counts["totals"]["neurons"],
        "connections": counts["totals"]["connections"],
        "params": counts["totals"]["params"],
        "fc_in_features": counts["plan"].fc_in_features,
    }

    # tier 2: train in float. These per-epoch reports are what ASHA prunes on --
    # a cheap, dense trajectory. They are NOT the objective.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def report_fn(epoch, train_loss, train_acc, val_acc, best_val_acc, lr, phase="float"):
        # Per-epoch progress across BOTH warmup ("float") and on-grid ("grid")
        # phases. ASHA prunes on float_val_accuracy, which we report every epoch
        # of both phases so the trajectory is continuous and a trial isn't judged
        # only on its pre-grid accuracy.
        tune.report({"val_accuracy": val_acc, "float_val_accuracy": val_acc,
                     "best_val_accuracy": best_val_acc, "phase": phase,
                     "train_accuracy": train_acc, "train_loss": train_loss,
                     "lr": lr, "epoch": epoch, **static})

    res = run_training(cfg, data_dir=config["data_dir"], device=device, report_fn=report_fn)

    # tier 3: the objective -- the accuracy of the artifact that actually
    # deploys (inline QAT: trained on the grid; tail/ptq: fold+quant at the end).
    # This is the LAST report, so scope="last" selection and Optuna's
    # on_trial_complete see the hardware number, not the float trajectory.
    # An undeployable config scores 0: a fold that drives a threshold <= 0 fires
    # unconditionally on chip regardless of how good its float accuracy looked.
    hw_acc = res["hw_val_accuracy"] if res["deployable"] else 0.0
    tune.report({
        "val_accuracy": hw_acc,                 # <- selection metric
        "hw_val_accuracy": res["hw_val_accuracy"],
        "float_val_accuracy": res["float_val_accuracy"],
        "quant_gap": res["quant_gap"],
        "grid_val_accuracy": res.get("grid_val_accuracy"),
        "weight_clip_frac": res["weight_clip_frac"],
        "max_abs_weight": res["max_abs_weight"],
        "min_threshold": res["min_threshold"],
        "deployable": res["deployable"],
        "deploy_reasons": res["deploy_reasons"],
        "deploy_warnings": res["deploy_warnings"],
        "flush_steps": res["flush_steps"],
        "phase": "deploy", "epoch": cfg["train"].epochs,
        **static})


# columns pulled out of each trial's flat config for the live leaderboard. Kept
# in sync with the (expanded) search space: norm/tdbn/trainable-neuron/optimizer
# dims are new since the streaming logging was first written.
_LEADERBOARD_COLS = [
    # val_accuracy IS hw_val_accuracy for completed trials (the last report);
    # float_val_accuracy and quant_gap are kept beside it so a config that only
    # looks good before conversion is obvious at a glance.
    "trial_id", "status", "val_accuracy", "hw_val_accuracy", "float_val_accuracy",
    "quant_gap", "deployable", "weight_clip_frac", "min_threshold",
    "feasible", "epochs_run", "stopped_early",
    "neurons", "connections", "params",
    "depth", "channels", "kernel_size", "stride", "downsample_mode",
    "resize_to", "T", "tau", "trainable_tau", "trainable_threshold",
    "fc_layers", "final_reduction", "dropout_rate", "norm", "tdbn_alpha",
    "optimizer", "lr", "weight_decay", "scheduler", "label_smoothing", "grad_clip",
]


def _make_streaming_callback(writer: "ResultsWriter", target: float):
    """
    Ray Tune Callback that runs IN THE DRIVER PROCESS -- the only safe place to
    write shared files, since trials are separate processes. Sees every
    tune.report() as it happens, so the search is inspectable mid-flight instead
    of only after tuner.fit() returns.
    """
    from ray import tune

    def _clean(cfg: dict) -> dict:
        return {k: v for k, v in cfg.items() if k not in ("data_dir", "results_dir")}

    class StreamingResults(tune.Callback):
        def __init__(self):
            self.best = -1.0
            self.best_trial = None
            self.n_done = 0
            self.n_infeasible = 0
            self.n_pruned = 0

        def on_trial_result(self, iteration, trials, trial, result, **info):
            writer.append_jsonl("trial_progress.jsonl", {
                "trial_id": trial.trial_id, "epoch": result.get("epoch"),
                "val_accuracy": result.get("val_accuracy"),
                "train_accuracy": result.get("train_accuracy"),
                "train_loss": result.get("train_loss"),
                "lr": result.get("lr"), "feasible": result.get("feasible"),
            })
            if not result.get("feasible", True):
                self.n_infeasible += 1
                writer.append_jsonl("infeasible.jsonl", {
                    "trial_id": trial.trial_id, "violations": result.get("violations"),
                    "config": _clean(trial.config)})
                return
            # Only the deploy-phase report counts as "best". Float epochs are
            # progress, not results -- ranking on them is what let a config win
            # the search and then lose accuracy on chip.
            if result.get("phase") != "deploy":
                return
            if not result.get("deployable", True):
                writer.append_jsonl("undeployable.jsonl", {
                    "trial_id": trial.trial_id, "reasons": result.get("deploy_reasons"),
                    "float_val_accuracy": result.get("float_val_accuracy"),
                    "config": _clean(trial.config)})
                return
            acc = result.get("hw_val_accuracy") or 0.0
            if acc > self.best:
                self.best, self.best_trial = acc, trial.trial_id
                writer.write_json("best.json", {
                    "trial_id": trial.trial_id, "val_accuracy": acc,
                    "hw_val_accuracy": result.get("hw_val_accuracy"),
                    "float_val_accuracy": result.get("float_val_accuracy"),
                    "quant_gap": result.get("quant_gap"),
                    "weight_clip_frac": result.get("weight_clip_frac"),
                    "min_threshold": result.get("min_threshold"),
                    "epoch": result.get("epoch"),
                    "neurons": result.get("neurons"), "connections": result.get("connections"),
                    "params": result.get("params"), "flat_config": _clean(trial.config),
                    "config": cfg_to_dict_safe(trial.config)})
                try:
                    writer.write_text("best_summary.txt", format_summary(config_to_specs(trial.config)))
                except Exception as exc:                 # never let logging kill a search
                    writer.write_text("best_summary.txt", f"(summary failed: {exc})")
                writer.log(f"[stream] NEW BEST (on hardware) {acc:.4f} "
                           f"(float {result.get('float_val_accuracy') or 0.0:.4f}, "
                           f"gap {result.get('quant_gap') or 0.0:+.4f}, trial {trial.trial_id})"
                           + ("  <-- TARGET REACHED" if acc >= target else ""))

        def on_trial_complete(self, iteration, trials, trial, **info):
            self.n_done += 1
            r = trial.last_result or {}
            flat = _clean(trial.config)
            epochs_run = (r.get("epoch") if r.get("epoch") is not None else -1) + 1
            requested = trial.config.get("epochs")
            stopped_early = bool(r.get("feasible")) and requested is not None and epochs_run < requested
            if stopped_early:
                self.n_pruned += 1
            writer.append_jsonl("trials.jsonl", {
                "trial_id": trial.trial_id, "status": "complete",
                "feasible": r.get("feasible"), "val_accuracy": r.get("val_accuracy"),
                "hw_val_accuracy": r.get("hw_val_accuracy"),
                "float_val_accuracy": r.get("float_val_accuracy"),
                "quant_gap": r.get("quant_gap"), "deployable": r.get("deployable"),
                "deploy_reasons": r.get("deploy_reasons"),
                "weight_clip_frac": r.get("weight_clip_frac"),
                "best_val_accuracy": r.get("best_val_accuracy"),
                "epochs_run": epochs_run, "epochs_requested": requested,
                "stopped_early": stopped_early, "neurons": r.get("neurons"),
                "connections": r.get("connections"), "params": r.get("params"),
                "violations": r.get("violations"), "config": flat})
            # A trial pruned by ASHA never reaches the deploy phase, so it has
            # no hardware number. Record that as "pruned" rather than 0.0, which
            # would be indistinguishable from a config that deployed and failed.
            deployed = r.get("phase") == "deploy"
            writer.append_csv("leaderboard.csv", {
                "trial_id": trial.trial_id,
                "status": "deployed" if deployed else "pruned_before_deploy",
                # the DEPLOYED number, never the best float epoch
                "val_accuracy": (r.get("hw_val_accuracy") if deployed else None),
                "hw_val_accuracy": r.get("hw_val_accuracy"),
                "float_val_accuracy": r.get("float_val_accuracy"),
                "quant_gap": r.get("quant_gap"), "deployable": r.get("deployable"),
                "weight_clip_frac": r.get("weight_clip_frac"),
                "min_threshold": r.get("min_threshold"),
                "feasible": r.get("feasible"), "epochs_run": epochs_run,
                "stopped_early": stopped_early, "neurons": r.get("neurons"),
                "connections": r.get("connections"), "params": r.get("params"), **flat,
            }, header_order=_LEADERBOARD_COLS)
            writer.write_json("progress.json", {
                "trials_completed": self.n_done, "infeasible_reports": self.n_infeasible,
                "stopped_early_by_scheduler": self.n_pruned, "best_val_accuracy": self.best,
                "best_trial": self.best_trial, "target": target,
                "target_reached": self.best >= target})

        def on_trial_error(self, iteration, trials, trial, **info):
            writer.append_jsonl("trials.jsonl", {
                "trial_id": trial.trial_id, "status": "error", "config": _clean(trial.config)})

    return StreamingResults()


def cfg_to_dict_safe(flat_config: dict) -> dict:
    """Best-effort structured dump of a flat trial config for best.json."""
    try:
        return {name: asdict(spec) for name, spec in config_to_specs(flat_config).items()}
    except Exception as exc:
        return {"error": str(exc)}


def run_search(args):
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    data_dir_abs = validate_data_dir(args.data_dir)   # before Ray, before any dir is made
    writer = ResultsWriter(default_results_dir("search", args.results_dir))
    T_CHOICES = [8, 16]

    writer.write_json("search_config.json", {
        "started": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args), "data_dir": data_dir_abs, "results_dir": writer.root,
        "T_choices": T_CHOICES, "axon_limits": AXON_LIMITS, "neuron_limits": NEURON_LIMITS})
    writer.log(f"[stream] results -> {writer.root}")
    writer.log("[stream] watch with:  Get-Content "
               + writer.path("trial_progress.jsonl") + " -Wait -Tail 20")

    warmup_dataset_cache(data_dir_abs, T_CHOICES)

    if args.compute == "cluster":
        ray.init(address="auto")          # connect to an existing multi-node Ray cluster
    else:
        # Single machine. The kwargs matter in containers (Kaggle, Colab, Docker):
        #   * include_dashboard=False   -- the dashboard's extra deps are usually
        #     absent and its absence otherwise prints a scary traceback;
        #   * _temp_dir                 -- Ray defaults under /tmp, which on a
        #     hosted notebook is small and not the persisted volume. Set RAY_TMPDIR
        #     to somewhere with room (on Kaggle: /kaggle/temp/ray).
        # A small /dev/shm makes Ray refuse to size its object store; export
        # RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 to let it fall back to disk.
        ray.init(include_dashboard=False,
                 ignore_reinit_error=True,
                 _temp_dir=os.environ.get("RAY_TMPDIR") or None)

    class TargetAccuracyStopper(tune.Stopper):
        """`run_config.stop={...}` only stops the TRIAL that reported the value.
        Stopping the whole study once the target is hit needs stop_all()."""
        def __init__(self, target: float):
            self.target = target
            self._reached = False

        def __call__(self, trial_id, result):
            # Only a DEPLOY-phase report can end the study. A float epoch that
            # crosses the target says nothing about what runs on chip.
            if (result.get("phase") == "deploy"
                    and result.get("deployable", False)
                    and result.get("hw_val_accuracy", 0.0) >= self.target):
                self._reached = True
            return self._reached

        def stop_all(self) -> bool:
            return self._reached

    def _short_trial_dirname(trial):
        # Ray's default dirname embeds every hyperparameter and blows past
        # Windows' 260-char path limit.
        return f"trial_{trial.trial_id}"

    space = make_define_by_run(args.batch_size, args.epochs, data_dir_abs, T_CHOICES,
                               per_layer=(args.search_space == "per_layer"))

    # Ray checkpoints the SEARCHER periodically, and OptunaSearch.save() pickles
    # the space. If it is not picklable the study dies at the first checkpoint --
    # minutes or hours in, after real GPU time is spent. Check it up front.
    import pickle as _pickle
    try:
        _pickle.loads(_pickle.dumps(space))
    except Exception as exc:
        raise RuntimeError(
            f"Search space is not picklable ({exc}). Ray checkpoints the searcher, so the "
            "space must be a module-level object -- not a closure or lambda."
        ) from exc

    # metric/mode wiring -- verified against ray 2.56, do not "simplify":
    #   * OptunaSearch MUST get them: its set_search_properties() returns early
    #     when a `space` is already set, so TuneConfig's metric/mode never reach
    #     it and the run dies at the first suggest() with "metric (None)".
    #   * TuneConfig ALSO needs them, for the scheduler and get_best_result().
    #   * ASHAScheduler must NOT get them -- Ray raises ValueError if a scheduler
    #     was instantiated with metric/mode AND they are passed to TuneConfig.
    # seed the search with the known winners, but ONLY for the uniform space --
    # the seed dicts use uniform keys (channels/kernel_size), not per-layer ones.
    seeds = SEED_CONFIGS if args.search_space == "uniform" else None
    if seeds:
        writer.log(f"[stream] seeding {len(seeds)} known-good configs before TPE explores")
    algo = OptunaSearch(space=space, metric="val_accuracy", mode="max",
                        points_to_evaluate=seeds)

    scheduler = None
    if args.scheduler == "asha":
        # every trial gets grace_period epochs; after that, at rungs
        # grace*rf^k, a trial is killed unless it is in the top 1/rf reaching
        # that rung. grace_period MUST stay > 1: cosine/onecycle LR do their work
        # late (early judging favours flat-LR configs), and infeasible trials
        # report once at epoch 0 -- with grace>1 they never reach a rung and so
        # cannot drag the promotion cutoff down.
        # ASHA prunes on float_val_accuracy -- the dense per-epoch trajectory.
        # It must NOT prune on val_accuracy: that key holds the float number
        # during training and the hardware number in the final report, so a
        # scheduler watching it would compare two different quantities across
        # rungs. Optuna still SELECTS on val_accuracy (last report = hardware).
        # max_t = epochs + 1, NOT epochs. ASHA returns STOP as soon as a trial
        # reports training_iteration >= max_t, and every tune.report increments
        # it -- so with max_t=epochs the trial is killed on its final epoch
        # report and the deploy-phase report (fold + INT16 + QAT) never lands.
        # The symptom is silent: hw_val_accuracy is NaN for every trial and
        # selection quietly falls back to float accuracy, i.e. exactly the
        # behaviour this whole rework exists to remove. The +1 buys the trial
        # one extra reporting slot for its deployment result.
        scheduler = ASHAScheduler(
            time_attr="training_iteration", max_t=args.epochs + 1,
            metric="float_val_accuracy", mode="max",
            grace_period=min(args.grace_period, args.epochs),
            reduction_factor=args.reduction_factor, brackets=args.brackets)
        rungs, r = [], min(args.grace_period, args.epochs)
        while r < args.epochs:
            rungs.append(r); r *= args.reduction_factor
        writer.log(f"[stream] ASHA on: grace={args.grace_period} rf={args.reduction_factor} "
                   f"max_t={args.epochs}; cull points at epochs {rungs or '(none)'}")
    else:
        writer.log("[stream] no scheduler -- every feasible trial runs the full epoch budget")

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0}),
        # NO metric/mode here. Ray raises ValueError if TuneConfig sets them
        # while the searcher or scheduler already has its own -- and here BOTH
        # do, deliberately and with DIFFERENT values:
        #   OptunaSearch  -> val_accuracy       (last report = post-INT16, selection)
        #   ASHAScheduler -> float_val_accuracy (per-epoch trajectory, pruning)
        # That split is the whole point, so the shared TuneConfig value is the
        # thing that has to go. Selection metric is named explicitly below in
        # get_best_result(...).
        tune_config=tune.TuneConfig(
            search_alg=algo, scheduler=scheduler,
            num_samples=args.trials,
            trial_dirname_creator=_short_trial_dirname,
        ),
        run_config=tune.RunConfig(
            stop=TargetAccuracyStopper(args.target),
            callbacks=[_make_streaming_callback(writer, args.target)],
        ),
    )
    results = tuner.fit()
    export_trial_records(results, out_dir=writer.root)   # final flat table, alongside the stream
    # scope="last": the final report of each trial is the deploy-phase one, so
    # this ranks by post-fold post-INT16 accuracy. The default scope would pick
    # each trial's best FLOAT epoch and hand back a config that never had that
    # accuracy on hardware.
    best = results.get_best_result(metric="val_accuracy", mode="max", scope="last")
    summary_text = format_summary(config_to_specs(best.config))
    writer.write_text("best_summary.txt", summary_text)
    writer.write_json("final.json", {
        "finished": datetime.now().isoformat(timespec="seconds"),
        "best_val_accuracy": best.metrics["val_accuracy"],
        "best_hw_val_accuracy": best.metrics.get("hw_val_accuracy"),
        "best_float_val_accuracy": best.metrics.get("float_val_accuracy"),
        "best_quant_gap": best.metrics.get("quant_gap"),
        "best_weight_clip_frac": best.metrics.get("weight_clip_frac"),
        "best_config": {k: v for k, v in best.config.items() if k not in ("data_dir", "results_dir")},
        "results_dir": writer.root})
    print("Best HARDWARE val_accuracy:", best.metrics["val_accuracy"])
    print("  float val_accuracy      :", best.metrics.get("float_val_accuracy"))
    print("  quantization gap        :", best.metrics.get("quant_gap"))
    print("  folded weights clipped  :", best.metrics.get("weight_clip_frac"))
    print("Best config:", best.config)
    print()
    print(summary_text)
    writer.log(f"[stream] all results in {writer.root}")


# =============================================================================
# 11. CLI entry point
# =============================================================================

def run_fold(args):
    """Load a trained checkpoint, fold BN, and report the folded-vs-BN comparison
    on the test set. Answers Q1's 'compare test accuracy of folded to BN'."""
    _require_torch()
    cfg = parse_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    if ckpt.get("encoder_layers_json"):
        cfg["encoder"].layers_json = ckpt["encoder_layers_json"]
    net = build_model(cfg).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    print("Folded-bias distribution (|b'|/theta per conv):")
    for row in fold_bias_report(net):
        print(f"  conv{row['conv_index']}: mean {row['mean_abs_b_over_theta']:.3f}  "
              f"max {row['max_abs_b_over_theta']:.3f}")
    print("  (mean ~0.1 folds cleanly to a threshold; ~0.5 needs bias on the conv)\n")

    _, _, test_loader = build_dataloaders(cfg, data_dir=args.data_dir)
    folded = fold_bn(net, bias_mode=args.bias_mode)
    rep = verify_fold(net, folded, test_loader, device)
    print(f"fold bias_mode={args.bias_mode}")
    print(f"  prediction agreement : {rep['pred_agreement']*100:.3f}%   (conv-bias fold -> 100%)")
    print(f"  max |logit| diff     : {rep['max_abs_logit_diff']:.2e}   (exact fold -> ~1e-5)")
    print(f"  test acc  BN         : {rep['acc_bn']*100:.3f}%")
    print(f"  test acc  folded     : {rep['acc_folded']*100:.3f}%   (delta {rep['acc_delta']*100:+.3f})")

    # quantize to the chip's grid and report what conversion will actually cost
    enable_weight_fake_quant(folded)
    acc_q = evaluate(folded, test_loader, device)
    bake_weight_fake_quant(folded)
    rep = deployment_report(folded)
    print(f"  test acc  folded+INT16: {acc_q*100:.3f}%")
    print(f"  folded weights clipped: {rep['weight_clip_frac']*100:.2f}%  "
          f"(max |w| = {rep['max_abs_weight']:.3f}, representable limit 1.0)")
    print(f"  min folded threshold  : {rep['min_threshold']}")
    for w in rep["warnings"]:
        print("  WARNING: " + w)
    if not rep["deployable"]:
        print("  NOT DEPLOYABLE: " + "; ".join(rep["blocking_reasons"]))

    # save the BN-free weights -- this is the deployable artifact
    out = args.out or (os.path.splitext(args.ckpt)[0] + f"_folded_{args.bias_mode}.pth")
    folded.eval()
    torch.save({
        "state_dict": folded.state_dict(),          # BN-free: conv weights carry the fused bias
        "encoder_layers_json": cfg["encoder"].layers_json,
        "config": {n: asdict(s) for n, s in cfg.items()},
        "bias_mode": args.bias_mode,
        "folded": True,
        "test_acc_bn": rep["acc_bn"], "test_acc_folded": rep["acc_folded"],
        "max_abs_logit_diff": rep["max_abs_logit_diff"],
    }, out)
    print(f"\nfolded weights saved -> {out}")
    # also drop a plain-numpy dump of every conv (weight + fused bias), which is
    # what a hardware/converter pipeline usually wants -- no torch needed to read
    npz = os.path.splitext(out)[0] + "_conv_params.npz"
    arrays = {}
    for idx, m in enumerate(folded.conv_fc):
        if isinstance(m, (layer.Conv2d, nn.Conv2d)):
            arrays[f"conv{idx}_weight"] = m.weight.detach().cpu().numpy()
            if m.bias is not None:
                arrays[f"conv{idx}_bias"] = m.bias.detach().cpu().numpy()
        elif isinstance(m, (layer.Linear, nn.Linear)):
            arrays[f"linear{idx}_weight"] = m.weight.detach().cpu().numpy()
            if m.bias is not None:
                arrays[f"linear{idx}_bias"] = m.bias.detach().cpu().numpy()
    np.savez(npz, **arrays)
    print(f"per-layer conv/linear arrays -> {npz}  ({len(arrays)} arrays)")

    # The per-layer neuron table. This is what removes the two hardcoded lines
    # ("threshold = 32767", "leak_lif = 63") from the conversion script: build
    # one LIF_neuron per entry and attach it to that layer's neuron keys instead
    # of reusing a single global N.
    hw = hardware_export(folded)
    hw_path = os.path.splitext(out)[0] + "_hardware.json"
    with open(hw_path, "w", encoding="utf-8") as fh:
        json.dump(hw, fh, indent=2)
    print(f"per-layer neuron table -> {hw_path}")
    for L in hw["neuron_layers"]:
        thr = L["threshold_int"]
        thr_s = (f"per-channel[{len(thr)}] {min(thr)}..{max(thr)}"
                 if L["per_channel_threshold"] else str(thr))
        print(f"  layer {L['layer_index']}: LIF_neuron(threshold={thr_s}, shift=0, leak={L['leak']})")
    print(f"  converter must run {hw['flush_steps']} zero-input steps after the last frame")


def main():
    # SequentialLR.step() calls the child scheduler's deprecated step(0) at the
    # warmup->main handoff (torch's own code, not ours). It fires once per trial
    # that drew warmup_epochs>0 and would otherwise spam the log. The closed form
    # it falls back to is correct for CosineAnnealingLR, so this is pure noise.
    warnings.filterwarnings("ignore", message=r".*epoch parameter in `scheduler\.step\(\)`.*")

    parser = argparse.ArgumentParser(description="DVS Gesture SNN -- summary, single run, search, or fold")
    sub = parser.add_subparsers(dest="mode", required=True)

    summary_p = sub.add_parser("summary", parents=[build_config_parser()],
                               help="print architecture, feasibility, and neuron/synapse counts (no torch needed)")

    single_p = sub.add_parser("single", parents=[build_config_parser()],
                              help="train one config directly -- laptop or single GPU")
    single_p.add_argument("--data-dir", required=True)
    single_p.add_argument("--epochs", type=int, default=None, help="overrides --train epochs=...")
    single_p.add_argument("--ckpt", default=None, help="save best-val weights here (needed for later folding)")
    single_p.add_argument("--results-dir", default=None,
                          help="stream per-epoch history/best/final here (default: results/single_<timestamp>)")

    fold_p = sub.add_parser("fold", parents=[build_config_parser()],
                            help="fold BN into convs and compare folded vs BN test accuracy")
    fold_p.add_argument("--data-dir", required=True)
    fold_p.add_argument("--ckpt", required=True, help="checkpoint written by `single --ckpt`")
    fold_p.add_argument("--bias-mode", choices=["conv", "threshold"], default="conv",
                        help="conv = exact (needs conv bias on chip); threshold = per-channel, approximate")
    fold_p.add_argument("--out", default=None,
                        help="where to save the folded weights (default: <ckpt>_folded_<mode>.pth)")

    search_p = sub.add_parser("search", help="Optuna search orchestrated by Ray Tune")
    search_p.add_argument("--compute", choices=["local", "cluster"], default="local",
                          help="local = this machine; cluster = attach to an existing Ray cluster")
    search_p.add_argument("--trials", type=int, default=50)
    search_p.add_argument("--epochs", type=int, default=40)
    search_p.add_argument("--batch-size", type=int, default=16)
    search_p.add_argument("--target", type=float, default=0.975,
                          help="stop the whole study once a trial reaches this val_accuracy")
    search_p.add_argument("--search-space", choices=["uniform", "per_layer"], default="uniform",
                          help="uniform (default) = the proven space, seeded from the winners; "
                               "per_layer varies kernel/stride/channels/pool independently per conv layer")
    search_p.add_argument("--results-dir", default=None,
                          help="stream trial_progress/best/leaderboard/final here (default: results/search_<timestamp>)")
    search_p.add_argument("--scheduler", choices=["asha", "none"], default="asha",
                          help="asha = kill trials clearly behind at each rung (default, ~2.6x faster); "
                               "none = every feasible trial runs all epochs")
    search_p.add_argument("--grace-period", type=int, default=8,
                          help="epochs every trial gets before ASHA may kill it. Keep >1")
    search_p.add_argument("--reduction-factor", type=int, default=3,
                          help="at each rung, keep the top 1/N of trials that reached it")
    search_p.add_argument("--brackets", type=int, default=1)
    search_p.add_argument("--data-dir", required=True)

    args = parser.parse_args()

    if args.mode == "summary":
        print(format_summary(parse_config()))
        return

    if args.mode == "single":
        cfg = parse_config()
        if args.epochs is not None:
            cfg["train"].epochs = args.epochs
        data_dir_abs = validate_data_dir(args.data_dir)   # before results dir is made
        summary_text = format_summary(cfg)
        print(summary_text)

        writer = ResultsWriter(default_results_dir("single", args.results_dir))
        writer.write_text("summary.txt", summary_text)
        writer.write_json("config.json", {n: asdict(s) for n, s in cfg.items()})
        writer.log(f"[stream] results -> {writer.root}")
        writer.log("[stream] watch with:  Get-Content " + writer.path("history.jsonl") + " -Wait -Tail 20")

        feasible, violations = check_feasibility(
            cfg["input"], cfg["encoder"], cfg["downsample"], cfg["head"], cfg["output"])
        writer.write_json("feasibility.json", {"feasible": feasible, "violations": violations})
        if not feasible:
            print("WARNING -- this config violates HiAER-Spike limits (training anyway):")
            for v in violations:
                print("  -", v)

        best_seen = {"acc": 0.0, "epoch": -1}

        def report_fn(epoch, train_loss, train_acc, val_acc, best_val_acc, lr, phase="float"):
            print(f"[{phase}] epoch {epoch}: lr={lr:.2e} train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f}")
            rec = writer.append_jsonl("history.jsonl", {
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_acc": val_acc, "best_val_acc": best_val_acc, "lr": lr})
            if val_acc > best_seen["acc"]:
                best_seen.update(acc=val_acc, epoch=epoch)
                writer.write_json("best.json", {"epoch": epoch, "val_accuracy": val_acc, "record": rec})

        res = {}
        try:
            res = run_training(cfg, data_dir=data_dir_abs, report_fn=report_fn, ckpt_path=args.ckpt)
        finally:
            # written from finally so it survives Ctrl-C / a crash mid-training
            writer.write_json("final.json", {
                "best_float_val_accuracy": best_seen["acc"], "best_epoch": best_seen["epoch"],
                "requested_epochs": cfg["train"].epochs,
                "hardware": {k: v for k, v in res.items() if k != "hw_net"},
                "config": {n: asdict(s) for n, s in cfg.items()}})
        if res:
            writer.write_json("hardware_export.json", hardware_export(res["hw_net"]))
            print()
            print("=" * 62)
            print("DEPLOYMENT (this is the number that matters)")
            print("=" * 62)
            print(f"  float/warmup val acc  : {res['float_val_accuracy']:.4f}")
            if "grid_val_accuracy" in res and res.get("grid_val_accuracy") is not None:
                # inline (Option A): trained on the grid, no separate fold/ptq stage
                print(f"  on-grid val_accuracy  : {res['grid_val_accuracy']:.4f}  (trained folded+INT16)")
            if "folded_val_accuracy" in res:            # tail/ptq only
                print(f"  folded (BN removed)   : {res['folded_val_accuracy']:.4f}")
                print(f"  + INT16, no QAT       : {res['ptq_val_accuracy']:.4f}")
            print(f"  DEPLOYED val_accuracy : {res['hw_val_accuracy']:.4f}   <-- runs on chip")
            print(f"  gap vs float          : {res['quant_gap']:+.4f}")
            print(f"  folded weights clipped: {res['weight_clip_frac']*100:.2f}%  "
                  f"(max |w| = {res['max_abs_weight']:.3f}, limit 1.0)")
            print(f"  min folded threshold  : {res['min_threshold']}")
            print(f"  pipeline flush steps  : {res['flush_steps']}")
            if not res["deployable"]:
                print(f"  NOT DEPLOYABLE        : {res['deploy_reasons']}")
        best = res.get("hw_val_accuracy", 0.0)
        print(f"\nbest hardware val_accuracy: {best:.4f}")
        writer.log(f"[stream] wrote {writer.path('final.json')}")
        if args.ckpt:
            print(f"best weights saved to {args.ckpt} -- fold with:  "
                  f"python Practice.py fold --ckpt {args.ckpt} --data-dir {args.data_dir} [same --encoder ...]")

    elif args.mode == "fold":
        run_fold(args)

    elif args.mode == "search":
        run_search(args)


if __name__ == "__main__":
    main()