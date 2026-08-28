"""Configuration dataclasses and the key=value command line surface.

Imports no deep-learning library, so `summary` runs on a bare Python.

Moved verbatim from Practice2.py lines 97-356 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

import argparse
from dataclasses import asdict, dataclass, field, fields
from typing import Optional


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
    # threshold mode only. HOW the folded bias is applied during inline QAT.
    #   "threshold" -- compare against theta - b', which is what export_deployed
    #                  writes and therefore what the chip runs.
    #   "input"     -- add b' to the neuron's input. The older behaviour.
    #
    # These are not the same operation once tau > 1. A bias arrives every
    # timestep and accumulates against the leak, reaching b'*tau, while a
    # threshold shift is worth b' once. On results/probe.pth (tau=63) the two
    # forms disagree on 9-35% of channel-timesteps, so training in "input" form
    # and deploying in threshold form validates a network that never runs.
    # tools/fold_bias_equivalence.py measures the gap on any checkpoint.
    fold_bias_qat_form: str = "threshold"


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
