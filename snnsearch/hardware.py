"""The chip's limits, and the check that a configuration fits them.

    AXON_LIMITS, NEURON_LIMITS   fan-in, fan-out and axon count, from hs_api
                                 and Christopher's L6m testing
    THRESHOLD_LIMITS             bits in a neuron's threshold field
    HW_TAU_CHOICES               the taus the chip can run (powers of two)
    check_feasibility            the limits checked layer by layer on the plan

Pure arithmetic on the plan, so it runs before any GPU time is spent and
needs no torch.
"""

import math

from .config import InputSpec, EncoderSpec, OutputSpec, DownsampleSpec, HeadSpec
from .planning import plan_network, InfeasibleConfig


AXON_LIMITS = {
    # MEASURED, not the register width. 16,383 is what the field can hold and
    # was what this used. Christopher's testing of bitstream L6m errored at
    # 16,383 and passed at 16,000, so the usable ceiling is lower than the
    # representable one and 16,000 is the number that has been run.
    #
    # For two-channel DVS input that caps a square frame at 89 x 89:
    # 89*89*2 = 15,842 fits, 90*90*2 = 16,200 does not.
    "total_axons": 16_000,
    "fan_out": 4_096,   # axonal fan-out: how many neurons one input axon feeds
    "fan_in": 8_191,    # axonal fan-in: how many axons feed one neuron
}
NEURON_LIMITS = {
    "fan_out": 4_095,   # neuron-to-neuron fan-out
    "fan_in": 8_159,    # neuron-to-neuron fan-in
}

THRESHOLD_LIMITS = {
    # 36 bits, the same as a neuron's state word: the FPGA keeps 2 neurons
    # per 72-bit URAM row, and its THRESHOLD register (CONFIG_WRITE, opcode
    # 0x06) is 36 bits ("3 Verilog Files Review", hiaerspike-web docs).
    # simpleSim's membrane format fxp-s35 is the same width. It matters behind
    # an average pool: deployment folds the pool into the next layer by
    # counting that layer's membrane in 4x finer units, so its threshold
    # becomes 4 * theta + 3, up to 131,071 (18 bits). Search 05's ranks 2 to
    # 4 need that, which a 16-bit field could not hold. Not used to reject a
    # network: deployment_report warns, and every run records threshold_bits.
    "bits": 36,
}

# A config's `limits:` block names these. Set_limits applies it.
_LIMIT_KEYS = {
    "total_axons": (AXON_LIMITS, "total_axons"),
    "axon_fan_out": (AXON_LIMITS, "fan_out"),
    "axon_fan_in": (AXON_LIMITS, "fan_in"),
    "neuron_fan_out": (NEURON_LIMITS, "fan_out"),
    "neuron_fan_in": (NEURON_LIMITS, "fan_in"),
    "threshold_bits": (THRESHOLD_LIMITS, "bits"),
}
_DEFAULT_LIMITS = {k: d[f] for k, (d, f) in _LIMIT_KEYS.items()}


def set_limits(overrides=None):
    """Apply a config's `limits:` block, for a chip build with other limits.

    Starts from the defaults every time, so a second config never inherits a
    first one's overrides, and changes the dicts in place, so every module
    that imported them sees the new values. Called when a config is loaded
    and at the start of every trial (Ray runs trials in fresh processes).
    """
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(_LIMIT_KEYS))
    if unknown:
        raise SystemExit(f"limits: unknown key(s) {unknown}; "
                         f"known: {', '.join(_LIMIT_KEYS)}")
    for key, (table, field) in _LIMIT_KEYS.items():
        table[field] = int(overrides.get(key, _DEFAULT_LIMITS[key]))


def check_feasibility(input_cfg: InputSpec, encoder_cfg: EncoderSpec,
                      downsample_cfg: DownsampleSpec,
                      head_cfg: HeadSpec = None, output_cfg: OutputSpec = None,
                      bias_neurons: bool = False, signed: bool = False):
    """
    Check the chip's connection limits at every layer, before any training.

    Returns (is_feasible: bool, violations: list[str]).

    FAN-IN COUNTS THE NEURONS THAT ACTUALLY CONNECT. An average pool has no
    neurons of its own on chip: it folds into the next layer's synapses
    (hs_export.py), so every neuron in front of the pool connects directly to
    that layer. A 2x2 pool therefore multiplies the next layer's fan-in by 4,
    and a flatten or GAP head connects to every neuron of the last conv layer,
    before its pool. Searches 01 and 02 counted the pooled values instead,
    which let through rank1, whose classifier needs 32,256 inputs per neuron
    against a limit of 8,159. simpleSim has no such limit, so it still ran.

    bias_neurons adds the bias neurons' share: behind a pool the bias is split
    across up to `scale` neurons (hs_export.split_bias), one fan-in slot each.

    MULTI-BIT INPUT (InputSpec.input_bits = B) takes B axons per input value,
    so it multiplies the axon count and the first layer's axonal fan-in by B.

    SIGNED SPIKES put two chip neurons behind every unit, one for +1 and one
    for -1, each connected to both neurons of the next unit. Fan-in and
    fan-out double wherever a signed layer sends or receives.

    MAX pooling is different. It runs as one OR neuron per window (OrPool in
    neuron.py), so the next layer reads the pooled map and its fan-in is not
    multiplied. Each OR neuron reads k*k inputs, far below any limit.
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

    def check(value, limit, label):
        if value > limit:
            violations.append(f"{label} {value} > {limit}")

    first_fc_width = plan.linears[0].out_features if plan.linears else 0
    folds = downsample_cfg.pool_type != "max"      # avg pools fold. Max pools are OR neurons
    bits = max(1, int(getattr(input_cfg, "input_bits", 1) or 1))
    sign = 2 if signed else 1          # chip neurons per unit
    prev_pool = 1                      # pool kernel between the previous neurons and this layer
    for b in plan.blocks:
        k, s = b.kernel_size, b.stride
        scale = prev_pool * prev_pool
        bias_in = scale if bias_neurons else 0
        is_last = b.index == len(plan.blocks) - 1
        # fan-out of whatever feeds this block: one input pixel (or one neuron
        # in front of a pool) reaches ceil(k/s)^2 positions in every channel,
        # and both neurons of each unit when the layer is signed
        fan_out = math.ceil(k / s) ** 2 * b.out_channels * sign
        tag = f"block{b.index}"
        if b.index == 0:
            check(b.in_hw[0] * b.in_hw[1] * b.in_channels * bits, AXON_LIMITS["total_axons"],
                  f"{tag}: total_axons")
            check(fan_out, AXON_LIMITS["fan_out"], f"{tag}: axonal_fan_out")
            check(k * k * b.in_channels * bits, AXON_LIMITS["fan_in"], f"{tag}: axonal_fan_in")
            check(bias_in, NEURON_LIMITS["fan_in"], f"{tag}: neuron_fan_in (bias)")
        else:
            fan_in = k * k * b.in_channels * scale * sign
            check(fan_out, NEURON_LIMITS["fan_out"], f"block{b.index - 1}: neuron_fan_out")
            check(fan_in + bias_in, NEURON_LIMITS["fan_in"], f"{tag}: neuron_fan_in")
        if is_last:
            # the last block's neurons feed the head
            check(first_fc_width * sign, NEURON_LIMITS["fan_out"], f"{tag}: neuron_fan_out(->fc)")
        prev_pool = b.pool_kernel if (b.pool and folds) else 1

    for lin in plan.linears:
        tag = "classifier" if lin.is_classifier else f"fc{lin.index}"
        if lin.index == 0 and plan.blocks:
            last = plan.blocks[-1]
            # flatten and GAP both connect to every neuron of the last conv
            if last.pool and not folds:
                # OR neurons already pooled the map. The head reads the pooled size
                fan_in = last.out_channels * last.out_hw[0] * last.out_hw[1]
                pool = 1
            else:
                fan_in = last.out_channels * last.conv_out_hw[0] * last.conv_out_hw[1]
                pool = (last.conv_out_hw[0] // last.out_hw[0]) if last.pool else 1
            if plan.reduction == "gap":
                pool *= last.out_hw[0]
            scale = pool * pool
        else:
            fan_in, scale = lin.in_features, 1
        check(fan_in * sign + (scale if bias_neurons else 0), NEURON_LIMITS["fan_in"],
              f"{tag}: neuron_fan_in")
        # the LAST linear's neurons are the outputs. They feed nothing on chip
        if not lin.is_classifier:
            check(lin.out_features * sign, NEURON_LIMITS["fan_out"], f"{tag}: neuron_fan_out")

    return len(violations) == 0, violations


def check_spec_feasibility(spec: dict):
    """check_feasibility on a full spec dict (the one config_to_specs returns),
    reading the bias mode from its TrainSpec when it has one."""
    train = spec.get("train")
    bias_mode = getattr(train, "fold_bias_mode", "threshold") if train is not None else "threshold"
    signed = bool(getattr(spec.get("neuron"), "signed_spikes", False))
    return check_feasibility(spec["input"], spec["encoder"], spec["downsample"],
                             spec["head"], spec["output"],
                             bias_neurons=bias_mode in ("neuron", "conv"), signed=signed)


W_BITS = 16
W_ALPHA = 1.0
INT16_MAX = 2 ** (W_BITS - 1) - 1            # 32767
W_DELTA = W_ALPHA / INT16_MAX                # 1/32767

# The leak values worth searching. The chip's leak register is a SHIFT:
# simpleSim on the pinned L6m commit computes V -= V // 2**leak, so tau must be
# a power of two. This list used to hold 3, 6 and 63 on the assumption that
# leak == round(tau). Those taus do not exist on chip, and rank1 trained to
# tau 3. (The converter's leak=63 is a shift of 63, i.e. no leak at all.)
HW_TAU_CHOICES = [2, 4, 8, 16]
HW_TAU_MIN, HW_TAU_MAX = 2, 128
# trainable tau is learned as log2(tau) and rounded, so it can only land on
# 2**HW_LOG2_TAU_MIN .. 2**HW_LOG2_TAU_MAX
HW_LOG2_TAU_MIN, HW_LOG2_TAU_MAX = 1, 7


def signed_bits(value) -> int:
    """Bits of a two's-complement field that holds +value: 32767 needs 16,
    131,071 needs 18."""
    return int(abs(int(value))).bit_length() + 1
