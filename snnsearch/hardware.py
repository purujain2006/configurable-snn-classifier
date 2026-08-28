"""HiAER-Spike limits and the INT16 deployment grid.

The constants are read off hs_api rather than chosen. The feasibility
check is arithmetic on a plan, so it needs no torch.

Moved verbatim from Practice2.py lines 516-590, 770-782 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

import math

from .config import InputSpec, EncoderSpec, OutputSpec, DownsampleSpec, HeadSpec
from .planning import plan_network, InfeasibleConfig


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
