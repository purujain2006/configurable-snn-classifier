"""Layer-shape planning: one source of truth for every derived size.

Pure arithmetic, no torch.

Moved verbatim from Practice2.py lines 367-509 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

from dataclasses import dataclass, field
from typing import Optional

from .config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                     HeadSpec, effective_hw, parse_fc_widths, resolve_conv_layers)


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
