"""The network, built straight from the plan so it cannot drift from it.

Moved verbatim from Practice2.py lines 1297-1477 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

from .config import InputSpec, EncoderSpec, OutputSpec, DownsampleSpec, HeadSpec, NeuronSpec
from .planning import plan_network, NetPlan
from .hardware import W_ALPHA, W_DELTA, INT16_MAX
from .quantgrid import (_ste, fake_quantize_weight, fake_quantize_threshold,
                        quantized_threshold_int, fold_bias_band, constrain_fold_bias,
                        weight_clip_fraction)
from .neuron import (build_neuron, HardwareLIFNode, TdBatchNorm2d,
                     ConvBNFoldQuant, enable_weight_fake_quant)
from ._torch import (_HAS_TORCH, _HAS_HS_API, _require_torch, _no_grad,
                     torch, nn, F, layer, functional, Custom_LIFNode, Custom_IFNode)


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

    def to_qat_folded(self, bias_mode: str = "threshold", fold_bias_margin: float = 0.05,
                      fold_bias_qat_form: str = "threshold"):
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
        # Validate here rather than letting a typo fall through to the neuron,
        # where anything != "threshold" silently selects the legacy input form.
        if fold_bias_qat_form not in ("threshold", "input"):
            raise ValueError(
                f"fold_bias_qat_form must be 'threshold' or 'input', got "
                f"{fold_bias_qat_form!r}. 'threshold' matches what "
                f"export_deployed writes; 'input' is the older form and drifts "
                f"from the deployed network as tau rises.")
        mods = list(self.conv_fc)
        new_mods, i = [], 0
        while i < len(mods):
            m = mods[i]
            nxt = mods[i + 1] if i + 1 < len(mods) else None
            nxt2 = mods[i + 2] if i + 2 < len(mods) else None
            if isinstance(m, (layer.Conv2d, nn.Conv2d)) and isinstance(nxt, (layer.BatchNorm2d, nn.BatchNorm2d)):
                block = ConvBNFoldQuant(m, nxt, bias_mode=bias_mode, quantize=True,
                                        fold_bias_margin=fold_bias_margin,
                                        fold_bias_qat_form=fold_bias_qat_form)
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
