"""Quantization-aware training, the deployment audit, and the export.

Moved verbatim from Practice2.py lines 1725-1921 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

import json
import math
import os
from copy import deepcopy

from .config import TrainSpec
from .hardware import W_ALPHA, W_DELTA, INT16_MAX, W_BITS
from .quantgrid import fake_quantize_weight, weight_clip_fraction
from .neuron import HardwareLIFNode, _WeightFakeQuant
from .folding import fold_bn, verify_fold, fold_bias_report
from ._torch import _HAS_TORCH, _require_torch, _no_grad, torch, nn, layer


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
    from .train import hardware_flush_steps      # local: see deploy_and_measure
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
    # local import: train.py imports deploy_and_measure from this module,
    # so these cannot be imported at module level.
    from .train import (evaluate, build_optimizer, build_scheduler,
                        train_one_epoch, hardware_flush_steps)
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
