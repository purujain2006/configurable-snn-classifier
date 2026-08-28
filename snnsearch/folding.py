"""Removing batch normalization by folding it into weights and thresholds.

Moved verbatim from Practice2.py lines 1503-1707 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

from .hardware import W_ALPHA, W_DELTA, INT16_MAX
from .quantgrid import fake_quantize_weight, quantized_threshold_int, fold_bias_band
from .neuron import HardwareLIFNode
from ._torch import _HAS_TORCH, _require_torch, _no_grad, torch, nn, layer, functional

# verify_fold needs the time loop from train.py. Imported inside the function
# rather than at module scope, because train.py imports this module.


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
    Per-conv distribution of the folded bias, for deciding whether
    bias_mode='threshold' is safe. Call on the trained eval-mode net BEFORE
    folding.

    Two different numbers matter, and only the second one tracks the leak:

      |b'|/theta       whether theta' = theta - b' is STORABLE. Above 1 the
                       threshold goes negative and the neuron fires every step.

      |b'|*tau/theta   whether the threshold form BEHAVES like the bias form.
                       A bias arrives every timestep and accumulates against the
                       leak, settling at b'*tau, while a threshold shift is
                       worth b' once. The old rule of thumb (0.1*theta folds
                       cleanly, 0.5*theta does not) holds only at small tau: at
                       tau=63 a bias of 0.01*theta already moves the membrane by
                       0.63*theta. Keep this ratio below ~0.5, or set
                       TrainSpec.fold_bias_qat_form='threshold' (the default) so
                       training runs the deployed form and the ratio stops
                       mattering.
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
            # the leak multiplies the bias's effect on the membrane, so the
            # behavioural ratio carries tau while the storability ratio does not
            tau = float(torch.as_tensor(node.hw_tau).item()) if node is not None \
                and hasattr(node, "hw_tau") else 1.0
            ab = b_prime.abs()
            # theta' = theta - b' must land in (0, w_alpha] to be storable.
            # Report how far outside that band the raw fold would fall, which is
            # what TrainSpec.fold_bias_margin exists to prevent during QAT.
            lo, hi = fold_bias_band(th, 0.0)
            below = (b_prime < lo)          # theta' > w_alpha, saturates at INT16_MAX
            above = (b_prime > hi)          # theta' <= 0, fires unconditionally
            rows.append({"conv_index": i, "theta": th, "tau": tau,
                         "mean_abs_b_over_theta": (ab.mean() / th).item(),
                         "max_abs_b_over_theta": (ab.max() / th).item(),
                         # behavioural ratio: how far the accumulated bias moves
                         # the membrane, relative to the threshold it competes with
                         "mean_abs_b_tau_over_theta": (ab.mean() * tau / th).item(),
                         "max_abs_b_tau_over_theta": (ab.max() * tau / th).item(),
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
