"""Differentiable stand-ins for the converter's integer arithmetic.

Every function here mirrors something hs_api does at conversion time, so
training sees the numbers the chip will use.

Moved verbatim from Practice2.py lines 783-854 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

from .hardware import (W_BITS, W_ALPHA, INT16_MAX, W_DELTA,
                       HW_TAU_CHOICES, HW_TAU_MIN, HW_TAU_MAX)
from ._torch import _HAS_TORCH, _require_torch, torch, nn, layer


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


if not _HAS_TORCH:
    # See the note in build_from_practice2.py: these must exist as names even
    # without torch, so that importing a dependent module gives a clear error
    # from _require_torch() rather than an ImportError about a private symbol.
    _ste = None
