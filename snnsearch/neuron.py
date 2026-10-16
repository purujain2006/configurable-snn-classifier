"""The neuron the chip runs, and the folded conv that feeds it during training.

    HardwareLIFNode   fires before it integrates, leaks by an integer shift,
                      and stores tau and threshold on the chip's grid
    ConvBNFoldQuant   a conv with its BatchNorm folded in and its weights
                      quantized on every forward pass (quantization-aware
                      training), with the folded bias sent to the threshold or
                      kept as a bias for bias neurons
    build_neuron      the one place a neuron is constructed
"""

import math
from typing import Callable

from .config import NeuronSpec
from .hardware import (W_ALPHA, W_DELTA, INT16_MAX, HW_TAU_CHOICES, HW_TAU_MIN, HW_TAU_MAX,
                       HW_LOG2_TAU_MIN, HW_LOG2_TAU_MAX)
from .quantgrid import (_ste, fake_quantize_weight, fake_quantize_threshold,
                        fake_quantize_bias, quantized_threshold_int, fold_bias_band,
                        constrain_fold_bias)
from ._torch import (_HAS_TORCH, _HAS_HS_API, _require_torch, _no_grad,
                     torch, nn, F, layer, functional, surrogate, neuron,
                     Custom_LIFNode, Custom_IFNode)


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
                     integer_leak: bool = True,
                     surrogate_function: Callable = None, detach_reset: bool = True,
                     step_mode: str = "s", store_v_seq: bool = False,
                     tau_per_channel: bool = False, num_channels: int = None,
                     signed: bool = False):
            super().__init__(v_threshold=float(v_threshold), v_reset=v_reset,
                             surrogate_function=surrogate_function or surrogate.ATan(),
                             detach_reset=detach_reset, step_mode=step_mode,
                             backend="torch", store_v_seq=store_v_seq)
            self.learn_tau = bool(learn_tau)
            self.learn_threshold = bool(learn_threshold)
            self.integer_leak = bool(integer_leak)
            # signed spikes: a second membrane integrates the negated input, as
            # the second chip neuron does; the unit outputs spike - spike_neg
            self.signed = bool(signed)
            if self.signed:
                self.register_memory("v_neg", 0.0)

            tau0 = float(min(max(int(round(float(tau))), HW_TAU_MIN), HW_TAU_MAX))
            if self.learn_tau:
                # Learned in log2 space and rounded there, so tau can only be
                # a power of two: the chip's leak is a shift (V -= V >> leak).
                # Checkpoints from before this stored raw_tau, a linear tau
                # that rounded to any integer; _load_from_state_dict keeps
                # those loading as they trained.
                if tau_per_channel and num_channels:
                    # one learned tau per output channel; the chip stores a
                    # neuron model per neuron, so this costs nothing on chip
                    self.raw_log2_tau = nn.Parameter(
                        torch.full((int(num_channels),), math.log2(tau0)))
                else:
                    self.raw_log2_tau = nn.Parameter(torch.tensor(math.log2(tau0)))
            else:
                self.register_buffer("_tau_const", torch.tensor(tau0))

            if self.learn_threshold:
                th0 = float(min(max(float(v_threshold), 1e-3), 1.0 - 1e-4))
                self.raw_threshold = nn.Parameter(torch.tensor(math.log(th0 / (1.0 - th0))))

        # ---- hardware parameters, always reported on the chip's grid --------
        @property
        def hw_tau(self):
            if self.learn_tau:
                if hasattr(self, "raw_log2_tau"):
                    e = torch.clamp(self.raw_log2_tau, HW_LOG2_TAU_MIN, HW_LOG2_TAU_MAX)
                    return 2.0 ** _ste(torch.round(e), e)   # power of two, STE gradient
                # legacy checkpoint: linear tau, any integer
                t = torch.clamp(self.raw_tau, HW_TAU_MIN, HW_TAU_MAX)
                return _ste(torch.round(t), t)
            return self._tau_const

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            # A checkpoint written before log2 tau holds `raw_tau`. Swap the
            # parameter so it loads unchanged and deploys the tau it trained
            # with; converting it to log2 would silently move it (3 -> 2 or 4).
            if (self.learn_tau and prefix + "raw_tau" in state_dict
                    and prefix + "raw_log2_tau" not in state_dict
                    and hasattr(self, "raw_log2_tau")):
                del self.raw_log2_tau
                self.raw_tau = nn.Parameter(state_dict[prefix + "raw_tau"].detach().clone())
            return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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

        def hw_leak(self):
            """The deployed tau: an int, or a list with one per channel."""
            t = torch.as_tensor(self.hw_tau).detach().reshape(-1)
            vals = [int(round(float(v))) for v in t]
            return vals[0] if len(vals) == 1 else vals

        def hw_leak_per_channel(self, n_channels: int) -> list:
            leak = self.hw_leak()
            return list(leak) if isinstance(leak, list) else [leak] * int(n_channels)

        def hw_threshold_int(self):
            """Integer threshold(s) written into LIF_neuron(threshold=...)."""
            return quantized_threshold_int(torch.as_tensor(self.v_threshold).detach())

        @property
        def v_threshold_neg(self):
            """Threshold of the negative neuron of a signed unit. It equals
            v_threshold unless threshold-mode folding gave the two different
            values: the folded bias lowers one threshold and raises the other."""
            v = getattr(self, "_v_threshold_neg_const", None)
            if v is None:
                return self.v_threshold
            return fake_quantize_threshold(torch.as_tensor(v))

        def hw_threshold_int_neg(self):
            return quantized_threshold_int(torch.as_tensor(self.v_threshold_neg).detach())

        @staticmethod
        def _per_channel(t, x):
            """Broadcast a per-channel tensor over (N, C, ...)."""
            if torch.is_tensor(t) and t.dim() == 1 and t.numel() > 1:
                return t.reshape([1, t.numel()] + [1] * (x.dim() - 2))
            return t

        def _fire_reset_leak(self, name, v_th, grid, tau, v_reset):
            """Steps 1 to 3 of the chip's update, on one membrane (self.<name>).

            1. Fire on the membrane carried in from the previous step. The chip
               compares v > theta (strictly). The membrane lives on a grid of
               `grid` (W_DELTA, or finer behind a pool), so v > theta means
               v >= theta + W_DELTA; the surrogate tests >= 0. Testing half a
               grid step lower gives the same answer for every value on the
               grid, so a float a hair off its grid point cannot flip it.
            2. Hard reset to v_reset.
            3. Leak by integer division on the grid: the chip subtracts
               floor(V / tau) in its integer units V. Adding half a unit before
               the floor gives the same result for every integer V and keeps a
               membrane that divides exactly by tau from rounding down a step
               in float. When |V| < tau the leak is zero, as on the chip. The
               straight-through estimator passes the float leak's gradient.
            """
            v = getattr(self, name)
            spike = self.surrogate_function(v - v_th - W_DELTA + 0.5 * grid)
            spike_d = spike.detach() if self.detach_reset else spike
            v = v_reset * spike_d + (1.0 - spike_d) * v
            leak = (v - v_reset) / tau
            if getattr(self, "integer_leak", True):
                units = (v - v_reset) / grid
                leak = _ste(torch.floor((units + 0.5) / tau) * grid, leak)
            setattr(self, name, v - leak)
            return spike

        # ---- dynamics -------------------------------------------------------
        def single_step_forward(self, x: "torch.Tensor"):
            self.v_float_to_tensor(x)
            if getattr(self, "signed", False) and not torch.is_tensor(self.v_neg):
                self.v_neg = torch.full_like(x.data, float(self.v_neg))
            v_reset = 0.0 if self.v_reset is None else float(self.v_reset)
            signed = getattr(self, "signed", False)
            v_th = self.v_threshold
            v_th_neg = self.v_threshold_neg if signed else None
            # exported per-channel thresholds are plain tensors, which
            # .to(device) does not move; follow the input instead
            if torch.is_tensor(v_th) and v_th.device != x.device:
                v_th = v_th.to(x.device)
            if torch.is_tensor(v_th_neg) and v_th_neg.device != x.device:
                v_th_neg = v_th_neg.to(x.device)

            # During inline-fold QAT in threshold mode, the folded bias b' is
            # delivered here each step instead of into the conv, and applied
            # the way the deployed chip applies it: as a threshold shift,
            # theta - b' (and theta + b' for the negative neuron of a signed
            # unit, which integrates the negated input). A bias added to the
            # input would build up against the leak (to b' * tau), which a
            # threshold shift does not, so the two are not interchangeable.
            fb = getattr(self, "_fold_bias", None)
            if fb is not None:
                fb_b = fb.reshape([1, fb.shape[0]] + [1] * (x.dim() - 2))
                if getattr(self, "_fold_bias_form", "threshold") == "threshold":
                    base = v_th
                    v_th = fake_quantize_threshold(base - fb_b)
                    if signed:
                        v_th_neg = fake_quantize_threshold(base + fb_b)
                else:
                    x = x + fb_b            # legacy form, kept for A/B only

            grid = getattr(self, "leak_grid", W_DELTA)
            tau = self._per_channel(self.hw_tau, x)
            spike = self._fire_reset_leak("v", v_th, grid, tau, v_reset)
            # 4. integrate this step's input, undecayed
            self.v = self.v + x
            if not signed:
                return spike
            spike_neg = self._fire_reset_leak("v_neg", v_th_neg, grid, tau, v_reset)
            self.v_neg = self.v_neg - x
            return spike - spike_neg

        def extra_repr(self):
            th = self.v_threshold
            th = torch.as_tensor(th).detach()
            th_s = "per-channel" if th.numel() > 1 else f"{float(th):.5f}"
            leak_s = "floored" if getattr(self, "integer_leak", True) else "float"
            leak = self.hw_leak()
            tau_s = f"per-channel ({len(leak)})" if isinstance(leak, list) else str(leak)
            return (f"tau={tau_s} ({leak_s} leak), v_threshold={th_s}, "
                    f"learn_tau={self.learn_tau}, learn_threshold={self.learn_threshold}, "
                    f"signed={getattr(self, 'signed', False)}")


    class _WeightFakeQuant(nn.Module):
        """Parametrization that makes `module.weight` return the INT16 value the
        chip stores, while gradients flow to the underlying float parameter."""
        def __init__(self, max_int: int = INT16_MAX):
            super().__init__()
            self.max_int = int(max_int)

        def forward(self, w):
            return fake_quantize_weight(w, self.max_int)


    class _BiasFakeQuant(nn.Module):
        """Same, for a bias delivered by bias neurons."""
        def forward(self, b):
            return fake_quantize_bias(b)


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
                     fold_bias_margin: float = 0.05,
                     fold_bias_qat_form: str = "threshold",
                     max_weight_int: int = INT16_MAX):
            super().__init__()
            # below INT16_MAX for a first layer reading multi-bit input
            self.max_weight_int = int(max_weight_int)
            self.fold_bias_qat_form = fold_bias_qat_form
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
                # STE: forward rounds, backward passes
                w = fake_quantize_weight(w, getattr(self, "max_weight_int", INT16_MAX))
            # In threshold mode b' becomes theta - b' on chip, so it has to stay
            # inside the storable band. Applied here rather than at export so the
            # forward pass, and therefore the loss, sees the deployed value.
            # conv mode needs no constraint: b' stays on the conv, untouched.
            if self.bias_mode == "threshold" and self.fold_bias_margin > 0:
                b = constrain_fold_bias(b, self._sink_threshold(), self.fold_bias_margin,
                                        signed=getattr(self._bias_sink, "signed", False))
            # bias neurons deliver b' as an INT16 synapse weight every step
            if self.bias_mode in ("neuron", "conv") and self.quantize:
                b = fake_quantize_bias(b)
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
            use_bias = b if self.bias_mode in ("neuron", "conv") else None
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
                # Tell the neuron HOW to apply it. "threshold" mirrors what
                # export_deployed writes; "input" is the older form, which
                # drifts from the deployed network as tau rises.
                self._bias_sink._fold_bias_form = getattr(
                    self, "fold_bias_qat_form", "threshold")
            return y

        @torch.no_grad()
        def export_folded_conv(self):
            """Freeze into a plain Conv2d holding the final folded+quantized
            weight -- what fold_bn would have produced, but already grid-trained."""
            w, b = self._folded_weight_bias()
            has_bias = self.bias_mode in ("neuron", "conv")
            conv = layer.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups, bias=has_bias)
            conv.weight.data.copy_(w)
            conv._max_weight_int = getattr(self, "max_weight_int", INT16_MAX)
            if has_bias:
                conv.bias.data.copy_(b)
            conv.step_mode = self.step_mode
            return conv, b


    from spikingjelly.activation_based import base as _sj_base

    class OrPool(_sj_base.MemoryModule):
        """Max pooling of spikes, as the chip runs it: one OR neuron per window.

        The chip has no pooling operation. Folding an AVERAGE pool into the next
        layer is exact but multiplies that layer's fan-in by k*k (hardware.py).
        A MAX pool of 0/1 spikes is an OR, and one memoryless neuron computes
        it exactly: weight 1 from each input in the window, threshold 0, so it
        fires when any input fired. That keeps the next layer's fan-in at the
        pooled size.

        The OR neuron fires before it integrates like every chip neuron, so its
        output arrives one timestep after its input. This module reproduces
        that delay, which is why it counts as a spiking layer for the flush.
        """

        def __init__(self, kernel_size: int = 2, stride: int = 2, step_mode: str = "s"):
            super().__init__()
            self.kernel_size = int(kernel_size)
            self.stride = int(stride)
            self.step_mode = step_mode
            self.register_memory("prev", None)

        def _pool(self, x):
            # A signed input is two spike trains, +1 and -1. Each gets its own
            # OR neurons on chip, so the pooled value is OR(+) - OR(-). For
            # ordinary 0/1 spikes the second term is zero.
            k, s = self.kernel_size, self.stride
            return F.max_pool2d(F.relu(x), k, s) - F.max_pool2d(F.relu(-x), k, s)

        def single_step_forward(self, x):
            pooled = self._pool(x)
            out = self.prev if self.prev is not None else torch.zeros_like(pooled)
            self.prev = pooled
            return out

        def multi_step_forward(self, x_seq):
            # x_seq is (T, N, C, H, W): pool every step, then shift one step later
            T, N = x_seq.shape[:2]
            pooled = self._pool(x_seq.flatten(0, 1))
            pooled = pooled.reshape(T, N, *pooled.shape[1:])
            first = self.prev if self.prev is not None else torch.zeros_like(pooled[0])
            self.prev = pooled[-1]
            return torch.cat([first.unsqueeze(0), pooled[:-1]], dim=0)

        def extra_repr(self):
            return f"kernel_size={self.kernel_size}, stride={self.stride}, one OR neuron per window"


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
    HardwareLIFNode = _WeightFakeQuant = _BiasFakeQuant = _LegacyLearnableThresholdLIF = None
    OrPool = None

LearnableThresholdLIF = _LegacyLearnableThresholdLIF   # back-compat alias


def build_neuron(neuron_cfg: NeuronSpec, tau, v_threshold: float, num_channels: int = None):
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
        integer_leak=getattr(neuron_cfg, "integer_leak", True),
        surrogate_function=surrogate.ATan(), detach_reset=True,
        tau_per_channel=bool(getattr(neuron_cfg, "tau_per_channel", False)
                             and neuron_cfg.trainable_tau),
        num_channels=num_channels,
        signed=bool(getattr(neuron_cfg, "signed_spikes", False)),
    )


if not _HAS_TORCH:
    # Same reason as quantgrid: model.py imports these by name at load time.
    TdBatchNorm2d = ConvBNFoldQuant = None
