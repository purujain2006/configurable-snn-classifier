"""Turn a deployed checkpoint into the integer network HiAER-Spike runs.

WHY THIS EXISTS INSTEAD OF hs_api's CRI_Converter

The converter on the pinned L6m-testing-suite commit (a8f0f7b5) cannot take
this project's networks:

  * it reads thresholds as `model[k+1].v_threshold` and only recognises
    SpikingJelly's own LIFNode/IFNode, so HardwareLIFNode is skipped;
  * it writes leak = 63 for every neuron, which simpleSim reads as a SHIFT
    (v -= v // 2**63), so every layer becomes a non-leaky IF neuron;
  * it has no AvgPool path, and its MaxPool path assigns into a tuple and
    raises before producing anything.

The checkpoint already holds everything needed (folded INT16 weights, the
per-channel thresholds that absorbed the BN bias, the leak), so this module
builds the integer network from the checkpoint directly and checks it against
the real simpleSim instead.

THE ARITHMETIC SIMPLESIM PERFORMS, PER TIMESTEP (hs_api/_simple_sim.py)

    spiked = V > threshold
    V[spiked] = 0
    V[LIF] -= V[LIF] // 2**leak          floor division, so leak is a SHIFT
    V += W_axon @ axons_t + W_neuron @ spiked

So tau must be a power of two. A tau of 3 has no representation.

AVERAGE POOLING

A spike is 0 or 1, and AvgPool2d(2) of spikes is a multiple of 1/4. That is
linear, so it folds into the next layer: each of the four spikes in a pooling
window gets the next layer's weight divided by 4. Dividing an INT16 weight by 4
leaves a fraction, so instead the next layer's membrane is kept in units 4x
finer. The synapse then carries the INT16 weight unchanged and the threshold is
scaled to match. `scale` below is that factor, 1 for a layer fed directly by
spikes and 4 (or 4*4 for two pools in a row) for a layer behind a pool.

The trained model fires when v - theta >= W_DELTA. With v = V * W_DELTA / scale
that is V >= scale*theta_int + scale, so the integer threshold for `>` is
scale*theta_int + scale - 1.
"""

import math

import numpy as np

from .hardware import INT16_MAX, W_DELTA

# simple_sim's membrane format: "fxp-s35/0" on a8f0f7b5, "fxp-s32/0" with
# two's-complement wraparound on later L6m-testing-suite commits (the FPGA's
# MP update is 32 bits). The stricter one applies. Weights are "fxp-s16/0".
MEMBRANE_LIMIT = 2 ** 31
from ._torch import _require_torch, torch, nn, F, layer


# ---------------------------------------------------------------------------
# loading the deployed network
# ---------------------------------------------------------------------------

def load_deployed(ckpt_path, device="cpu"):
    """Rebuild the exact network `hw_test_accuracy` was measured on.

    `hw_state_dict` cannot do this alone: export_deployed() stores the
    per-channel thresholds as a plain attribute, which state_dict() skips. So
    the QAT-form network is rebuilt, its state loaded, and export_deployed()
    run again, which is the same path run_training() took.

    Returns (hw_net, spec, ckpt).
    """
    _require_torch()
    from .config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                         HeadSpec, NeuronSpec, TrainSpec)
    from .model import build_model

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["config"]
    spec = {
        "input": InputSpec(**c["input"]),
        "encoder": EncoderSpec(**c["encoder"]),
        "output": OutputSpec(**c["output"]),
        "downsample": DownsampleSpec(**c["downsample"]),
        "head": HeadSpec(**c["head"]),
        "neuron": NeuronSpec(**c["neuron"]),
        "train": TrainSpec(**c["train"]),
    }
    t = spec["train"]
    net = build_model(spec).to(device)
    if ckpt.get("qat_mode") == "inline":
        net.to_qat_folded(bias_mode=t.fold_bias_mode,
                          fold_bias_margin=t.fold_bias_margin,
                          fold_bias_qat_form=t.fold_bias_qat_form)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        # export_deployed builds its plain Conv2d on the CPU
        net = net.export_deployed().to(device)
    else:
        raise NotImplementedError(
            f"qat_mode={ckpt.get('qat_mode')!r}: only inline-QAT checkpoints "
            "are wired here. The thresholds of a tail/ptq checkpoint come from "
            "fold_bn and would need that path instead.")
    net.eval()
    if ckpt.get("leak_grid") != "chip":
        # Trained before the model floored its leak on the chip's grid: every
        # layer floored on W_DELTA. Restore that so the checkpoint scores what
        # it scored. make_chip_faithful() switches it to what the chip does.
        from .neuron import HardwareLIFNode
        for m in net.conv_fc:
            if isinstance(m, HardwareLIFNode):
                m.leak_grid = W_DELTA
    _check_against_export(net, ckpt.get("hardware_export"))
    return net, spec, ckpt


def _check_against_export(net, export):
    """The rebuilt thresholds must equal the ones saved at training time."""
    if not export:
        return
    from .neuron import HardwareLIFNode
    nodes = [m for m in net.conv_fc if isinstance(m, HardwareLIFNode)]
    for node, rec in zip(nodes, export["neuron_layers"]):
        th = torch.as_tensor(node.hw_threshold_int()).reshape(-1).tolist()
        saved = rec["threshold_int"]
        saved = saved if isinstance(saved, list) else [saved]
        if len(th) == 1 and len(saved) > 1:
            th = th * len(saved)
        if th != saved or node.hw_leak() != rec["leak"]:
            raise RuntimeError(
                f"rebuilt layer {rec['layer_index']} does not match the "
                f"checkpoint's hardware_export (threshold or leak differ). The "
                f"network being converted is not the one that was scored.")


# ---------------------------------------------------------------------------
# the integer description
# ---------------------------------------------------------------------------

def leak_shift_for(tau):
    """tau -> the shift simpleSim uses, or None when tau is not a power of 2."""
    tau = int(tau)
    if tau >= 1 and tau & (tau - 1) == 0:
        return int(math.log2(tau))
    return None


def integer_layers(net, tau_override=None):
    """Read the deployed net into a list of integer layers, in order.

    Each entry:
        kind          "conv", "linear", or "orpool" (a layer of OR neurons: one
                      memoryless neuron per pooling window, weight 1 from each
                      input, threshold 0, so it fires when any input fired)
        w             int64 array, the INT16 weight the chip stores
        stride, padding
        pool          avg-pool kernel applied to this layer's INPUT spikes (1 = none);
                      a global average pool counts as a pool the size of the map
        scale         membrane unit relative to the model's W_DELTA grid
        threshold     int64 per output channel, for the `V > threshold` compare
        signed        True when every unit is two neurons, +1 and -1
        threshold_neg the negative neurons' thresholds, for a signed layer
        bias          int64 per output channel in membrane units, added every
                      step by bias neurons (subtracted for negative neurons), or None
        tau, leak     per output channel: tau, and the simpleSim shift (None
                      when tau is not a power of two)
        input_bits    on the first layer: bits per input value (B axons each)
    """
    from .neuron import HardwareLIFNode, OrPool

    mods = list(net.conv_fc)
    out, pending_pool = [], 1
    last_signed = False
    i = 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, (layer.AvgPool2d, nn.AvgPool2d)):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            if k != s:
                raise NotImplementedError("overlapping avg pool does not fold exactly")
            pending_pool *= k
            i += 1
            continue
        if isinstance(m, (layer.AdaptiveAvgPool2d, nn.AdaptiveAvgPool2d)):
            # GAP: an average over the whole (square) map, folded the same way
            h, w_ = net.plan.encoder_out_hw
            if h != w_:
                raise NotImplementedError("GAP over a non-square map")
            pending_pool *= h
            i += 1
            continue
        if OrPool is not None and isinstance(m, OrPool):
            if pending_pool != 1:
                raise NotImplementedError("an average pool directly before an OR pool")
            if m.kernel_size != m.stride:
                raise NotImplementedError("overlapping OR pool")
            out.append({"kind": "orpool", "pool": m.kernel_size, "scale": 1,
                        "threshold": np.zeros(1, dtype=np.int64),
                        "threshold_neg": np.zeros(1, dtype=np.int64) if last_signed else None,
                        "signed": last_signed, "bias": None, "tau": None,
                        "leak": np.zeros(1, dtype=np.int64), "memoryless": True})
            i += 1
            continue
        if isinstance(m, (layer.MaxPool2d, nn.MaxPool2d)):
            raise NotImplementedError(
                "a plain MaxPool2d has no chip timing; build max pooling as OrPool")
        if isinstance(m, (layer.Conv2d, nn.Conv2d, layer.Linear, nn.Linear)):
            node = next(x for x in mods[i + 1:] if isinstance(x, HardwareLIFNode))
            w = m.weight.detach().double().cpu().numpy()
            w_int = np.rint(w * INT16_MAX).astype(np.int64)
            # stored as float32, so k/32767 comes back within ~1e-3 of k
            if np.abs(w * INT16_MAX - w_int).max() > 1e-2:
                raise RuntimeError(f"{type(m).__name__} weights are not on the INT16 grid")
            bits = int(getattr(net, "input_bits", 1) or 1) if not out else 1
            if np.abs(w_int).max() > (INT16_MAX >> (bits - 1)):
                raise RuntimeError(
                    f"weight {np.abs(w_int).max()} too large for {bits}-bit input: the "
                    f"top bit's synapse carries it times {2 ** (bits - 1)}, past INT16")
            # the pool sits between the previous neuron and this layer's
            # synapses, so it scales THIS layer's membrane
            pool = pending_pool
            scale = pool * pool
            pending_pool = 1
            n_out = w.shape[0]

            def per_channel(t):
                t = torch.as_tensor(t).reshape(-1).cpu().numpy().astype(np.int64)
                return np.full(n_out, int(t[0]), dtype=np.int64) if t.size == 1 else t

            th = per_channel(node.hw_threshold_int())
            signed = bool(getattr(node, "signed", False))
            th_neg = per_channel(node.hw_threshold_int_neg()) if signed else None
            taus = ([int(tau_override)] * n_out if tau_override is not None
                    else node.hw_leak_per_channel(n_out))
            bias = None
            if getattr(m, "bias", None) is not None:
                b = m.bias.detach().double().cpu().numpy()
                b_int = np.rint(b * INT16_MAX).astype(np.int64)
                if np.abs(b * INT16_MAX - b_int).max() > 1e-2:
                    raise RuntimeError(f"{type(m).__name__} bias is not on the INT16 grid")
                bias = scale * b_int
            entry = {
                "kind": "conv" if w.ndim == 4 else "linear",
                "w": w_int,
                "pool": pool,
                "scale": scale,
                "threshold": scale * th + scale - 1,
                "threshold_int_model": th,
                "signed": signed,
                "threshold_neg": (scale * th_neg + scale - 1) if signed else None,
                "bias": bias,
                "tau": taus,
                "leak": [leak_shift_for(t) for t in taus],
                "input_bits": bits,
            }
            if w.ndim == 4:
                entry["stride"] = tuple(m.stride)
                entry["padding"] = tuple(m.padding)
            out.append(entry)
            last_signed = signed
            i += 1
            continue
        i += 1      # neuron, dropout, flatten: no synapses of their own
    return out


def make_chip_faithful(net, tau=None):
    """Make the PyTorch model compute what the chip computes, in place.

    Two changes, both evaluation-only:
      * tau -> the given power of two, on every spiking layer (None keeps the
        trained tau, which only works if it already is a power of two);
      * the leak floors on the membrane's own grid, W_DELTA / scale, instead of
        W_DELTA. Training floored every layer on W_DELTA, which behind a pool
        rounds the leak to a multiple of `scale` LSBs where the chip rounds to 1.
    Returns the per-layer scales applied.
    """
    from .neuron import HardwareLIFNode
    nodes = [m for m in net.conv_fc if isinstance(m, HardwareLIFNode)]
    scales = [L["scale"] for L in integer_layers(net) if L["kind"] != "orpool"]
    for node, s in zip(nodes, scales):
        if tau is not None:
            if node.learn_tau and hasattr(node, "raw_log2_tau"):
                node.raw_log2_tau.data.fill_(math.log2(float(tau)))
            elif node.learn_tau:
                node.raw_tau.data.fill_(float(tau))
            else:
                node._tau_const.fill_(float(tau))
        node.leak_grid = W_DELTA / s
    return scales


def _normalize(L):
    """Fill defaults so hand-built layer lists (the selftest's) use the same code."""
    L = dict(L)
    L.setdefault("signed", False)
    L.setdefault("threshold_neg", None)
    L.setdefault("bias", None)
    L.setdefault("memoryless", L["kind"] == "orpool")
    L.setdefault("input_bits", 1)
    n = L["w"].shape[0] if "w" in L else 1
    leak = L.get("leak", 0)
    if not isinstance(leak, (list, tuple, np.ndarray)):
        leak = [leak] * n
    L["leak"] = list(leak)
    if L["signed"] and L["threshold_neg"] is None:
        L["threshold_neg"] = L["threshold"]
    return L


# ---------------------------------------------------------------------------
# the fast simulator
# ---------------------------------------------------------------------------

class IntegerSim:
    """simpleSim's arithmetic, layer by layer, on whole tensors.

    Same update as hs_api's simple_sim.step_run, but it keeps the convolution
    as a convolution instead of unrolling millions of synapses into a sparse
    matrix, so a clip takes a fraction of a second on a GPU. float64 holds
    every membrane exactly (all values are integers far below 2**53).

    A signed layer is two neuron groups: one integrates the drive, the other
    its negative, and the unit outputs spike - spike_neg. B-bit input needs no
    special case: the first layer's drive from integer levels equals the sum
    over bits of bit_k * (w * 2**k), which is what the chip's B axons deliver.

    tools/match_simplesim.py --selftest is what earns this the right to stand
    in for simpleSim: it unrolls small networks of every layer type into
    hs_api's axon/neuron dicts, runs the real simple_sim, and requires every
    membrane and spike of every neuron at every timestep to agree.
    """

    def __init__(self, layers, in_shape, device="cpu"):
        layers = [_normalize(L) for L in layers]
        for L in layers:
            if L["kind"] != "orpool" and any(lk is None for lk in L["leak"]):
                raise ValueError(
                    f"tau={L.get('tau')} includes a value that is not a power of two, so "
                    f"simpleSim cannot represent it (its leak is v // 2**leak). Pass a "
                    f"power-of-two tau_override.")
        self.layers = layers
        self.in_shape = tuple(in_shape)
        self.device = device
        dt = torch.float64

        def shaped(values, L):
            t = torch.as_tensor(np.asarray(values), dtype=dt, device=device).reshape(-1)
            return t.reshape(-1, 1, 1) if L["kind"] in ("conv", "orpool") else t

        self.w = [torch.as_tensor(L["w"], dtype=dt, device=device) if "w" in L else None
                  for L in layers]
        self.th = [shaped(L["threshold"], L) for L in layers]
        self.th_neg = [shaped(L["threshold_neg"], L) if L["signed"] else None for L in layers]
        self.div = [shaped([2.0 ** lk for lk in L["leak"]], L) for L in layers]
        self.b = [shaped(L["bias"], L) if L["bias"] is not None else None for L in layers]

    def _drive(self, idx, x):
        """Synaptic input to layer idx from the previous layer's spikes (signed
        spikes are -1/0/+1). An OR-pool layer returns (positive, negative)
        window counts, one per neuron group."""
        # Every drive is an integer. Rounding makes it exactly one whatever
        # algorithm the convolution used: on a GPU, cuDNN may pick an FFT
        # convolution whose float64 error is tiny but not zero, and a membrane
        # exactly on its threshold then lands on the wrong side.
        L = self.layers[idx]
        if L["kind"] == "orpool":
            k = L["pool"]
            pos = torch.round(F.avg_pool2d(F.relu(x), k) * (k * k))
            neg = torch.round(F.avg_pool2d(F.relu(-x), k) * (k * k))
            return pos, neg
        if L["pool"] > 1:
            # sum, not mean: the 1/pool^2 lives in `scale`, which keeps it integer
            x = torch.round(F.avg_pool2d(x, L["pool"]) * (L["pool"] ** 2))
        if L["kind"] == "conv":
            return torch.round(F.conv2d(x, self.w[idx], stride=L["stride"], padding=L["padding"]))
        return torch.round(F.linear(x.flatten(1), self.w[idx]))

    @torch.no_grad()
    def steps(self, frames, flush_steps):
        """Run one clip, yielding (groups, outputs) after every timestep.

        groups   [(membranes, spikes), ...] for every neuron group in the order
                 unroll_to_hs_api names them: per layer, the positive group,
                 then the negative group if the layer is signed
        outputs  per layer, the unit outputs the next layer reads

        frames: (T, C, H, W) of input levels (0/1 spikes for 1-bit input). Runs
        T steps with input, then flush_steps with none, so the last frame
        reaches the output. Fresh membranes each call, as the chip is cleared.

        Each step is simpleSim's, in its order:
          phase one   every neuron fires on the membrane carried in (V > threshold)
                      and fired neurons reset to 0; LIF neurons leak
                      V -= floor(V / 2**leak); memoryless (OR) neurons reset to 0
          phase two   add the input axons, the spikes fired in phase one, and
                      the bias neurons, which fire on every step (flush included)
        """
        T = frames.shape[0]
        zeros = torch.zeros_like(frames[0])
        V = Vn = None
        for t in range(T + flush_steps):
            x = (frames[t] if t < T else zeros).to(self.device, torch.float64).unsqueeze(0)
            if V is None:
                V, Vn, h = [], [], x
                for i, L in enumerate(self.layers):
                    d = self._drive(i, h)
                    d = d[0] if isinstance(d, tuple) else d
                    V.append(torch.zeros(d.shape, dtype=torch.float64, device=self.device))
                    Vn.append(torch.zeros_like(V[-1]) if L["signed"] else None)
                    h = torch.zeros_like(V[-1])
            # phase one
            groups, outputs = [], []
            for i, L in enumerate(self.layers):
                sp = (V[i] > self.th[i]).to(torch.float64)
                if L["memoryless"]:
                    V[i] = torch.zeros_like(V[i])
                else:
                    V[i] = V[i] * (1.0 - sp)
                    V[i] = V[i] - torch.floor(V[i] / self.div[i])
                if L["signed"]:
                    sn = (Vn[i] > self.th_neg[i]).to(torch.float64)
                    if L["memoryless"]:
                        Vn[i] = torch.zeros_like(Vn[i])
                    else:
                        Vn[i] = Vn[i] * (1.0 - sn)
                        Vn[i] = Vn[i] - torch.floor(Vn[i] / self.div[i])
                    groups.append([V[i], sp])
                    groups.append([Vn[i], sn])
                    outputs.append(sp - sn)
                else:
                    groups.append([V[i], sp])
                    outputs.append(sp)
            # phase two
            for i, L in enumerate(self.layers):
                d = self._drive(i, x if i == 0 else outputs[i - 1])
                if L["kind"] == "orpool":
                    V[i] = V[i] + d[0]
                    if L["signed"]:
                        Vn[i] = Vn[i] + d[1]
                    continue
                V[i] = V[i] + d
                if self.b[i] is not None:
                    V[i] = V[i] + self.b[i]
                if L["signed"]:
                    Vn[i] = Vn[i] - d
                    if self.b[i] is not None:
                        Vn[i] = Vn[i] - self.b[i]
            # the membranes after phase two are what simpleSim reports
            k = 0
            for i, L in enumerate(self.layers):
                groups[k][0] = V[i]
                k += 1
                if L["signed"]:
                    groups[k][0] = Vn[i]
                    k += 1
            # simpleSim's membrane is a 32-bit integer that wraps (35-bit and
            # saturating on older commits); float64 does neither, so the two
            # agree only while nothing gets near the rail
            peak = max(float(v.abs().max()) for v, _s in groups)
            self.max_abs_membrane = max(getattr(self, "max_abs_membrane", 0.0), peak)
            if peak >= MEMBRANE_LIMIT:
                raise OverflowError(
                    f"membrane reached {peak:.0f}, past simpleSim's 32-bit range "
                    f"({MEMBRANE_LIMIT}); IntegerSim no longer matches it")
            yield groups, outputs

    def run(self, frames, flush_steps, record=False, record_output=False):
        """One clip. Returns (output spike count per class, extra), where extra is
        the net spike count of every layer per step (record=True), the output
        layer's spikes per step (record_output=True), or None. For signed
        output a -1 spike counts against its class."""
        counts, per_layer, rows = None, [[] for _ in self.layers], []
        for _groups, outputs in self.steps(frames, flush_steps):
            out = outputs[-1].reshape(-1)
            counts = out if counts is None else counts + out
            if record:
                for i, s in enumerate(outputs):
                    per_layer[i].append(int(s.sum().item()))
            if record_output:
                rows.append([int(v) for v in out.tolist()])
        extra = rows if record_output else per_layer if record else None
        return counts.cpu(), extra


# ---------------------------------------------------------------------------
# unrolling into hs_api's dictionaries
# ---------------------------------------------------------------------------

def hs_lif_neuron(threshold, leak):
    """hs_api's LIF_neuron with noise off, on any L6m-testing-suite commit.

    Positional on purpose: the keywords were renamed after a8f0f7b5
    (threshold, shift, leak -> theta, nu, Lambda) but the order was kept.

    nu = -17 is what makes the noise zero. Later commits ADD a perturbation
    every step: an odd magnitude below 2**17, shifted right by -nu. At -17
    that is always 0; at anything larger it is not, and simpleSim stops being
    deterministic.
    """
    from hs_api.neuron_models import LIF_neuron
    return LIF_neuron(int(threshold), -17, int(leak))


def hs_bias_neuron():
    """An always-on neuron, the bias neuron of Christopher's converter_v2.

    Memoryless (ANN), threshold -1: its membrane is 0 at every step, 0 > -1,
    so it fires every step from the first. Noise off as in hs_lif_neuron.
    """
    from hs_api.neuron_models import ANN_neuron
    return ANN_neuron(-1, -17, 0)


def hs_or_neuron():
    """One OR-pool neuron: memoryless (ANN), threshold 0, so it fires on the
    step after any input in its window fired. Noise off as in hs_lif_neuron."""
    from hs_api.neuron_models import ANN_neuron
    return ANN_neuron(0, -17, 0)


# the chip's limits on one synapse weight and one neuron's fan-out
_MAX_WEIGHT = INT16_MAX
_MAX_NEURON_FANOUT = 4095


def split_bias(value, n_targets):
    """How bias neurons deliver one channel's bias.

    One synapse holds at most INT16_MAX, so a larger bias (common behind a pool,
    where the membrane unit is finer) is split into parts that sum exactly. One
    neuron feeds at most 4095 others, so a large channel is split across groups.
    Returns [(weight, target_slice), ...], one entry per bias neuron.
    """
    value = int(value)
    n_parts = max(1, -(-abs(value) // _MAX_WEIGHT))
    base, extra = divmod(abs(value), n_parts)
    sign = 1 if value >= 0 else -1
    parts = [sign * (base + (1 if k < extra else 0)) for k in range(n_parts)]
    out = []
    for w in parts:
        for start in range(0, n_targets, _MAX_NEURON_FANOUT):
            out.append((w, slice(start, min(start + _MAX_NEURON_FANOUT, n_targets))))
    return out


def _layer_shapes(layers, in_shape):
    shapes, shp = [], tuple(in_shape)
    for L in layers:
        if L["kind"] == "orpool":
            c, h, w = shp
            shp = (c, h // L["pool"], w // L["pool"])
        elif L["kind"] == "conv":
            c, h, w = shp
            if L["pool"] > 1:
                h, w = h // L["pool"], w // L["pool"]
            kh, kw = L["w"].shape[2:]
            ph, pw = L["padding"]
            sh, sw = L["stride"]
            shp = (L["w"].shape[0], (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1)
        else:
            shp = (L["w"].shape[0],)
        shapes.append(shp)
    return shapes


def _synapses(L, prev_shape, out_shape):
    """Every (pre index, post index, weight) the layer stores, over flat
    (C, H, W) indices, with any average pool folded in. OR pools give weight 1."""
    w = L.get("w")
    if L["kind"] == "orpool":
        c_in, h_in, w_in = prev_shape
        p = L["pool"]
        hp, wp = h_in // p, w_in // p
        for ci in range(c_in):
            for y in range(hp * p):
                for x in range(wp * p):
                    yield (ci * h_in + y) * w_in + x, (ci * hp + y // p) * wp + x // p, 1
        return
    if L["kind"] == "conv":
        c_in, h_in, w_in = prev_shape
        p = L["pool"]
        hp, wp = h_in // p, w_in // p
        c_out, h_out, w_out = out_shape
        kh, kw = w.shape[2:]
        ph, pw = L["padding"]
        sh, sw = L["stride"]
        for ci in range(c_in):
            for y in range(hp * p):
                for x in range(wp * p):
                    pre = (ci * h_in + y) * w_in + x
                    py, px = y // p, x // p
                    for oy in range(h_out):
                        ky = py + ph - oy * sh
                        if not 0 <= ky < kh:
                            continue
                        for ox in range(w_out):
                            kx = px + pw - ox * sw
                            if not 0 <= kx < kw:
                                continue
                            for co in range(c_out):
                                yield pre, (co * h_out + oy) * w_out + ox, w[co, ci, ky, kx]
        return
    if len(prev_shape) == 3 and L["pool"] > 1:
        c_in, h_in, w_in = prev_shape
        p = L["pool"]
        hp, wp = h_in // p, w_in // p
        for ci in range(c_in):
            for y in range(hp * p):
                for x in range(wp * p):
                    flat = (ci * hp + y // p) * wp + x // p
                    for o in range(w.shape[0]):
                        yield (ci * h_in + y) * w_in + x, o, w[o, flat]
        return
    for j in range(int(np.prod(prev_shape))):
        for o in range(w.shape[0]):
            yield j, o, w[o, j]


def unroll_to_hs_api(layers, in_shape, neuron_model_factory=None):
    """Build (axons, connections, groups) in CRI_network's format.

    Keys: input axons "A{i * B + k}" for bit k of flat (C, H, W) input value i
    (with 1-bit input, "A{i}", the order of Christopher's DVS example).
    Neurons "L{layer}_{flat index}", and "L{layer}n_{flat index}" for the
    negative neuron of a signed unit.

    groups is, per layer, a list of (keys, sign): the positive group, then the
    negative group if the layer is signed. A unit's output is the positive
    neuron's spike minus the negative one's.

    neuron_model_factory(threshold:int, leak:int) -> an hs_api LIF_neuron;
    defaults to hs_lif_neuron.

    Only call this on a small network: hs_api holds every synapse as a Python
    tuple, and a DVS-sized network has hundreds of millions.
    """
    neuron_model_factory = neuron_model_factory or hs_lif_neuron
    layers = [_normalize(L) for L in layers]
    C, H, W = in_shape
    n_in = C * H * W
    bits = int(layers[0].get("input_bits", 1) or 1)
    axons = {f"A{i * bits + k}": [] for i in range(n_in) for k in range(bits)}
    shapes = _layer_shapes(layers, in_shape)

    groups, connections = [], {}
    for li, L in enumerate(layers):
        shp = shapes[li]
        n = int(np.prod(shp))
        per_ch = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        layer_groups = [([f"L{li}_{j}" for j in range(n)], 1)]
        if L["signed"]:
            layer_groups.append(([f"L{li}n_{j}" for j in range(n)], -1))
        groups.append(layer_groups)
        for keys, sign in layer_groups:
            if L["kind"] == "orpool":
                or_model = hs_or_neuron()
                for k in keys:
                    connections[k] = ([], or_model)
                continue
            th = L["threshold"] if sign > 0 else L["threshold_neg"]
            models = {}
            for j, k in enumerate(keys):
                ch = j // per_ch
                key = (int(th[ch]), int(L["leak"][ch]))
                if key not in models:
                    models[key] = neuron_model_factory(*key)
                connections[k] = ([], models[key])

    # bias neurons: +b into positive neurons, -b into negative ones
    bias_model = None
    for li, L in enumerate(layers):
        if L["bias"] is None:
            continue
        shp = shapes[li]
        per_ch = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        for keys, sign in groups[li]:
            tag = "" if sign > 0 else "n"
            for ch, value in enumerate(L["bias"]):
                value = int(value) * sign
                if value == 0:
                    continue
                targets = keys[ch * per_ch:(ch + 1) * per_ch]
                for j, (w, sl) in enumerate(split_bias(value, len(targets))):
                    if bias_model is None:
                        bias_model = hs_bias_neuron()
                    connections[f"BiasN.L{li}{tag}.c{ch}.{j}"] = (
                        [(t, w) for t in targets[sl]], bias_model)

    def add(src, pre_key, post_key, weight):
        weight = int(weight)
        if weight == 0:
            return
        if src == "axon":
            axons[pre_key].append((post_key, weight))
        else:
            connections[pre_key][0].append((post_key, weight))

    prev_shape = (C, H, W)
    for li, L in enumerate(layers):
        # where this layer's input comes from: the axons (one group per bit,
        # weight times 2**k) or the previous layer's neuron groups (sign +-1)
        if li == 0:
            sources = [("axon", [f"A{i * bits + k}" for i in range(n_in)], 1, 2 ** k)
                       for k in range(bits)]
        else:
            sources = [("neuron", keys, sign, 1) for keys, sign in groups[li - 1]]
        for pre, post, weight in _synapses(L, prev_shape, shapes[li]):
            for src, src_keys, src_sign, mult in sources:
                for tgt_keys, tgt_sign in groups[li]:
                    if L["kind"] == "orpool":
                        # an OR neuron reads only the spikes of its own sign
                        if src_sign == tgt_sign:
                            add(src, src_keys[pre], tgt_keys[post], 1)
                    else:
                        add(src, src_keys[pre], tgt_keys[post],
                            weight * src_sign * tgt_sign * mult)
        prev_shape = shapes[li]
    return axons, connections, groups


def input_axons(frame, bits=1):
    """The axon keys that fire for one input frame of levels (C, H, W)."""
    flat = torch.as_tensor(frame).reshape(-1).to(torch.int64)
    keys = []
    for i in torch.nonzero(flat).flatten().tolist():
        v = int(flat[i])
        for k in range(bits):
            if (v >> k) & 1:
                keys.append(f"A{i * bits + k}")
    return keys


# ---------------------------------------------------------------------------
# checking a trained network on the simulator
# ---------------------------------------------------------------------------

def exact_copy(hw_net):
    """The deployed network in float64, every number exactly on its grid.

    The float32 model fires a spike when v - theta >= W_DELTA, tested half a
    grid step early so rounding cannot flip it. Behind a GAP layer the grid
    is W_DELTA / (map area), 1/121 of a weight step for an 11x11 map, and
    float32 cannot hold a weight of k/32767 that closely: summed over a few
    thousand inputs the error passes half a grid step and a spike flips. On
    a cluster GPU, with its faster convolution algorithms, that showed up as
    4 of 216 validation clips predicted differently from IntegerSim.

    float64 with weights, biases and thresholds re-rounded to k/32767 keeps
    the error some ten thousand times below that, so measurements of the
    deployed network use this copy. Training does not need it.
    """
    _require_torch()
    from torch.nn.utils import parametrize
    from copy import deepcopy
    from .neuron import HardwareLIFNode
    net = deepcopy(hw_net)
    for m in net.modules():
        for name in ("weight", "bias"):
            if parametrize.is_parametrized(m, name):
                parametrize.remove_parametrizations(m, name, leave_parametrized=True)
    net = net.double()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.copy_(torch.round(m.weight * INT16_MAX) / INT16_MAX)
                if m.bias is not None:
                    m.bias.copy_(torch.round(m.bias * INT16_MAX) / INT16_MAX)
            if isinstance(m, HardwareLIFNode):
                # plain attributes, which .double() does not reach; the
                # v_threshold properties round them onto the grid in float64
                for attr in ("_v_threshold_const", "_v_threshold_neg_const"):
                    v = getattr(m, attr, None)
                    if torch.is_tensor(v):
                        setattr(m, attr, v.double())
    return net


def verify_deployment(hw_net, loader, device, flush_steps, max_clips=None):
    """Run every clip through both the PyTorch model and IntegerSim.

    Called on every trial, so `deployable` is measured instead of assumed. On a
    GPU it costs seconds against a training run of minutes. Returns
        sim_val_accuracy     IntegerSim's accuracy on the loader
        sim_disagreements    clips where IntegerSim and PyTorch predict differently
        sim_reason           why the network could not be simulated, else ""
    A network that cannot be exported (a non-power-of-two tau, a max pool, a
    weight off the grid) reports sim_val_accuracy None and the reason.
    """
    _require_torch()
    from .train import forward_over_time, exact_arithmetic
    try:
        layers = integer_layers(hw_net)
        C, H, W = hw_net.plan.blocks[0].in_channels, *hw_net.plan.blocks[0].in_hw
        sim = IntegerSim(layers, (C, H, W), device=device)
    except Exception as exc:
        return {"sim_val_accuracy": None, "sim_disagreements": None,
                "sim_reason": f"{type(exc).__name__}: {exc}"[:300]}
    if next(hw_net.parameters()).dtype != torch.float64:
        hw_net = exact_copy(hw_net)
    was_training = hw_net.training
    hw_net.eval()
    correct = disagree = seen = 0
    try:
        with torch.no_grad(), exact_arithmetic():
            for x, y, *_ in loader:
                x = x.to(device)
                top = 2 ** int(getattr(hw_net, "input_bits", 1) or 1) - 1
                if not torch.all((x == torch.round(x)) & (x >= 0) & (x <= top)):
                    # an axon carries one bit; analog input (the "direct"
                    # coding) or levels past the input bits have no
                    # representation on chip
                    return {"sim_val_accuracy": None, "sim_disagreements": None,
                            "sim_reason": f"input is not whole levels 0..{top}; "
                                          f"axons cannot carry it"}
                p_torch = forward_over_time(hw_net, x, flush_steps=flush_steps).argmax(1).cpu()
                for n in range(x.shape[0]):
                    counts, _ = sim.run(x[n], flush_steps)
                    p_sim = int(counts.argmax())
                    correct += p_sim == int(y[n])
                    disagree += p_sim != int(p_torch[n])
                    seen += 1
                if max_clips and seen >= max_clips:
                    break
    except OverflowError as exc:
        return {"sim_val_accuracy": None, "sim_disagreements": None,
                "sim_reason": str(exc)[:300]}
    finally:
        hw_net.train(was_training)
    return {"sim_val_accuracy": correct / max(1, seen), "sim_disagreements": disagree,
            "sim_reason": ""}
