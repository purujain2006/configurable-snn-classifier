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
    """Read the deployed net into a list of integer layers.

    Each entry:
        kind        "conv", "linear", or "orpool" (a layer of OR neurons: one
                    memoryless neuron per pooling window, weight 1 from each
                    input, threshold 0, so it fires when any input fired)
        w           int64 array, the INT16 weight the chip stores
        stride, padding
        pool        avg-pool kernel applied to this layer's INPUT spikes (1 = none);
                    a global average pool counts as a pool the size of the map
        scale       membrane unit relative to the model's W_DELTA grid
        threshold   int64 per output channel, for the `V > threshold` compare
        bias        int64 per output channel in membrane units, added every
                    timestep by bias neurons, or None
        tau, leak   tau the model trained with, and the simpleSim shift
    """
    from .neuron import HardwareLIFNode, OrPool

    mods = list(net.conv_fc)
    out, pending_pool, scale = [], 1, 1
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
                        "threshold": np.zeros(1, dtype=np.int64), "bias": None,
                        "tau": None, "leak": 0, "memoryless": True})
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
            if np.abs(w_int).max() > INT16_MAX:
                raise RuntimeError("weight outside INT16")
            # the pool sits between the previous neuron and this layer's
            # synapses, so it scales THIS layer's membrane
            pool = pending_pool
            scale = pool * pool
            pending_pool = 1
            th = torch.as_tensor(node.hw_threshold_int()).reshape(-1).cpu().numpy().astype(np.int64)
            n_out = w.shape[0]
            if th.size == 1:
                th = np.full(n_out, int(th[0]), dtype=np.int64)
            tau = node.hw_leak() if tau_override is None else int(tau_override)
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
                "bias": bias,
                "tau": tau,
                "leak": leak_shift_for(tau),
            }
            if w.ndim == 4:
                entry["stride"] = tuple(m.stride)
                entry["padding"] = tuple(m.padding)
            out.append(entry)
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


# ---------------------------------------------------------------------------
# the fast simulator
# ---------------------------------------------------------------------------

class IntegerSim:
    """simpleSim's arithmetic, layer by layer, on whole tensors.

    Same update as hs_api's simple_sim.step_run, but it keeps the convolution
    as a convolution instead of unrolling ~260M synapses into a sparse matrix,
    so a clip takes a fraction of a second on a GPU. float64 holds every
    membrane exactly (all values are integers far below 2**53).

    tools/match_simplesim.py --selftest is what earns this the right to stand
    in for simpleSim: it unrolls a small network of the same layer types into
    hs_api's axon/neuron dicts, runs the real simple_sim, and requires every
    spike of every neuron at every timestep to agree.
    """

    def __init__(self, layers, in_shape, device="cpu"):
        for L in layers:
            if L["kind"] != "orpool" and L["leak"] is None:
                raise ValueError(
                    f"tau={L['tau']} is not a power of two, so simpleSim cannot "
                    f"represent it (its leak is v // 2**leak). Pass a power-of-two "
                    f"tau_override.")
        self.layers = layers
        self.in_shape = tuple(in_shape)
        self.device = device
        dt = torch.float64
        self.w = [torch.as_tensor(L["w"], dtype=dt, device=device) if "w" in L else None
                  for L in layers]
        self.th = []
        for L in layers:
            t = torch.as_tensor(L["threshold"], dtype=dt, device=device)
            self.th.append(t.reshape(-1, 1, 1) if L["kind"] in ("conv", "orpool") else t)
        self.div = [float(2 ** L["leak"]) for L in layers]
        # memoryless (ANN) neurons reset to 0 every step, fired or not
        self.memoryless = [bool(L.get("memoryless")) for L in layers]
        self.b = []
        for L in layers:
            if L.get("bias") is None:
                self.b.append(None)
                continue
            b = torch.as_tensor(L["bias"], dtype=dt, device=device)
            self.b.append(b.reshape(-1, 1, 1) if L["kind"] == "conv" else b)

    def _drive(self, idx, spikes_in):
        """Synaptic input to layer idx from the previous layer's 0/1 spikes."""
        L, x = self.layers[idx], spikes_in
        if L["kind"] == "orpool":
            # each OR neuron counts the spikes in its window (weight 1 each)
            return F.avg_pool2d(x, L["pool"]) * (L["pool"] ** 2)
        if L["pool"] > 1:
            # sum, not mean: the 1/pool^2 lives in `scale`, which keeps it integer
            x = F.avg_pool2d(x, L["pool"]) * (L["pool"] ** 2)
        if L["kind"] == "conv":
            return F.conv2d(x, self.w[idx], stride=L["stride"], padding=L["padding"])
        return F.linear(x.flatten(1), self.w[idx])

    @torch.no_grad()
    def steps(self, frames, flush_steps):
        """Run one clip, yielding (membranes, spikes) after every timestep.

        frames: (T, C, H, W) of 0/1. Runs T steps with input and then
        flush_steps with none, so the last frame reaches the output. Fresh
        membranes each call, as the chip is cleared between clips.

        Each step is simpleSim's, in its order:
          phase one   every neuron fires on the membrane carried in (V > threshold),
                      fired neurons reset to 0, then every neuron leaks
                      V -= floor(V / 2**leak)
          phase two   add the input axons, the spikes fired in phase one, and
                      the bias neurons, which fire on every step (flush included)
        """
        T = frames.shape[0]
        zeros = torch.zeros_like(frames[0])
        V = None
        for t in range(T + flush_steps):
            x = (frames[t] if t < T else zeros).to(self.device, torch.float64).unsqueeze(0)
            if V is None:
                V, h = [], x
                for i in range(len(self.layers)):
                    h = self._drive(i, h)
                    V.append(torch.zeros(h.shape, dtype=torch.float64, device=self.device))
            # phase one
            spikes = []
            for i in range(len(self.layers)):
                s = (V[i] > self.th[i]).to(torch.float64)
                V[i] = V[i] * (1.0 - s)
                if self.memoryless[i]:
                    V[i] = torch.zeros_like(V[i])
                else:
                    V[i] = V[i] - torch.floor(V[i] / self.div[i])
                spikes.append(s)
            # phase two
            V[0] = V[0] + self._drive(0, x)
            for i in range(1, len(self.layers)):
                V[i] = V[i] + self._drive(i, spikes[i - 1])
            for i, b in enumerate(self.b):
                if b is not None:
                    V[i] = V[i] + b
            # simpleSim's membrane is a 32-bit integer that wraps (35-bit and
            # saturating on older commits); float64 does neither, so the two
            # agree only while nothing gets near the rail
            peak = max(float(v.abs().max()) for v in V)
            self.max_abs_membrane = max(getattr(self, "max_abs_membrane", 0.0), peak)
            if peak >= MEMBRANE_LIMIT:
                raise OverflowError(
                    f"membrane reached {peak:.0f}, past simpleSim's 32-bit range "
                    f"({MEMBRANE_LIMIT}); IntegerSim no longer matches it")
            yield V, spikes

    def run(self, frames, flush_steps, record=False, record_output=False):
        """One clip. Returns (output spike count per class, extra), where extra is
        the spike count of every layer per step (record=True), the output
        layer's 0/1 spikes per step (record_output=True), or None."""
        counts, per_layer, rows = None, [[] for _ in self.layers], []
        for _V, spikes in self.steps(frames, flush_steps):
            out = spikes[-1].reshape(-1)
            counts = out if counts is None else counts + out
            if record:
                for i, s in enumerate(spikes):
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


def unroll_to_hs_api(layers, in_shape, neuron_model_factory=None):
    """Build (axons, connections, outputs) in CRI_network's format.

    Keys: input axons "A{i}" over the flattened (C, H, W) frame, the same
    order as Christopher's DVS example; neurons "L{layer}_{flat index}".

    neuron_model_factory(threshold:int, leak:int) -> an hs_api LIF_neuron;
    defaults to hs_lif_neuron.

    Only call this on a small network. rank1 unrolls to roughly 260M synapses,
    which hs_api holds as Python tuples.
    """
    neuron_model_factory = neuron_model_factory or hs_lif_neuron
    C, H, W = in_shape
    n_in = C * H * W
    axons = {f"A{i}": [] for i in range(n_in)}
    shapes = []
    shp = (C, H, W)
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

    keys = [[f"L{li}_{j}" for j in range(int(np.prod(s)))] for li, s in enumerate(shapes)]
    connections = {}
    for li, L in enumerate(layers):
        shp = shapes[li]
        per_ch = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        if L["kind"] == "orpool":
            or_model = hs_or_neuron()
            for k in keys[li]:
                connections[k] = ([], or_model)
            continue
        models = {}
        for j, k in enumerate(keys[li]):
            ch = j // per_ch
            th = int(L["threshold"][ch])
            if th not in models:
                models[th] = neuron_model_factory(th, L["leak"])
            connections[k] = ([], models[th])

    # bias neurons, one or more per channel that has a nonzero bias
    bias_model = None
    for li, L in enumerate(layers):
        if L.get("bias") is None:
            continue
        shp = shapes[li]
        per_ch = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        for ch, value in enumerate(L["bias"]):
            if int(value) == 0:
                continue
            targets = keys[li][ch * per_ch:(ch + 1) * per_ch]
            for j, (w, sl) in enumerate(split_bias(value, len(targets))):
                if bias_model is None:
                    bias_model = hs_bias_neuron()
                connections[f"BiasN.L{li}.c{ch}.{j}"] = (
                    [(t, w) for t in targets[sl]], bias_model)

    def add(pre_key, post_key, weight, is_axon):
        if weight == 0:
            return
        if is_axon:
            axons[pre_key].append((post_key, int(weight)))
        else:
            connections[pre_key][0].append((post_key, int(weight)))

    prev_shape = (C, H, W)
    for li, L in enumerate(layers):
        is_axon = li == 0
        pre_keys = [f"A{i}" for i in range(n_in)] if is_axon else keys[li - 1]
        w = L.get("w")
        if L["kind"] == "orpool":
            # every input in a window connects to that window's OR neuron, weight 1
            c_in, h_in, w_in = prev_shape
            p = L["pool"]
            hp, wp = h_in // p, w_in // p
            for ci in range(c_in):
                for y in range(hp * p):
                    for x in range(wp * p):
                        pre = pre_keys[(ci * h_in + y) * w_in + x]
                        add(pre, keys[li][(ci * hp + y // p) * wp + x // p], 1, is_axon)
        elif L["kind"] == "conv":
            c_in, h_in, w_in = prev_shape
            p = L["pool"]
            hp, wp = h_in // p, w_in // p
            c_out, h_out, w_out = shapes[li]
            kh, kw = w.shape[2:]
            ph, pw = L["padding"]
            sh, sw = L["stride"]
            for ci in range(c_in):
                for y in range(h_in):
                    for x in range(w_in):
                        if y // p >= hp or x // p >= wp:
                            continue
                        py, px = y // p, x // p
                        pre = pre_keys[(ci * h_in + y) * w_in + x]
                        for oy in range(h_out):
                            ky = py + ph - oy * sh
                            if not 0 <= ky < kh:
                                continue
                            for ox in range(w_out):
                                kx = px + pw - ox * sw
                                if not 0 <= kx < kw:
                                    continue
                                for co in range(c_out):
                                    post = keys[li][(co * h_out + oy) * w_out + ox]
                                    add(pre, post, w[co, ci, ky, kx], is_axon)
        else:
            if prev_shape and len(prev_shape) == 3 and L["pool"] > 1:
                c_in, h_in, w_in = prev_shape
                p = L["pool"]
                hp, wp = h_in // p, w_in // p
                for ci in range(c_in):
                    for y in range(h_in):
                        for x in range(w_in):
                            if y // p >= hp or x // p >= wp:
                                continue
                            pre = pre_keys[(ci * h_in + y) * w_in + x]
                            flat = (ci * hp + y // p) * wp + x // p
                            for o in range(w.shape[0]):
                                add(pre, keys[li][o], w[o, flat], is_axon)
            else:
                for j, pre in enumerate(pre_keys):
                    for o in range(w.shape[0]):
                        add(pre, keys[li][o], w[o, j], is_axon)
        prev_shape = shapes[li]
    return axons, connections, keys


# ---------------------------------------------------------------------------
# checking a trained network on the simulator
# ---------------------------------------------------------------------------

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
    from .train import forward_over_time
    try:
        layers = integer_layers(hw_net)
        C, H, W = hw_net.plan.blocks[0].in_channels, *hw_net.plan.blocks[0].in_hw
        sim = IntegerSim(layers, (C, H, W), device=device)
    except Exception as exc:
        return {"sim_val_accuracy": None, "sim_disagreements": None,
                "sim_reason": f"{type(exc).__name__}: {exc}"[:300]}
    was_training = hw_net.training
    hw_net.eval()
    correct = disagree = seen = 0
    try:
        with torch.no_grad():
            for x, y, *_ in loader:
                x = x.to(device)
                if not torch.all((x == 0) | (x == 1)):
                    # an axon carries a spike or nothing; analog input (the
                    # "direct" coding) has no representation on chip
                    return {"sim_val_accuracy": None, "sim_disagreements": None,
                            "sim_reason": "input is not 0/1 spikes, axons cannot carry it"}
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
