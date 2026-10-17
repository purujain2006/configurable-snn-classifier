"""Synaptic operations: the hardware-independent energy proxy.

WHAT THIS IS

A spiking network does arithmetic only where a spike arrives. So the work it
performs is not a property of the architecture alone -- it depends on how often
neurons actually fire on real data. The accepted measure is:

    SynOps = sum over layers of  (spikes emitted) x (fan-out of that layer)

One SynOp is one accumulate at a downstream neuron. It is the quantity a
neuromorphic chip spends energy on, and it is comparable across platforms
because it counts events rather than seconds or watts.

WHAT THIS IS NOT

SynOps is not joules. Real energy also includes memory traffic, data movement
between cores, and static power -- and on an FPGA static power may dominate
everything else. Reporting `synops` is honest; calling it "energy" is not,
unless a wattmeter was involved. The naming here is deliberate.

WHY IT IS WORTH REPORTING ANYWAY

  * It is free. Counting spikes during evaluation costs one .sum() per layer.
  * It is the number published SNN work quotes, so it makes results comparable
    against Loihi and SpiNNaker results without owning either.
  * The dense-equivalent ratio below states the case for spiking at all: how
    many multiply-accumulates an ANN of the same shape would have performed.
"""

import math

from ._torch import torch, F


class SynOpsCounter:
    """Accumulates input-driven operations at each weight layer.

    Attach with `attach(net)`, run evaluation, then read `.summary(plan)`.
    Hooks are removed by `detach()`; the context-manager form does both.

    Counting happens under no_grad and adds one reduction per layer per
    timestep, so the overhead is negligible next to the forward pass itself.
    """

    def __init__(self):
        self.ops = {}           # weight layer name -> accumulate operations
        self.dense_macs = {}    # weight layer name -> dense MACs over all calls
        self.spikes = {}        # spiking layer name -> spikes emitted
        self.spike_elements = {}  # spiking layer name -> neuron-timesteps
        self.calls = {}         # weight layer name -> forward calls
        self.samples = 0
        # id(tensor) -> (tensor, events behind each element, sources per element)
        self._mag = {}
        self._or_prev = {}      # OR pool name -> what it emits next step, as |spikes|
        self._input_problem = None  # why the input has no on-chip form, if it has none
        self._handles = []

    # ---- collection -----------------------------------------------------
    def attach(self, net):
        """Hook the WEIGHT-BEARING layers, and the spiking ones for reporting.

        The obvious approach counts spikes at each neuron and multiplies by a
        fan-out looked up from the cost table. That pairs two lists by position,
        and the moment the model gains a layer the table does not describe the
        same way -- a fully-connected head, say -- the pairing slides and the
        result can exceed the number of connections that exist. It did: a run
        reported 73.2M SynOps against a hard ceiling of 44.9M.

        So fan-out is no longer looked up. Each Conv2d and Linear knows its own
        arithmetic, and at hook time its input tensor says how much of that
        arithmetic a spike actually triggered. Nothing is paired, nothing can
        slide, and the number cannot exceed the layer's own dense cost.

        Signed spikes are +1 or -1, and each is one event, so activity is
        counted on magnitudes. Behind a pool that is not enough: the average
        of a +1 and a -1 is 0, and so is OR(+) - OR(-) when both fired. So the
        pooling and flatten layers are hooked too, and carry |spikes| beside
        the values: a weight layer counts what the same network would with
        every spike positive. For 0/1 spikes that is the tensor itself.

        B-bit input (net.input_bits > 1) arrives as levels 0 to 2**B - 1, but
        the chip carries each value on B axons, one per bit (hs_export). So
        the network's input counts one event per set bit, and the layer that
        reads it has B synapses per value: B times the dense ceiling.
        """
        from .neuron import HardwareLIFNode, OrPool
        try:
            from torch import nn as _nn
        except Exception:                     # pragma: no cover - torch absent
            return self
        self.detach()
        bits = int(getattr(net, "input_bits", 1) or 1)
        if bits > 1:
            self._handles.append(net.register_forward_pre_hook(self._make_input_hook(bits)))
        for name, mod in net.named_modules():
            if isinstance(mod, (_nn.Conv2d, _nn.Linear)):
                self._handles.append(
                    mod.register_forward_hook(self._make_op_hook(name)))
            elif isinstance(mod, HardwareLIFNode):
                self._handles.append(
                    mod.register_forward_hook(self._make_spike_hook(name)))
            elif OrPool is not None and isinstance(mod, OrPool):
                pre, post = self._make_or_hooks(name)
                self._handles.append(mod.register_forward_pre_hook(pre))
                self._handles.append(mod.register_forward_hook(post))
            elif isinstance(mod, (_nn.AvgPool2d, _nn.AdaptiveAvgPool2d, _nn.Flatten)):
                self._handles.append(mod.register_forward_hook(self._pass_hook))
        return self

    @staticmethod
    def _dense_macs(mod, inp, out):
        """Multiply-accumulates this layer performs on a fully dense input."""
        from torch import nn as _nn
        if isinstance(mod, _nn.Linear):
            return out.numel() * mod.in_features
        # Conv2d: one MAC per output element per weight in the kernel window.
        kh, kw = mod.kernel_size
        return out.numel() * (mod.in_channels // mod.groups) * kh * kw

    def _make_op_hook(self, name):
        def hook(mod, inp, out):
            x = inp[0]
            if x is None or x.numel() == 0:
                return
            with torch.no_grad():
                dense = self._dense_macs(mod, x, out)
                # The share of the dense arithmetic that a spike drove. Exact
                # for Linear with spike input, signed or not; Conv2d uses a
                # mean-activity estimate (padding and stride can vary a spike's
                # fan-out). Fractional inputs, including average pooling,
                # retain their activity rather than being rounded to binary
                # spikes.
                mag, sources = self._magnitude(x)
                active = float(mag.sum(dtype=torch.float64).item()) / float(x.numel())
                self.ops[name] = self.ops.get(name, 0.0) + active * dense
                # Count the actual tensor shape on EVERY invocation. This
                # includes the batch/time axes in multi-step mode and handles
                # uneven final batches without confusing calls with timesteps.
                # B-bit input is B axons per value, each with its own synapse.
                self.dense_macs[name] = self.dense_macs.get(name, 0) + dense * sources
                self.calls[name] = self.calls.get(name, 0) + 1
        return hook

    def _make_spike_hook(self, name):
        def hook(_mod, _inp, out):
            # `out` is the spike tensor: 0 or 1, or 0, +1 or -1 for a signed
            # unit (two chip neurons, out = spike - spike_neg). Every nonzero
            # is one spike, so sum |out| == spike count, and the firing rate
            # is spikes per unit per step. Reported for the firing rate; it no
            # longer feeds the SynOps total.
            with torch.no_grad():
                self.spikes[name] = self.spikes.get(name, 0) + float(
                    out.abs().sum(dtype=torch.float64).item())
                self.spike_elements[name] = self.spike_elements.get(name, 0) + out.numel()
            # Pooled tensors from before this layer are not what any later
            # weight layer reads in a feed-forward stack; do not hold them.
            self._mag.clear()
        return hook

    def _magnitude(self, x):
        """(events behind each element of x, sources per element).

        What a hook recorded for x, else (|x|, 1). The stored reference keeps
        the tensor alive, so its id cannot be reused while the entry exists.
        Each entry is read once.
        """
        entry = self._mag.pop(id(x), None)
        return (entry[1], entry[2]) if entry is not None else (x.abs(), 1)

    def _make_input_hook(self, bits):
        """The network's B-bit input as the chip sends it: a spike per set bit."""
        top = 2 ** bits - 1

        def hook(_mod, args):
            if not args or not torch.is_tensor(args[0]):
                return
            x = args[0]
            with torch.no_grad():
                # hs_export refuses the same input: an axon carries one bit
                if not bool(torch.all((x == torch.round(x)) & (x >= 0) & (x <= top))):
                    self._input_problem = (f"input is not whole levels 0..{top}; "
                                           f"{bits} axons per value cannot carry it")
                    return
                v = x.to(torch.int64)
                mag = sum((v >> k) & 1 for k in range(bits)).to(x.dtype)
            self._mag[id(x)] = (x, mag, bits)
        return hook

    def _pass_hook(self, mod, inp, out):
        """Average pooling, GAP and flatten: run the same layer on |spikes|.

        forward() rather than mod(...), so this hook does not fire again.
        """
        if not inp or not torch.is_tensor(inp[0]):
            return
        with torch.no_grad():
            mag, sources = self._magnitude(inp[0])
            mag = mod.forward(mag)
        self._mag[id(out)] = (out, mag, sources)

    def _make_or_hooks(self, name):
        """OR pooling on |spikes|: one event per window that held a spike.

        On chip a signed OR pool is one OR neuron per sign, so a window where
        both signs fired in one step is two events; this counts one, as the
        network with every spike positive would, and keeps every input's
        activity at most 1 so the dense ceiling stays a hard bound. The OR
        neurons emit one step late (neuron.OrPool), and so does this.
        """
        def pre(mod, _inp):
            if mod.prev is None:              # reset_net: a new sample starts
                self._or_prev.pop(name, None)

        def post(mod, inp, out):
            if not inp or not torch.is_tensor(inp[0]):
                return
            k, s = mod.kernel_size, mod.stride
            with torch.no_grad():
                src, sources = self._magnitude(inp[0])
                last = self._or_prev.get(name)
                if mod.step_mode == "m":
                    T, N = src.shape[:2]
                    now = F.max_pool2d(src.flatten(0, 1), k, s)
                    now = now.reshape(T, N, *now.shape[1:])
                    first = torch.zeros_like(now[0]) if last is None else last
                    mag = torch.cat([first.unsqueeze(0), now[:-1]], dim=0)
                    self._or_prev[name] = now[-1]
                else:
                    now = F.max_pool2d(src, k, s)
                    mag = torch.zeros_like(now) if last is None else last
                    self._or_prev[name] = now
            self._mag[id(out)] = (out, mag, sources)
        return pre, post

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._mag.clear()
        self._or_prev.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()
        return False

    def reset(self):
        self.ops.clear()
        self.dense_macs.clear()
        self.spikes.clear()
        self.spike_elements.clear()
        self.calls.clear()
        self._mag.clear()
        self._or_prev.clear()
        self._input_problem = None
        self.samples = 0

    def add_samples(self, n):
        self.samples += int(n)

    # ---- reporting ------------------------------------------------------
    def summary(self, plan=None, cost_rows=None):
        """SynOps per sample, the firing rate, and the dense-equivalent ratio.

        `cost_rows` is only used for the one-pass ANN comparison. Both the
        SynOps total and its dense ceiling come from the weight layers' actual
        inputs/outputs, so neither depends on matching a cost table to modules.
        """
        if not self.samples:
            return {"synops_per_sample": None, "reason": "no samples counted"}

        synops = sum(self.ops.values())
        total_spikes = sum(self.spikes.values())
        total_elements = sum(self.spike_elements.values())
        per_layer = {name: {"synops_per_sample": ops / self.samples,
                            "calls": self.calls.get(name, 0)}
                     for name, ops in self.ops.items()}
        for name, sp in self.spikes.items():
            per_layer.setdefault(name, {})["spikes_per_sample"] = sp / self.samples
            elements = self.spike_elements.get(name, 0)
            per_layer[name]["firing_rate"] = sp / elements if elements else None

        # Sum each invocation's dense work BEFORE normalizing by samples.
        # A cost-table total times max(calls) gets looser as more batches are
        # measured and gets too tight when one call contains multiple timesteps.
        # Checking each layer also catches an impossible layer hidden by silent
        # layers elsewhere in the network. No cost table is needed for safety.
        problems = [self._input_problem] if self._input_problem else []
        for name, ops in self.ops.items():
            dense = self.dense_macs.get(name)
            if dense is None:
                problems.append(f"layer {name!r}: missing dense MAC ceiling")
            elif not math.isfinite(ops) or ops < 0:
                problems.append(f"layer {name!r}: invalid SynOps total {ops}")
            elif ops > dense + max(1.0, dense) * 1e-6:
                problems.append(
                    f"layer {name!r}: {ops / self.samples:,.0f} SynOps/sample "
                    f"exceeds the dense ceiling of {dense / self.samples:,.0f}")
            else:
                continue
            per_layer[name]["synops_per_sample"] = None

        ceiling = sum(self.dense_macs.values()) / self.samples

        out = {
            "samples": self.samples,
            "spikes_per_sample": total_spikes / self.samples,
            "firing_rate": total_spikes / total_elements if total_elements else None,
            "synops_per_sample": None if problems else synops / self.samples,
            "synops_ceiling_per_sample": ceiling,
            "per_layer": per_layer,
        }

        # An ANN of the same shape performs one multiply-accumulate per
        # connection per inference, whether or not anything was active.
        if cost_rows:
            dense = sum(r.get("connections", 0) for r in cost_rows)
            out["dense_macs_per_inference"] = dense
            if dense:
                out["synops_over_dense"] = None if problems else (synops / self.samples) / dense
        if problems:
            out["reason"] = "invalid SynOps measurement: " + "; ".join(problems)
        return out


def measure_synops(net, loader, device, spec=None, max_batches=None):
    """Count SynOps per sample on the network that will actually deploy.

    Measured after conversion rather than during training, because the float
    network and the converted one do not fire identically: folding shifts every
    threshold, and quantization moves weights onto the INT16 grid. The number
    worth reporting is the one the chip would produce.

    `max_batches` bounds the cost during a search. The spike rate settles within
    a few hundred samples, so counting the whole validation set every trial buys
    precision nobody reads.

    Returns the summary dict, or a dict carrying `reason` when it could not
    measure. Never raises: a missing SynOps figure should not lose a training
    run that otherwise succeeded.
    """
    from .train import forward_over_time

    try:
        from .cost import count_neurons_and_synapses
        cost_rows = count_neurons_and_synapses(spec)["rows"] if spec else None
    except Exception as exc:                  # a plan that will not cost is not fatal
        cost_rows = None
        if spec is not None:
            print(f"  synops: dense comparison unavailable ({type(exc).__name__})")

    counter = SynOpsCounter()
    was_training = net.training
    net.eval()
    try:
        with counter.attach(net):
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if max_batches is not None and i >= max_batches:
                        break
                    x = batch[0].to(device)
                    forward_over_time(net, x)
                    counter.add_samples(x.shape[0])
    except Exception as exc:
        return {"synops_per_sample": None, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        net.train(was_training)

    return counter.summary(cost_rows=cost_rows)


def score_with_energy(accuracy, synops, mode="accuracy",
                      synops_budget=None, synops_reference=None, weight=0.0):
    """Combine accuracy and SynOps into whatever the search should maximize.

    Four modes, because the right one depends on a decision the user has to
    make rather than one this code can make for them:

      "accuracy"    ignore SynOps entirely. The default, and what every
                    previously recorded run used.

      "constrained" maximize accuracy, but score 0 above `synops_budget`.
                    Matches how deployment actually works: there is a power
                    envelope and you either fit or you do not.

      "weighted"    accuracy - weight * (synops / synops_reference). Requires
                    choosing `weight`, which encodes how much accuracy a
                    halving of energy is worth. Nothing here can guess it.

      "pareto"      return both and let a multi-objective sampler handle it.
                    Preferred when the trade-off is not yet decided, since it
                    defers the choice until the front can be looked at.
    """
    if synops is None or mode == "accuracy":
        return accuracy
    if mode == "constrained":
        if synops_budget is None:
            raise ValueError("mode='constrained' needs synops_budget")
        return accuracy if synops <= synops_budget else 0.0
    if mode == "weighted":
        ref = synops_reference or synops or 1.0
        return accuracy - weight * (synops / ref)
    if mode == "pareto":
        return (accuracy, -synops)          # both maximized
    raise ValueError(f"unknown energy mode {mode!r}; expected accuracy, "
                     "constrained, weighted or pareto")
