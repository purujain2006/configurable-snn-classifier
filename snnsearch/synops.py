"""Synaptic operations: the hardware-independent energy proxy.

A spiking network does arithmetic only where a spike arrives. So the work it
performs is not a property of the architecture alone. It depends on how often
neurons fire on real data. The accepted measure is:

    SynOps = sum over layers of  (spikes emitted) x (fan-out of that layer)

One SynOp is one accumulate at a downstream neuron. It is the quantity a
neuromorphic chip spends energy on, and it is comparable across platforms
because it counts events rather than seconds or watts.

The fan-out is the one the chip stores, as hs_export writes it: an average
pool or GAP has no neurons and is folded into the next layer, so each neuron
in front of it keeps its own synapses. A signed unit is two neurons, and a
spike reaches both neurons of every signed unit it feeds. An OR pool is a
layer of neurons with one synapse from each input. Padding and zero weights
hold no synapse. So SynOps is the number of synaptic events HiAER-Spike
delivers for the clip.

SynOps count work. Energy in joules also includes memory traffic, data
movement between cores and static power, and on an FPGA static power may
dominate everything else. So the field is called `synops`, and nothing here
calls it energy, because no power was measured.

It is still worth reporting:

  * It is free. Counting spikes during evaluation costs one .sum() per layer.
  * It is the number published SNN work quotes, so it makes results comparable
    against Loihi and SpiNNaker results without owning either.
  * The dense-equivalent ratio below compares it with the multiply-accumulates
    an ANN of the same shape would perform, which is what spiking can save.
    That denominator is the ANN's, one pass with one MAC per weight per
    output (cost.py "macs"): no folded pools, no second neuron per sign. It
    is not the chip's synapse count, which cost.py reports as "connections".
"""

import math

from ._torch import torch, F


class SynOpsCounter:
    """Accumulates the synaptic events each chip layer receives.

    Attach with `attach(net)`, run evaluation, then read `.summary(plan)`.
    Hooks are removed by `detach()`. The context-manager form does both.

    Counting happens under no_grad and adds one multiply and one reduction per
    layer per timestep, so the overhead is negligible next to the forward pass.
    """

    def __init__(self):
        self.ops = {}           # chip layer name -> synaptic events delivered to it
        self.ceiling = {}       # chip layer name -> events if every source fired every call
        self.spikes = {}        # spiking layer name -> spikes emitted
        self.spike_elements = {}  # spiking layer name -> unit-timesteps
        self.calls = {}         # chip layer name -> forward calls
        self.pool_area = {}     # weight layer name -> pre-pool values behind each input value
        self.samples = 0
        # id(tensor) -> (tensor, events behind each element, sources per
        # element, pool area folded into it)
        self._mag = {}
        self._or_prev = {}      # OR pool name -> what it emits next step, as |spikes|
        self._input_problem = None  # why the input has no on-chip form, if it has none
        self._layers = []       # chip layers in order: kind, module, neuron
        self._targets = {}      # weight layer name -> neurons per target unit (2 if signed)
        self._or_signed = {}    # OR pool name -> whether it pools a signed layer
        self._fanout = {}       # weight layer name -> (key, synapses of each input value)
        self._handles = []

    # collection
    def attach(self, net):
        """Hook the WEIGHT-BEARING layers, and the spiking ones for reporting.

        The obvious approach counts spikes at each neuron and multiplies by a
        fan-out looked up from the cost table. That pairs two lists by position,
        and the moment the model gains a layer the table does not describe the
        same way, a fully-connected head, say, the pairing slides and the
        result can exceed the number of connections that exist. It did: a run
        reported 73.2M SynOps against a hard ceiling of 44.9M.

        So fan-out is no longer looked up. Each Conv2d and Linear knows its own
        synapses, and at hook time its input tensor says which of them a spike
        fired: every input value's activity times that value's own
        fan-out, from the layer's nonzero weights (_value_fanout). That is
        exact at a border, where a padded window reaches fewer outputs, and
        after pruning, where a zero weight is no synapse.

        Signed spikes are +1 or -1, and each is one event, so activity is
        counted on magnitudes. A spike from a signed unit is one of its two
        neurons firing, and that neuron has synapses to both neurons of every
        signed unit it feeds, so a layer whose neurons are signed receives
        twice the events. Behind a pool magnitudes are not enough: the average
        of a +1 and a -1 is 0, and so is OR(+) - OR(-) when both fired. So the
        pooling and flatten layers are hooked too, and carry |spikes| beside
        the values.

        An average pool or GAP has no neurons on chip. hs_export folds it into
        the next layer, so each neuron in front of the pool keeps its own
        synapses and every one of its spikes fires them. The pool hooks carry
        the SUM of |spikes| in each window: the next layer's events, and its
        ceiling, grow by the pool's area (4 for 2x2, H*W for GAP). An OR pool
        is a layer of neurons, and its inputs' synapses count as its events.

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
        self._layers, self._targets, self._or_signed, self._fanout = [], {}, {}, {}
        bits = int(getattr(net, "input_bits", 1) or 1)
        if bits > 1:
            self._handles.append(net.register_forward_pre_hook(self._make_input_hook(bits)))
        # Module order is layer order, as hs_export.integer_layers reads it: a
        # weight layer's neurons are the next spiking node, and an OR pool
        # pools the last one.
        current, last_node = None, None
        for name, mod in net.named_modules():
            if isinstance(mod, (_nn.Conv2d, _nn.Linear)):
                current = {"kind": "conv" if isinstance(mod, _nn.Conv2d) else "linear",
                           "module": name, "neuron": None}
                self._layers.append(current)
                self._targets[name] = 1
                self._handles.append(
                    mod.register_forward_hook(self._make_op_hook(name)))
            elif isinstance(mod, HardwareLIFNode):
                if current is not None and current["neuron"] is None:
                    current["neuron"] = name
                    if getattr(mod, "signed", False):
                        self._targets[current["module"]] = 2
                last_node = mod
                self._handles.append(
                    mod.register_forward_hook(self._make_spike_hook(name)))
            elif OrPool is not None and isinstance(mod, OrPool):
                current = {"kind": "orpool", "module": name, "neuron": name}
                self._layers.append(current)
                self._or_signed[name] = bool(getattr(last_node, "signed", False))
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

    def _value_fanout(self, name, mod, x):
        """Synapses of each input value of `mod`: the outputs it reaches
        through a nonzero weight.

        Linear: the nonzero weights in the value's column. Conv2d: the
        gradient of sum(conv(x, w != 0)) with respect to x, which counts each
        input position once per output window that covers it through a
        nonzero tap, so padding and stride are exact. Computed once per input
        shape and weight version, in float64, as (C, H, W) or (in_features,).
        """
        from torch import nn as _nn
        w = mod.weight
        shape = tuple(x.shape[-1:]) if isinstance(mod, _nn.Linear) else tuple(x.shape[-3:])
        key = (shape, w._version, w.data_ptr(), x.device)
        cached = self._fanout.get(name)
        if cached is not None and cached[0] == key:
            return cached[1]
        nz = (w.detach() != 0).to(torch.float64)
        if isinstance(mod, _nn.Linear):
            fan = nz.sum(0)
        else:
            oh, ow = F.conv2d(torch.zeros((1,) + shape, dtype=torch.float64, device=w.device),
                              nz, stride=mod.stride, padding=mod.padding,
                              dilation=mod.dilation, groups=mod.groups).shape[-2:]
            fan = torch.nn.grad.conv2d_input(
                (1,) + shape, nz,
                torch.ones(1, mod.out_channels, oh, ow, dtype=torch.float64, device=w.device),
                stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                groups=mod.groups)[0]
        fan = fan.to(x.device)
        self._fanout[name] = (key, fan)
        return fan

    def _make_op_hook(self, name):
        def hook(mod, inp, out):
            x = inp[0]
            if x is None or x.numel() == 0:
                return
            with torch.no_grad():
                # Every input value's events times its own fan-out. Exact for
                # Linear and Conv2d alike, signed or not, behind any pool:
                # the magnitude already holds every spike folded into it.
                mag, sources, area = self._magnitude(x)
                fan = self._value_fanout(name, mod, x)
                targets = self._targets.get(name, 1)
                events = float((mag.to(torch.float64) * fan).sum().item())
                self.ops[name] = self.ops.get(name, 0.0) + events * targets
                # The ceiling: every source fires on this call. Count the
                # actual tensor shape on EVERY invocation. This includes the
                # batch/time axes in multi-step mode and handles uneven final
                # batches without confusing calls with timesteps. B-bit input
                # is B axons per value, and a pooled value is `area` neurons,
                # each with its own synapses.
                dense = self._dense_macs(mod, x, out)
                self.ceiling[name] = self.ceiling.get(name, 0) + dense * sources * targets
                self.pool_area[name] = area
                self.calls[name] = self.calls.get(name, 0) + 1
        return hook

    def _make_spike_hook(self, name):
        def hook(_mod, _inp, out):
            # `out` is the spike tensor: 0 or 1, or 0, +1 or -1 for a signed
            # unit (two chip neurons, out = spike - spike_neg; only one of the
            # two can fire in a step). Every nonzero is one spike, so sum |out|
            # == spike count, and the firing rate is spikes per unit per step.
            # Reported per layer. The SynOps come from the next layer's input.
            with torch.no_grad():
                self.spikes[name] = self.spikes.get(name, 0) + float(
                    out.abs().sum(dtype=torch.float64).item())
                self.spike_elements[name] = self.spike_elements.get(name, 0) + out.numel()
            # Pooled tensors from before this layer are not what any later
            # weight layer reads in a feed-forward stack. Do not hold them.
            self._mag.clear()
        return hook

    def _magnitude(self, x):
        """(events behind each element of x, sources per element, pool area).

        What a hook recorded for x, else (|x|, 1, 1). The stored reference
        keeps the tensor alive, so its id cannot be reused while the entry
        exists. Each entry is read once.
        """
        entry = self._mag.pop(id(x), None)
        return entry[1:] if entry is not None else (x.abs(), 1, 1)

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
            self._mag[id(x)] = (x, mag, bits, 1)
        return hook

    def _pass_hook(self, mod, inp, out):
        """Average pooling, GAP and flatten: the events behind each output.

        On chip the average pool is not there (hs_export folds it into the
        next layer), so an output value stands for the k*k neurons of its
        window, each with its own synapses. It carries their SUM of |spikes|:
        the layer's own mean times its area, run with forward() rather than
        mod(...), so this hook does not fire again. Flatten only reshapes.
        """
        from torch import nn as _nn
        if not inp or not torch.is_tensor(inp[0]):
            return
        x = inp[0]
        if isinstance(mod, _nn.AvgPool2d):
            ks = mod.kernel_size
            area = ks * ks if isinstance(ks, int) else ks[0] * ks[1]
        elif isinstance(mod, _nn.AdaptiveAvgPool2d):
            # GAP: the whole map behind each value
            area = (x.shape[-2] * x.shape[-1]) // (out.shape[-2] * out.shape[-1])
        else:
            area = 1
        with torch.no_grad():
            mag, sources, folded = self._magnitude(x)
            mag = mod.forward(mag)
            if area > 1:
                mag = mag * area
        self._mag[id(out)] = (out, mag, sources * area, folded * area)

    def _make_or_hooks(self, name):
        """OR pooling: one OR neuron per window and sign.

        Each input value has one synapse to the OR neuron of its window and
        sign, so the pool's events are the |spikes| it reads. It emits one
        spike per window and sign that held a spike: a window where both
        signs fired in one step emits two, so a signed OR pool's output has
        two sources per value. The OR neurons emit one step late
        (neuron.OrPool), and so does this.
        """
        def pre(mod, _inp):
            if mod.prev is None:              # reset_net: a new sample starts
                self._or_prev.pop(name, None)

        def post(mod, inp, out):
            if not inp or not torch.is_tensor(inp[0]):
                return
            x = inp[0]
            k, s = mod.kernel_size, mod.stride
            multi = mod.step_mode == "m"
            with torch.no_grad():
                entry = self._mag.pop(id(x), None)
                src, sources, _folded = entry[1:] if entry is not None else (x.abs(), 1, 1)
                flat = src.flatten(0, 1) if multi else src
                # one synapse per input per window it sits in
                reached = F.avg_pool2d(flat, k, s) * (k * k)
                self.ops[name] = self.ops.get(name, 0.0) + float(
                    reached.sum(dtype=torch.float64).item())
                self.ceiling[name] = self.ceiling.get(name, 0) + reached.numel() * k * k * sources
                self.calls[name] = self.calls.get(name, 0) + 1
                if entry is None and self._or_signed.get(name, False):
                    xs = x.flatten(0, 1) if multi else x
                    now = F.max_pool2d(F.relu(xs), k, s) + F.max_pool2d(F.relu(-xs), k, s)
                    sources = 2
                else:
                    now = F.max_pool2d(flat, k, s)
                last = self._or_prev.get(name)
                if multi:
                    T, N = x.shape[:2]
                    now = now.reshape(T, N, *now.shape[1:])
                    first = torch.zeros_like(now[0]) if last is None else last
                    mag = torch.cat([first.unsqueeze(0), now[:-1]], dim=0)
                    self._or_prev[name] = now[-1]
                else:
                    mag = torch.zeros_like(now) if last is None else last
                    self._or_prev[name] = now
                # the OR neurons' own spikes, for the per-layer report
                self.spikes[name] = self.spikes.get(name, 0) + float(
                    mag.sum(dtype=torch.float64).item())
                self.spike_elements[name] = self.spike_elements.get(name, 0) + mag.numel()
            self._mag[id(out)] = (out, mag, sources, 1)
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
        self.ceiling.clear()
        self.spikes.clear()
        self.spike_elements.clear()
        self.calls.clear()
        self.pool_area.clear()
        self._mag.clear()
        self._or_prev.clear()
        self._input_problem = None
        self.samples = 0

    def add_samples(self, n):
        self.samples += int(n)

    # reporting
    def summary(self, plan=None, cost_rows=None):
        """SynOps per sample, the firing rate, and the dense-equivalent ratio.

        `cost_rows` (cost.count_neurons_and_synapses) is only used for the
        one-pass ANN comparison, the chip's synapse total, and each layer's
        chip neurons and synapses in `layers`. The SynOps total and its
        ceiling come from the weight layers' actual inputs/outputs, so neither
        depends on matching a cost table to modules.

        `firing_rate` and `spikes_per_sample` are over the LIF layers, every
        spike over every unit-step; OR pools are reported in `layers` only.
        """
        if not self.samples:
            return {"synops_per_sample": None, "reason": "no samples counted"}

        synops = sum(self.ops.values())
        lif = [n for n in self.spikes if n not in self._or_signed]
        total_spikes = sum(self.spikes[n] for n in lif)
        total_elements = sum(self.spike_elements.get(n, 0) for n in lif)
        per_layer = {name: {"synops_per_sample": ops / self.samples,
                            "calls": self.calls.get(name, 0)}
                     for name, ops in self.ops.items()}
        for name, sp in self.spikes.items():
            per_layer.setdefault(name, {})["spikes_per_sample"] = sp / self.samples
            elements = self.spike_elements.get(name, 0)
            per_layer[name]["firing_rate"] = sp / elements if elements else None

        # Sum each invocation's ceiling BEFORE normalizing by samples.
        # A cost-table total times max(calls) gets looser as more batches are
        # measured and gets too tight when one call contains multiple timesteps.
        # Checking each layer also catches an impossible layer hidden by silent
        # layers elsewhere in the network. No cost table is needed for safety.
        problems = [self._input_problem] if self._input_problem else []
        for name, ops in self.ops.items():
            ceiling = self.ceiling.get(name)
            if ceiling is None:
                problems.append(f"layer {name!r}: missing dense MAC ceiling")
            elif not math.isfinite(ops) or ops < 0:
                problems.append(f"layer {name!r}: invalid SynOps total {ops}")
            elif ops > ceiling + max(1.0, ceiling) * 1e-6:
                problems.append(
                    f"layer {name!r}: {ops / self.samples:,.0f} SynOps/sample "
                    f"exceeds the dense ceiling of {ceiling / self.samples:,.0f}")
            else:
                continue
            per_layer[name]["synops_per_sample"] = None

        out = {
            "samples": self.samples,
            "spikes_per_sample": total_spikes / self.samples,
            "firing_rate": total_spikes / total_elements if total_elements else None,
            "synops_per_sample": None if problems else synops / self.samples,
            "synops_ceiling_per_sample": sum(self.ceiling.values()) / self.samples,
            "per_layer": per_layer,
            "layers": self._layer_rows(per_layer, cost_rows),
        }

        if cost_rows:
            # An ANN of the same shape performs one multiply-accumulate per
            # weight per output per inference, whether or not anything was
            # active. The chip's synapses are a different count: folded
            # pools, two neurons per signed unit, B axons per input value.
            dense = sum(r.get("macs", 0) for r in cost_rows)
            out["dense_macs_per_inference"] = dense
            out["chip_synapses"] = sum(r.get("connections", 0) for r in cost_rows)
            if dense:
                out["synops_over_dense"] = None if problems else (synops / self.samples) / dense
        if problems:
            out["reason"] = "invalid SynOps measurement: " + "; ".join(problems)
        return out

    def _layer_rows(self, per_layer, cost_rows=None):
        """One row per chip layer, in order: what reaches it and what it fires.

        synops_per_sample   events its synapses receive (from the previous
                            layer's spikes, or the input axons)
        spikes_per_sample   spikes its own neurons emit, and firing_rate per
                            unit and step; None for a layer without neurons
        input_pool_area     neurons folded into each of its input values
        chip_neurons, chip_synapses   from cost_rows, when its chip layers
                            (rows with neurons) match these one for one
        """
        chip = [r for r in (cost_rows or ()) if r.get("neurons")]

        def kind(r):
            name = r.get("layer", "")
            return "conv" if name.startswith("conv") else "orpool" if name.startswith("pool") \
                else "linear"
        paired = (len(chip) == len(self._layers)
                  and all(kind(r) == L["kind"] for r, L in zip(chip, self._layers)))
        rows = []
        for i, L in enumerate(self._layers):
            got = per_layer.get(L["module"], {})
            fired = per_layer.get(L["neuron"], {}) if L["neuron"] else {}
            row = {"layer": i, "kind": L["kind"], "module": L["module"], "neuron": L["neuron"],
                   "input_pool_area": self.pool_area.get(L["module"], 1),
                   "synops_per_sample": got.get("synops_per_sample"),
                   "synops_ceiling_per_sample": self.ceiling.get(L["module"], 0) / self.samples,
                   "spikes_per_sample": fired.get("spikes_per_sample"),
                   "firing_rate": fired.get("firing_rate")}
            if paired:
                row["chip_neurons"] = chip[i]["neurons"]
                row["chip_synapses"] = chip[i]["connections"]
            rows.append(row)
        return rows


def measure_synops(net, loader, device, spec=None, max_batches=None):
    """Count SynOps per sample on the network that will deploy.

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
                    Matches how deployment works: there is a power
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
