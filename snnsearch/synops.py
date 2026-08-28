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

from ._torch import _HAS_TORCH, torch


class SynOpsCounter:
    """Accumulates spike counts per spiking layer during a forward pass.

    Attach with `attach(net)`, run evaluation, then read `.summary(plan)`.
    Hooks are removed by `detach()`; the context-manager form does both.

    Counting happens under no_grad and adds one reduction per layer per
    timestep, so the overhead is negligible next to the forward pass itself.
    """

    def __init__(self):
        self.ops = {}           # weight layer name -> accumulate operations
        self.spikes = {}        # spiking layer name -> spikes emitted
        self.calls = {}         # weight layer name -> forward calls
        self.samples = 0
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
        """
        from .neuron import HardwareLIFNode
        try:
            from torch import nn as _nn
        except Exception:                     # pragma: no cover - torch absent
            return self
        self.detach()
        for name, mod in net.named_modules():
            if isinstance(mod, (_nn.Conv2d, _nn.Linear)):
                self._handles.append(
                    mod.register_forward_hook(self._make_op_hook(name)))
            elif isinstance(mod, HardwareLIFNode):
                self._handles.append(
                    mod.register_forward_hook(self._make_spike_hook(name)))
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
                # for binary input; for analog input (the `direct` coding at the
                # first layer) it degrades to the mean activation, which is the
                # conventional treatment.
                active = float(x.sum().item()) / float(x.numel())
                self.ops[name] = self.ops.get(name, 0.0) + active * dense
                self.calls[name] = self.calls.get(name, 0) + 1
        return hook

    def _make_spike_hook(self, name):
        def hook(_mod, _inp, out):
            # `out` is the spike tensor: exactly 0 or 1, so sum == spike count.
            # Reported for the firing rate; it no longer feeds the SynOps total.
            with torch.no_grad():
                self.spikes[name] = self.spikes.get(name, 0) + float(out.sum().item())
        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()
        return False

    def reset(self):
        self.ops.clear()
        self.spikes.clear()
        self.calls.clear()
        self.samples = 0

    def add_samples(self, n):
        self.samples += int(n)

    # ---- reporting ------------------------------------------------------
    def summary(self, plan=None, cost_rows=None):
        """SynOps per sample, the firing rate, and the dense-equivalent ratio.

        `cost_rows` is only used for the dense comparison and the ceiling check
        now. The SynOps total comes from what the weight layers were actually
        asked to do, so it does not depend on the cost table being lined up
        with the module list.
        """
        if not self.samples:
            return {"synops_per_sample": None, "reason": "no samples counted"}

        synops = sum(self.ops.values())
        total_spikes = sum(self.spikes.values())
        per_layer = {name: {"synops_per_sample": ops / self.samples,
                            "calls": self.calls.get(name, 0)}
                     for name, ops in self.ops.items()}
        for name, sp in self.spikes.items():
            per_layer.setdefault(name, {})["spikes_per_sample"] = sp / self.samples

        out = {
            "samples": self.samples,
            "spikes_per_sample": total_spikes / self.samples,
            "synops_per_sample": synops / self.samples,
            "per_layer": per_layer,
        }

        # An ANN of the same shape performs one multiply-accumulate per
        # connection per inference, whether or not anything was active.
        if cost_rows:
            dense = sum(r.get("connections", 0) for r in cost_rows)
            out["dense_macs_per_inference"] = dense
            if dense:
                out["synops_over_dense"] = (synops / self.samples) / dense
                # Every weight layer is driven once per pass, and no pass can
                # accumulate more than the layer's dense cost. So the total
                # cannot exceed dense x passes. Exceeding it is not a surprising
                # measurement, it is a broken one, and a broken number that
                # reaches a leaderboard gets quoted.
                passes = max(self.calls.values()) if self.calls else 1
                ceiling = dense * passes
                if synops / self.samples > ceiling * 1.001:
                    out["synops_per_sample"] = None
                    out["reason"] = (
                        f"impossible: {synops / self.samples:,.0f} SynOps/sample "
                        f"exceeds the ceiling of {ceiling:,.0f} "
                        f"({dense:,} connections x {passes} passes)")
        return out


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
