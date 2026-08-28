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
        self.spikes = {}        # layer name -> total spikes emitted
        self.calls = {}         # layer name -> forward calls (timesteps x batches)
        self.samples = 0
        self._handles = []

    # ---- collection -----------------------------------------------------
    def attach(self, net):
        """Hook every spiking layer. Returns self so it can be chained."""
        from .neuron import HardwareLIFNode
        self.detach()
        for name, mod in net.named_modules():
            if isinstance(mod, HardwareLIFNode):
                self._handles.append(
                    mod.register_forward_hook(self._make_hook(name)))
        return self

    def _make_hook(self, name):
        def hook(_mod, _inp, out):
            # `out` is the spike tensor: exactly 0 or 1, so sum == spike count
            with torch.no_grad():
                self.spikes[name] = self.spikes.get(name, 0) + float(out.sum().item())
                self.calls[name] = self.calls.get(name, 0) + 1
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
        self.spikes.clear()
        self.calls.clear()
        self.samples = 0

    def add_samples(self, n):
        self.samples += int(n)

    # ---- reporting ------------------------------------------------------
    def summary(self, plan=None, cost_rows=None):
        """SynOps per sample, plus the dense-equivalent comparison.

        `cost_rows` is the `rows` list from count_neurons_and_synapses, which
        already carries per-layer connection counts. Fan-out per spiking layer
        is taken from the NEXT layer's connection count divided by its input
        neuron count, because that is what one spike actually drives.
        """
        if not self.samples:
            return {"synops_per_sample": None, "reason": "no samples counted"}

        total_spikes = sum(self.spikes.values())
        per_layer = {}
        synops = 0.0

        fanouts = self._fanouts(cost_rows)
        for name, spikes in self.spikes.items():
            fan = fanouts.get(name, 0)
            ops = spikes * fan
            synops += ops
            per_layer[name] = {
                "spikes_per_sample": spikes / self.samples,
                "fan_out": fan,
                "synops_per_sample": ops / self.samples,
            }

        out = {
            "samples": self.samples,
            "spikes_per_sample": total_spikes / self.samples,
            "synops_per_sample": synops / self.samples,
            "per_layer": per_layer,
        }

        # An ANN of the same shape performs one multiply-accumulate per
        # connection per inference, whether or not anything was active.
        if cost_rows is not None:
            dense = sum(r.get("connections", 0) for r in cost_rows)
            out["dense_macs_per_inference"] = dense
            if dense:
                out["synops_over_dense"] = (synops / self.samples) / dense
        return out

    def _fanouts(self, cost_rows):
        """Spikes from layer i drive the connections of layer i+1.

        Without the cost rows there is nothing to multiply by, so fan-out is
        reported as zero and only raw spike counts are meaningful.
        """
        if not cost_rows:
            return {}
        conn = [r.get("connections", 0) for r in cost_rows if r.get("neurons", 0) > 0]
        neur = [r.get("neurons", 0) for r in cost_rows if r.get("neurons", 0) > 0]
        fan = []
        for i in range(len(conn)):
            nxt = conn[i + 1] if i + 1 < len(conn) else 0
            fan.append(nxt / neur[i] if neur[i] else 0)
        names = list(self.spikes.keys())
        return {n: (fan[i] if i < len(fan) else 0) for i, n in enumerate(names)}


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
