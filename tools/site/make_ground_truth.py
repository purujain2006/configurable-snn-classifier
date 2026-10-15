"""Regenerate tests/ground_truth.json from the Python package.

    python tools/site/make_ground_truth.py

The website's interactive builder runs a JavaScript copy of the planner, the
cost model and the feasibility check (docs/assets/snn.js). tests/test_snn_port.js
compares that copy against this file. Regenerate it whenever the Python rules
change, then run the JavaScript test; a failure means the site and the package
disagree.

The configurations below must match makeCfg() in tests/test_snn_port.js.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from snnsearch.config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                              HeadSpec, NeuronSpec)
from snnsearch.cost import count_neurons_and_synapses
from snnsearch.hardware import (check_feasibility, AXON_LIMITS, NEURON_LIMITS,
                                HW_TAU_CHOICES, W_BITS, W_ALPHA, INT16_MAX, W_DELTA)
from snnsearch.planning import InfeasibleConfig
from snnsearch.quantgrid import quantized_threshold_int


def cfg(input=None, encoder=None, downsample=None, head=None):
    return {"input": input or InputSpec(), "encoder": encoder or EncoderSpec(),
            "output": OutputSpec(), "downsample": downsample or DownsampleSpec(),
            "head": head or HeadSpec(), "neuron": NeuronSpec()}


CONFIGS = {
    "seed_d2_k7": cfg(InputSpec(resize_to=64, T=16),
                      EncoderSpec(depth=2, channels=32, kernel_size=7, stride=2),
                      head=HeadSpec(fc_widths="", dropout_rate=0.28)),
    "seed_d3_k7": cfg(InputSpec(resize_to=64, T=16),
                      EncoderSpec(depth=3, channels=32, kernel_size=7, stride=2),
                      head=HeadSpec(fc_widths="", dropout_rate=0.35)),
    "default_fail": cfg(),
    "gap_fc512": cfg(InputSpec(resize_to=64, T=16),
                     EncoderSpec(depth=3, channels=64, kernel_size=5, stride=2),
                     head=HeadSpec(final_reduction="gap", fc_widths="512")),
    "pool_mode": cfg(InputSpec(resize_to=64, T=8),
                     EncoderSpec(depth=2, channels=32, kernel_size=5, stride=1),
                     DownsampleSpec(mode="pool", pool_type="avg")),
    "per_layer": cfg(InputSpec(resize_to=64, T=16),
                     EncoderSpec(layers_json=json.dumps([
                         {"out_channels": 32, "kernel_size": 7, "stride": 2},
                         {"out_channels": 64, "kernel_size": 5, "stride": 1, "pool": True},
                         {"out_channels": 32, "kernel_size": 5, "stride": 2,
                          "tau": 4, "v_threshold": 0.8}])),
                     head=HeadSpec(fc_widths="256")),
    "tiny_maps": cfg(InputSpec(resize_to=32, T=8),
                     EncoderSpec(depth=4, channels=32, kernel_size=5, stride=2),
                     head=HeadSpec(fc_widths="")),
    "collapse": cfg(InputSpec(resize_to=32, T=8),
                    EncoderSpec(depth=5, channels=32, kernel_size=7, stride=2),
                    head=HeadSpec(fc_widths="")),
}


def entry(c):
    feasible, violations = check_feasibility(c["input"], c["encoder"], c["downsample"],
                                             c["head"], c["output"])
    out = {"feasible": feasible, "violations": violations}
    try:
        counts = count_neurons_and_synapses(c)
    except InfeasibleConfig as exc:
        out["infeasible_error"] = str(exc)
        return out
    plan = counts["plan"]
    out.update({
        "rows": counts["rows"], "totals": counts["totals"],
        "input_axons": counts["input_axons"],
        "neuron_updates_per_sample": counts["neuron_updates_per_sample"],
        "blocks": [{"index": b.index, "in_channels": b.in_channels,
                    "out_channels": b.out_channels, "kernel_size": b.kernel_size,
                    "stride": b.stride, "padding": b.padding, "in_hw": list(b.in_hw),
                    "conv_out_hw": list(b.conv_out_hw), "pool": b.pool,
                    "out_hw": list(b.out_hw)} for b in plan.blocks],
        "linears": [{"index": l.index, "in_features": l.in_features,
                     "out_features": l.out_features, "is_classifier": l.is_classifier}
                    for l in plan.linears],
        "fc_in_features": plan.fc_in_features,
    })
    return out


def main():
    gt = {name: entry(c) for name, c in CONFIGS.items()}
    gt["_constants"] = {"W_BITS": W_BITS, "W_ALPHA": W_ALPHA, "INT16_MAX": INT16_MAX,
                        "W_DELTA": W_DELTA, "HW_TAU_CHOICES": HW_TAU_CHOICES,
                        "AXON_LIMITS": AXON_LIMITS, "NEURON_LIMITS": NEURON_LIMITS}
    gt["_threshold_int_examples"] = {
        str(t): quantized_threshold_int(t)
        for t in (1.0, 0.5, 0.25, 0.1, 0.01, 0.001, 3.0517578125e-05, 1e-06)}
    path = os.path.join(ROOT, "tests", "ground_truth.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
