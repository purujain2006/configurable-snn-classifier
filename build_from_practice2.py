"""Split Practice2.py into the snnsearch package, by line range.

Copying 3,400 lines by hand invites transcription errors that are invisible
until something silently computes the wrong number. This reads the original,
slices it at boundaries verified against its AST, and writes each module with
its own imports. Re-running it regenerates the extracted modules exactly.

Modules written here hold ONLY moved code. Anything new -- encoders, dataset
plugins, SynOps, the report -- is hand-written and never touched by this script.

    python build_from_practice2.py /path/to/Practice2.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "snnsearch")

BANNER = '''"""{doc}

Moved verbatim from Practice2.py lines {span} by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""
'''

# Text appended after the moved code. Practice2.py could leave a name
# undefined when torch was absent, because every reference to it was inside the
# same guard. Once split, a dependent module imports the name at load time, so
# an absent name surfaces as "cannot import name '_ste'" -- which reads like a
# bug in this package rather than a missing dependency. Binding it to None keeps
# the import working and lets _require_torch() give the real message.
TAIL = {
    "neuron.py": '''

if not _HAS_TORCH:
    # Same reason as quantgrid: model.py imports these by name at load time.
    TdBatchNorm2d = ConvBNFoldQuant = None
    enable_weight_fake_quant = bake_weight_fake_quant = None
''',
    "quantgrid.py": '''

if not _HAS_TORCH:
    # See the note in build_from_practice2.py: these must exist as names even
    # without torch, so that importing a dependent module gives a clear error
    # from _require_torch() rather than an ImportError about a private symbol.
    _ste = None
''',
}

# module -> (docstring, imports, [(first, last), ...])
# Line numbers are 1-based inclusive and were read off the AST, not guessed.
PLAN = {
    "config.py": (
        "Configuration dataclasses and the key=value command line surface.\n\n"
        "Imports no deep-learning library, so `summary` runs on a bare Python.",
        "import argparse\n"
        "from dataclasses import asdict, dataclass, field, fields\n"
        "from typing import Optional\n",
        [(98, 356)],
    ),
    "planning.py": (
        "Layer-shape planning: one source of truth for every derived size.\n\n"
        "Pure arithmetic, no torch.",
        "from dataclasses import dataclass, field\n"
        "from typing import Optional\n\n"
        "from .config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,\n"
        "                     HeadSpec, effective_hw, parse_fc_widths, resolve_conv_layers)\n",
        [(367, 509)],
    ),
    "hardware.py": (
        "HiAER-Spike limits and the INT16 deployment grid.\n\n"
        "The constants are read off hs_api rather than chosen. The feasibility\n"
        "check is arithmetic on a plan, so it needs no torch.",
        "import math\n\n"
        "from .config import InputSpec, EncoderSpec, OutputSpec, DownsampleSpec, HeadSpec\n"
        "from .planning import plan_network, InfeasibleConfig\n",
        [(516, 590), (770, 782)],
    ),
    "cost.py": (
        "What a configuration occupies: neurons, connections, parameters.\n\n"
        "Counted from the plan, so it agrees with the feasibility check by\n"
        "construction. SynOps live in synops.py, which needs a trained network.",
        "from .config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,\n"
        "                     HeadSpec, NeuronSpec, TrainSpec, effective_hw,\n"
        "                     parse_fc_widths, resolve_conv_layers)\n"
        "from .planning import plan_network, plan_encoder, conv_out_size, InfeasibleConfig\n"
        "from .hardware import (check_feasibility, AXON_LIMITS, NEURON_LIMITS, HW_TAU_CHOICES,\n"
        "                       W_BITS, W_ALPHA, INT16_MAX, W_DELTA)\n",
        [(603, 742)],
    ),
    "quantgrid.py": (
        "Differentiable stand-ins for the converter's integer arithmetic.\n\n"
        "Every function here mirrors something hs_api does at conversion time, so\n"
        "training sees the numbers the chip will use.",
        "from .hardware import (W_BITS, W_ALPHA, INT16_MAX, W_DELTA,\n"
        "                       HW_TAU_CHOICES, HW_TAU_MIN, HW_TAU_MAX)\n"
        "from ._torch import _HAS_TORCH, _require_torch, torch, nn, layer\n",
        [(783, 854)],
    ),
    "neuron.py": (
        "The neuron HiAER-Spike implements, and the layers around it.",
        "import math\n\n"
        "from .config import NeuronSpec\n"
        "from .hardware import W_ALPHA, W_DELTA, HW_TAU_CHOICES, HW_TAU_MIN, HW_TAU_MAX\n"
        "from .quantgrid import (_ste, fake_quantize_weight, fake_quantize_threshold,\n"
        "                        quantized_threshold_int, fold_bias_band, constrain_fold_bias)\n"
        "from ._torch import (_HAS_TORCH, _HAS_HS_API, _require_torch, _no_grad,\n"
        "                     torch, nn, F, layer, functional, surrogate, neuron,\n"
        "                     Custom_LIFNode, Custom_IFNode)\n",
        [(861, 1294)],
    ),
    "model.py": (
        "The network, built straight from the plan so it cannot drift from it.",
        "from .config import InputSpec, EncoderSpec, OutputSpec, DownsampleSpec, HeadSpec, NeuronSpec\n"
        "from .planning import plan_network, NetPlan\n"
        "from .hardware import W_ALPHA, W_DELTA, INT16_MAX\n"
        "from .quantgrid import (_ste, fake_quantize_weight, fake_quantize_threshold,\n"
        "                        quantized_threshold_int, fold_bias_band, constrain_fold_bias,\n"
        "                        weight_clip_fraction)\n"
        "from .neuron import (build_neuron, HardwareLIFNode, TdBatchNorm2d,\n"
        "                     ConvBNFoldQuant, enable_weight_fake_quant)\n"
        "from ._torch import (_HAS_TORCH, _HAS_HS_API, _require_torch, _no_grad,\n"
        "                     torch, nn, F, layer, functional, Custom_LIFNode, Custom_IFNode)\n",
        [(1297, 1477)],
    ),
    "folding.py": (
        "Removing batch normalization by folding it into weights and thresholds.",
        "from .hardware import W_ALPHA, W_DELTA, INT16_MAX\n"
        "from .quantgrid import fake_quantize_weight, quantized_threshold_int, fold_bias_band\n"
        "from .neuron import HardwareLIFNode\n"
        "from ._torch import _HAS_TORCH, _require_torch, _no_grad, torch, nn, layer, functional\n\n"
"# verify_fold needs the time loop from train.py. Imported inside the function\n"
"# rather than at module scope, because train.py imports this module.\n",
        [(1503, 1707)],
    ),
    "quantize.py": (
        "Quantization-aware training, the deployment audit, and the export.",
        "import json\nimport os\n\n"
        "from .hardware import W_ALPHA, W_DELTA, INT16_MAX\n"
        "from .quantgrid import fake_quantize_weight, weight_clip_fraction\n"
        "from .neuron import HardwareLIFNode\n"
        "from .folding import fold_bn, verify_fold, fold_bias_report\n"
        "from ._torch import _HAS_TORCH, _require_torch, _no_grad, torch, nn, layer\n",
        [(1725, 1921)],
    ),
    "train.py": (
        "Optimizer construction, the training loop, and the pipeline flush.",
        "import math\nfrom copy import deepcopy\n\n"
        "from .config import TrainSpec\n"
        "from .neuron import HardwareLIFNode\n"
        "from .quantize import deploy_and_measure\n"
        "from .synops import SynOpsCounter\n"
        "from ._torch import _HAS_TORCH, _require_torch, torch, nn, functional\n",
        [(1990, 2241)],
    ),
    "spaces.py": (
        "The searchable space, as a picklable module-level object.\n\n"
        "Ray checkpoints the searcher by pickling it, so this cannot be a closure.",
        "from .config import (InputSpec, ConvLayerSpec, EncoderSpec, OutputSpec,\n"
        "                     DownsampleSpec, HeadSpec, NeuronSpec, TrainSpec,\n"
        "                     parse_fc_widths)\n"
        "from .hardware import HW_TAU_CHOICES\n",
        [(2366, 2578)],
    ),
    "results.py": (
        "Incremental result writing, so an interrupted search keeps its work.",
        "import csv\nimport json\nimport os\nimport shutil\nimport threading\nimport time\n"
        "from datetime import datetime\n\n"
        "from .planning import InfeasibleConfig\n"
        "from .hardware import check_feasibility\n"
        "from .cost import count_neurons_and_synapses\n"
        "from .spaces import config_to_specs\n",
        [(2633, 2776)],
    ),
}


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().split("\n")


def decorator_starts(path):
    """body line -> first decorator line, for everything in the file.

    A span that begins on a decorated `class` or `def` would otherwise leave
    its @dataclass behind, and the class silently loses its generated __init__.
    That failure is quiet: the class still imports and only breaks when
    constructed. So spans are snapped backwards automatically.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.decorator_list:
            out[node.lineno] = min(d.lineno for d in node.decorator_list)
    return out


def snap(spans, decs):
    fixed = []
    for a, b in spans:
        if a in decs and decs[a] < a:
            print(f"    span {a}-{b} snapped back to {decs[a]} to keep its decorator")
            a = decs[a]
        fixed.append((a, b))
    return fixed


def write_module(name, doc, imports, spans, lines):
    body = []
    for a, b in spans:
        body.append("\n".join(lines[a - 1:b]).rstrip())
    span_txt = ", ".join(f"{a}-{b}" for a, b in spans)
    out = BANNER.format(doc=doc, span=span_txt) + "\n" + imports + "\n\n" + "\n\n\n".join(body) + "\n"
    out += TAIL.get(name, "")
    dest = os.path.join(PKG, name)
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    n = out.count("\n")
    print(f"  {name:<16} {span_txt:<20} {n:>5} lines")
    return n


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(HERE), "configurable-snn-classifier", "Practice2.py")
    if not os.path.isfile(src):
        sys.exit(f"source not found: {src}")
    lines = read(src)
    decs = decorator_starts(src)
    os.makedirs(PKG, exist_ok=True)
    print(f"source: {src} ({len(lines)} lines)\n")
    total = 0
    for name, (doc, imports, spans) in PLAN.items():
        total += write_module(name, doc, imports, snap(spans, decs), lines)
    print(f"\n  {'total':<16} {'':<20} {total:>5} lines extracted")
    print("\nHand-written modules are not touched: _torch.py, synops.py, encoders.py,")
    print("data/, search.py, report.py, analysis.py, cli.py")


if __name__ == "__main__":
    main()
