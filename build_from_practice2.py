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
# Exact-string replacements applied after extraction. Practice2.py is one
# module, so it can call anything from anywhere; split into packages, some of
# those calls point backwards and would import in a circle. The fix is a local
# import at the point of use, which cannot be expressed as a module header --
# hence a patch. Each entry raises if its `old` text is absent, so a change in
# Practice2.py surfaces as a loud failure here rather than a silent no-op.
PATCHES = {
    "model.py": [
        ("        enable_weight_fake_quant(self)\n",
         "        # local import: quantize -> folding -> model, so a module-level\n"
         "        # import here would close the circle.\n"
         "        from .quantize import enable_weight_fake_quant\n"
         "        enable_weight_fake_quant(self)\n"),
        ("        bake_weight_fake_quant(self)          # collapse Linear fake-quant params\n",
         "        from .quantize import bake_weight_fake_quant   # local: see above\n"
         "        bake_weight_fake_quant(self)          # collapse Linear fake-quant params\n"),
    ],
    "folding.py": [
        ("    _require_torch()\n    bn_net.eval(); folded_net.eval()\n",
         "    _require_torch()\n"
         "    from .train import forward_over_time    # local: train imports quantize\n"
         "                                            # imports folding.\n"
         "    bn_net.eval(); folded_net.eval()\n"),
    ],
    "quantize.py": [
        ("    _require_torch()\n    train_cfg: TrainSpec = cfg[\"train\"]\n    net.eval()\n",
         "    _require_torch()\n"
         "    # local import: train.py imports deploy_and_measure from this module,\n"
         "    # so these cannot be imported at module level.\n"
         "    from .train import (evaluate, build_optimizer, build_scheduler,\n"
         "                        train_one_epoch, hardware_flush_steps)\n"
         "    train_cfg: TrainSpec = cfg[\"train\"]\n"
         "    net.eval()\n"),
        ("    _require_torch()\n    layers, li = [], 0\n",
         "    _require_torch()\n"
         "    from .train import hardware_flush_steps      # local: see deploy_and_measure\n"
         "    layers, li = [], 0\n"),
    ],
    # run_training was written when a dataset meant one hard-coded directory.
    # The package builds a DatasetBundle first and hands over ready loaders, so
    # the parameter changes from a path to the loaders themselves. data_dir is
    # kept as a fallback so Practice2.py's own calling convention still works.
    "train.py": [
        # quant_gap subtracted a warmup-phase number from an end-of-training
        # one. With qat_warmup_frac=0.25 and epochs=40 those sit 30 epochs
        # apart, so the "gap" was dominated by training progress and came out
        # NEGATIVE on good runs -- reading as though quantization had improved
        # accuracy. The comparison that answers "what did conversion cost" uses
        # the same weights at the same moment: the grid-trained network just
        # before export against the exported one. Both numbers already existed.
        ('    hw["float_val_accuracy"] = float_best\n'
         '    hw["quant_gap"] = float_best - hw["hw_val_accuracy"]\n',
         '    hw["float_val_accuracy"] = float_best\n'
         '\n'
         '    # What conversion cost: same weights, same point in training,\n'
         '    # measured before and after export. Positive means export lost\n'
         '    # accuracy, which is the only direction this can honestly go.\n'
         '    # inline mode names it grid_val_accuracy, tail/ptq names it\n'
         '    # qat_val_accuracy; both are the trained net just before export.\n'
         '    pre_export = hw.get("grid_val_accuracy") or hw.get("qat_val_accuracy")\n'
         '    hw["quant_gap"] = ((pre_export - hw["hw_val_accuracy"])\n'
         '                       if pre_export else None)\n'
         '    hw["pre_export_val_accuracy"] = pre_export\n'
         '\n'
         '    # What the whole schedule bought, warmup best to deployed. Useful,\n'
         '    # but it is a training-progress figure and not a conversion cost,\n'
         '    # so it gets a name that says so.\n'
         '    hw["end_to_end_gain"] = hw["hw_val_accuracy"] - float_best\n'),
        ("def run_training(cfg: dict, data_dir: str, device=None, report_fn=None, "
         "ckpt_path: str = None) -> float:",
         "def run_training(cfg: dict, data_dir: str = None, device=None, report_fn=None,\n"
         "                 ckpt_path: str = None, loaders=None) -> float:"),
        ("    train_loader, val_loader, _ = build_dataloaders(cfg, data_dir=data_dir)\n",
         "    if loaders is None:\n"
         "        raise ValueError(\n"
         "            \"run_training needs `loaders`. Build them with\\n\"\n"
         "            \"    snnsearch.pipeline.prepare(cfg)\\n\"\n"
         "        \"which resolves the dataset through the registry rather than \"\n"
         "        \"assuming one directory layout.\")\n"
         "    train_loader, val_loader, _ = loaders\n"),
    ],
}

TAIL = {
    "neuron.py": '''

if not _HAS_TORCH:
    # Same reason as quantgrid: model.py imports these by name at load time.
    TdBatchNorm2d = ConvBNFoldQuant = None
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
        "import math\n"
        "from typing import Callable\n\n"
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
        "                     ConvBNFoldQuant)\n"
        "from ._torch import (_HAS_TORCH, _HAS_HS_API, _require_torch, _no_grad,\n"
        "                     torch, nn, F, layer, functional, Custom_LIFNode, Custom_IFNode)\n",
        [(1297, 1477)],
    ),
    "folding.py": (
        "Removing batch normalization by folding it into weights and thresholds.",
        "from .hardware import W_ALPHA, W_DELTA, INT16_MAX\n"
        "from .quantgrid import fake_quantize_weight, quantized_threshold_int, fold_bias_band\n"
        "from .neuron import HardwareLIFNode\n"
        "from .model import DVSGesturePuru\n"
        "from ._torch import _HAS_TORCH, _require_torch, _no_grad, torch, nn, layer, functional\n\n"
"# verify_fold needs the time loop from train.py. Imported inside the function\n"
"# rather than at module scope, because train.py imports this module.\n",
        [(1503, 1707)],
    ),
    "quantize.py": (
        "Quantization-aware training, the deployment audit, and the export.",
        "import json\nimport math\nimport os\nfrom copy import deepcopy\n\n"
        "from .config import TrainSpec\n"
        "from .hardware import W_ALPHA, W_DELTA, INT16_MAX, W_BITS\n"
        "from .quantgrid import fake_quantize_weight, weight_clip_fraction\n"
        "from .neuron import HardwareLIFNode, _WeightFakeQuant\n"
        "from .folding import fold_bn, verify_fold, fold_bias_report\n"
        "from ._torch import _HAS_TORCH, _require_torch, _no_grad, torch, nn, layer\n",
        [(1725, 1921)],
    ),
    "train.py": (
        "Optimizer construction, the training loop, and the pipeline flush.",
        "import math\nfrom copy import deepcopy\nfrom dataclasses import asdict\n\n"
        "from .config import TrainSpec\n"
        "from .neuron import HardwareLIFNode\n"
        "from .model import build_model\n"
        "from .quantize import deploy_and_measure, deployment_report, hardware_export\n"
        # SynOps is measured after training, by pipeline.measure_run_synops, on
        # the converted network. Importing the counter here made it look wired
        # in when nothing called it.

        "from ._torch import _HAS_TORCH, _require_torch, torch, nn, F, functional\n",
        [(1990, 2241)],
    ),
    "spaces.py": (
        "The searchable space, as a picklable module-level object.\n\n"
        "Ray checkpoints the searcher by pickling it, so this cannot be a closure.",
        "from .config import (InputSpec, ConvLayerSpec, EncoderSpec, OutputSpec,\n"
        "                     DownsampleSpec, HeadSpec, NeuronSpec, TrainSpec,\n"
        "                     parse_fc_widths)\n"
        "from .hardware import HW_TAU_CHOICES\n",
        # starts at 2360, not 2366: FC_WIDTH_CHOICES and the comment recording
        # the trials that narrowed it belong to this module, not to the gap
        # between modules.
        [(2360, 2578)],
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


def apply_patches(name, text):
    """Apply this module's PATCHES, insisting each one matches exactly once.

    A patch that silently fails to apply is worse than no patch: the module
    still imports, and the missing local import only surfaces when the function
    is finally called, which for a training path can be twenty minutes in. So a
    miss and an ambiguous match are both errors.
    """
    for old, new in PATCHES.get(name, []):
        hits = text.count(old)
        if hits != 1:
            sys.exit(
                f"patch for {name} matched {hits} times, expected exactly 1.\n"
                f"Practice2.py has changed under this patch. The text sought was:\n"
                f"---\n{old}---")
        text = text.replace(old, new)
    return text


def write_module(name, doc, imports, spans, lines):
    body = []
    for a, b in spans:
        body.append("\n".join(lines[a - 1:b]).rstrip())
    span_txt = ", ".join(f"{a}-{b}" for a, b in spans)
    out = BANNER.format(doc=doc, span=span_txt) + "\n" + imports + "\n\n" + "\n\n\n".join(body) + "\n"
    out += TAIL.get(name, "")
    out = apply_patches(name, out)
    dest = os.path.join(PKG, name)
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    n = out.count("\n")
    npatch = len(PATCHES.get(name, []))
    print(f"  {name:<16} {span_txt:<20} {n:>5} lines"
          + (f"   +{npatch} patch" + ("es" if npatch > 1 else "") if npatch else ""))
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
