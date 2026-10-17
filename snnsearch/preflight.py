"""Everything that can sink a search, checked in minutes, before it runs for days.

    python main.py preflight -c configs/dvs128_search05.yaml           # ~15 minutes
    python main.py preflight -c configs/dvs128_search05.yaml --quick   # ~2 minutes

Run it on the machine the search will run on, with the same environment. Each
of these cost a real search hours or a lost night before this existed:

    config        settings that contradict each other, such as an early-stopping
                  cut that falls inside the float warm-up
    environment   a missing package (ray without pydantic), a package file that
                  never reached the cluster, no GPU, no room for Ray's scratch
    dataset       a dataset root that does not exist, frame caches not built yet,
                  inputs an axon cannot carry
    search space  options that never fit the chip, a balanced block that cannot
                  be drawn, a combination of options that cannot be exported
    memory        a trial too large for its share of the GPU, measured on the
                  heaviest of the balanced block; a warning for the heavier
                  ones Optuna can propose after it
    exactness     two trained networks, checked clip by clip against the
                  simulator on this machine's GPU
    mini search   a real search through Ray: six trials, the balanced block,
                  early stopping, the simulator check, finalists and the report

A search will not start until a full preflight has passed for the same config
and the same code (preflight.json in the results directory, matched by a
fingerprint). Change either and it asks for a new one. --skip-preflight on
`search` overrides that, for when you know why.
"""

import glob
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback

from .config import TrainSpec
from .planning import plan_network
from .spaces import RandomTrial, config_to_specs, sample_config


# Settings that change how long a search runs but not whether it works, so
# changing them keeps a passed preflight valid.
_SCALE_ONLY = {("search", "trials"), ("search", "skip_preflight"),
               ("search", "max_failures"), ("search", "target")}


# ---------------------------------------------------------------------------
# rules every search config must satisfy
# ---------------------------------------------------------------------------

def warmup_epochs(epochs: int) -> int:
    """Float epochs before inline QAT folds onto the integer grid (train.py)."""
    frac = TrainSpec().qat_warmup_frac
    w = max(1, round(epochs * frac))
    return min(w, epochs - 1) if epochs > 1 else 1


def n_balanced(search: dict) -> int:
    """Trials in the balanced block; balanced_repeats is the older setting."""
    return int(search.get("balanced_trials") or 48 * int(search.get("balanced_repeats") or 0))


def check_search_config(cfg):
    """(errors, warnings) for a search config. An error stops the search
    before Ray starts; a warning is printed and the search goes ahead."""
    errors, warnings = [], []
    s = cfg["search"]
    epochs = int(s.get("epochs") or 0)
    if epochs < 2:
        errors.append("search.epochs must be at least 2: one float epoch, then at "
                      "least one on the integer grid")
        return errors, warnings
    warm = warmup_epochs(epochs)
    if (s.get("scheduler") or "asha") == "asha":
        grace = int(s.get("grace_period") or 0)
        if grace <= warm:
            errors.append(
                f"search.grace_period is {grace}, but the first {warm} of {epochs} epochs "
                f"are the float warm-up. The first early-stopping cut would judge trials "
                f"before they have trained on the chip's integer weights. Set "
                f"grace_period to at least {warm + 1}; {warm + 10} gives the integer "
                f"phase room to settle.")
        elif grace < warm + 5:
            warnings.append(
                f"search.grace_period {grace} is only {grace - warm} epoch(s) after the "
                f"float warm-up ends at {warm}; accuracy dips right after the switch to "
                f"integer weights, so the first cut may catch trials mid-dip")
    trials, nb = int(s.get("trials") or 0), n_balanced(s)
    if nb > trials:
        errors.append(f"search.balanced_trials ({nb}) is more than search.trials ({trials})")
    if nb and (s.get("space") or "uniform") == "uniform":
        warnings.append("search.balanced_trials is ignored with space: uniform "
                        "(the block draws per-layer shapes); use space: per_layer")
    if nb and not s.get("protect_balanced", True):
        warnings.append("search.protect_balanced is off, so early stopping can cut "
                        "balanced-block trials and unbalance the comparison")
    if int(s.get("finalists") or 0) > 0 and int(s.get("finalist_seeds") or 0) < 2:
        warnings.append("search.finalist_seeds below 2 cannot average out a lucky seed")
    t_choices = s.get("T_choices") or [cfg["encoding"].get("T", 16)]
    if not all(isinstance(t, int) and t >= 1 for t in t_choices):
        errors.append(f"search.T_choices must be whole numbers of at least 1, got {t_choices}")
    ds = cfg.get("dataset") or {}
    if ds.get("name") == "dvs128" and not ds.get("module"):
        root = os.path.expanduser(str(ds.get("root") or ""))
        if not root or not os.path.isdir(root):
            errors.append(f"dataset.root does not exist: {root or '(not set)'}. Point it "
                          f"at the folder holding events_np/ (or download/), or pass "
                          f"--data-root")
    return errors, warnings


# ---------------------------------------------------------------------------
# the record a search checks for
# ---------------------------------------------------------------------------

def package_files():
    here = os.path.dirname(os.path.abspath(__file__))
    return sorted(glob.glob(os.path.join(here, "**", "*.py"), recursive=True))


def fingerprint(cfg) -> str:
    """Hash of the config and of the package's source. Line endings are
    normalized, so a checkout on Windows and one on Linux agree."""
    c = json.loads(json.dumps({k: v for k, v in cfg.items()
                               if not k.startswith("_") and k != "report"}, default=str))
    for section, key in _SCALE_ONLY:
        (c.get(section) or {}).pop(key, None)
    h = hashlib.sha256(json.dumps(c, sort_keys=True).encode())
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in package_files():
        h.update(os.path.relpath(path, root).replace("\\", "/").encode())
        with open(path, "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def preflight_gate(cfg, out_dir):
    """None if the search may start, otherwise what to do first."""
    if cfg["search"].get("skip_preflight"):
        return None
    src = cfg.get("_source") or "<your config>"
    run = f"  python main.py preflight -c {src}"
    path = os.path.join(out_dir, "preflight.json")
    if not os.path.isfile(path):
        return ("No preflight has passed for this search yet. Run it first (about 15 "
                "minutes, on this machine):\n" + run +
                "\nor start anyway with --skip-preflight.")
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if not rec.get("passed"):
        return "The last preflight failed (preflight.json). Fix it and run it again:\n" + run
    if rec.get("quick"):
        return ("Only a quick preflight has passed. The search needs the full one, "
                "which runs a small search end to end:\n" + run)
    if rec.get("fingerprint") != fingerprint(cfg):
        return (f"The config or the code changed since the preflight of {rec.get('when')}. "
                f"Run it again:\n" + run)
    return None


# ---------------------------------------------------------------------------
# the stages
# ---------------------------------------------------------------------------

class _Report:
    def __init__(self):
        self.stages = []

    def add(self, name, status, lines, seconds):
        self.stages.append({"stage": name, "status": status, "notes": lines,
                            "seconds": round(seconds, 1)})
        print(f"  [{status}] {name:<14s} ({seconds:.0f}s)", flush=True)
        for line in lines:
            print(f"         {line}", flush=True)

    @property
    def failed(self):
        return any(s["status"] == "FAIL" for s in self.stages)


def _run_stage(report, name, fn):
    """fn returns (status, lines). Any exception is a FAIL with its traceback
    tail, because an unexpected crash here is what the preflight is for."""
    t0 = time.time()
    try:
        status, lines = fn()
    except SystemExit as exc:
        status, lines = "FAIL", [str(exc)]
    except Exception as exc:
        tb = traceback.format_exc().strip().splitlines()
        status, lines = "FAIL", [f"{type(exc).__name__}: {exc}"] + tb[-6:-1]
    report.add(name, status, lines, time.time() - t0)
    return status != "FAIL"


def _missing_package_modules():
    """Package modules that are imported somewhere but whose file is missing,
    as when a new file was never committed. A lazy import inside a function
    only fails when that function runs, which can be hours in."""
    import ast
    here = os.path.dirname(os.path.abspath(__file__))
    missing = set()
    for path in package_files():
        pkg = os.path.dirname(path)
        try:
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except SyntaxError as exc:
            missing.add(f"{os.path.relpath(path, here)}: syntax error {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level >= 1 and node.module:
                base = pkg
                for _ in range(node.level - 1):
                    base = os.path.dirname(base)
                target = os.path.join(base, *node.module.split("."))
                if not (os.path.isfile(target + ".py") or os.path.isdir(target)):
                    missing.add(f"{node.module} (imported by {os.path.relpath(path, here)})")
    return sorted(missing)


# ---------------------------------------------------------------------------
# memory: the heaviest trial the search can propose, and whether it fits
# ---------------------------------------------------------------------------

# What each trial's process holds on the GPU before its first tensor. It sits
# outside the cap a trial sets with set_per_process_memory_fraction.
CUDA_CONTEXT = 0.7 * 2**30
# Headroom over the measured peak, for fragmentation late in a run
MEMORY_MARGIN = 1.3

# Training options that leave the network's shape alone but can add to its
# memory: every switch on, the largest amounts. Each measured trial uses these
# with both norms: tdBN runs every timestep at once, and on DVS128 Gesture it
# needed 25% more than BN for the same shape. A value the space no longer
# offers is drawn at random instead (_PreferTrial).
HEAVY_TRAINING = {"tau_mode": "channel", "trainable_threshold": True,
                  "use_conv_dropout": True, "conv_dropout": 0.3,
                  "use_rate_penalty": True, "rate_penalty": 1e-4,
                  "use_augment": True, "aug_shift": 0.2, "aug_cutout": 0.4, "aug_drop": 0.3,
                  "loss": "tet", "grad_clip": 5.0}
HEAVY_NORMS = ("bn", "tdbn")


def ray_trials_per_gpu(frac: float) -> int:
    """How many trials Ray puts on one GPU at this gpu_fraction. Ray counts
    resources in units of 1/10,000, so 0.167 packs 5 (6 x 0.167 > 1), not 6."""
    units = max(1, int(round(float(frac) * 10_000)))
    return max(1, 10_000 // units)


def memory_verdict(peak: float, gpu_total: float, frac: float, cpus: int,
                   cpu_per_trial: int) -> dict:
    """Whether the heaviest trial fits, from its measured peak in bytes.

    Two things must hold. The trial's own cap (gpu_fraction of the card, set
    in every trial) must cover its peak with margin, or it runs out of memory
    on its own budget however empty the card is. And the trials Ray packs onto
    one card, each at that size plus its CUDA context, must fit the card, or
    one of them runs out when the others are resident.
    """
    need = peak * MEMORY_MARGIN
    per_trial = need + CUDA_CONTEXT
    cap = float(frac) * gpu_total
    resident = ray_trials_per_gpu(frac)
    by_memory = int(gpu_total // per_trial)
    by_cpu = max(1, int(cpus) // max(1, int(cpu_per_trial)))
    fit = min(by_memory, by_cpu, 16)
    # rounded down, so Ray still packs `fit` trials at the suggested value
    suggest = math.floor(1000 / fit) / 1000 if fit >= 1 else None
    return {"need": need, "per_trial": per_trial, "cap": cap, "resident": resident,
            "fits_cap": need <= cap, "fits_card": resident * per_trial <= gpu_total,
            "fit": fit, "suggest": suggest, "cpu_limited": by_cpu < by_memory}


def activation_load(flat: dict) -> int:
    """What a trial keeps for backpropagation through time, up to a constant:
    the input and every conv and pool output, at every timestep."""
    spec = config_to_specs(flat)
    inp = spec["input"]
    plan = plan_network(inp, spec["encoder"], spec["output"], spec["downsample"], spec["head"])
    side = inp.resize_to or inp.H
    units = inp.C * side * side
    for b in plan.blocks:
        units += b.out_channels * b.conv_out_hw[0] * b.conv_out_hw[1]
        if b.pool:
            units += b.out_channels * b.out_hw[0] * b.out_hw[1]
    return units * int(inp.T)


def _shape_signature(flat: dict) -> tuple:
    spec = config_to_specs(flat)
    inp, head = spec["input"], spec["head"]
    return (inp.T, inp.resize_to, inp.input_bits, spec["encoder"].layers_json,
            spec["encoder"].depth, spec["encoder"].channels, spec["downsample"].pool_type,
            head.final_reduction, head.fc_widths, spec["neuron"].signed_spikes,
            flat.get("bias_mode"))


def _prefer_sample(space, rng, fixed: dict) -> dict:
    """sample_config with `fixed` where the space offers it. Changing one
    option can move another's range (adamw and adam decay differently), and
    that one is then drawn again."""
    trial = _PreferTrial(rng, fixed)
    consts = space(trial)
    return {**trial.params, **consts}


def distinct_heaviest(points, keep: int) -> list:
    """The `keep` heaviest of `points` that fit the chip, one per shape."""
    found = {}
    for f in sorted((f for f in points if not f.get("infeasible_geometry")),
                    key=activation_load, reverse=True):
        found.setdefault(_shape_signature(f), f)
        if len(found) == keep:
            break
    return list(found.values())


def heaviest_shapes(space, starts, keep: int = 2, seed: int = 0) -> list:
    """Up to `keep` different trials that fit the chip and keep the most for
    backpropagation through time (activation_load), heaviest first.

    Random proposals rarely land where the heaviest trials are: the largest
    input, the most channels, the least downsampling and the longest T the
    chip allows, all at once. So this climbs from the heaviest few of
    `starts`: each step tries every value of one integer or categorical
    parameter, as the space offers them, and keeps the heaviest that fits.
    """
    import random
    rng = random.Random(seed)
    found = {}
    for start in distinct_heaviest(starts, keep + 1):
        point, load = start, activation_load(start)
        for _ in range(10):
            rec = _ChoiceRecorder(random.Random(0), fixed=point)
            space(rec)
            moved = False
            for name, values in rec.choices.items():
                for v in values:
                    if v == point.get(name):
                        continue
                    cand = _prefer_sample(space, rng, {**point, name: v})
                    if cand.get("infeasible_geometry"):
                        continue
                    cand_load = activation_load(cand)
                    if cand_load > load:
                        point, load, moved = cand, cand_load, True
            if not moved:
                break
        found.setdefault(_shape_signature(point), (load, point))
    ranked = sorted(found.values(), key=lambda lp: -lp[0])
    return [p for _, p in ranked[:keep]]


def heavy_variants(space, shape: dict, seed: int = 0) -> list:
    """`shape` with HEAVY_TRAINING, once per norm in HEAVY_NORMS."""
    import random
    rng = random.Random(seed)
    out = []
    for norm in HEAVY_NORMS:
        flat = _prefer_sample(space, rng, {**shape, **HEAVY_TRAINING, "norm": norm})
        if flat.get("infeasible_geometry"):
            raise RuntimeError(f"the heaviest shape stopped fitting the chip: {shape}")
        if flat not in out:
            out.append(flat)
    return out


def measure_training_peak(spec, device) -> int:
    """Peak GPU memory reserved, in bytes, by one training step in each
    phase: on float weights, then folded onto the integer grid, through the
    same train_one_epoch a trial runs."""
    import torch
    from .model import build_model
    from .train import build_optimizer, train_one_epoch
    from ._torch import nn
    inp, tr = spec["input"], spec["train"]
    side = inp.resize_to or inp.H
    top = 2 ** int(inp.input_bits) - 1
    x = torch.randint(0, top + 1, (inp.N, inp.T, inp.C, side, side)).float()
    y = torch.randint(0, spec["output"].num_classes, (inp.N,))
    batch = [(x, y, None)]
    criterion = nn.CrossEntropyLoss(label_smoothing=tr.label_smoothing)
    augment = None
    if tr.augment:
        from .augment import augment_batch
        augment = lambda b: augment_batch(b, tr, bool(inp.hflip_ok))     # noqa: E731

    def step(net, optimizer):
        train_one_epoch(net, batch, optimizer, device, criterion, grad_clip=tr.grad_clip,
                        rate_penalty=tr.rate_penalty or 0.0, loss_kind=tr.loss,
                        augment=augment)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    net = build_model(spec).to(device)
    float_opt = build_optimizer(net, tr)    # run_training keeps it through the grid phase
    step(net, float_opt)
    net.eval()
    net.to_qat_folded(bias_mode=tr.fold_bias_mode,
                      fold_bias_margin=getattr(tr, "fold_bias_margin", 0.05),
                      fold_bias_qat_form=getattr(tr, "fold_bias_qat_form", "threshold"))
    net.to(device)
    step(net, build_optimizer(net, tr))
    peak = torch.cuda.max_memory_reserved(device)
    del net, float_opt
    torch.cuda.empty_cache()
    return peak


def is_oom(exc) -> bool:
    """Out of GPU memory, from PyTorch's allocator or from a kernel that
    could not get its workspace (torch.AcceleratorError)."""
    import torch
    return isinstance(exc, torch.cuda.OutOfMemoryError) or         (isinstance(exc, RuntimeError) and "out of memory" in str(exc))


def estimate_training_peak(flat, device):
    """(bytes, exact) for a trial that may not fit the GPU at all.

    Measured directly at the trial's batch size when it fits, so exact. When
    it does not, extrapolated from the largest pair of halved batch sizes
    that does, since memory grows about linearly with the batch. None if not
    even a batch of two fits.
    """
    direct = _peak_at(flat, device, int(flat["N"]))
    if direct is not None:
        return direct, True
    hi = int(flat["N"]) // 2
    while hi >= 2:
        est = extrapolated_peak(flat, device, hi)
        if est is not None:
            return est, False
        hi //= 2
    return None, False


def extrapolated_peak(flat, device, hi: int):
    """Peak at the trial's batch size, extrapolated in a line through the
    peaks at batches hi and hi // 2. None if either does not fit."""
    lo = hi // 2
    p_hi = _peak_at(flat, device, hi)
    p_lo = _peak_at(flat, device, lo) if p_hi is not None else None
    if p_lo is None:
        return None
    return p_lo + (p_hi - p_lo) * (int(flat["N"]) - lo) / (hi - lo)


def _peak_at(flat, device, batch):
    """measure_training_peak at this batch size, or None when it does not fit."""
    import torch
    try:
        return measure_training_peak(config_to_specs({**flat, "N": batch}), device)
    except Exception as exc:
        if not is_oom(exc):
            raise
        torch.cuda.empty_cache()
        return None


def _describe(spec) -> str:
    inp = spec["input"]
    side = inp.resize_to or inp.H
    return (f"{inp.T} steps, {side}x{side}, "
            f"{'signed' if spec['neuron'].signed_spikes else 'unsigned'}, "
            f"{spec['encoder'].norm}")


class _ChoiceRecorder(RandomTrial):
    """Follows `fixed` like RandomTrial and records every value each integer
    or categorical parameter on the way could have taken."""

    def __init__(self, rng, fixed=None):
        super().__init__(rng, fixed)
        self.choices = {}

    def suggest_int(self, name, low, high, step=1, log=False):
        self.choices[name] = list(range(low, high + 1, step))
        return super().suggest_int(name, low, high, step, log)

    def suggest_categorical(self, name, choices):
        self.choices[name] = list(choices)
        return super().suggest_categorical(name, choices)


class _PreferTrial(RandomTrial):
    """RandomTrial that takes a fixed value only where the space offers it and
    draws at random otherwise, so an option the space renames or narrows
    cannot break the preflight."""

    def _drop_unless(self, name, ok):
        if name in self.fixed and not ok(self.fixed[name]):
            del self.fixed[name]

    def suggest_int(self, name, low, high, step=1, log=False):
        self._drop_unless(name, lambda v: low <= v <= high and (v - low) % step == 0)
        return super().suggest_int(name, low, high, step, log)

    def suggest_float(self, name, low, high, log=False):
        self._drop_unless(name, lambda v: low <= v <= high)
        return super().suggest_float(name, low, high, log)

    def suggest_categorical(self, name, choices):
        self._drop_unless(name, lambda v: v in choices)
        return super().suggest_categorical(name, choices)


class _Context:
    """What the stages learn and pass along."""
    bundle = data = coding = None
    max_bits = 1
    t_small = None
    points = []
    fits = []
    space = None
    gpu_total = None


def run_preflight(cfg, quick=False):
    """Run every stage, print a checklist, write preflight.json. Returns 0 if
    nothing failed."""
    from .pipeline import results_dir
    out_dir = results_dir(cfg)
    s = cfg["search"]
    ctx = _Context()
    report = _Report()
    t_start = time.time()
    print(f"preflight for {cfg.get('_source') or 'config'} -> {out_dir}\n", flush=True)

    # -- config ---------------------------------------------------------------
    def config_stage():
        errors, warnings = check_search_config(cfg)
        if errors:
            return "FAIL", errors + warnings
        return ("WARN" if warnings else "PASS"), warnings or ["settings are consistent"]

    # -- environment ----------------------------------------------------------
    def environment_stage():
        lines, warn = [], False
        problems = []
        for mod in ("torch", "spikingjelly.activation_based", "ray.tune",
                    "ray.tune.search.optuna", "optuna"):
            try:
                __import__(mod)
            except ImportError as exc:
                fix = ' (pip install "pydantic<3" "ray[tune]")' if mod.startswith("ray") else ""
                problems.append(f"cannot import {mod}: {exc}{fix}")
        missing = _missing_package_modules()
        if missing:
            problems.append("package files missing (never committed or copied?): "
                            + "; ".join(missing))
        if problems:
            return "FAIL", problems
        import torch
        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            ctx.gpu_total = torch.cuda.get_device_properties(0).total_memory
            lines.append(f"{len(names)} GPU(s): {', '.join(names)}, "
                         f"{ctx.gpu_total / 2**30:.0f} GiB each")
        else:
            warn = True
            lines.append("no CUDA GPU visible: the search would train on the CPU, "
                         "which is far too slow for a real search")
        from .search import _ray_temp_dir

        class _Quiet:
            def log(self, msg):
                pass
        tmp = _ray_temp_dir(s.get("ray_temp_dir"), _Quiet())
        if tmp:
            os.makedirs(tmp, exist_ok=True)
            probe = os.path.join(tmp, ".preflight_write_test")
            open(probe, "w").close()
            os.remove(probe)
            free = shutil.disk_usage(tmp).free / 2**30
            lines.append(f"Ray scratch {tmp}: writable, {free:.0f} GiB free")
            if free < 20:
                warn = True
                lines.append("under 20 GiB free for Ray's scratch; set search.ray_temp_dir "
                             "to a short path on a larger disk")
        cpus = os.cpu_count() or 1
        per = s.get("cpu_per_trial") or (int(s.get("num_workers", 0)) + 1)
        lines.append(f"{cpus} CPU cores, {per} per trial: at most {cpus // per} trials at once")
        return ("WARN" if warn else "PASS"), lines

    # -- dataset --------------------------------------------------------------
    def dataset_stage():
        from .data import build_dataset
        from .spaces import data_facts, MULTI_BIT_CODINGS, MAX_INPUT_BITS
        lines, warn = [], False
        t_choices = [int(t) for t in (s.get("T_choices") or [cfg["encoding"].get("T", 16)])]
        ds = cfg["dataset"]
        ready = list(t_choices)
        if ds.get("name") == "dvs128" and not ds.get("module"):
            from .data.builtin import frame_cache_is_complete, validate_dvs_root
            root = validate_dvs_root(ds["root"])
            ready = [t for t in t_choices if frame_cache_is_complete(root, t)]
            todo = [t for t in t_choices if t not in ready]
            if todo:
                warn = True
                lines.append(f"frame caches for T={todo} are not built yet; the search "
                             f"builds them before starting, tens of minutes each")
        ctx.t_small = min(ready) if ready else min(t_choices)
        ctx.bundle = build_dataset({**ds, "T": ctx.t_small})
        ctx.data = data_facts(ctx.bundle)
        ctx.coding = cfg["encoding"].get("coding") or \
            ("passthrough" if ctx.bundle.is_event else "levels")
        ctx.max_bits = int(s.get("max_input_bits") or MAX_INPUT_BITS) \
            if ctx.coding in MULTI_BIT_CODINGS else 1
        lines.append(f"{ctx.bundle.name}: {ctx.data}, coding {ctx.coding}, "
                     f"input bits 1..{ctx.max_bits}")
        if ctx.coding not in MULTI_BIT_CODINGS:
            warn = True
            lines.append(f"coding {ctx.coding!r} cannot be checked on the simulator unless "
                         f"its inputs are whole numbers; levels or passthrough deploy")
        return ("WARN" if warn else "PASS"), lines

    # -- search space ---------------------------------------------------------
    def space_stage():
        import random
        from .spaces import (make_define_by_run, balanced_points, sample_config,
                             config_to_specs)
        from .model import build_model
        from .hs_export import integer_layers
        t_choices = [int(t) for t in (s.get("T_choices") or [cfg["encoding"].get("T", 16)])]
        space = make_define_by_run(
            batch_size=s.get("batch_size", 16), epochs=s["epochs"], data_dir_abs=".",
            t_choices=t_choices, per_layer=(s.get("space", "uniform") != "uniform"),
            require=s.get("require") or [], data=ctx.data, max_input_bits=ctx.max_bits,
            seed=cfg["run"].get("seed", 1))
        rng = random.Random(0)
        n_block = n_balanced(s) if space.per_layer else 0
        block = balanced_points(n_block, seed=cfg["run"].get("seed", 1), t_choices=t_choices,
                                batch_size=s.get("batch_size", 16), epochs=s["epochs"],
                                data=ctx.data, max_input_bits=ctx.max_bits,
                                run_seed=cfg["run"].get("seed", 1))
        # every block point through the real space, amounts drawn at random
        ctx.points = [sample_config(space, rng, fixed=p) for p in block]
        sampled = [sample_config(space, rng) for _ in range(300)]
        fits = [f for f in sampled if not f.get("infeasible_geometry")]
        ctx.space, ctx.fits = space, fits
        lines = [f"balanced block: {len(block)} trials, every one fits the chip",
                 f"{len(fits)} of {len(sampled)} random proposals fit the chip"]
        if not fits:
            return "FAIL", lines + ["no random proposal fits the chip; the search would "
                                    "score zero after the block"]
        # every block configuration must build, fold and export, so no
        # combination of options fails at the end of a trial
        candidates = ctx.points or fits[:24]
        for f in candidates:
            spec = config_to_specs(f)
            net = build_model(spec).eval()
            net.to_qat_folded(bias_mode=spec["train"].fold_bias_mode)
            integer_layers(net.export_deployed())
        lines.append(f"{len(candidates)} configurations build, fold and export")
        if not ctx.points:
            ctx.points = fits[:24]
        status = "WARN" if len(fits) < 0.2 * len(sampled) else "PASS"
        if status == "WARN":
            lines.append("under 20% of proposals fit; consider narrowing the ranges")
        return status, lines

    # -- memory ---------------------------------------------------------------
    def memory_stage():
        import random
        import torch
        if not torch.cuda.is_available():
            return "WARN", ["no GPU, nothing to measure"]
        dev = torch.device("cuda")
        frac = float(s.get("gpu_fraction", 1.0))
        cpus = os.cpu_count() or 1
        cpu_per_trial = s.get("cpu_per_trial") or (int(s.get("num_workers", 0)) + 1)
        total = ctx.gpu_total
        gib = lambda b: b / 2**30                                          # noqa: E731

        # The balanced block runs every one of its trials to the end, so its
        # heaviest must fit: the two heaviest shapes, each with both norms and
        # every memory-adding option on.
        measured = []
        for shape in distinct_heaviest(ctx.points, 2):
            for flat in heavy_variants(ctx.space, shape):
                spec = config_to_specs(flat)
                try:
                    peak = measure_training_peak(spec, dev)
                except Exception as exc:
                    if not is_oom(exc):
                        raise
                    torch.cuda.empty_cache()
                    return "FAIL", [f"a balanced-block trial ({_describe(spec)}, batch "
                                    f"{spec['input'].N}) does not fit the whole "
                                    f"{gib(total):.0f} GiB GPU even alone",
                                    "drop the largest of search.T_choices, or lower "
                                    "search.batch_size"]
                measured.append((peak, flat, spec))
        peak, block_flat, spec = max(measured, key=lambda m: m[0])
        v = memory_verdict(peak, total, frac, cpus, cpu_per_trial)
        lines = [f"heaviest balanced-block trial ({_describe(spec)}) peaked at "
                 f"{gib(peak):.2f} GiB with every memory-adding option on",
                 f"it needs {gib(v['need']):.1f} GiB with margin, plus "
                 f"{gib(CUDA_CONTEXT):.1f} GiB of CUDA context outside its cap",
                 f"gpu_fraction {frac} caps each trial at {gib(v['cap']):.1f} GiB and packs "
                 f"{v['resident']} per GPU"]
        if v["fit"] < 1:
            ctx.suggested_gpu_fraction = None
            return "FAIL", lines + [f"one such trial needs {gib(v['per_trial']):.1f} GiB, more "
                                    f"than the {gib(total):.1f} GiB GPU; drop the largest of "
                                    "search.T_choices, or lower search.batch_size"]
        lines.append(f"suggested gpu_fraction: {v['suggest']} ({v['fit']} "
                     f"trial{'s' if v['fit'] != 1 else ''} per GPU"
                     f"{', limited by CPU cores' if v['cpu_limited'] else ''})")
        ctx.suggested_gpu_fraction = v["suggest"]
        problems = []
        if not v["fits_cap"]:
            problems.append("the heaviest block trial would run out of memory under its "
                            "own cap")
        if not v["fits_card"]:
            problems.append(f"{v['resident']} such trials at once need "
                            f"{gib(v['resident'] * v['per_trial']):.1f} GiB of the GPU's "
                            f"{gib(total):.1f}")
        if problems:
            return "FAIL", lines + problems + [f"set search.gpu_fraction to {v['suggest']}"]

        # After the block, Optuna can propose heavier trials than any in it. One
        # over its cap runs out of memory at its first batch, and Ray retries it
        # and then drops it, so this warns instead of failing: how heavy the
        # space gets, and about how many proposals would be lost.
        top = heaviest_shapes(ctx.space, list(ctx.points) + list(ctx.fits), keep=1,
                              seed=cfg["run"].get("seed", 1) or 0)
        worst, worst_spec, exact = None, None, True
        for flat in heavy_variants(ctx.space, top[0]) if top else []:
            est, is_exact = estimate_training_peak(flat, dev)
            est = float("inf") if est is None else est
            if worst is None or est > worst:
                worst, worst_spec, exact = est, config_to_specs(flat), is_exact
        if worst is None or worst * MEMORY_MARGIN <= v["cap"]:
            lines.append("the heaviest trial the space can propose after the block also "
                         "fits its cap")
            return "PASS", lines
        # bytes per unit of activation_load, from the block measurements
        per_unit = max(p / activation_load(f) for p, f, _ in measured)
        rng = random.Random(0)
        props = [f for f in (sample_config(ctx.space, rng) for _ in range(2000))
                 if not f.get("infeasible_geometry")]
        over = sum(activation_load(f) * per_unit * MEMORY_MARGIN > v["cap"] for f in props)
        if worst == float("inf"):
            size = "more than the whole GPU"
        elif exact:
            size = f"{gib(worst):.1f} GiB"
        else:
            size = f"about {gib(worst):.0f} GiB, more than the whole GPU,"
        lines += [f"after the block the space can propose heavier trials: the heaviest "
                  f"({_describe(worst_spec)}) needs {size} at batch "
                  f"{worst_spec['input'].N}",
                  f"{over} of {len(props)} random proposals that fit the chip "
                  f"({100 * over / max(1, len(props)):.1f}%) would exceed the "
                  f"{gib(v['cap']):.1f} GiB cap; each such trial runs out of memory, is "
                  f"retried and then dropped"]
        if over:
            lines.append("to lose fewer, narrow the space (search.T_choices, the channel "
                         "range)" + (" or raise search.gpu_fraction" if frac < 1 else ""))
        return "WARN", lines

    # -- exactness ------------------------------------------------------------
    def exactness_stage():
        from .spaces import config_to_specs
        from .cost import count_neurons_and_synapses
        from .pipeline import resolve_specs, make_loaders
        from .train import run_training

        def size(f):
            spec = config_to_specs(f)
            return count_neurons_and_synapses(spec)["totals"]["neurons"]
        pool = sorted(ctx.points, key=size)
        wanted = [lambda f: f.get("signed_spikes") and f.get("final_reduction") == "gap",
                  lambda f: f.get("bias_mode") == "neuron" and f.get("pool_type") == "max"]
        chosen = []
        for want in wanted:
            pick = next((f for f in pool if want(f) and f not in chosen), None)
            if pick is not None:
                chosen.append(pick)
        for f in pool:
            if len(chosen) >= 2:
                break
            if f not in chosen:
                chosen.append(f)
        lines, bad = [], False
        for f in chosen:
            flat = {**f, "epochs": 2, "T": ctx.t_small}
            bundle, enc, spec = resolve_specs(cfg, flat, quiet=True)
            loaders = make_loaders(cfg, bundle, enc, spec, quiet=True)
            res = run_training(spec, loaders=loaders)
            ok = res["sim_disagreements"] == 0 and not res["sim_reason"]
            bad |= not ok
            lines.append(f"{'ok ' if ok else 'BAD'} signed {f.get('signed_spikes')}, "
                         f"head {f.get('final_reduction')}, bias {f.get('bias_mode')}, "
                         f"pool {f.get('pool_type', '-')}: pytorch {res['hw_val_accuracy']:.4f}, "
                         f"simulator {res['sim_val_accuracy']}, disagreements "
                         f"{res['sim_disagreements']}"
                         + (f", {res['sim_reason']}" if res["sim_reason"] else ""))
        return ("FAIL" if bad else "PASS"), lines

    # -- a real search, small -------------------------------------------------
    def mini_search_stage():
        import copy
        import csv
        import ray
        from .search import run_search
        from . import report as report_mod
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        mini = copy.deepcopy(cfg)
        sub = os.path.join(out_dir, "preflight_search")
        shutil.rmtree(sub, ignore_errors=True)
        epochs = 4
        mini["run"]["name"] = f"{cfg['run'].get('name', 'search')}_preflight"
        mini["run"]["results_dir"] = sub
        mini["search"].update({
            "trials": 6, "balanced_trials": 6, "balanced_repeats": 0, "epochs": epochs,
            "grace_period": warmup_epochs(epochs) + 1, "brackets": 1, "startup_trials": 0,
            "finalists": 1, "finalist_seeds": 2, "finalist_epochs": None,
            "T_choices": [ctx.t_small], "skip_preflight": True})
        os.makedirs(sub, exist_ok=True)
        try:
            run_search(mini, sub)
        finally:
            if ray.is_initialized():
                ray.shutdown()
        lines, bad = [], []
        rows = [json.loads(line) for line in open(os.path.join(sub, "trials.jsonl"),
                                                  encoding="utf-8") if line.strip()]
        done = [r for r in rows if r.get("status") == "complete"]
        lines.append(f"{len(done)} of 6 trials completed")
        if len(done) < 6:
            bad.append("not every trial completed; see the Ray errors above")
        cut = [r["trial_id"] for r in done if r.get("stopped_early")]
        if cut:
            bad.append(f"balanced-block trials were stopped early: {cut}")
        dis = [r["trial_id"] for r in done if r.get("sim_disagreements")]
        if dis:
            bad.append(f"PyTorch and the simulator disagree in trials {dis}")
        with open(os.path.join(sub, "leaderboard.csv"), encoding="utf-8") as fh:
            n_lb = sum(1 for _ in csv.DictReader(fh))
        lines.append(f"leaderboard.csv has {n_lb} rows")
        for name in ("best.json", "finalists.csv", "best_finalist.json"):
            if not os.path.isfile(os.path.join(sub, name)):
                bad.append(f"{name} was not written")
        path = report_mod.build(sub, cfg=mini)
        lines.append(f"finalists retrained and {os.path.basename(path)} built")
        return ("FAIL" if bad else "PASS"), lines + bad

    stages = [("config", config_stage), ("environment", environment_stage),
              ("dataset", dataset_stage), ("search space", space_stage),
              ("memory", memory_stage)]
    if not quick:
        stages += [("exactness", exactness_stage), ("mini search", mini_search_stage)]
    for name, fn in stages:
        if not _run_stage(report, name, fn):
            break

    passed = not report.failed and len(report.stages) == len(stages)
    minutes = (time.time() - t_start) / 60
    record = {"passed": passed, "quick": bool(quick), "fingerprint": fingerprint(cfg),
              "when": time.strftime("%Y-%m-%d %H:%M"), "minutes": round(minutes, 1),
              "config": cfg.get("_source"), "python": sys.executable,
              "suggested_gpu_fraction": getattr(ctx, "suggested_gpu_fraction", None),
              "stages": report.stages}
    with open(os.path.join(out_dir, "preflight.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    print()
    if passed:
        print(f"PREFLIGHT PASSED in {minutes:.1f} min"
              + (" (quick: the search still needs the full preflight)" if quick else ""))
        if not quick:
            print(f"start the search with\n  python main.py search -c "
                  f"{cfg.get('_source') or '<config>'}")
    else:
        print(f"PREFLIGHT FAILED after {minutes:.1f} min; fix the FAIL above and run it again")
    return 0 if passed else 1


__all__ = ["check_search_config", "fingerprint", "preflight_gate", "run_preflight",
           "warmup_epochs", "n_balanced"]
