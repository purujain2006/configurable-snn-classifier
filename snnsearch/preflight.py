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
                  largest network the search can propose
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
import os
import shutil
import sys
import time
import traceback

from .config import TrainSpec


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


class _Context:
    """What the stages learn and pass along."""
    bundle = data = coding = None
    max_bits = 1
    t_small = None
    points = []
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
        import torch
        if not torch.cuda.is_available():
            return "WARN", ["no GPU, nothing to measure"]
        from .spaces import config_to_specs
        from .cost import count_neurons_and_synapses
        from .model import build_model
        from .train import forward_over_time
        from ._torch import F

        def load(f):
            spec = config_to_specs(f)
            return count_neurons_and_synapses(spec)["totals"]["neurons"] * spec["input"].T
        heavy = max(ctx.points, key=load)
        spec = config_to_specs(heavy)
        dev = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        net = build_model(spec).to(dev).train()
        inp = spec["input"]
        side = inp.resize_to or inp.H
        top = 2 ** int(inp.input_bits) - 1
        x = torch.randint(0, top + 1, (inp.N, inp.T, inp.C, side, side), device=dev).float()
        y = torch.randint(0, spec["output"].num_classes, (inp.N,), device=dev)
        for _ in range(2):           # float phase, then the folded integer phase
            F.cross_entropy(forward_over_time(net, x), y).backward()
            net.zero_grad(set_to_none=True)
            if not getattr(net, "_qat_folded", False):
                net.eval()
                net.to_qat_folded(bias_mode=spec["train"].fold_bias_mode)
                net.to(dev).train()
        peak = torch.cuda.max_memory_reserved()
        del net, x, y
        torch.cuda.empty_cache()
        need = peak * 1.3                                 # margin for fragmentation
        context = 0.7 * 2**30                             # CUDA context per process
        frac = float(s.get("gpu_fraction", 1.0))
        budget = frac * ctx.gpu_total
        per_gpu = max(1, int(ctx.gpu_total // (need + context)))
        cpus = os.cpu_count() or 1
        per_cpu = cpus // (s.get("cpu_per_trial") or (int(s.get("num_workers", 0)) + 1))
        fit = min(per_gpu, max(1, per_cpu), 16)
        suggest = round(1.0 / fit, 3)
        lines = [f"largest proposal ({inp.T} steps, {side}x{side}, "
                 f"{'signed' if spec['neuron'].signed_spikes else 'unsigned'}) peaked at "
                 f"{peak / 2**30:.2f} GiB",
                 f"gpu_fraction {frac} gives each trial {budget / 2**30:.1f} GiB, "
                 f"{int(round(1 / frac))} trials per GPU",
                 f"suggested gpu_fraction: {suggest} ({fit} trials per GPU"
                 f"{', limited by CPU cores' if per_cpu < per_gpu else ''})"]
        ctx.suggested_gpu_fraction = suggest
        if need > budget:
            return "FAIL", lines + ["the largest trial would run out of memory in its share; "
                                    f"set search.gpu_fraction to {suggest} or less"]
        return "PASS", lines

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
