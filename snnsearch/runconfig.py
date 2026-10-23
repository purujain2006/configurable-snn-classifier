"""The run configuration file.

It is YAML because every narrowed range in this project carries its evidence
beside it, the statistics that justified it, in a comment on the same line.
JSON has no comments, so a JSON search space would lose them. In YAML:

    channels: [32, 64]   # 128/256 were 0/14 feasible; V=0.85, p<1e-4

A dataset is code, such as a Dataset subclass, a transform and sometimes a
loader, so it is not in the file. The config holds values and points at a
Python callable for the rest.

Falls back to a small parser when PyYAML is absent, so `summary` still runs on
a machine with nothing installed.
"""

import json
import os


DEFAULTS = {
    "run": {"name": "run", "results_dir": None, "seed": 1},
    "dataset": {"name": "dvs128", "root": None},
    # input_bits: bits per input value when the search does not choose it
    # (`single` runs). A search samples it from 1 to search.max_input_bits
    "encoding": {"coding": None, "T": 16, "resize_to": None, "input_bits": None},
    # chip limits for a different build: total_axons, axon_fan_out,
    # axon_fan_in, neuron_fan_out, neuron_fan_in (hardware.set_limits)
    "limits": {},
    # The network `single` builds. Empty means the dataclass defaults, which is
    # what the search samples around. Field names match the dataclasses in
    # config.py, and an unknown one is an error rather than a silent default.
    # `single --from-best best.json` replaces this wholesale with a trial that
    # ran, which is the reliable way to reproduce a search winner.
    "architecture": {"encoder": {}, "downsample": {}, "head": {}, "neuron": {}},
    "train": {},           # TrainSpec fields: optimizer, lr, scheduler, ...
    "search": {
        # grace_period: the first early-stopping cut, in epochs. It must come
        # after the float warm-up (the first quarter of the epochs), or it
        # judges trials before they train on integer weights.
        "trials": 25, "epochs": 40, "grace_period": 15, "reduction_factor": 3,
        "brackets": 1, "target": 0.975, "space": "uniform",
        # "asha" stops weak trials early; "none" runs every trial in full
        "scheduler": "asha",
        # random trials after the balanced block, before Optuna steers
        "startup_trials": 10,
        # a trial that crashes (CUDA out of memory, say) is retried this often
        "max_failures": 3,
        # where Ray keeps its scratch. Must be a short path (unix socket limit)
        "ray_temp_dir": None,
        # regularizers every trial uses: conv_dropout, rate_penalty, augment
        "require": [],
        "gpu_fraction": 1.0, "batch_size": 16,
        # DataLoader worker processes per trial. The frame cache is decoded,
        # resized and encoded on the CPU, single-threaded, in step with the
        # forward pass, so at 0 the GPU waits on one core per trial. Set
        # cpu_per_trial to null to have the scheduler reserve num_workers + 1.
        "num_workers": 4, "cpu_per_trial": None,
        # Left unset, the search holds T at whatever `encoding` says. Listing
        # more values here searches over them, which needs a frame cache per
        # value: building those lazily inside parallel trials has several
        # processes decoding the same events into the same directory. Build
        # them sequentially before Ray starts (or with tools/build_cache.py).
        "T_choices": None,
        # "subject" holds out whole people for validation, like the test set
        # does. "clip" is the split every run before October 2026 used, kept
        # only to re-score old checkpoints on the clips they were selected on.
        "val_split": "subject",
        # k-fold cross-subject validation for scoring a chosen network: k > 0
        # splits the training people into k groups of near-equal clip counts
        # (shuffled by run.seed) and group val_fold is validation. 0 keeps the
        # single split above. `single --folds k` trains every fold in turn.
        "val_folds": 0, "val_fold": 0,
        # Trials at the start in which every option of every on/off or
        # either/or choice appears equally often (spaces.balanced_points).
        "balanced_trials": 0,
        # Bits per input value are searched from 1 to this, when the coding
        # can send more than one (passthrough, levels). None means 4.
        "max_input_bits": None,
        # After the search, retrain the top `finalists` trials with
        # `finalist_seeds` new seeds each and pick the best mean. 0 turns it
        # off. finalist_epochs None keeps the search's epoch count.
        "finalists": 3, "finalist_seeds": 3, "finalist_epochs": None,
        # Early stopping never cuts the balanced block's trials, so every
        # option is judged on a full run.
        "protect_balanced": True,
        # A search refuses to start until `main.py preflight` has passed for the
        # same config and code. True starts it anyway (--skip-preflight).
        "skip_preflight": False,
    },
    "objective": {
        "mode": "accuracy",          # accuracy | constrained | weighted | pareto
        "synops_budget": None,
        "synops_reference": None,
        "weight": 0.0,
    },
    "report": {"html": True},
}


def _deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand(node):
    """Expand $VAR and a leading ~ in every string in the config.

    A shared lab config names paths like /local_disk/$USER/DVS128Gesture so one
    file serves every account. Python does not expand that: open() takes the
    dollar sign literally and reports a missing directory whose name contains
    "$USER", which reads like a typo rather than a missing expansion. Doing it
    once here means every consumer downstream receives a real path, so no call
    site has to remember. os.path.expandvars leaves an unset variable in place
    rather than replacing it with an empty string, so a genuine mistake stays
    visible instead of collapsing to the filesystem root.
    """
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    if isinstance(node, str):
        return os.path.expanduser(os.path.expandvars(node))
    return node


def apply_train_overrides(cfg, train):
    """Copy config values onto a TrainSpec, in one place.

    `summary` and `single` each construct their own TrainSpec. An override
    applied in one and missed in the other makes the summary describe a run
    that will never happen, which is the opposite of what a cheap preview is
    for. Both call this, so the two cannot disagree.
    """
    train.epochs = cfg["search"].get("epochs", train.epochs)
    return train


def load(path=None, overrides=None):
    """Read a config file, merge over the defaults, and sanity-check it."""
    raw = {}
    if path:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise SystemExit(f"config file not found: {path}")
        raw = _parse(path)
    cfg = _deep_merge(DEFAULTS, raw)
    cfg = _deep_merge(cfg, overrides or {})
    cfg = _expand(cfg)
    _validate(cfg)
    from .hardware import set_limits
    set_limits(cfg.get("limits"))
    cfg["_source"] = path
    return cfg


def _parse(path):
    text = open(path, encoding="utf-8").read()
    if path.endswith((".json",)):
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        return _mini_yaml(text)


def _mini_yaml(text):
    """A deliberately small YAML subset: nested maps, scalars, inline lists.

    Enough for the shipped configs so that `summary` works before anyone has
    installed PyYAML. Anything more elaborate raises rather than guessing.
    """
    # Two statements, not one tuple assignment: the right-hand side of
    # `root, stack = {}, [(-1, root)]` is evaluated before either name is
    # bound, so the `root` inside the list refers to nothing yet.
    root = {}
    stack = [(-1, root)]
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if ":" not in line:
            raise SystemExit(
                f"{path_hint()}:{lineno}: cannot parse {raw.strip()!r} without PyYAML.\n"
                "  pip install pyyaml")
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        val = val.strip()
        if val == "":
            node = {}
            parent[key.strip()] = node
            stack.append((indent, node))
        else:
            parent[key.strip()] = _scalar(val)
    return root


def path_hint():
    return "config"


def _scalar(v):
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_scalar(p.strip()) for p in inner.split(",")] if inner else []
    low = v.lower()
    if low in ("null", "none", "~"):
        return None
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _validate(cfg):
    obj = cfg["objective"]
    mode = (obj.get("mode") or "accuracy").lower()
    if mode not in ("accuracy", "constrained", "weighted", "pareto"):
        raise SystemExit(
            f"objective.mode is {mode!r}; expected accuracy, constrained, weighted or pareto.\n"
            "  accuracy     ignore SynOps\n"
            "  constrained  maximize accuracy subject to synops_budget\n"
            "  weighted     accuracy - weight * synops/synops_reference\n"
            "  pareto       optimize both, choose from the front afterwards")
    if mode == "constrained" and not obj.get("synops_budget"):
        raise SystemExit("objective.mode is 'constrained' but no objective.synops_budget was set.")
    if mode == "weighted" and not obj.get("weight"):
        raise SystemExit(
            "objective.mode is 'weighted' but objective.weight is 0, which is the\n"
            "  same as 'accuracy'. Set a weight, or use 'pareto' to defer the choice.")
    k, i = cfg["search"].get("val_folds") or 0, cfg["search"].get("val_fold") or 0
    if not isinstance(k, int) or k < 0 or k == 1:
        raise SystemExit(f"search.val_folds must be 0 (one split) or 2 or more, got {k!r}")
    if k and (not isinstance(i, int) or not 0 <= i < k):
        raise SystemExit(f"search.val_fold must be 0 to {k - 1} "
                         f"(search.val_folds is {k}), got {i!r}")
    gf = cfg["search"].get("gpu_fraction", 1.0)
    if not (0 < gf <= 1):
        raise SystemExit(f"search.gpu_fraction must be in (0, 1], got {gf}")


def describe(cfg):
    """A short human summary, for the top of a log or a report."""
    d, e, s, o = cfg["dataset"], cfg["encoding"], cfg["search"], cfg["objective"]
    src = d.get("module") or d.get("name")
    lines = [
        f"dataset   : {src}  root={d.get('root')}",
        f"encoding  : {e.get('coding') or 'auto'}  T={e.get('T')}  resize_to={e.get('resize_to')}",
        f"search    : {s['trials']} trials x {s['epochs']} epochs  space={s['space']}"
        f"  gpu_fraction={s['gpu_fraction']}",
        f"objective : {o['mode']}"
        + (f"  budget={o['synops_budget']}" if o["mode"] == "constrained" else "")
        + (f"  weight={o['weight']}" if o["mode"] == "weighted" else ""),
    ]
    return "\n".join(lines)
