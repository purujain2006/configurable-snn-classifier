"""The run configuration file.

WHY YAML AND NOT JSON

Every narrowed range in this project carries its evidence beside it -- the
statistics that justified it, in a comment on the same line. JSON has no
comments, so moving the search space into JSON would strip exactly the property
that makes the file trustworthy. YAML keeps it:

    channels: [32, 64]   # 128/256 were 0/14 feasible; V=0.85, p<1e-4

WHY A DATASET IS NOT IN THE FILE

A dataset is code, not values: a Dataset subclass, a transform, sometimes a
loader. The config therefore carries values and points at a Python callable for
the rest, which keeps the file declarative without pretending code can be
declared.

Falls back to a small parser when PyYAML is absent, so `summary` still runs on
a machine with nothing installed.
"""

import json
import os


DEFAULTS = {
    "run": {"name": "run", "results_dir": None, "seed": 1},
    "dataset": {"name": "dvs128", "root": None},
    "encoding": {"coding": None, "T": 16, "resize_to": None},
    "limits": {},          # overrides for AXON_LIMITS / NEURON_LIMITS
    "search": {
        "trials": 25, "epochs": 40, "grace_period": 8, "reduction_factor": 3,
        "brackets": 1, "target": 0.975, "space": "uniform",
        "gpu_fraction": 1.0, "cpu_per_trial": 1, "batch_size": 16,
        # Left unset, the search holds T at whatever `encoding` says. Listing
        # more values here searches over them, which needs a frame cache per
        # value: building those lazily inside parallel trials has several
        # processes decoding the same events into the same directory. Build
        # them first with tools/build_cache.py.
        "T_choices": None,
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
