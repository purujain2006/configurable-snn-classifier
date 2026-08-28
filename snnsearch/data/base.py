"""The dataset seam.

A dataset cannot be described in YAML, because it is code: a Dataset subclass,
a transform, sometimes a custom loader. So the config carries *values* and
points at a Python callable for the rest.

A provider is any callable returning a DatasetBundle. Register it by name for
the built-ins, or point the config at a file and function for your own:

    dataset:
      module: examples/cifar10_data.py
      factory: make_datasets
      root: /local_disk/$USER/cifar10

Everything downstream sees only the bundle, so the planner, the feasibility
check and the search never learn what the data is.
"""

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class DatasetBundle:
    """Everything the rest of the package needs to know about a dataset.

    `train` and `test` are torch Datasets. The shape fields drive the planner
    and the feasibility check, so they must describe one sample BEFORE any
    resize the encoder applies.

    `is_event` decides the default coding: event data passes through, static
    data needs an encoder to give it a time axis.
    """

    train: Any
    test: Any
    C: int
    H: int
    W: int
    num_classes: int
    name: str = "dataset"
    is_event: bool = False
    #: collate_fn for the DataLoader; event datasets need padding
    collate_fn: Optional[Callable] = None
    #: anything worth recording in the run report
    meta: dict = field(default_factory=dict)

    def describe(self):
        return {"name": self.name, "C": self.C, "H": self.H, "W": self.W,
                "num_classes": self.num_classes, "is_event": self.is_event,
                "train_size": _safe_len(self.train), "test_size": _safe_len(self.test),
                **self.meta}


def _safe_len(ds):
    try:
        return len(ds)
    except Exception:
        return None


_REGISTRY = {}


def register(name):
    """Decorator registering a built-in provider under `name`."""
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def available():
    return sorted(_REGISTRY)


def load_from_file(path, factory="make_datasets"):
    """Import a user's Python file and pull one callable out of it.

    Kept deliberately simple: the file is imported as a module, so it can do
    anything Python can do, and only the named function is called.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise SystemExit(f"dataset module not found: {path}")
    mod_name = "_snnsearch_userdata_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod            # dataclasses in the file need this
    spec.loader.exec_module(mod)
    if not hasattr(mod, factory):
        have = ", ".join(n for n in dir(mod) if not n.startswith("_")) or "(nothing public)"
        raise SystemExit(f"{path} has no function {factory!r}.\n  Found: {have}")
    return getattr(mod, factory)


def build_dataset(spec: dict) -> DatasetBundle:
    """Resolve a `dataset:` config block into a bundle.

    Either a registered name:      {"name": "dvs128", "root": "..."}
    or a user file:                {"module": "my.py", "factory": "make", ...}
    """
    spec = dict(spec or {})
    module = spec.pop("module", None)
    factory_name = spec.pop("factory", "make_datasets")
    name = spec.pop("name", None)

    if module:
        factory = load_from_file(module, factory_name)
    elif name in _REGISTRY:
        factory = _REGISTRY[name]
    else:
        raise SystemExit(
            f"dataset {name!r} is not registered and no module: was given.\n"
            f"  Built in: {', '.join(available())}\n"
            "  Or point at your own:\n"
            "    dataset:\n"
            "      module: examples/my_data.py\n"
            "      factory: make_datasets")

    bundle = factory(**spec)
    if not isinstance(bundle, DatasetBundle):
        raise SystemExit(
            f"{factory_name} returned {type(bundle).__name__}, expected DatasetBundle.\n"
            "  from snnsearch.data.base import DatasetBundle\n"
            "  return DatasetBundle(train=..., test=..., C=3, H=32, W=32, num_classes=10)")
    return bundle
