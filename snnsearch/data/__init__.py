"""Dataset plugins.

Importing this registers the built-ins, so `build_dataset({"name": "cifar10"})`
works without the caller knowing where they live.
"""

from .base import DatasetBundle, build_dataset, register, available, load_from_file
from . import builtin  # noqa: F401  -- import registers the built-in providers

__all__ = ["DatasetBundle", "build_dataset", "register", "available", "load_from_file"]
