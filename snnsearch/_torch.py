"""One place that knows whether torch exists.

Practice2.py guarded its torch import so that `summary` -- architecture
planning and the feasibility check -- runs on a machine with no deep-learning
stack at all. Splitting into modules would normally lose that, because each
module would import torch for itself.

Instead every module imports the names from here. If torch is absent the names
are None and `_HAS_TORCH` is False, so config, planning, hardware and cost still
import cleanly and `summary` still works.
"""

_TORCH_IMPORT_ERROR = None
_SPIKINGJELLY_IMPORT_ERROR = None

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    _HAS_TORCH = True
except ImportError as _err:                      # summary/feasibility still work
    _HAS_TORCH = False
    _TORCH_IMPORT_ERROR = _err
    np = torch = F = None
    DataLoader = Subset = None

    class _NoTorchModule:
        """Placeholder base class so modules still import without torch."""

    class nn:                                    # noqa: N801 - shim, not a real class
        Module = _NoTorchModule


try:
    from spikingjelly.datasets import pad_sequence_collate
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    _HAS_SPIKINGJELLY = True
except ImportError as _err:
    _HAS_SPIKINGJELLY = False
    _SPIKINGJELLY_IMPORT_ERROR = _err
    pad_sequence_collate = neuron = functional = surrogate = None

    class layer:                                 # noqa: N801 - shim
        Conv2d = Linear = BatchNorm2d = Dropout = Flatten = None
        AdaptiveAvgPool2d = AvgPool2d = MaxPool2d = SeqToANNContainer = None


try:
    from hs_api.custom_neurons import Custom_LIFNode, Custom_IFNode
    _HAS_HS_API = True
except ImportError:
    _HAS_HS_API = False
    Custom_LIFNode = Custom_IFNode = None


def _require_torch():
    """Raise with the real cause rather than a bare NameError deeper down."""
    if not _HAS_TORCH:
        raise SystemExit(
            "This mode needs PyTorch, which failed to import:\n"
            f"    {_TORCH_IMPORT_ERROR}\n\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  (`summary` runs without it.)")
    if not _HAS_SPIKINGJELLY:
        raise SystemExit(
            "This mode needs spikingjelly, which failed to import:\n"
            f"    {_SPIKINGJELLY_IMPORT_ERROR}\n\n"
            "  pip install spikingjelly\n"
            "  Install it AFTER torch, or it may pull a different build.")


def _no_grad(fn):
    """`@torch.no_grad()` that survives being imported without torch.

    A bare @torch.no_grad() on a method of a module-level class is evaluated at
    import time, which raised when torch was absent and broke `summary`.
    """
    return torch.no_grad()(fn) if _HAS_TORCH else fn


__all__ = [
    "_HAS_TORCH", "_HAS_SPIKINGJELLY", "_HAS_HS_API",
    "_require_torch", "_no_grad",
    "np", "torch", "nn", "F", "DataLoader", "Subset",
    "pad_sequence_collate", "neuron", "functional", "surrogate", "layer",
    "Custom_LIFNode", "Custom_IFNode",
]
