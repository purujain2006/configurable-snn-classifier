"""A tiny synthetic dataset, for exercising the pipeline without real data.

Two classes of noisy blobs, small enough that a full run finishes in under a
minute on a CPU. Nothing here is meant to learn anything useful: the point is
to drive every stage of the pipeline (build, train, fold, quantize, audit,
export, report) so a broken stage shows up in seconds rather than after a
dataset download and a GPU allocation.

    python tools/smoke_test.py
"""

import torch
from torch.utils.data import TensorDataset

from snnsearch.data.base import DatasetBundle


def make_datasets(root=None, n_train=48, n_test=16, C=2, H=16, W=16,
                  num_classes=2, seed=0):
    g = torch.Generator().manual_seed(seed)

    def build(n):
        y = torch.randint(0, num_classes, (n,), generator=g)
        x = torch.rand(n, C, H, W, generator=g) * 0.3
        # Give each class a bright quadrant, so the task is learnable and a
        # collapsed model is distinguishable from a working one.
        for i, cls in enumerate(y.tolist()):
            r = (cls // 2) * (H // 2)
            c = (cls % 2) * (W // 2)
            x[i, :, r:r + H // 2, c:c + W // 2] += 0.7
        return TensorDataset(x.clamp(0, 1), y)

    return DatasetBundle(
        train=build(n_train), test=build(n_test),
        C=C, H=H, W=W, num_classes=num_classes,
        name="synthetic", is_event=False,
        meta={"purpose": "smoke test", "seed": seed},
    )
