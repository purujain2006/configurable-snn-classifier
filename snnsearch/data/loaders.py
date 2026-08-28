"""DataLoader construction, for any dataset rather than one.

Replaces Practice2.py's `build_dataloaders`, which constructed DVS128Gesture
directly. The three settings that made the original's numbers comparable are
kept exactly, because each was there for a reason:

  SEEDED SPLIT   The validation set is carved from train after
                 torch.manual_seed(1), so every run and every search trial is
                 scored against the same held-out clips. Without it a trial
                 could look better purely by drawing an easier split, and you
                 would be comparing configurations that sat different exams.

  drop_last ON TRAIN ONLY
                 A partial final batch gives a noisier gradient and, with batch
                 normalization, batch statistics from very few samples. On
                 validation it would silently skip up to N-1 samples -- and
                 with shuffle=False, always the same trailing ones -- biasing
                 every accuracy figure reported.

  num_workers=0  Each search trial is already its own process with one CPU.
                 Worker processes would nest multiprocessing, oversubscribing
                 the CPUs and breaking on Windows.
"""

from .._torch import _require_torch, torch, DataLoader, Subset
from .base import DatasetBundle


def build_dataloaders(bundle: DatasetBundle, batch_size: int, encoder=None,
                      val_fraction: float = 0.15, num_workers: int = 0,
                      seed: int = 1):
    """Returns (train_loader, val_loader, test_loader).

    `encoder` is applied inside the collate step rather than as a dataset
    transform, so the same cached dataset serves every coding. That matters for
    event data, where the frame cache is expensive and depends only on T.
    """
    _require_torch()

    n = len(bundle.train)
    n_val = int(val_fraction * n)
    torch.manual_seed(seed)
    indices = torch.randperm(n)
    train_set = Subset(bundle.train, indices[n_val:])
    val_set = Subset(bundle.train, indices[:n_val])

    collate = _make_collate(bundle, encoder)
    kwargs = dict(batch_size=batch_size, pin_memory=True,
                  collate_fn=collate, num_workers=num_workers)

    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **kwargs),
        DataLoader(val_set, shuffle=False, drop_last=False, **kwargs),
        DataLoader(bundle.test, shuffle=False, drop_last=False, **kwargs),
    )


def _make_collate(bundle, encoder):
    """Wrap the dataset's own collate, then encode the batch.

    Event datasets need spikingjelly's padding collate because clips can differ
    in length. Static datasets use the default. Either way the encoder runs
    once per batch on the assembled tensor, which is where it belongs: doing it
    per-sample would repeat the same work T times.
    """
    base = bundle.collate_fn

    def collate(items):
        if base is not None:
            out = base(items)
            # pad_sequence_collate returns (x, y, lengths)
            x, y = out[0], out[1]
        else:
            xs, ys = zip(*items)
            x = torch.stack([_as_tensor(v) for v in xs])
            y = torch.as_tensor(ys)
        x = x.float()
        if encoder is not None:
            x = encoder(x)
        return x, y

    return collate


def _as_tensor(v):
    if torch.is_tensor(v):
        return v
    return torch.as_tensor(v)
