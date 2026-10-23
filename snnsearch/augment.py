"""Training-time augmentation for any input of shape (N, T, C, H, W).

Each transform is drawn once per sample and applied to every timestep alike,
so motion inside a clip survives (as in Li et al. 2022, "Neuromorphic Data
Augmentation for Training Spiking Neural Networks"). Nothing here knows what
the data is: an event clip and a still image repeated over T get the same
treatment.

Every output is still an input the chip can receive. Shift and flip move
values, cutout sets them to zero, and drop thins integer counts, so an input
of integer levels 0..2**B-1 stays one.

    shift    translate by up to aug_shift of the image size, zero filled
    cutout   zero one square of side aug_cutout of the image size
    drop     remove each unit of input with probability aug_drop (EventDrop
             for event counts, pixel dropout for levels)
    flip     mirror left-right, half the time, only when the dataset says a
             mirrored example keeps its label (DatasetBundle.hflip_ok)

Training only. Validation, the deployed network and the simulator check all
see the inputs exactly as encoded.
"""

from ._torch import torch, F


def _shift(x, frac):
    N, _T, _C, H, W = x.shape
    s = int(round(frac * min(H, W)))
    if s <= 0:
        return x
    padded = F.pad(x, (s, s, s, s))
    dy = torch.randint(0, 2 * s + 1, (N,)).tolist()
    dx = torch.randint(0, 2 * s + 1, (N,)).tolist()
    return torch.stack([padded[i, ..., dy[i]:dy[i] + H, dx[i]:dx[i] + W]
                        for i in range(N)])


def _cutout(x, frac):
    N, _T, _C, H, W = x.shape
    side = int(round(frac * min(H, W)))
    if side <= 0:
        return x
    cy = torch.randint(0, H, (N, 1, 1), device=x.device)
    cx = torch.randint(0, W, (N, 1, 1), device=x.device)
    ys = torch.arange(H, device=x.device).view(1, H, 1)
    xs = torch.arange(W, device=x.device).view(1, 1, W)
    half = side // 2
    hole = ((ys - cy).abs() <= half) & ((xs - cx).abs() <= half)     # (N, H, W)
    return x * (~hole).to(x.dtype)[:, None, None]


def _drop(x, p):
    if p <= 0:
        return x
    if bool((x >= 0).all()) and bool((x == x.round()).all()):
        # binomial thinning keeps counts integer. For 0/1 spikes it is plain dropout
        return torch.binomial(x, torch.full_like(x, 1.0 - p))
    # analog inputs ("direct" coding): drop whole values
    return x * (torch.rand_like(x) >= p).to(x.dtype)


def _flip(x):
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    return torch.where(flip[:, None, None, None, None], x.flip(-1), x)


def augment_batch(x, train_cfg, hflip_ok=False):
    """x: (N, T, C, H, W) training batch -> augmented batch of the same shape."""
    if not getattr(train_cfg, "augment", False):
        return x
    if getattr(train_cfg, "aug_flip", False) and hflip_ok:
        x = _flip(x)
    x = _shift(x, float(getattr(train_cfg, "aug_shift", 0.0) or 0.0))
    x = _cutout(x, float(getattr(train_cfg, "aug_cutout", 0.0) or 0.0))
    x = _drop(x, float(getattr(train_cfg, "aug_drop", 0.0) or 0.0))
    return x
