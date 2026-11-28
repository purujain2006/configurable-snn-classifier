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
    flip     mirror left-right, half the time. With DatasetBundle.hflip_ok
             every label stays. With DatasetBundle.hflip_labels each mirrored
             clip takes its class's mirror label (DVS128 Gesture: right hand
             wave becomes left hand wave), and a class mapped to -1 is never
             mirrored
    time     keep aug_time_keep of a clip's frames, a random set per clip in
             their original order (random temporal deletion, SEW ResNet), so
             the network sees a gesture done faster or with frames missing.
             Last, so the clip is shorter than T from here on

Training only. Validation, the deployed network and the simulator check all
see the inputs exactly as encoded, every frame of them.
"""

from ._torch import torch, F


def mirror_labels(text):
    """InputSpec.hflip_labels ("0,2,1,...") as a tuple of ints, or None."""
    text = (text or "").strip()
    return tuple(int(c) for c in text.split(",")) if text else None


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


def _mirror(x, y, labels):
    """Mirror half the clips whose class has a mirror and relabel them."""
    to = torch.as_tensor(labels, dtype=y.dtype, device=y.device)[y]
    flip = (torch.rand(y.shape[0], device=y.device) < 0.5) & (to >= 0)
    x = torch.where(flip.to(x.device)[:, None, None, None, None], x.flip(-1), x)
    return x, torch.where(flip, to, y)


def _time_keep(x, keep):
    N, T = x.shape[:2]
    k = max(1, min(T, int(round(keep * T))))
    if k >= T:
        return x
    order = torch.rand(N, T, device=x.device).argsort(1)[:, :k].sort(1).values   # (N, k)
    return x[torch.arange(N, device=x.device)[:, None], order]


def augment_batch(x, train_cfg, hflip_ok=False, y=None, hflip_labels=None):
    """x: (N, T, C, H, W) training batch -> the augmented batch, or (batch,
    labels) when the labels y are given. A mirror can change labels, so it
    only runs with y; temporal deletion returns fewer than T frames."""
    if not getattr(train_cfg, "augment", False):
        return x if y is None else (x, y)
    if getattr(train_cfg, "aug_flip", False):
        if hflip_labels and y is not None:
            x, y = _mirror(x, y, hflip_labels)
        elif hflip_ok:
            x = _flip(x)
    x = _shift(x, float(getattr(train_cfg, "aug_shift", 0.0) or 0.0))
    x = _cutout(x, float(getattr(train_cfg, "aug_cutout", 0.0) or 0.0))
    x = _drop(x, float(getattr(train_cfg, "aug_drop", 0.0) or 0.0))
    keep = float(getattr(train_cfg, "aug_time_keep", 0.0) or 0.0)
    if 0.0 < keep < 1.0:
        x = _time_keep(x, keep)
    return x if y is None else (x, y)
