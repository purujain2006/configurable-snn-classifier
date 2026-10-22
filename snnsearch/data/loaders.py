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

  num_workers    Was pinned to 0, on the reasoning that each trial is already
                 its own process holding one CPU, so workers would nest
                 multiprocessing and oversubscribe. That reasoning depends
                 entirely on the "one CPU" part. On a 384-core box running 16
                 trials, 368 cores sit idle while each trial decodes frames
                 single-threaded and the GPU waits. It is a setting now, and
                 the scheduler reserves num_workers + 1 CPUs per trial so the
                 oversubscription the original comment feared cannot happen.
"""

import os
import random
import re

from .._torch import _require_torch, torch, DataLoader, Subset
from .base import DatasetBundle


_SUBJECT = re.compile(r"user(\d+)", re.IGNORECASE)


def subject_ids(dataset):
    """Subject per sample, read from the file names, or None if unavailable.

    DVS128 Gesture names every clip after its recording, e.g.
    user07_fluorescent_led_3.npz, and spikingjelly's DatasetFolder keeps the
    paths in `.samples`.
    """
    samples = getattr(dataset, "samples", None)
    if not samples:
        return None
    ids = []
    for s in samples:
        path = s[0] if isinstance(s, (tuple, list)) else s
        m = _SUBJECT.search(os.path.basename(str(path)))
        if m is None:
            return None
        ids.append(int(m.group(1)))
    return ids


def split_settings(cfg):
    """The split_indices arguments a run config names, so every tool that
    rebuilds a run's validation set reads the same settings."""
    s = cfg["search"]
    return {"seed": cfg["run"].get("seed", 1), "val_split": s.get("val_split", "subject"),
            "folds": int(s.get("val_folds") or 0), "fold": int(s.get("val_fold") or 0)}


def subject_folds(ids, folds, seed=1):
    """`folds` lists of people, together every person once, with near-equal
    clip counts.

    The people are shuffled by `seed` and dealt out one at a time, each to the
    fold with the fewest clips so far. People differ in clip count (DVS128
    Gesture has 2 to 5 recordings each), so that can leave folds a person
    apart; moving one person to a smaller fold, or swapping two, while that
    evens the sizes out, closes most of it.
    """
    count = {}
    for s in ids:
        count[s] = count.get(s, 0) + 1
    people = sorted(count)
    if not 2 <= folds <= len(people):
        raise SystemExit(f"search.val_folds is {folds}; it needs 2 to {len(people)} "
                         f"(this training set has {len(people)} people)")
    random.Random(seed).shuffle(people)
    groups, sizes = [[] for _ in range(folds)], [0] * folds
    for p in people:
        j = min(range(folds), key=lambda g: (sizes[g], g))
        groups[j].append(p)
        sizes[j] += count[p]
    while True:
        # Moving d clips from fold i to a fold `gap` clips smaller lowers the
        # sum of squared sizes by 2 d (gap - d), so it helps when 0 < d < gap,
        # and as the sum falls with every move, this ends.
        best = None
        for i in range(folds):
            for j in range(folds):
                gap = sizes[i] - sizes[j]
                for a in groups[i]:
                    for b in groups[j] + [None]:
                        d = count[a] - (count[b] if b is not None else 0)
                        if 0 < d < gap and (best is None or d * (gap - d) > best[0]):
                            best = (d * (gap - d), i, j, a, b)
        if best is None:
            return [sorted(g) for g in groups]
        _, i, j, a, b = best
        groups[i].remove(a)
        groups[j].append(a)
        sizes[i] -= count[a]
        sizes[j] += count[a]
        if b is not None:
            groups[j].remove(b)
            groups[i].append(b)
            sizes[j] -= count[b]
            sizes[i] += count[b]


def split_indices(dataset, val_fraction=0.15, seed=1, val_split="subject", folds=0, fold=0):
    """(train_idx, val_idx, held_out_subjects) for carving val out of train.

    "subject" holds out whole people. The test set is six people absent from
    training, so validation should be people absent from training too.
    Splitting by clip puts every subject on both sides, which makes validation
    a memorization score: rank1 read 0.92 on it and 0.83 on test.

    "clip" is the old split, kept so checkpoints scored with it can be
    re-scored on the same clips.

    `folds` = k > 0 is k-fold cross-subject validation (search.val_folds): the
    people are split into k groups by subject_folds and group `fold` is
    validation. One fixed set of people can be easier or harder than average,
    which moves every score it gives; over k folds every training person is
    held out once.
    """
    n = len(dataset)
    if folds:
        if val_split != "subject":
            raise SystemExit("search.val_folds holds out people, so it needs "
                             f"search.val_split 'subject', not {val_split!r}")
        ids = subject_ids(dataset)
        if ids is None:
            raise SystemExit("search.val_folds needs a person id in every clip's file "
                             "name (e.g. user07_led_3.npz), and this dataset has none; "
                             "set search.val_folds to 0")
        if not 0 <= fold < folds:
            raise SystemExit(f"search.val_fold is {fold}; with {folds} folds it is 0 to "
                             f"{folds - 1}")
        held_set = set(subject_folds(ids, folds, seed)[fold])
        train_idx = [i for i, s in enumerate(ids) if s not in held_set]
        val_idx = [i for i, s in enumerate(ids) if s in held_set]
        return train_idx, val_idx, sorted(held_set)
    if val_split == "subject":
        ids = subject_ids(dataset)
        if ids is None:
            print("loaders   : no subject ids in this dataset, falling back to a clip split")
            val_split = "clip"
    if val_split == "clip":
        torch.manual_seed(seed)
        indices = torch.randperm(n).tolist()
        n_val = int(val_fraction * n)
        return indices[n_val:], indices[:n_val], None
    if val_split != "subject":
        raise ValueError(f"val_split must be 'subject' or 'clip', got {val_split!r}")

    # whole subjects, in a seeded order, until val holds val_fraction of clips
    subjects = sorted(set(ids))
    random.Random(seed).shuffle(subjects)
    held, count = [], 0
    for s in subjects:
        if count >= val_fraction * n:
            break
        held.append(s)
        count += ids.count(s)
    held_set = set(held)
    train_idx = [i for i, s in enumerate(ids) if s not in held_set]
    val_idx = [i for i, s in enumerate(ids) if s in held_set]
    return train_idx, val_idx, sorted(held)


def build_dataloaders(bundle: DatasetBundle, batch_size: int, encoder=None,
                      val_fraction: float = 0.15, num_workers: int = 0,
                      seed: int = 1, val_split: str = "subject",
                      folds: int = 0, fold: int = 0):
    """Returns (train_loader, val_loader, test_loader).

    `encoder` is applied inside the collate step rather than as a dataset
    transform, so the same cached dataset serves every coding. That matters for
    event data, where the frame cache is expensive and depends only on T.
    """
    _require_torch()

    train_idx, val_idx, held = split_indices(bundle.train, val_fraction, seed, val_split,
                                             folds, fold)
    if held is not None:
        print(f"loaders   : validation = "
              + (f"fold {fold} of {folds}, " if folds else "")
              + f"subjects {held} ({len(val_idx)} clips), none of them in train")
    train_set = Subset(bundle.train, train_idx)
    val_set = Subset(bundle.train, val_idx)

    collate = _make_collate(bundle, encoder)
    kwargs = dict(batch_size=batch_size, pin_memory=True,
                  collate_fn=collate, num_workers=num_workers)
    if num_workers > 0:
        # Workers are torn down and rebuilt every epoch by default. At 40 epochs
        # and 16 trials that is 640 process spawns, each re-importing torch.
        kwargs["persistent_workers"] = True
        # Two batches queued per worker, so the GPU has something to start on
        # while the next one is still being decoded and encoded.
        kwargs["prefetch_factor"] = 2

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

    ALWAYS RETURNS THREE VALUES.  The training loop, the evaluator and the fold
    verifier all read `for x, y, lengths in loader`, because they were written
    when the only dataset was DVS128 and pad_sequence_collate always supplied a
    length. Returning a pair here works for event data and raises
    "not enough values to unpack" on every static dataset. Padding the third
    slot in one place beats editing three consumers, and keeps `lengths`
    meaningful: real per-clip lengths when the dataset supplies them, and a
    constant T when every sample genuinely has the same length.
    """
    base = bundle.collate_fn

    def collate(items):
        lengths = None
        if base is not None:
            out = base(items)
            x, y = out[0], out[1]
            if len(out) > 2:
                lengths = out[2]
        else:
            xs, ys = zip(*items)
            x = torch.stack([_as_tensor(v) for v in xs])
            y = torch.as_tensor(ys)
        x = x.float()
        if encoder is not None:
            x = encoder(x)
        if lengths is None:
            # x is (N, T, C, H, W) after encoding; before it, static input has
            # no time axis at all and every sample is one step long.
            T = x.shape[1] if x.dim() == 5 else 1
            lengths = torch.full((x.shape[0],), T, dtype=torch.long)
        return x, y, lengths

    return collate


def _as_tensor(v):
    if torch.is_tensor(v):
        return v
    return torch.as_tensor(v)
