"""Datasets that ship with the tool.

DVS128 Gesture is the one the project was built around. The torchvision
entries exist so that the generalization is exercised rather than merely
claimed: if CIFAR-10 does not run, the abstraction is wrong.
"""

import os

from .base import DatasetBundle, register
from .._torch import _require_torch, pad_sequence_collate


@register("dvs128")
def dvs128_gesture(root, T=16, **_ignored):
    """DVS128 Gesture: 11 hand gestures from a 128x128 event camera.

    Frames are built by splitting each recording into T groups of EQUAL EVENT
    COUNT rather than equal duration, so every frame carries a similar amount
    of evidence. The cost is that a frame's wall-clock span varies with how
    fast the gesture was performed.
    """
    _require_torch()
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    root = validate_dvs_root(root)
    train = DVS128Gesture(root=root, frames_number=T, split_by="number",
                          train=True, data_type="frame")
    test = DVS128Gesture(root=root, frames_number=T, split_by="number",
                         train=False, data_type="frame")
    return DatasetBundle(
        train=train, test=test, C=2, H=128, W=128, num_classes=11,
        name="dvs128_gesture", is_event=True, collate_fn=pad_sequence_collate,
        meta={"split_by": "number", "T": T, "root": root},
    )


def _vision(name, cls_name, C, H, W, num_classes):
    """Shared body for the torchvision image datasets."""
    @register(name)
    def _provider(root, download=True, **_ignored):
        _require_torch()
        import torchvision
        import torchvision.transforms as tt

        transform = tt.Compose([tt.ToTensor()])       # keep in [0, 1] for the encoders
        cls = getattr(torchvision.datasets, cls_name)
        root = os.path.abspath(os.path.expanduser(root))
        train = cls(root=root, train=True, download=download, transform=transform)
        test = cls(root=root, train=False, download=download, transform=transform)
        return DatasetBundle(train=train, test=test, C=C, H=H, W=W,
                             num_classes=num_classes, name=name, is_event=False,
                             meta={"root": root, "source": "torchvision"})
    return _provider


cifar10 = _vision("cifar10", "CIFAR10", 3, 32, 32, 10)
cifar100 = _vision("cifar100", "CIFAR100", 3, 32, 32, 100)
mnist = _vision("mnist", "MNIST", 1, 28, 28, 10)
fashion_mnist = _vision("fashion_mnist", "FashionMNIST", 1, 28, 28, 10)


def _class_dirs_have_npz(split_dir, num_classes=11):
    """Require at least one sample file in every expected class directory."""
    if not os.path.isdir(split_dir):
        return False
    for c in range(num_classes):
        class_dir = os.path.join(split_dir, str(c))
        if not os.path.isdir(class_dir):
            return False
        if not any(f.endswith(".npz") and os.path.isfile(os.path.join(class_dir, f))
                   for f in os.listdir(class_dir)):
            return False
    return True


def frame_cache_is_complete(root, T):
    """True when both splits of the T-frame cache have samples in every class.

    Presence of the directory is not enough. A build killed partway leaves the
    folder there with some classes missing, and spikingjelly treats an existing
    folder as done, so training would silently run on a fraction of the data.
    """
    base = os.path.join(root, f"frames_number_{int(T)}_split_by_number")
    return (_class_dirs_have_npz(os.path.join(base, "train"))
            and _class_dirs_have_npz(os.path.join(base, "test")))


def warmup_frame_cache(root, T_values, log=print):
    """Build the frame cache for each T, sequentially, before anything parallel.

    Event clips are cached as a fixed number of frames, and the cache for a
    given T is built on first use: pure numpy, single-threaded, tens of minutes.
    Caching depends only on (frames_number, split_by), since transforms are
    applied per sample at load time, so warming T alone is sufficient.

    Doing this up front matters because trials run as separate processes. Eight
    of them sampling the same unbuilt T would race to write one shared folder
    and can leave it corrupted, which then reads as a complete cache.
    """
    import shutil

    _require_torch()
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

    root = validate_dvs_root(root)
    for T in sorted({int(t) for t in T_values}):
        if frame_cache_is_complete(root, T):
            log(f"[warmup] frame cache for T={T} already complete")
            continue
        stale = os.path.join(root, f"frames_number_{T}_split_by_number")
        if os.path.isdir(stale):
            log(f"[warmup] frame cache for T={T} incomplete, removing {stale}")
            # A failed removal must stop warmup: spikingjelly otherwise reuses
            # the surviving directory instead of rebuilding it.
            shutil.rmtree(stale)
        log(f"[warmup] building frame cache for T={T}. "
            "CPU only, tens of minutes, one time.")
        for train in (True, False):
            DVS128Gesture(root=root, frames_number=T, split_by="number",
                          train=train, data_type="frame")
        if not frame_cache_is_complete(root, T):
            raise SystemExit(
                f"frame cache for T={T} is still incomplete after building it.\n"
                f"  {stale}\n"
                "  Check free space on that volume and run "
                "tools/build_cache.py by hand.")
        log(f"[warmup] frame cache for T={T} ready")
    return root


def validate_dvs_root(data_dir: str) -> str:
    """Fail early and legibly on a bad DVS root.

    spikingjelly calls os.mkdir(root/'download') without creating parents, so a
    missing root surfaces as a bare FileNotFoundError from deep inside the
    library, naming a path the user never typed.
    """
    if not data_dir or data_dir.strip(".") == "":
        raise SystemExit(f"dataset root is a placeholder, not a path: {data_dir!r}")
    abs_dir = os.path.abspath(os.path.expanduser(data_dir))
    if not os.path.isdir(abs_dir):
        raise SystemExit(
            f"dataset root does not exist:\n    {abs_dir}\n\n"
            "  It must be the DVS128 Gesture ROOT folder, containing:\n"
            "      <root>/download/DvsGesture.tar.gz\n"
            "      <root>/download/gesture_mapping.csv\n"
            "  spikingjelly builds extract/, events_np/ and frames_number_*/ beside them.")
    if os.path.isdir(os.path.join(abs_dir, "extract")) or \
       os.path.isdir(os.path.join(abs_dir, "events_np")):
        return abs_dir                       # caches exist; the archive is spent
    # spikingjelly wants four files, not just the archive, and reports a missing
    # one from deep inside the library as "does not exist or is corrupted". Ask
    # it for the list rather than hard-coding a guess that will drift.
    dl = os.path.join(abs_dir, "download")
    try:
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        wanted = [r[0] for r in DVS128Gesture.resource_url_md5()]
    except Exception:
        wanted = ["DvsGesture.tar.gz"]

    missing = [n for n in wanted if not os.path.isfile(os.path.join(dl, n))]
    if missing:
        try:
            found = ", ".join(sorted(os.listdir(dl))) or "(empty)"
        except OSError:
            found = "(no download/ directory)"
        raise SystemExit(
            f"dataset root is missing files spikingjelly requires:\n    {abs_dir}\n\n"
            f"  Missing from download/: {', '.join(missing)}\n"
            f"  Present:                {found}\n\n"
            "  All of them come from the same Box folder, and IBM gates it behind\n"
            "  a click-through, so none can be fetched automatically:\n"
            "      https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8\n\n"
            "  If your root is one level deeper, point at the inner one.")
    return abs_dir
