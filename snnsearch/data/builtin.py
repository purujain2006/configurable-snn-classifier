"""Datasets that ship with the tool.

DVS128 Gesture is the one the project was built around. The torchvision
entries exist so that the generalization is exercised rather than merely
claimed: if CIFAR-10 does not run, the abstraction is wrong.
"""

import os

from .base import DatasetBundle, register
from .._torch import _HAS_TORCH, _require_torch, pad_sequence_collate


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
