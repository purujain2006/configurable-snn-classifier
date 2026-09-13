"""Test cache completeness and sequential warmup without the real dataset.

    python tools/test_frame_cache.py

Only temporary directories are changed; the dataset constructor is mocked.
"""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snnsearch.data import builtin  # noqa: E402


class FrameCacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="snnsearch-frame-cache-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        # Make the temporary root valid without requiring download archives.
        (self.root / "events_np").mkdir()
        self.logs = []
        self.build = Mock(side_effect=self._build_split)
        dataset_module = ModuleType("spikingjelly.datasets.dvs128_gesture")
        dataset_module.DVS128Gesture = self.build
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(patch.dict(sys.modules, {
            "spikingjelly.datasets.dvs128_gesture": dataset_module,
        }))
        patches.enter_context(patch.object(builtin, "_require_torch"))

    def _cache(self, T):
        target = self.root / f"frames_number_{int(T)}_split_by_number"
        self.assertEqual(target.resolve().parent, self.root)
        return target

    def _write_split(self, T, split, num_classes=11):
        for c in range(num_classes):
            class_dir = self._cache(T) / split / str(c)
            class_dir.mkdir(parents=True, exist_ok=True)
            (class_dir / "sample.npz").write_bytes(b"mock sample")

    def _write_cache(self, T):
        for split in ("train", "test"):
            self._write_split(T, split)

    def _build_split(self, *, root, frames_number, split_by, train, data_type):
        self.assertEqual(Path(root), self.root)
        self.assertEqual(split_by, "number")
        self.assertEqual(data_type, "frame")
        self._write_split(frames_number, "train" if train else "test")

    def _warm(self, T_values):
        return builtin.warmup_frame_cache(str(self.root), T_values, self.logs.append)

    def test_missing_cache_and_missing_class_are_incomplete(self):
        self.assertFalse(builtin.frame_cache_is_complete(self.root, 8))
        self._write_split(8, "train")
        self.assertFalse(builtin.frame_cache_is_complete(self.root, 8))
        self._write_split(8, "test", num_classes=10)
        self.assertFalse(builtin.frame_cache_is_complete(self.root, 8))

    def test_requires_sample_files_in_every_class_of_both_splits(self):
        self._write_cache(8)
        self.assertTrue(builtin.frame_cache_is_complete(self.root, "8"))
        sample = self._cache(8) / "test" / "10" / "sample.npz"
        sample.unlink()
        (sample.parent / "notes.txt").write_text("not a sample", encoding="utf-8")
        self.assertFalse(builtin.frame_cache_is_complete(self.root, 8))
        sample.mkdir()
        self.assertFalse(builtin.frame_cache_is_complete(self.root, 8))

    def test_complete_cache_is_skipped(self):
        self._write_cache(8)
        sentinel = self._cache(8) / "preserve.txt"
        sentinel.write_text("complete cache", encoding="utf-8")
        self.assertEqual(self._warm([8]), str(self.root))
        self.build.assert_not_called()
        self.assertTrue(sentinel.is_file())
        self.assertTrue(any("already complete" in message for message in self.logs))

    def test_missing_caches_build_sequentially_once_per_distinct_T(self):
        self._warm([16, "8", 16, 8])
        self.assertEqual(
            [(call.kwargs["frames_number"], call.kwargs["train"])
             for call in self.build.call_args_list],
            [(8, True), (8, False), (16, True), (16, False)],
        )
        for T in (8, 16):
            self.assertTrue(builtin.frame_cache_is_complete(self.root, T))

    def test_partial_cache_is_removed_before_rebuilding_only_that_T(self):
        self._write_split(8, "train", num_classes=3)
        stale = self._cache(8) / "partial.txt"
        stale.write_text("partial cache", encoding="utf-8")
        self._write_cache(16)
        untouched = self._cache(16) / "preserve.txt"
        untouched.write_text("other T", encoding="utf-8")

        def rebuild(**kwargs):
            if kwargs["train"]:
                self.assertFalse(self._cache(8).exists())
            self._build_split(**kwargs)

        self.build.side_effect = rebuild
        self._warm([8])
        self.assertFalse(stale.exists())
        self.assertTrue(untouched.is_file())
        self.assertTrue(builtin.frame_cache_is_complete(self.root, 8))

    def test_incomplete_rebuild_raises_before_the_next_T(self):
        self.build.side_effect = lambda **kwargs: None
        with self.assertRaisesRegex(SystemExit, "T=8 is still incomplete"):
            self._warm([8, 16])
        self.assertEqual(self.build.call_count, 2)
        self.assertFalse(any("ready" in message for message in self.logs))

    def test_failed_removal_stops_before_dataset_construction(self):
        self._write_split(8, "train", num_classes=3)
        with patch("shutil.rmtree", side_effect=PermissionError("cache is locked")):
            with self.assertRaisesRegex(PermissionError, "cache is locked"):
                self._warm([8])
        self.build.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
