"""Build the DVS128 Gesture frame cache without touching the GPU.

The event-to-frame conversion is the slow, one-time part of setting up: pure
numpy, single-threaded, tens of minutes per value of T. `search` mode warms it
before starting Ray, and `single` builds it lazily on first access, but both of
those also want a GPU.

Running it separately means the preprocessing can happen while the card is busy
with someone else's job, so the machine is not idle in either direction.

    python tools/build_cache.py /local_disk/$USER/DVS128Gesture
    python tools/build_cache.py /local_disk/$USER/DVS128Gesture 8 16
"""
import importlib.util
import os
import sys
import time


def load_practice(root):
    path = os.path.join(root, "Practice2.py")
    if not os.path.isfile(path):
        sys.exit(f"Practice2.py not found next to tools/: {path}")
    spec = importlib.util.spec_from_file_location("practice2", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["practice2"] = mod          # dataclasses needs it importable
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data_dir = sys.argv[1]
    T_values = [int(a) for a in sys.argv[2:]] or None

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p2 = load_practice(repo)

    if not getattr(p2, "_HAS_TORCH", False):
        sys.exit("torch and spikingjelly are required to build the cache.\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
                 "  pip install spikingjelly")

    data_dir = p2.validate_data_dir(data_dir)
    if T_values is None:
        T_values = list(getattr(p2, "T_CHOICES", [16]))

    print(f"dataset : {data_dir}")
    print(f"T values: {T_values}")
    print("This is CPU-only. It is safe to run while the GPU is busy.\n")

    start = time.time()
    p2.warmup_dataset_cache(data_dir, T_values)
    mins = (time.time() - start) / 60

    print(f"\ndone in {mins:.1f} min")
    for T in T_values:
        built = p2._frame_cache_is_complete(data_dir, T)
        print(f"  T={T:<3} cache complete: {built}")
    print("\nThe GPU run will now skip straight to training.")


if __name__ == "__main__":
    main()
