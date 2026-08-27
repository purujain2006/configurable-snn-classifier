"""Check whether this machine can run Practice2.py, and say what is missing.

Run it first on a fresh lab account. It imports nothing that is not already
required, reports per-mode readiness, and never raises: a missing package is a
finding, not a crash.

    python tools/check_env.py                      # environment only
    python tools/check_env.py /path/to/DVS128      # also check the dataset
"""
import importlib
import os
import shutil
import sys

OK, WARN, BAD = "  ok  ", " warn ", " MISS "


def line(state, name, detail=""):
    print(f"[{state}] {name:<26} {detail}")


def probe(mod, purpose):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "")
        line(OK, mod, f"{v}  {purpose}")
        return True
    except Exception as exc:
        line(BAD, mod, f"{purpose}  ->  {type(exc).__name__}")
        return False


def main():
    print("=" * 74)
    print("Practice2.py environment check")
    print("=" * 74)

    # ---- interpreter -------------------------------------------------------
    v = sys.version_info
    py_ok = v >= (3, 9)
    line(OK if py_ok else BAD, "python", f"{v.major}.{v.minor}.{v.micro}  (needs >= 3.9)")
    print(f"        interpreter: {sys.executable}")
    if "conda" not in sys.executable and "envs" not in sys.executable:
        line(WARN, "conda env", "does not look like a conda env; did you `conda activate myenv`?")

    # ---- packages ----------------------------------------------------------
    print("\n-- packages --")
    numpy = probe("numpy", "always required")
    torch_ok = probe("torch", "everything except `summary`")
    sj = probe("spikingjelly", "dataset + neuron library")
    ray_ok = probe("ray", "`search` mode only")
    optuna_ok = probe("optuna", "`search` mode only")
    try:
        importlib.import_module("hs_api")
        line(OK, "hs_api", "real converter (optional)")
        hs = True
    except Exception:
        line(WARN, "hs_api", "absent; Practice2.py mirrors the arithmetic itself, so this is fine")
        hs = False

    if sj:
        try:
            importlib.import_module("spikingjelly.activation_based")
            line(OK, "spikingjelly API", "activation_based present")
        except Exception:
            line(BAD, "spikingjelly API", "too old: needs the activation_based API (>= 0.0.0.0.14)")
            sj = False

    # ---- compute -----------------------------------------------------------
    print("\n-- compute --")
    if torch_ok:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            line(OK, "CUDA", f"{n} device(s): " + ", ".join(
                torch.cuda.get_device_name(i) for i in range(n)))
        else:
            line(WARN, "CUDA", "not available -- torch is CPU-only")
            print("        The setup guide installs `cpuonly`. Training runs T timesteps")
            print("        per sample, so CPU is roughly one to two orders of magnitude")
            print("        slower here. Check `nvidia-smi`; if the machine has a GPU,")
            print("        reinstall torch with a CUDA build.")
    if shutil.which("nvidia-smi"):
        line(OK, "nvidia-smi", "present, so the machine has NVIDIA drivers")
    else:
        line(WARN, "nvidia-smi", "not found; this box may genuinely have no GPU")

    # ---- disk --------------------------------------------------------------
    print("\n-- disk --")
    user = os.environ.get("USER", "")
    for path in [os.path.expanduser("~"), f"/local_disk/{user}" if user else "/local_disk"]:
        if os.path.isdir(path):
            try:
                free = shutil.disk_usage(path).free / 2**30
                state = OK if free > 40 else WARN
                line(state, path, f"{free:.0f} GiB free")
            except OSError:
                line(WARN, path, "unreadable")
        else:
            line(WARN, path, "does not exist")
    print("        The frame cache is tens of GiB. Keep the dataset off your home")
    print("        directory if it is a small network mount.")

    # ---- dataset -----------------------------------------------------------
    print("\n-- dataset --")
    root = sys.argv[1] if len(sys.argv) > 1 else None
    if not root:
        line(WARN, "--data-dir", "not given; pass the DVS128 root to check it")
    else:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            line(BAD, "root", f"{root} does not exist")
        else:
            line(OK, "root", root)
            tar = os.path.join(root, "download", "DvsGesture.tar.gz")
            csvf = os.path.join(root, "download", "gesture_mapping.csv")
            built = [d for d in ("extract", "events_np") if os.path.isdir(os.path.join(root, d))]
            frames = sorted(d for d in os.listdir(root) if d.startswith("frames_number_"))
            line(OK if os.path.isfile(tar) else (WARN if built else BAD), "DvsGesture.tar.gz",
                 "present" if os.path.isfile(tar) else
                 ("absent, but caches exist so it is no longer needed" if built else
                  "absent -- download it manually, spikingjelly cannot fetch it"))
            line(OK if os.path.isfile(csvf) else WARN, "gesture_mapping.csv",
                 "present" if os.path.isfile(csvf) else "absent")
            line(OK if built else WARN, "caches",
                 ", ".join(built) if built else "none yet; the first run builds them")
            line(OK if frames else WARN, "frame caches",
                 ", ".join(frames) if frames else "none yet (one per value of T)")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 74)
    modes = [
        ("summary", py_ok, "architecture table + feasibility, no GPU or dataset needed"),
        ("single", py_ok and torch_ok and sj, "train one configuration"),
        ("fold", py_ok and torch_ok and sj, "fold, quantize, verify, export"),
        ("search", py_ok and torch_ok and sj and ray_ok and optuna_ok, "the automated search"),
    ]
    for name, ready, what in modes:
        line(OK if ready else BAD, f"mode: {name}", what)
    if not all(m[1] for m in modes):
        print("\nTo finish the setup inside myenv:")
        if not sj:
            print("    pip install spikingjelly")
        if not (ray_ok and optuna_ok):
            print('    pip install "ray[tune]" optuna')
        if not hs:
            print("    # hs_api only if you want the real converter; optional")
    print()


if __name__ == "__main__":
    main()
