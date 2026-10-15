"""Train small networks briefly, then check each one on the simulator.

    python tools/checks/train_and_verify.py /local_disk/$USER/DVS128Gesture

Run this before a search. It answers one question: does a network trained by
this code predict, clip for clip, what simpleSim's arithmetic (IntegerSim)
predicts? It trains one small network for every combination of head
(flatten, GAP) and bias mode (threshold, bias neurons), a few epochs each,
and prints the per-trial simulator check the search records.

Every line should end in "disagreements 0". Anything else means training and
deployment have drifted apart, and a search run now would select on a number
the chip does not reproduce.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


# A small network that the chip can hold: a pooled layer, then a strided one.
BASE = {"N": 16, "T": 16, "resize_to": 32, "data_dir": ".",
        "depth": 2, "k_0": 3, "ch_0": 16, "ds_0": "pool", "pool_0": 1,
        "k_1": 3, "ch_1": 24, "ds_1": "stride", "stride_1": 2,
        "fc_layers": 0, "tau": 4, "trainable_tau": True, "trainable_threshold": True,
        "dropout_rate": 0.1, "norm": "bn", "label_smoothing": 0.1, "grad_clip": 1.0,
        "optimizer": "adam", "lr": 2e-3, "weight_decay": 1e-4, "scheduler": "cosine",
        "warmup_epochs": 0, "pool_type": "avg"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", help="DVS128 Gesture root")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    import torch
    from snnsearch.spaces import config_to_specs
    from snnsearch.hardware import check_spec_feasibility
    from snnsearch.data.builtin import dvs128_gesture
    from snnsearch.data.loaders import build_dataloaders
    from snnsearch.encoders import build_encoder
    from snnsearch.train import run_training

    bundle = dvs128_gesture(args.data_root, T=BASE["T"])
    enc = build_encoder("passthrough", T=BASE["T"], resize_to=BASE["resize_to"])
    loaders = build_dataloaders(bundle, batch_size=BASE["N"], encoder=enc, seed=1,
                                val_split="subject", num_workers=4)

    failures = 0
    for head in ("flatten", "gap"):
        for bias_mode in ("threshold", "neuron"):
            cfg = config_to_specs({**BASE, "epochs": args.epochs,
                                   "final_reduction": head, "bias_mode": bias_mode})
            ok, why = check_spec_feasibility(cfg)
            t0 = time.time()
            res = run_training(cfg, loaders=loaders)
            fine = res["sim_disagreements"] == 0 and not res["sim_reason"]
            failures += not fine
            print(f"head {head:7s} bias {bias_mode:9s} fits chip {ok!s:5s}  "
                  f"pytorch {res['hw_val_accuracy']:.4f}  simulator {res['sim_val_accuracy']}  "
                  f"tau {res['deployed_tau']}  ({time.time() - t0:.0f}s)  "
                  f"disagreements {res['sim_disagreements']}"
                  + (f"  REASON {res['sim_reason']}" if res["sim_reason"] else ""),
                  flush=True)
    print("\nALL MATCH" if not failures else f"\n{failures} CONFIGURATION(S) DO NOT MATCH")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
