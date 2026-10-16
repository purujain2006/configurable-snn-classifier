"""Train a balanced set of networks briefly, then check each one on the simulator.

    python tools/checks/train_and_verify.py configs/dvs128_search04.yaml
    python tools/checks/train_and_verify.py configs/cifar10.yaml --trials 12 --epochs 2

Run this on the cluster before a search, with the search's own config file,
so it uses the same dataset, coding and T choices. It answers one question:
does a network trained by this code predict, clip for clip, what simpleSim's
arithmetic (IntegerSim) predicts?

The networks are the start of the search's own balanced block
(spaces.balanced_points), so every option the search can pick is covered:
pooling (none, average, max as OR neurons), head (flatten, GAP), bias
(threshold, bias neurons), signed spikes, bits per input value, how tau is
learned, the loss, augmentation, conv dropout and the firing-rate penalty.
Training settings are drawn at random like the block's.

Every line should end in "disagreements 0". Anything else means training and
deployment have drifted apart, and a search run now would select on a number
the chip does not reproduce.
"""
import argparse
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def training_settings(rng, point, hflip_ok):
    """The knobs the balanced block leaves to the sampler, drawn from the
    same ranges as DefineByRunSpace."""
    flat = {"tau": rng.choice([2, 4, 8, 16]), "trainable_threshold": rng.random() < 0.5,
            "dropout_rate": rng.uniform(0.1, 0.45), "norm": "bn",
            "label_smoothing": rng.uniform(0, 0.2), "grad_clip": 1.0,
            "optimizer": "adam", "lr": 2e-3, "weight_decay": 1e-4,
            "scheduler": "cosine", "warmup_epochs": 0}
    if point.get("use_conv_dropout"):
        flat["conv_dropout"] = rng.uniform(0.05, 0.3)
    if point.get("use_rate_penalty"):
        flat["rate_penalty"] = 10 ** rng.uniform(-4, -1)
    if point.get("use_augment"):
        flat.update(aug_shift=rng.uniform(0, 0.2), aug_cutout=rng.uniform(0, 0.4),
                    aug_drop=rng.uniform(0, 0.3), aug_flip=hflip_ok and rng.random() < 0.5)
    return flat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the search's config file")
    ap.add_argument("--trials", type=int, default=12,
                    help="networks to train (rounded up to a multiple of 6)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from snnsearch import runconfig
    from snnsearch.data import build_dataset
    from snnsearch.hardware import check_spec_feasibility
    from snnsearch.pipeline import resolve_specs, make_loaders
    from snnsearch.spaces import (balanced_points, data_facts, MULTI_BIT_CODINGS,
                                  MAX_INPUT_BITS)
    from snnsearch.train import run_training

    cfg = runconfig.load(args.config)
    s = cfg["search"]
    t_choices = s.get("T_choices") or [cfg["encoding"].get("T", 16)]
    bundle = build_dataset({**cfg["dataset"], "T": int(t_choices[0])})
    data = data_facts(bundle)
    coding = cfg["encoding"].get("coding") or ("passthrough" if bundle.is_event else "levels")
    max_bits = MAX_INPUT_BITS if coding in MULTI_BIT_CODINGS else 1
    print(f"dataset {bundle.name}: {data}, coding {coding}, input bits 1..{max_bits}")

    points = balanced_points(args.trials, seed=args.seed, t_choices=t_choices,
                             batch_size=s.get("batch_size", 16), epochs=args.epochs,
                             data=data, max_input_bits=max_bits, run_seed=args.seed)
    rng = random.Random(args.seed)
    failures = 0
    for i, point in enumerate(points):
        flat = {**point, **training_settings(rng, point, data["hflip_ok"]),
                "N": s.get("batch_size", 16), "epochs": args.epochs, "seed": args.seed}
        _bundle, enc, spec = resolve_specs(cfg, flat, quiet=True)
        ok, _why = check_spec_feasibility(spec)
        loaders = make_loaders(cfg, _bundle, enc, spec, quiet=True)
        t0 = time.time()
        res = run_training(spec, loaders=loaders)
        fine = res["sim_disagreements"] == 0 and not res["sim_reason"]
        failures += not fine
        pools = any(point.get(f"ds_{j}") == "pool" for j in range(point["depth"]))
        print(f"{i:2d} pool {point.get('pool_type', 'avg') if pools else 'none':4s} "
              f"head {point['final_reduction']:7s} bias {point['bias_mode']:9s} "
              f"signed {point['signed_spikes']!s:5s} bits {point.get('input_bits', 1)} "
              f"tau {point['tau_mode']:7s} loss {point['loss']:4s} "
              f"aug {point['use_augment']!s:5s} fits chip {ok!s:5s}  "
              f"pytorch {res['hw_val_accuracy']:.4f}  simulator {res['sim_val_accuracy']}  "
              f"tau {res['deployed_tau']}  ({time.time() - t0:.0f}s)  "
              f"disagreements {res['sim_disagreements']}"
              + (f"  REASON {res['sim_reason']}" if res["sim_reason"] else ""),
              flush=True)
    print("\nALL MATCH" if not failures else f"\n{failures} CONFIGURATION(S) DO NOT MATCH")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
