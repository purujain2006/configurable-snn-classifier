"""Compare the spec a search trial builds against the one --from-best rebuilds.

WHY THIS EXISTS

Trial df8c5229 climbed to 0.892 with train accuracy 0.988. Replaying its config
through `single` reached 0.750, and diverged from the first epoch: the trial
scored 0.500 after epoch 0, both replays scored 0.409. Two replays agreeing with
each other and disagreeing with the trial is not noise, it is a different
network or a different training setup.

A tool whose winners cannot be replayed is not useful, so this compares the two
construction paths field by field.

    python tools/diff_spec.py results/dvs128/best.json
"""
import json
import os
import sys
from dataclasses import asdict, fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def build_trial_side(flat):
    """What snnsearch/search.py does inside a trial."""
    from snnsearch.spaces import config_to_specs
    return config_to_specs(dict(flat))


def build_single_side(flat, batch_size):
    """What snnsearch/pipeline.py does for --from-best."""
    from snnsearch.pipeline import specs_from_flat
    return specs_from_flat(dict(flat), batch_size=batch_size)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/dvs128/best.json"
    with open(os.path.expanduser(path), encoding="utf-8") as fh:
        data = json.load(fh)
    flat = data.get("flat_config") or data

    print("=" * 78)
    print(f"spec comparison for {data.get('trial_id', '?')}   "
          f"reported hw_val {data.get('hw_val_accuracy')}")
    print("=" * 78)

    # The saved config has data_dir stripped by the streaming callback, and the
    # trial side wants it present, so both sides get the same stand-in.
    trial_flat = dict(flat)
    trial_flat.setdefault("data_dir", ".")
    trial_flat.setdefault("epochs", 40)

    a = build_trial_side(trial_flat)
    b = build_single_side(flat, batch_size=trial_flat.get("N", 16))

    diffs = 0
    for section in sorted(set(a) | set(b)):
        if section not in a or section not in b:
            print(f"\n  [{section}] present on only one side")
            diffs += 1
            continue
        da, db = asdict(a[section]), asdict(b[section])
        rows = [(k, da.get(k), db.get(k)) for k in sorted(set(da) | set(db))
                if da.get(k) != db.get(k)]
        if rows:
            print(f"\n  [{section}]")
            for k, va, vb in rows:
                print(f"    {k:<22} trial={va!r:<24} single={vb!r}")
            diffs += len(rows)

    print("\n" + "=" * 78)
    if diffs:
        print(f"{diffs} field(s) differ. The replay is not training the same network.")
    else:
        print("Specs are IDENTICAL.")
        print("So the divergence is not in the configuration. What is left:")
        print("  - RNG: weight init and batch order. build_dataloaders seeds")
        print("    torch before the split, and the model is built afterwards, so")
        print("    anything that consumes randomness in between shifts the init.")
        print("  - the dataloader: num_workers, shuffle, drop_last")
        print("  - the input pipeline: encoder settings, T, resize")
        print("Run both paths with the same seed and compare epoch 0 exactly.")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
