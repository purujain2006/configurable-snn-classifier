"""Compare the spec a search trial builds against the one --from-best rebuilds.

WHY THIS EXISTS

Trial df8c5229 climbed to 0.892 with train accuracy 0.988. Replaying its config
through `single` reached 0.750, and diverged from the first epoch: the trial
scored 0.500 after epoch 0, both replays scored 0.409. Two replays agreeing with
each other and disagreeing with the trial is not noise, it is a different
network or a different training setup.

WHAT IT FOUND, AND WHAT IT MISSED

The specs matched exactly, which ruled out the architecture and the training
schedule and left the data pipeline. That was the answer: the search built its
dataloaders from the config file rather than from the sampled trial, so every
trial trained at the file's T and resize_to whatever the sampler chose, and the
replay honoured the recorded 32x32 at T=8. A smaller, weaker network.

This tool could not see that, because both sides it compares read the same flat
config. The check that catches it now lives in the trial itself, where the
sampled input can be compared against the spec that actually builds the model
(`_assert_records_what_ran` in snnsearch/search.py).

So this remains a regression check on the two spec builders, and the input
section below is the part worth reading.

    python tools/diff_spec.py results/dvs128/best.json
"""
import json
import os
import sys
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def build_trial_side(flat):
    """The original direct flat builder, before shared pipeline resolution."""
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
    from snnsearch.pipeline import load_flat_config
    flat = load_flat_config(path)

    print("=" * 78)
    print(f"spec comparison for {data.get('trial_id', '?')}   "
          f"reported hw_val {data.get('hw_val_accuracy')}")
    print("=" * 78)

    # The saved config has data_dir stripped by the streaming callback, and the
    # trial side wants it present, so both sides get the same stand-in.
    trial_flat = dict(flat)
    trial_flat.setdefault("data_dir", ".")
    trial_flat.setdefault("epochs", 40)
    trial_flat.setdefault("N", 16)

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
        print("Specs are IDENTICAL. Both builders agree, which is what this checks.")
        print("")
        print(f"This trial recorded T={flat.get('T')} resize_to={flat.get('resize_to')}.")
        print("Confirm the run used them: `single` prints a loaders line with")
        print("T and resize, and the search asserts the same thing per trial.")
        print("Records written before that assertion existed may name an input")
        print("size the run did not use.")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
