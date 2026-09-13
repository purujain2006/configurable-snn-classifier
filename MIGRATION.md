# Migrating the search from the broken input pipeline

A change spec for an agent working on this repository. Each entry states the
invariant that was violated, the exact site, what feeds it, what consumes it,
and how to tell whether the change landed.

Read this section first, because it governs where edits are allowed.

## 0. Which files are generated

`build_from_practice2.py` extracts most of `snnsearch/` out of `Practice2.py`
by line range. These modules are GENERATED and an edit to them survives only
until someone regenerates:

    config.py  neuron.py  model.py  folding.py  quantize.py  train.py  spaces.py

Behavioral changes to those files go in the `PATCHES` dict in
`build_from_practice2.py`, keyed by generated filename, as exact-string
`(old, new)` pairs. `apply_patches` exits if a pattern matches zero times or
more than once, so a patch that silently stops applying is impossible.

When you patch a generated file, apply the same edit to the checked-in copy as
well, so the tree is consistent before the next regeneration.

These are hand-written; edit directly:

    pipeline.py  search.py  streaming.py  synops.py  cli.py  results.py
    data/*  hardware.py  planning.py  cost.py  encoders.py  report.py

---

## 1. The trial config must reach the data pipeline

**Invariant violated.** A search records the configuration it sampled and
reports a score beside it. Nothing checked that the two belonged together.

`search.py` built the model spec from `trial_cfg`, built the dataloaders from
the YAML, and then overwrote the spec's input section with the YAML's. So every
trial trained at `encoding.resize_to` and `encoding.T` from the config file no
matter what Optuna chose, while the leaderboard recorded the sampled values.
Replaying a winner produced a different and weaker network.

### 1a. Split `prepare` in `snnsearch/pipeline.py`

Replace the single `prepare(cfg, flat_config=None)` with three functions:

- `resolve_specs(cfg, flat_config=None, quiet=False)` returns
  `(bundle, encoder, spec)`. No loaders.
- `make_loaders(cfg, bundle, encoder, spec, quiet=False)` returns the three
  loaders.
- `prepare(cfg, flat_config=None)` calls both and returns the old 4-tuple.

Keep `prepare`'s signature and return shape. `tools/smoke_test.py` and
`run_single` both unpack four values from it.

The split exists so the connection-limit check can run between the two. That
check needs the final input spec, and it must not run against a resolution the
loaders will contradict.

Add `quiet` so the trial path can suppress the per-run printouts. Four hundred
trials each printing a dataset description is unreadable.

### 1b. Pass the trial config through, in `snnsearch/search.py`

Inside `_make_trainable`'s `trainable`, replace:

```python
spec = config_to_specs(trial_cfg)
...                                        # feasibility check here
_bundle, _enc, loaders, base = prepare(run_cfg)
for k in ("input", "output"):
    spec[k] = base[k]
```

with:

```python
_bundle, _enc, spec = resolve_specs(run_cfg, trial_cfg, quiet=True)
_assert_records_what_ran(trial_cfg, spec, _enc)
...                                        # feasibility check, now after
loaders = make_loaders(run_cfg, _bundle, _enc, spec, quiet=True)
```

Drop the now-unused `from .spaces import config_to_specs` from the trial's
local imports. `tools/check_static.py` catches it if you forget.

**Upstream.** `run_cfg` is the loaded YAML minus underscore-prefixed keys,
captured once at closure time. `trial_cfg` is what Optuna sampled for this
trial.

**Downstream.** `spec` feeds `check_feasibility` and `run_training`.
`spec["input"].N` sets the loader batch size. `streaming.py` records
`trial.config`, which is `trial_cfg`, so the record now describes the run.

**Watch.** `resolve_specs` routes a non-empty `flat_config` through
`specs_from_flat`, which forces `flat["N"] = cfg["search"]["batch_size"]` and
defaults `epochs` and `data_dir`. The old path reached the same values by a
different route. Confirm with `tools/diff_spec.py` that the two spec builders
still agree field for field.

---

## 2. T must reach the dataset, not only the encoder

**Invariant violated.** Event clips are cached as a fixed number of frames per
clip, so the frame count is decided when the dataset object is constructed.
Passing T only to the encoder leaves every run on whatever cache the dataset
factory defaults to.

In `resolve_specs`:

```python
bundle = build_dataset({**cfg["dataset"], "T": T})
```

`data/base.py::build_dataset` pops `module`, `factory` and `name`, then calls
`factory(**spec)`. `data/builtin.py::dvs128_gesture(root, T=16, **_ignored)`
reads T and passes it as `frames_number`. Before this change T was never in
`spec`, so the default of 16 applied to every run in the project's history.

**Downstream.** Each distinct T needs its own cache directory on disk. See
section 5.

**Watch.** A user-supplied dataset factory that does not accept `**kwargs` now
fails on the unexpected `T`. Both built-in factories already absorb extras.

---

## 3. Fall back to the config file, not to nothing

In `resolve_specs`, resolve each input knob as trial first, config second:

```python
flat = flat_config or {}
T = int(flat["T"]) if flat.get("T") else int(enc_cfg.get("T", 16))
resize = flat["resize_to"] if flat.get("resize_to") is not None \
    else enc_cfg.get("resize_to")
```

A trial that did not sample `resize_to` must inherit the file's value. Falling
back to `None` means native resolution, which on DVS128 is 128x128 and 32,768
input axons against a limit of 16,383, so every such trial would be rejected
for a reason unrelated to the configuration.

---

## 4. Assert that the record matches the run

**This is the general fix; sections 1 through 3 are the instance.**

Add a module-level function to `snnsearch/search.py`:

```python
def _assert_records_what_ran(trial_cfg, spec, encoder):
    inp = spec["input"]
    want_T = trial_cfg.get("T")
    want_resize = trial_cfg.get("resize_to")
    problems = []
    if want_T is not None and int(want_T) != int(inp.T):
        problems.append(f"T: sampled {want_T}, spec has {inp.T}")
    if want_T is not None and int(want_T) != int(getattr(encoder, "T", want_T)):
        problems.append(f"T: sampled {want_T}, encoder has {encoder.T}")
    if want_resize is not None and int(want_resize) != int(inp.resize_to):
        problems.append(f"resize_to: sampled {want_resize}, "
                        f"spec has {inp.resize_to}")
    if problems:
        raise RuntimeError(
            "the trial would be recorded under a config it is not running:\n  "
            + "\n  ".join(problems))
```

Skip a check when the key is absent, so a config that did not sample a knob is
not a failure.

It must be module level. The trainable is pickled and shipped to Ray workers,
and a nested function is a local object that pickle cannot serialize.

Dying here costs one trial. Not checking cost four hundred.

---

## 5. Build the frame caches before anything runs in parallel

**Invariant violated.** Cache construction is lazy, single-threaded numpy, and
tens of minutes long. Now that T varies, several trial processes can discover
the same missing directory at once and race to write it. A half-written cache
reads as complete, so training silently runs on a fraction of the data.

`Practice2.py` had `warmup_dataset_cache` for exactly this reason and the
refactor dropped it, the same way it dropped the streaming callback.

### 5a. Port the helpers into `snnsearch/data/builtin.py`

Three functions, from `Practice2.py` around lines 2296 to 2353:

- `_class_dirs_have_npz(split_dir, num_classes=11)`
- `frame_cache_is_complete(root, T)` — both splits, all classes, at least one
  `.npz` each. Directory existence alone is not enough.
- `warmup_frame_cache(root, T_values, log=print)` — for each T, skip if
  complete, delete and rebuild if partial, then verify and raise if it is still
  incomplete.

### 5b. Call it from `snnsearch/search.py::run_search`

Hoist `t_choices` to a local so the warmup and `make_define_by_run` use one
list, then call `_warm_frame_caches(cfg, t_choices, writer)` before the space
is built and well before Ray schedules anything. Guard on
`cfg["dataset"]["name"] == "dvs128"`, since only that dataset caches this way.

**Downstream.** First `run_search` for a new T is now slow before any trial
starts. That is the intent.

---

## 6. An explicit flag beats a recorded trial

A record can be wrong about what its run did. Reproducing the run then means
contradicting the record deliberately, so that has to be expressible.

`cli.py` already merges `--T` and `--resize-to` into `cfg["encoding"]`, but
with `--from-best` the recorded values now win. Pass the typed values to
`run_single` and drop the corresponding keys from `flat`:

```python
# cli.py, single branch
input_overrides={"T": getattr(args, "T", None),
                 "resize_to": getattr(args, "resize_to", None)}

# pipeline.py, run_single
for key, val in (input_overrides or {}).items():
    if val is None or not flat:
        continue
    if flat.pop(key, None) is not None:
        print(f"override  : {key}={val} from the command line, ...")
```

Print when an override fires. A silent override is the same class of problem
this whole document is about.

---

## 7. Count SynOps from the weight layers

**Invariant violated.** The old counter hooked the spiking layers, counted
spikes, and multiplied by a fan-out looked up from the cost table by list
position. Pairing two lists positionally slides as soon as the module list
contains something the cost table describes differently, such as a
fully-connected head. A run reported 73.2M SynOps against a hard ceiling of
44.9M.

In `snnsearch/synops.py`, hook `nn.Conv2d` and `nn.Linear` instead. Each knows
its own arithmetic, and its input tensor at hook time says what fraction of
that arithmetic a spike actually drove:

```python
dense = out.numel() * mod.in_features                      # Linear
dense = out.numel() * (mod.in_channels // mod.groups) * kh * kw   # Conv2d
active = float(x.sum().item()) / float(x.numel())
self.ops[name] += active * dense
```

Keep hooking `HardwareLIFNode`, but only to report the firing rate. It no
longer feeds the total.

Add a ceiling guard in `summary()`: no pass can accumulate more than the
layer's dense cost, so a total above `dense * passes` is a broken measurement
rather than a surprising one. Return `None` plus a `reason` instead of a
number, because a broken number that reaches a leaderboard gets quoted.

**Downstream.** `measure_synops(net, loader, device, spec, max_batches)` must
keep its signature; `pipeline.measure_run_synops` and `search.py` call it.
Every `synops_per_sample` recorded before this change is void.

---

## 8. Widen the search space, and unpin the head

These are choices rather than corrections, but they are void without section 1,
since two of the knobs never reached the model.

All of these go through `PATCHES["spaces.py"]` in `build_from_practice2.py`.

Use `suggest_int` rather than `suggest_categorical` for ordered knobs. TPE
models a categorical as a set of unrelated labels, so learning that 64 beat 32
says nothing about 88. As an ordered integer it interpolates, which is why a
search over nine sizes costs barely more than a search over two.

    resize_to    [32, 64]              -> suggest_int(24, 88, step=8)
    channels     CHANNEL_CHOICES       -> suggest_int(8, 128, log=True)
    ch_{i}       CHANNEL_CHOICES       -> suggest_int(8, 128, log=True)
    kernel_size  KERNEL_CHOICES        -> suggest_int(3, 9, step=2)
    k_{i}        KERNEL_CHOICES        -> suggest_int(3, 9, step=2)

Then unpin the head reduction. Remove `"final_reduction": "flatten"` from the
constant dict the space returns and add:

```python
trial.suggest_categorical("final_reduction", ["flatten", "gap"])
```

GAP is not optional here. Flatten sends `H*W*C` into the first fully-connected
layer, so resolution drives the fan-in directly and most of the widened resize
range would fail the limit before training. GAP collapses `H*W` first, leaving
a fan-in equal to the channel count at any resolution. Pinning flatten makes
the widening decorative.

The prior verdict that GAP costs about 40 points was measured when resolution
never varied, so it compared the two at a single input size. Update the comment
that states it, or the file will contradict itself.

**Downstream, all already correct, verify rather than change.**
`spaces.config_to_specs` reads `config.get("final_reduction", "flatten")`.
`planning.plan_network` branches on gap versus flatten to compute `fc_in`.
`streaming.LEADERBOARD_COLS` already lists the column.

---

## Verification

Static, no GPU needed:

    python tools/check_static.py     # pyflakes, call signatures, intra-package imports
    python tools/smoke_test.py       # 7-stage pipeline on synthetic data
    python tools/test_synops.py      # arithmetic, silent input costs zero, ceiling
    python tools/diff_spec.py <best.json>

`tools/check_static.py` is the one that matters most after this set of changes,
because it resolves every `from .mod import name` and checks intra-package call
signatures. The class of failure it catches, a name that moved or a caller that
did not follow, is how several of these bugs survived.

Behavioral, requires the dataset:

    python main.py summary -c configs/dvs128.yaml

The summary must stay byte-identical to `Practice2.py`'s for an unpatched
configuration. Any drift means a patch changed something it should not have.

Empirical, requires a GPU. This is the test the whole document exists for:

    python main.py single -c configs/dvs128.yaml \
        --from-best <best.json> --T 16 --resize-to 64 \
        --results-dir results/r64 --ckpt r64.pth

Compare epoch 0 against the original trial's logged
`val 0.5000 / train 0.3609 / loss 2.1691`. A match means the input pipeline was
the whole story. A mismatch means something else differs, and the next place to
look is weight initialization, which currently inherits whatever random state
`build_dataloaders` leaves behind rather than being seeded on purpose.
