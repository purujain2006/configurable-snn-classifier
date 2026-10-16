# snnsearch

Finds good hyperparameters for a **spiking neural network** under hardware
constraints, and tells you whether the result can actually be deployed.

Built around **HiAER-Spike**, but the limits are configuration rather than
constants, so it retargets as the hardware changes. Any dataset, any input
coding, any architecture the configuration can express.

**New here? Start with [GUIDE.md](GUIDE.md):** install, data, configs, the
preflight, running a search, reading results, the option study, and a
troubleshooting table.

## What to run

```bash
python main.py check                                        # what this machine is missing
python main.py summary   -c configs/dvs128_search05.yaml    # size and chip fit, no GPU
python main.py preflight -c configs/dvs128_search05.yaml    # every stage of a search, ~15 min
python main.py search    -c configs/dvs128_search05.yaml    # the search
python main.py study     -c configs/dvs128_search05.yaml --from-best results/dvs128_search05/best_finalist.json
python main.py single    -c configs/dvs128_search05.yaml --from-best results/dvs128_search05/best_finalist.json --epochs 300
python main.py report    results/dvs128_search05            # rebuild report.html
```

A search will not start until `preflight` has passed for the same config and
code. The preflight runs a miniature search end to end on the real data and
GPU, so a problem that used to surface hours into a run (a missing package, a
dataset path, a trial too big for its share of the GPU, a bug in the steps
after the trials) surfaces in minutes instead.

The tests, each runnable on its own on a laptop:

```bash
python tools/checks/check_static.py      # undefined names and call signatures, no torch
python tests/smoke_test.py               # the whole pipeline on synthetic data, ~1 min
python tests/test_chip_rules.py          # tau, weight decay, fan-in behind pools, bias neurons
python tests/test_search_space.py        # every option sampled, balanced block, any dataset's shape
python tests/test_sim_match.py           # PyTorch and IntegerSim fire identical spikes, every feature
python tests/test_training_options.py    # augmentation, TET loss, seeds, exact measurement
python tests/test_preflight.py           # config rules, the preflight record, early stopping
python tests/test_study.py               # the option study's variants and verdicts
python tests/test_docs.py                # GUIDE.md covers every setting, command and test
python tests/test_input_pipeline.py      # sampled inputs, replay flags, warmup ordering
python tests/test_frame_cache.py         # incomplete caches and sequential rebuilds
python tests/test_synops.py              # weight-layer arithmetic and dense ceilings
python tests/test_search_e2e.py          # a real search and study through Ray, ~5 min
node tests/test_snn_port.js              # the site's JavaScript math against Python
```

## Why it exists

A network trained in floating point is not the network the chip runs. They
differ in weight precision, threshold precision, leak resolution, neuron update
order, and whether batch normalization is still present. Every one of those
changes the output, and none of them appear in a training accuracy figure.

So the search optimizes accuracy measured **after** conversion. A configuration
must fit the connection limits, survive batch-norm folding, and quantize to
INT16 without driving a threshold out of range. One that cannot be represented
on chip scores zero regardless of how well it trained.

## What the search tries

Nothing in the search is specific to one dataset: the input channels, size
and class count come from the dataset, and the range of input sizes is worked
out from them and the chip's 16,000 input axons. Besides the network's shape
and the training recipe, it tries each of these on and off:

| option | what it is |
|---|---|
| pooling | none, average (folded into the next layer), or max (one OR neuron per window) |
| bias | folded into the threshold, or delivered by always-on bias neurons |
| head | flatten or global average pooling |
| signed spikes | a unit sends +1 or -1, as two chip neurons |
| bits per input value | 1 to 4, one input axon per bit |
| tau | fixed, learned per layer, or learned per channel (always a power of two) |
| loss | spike rate over the clip, or TET (a loss at every timestep) |
| augmentation | shifts, cutout, dropped inputs, and flips where they keep the label |
| conv dropout, firing-rate penalty | regularizers |

A search starts with a balanced block (`search.balanced_trials`) in which each
option appears equally often, so all of them get a fair test before Optuna
starts steering. When it ends, the top `search.finalists` trials are trained
again with `search.finalist_seeds` new seeds each, and the one with the best
mean validation score is written to `best_finalist.json`. Replay it with
`python main.py single -c <config> --from-best results/<run>/best_finalist.json`.

## Bringing your own data

Values go in the config. A dataset is code, so the config points at it:

```yaml
dataset:
  module: examples/cifar10_data.py
  factory: make_datasets
  root: ./data/cifar10
encoding:
  coding: levels     # levels | passthrough | poisson | temporal | direct
  T: 8
```

The factory returns a `DatasetBundle`; see `examples/cifar10_data.py`. Built-in
names (`dvs128`, `cifar10`, `cifar100`, `mnist`, `fashion_mnist`) need no module.

### Coding

Event data is already spikes. A static image is not, and how you give it a time
axis matters:

| coding | what it does | when |
|---|---|---|
| `levels` | each value as a whole number 0..2^B-1, repeated each step, on B axons | default for images; B is searched |
| `passthrough` | resize existing events; 1 bit = any event, B bits = event counts | DVS and other event cameras |
| `poisson` | stochastic spikes, probability = intensity | fully spiking, noisier at small T |
| `temporal` | one spike per unit, earlier = brighter | sparsest by SynOps, harder to train |
| `direct` | analog value repeated each step | software only: no axon carries an analog value |

Which one wins is task-dependent, so it is a config field rather than a constant.

## SynOps

Every run reports **synaptic operations per sample**: how many accumulate
events the network actually performs, counted from real spike activity.

This is the number neuromorphic papers quote, so it compares against Loihi and
SpiNNaker results without owning either. It is **not joules** -- real energy also
includes memory traffic and static power, which on an FPGA may dominate. The
column is called `synops` for that reason.

Optimizing for it is opt-in:

```yaml
objective:
  mode: pareto          # accuracy | constrained | weighted | pareto
  synops_budget: 2.0e6  # for constrained
```

`pareto` is the honest default when the trade-off has not been decided: it
returns the front and lets you choose with the data in front of you.

## Documentation site

A full site under `docs/`, written for readers with no background in spiking
networks:

> **https://purujain2006.github.io/configurable-snn-classifier/**

Eight pages in reading order. The interactive demonstrations use arithmetic
ported from the package and verified against it by `tests/test_snn_port.js`.

| Page | Covers |
|---|---|
| [01 · The problem](docs/index.html) | HiAER-Spike and its constraints, conventional networks, spiking networks, why their costs differ, and how the code is organized |
| [02 · The neuron](docs/neurons.html) | The neuron the chip implements: update order, integer leak, quantized threshold, surrogate gradients |
| [03 · The architecture](docs/architecture.html) | The network as configuration, shape planning, the four measures of size including SynOps, an interactive builder |
| [04 · Fitting the chip](docs/hardware.html) | Connection limits as arithmetic, the INT16 grid, batch-norm folding, quantization-aware training |
| [05 · Training](docs/training.html) | Turning an input into spikes, reading a prediction, the pipeline flush, the training schedule |
| [06 · The search](docs/search.html) | Evaluation tiers, early stopping, the conditional search space, results recording |
| [07 · Results](docs/results.html) | Every run on record, the statistics recomputed from the raw trial files, what the data cannot answer |
| [08 · Reference](docs/reference.html) | Every class and function, grouped by module, searchable |

To read it locally, open `docs/index.html`. Network access is used only for
fonts and syntax highlighting.

## Layout

```
main.py                  launcher
configs/                 YAML run configurations
examples/                a user-supplied dataset, worked
snnsearch/               the package; snnsearch/README.md maps every file
docs/                    the explainer site (GitHub Pages serves this folder)
tests/                   every test, each runnable on its own
GUIDE.md                 how to use the tool, start to finish
tools/
  match_simplesim.py     per-clip match against the simulator; --selftest against hs_api
  replay_top.py          retrain the best N trials of a search
  build_cache.py         build the frame cache without a GPU
  hpo_analysis.py        which knobs moved accuracy, with multiple-comparison control
  leaderboard_from_trials.py   rebuild a complete leaderboard from trials.jsonl
  read_ckpt.py           read a .pth into numpy with no torch installed
  checks/                check_env, check_static, train_and_verify
  analysis/              top_trials, factor_effects, and one-off studies
  site/                  website tooling: page validation, prose and jargon checks
  archive/               Practice2.py, the single file the package was split from
```

## Dependencies

`summary` and `check` need nothing but Python 3.9+. Training needs `torch` and
`spikingjelly`; the search additionally needs `ray[tune]` and `optuna`. Install
torch first, or spikingjelly may pull a different build.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install spikingjelly
pip install "pydantic<3" "ray[tune]" optuna
python tools/checks/check_env.py /path/to/dataset
```
