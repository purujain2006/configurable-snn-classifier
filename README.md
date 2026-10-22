# configurable-snn-classifier

This tool searches for spiking neural networks that can run on HiAER-Spike, a
neuromorphic computer built at UC San Diego. Every network it reports fits the
chip's limits, was trained with the chip's own arithmetic, and was checked
clip by clip against a simulator of the chip.

The chip's limits are settings in the config, so the search still works if the
hardware changes. It runs on any dataset the config can load and any network
the config can describe.

The website explains the project from the beginning, with a demo for each idea,
and assumes no background in neural networks or chips:
https://purujain2006.github.io/configurable-snn-classifier/

To install the tool and run it, read [GUIDE.md](GUIDE.md).

## Commands

```bash
python main.py check                                        # what this machine is missing
python main.py summary   -c configs/dvs128_search05.yaml    # network size and chip fit, no GPU needed
python main.py preflight -c configs/dvs128_search05.yaml    # every stage of a search in miniature
python main.py search    -c configs/dvs128_search05.yaml    # the search itself
python main.py study     -c configs/dvs128_search05.yaml --from-best results/dvs128_search05/best_finalist.json
python main.py single    -c configs/dvs128_search05.yaml --from-best results/dvs128_search05/best_finalist.json --epochs 300
python main.py single    -c configs/dvs128_search05.yaml --from-best results/dvs128_search05/best_finalist.json --epochs 300 --folds 5
python main.py report    results/dvs128_search05            # rebuild report.html
```

A search will not start until `preflight` has passed for the same config and
code. The preflight runs a small search from start to finish on the real data
and GPU. Problems that would otherwise show up hours into a search, such as a
missing package, a wrong dataset path, a trial too large for its share of the
GPU or a bug in the steps after training, show up in minutes.

`single --folds 5` scores a chosen network with five-fold cross-validation.
Every training person is held out once, and the result is an average with a
standard error. It costs five trainings, so it is for comparing a few finished
networks.

## Tests

Each test runs on its own on a laptop, without a GPU. The last one needs ray and optuna and takes about five minutes.

```bash
python tools/checks/check_static.py      # undefined names and call signatures, no torch needed
python tests/smoke_test.py               # the whole pipeline on synthetic data, about a minute
python tests/test_chip_rules.py          # tau, weight decay, fan-in behind pools, bias neurons
python tests/test_search_space.py        # every option sampled, the balanced block, any dataset's shape
python tests/test_sim_match.py           # PyTorch and the integer simulator fire the same spikes
python tests/test_training_options.py    # augmentation, TET loss, seeds, exact measurement
python tests/test_preflight.py           # config rules, the preflight record, early stopping
python tests/test_study.py               # the option study's variants and verdicts
python tests/test_docs.py                # GUIDE.md covers every setting, command and test
python tests/test_input_pipeline.py      # sampled inputs, replay flags, warm-up ordering
python tests/test_frame_cache.py         # incomplete caches and rebuilds
python tests/test_synops.py              # SynOps counted the way the chip does the work
python tests/test_per_subject.py         # accuracy per person
python tests/test_val_folds.py           # the cross-validation folds
python tests/test_early_exit.py          # accuracy and SynOps from the first k frames
python tests/test_recount_synops.py      # recounting SynOps for saved networks
node tests/test_snn_port.js              # the website's arithmetic against the Python package
python tests/test_search_e2e.py          # a real search and option study through Ray, about 5 minutes
```

## Why the search scores the converted network

A network trained the usual way, with 32-bit decimal numbers, is a different
network from the one the chip runs. The chip rounds every weight and threshold
to a 16-bit whole number, leaks by halving, updates its neurons in a fixed
order, and has no batch normalization. Each of these changes the output, and
none of them shows up in a training accuracy.

So the search measures accuracy after the network has been converted for the
chip. A network has to fit the connection limits, keep working once batch norm
is folded into its weights, and keep its thresholds in range after rounding. A
network that cannot run on the chip scores zero, however well it trained.

## What the search tries

The input channels, input size and number of classes come from the dataset,
and the range of input sizes to try is worked out from them and the chip's
16,000 input lines. Besides the network's shape and the training recipe, the
search tries each of these options:

| option | values |
|---|---|
| pooling | none, average (merged into the next layer), or max (one OR neuron per window) |
| bias | merged into the threshold, or added by bias neurons that fire on every step |
| head | flatten, or global average pooling |
| signed spikes | off, or each unit sends +1 or -1 using two chip neurons |
| bits per input value | 1 to 4, one input line per bit |
| tau (the leak) | fixed, learned per layer, or learned per channel, always a power of two |
| loss | spike count over the clip, or TET (a loss at every timestep) |
| augmentation | shifts, cutout, dropped events, and flips where a flip keeps the label |
| dropout between layers, firing-rate penalty | off or on |

A search starts with a balanced block of trials (`search.balanced_trials`) in
which every option appears equally often, so each one gets a fair test before
Optuna starts choosing. At the end, the best `search.finalists` trials are
trained again with `search.finalist_seeds` new seeds each. The one with the
best average validation score is written to `best_finalist.json`, and
`python main.py single -c <config> --from-best results/<run>/best_finalist.json`
trains it again.

## Your own data

Settings go in the config. A dataset is code, so the config names the file
that builds it:

```yaml
dataset:
  module: examples/cifar10_data.py
  factory: make_datasets
  root: ./data/cifar10
encoding:
  coding: levels     # levels, passthrough, poisson, temporal or direct
  T: 8
```

The factory returns a `DatasetBundle`, as in `examples/cifar10_data.py`. The
built-in datasets (`dvs128`, `cifar10`, `cifar100`, `mnist`, `fashion_mnist`)
need no module.

Event-camera data is already made of spikes. A still image has no time axis,
and the coding decides what the network receives at each timestep.

| coding | what it sends | use it for |
|---|---|---|
| `levels` | each value as a whole number from 0 to 2^B - 1 on B input lines, every step | still images (the default) |
| `passthrough` | resized events, as 1 bit for any event or B bits for the count | DVS and other event cameras |
| `poisson` | a random spike with probability equal to the brightness | still images, noisy when T is small |
| `temporal` | one spike per pixel, earlier for brighter pixels | the fewest spikes, but harder to train |
| `direct` | the raw value at every step | software only, since the chip's input lines carry spikes |

The best coding depends on the task, so it is a config setting.

## SynOps

Every run counts synaptic operations (SynOps) per sample, the number of times
a spike adds a weight to a neuron on the chip, measured on real inputs. Papers on
neuromorphic hardware report the same count, so results can be compared with
other chips. SynOps count work, and energy also depends on memory traffic and
on the power the chip draws while idle, which can be the larger part on an FPGA.

Searching for fewer SynOps is optional:

```yaml
objective:
  mode: pareto          # accuracy, constrained, weighted or pareto
  synops_budget: 2.0e6  # for constrained
```

`pareto` keeps every network that no other network beats on both accuracy and
SynOps, which suits a search where the trade-off is not settled yet.

## The website

The site lives in `docs/` and GitHub Pages serves it. Its demos use arithmetic
ported from the package, and `tests/test_snn_port.js` checks that port against
the Python. It has a light theme and a dark theme.

| Page | Covers |
|---|---|
| [Home](docs/index.html) | what the project is, the chip's rules, and where it stands |
| [0 · Why](docs/why.html) | the chain of problems that led to this project, from the energy cost of neural networks to the search |
| [A · Neural networks](docs/basics.html) | neurons, layers, learning, convolutions, spiking neurons, and the tools that help a network train |
| [B · Data and testing](docs/data.html) | the event camera, frames, the train, validation and test split, overfitting, cross-validation, augmentation |
| [C · Pipeline and search](docs/pipeline.html) | one trial from start to finish, then the search and what it has found so far |
| [D · Every setting](docs/settings.html) | each setting the search tries, with a picture and how it moved accuracy |
| [E · Bit encoding](docs/bits.html) | sending a pixel's event count to the chip one bit per input line |
| [F · The chip's neuron](docs/neurons.html) | the update order, the leak and the threshold, and how training matches the chip |
| [G · Fitting the chip](docs/hardware.html) | layer sizes, connection limits, 16-bit weights, removing batch norm |
| [H · Results](docs/results.html) | the best networks so far and how far to trust the numbers |
| [Guide](docs/guide.html) | questions such as "will my network fit?", each with sections to read and commands to run |
| [Glossary](docs/glossary.html) | every term, in plain words |
| [Code reference](docs/reference.html) | every class and function, grouped by module |

To read it offline, open `docs/index.html`. It uses the network only for fonts.

## Layout

```
main.py                  launcher
configs/                 YAML run configurations
examples/                a user-supplied dataset, worked through
snnsearch/               the package (snnsearch/README.md describes every file)
docs/                    the website
tests/                   the tests, each runnable on its own
GUIDE.md                 how to use the tool, start to finish
tools/
  match_simplesim.py     compare the tool's simulator with HiAER-Spike's, clip by clip
  fpga_diff.py           find where the FPGA's spikes depart from the simulator's
  replay_top.py          retrain the best N trials of a search
  build_cache.py         build the frame cache without a GPU
  hpo_analysis.py        which settings moved accuracy, corrected for many comparisons
  leaderboard_from_trials.py   rebuild a complete leaderboard from trials.jsonl
  read_ckpt.py           read a .pth file into numpy without torch
  checks/                check_env, check_static, train_and_verify
  analysis/              top_trials, factor_effects, per_subject, early_exit, recount_synops
  site/                  website tools: page checks, the glossary build
  archive/               Practice2.py, the single file the package was split from
```

## Dependencies

`summary` and `check` need only Python 3.9 or later. Training needs `torch` and
`spikingjelly`, and the search also needs `ray[tune]` and `optuna`. Install
torch first, or spikingjelly may install a different build of it.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install spikingjelly
pip install "pydantic<3" "ray[tune]" optuna
python tools/checks/check_env.py /path/to/dataset
```
