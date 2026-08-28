# snnsearch

Finds good hyperparameters for a **spiking neural network** under hardware
constraints, and tells you whether the result can actually be deployed.

Built around **HiAER-Spike**, but the limits are configuration rather than
constants, so it retargets as the hardware changes. Any dataset, any input
coding, any architecture the configuration can express.

## What to run

```bash
python main.py check                              # what this machine is missing
python main.py summary -c configs/dvs128.yaml     # no torch, no dataset needed
python main.py single  -c configs/dvs128.yaml     # train one configuration
python main.py search  -c configs/cifar10.yaml    # the automated search
python main.py report  results/dvs128             # rebuild report.html
```

`summary` imports no deep-learning library. If it prints the architecture table,
the config parser and the planner agree and you have spent nothing.

That is a weaker claim than it looks, because Python binds names when a line
runs rather than when a module loads, so a module can import cleanly while the
functions inside it reference names that were never imported. Two checks cover
what an import cannot:

```bash
python tools/check_static.py      # undefined names + call signatures, no torch
python tools/smoke_test.py        # the whole pipeline on synthetic data, ~1 min
```

Run both before a real training run.

## Why it exists

A network trained in floating point is not the network the chip runs. They
differ in weight precision, threshold precision, leak resolution, neuron update
order, and whether batch normalization is still present. Every one of those
changes the output, and none of them appear in a training accuracy figure.

So the search optimizes accuracy measured **after** conversion. A configuration
must fit the connection limits, survive batch-norm folding, and quantize to
INT16 without driving a threshold out of range. One that cannot be represented
on chip scores zero regardless of how well it trained.

## Bringing your own data

Values go in the config. A dataset is code, so the config points at it:

```yaml
dataset:
  module: examples/cifar10_data.py
  factory: make_datasets
  root: ./data/cifar10
encoding:
  coding: direct     # direct | poisson | temporal | passthrough
  T: 8
```

The factory returns a `DatasetBundle`; see `examples/cifar10_data.py`. Built-in
names (`dvs128`, `cifar10`, `cifar100`, `mnist`, `fashion_mnist`) need no module.

### Coding

Event data is already spikes. A static image is not, and how you give it a time
axis matters:

| coding | what it does | when |
|---|---|---|
| `direct` | analog value repeated each step | default for images, converges fastest |
| `poisson` | stochastic spikes, probability = intensity | fully spiking, noisier at small T |
| `temporal` | one spike per unit, earlier = brighter | sparsest by SynOps, harder to train |
| `passthrough` | resize and binarize existing events | DVS and other event cameras |

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
ported from the package and verified against it by `tools/test_snn_port.js`.

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
snnsearch/
  config planning hardware cost     no torch -- summary runs anywhere
  quantgrid neuron model folding    the network and its deployed form
  quantize train synops             training and what it measures
  data/ encoders                    what makes it work on any dataset
  spaces search results report      the search and what it writes
  cli pipeline runconfig            wiring
docs/                    the explainer site (GitHub Pages serves this folder)
tools/
  check_env.py           what a fresh machine still needs
  check_static.py        undefined names and call signatures, without torch
  smoke_test.py          the whole pipeline on synthetic data, one minute, CPU
  build_cache.py         build the frame cache without a GPU
  trial_analysis.py      recompute the recorded statistics from raw trial files
  fold_bias_equivalence.py   measure the train/deploy gap on a checkpoint
  read_ckpt.py           read a .pth into numpy with no torch installed
  test_snn_port.js       verify the site's JS math against Python ground truth
  validate_pages.js      structural checks on the site
  prose_audit.js         writing checks
  jargon_check.js        every term defined before first use
  make_fileviews.py      regenerate the site's source excerpts from real files
Practice2.py             the original single file the package was split from
build_from_practice2.py  the splitter, kept so the extraction is reproducible
```

`Practice2.py` is retained as the historical source. `build_from_practice2.py`
regenerates the extracted modules from it by line range, which is how the split
was done without retyping 2,500 lines. `summary` output from the package is
byte-identical to the original for the same configuration.

## Dependencies

`summary` and `check` need nothing but Python 3.9+. Training needs `torch` and
`spikingjelly`; the search additionally needs `ray[tune]` and `optuna`. Install
torch first, or spikingjelly may pull a different build.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install spikingjelly
pip install "ray[tune]" optuna
python tools/check_env.py /path/to/dataset
```
