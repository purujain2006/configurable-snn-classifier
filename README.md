# Configurable SNN Classifier

A single-file, fully configurable **spiking neural network (SNN)** classifier for the
[DVS128 Gesture](https://research.ibm.com/interactive/dvsgesture/) dataset, targeting the
**HiAER-Spike** neuromorphic hardware platform.

The architecture, the neuron parameters and the optimization recipe are all expressed as
configuration, so the same code serves hand-designed experiments and an automated search over the
configuration space.

The search optimizes accuracy measured on the converted network rather than on the floating-point
one. A configuration must fit HiAER-Spike's axon and fan-in/fan-out limits, and its accuracy is
recorded after batch normalization is folded away, weights are quantized to INT16, and the chip's
neuron dynamics are applied.

## Documentation site

A full documentation site is included under `docs/`, written for readers with no background in spiking networks:

> **https://purujain2006.github.io/configurable-snn-classifier/**

Seven pages in reading order. The interactive demonstrations use arithmetic ported from
`Practice2.py` and verified against it by `tools/test_snn_port.js`.

| Page | Covers |
|---|---|
| [01 · The problem](docs/index.html) | HiAER-Spike and its constraints, conventional networks, spiking networks, why their costs differ, and what the project sets out to do |
| [02 · The neuron](docs/neurons.html) | The neuron the chip implements: update order, integer leak, quantized threshold, surrogate gradients |
| [03 · The architecture](docs/architecture.html) | The network as configuration, shape planning computed once, the cost model, and an interactive builder |
| [04 · Fitting the chip](docs/hardware.html) | Routing limits as arithmetic, the INT16 grid, batch-norm folding, quantization-aware training |
| [05 · Training](docs/training.html) | Event data within the axon budget, rate coding, pipeline flush, the training schedule |
| [06 · The search](docs/search.html) | Evaluation tiers, early stopping, the conditional search space, results recording |
| [07 · Reference](docs/reference.html) | Every class and function, grouped by section, searchable |

To read it locally, open `docs/index.html` in a browser. Network access is used only for fonts and
syntax highlighting.

## Running the code

```bash
# architecture table, feasibility and counts. No GPU stack required.
python Practice2.py summary --input resize_to=64 T=16 \
    --encoder depth=2 channels=32 kernel_size=7 stride=2 --head fc_widths=

# train one configuration
python Practice2.py single --data-dir <DVS128_root> --encoder ... --ckpt best.pth

# fold, quantize, verify, and write the conversion tables
python Practice2.py fold --ckpt best.pth --data-dir <DVS128_root>

# search the configuration space
python Practice2.py search --compute local --trials 50 --data-dir <DVS128_root>
```

Dependencies for training: `torch`, `spikingjelly`, and for search additionally `ray[tune]` and
`optuna`. `summary` mode is designed to run without the GPU stack.

## Repo layout

```
Practice2.py            the whole system (config → plan → feasibility → model → QAT → search)
docs/                   the interactive explainer site (GitHub Pages serves this folder)
tools/
  test_snn_port.js      verifies the site's JS math against Python-generated ground truth
  ground_truth.json     reference outputs from Practice2.py sections 1–4
```
