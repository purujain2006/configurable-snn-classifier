# Configurable SNN Classifier

A single-file, fully configurable **spiking neural network (SNN)** classifier for the
[DVS128 Gesture](https://research.ibm.com/interactive/dvsgesture/) dataset, targeting the
**HiAER-Spike** neuromorphic hardware platform.

Depth, width, kernel sizes, downsampling strategy, head shape, neuron parameters, and the whole
optimization recipe are all configuration — so the same code serves hand-designed experiments
*and* an Optuna/Ray Tune hyperparameter search that only rewards configs which are
**hardware-feasible** (fit HiAER-Spike's axon/fan-in/fan-out limits) and accurate **as deployed**
(post BatchNorm-fold, post INT16 quantization, with the chip's exact neuron dynamics).

## 📖 The interactive explainer

**This repo ships with a full interactive website explaining how everything works, written for
readers who have never touched an SNN:**

> **https://purujain2006.github.io/configurable-snn-classifier/**

Seven chapters with live demos whose math is ported line-for-line from `Practice2.py` and
verified against it (`tools/test_snn_port.js`):

| Chapter | Covers |
|---|---|
| [01 · The big picture](docs/index.html) | Event cameras, spikes vs. numbers, the train-vs-deploy gap, the whole pipeline |
| [02 · Spiking neurons](docs/neurons.html) | LIF simulator, the chip's fire→reset→leak→integrate order, integer leaks, surrogate gradients |
| [03 · Architecture](docs/architecture.html) | Config dataclasses, shape planning, an interactive network builder with live cost + feasibility |
| [04 · Hardware & quantization](docs/hardware.html) | Wiring budgets, the INT16 grid, BatchNorm folding, quantization-aware training |
| [05 · Data & training](docs/training.html) | Events→frames, OR-pool binarization, rate coding, the pipeline flush, training phases |
| [06 · The search](docs/search.html) | Optuna define-by-run, Ray Tune, ASHA, the three-tier evaluation funnel, crash-proof results |
| [07 · Code reference](docs/reference.html) | Every function and class, searchable |

To browse locally, just open `docs/index.html` in a browser (an internet connection is used only
for fonts and syntax highlighting).

## Running the code

```bash
# architecture table + feasibility + neuron/synapse counts — no torch needed
python Practice2.py summary --input resize_to=64 T=16 \
    --encoder depth=2 channels=32 kernel_size=7 stride=2 --head fc_widths=

# one training run (laptop or single GPU)
python Practice2.py single --data-dir <DVS128_root> --encoder ... --ckpt best.pth

# fold BatchNorm, quantize, verify, export hardware tables
python Practice2.py fold --ckpt best.pth --data-dir <DVS128_root>

# hyperparameter search (Optuna + Ray Tune + ASHA)
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
