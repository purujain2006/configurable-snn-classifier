# snnsearch, file by file

A configuration becomes a **plan** (every layer's size), and everything else
reads the plan. The files below are in the order a configuration passes
through them.

## 1. Describe a configuration

| file | what it does |
|---|---|
| `config.py` | the dataclasses: `InputSpec`, `EncoderSpec`, `DownsampleSpec`, `HeadSpec`, `NeuronSpec`, `TrainSpec` |
| `runconfig.py` | loads the YAML run file (`configs/*.yaml`) and its defaults |
| `spaces.py` | the search space: what a trial samples, geometry first, sized from the dataset; the balanced block |

## 2. Judge it before training (no GPU, no torch)

| file | what it does |
|---|---|
| `planning.py` | `plan_network`: every layer's input and output size, once |
| `hardware.py` | the chip's limits and `check_feasibility` against the plan |
| `cost.py` | neurons, connections and parameters per layer; the `summary` table |

## 3. Build and train

| file | what it does |
|---|---|
| `neuron.py` | `HardwareLIFNode` (the chip's neuron) and `ConvBNFoldQuant` (conv with BatchNorm folded in, trained on the INT16 grid) |
| `quantgrid.py` | rounding weights, biases and thresholds to the INT16 grid, differentiably |
| `model.py` | `SpikingCNN`: the network built from the plan |
| `encoders.py` | turns input into spikes or B-bit levels (events pass through; images become levels) |
| `data/` | datasets, the frame cache, and the train/val/test loaders (val holds out whole subjects, or one of k folds of them) |
| `augment.py` | training-time shifts, cutout, dropped inputs and flips, the same at every timestep |
| `train.py` | the training loop (rate or TET loss, seeding) and `forward_over_time`, the forward pass every number comes from |

## 4. Deploy and check

| file | what it does |
|---|---|
| `folding.py` | folds BatchNorm into the weights (tail and PTQ modes) |
| `quantize.py` | the deployment audit and `hardware_export` |
| `hs_export.py` | the deployed network as integers; `IntegerSim`, which runs simpleSim's arithmetic; the bias neurons; the unroll into hs_api's dictionaries |
| `synops.py` | synaptic operations, the energy proxy |

## 5. Run many of them

| file | what it does |
|---|---|
| `pipeline.py` | config to dataset to model to report, shared by `single` and `search` |
| `preflight.py` | the config rules a search must pass, and `main.py preflight`: every stage of a search, small, before the real one |
| `search.py` | the Ray trial, the Optuna/ASHA wiring, re-scoring the finalists with new seeds, and `run_jobs` for fixed lists of runs |
| `scheduling.py` | early stopping that never cuts the balanced block |
| `study.py` | `main.py study`: one option flipped at a time on a chosen network, trained in full |
| `folds.py` | `single --folds k`: a chosen network trained once per cross-subject fold, and `folds.json`, validation per fold with its mean and standard error |
| `streaming.py` | writes each trial to `leaderboard.csv` and `trials.jsonl` as it finishes |
| `results.py` | the results directory and the end-of-search trial table |
| `report.py` | `report.html` |
| `cli.py` | `python main.py check | summary | preflight | search | study | single | report` |
| `_torch.py` | imports torch if present, so stages 1 and 2 run without it |

## The numbers a trial reports

| column | what it measures |
|---|---|
| `hw_val_accuracy` | the deployed network in PyTorch, on validation (held-out subjects) |
| `sim_val_accuracy` | the same clips through `IntegerSim` |
| `sim_disagreements` | clips where those two predict differently; should be 0 |
| `deployed_tau` | the tau each layer deploys with (`tau` is only where training started); a per-channel layer shows its range |
| `hw_test_accuracy` | the test set; measured once by `single`, never inside the search |

After the search, `finalists.csv` has each finalist's score for every seed,
and `best_finalist.json` the one with the best mean.
