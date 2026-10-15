# snnsearch, file by file

A configuration becomes a **plan** (every layer's size), and everything else
reads the plan. The files below are in the order a configuration passes
through them.

## 1. Describe a configuration

| file | what it does |
|---|---|
| `config.py` | the dataclasses: `InputSpec`, `EncoderSpec`, `DownsampleSpec`, `HeadSpec`, `NeuronSpec`, `TrainSpec` |
| `runconfig.py` | loads the YAML run file (`configs/*.yaml`) and its defaults |
| `spaces.py` | the search space: what a trial samples, geometry first |

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
| `model.py` | `DVSGesturePuru`: the network built from the plan |
| `encoders.py` | turns input into spikes (DVS frames pass through and are binarized) |
| `data/` | datasets, the frame cache, and the train/val/test loaders (val holds out whole subjects) |
| `train.py` | the training loop and `forward_over_time`, the forward pass every number comes from |

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
| `search.py` | the Ray trial and the Optuna/ASHA wiring |
| `streaming.py` | writes each trial to `leaderboard.csv` and `trials.jsonl` as it finishes |
| `results.py` | the results directory and the end-of-search trial table |
| `report.py` | `report.html` |
| `cli.py` | `python main.py check | summary | single | search | report` |
| `_torch.py` | imports torch if present, so stages 1 and 2 run without it |

## The numbers a trial reports

| column | what it measures |
|---|---|
| `hw_val_accuracy` | the deployed network in PyTorch, on validation (held-out subjects) |
| `sim_val_accuracy` | the same clips through `IntegerSim` |
| `sim_disagreements` | clips where those two predict differently; should be 0 |
| `deployed_tau` | the tau each layer deploys with (`tau` is only where training started) |
| `hw_test_accuracy` | the six test subjects; measured once by `single`, never inside the search |
