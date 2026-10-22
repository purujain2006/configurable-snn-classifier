# snnsearch user guide

snnsearch searches for spiking convolutional networks that run on HiAER-Spike.
You give it a dataset and a config file. It trains hundreds of networks, keeps
only the ones that fit the chip's limits, and scores each one the way the chip
would run it. The best network comes back checked against the chip's simulator,
clip by clip.

This guide explains how to use the tool. The website explains why it works the
way it does, at https://purujain2006.github.io/configurable-snn-classifier/

Each command is a single line, so it works in any shell (bash, zsh or tcsh).

## Contents

1. [The workflow](#1-the-workflow)
2. [Install](#2-install)
3. [Data](#3-data)
4. [Write a config](#4-write-a-config)
5. [Preflight](#5-preflight)
6. [Run the search](#6-run-the-search)
7. [Read the results](#7-read-the-results)
8. [Option study](#8-option-study)
9. [Train the winner and test it](#9-train-the-winner-and-test-it)
10. [Check on the chip's simulator](#10-check-on-the-chips-simulator)
11. [Troubleshooting](#11-troubleshooting)
12. [Config reference](#12-config-reference)
13. [What the search tries](#13-what-the-search-tries)
14. [For developers](#14-for-developers)

## 1. The workflow

| step | command | how long |
|---|---|---|
| install, once per machine | `python main.py check` | minutes |
| preview a network's size and fit, no GPU needed | `python main.py summary -c CONFIG` | seconds |
| preflight, whenever the config or the code changes | `python main.py preflight -c CONFIG` | about 15 minutes |
| search | `python main.py search -c CONFIG` | hours to days, depending on the GPUs |
| look at results while it runs | `python tools/analysis/top_trials.py RESULTS` | seconds |
| retrain the best trials of a stopped search | `python main.py finalists -c CONFIG --top 3` | hours |
| which options help | `python main.py study -c CONFIG --from-best RESULTS/best_finalist.json` | hours |
| choose between finished networks on every person (k folds) | `python main.py single -c CONFIG --from-best RESULTS/best_finalist.json --epochs 300 --folds 5` | k times one `single` |
| train the winner long, get a test score | `python main.py single -c CONFIG --from-best RESULTS/best_finalist.json --epochs 300` | hours |

`CONFIG` is a file in `configs/`, and `RESULTS` is the folder named by
`run.results_dir` in it, for example `results/dvs128_search05`. The times are
rough and scale with your hardware. More or faster GPUs shorten every step that
trains.

## 2. Install

Use a conda environment on the machine with the GPU. Install PyTorch first,
because spikingjelly can otherwise pull a different build:

```
conda create -n snn python=3.9
conda activate snn
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install spikingjelly
pip install "pydantic<3" "ray[tune]" optuna pyyaml
```

`ray[tune]` alone does not always bring pydantic, and without it the search
dies at startup with `ray.train.v2 requires the pydantic package`. Name it
explicitly, as above.

hs_api (the chip's software) is only needed to compare against the real
simpleSim (section 10). Install it with poetry from its own repository.

Then check:

```
python main.py check
```

Every line should say `ok` except, possibly, `hs_api`.

If you copy the code to another machine by hand rather than with git, copy the
whole `snnsearch/` folder. `git commit -a` leaves out new files, so `git add`
them first. The preflight catches a missing file.

## 3. Data

### Built-in datasets

| `dataset.name` | what | input |
|---|---|---|
| `dvs128` | DVS128 Gesture, 11 hand gestures from an event camera | 2 x 128 x 128 events |
| `cifar10`, `cifar100` | photos | 3 x 32 x 32 |
| `mnist`, `fashion_mnist` | digits, clothing | 1 x 28 x 28 |

For DVS128 Gesture, `dataset.root` is the folder holding `events_np/` (or, the
first time, `download/` with `DvsGesture.tar.gz`, `gesture_mapping.csv`,
`LICENSE.txt` and `README.txt` from IBM's Box folder). The tool builds one
frame cache per value of T beside them, `frames_number_16_split_by_number/`
and so on, tens of minutes each the first time.

Keep the dataset on storage that persists between jobs. A copy on a compute
node's local scratch disk can be wiped.

### Your own dataset

Write a Python file with a function that returns a `DatasetBundle`, and point
the config at it:

```yaml
dataset:
  module: /path/to/my_data.py
  factory: make_datasets
  root: /path/to/data
```

`examples/cifar10_data.py` is a worked example and
`examples/synthetic_data.py` a minimal one. The bundle says:

| field | meaning |
|---|---|
| `train`, `test` | PyTorch datasets returning `(x, label)` |
| `C`, `H`, `W` | one sample's shape before any resize |
| `num_classes` | number of classes |
| `is_event` | True for event data (already spikes), False for images |
| `hflip_ok` | True if a mirrored sample keeps its label, which allows flips in augmentation |
| `collate_fn` | only if samples need padding to batch, as event clips do |

Images should be floats in [0, 1]. Event data should be per-pixel event counts.

The search reads the shape and class count from the bundle and works out the
range of input sizes from them, so nothing in the config is specific to a
dataset except `dataset:` itself.

## 4. Write a config

Copy the nearest example and edit it:

* events: `configs/dvs128_search06.yaml` (the next search, after `configs/dvs128_search05.yaml`)
* images: `configs/cifar10.yaml`

Every line in those files is commented. The settings people change most:

| setting | what to consider |
|---|---|
| `search.trials` | 400 is a thorough search, 100 a quick one |
| `search.epochs` | the budget for one trial. The first quarter trains in floating point and the rest on the chip's integer weights |
| `search.grace_period` | when early stopping first judges a trial, which must come after the first quarter of the epochs |
| `search.balanced_trials` | trials at the start in which every option is tried equally often. Use about a quarter of the trials, a multiple of 6 (12 with four T values) |
| `search.T_choices` | timesteps per sample to try. Each one needs its own frame cache for event data, and a trial's time and memory grow with its T. With more than one, the balanced block gives each the same number of trials |
| `search.gpu_fraction` | the share of a GPU each trial gets, and a hard cap on its memory. 0.25 runs 4 at once per GPU, and 0.167 only 5 (Ray rounds down). The preflight measures what fits |
| `search.finalists`, `search.finalist_seeds` | how many of the best trials to retrain at the end, and with how many seeds |

Section 12 lists every setting.

## 5. Preflight

```
python main.py preflight -c configs/dvs128_search05.yaml
```

Run it on the machine and in the environment the search will use. It takes
about 15 minutes and checks, in order:

| stage | checks |
|---|---|
| config | settings that contradict each other |
| environment | every package imports, every package file exists, a GPU is visible, Ray's scratch folder is writable and has room |
| dataset | the dataset loads, which frame caches still need building |
| search space | the balanced block can be drawn, how many random proposals fit the chip, every block configuration builds, folds and exports |
| memory | the heaviest balanced-block trials, with both norms and every memory-adding option on, trained for a step in each phase on the GPU. FAIL unless the heaviest fits its own cap (`gpu_fraction` of the card) and the trials Ray packs onto one card, each with its CUDA context, fit the card. Prints the `gpu_fraction` that fits the most trials. Then it climbs to the heaviest trial Optuna could propose after the block and WARNs if it exceeds the cap, with the share of proposals that would |
| exactness | two small networks trained on your data and checked clip by clip against the simulator |
| mini search | a real search through Ray with six trials, the balanced block, early stopping, the simulator check, finalists and the report |

It prints PASS, WARN or FAIL for each, with what to fix, and writes
`preflight.json` into the results folder. `--quick` skips the last two stages
(about 2 minutes) but a search still needs the full preflight.

The search will not start until a full preflight has passed for the same
config and the same code. Changing `search.trials` keeps it valid. Changing
anything else, or pulling new code, needs a new one. `--skip-preflight` on
`search` starts anyway.

## 6. Run the search

On a remote machine, the search must keep running after you log out. Use tmux:

```
tmux new -s search
```

then, inside it:

```
conda activate snn
cd /path/to/configurable-snn-classifier
python main.py search -c configs/dvs128_search05.yaml |& tee search.log
```

Press `Ctrl-b` then `d` to leave it running. `tmux attach -t search` brings it
back. Without tmux:

```
nohup python main.py search -c configs/dvs128_search05.yaml >& search.log &
```

What happens:

1. Frame caches for every T are built if missing.
2. The balanced block runs. Every option appears equally often, each with a
   random network that fits the chip, and none is stopped early.
3. Optuna proposes the rest, steering toward what has scored well. Early
   stopping cuts the weaker ones at `grace_period`.
4. The top `finalists` are retrained from scratch with `finalist_seeds` new
   seeds each, and the best average wins.

To stop it, run `pkill -f "main.py search"` and then `ray stop`. A stopped search
cannot be resumed, and it skips step 4. To run step 4 on what it finished:

```
python main.py finalists -c configs/dvs128_search05.yaml --top 3
```

It reads `trials.jsonl`, retrains the best finished trials with new seeds
(`--seeds`, default `search.finalist_seeds`), and writes `finalists.csv` and
`best_finalist.json` as the search would have.

Every trial is scored on the validation set only. The search never uses the
test set, so the test score in section 9 is measured on clips nothing was
chosen by.

## 7. Read the results

Everything is in the results folder, written as trials finish:

| file | contents |
|---|---|
| `leaderboard.csv` | one row per trial, with its settings, scores and size |
| `trials.jsonl` | the same, with every setting, one JSON record per line |
| `trial_progress.jsonl` | every trial's score after every epoch |
| `best.json` | the best trial so far |
| `infeasible.jsonl` | proposals the chip check rejected, and why |
| `finalists.csv` | at the end, each finalist's score for every new seed |
| `best_finalist.json` | at the end, the winner by its average over seeds |
| `report.html` | charts, which `python main.py report RESULTS` rebuilds at any time |
| `preflight.json` | the last preflight |

The scores:

| column | meaning |
|---|---|
| `hw_val_accuracy` | the deployed network (integer weights, chip arithmetic) on the validation set |
| `sim_val_accuracy` | the same clips through the simulator's arithmetic (IntegerSim) |
| `sim_disagreements` | clips where the two predict differently, always 0 |
| `deployable` | fits the chip and survived export |
| `deployed_tau` | the leak each layer deploys with |
| `threshold_bits` | bits the widest threshold needs on chip: 16 normally, 18 in a layer behind an average pool (the fold makes it 4 x theta + 3). The chip field is `limits.threshold_bits` |
| `float_val_accuracy` | the best validation score of the float warm-up (the first quarter of the epochs) |
| `grid_val_accuracy` | the best validation score of the integer phase, averaged over `val_smoothing_epochs` epochs. It picks the checkpoint that deploys |
| `pre_export_val_accuracy` | that checkpoint on its own, before export |
| `quant_gap` | `pre_export_val_accuracy` minus `hw_val_accuracy`, what export cost. Export is exact, so it is zero up to rounding. A number far from zero means the deployed network is not the trained one |
| `end_to_end_gain` | `hw_val_accuracy` minus `float_val_accuracy`, the progress made after the warm-up. It comes mostly from the integer phase's extra epochs and says nothing about export |

The leaders so far:

```
python tools/analysis/top_trials.py results/dvs128_search05
```

Which options did well, judged fairly on the balanced block:

```
python tools/analysis/factor_effects.py results/dvs128_search05 --block 96
```

Read its last column (every trial compared at the same epoch) together with
the finished medians. Validation sets are small. With about 200 clips, a score
moves by roughly 3 points by chance, so treat smaller gaps as ties. For a
clean answer about one option, use the option study.

Accuracy per person and per lighting, for checkpoints `single` wrote:

```
python tools/analysis/per_subject.py -c configs/dvs128_search05.yaml results/search05_top5/rank1 results/search05_top5/rank2
```

Validation and test are different people, so a run can score higher on test
than on validation when the validation people are harder to read. This shows
it person by person, one column per checkpoint, and checks that its overall
scores equal the ones in each `single.json`. A person has about 50 clips, so a
pattern every checkpoint shares means more than one number. A folder of fold
runs (`single --folds`, section 9) counts as one checkpoint. Each fold's
network is scored on the people it held out, so every training person gets a
score from a network that never trained on them. Fold runs have no test
column.

Accuracy and SynOps when the chip gets only the first k of a clip's T frames:

```
python tools/analysis/early_exit.py -c configs/dvs128_search05.yaml --test results/search05_top5/rank1 results/search05_top5/rank2
```

SynOps grow with the frames processed, and the host can stop sending input
early. Each DVS128 frame holds the same number of events, so k frames are the
first k/T of the gesture. This feeds each deployed network the first k frames
(every 8 up to T by default, `--k 4,12,20` for others) plus the flush steps,
in the chip's arithmetic, one column per checkpoint, and checks that the
k = T row equals each `single.json`. It also shows by which frame each clip's
running prediction had stopped changing, a cheap early-exit signal. No
training.

SynOps again, for checkpoints already trained, with today's counter:

```
python tools/analysis/recount_synops.py -c configs/dvs128_search05.yaml results/search05_top5 --out results/search05_top5/synops_recount.csv
```

A run measures SynOps once, when it finishes. When the counter is corrected
later, as it was for signed spikes and for layers behind an average pool, the
old figure in `single.json` stays wrong. This rebuilds each saved network exactly as it was scored, runs the
validation clips through the current counter, and prints the new figure next
to the old one. No training.

## 8. Option study

```
python main.py study -c configs/dvs128_search05.yaml --from-best results/dvs128_search05/best_finalist.json
```

Takes the chosen network and, for every option the search covers (pooling,
bias, head, signed spikes, input bits, tau, loss, augmentation, conv dropout,
firing-rate penalty), trains a copy with only that option changed. Every
variant trains for the full epochs with the same 3 seeds, with no early
stopping. A variant the chip cannot hold is listed with the reason and skipped.

It writes `RESULTS/study/study_summary.csv`, with one line per variant:

| column | meaning |
|---|---|
| `mean_val_accuracy` | average over seeds |
| `diff_vs_baseline` | the variant minus the unchanged network, seed by seed, averaged |
| `diff_se` | its standard error |
| `verdict` | `helps` or `hurts` when the difference is more than twice its standard error, otherwise `no clear difference` |

About 13 variants times 3 seeds, so roughly 40 full trainings. `--only
augment,loss` studies just those; `--seeds 5` tightens the verdicts.

## 9. Train the winner and test it

```
python main.py single -c configs/dvs128_search05.yaml --from-best results/dvs128_search05/best_finalist.json --epochs 300
```

Replays the winning configuration exactly (architecture, input size, timesteps,
training recipe), for as long as you ask, then scores it once on the test set:
`hw_test_accuracy` in `single.json`. That is the number to report. The
checkpoint (`best.pth`) holds the deployed network.

`single.json` also holds `trained`, a summary of what trained (input
size and timesteps, every layer with its map size, head, neuron, recipe and
seed), and `config`, the full resolved settings. `config_file` only names the
YAML given with `-c`. On a `--from-best` run the trial decides the
architecture, input size and schedule, so the file's own values do not describe
the run.

`--train key=value` changes one training setting for an experiment, for
example `--train weight_decay=1e-3`, and prints every change.

### Choose between networks on every person: k folds

Validation is one fixed set of people. On DVS128 Gesture, `run.seed` 1 holds
out people 6, 10, 17 and 21 (216 clips). People differ in how hard they are to
read, and three of those four score below the test average for every network,
so validation reads about 6 points low, and another draw of people would move
every score. To choose between a few finished networks without that luck,
score each one with k-fold cross-subject validation:

```
python main.py single -c configs/dvs128_search05.yaml --from-best results/search05_top5/rank1/trial.json --epochs 300 --folds 5 --results-dir results/search05_folds/rank1
```

`--folds 5` splits the training people into 5 groups of near-equal clip counts
(`search.val_folds`, shuffled by `run.seed` so every network gets the same
groups) and trains the network 5 times, one after another. Each time one group
is validation and the other people train. Every training person is held out
exactly once. Each fold is an ordinary `single` run in `RESULTS/fold0` to
`fold4`. After each fold, `RESULTS/folds.json` is rewritten from the fold
folders. It holds each fold's validation people, `hw_val_accuracy` and last
epoch's training accuracy, then the mean over folds with its standard error
and the score pooled over every clip. It also warns when the folds did not
all train the same network and recipe. Fold runs never score the test set
(`single.json` says `test_skipped`).

It costs k times the training of one run, per network: 5 folds of a 300-epoch
replay take as long as five 300-epoch replays. So it is for the finalists and
networks you are choosing between, not for search trials. To spread the folds
over GPUs, train one fold per GPU with `--fold`. Each finishing fold rewrites
`folds.json` with every fold done so far and lists the missing ones.

```
env CUDA_VISIBLE_DEVICES=0 python main.py single -c configs/dvs128_search05.yaml --from-best results/search05_top5/rank1/trial.json --epochs 300 --folds 5 --fold 0 --results-dir results/search05_folds/rank1
```

For several training seeds as well, give each seed its own folder, for
example `--train seed=1001 --results-dir results/search05_folds/rank1_seed1001`.
Compare networks by their mean over folds. A gap smaller than about twice the
two standard errors combined is a tie. Then train the chosen one on the usual
split and open the test set once, as above. Every training person's accuracy
across the folds:

```
python tools/analysis/per_subject.py -c configs/dvs128_search05.yaml results/search05_folds/rank1 results/search05_folds/rank2
```

## 10. Check on the chip's simulator

Every trial already runs its validation clips through IntegerSim, an exact
copy of the simulator's arithmetic. To check IntegerSim itself against hs_api's
real simpleSim, on small networks of every layer type (needs hs_api):

```
python tools/match_simplesim.py --selftest
```

To compare a trained checkpoint clip by clip against IntegerSim, and with
`--real-simplesim K` also run K clips through the real simpleSim. That needs
hs_api and is slow, about a minute per clip for a full network.

```
python tools/match_simplesim.py --ckpt results/RUN/best.pth --data-root /path/to/data --real-simplesim 2
```

Where hs_api is needed, run from its poetry environment (`poetry run python
...` inside hs_api's folder).

## 11. Troubleshooting

| message | cause | fix |
|---|---|---|
| `dataset root does not exist` | `dataset.root` points nowhere on this machine | fix the path in the config, or pass `--data-root` |
| `ray.train.v2 requires the pydantic package` | Ray installed without its tune extras | `python -m pip install "pydantic<3" "ray[tune]"` |
| `No module named 'snnsearch.something'` | a package file never reached this machine | `git add` the file, then commit, push and pull. The preflight's environment stage lists missing files |
| `No preflight has passed for this search yet` | the search needs a preflight first | run `main.py preflight -c CONFIG` |
| `The config or the code changed since the preflight` | you edited the config or pulled code | run the preflight again |
| `search.grace_period is N, but the first M of E epochs are the float warm-up` | early stopping would judge trials before they train on integer weights | raise `grace_period` above M |
| `CUDA out of memory` in trials | too many trials share a GPU | raise `search.gpu_fraction` for fewer trials at once. The preflight suggests a value |
| `ray scratch: skipping ... no room for Ray's socket path` | the path is too long for a unix socket | set `search.ray_temp_dir` to a short path on a large disk |
| frame cache incomplete | a cache build was interrupted | `python tools/build_cache.py ROOT 16`, or delete that `frames_number_*` folder |
| the search stops when you log out | it ran in your login shell | run it in tmux or with nohup (section 6) |
| `sim_disagreements` above 0 | should not happen, since the deployed network is measured in float64 | report it with the trial's config |

## 12. Config reference

Every setting, with its default. A config file only needs what differs.

### run

| key | default | meaning |
|---|---|---|
| `name` | `run` | name of the run, used for Ray's folders |
| `results_dir` | `results/<name>` | where everything is written |
| `seed` | 1 | fixes the validation split, the balanced block, the sampler and every trial's initial weights |

### dataset

| key | default | meaning |
|---|---|---|
| `name` | `dvs128` | a built-in dataset (section 3) |
| `root` | none | where the data is, with `$USER` and `~` expanded |
| `module` | none | your own dataset file (section 3), used instead of `name` |
| `factory` | `make_datasets` | the function in `module` that returns the bundle |

Any other key is passed to the dataset function, for example `download: true`.

### encoding

| key | default | meaning |
|---|---|---|
| `coding` | `passthrough` for events, `levels` for images | how input becomes chip input. `passthrough` (events), `levels` (pixel values as B-bit numbers), `poisson`, `temporal` or `direct` (analog, cannot run on the chip) |
| `T` | 16 | timesteps per sample when the search does not choose it |
| `resize_to` | none | square input size when the search does not choose it |
| `input_bits` | none (1) | bits per input value when the search does not choose it |

### search

| key | default | meaning |
|---|---|---|
| `trials` | 25 | trials in total, the balanced block included |
| `epochs` | 40 | epochs per trial, the first quarter in floating point |
| `space` | `uniform` | `per_layer` samples every layer's kernel, channels and downsampling separately, and `uniform` shares one. Use `per_layer` |
| `T_choices` | `[encoding.T]` | timesteps to choose from |
| `max_input_bits` | none (4) | bits per input value are searched from 1 to this, for `levels` and `passthrough` |
| `balanced_trials` | 0 | trials at the start in which every option appears equally often (rounded up to a multiple of every choice's level count: 6, or 12 with four T values) |
| `protect_balanced` | true | early stopping never cuts the balanced block |
| `startup_trials` | 10 | random trials after the block, before Optuna steers |
| `scheduler` | `asha` | `asha` stops weak trials early, and `none` runs every trial in full |
| `grace_period` | 15 | epoch of the first early-stopping cut, which must be after the float warm-up |
| `reduction_factor` | 3 | at each cut, the top 1 in this many continue |
| `brackets` | 1 | 2 or more gives some trials a later first cut |
| `finalists` | 3 | best trials retrained at the end (0: off) |
| `finalist_seeds` | 3 | new seeds per finalist |
| `finalist_epochs` | none (`epochs`) | epochs for the finalists |
| `batch_size` | 16 | samples per training step |
| `gpu_fraction` | 1.0 | share of a GPU per trial |
| `num_workers` | 4 | data-loading processes per trial |
| `cpu_per_trial` | none (`num_workers` + 1) | CPU cores reserved per trial |
| `val_split` | `subject` | `subject` holds out whole people (when the dataset names them), and `clip` draws clips at random |
| `val_folds` | 0 | k-fold cross-subject validation for chosen networks (section 9). 2 or more splits the training people into that many groups of near-equal clip counts, shuffled by `run.seed`, and 0 keeps the one `val_split` split. Needs a person id in every clip's file name (`user07_led_3.npz`). `single --folds K` sets it and trains every fold |
| `val_fold` | 0 | which fold is validation when `val_folds` is set, 0 to `val_folds` - 1 (`single --fold I`) |
| `target` | 0.975 | accuracy to call out in the log when reached |
| `require` | `[]` | regularizers every trial uses, from `conv_dropout`, `rate_penalty` and `augment` |
| `max_failures` | 3 | retries for a trial that crashes |
| `ray_temp_dir` | none | Ray's scratch folder, a short path on a large disk |
| `skip_preflight` | false | start without a passed preflight (`--skip-preflight`) |

### train

Settings of the training recipe the search does not choose itself, applied to
every trial. The one worth knowing:

| key | default | meaning |
|---|---|---|
| `qat_schedule_epochs` | none | length of the learning-rate schedule on integer weights, independent of the epoch budget. Set it (40) for long runs |

The rest are the fields of `TrainSpec` in `snnsearch/config.py`. Settings the
search chooses (learning rate, weight decay, dropout and so on) cannot be set
here.

### architecture

Only for `single` without `--from-best`. The network to train, as fields of
`EncoderSpec`, `DownsampleSpec`, `HeadSpec` and `NeuronSpec` in
`snnsearch/config.py`, under `encoder:`, `downsample:`, `head:` and `neuron:`.

### objective

| key | default | meaning |
|---|---|---|
| `mode` | `accuracy` | what the search maximizes. `accuracy`, `constrained` (accuracy within `synops_budget`), `weighted` (accuracy minus `weight` times SynOps over `synops_reference`) or `pareto` (report the trade-off) |
| `synops_budget` | none | for `constrained` |
| `synops_reference` | none | for `weighted` |
| `weight` | 0 | for `weighted` |

SynOps is the number of synaptic events per sample, the usual proxy for a
spiking network's energy. It counts what HiAER-Spike does with the network as
hs_export lays it out, every spike times the synapses of the neuron that fired.

- An average pool or GAP has no neurons on chip. It is folded into the next
  layer, which reads every neuron in front of the pool, each through its own
  synapses. So a layer behind a 2x2 pool counts 4 times what its pooled input
  suggests, and a head behind GAP H x W times.
- A signed spike, +1 or -1, is one neuron firing. It reaches both neurons of
  every signed unit it feeds, so a signed layer receives twice the events.
- A max pool is a layer of OR neurons. It receives one event per spike and
  sends one spike per window and sign.
- With `input_bits` B > 1 each input value is B axons, one per bit, and each
  set bit is one event.
- Padding and zero weights hold no synapse, so pruning lowers SynOps.

Old figures are low. Before 2026-10-20 a layer behind an average pool or GAP
was counted from the pooled mean, and a signed layer like an unsigned one:
Search 05 rank2's conv1, signed behind a 2x2 pool, by about 8 times. Before
2026-10-17 a -1 cancelled a +1 and a level counted as its value. Recount saved
networks with `tools/analysis/recount_synops.py` (section 7).

`single.json` and each record in `trials.jsonl` carry:

| field | meaning |
|---|---|
| `synops_per_sample` | synaptic events per clip, all layers |
| `synops_layers` | one row per chip layer, in order (below) |
| `chip_synapses` | synapses the chip stores, with every weight nonzero (the leaderboard's `connections`) |
| `dense_macs_per_inference` | multiply-accumulates of an ANN of the same shape in one pass, one per weight per output, with no folded pools and no second neuron per sign |
| `synops_over_dense` | `synops_per_sample` over `dense_macs_per_inference` |

Each row of `synops_layers` holds `kind` (conv, linear or orpool), `module` and
`neuron` (the PyTorch modules), `input_pool_area` (neurons folded into each
input value: 4 behind a 2x2 average pool), `synops_per_sample` (events its
synapses receive, from the previous layer's spikes or the input),
`synops_ceiling_per_sample` (if every source fired every step),
`spikes_per_sample` and `firing_rate` (what its own neurons fire, with the rate
in spikes per unit per step), `chip_neurons` and `chip_synapses`.

`firing_rate` in the training records (`trial_progress.jsonl`,
`progress.jsonl`, `trials.jsonl`, the leaderboard) is something else. It is the
unweighted mean of each spiking layer's rate, without the output layer, so a
small layer counts as much as a large one. It is recorded
only when the firing-rate penalty is on. The per-layer rates of the deployed
network are in `synops_layers`.

### limits

For a chip build with other limits. Empty by default, which uses the values in
`snnsearch/hardware.py`:

| key | default | meaning |
|---|---|---|
| `total_axons` | 16000 | input axons in total |
| `axon_fan_out` | 4096 | neurons one input axon may feed |
| `axon_fan_in` | 8191 | input axons that may feed one neuron |
| `neuron_fan_out` | 4095 | neurons one neuron may feed |
| `neuron_fan_in` | 8159 | neurons that may feed one neuron |
| `threshold_bits` | 36 | bits in a neuron's threshold field, the same as its state word on the FPGA. Behind an average pool the chip threshold is 4 x theta + 3, up to 18 bits. A network past the limit deploys with a warning, and `threshold_bits` records what it needs |

### report

| key | default | meaning |
|---|---|---|
| `html` | true | build `report.html` at the end |

## 13. What the search tries

The network's shape covers 2 to 4 convolution layers, each with its own kernel
(3 to 9), channels (8 to 128) and downsampling (stride 2, a 2x2 pool, or none),
the input size and timesteps, and a flatten or global-average-pooling head with
up to one hidden layer. Every proposal is checked against the chip's limits before
it trains.

The options it tries on and off, and that the balanced block and the option
study cover:

| option | levels |
|---|---|
| pooling | none, average (folded into the next layer), max (one OR neuron per window) |
| bias | folded into the threshold, or delivered by always-on bias neurons |
| head | flatten, global average pooling |
| signed spikes | off, on (a unit sends +1 or -1, as two chip neurons) |
| input bits | 1, more (up to 4 axons per input value) |
| tau | fixed, learned per layer, learned per channel (always a power of two) |
| loss | spike rate over the sample, TET (a loss at every timestep) |
| augmentation | off, on (shifts, cutout, dropped inputs, flips where allowed) |
| conv dropout | off, on |
| firing-rate penalty | off, on |
| timesteps | every value of `search.T_choices` (the option study leaves T alone) |

It also chooses the training recipe, meaning the optimizer, learning rate,
weight decay, schedule, dropout, label smoothing, gradient clipping and
normalization.

Pages C and D of the website explain each choice.

## 14. For developers

`snnsearch/README.md` maps every file. Before pushing a change, run:

```
python tools/checks/check_static.py
python tests/test_chip_rules.py
python tests/test_search_space.py
python tests/test_sim_match.py
python tests/test_training_options.py
python tests/test_preflight.py
python tests/test_study.py
python tests/test_input_pipeline.py
python tests/test_frame_cache.py
python tests/test_synops.py
python tests/test_recount_synops.py
python tests/test_per_subject.py
python tests/test_val_folds.py
python tests/test_early_exit.py
python tests/test_docs.py
python tests/smoke_test.py
```

All run on a laptop in a few minutes. Two more:

* `python tests/test_search_e2e.py` runs a real search and option study
  through Ray on synthetic data (about 5 minutes, needs ray and optuna). Run it
  after any change to `search.py`, `scheduling.py`, `study.py` or `streaming.py`.
* `python tools/checks/train_and_verify.py CONFIG` trains a dozen networks on
  real data and checks each against the simulator (on a machine with the dataset and a GPU).

`node tests/test_snn_port.js` checks the website's arithmetic against Python,
and `node tools/site/validate_pages.js` the website's structure.
