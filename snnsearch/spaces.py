"""The searchable space, as a picklable module-level object.

Ray checkpoints the searcher by pickling it, so this cannot be a closure.

Moved verbatim from Practice2.py lines 2360-2578 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

from .config import (InputSpec, ConvLayerSpec, EncoderSpec, OutputSpec,
                     DownsampleSpec, HeadSpec, NeuronSpec, TrainSpec,
                     parse_fc_widths)
from .hardware import HW_TAU_CHOICES, check_feasibility


# NARROWED after the 65-trial analysis (see DVS_Search_Statistical_Analysis).
# 512 was the only hidden-FC width that showed up in a strong trial; the head is
# rarely used at all now (fc_layers capped at 1), so keep a small set.
FC_WIDTH_CHOICES = [128, 256, 512]


def config_to_specs(config: dict) -> dict:
    """Flat Ray/Optuna config dict -> the structured cfg every other function takes.

    Encoder is assembled per-layer when per-layer keys (k_0, ch_0, ...) are
    present, otherwise from the uniform depth/channels/kernel_size fields."""
    import json
    n_fc = int(config.get("fc_layers", 1))
    widths = [str(config[f"fc_width_{i}"]) for i in range(n_fc) if f"fc_width_{i}" in config]
    norm = config.get("norm", "bn")
    depth = int(config["depth"])

    if "ch_0" in config:  # per-layer search emitted individual layer keys
        layers = []
        for i in range(depth):
            layers.append({
                "out_channels": config[f"ch_{i}"],
                "kernel_size": config[f"k_{i}"],
                "stride": config.get(f"stride_{i}", 1),
                "pool": bool(config.get(f"pool_{i}", False)),
            })
        encoder = EncoderSpec(layers_json=json.dumps(layers), bias=(norm == "none"),
                              norm=norm, tdbn_alpha=config.get("tdbn_alpha", 1.0),
                              dropout_rate=config.get("conv_dropout", 0.0))
    else:                 # uniform
        encoder = EncoderSpec(depth=depth, channels=config["channels"],
                              kernel_size=config["kernel_size"], stride=config.get("stride", 1),
                              padding=0, dilation=1, bias=(norm == "none"),
                              norm=norm, tdbn_alpha=config.get("tdbn_alpha", 1.0),
                              dropout_rate=config.get("conv_dropout", 0.0))

    return {
        "input": InputSpec(N=config["N"], C=2, H=128, W=128,
                           T=config["T"], resize_to=config["resize_to"]),
        "encoder": encoder,
        "output": OutputSpec(num_classes=11),
        "downsample": DownsampleSpec(mode=config.get("downsample_mode", "stride"),
                                     pool_type=config.get("pool_type", "avg"),
                                     pool_kernel_size=2, pool_stride=2),
        "head": HeadSpec(final_reduction=config.get("final_reduction", "flatten"),
                         fc_widths=",".join(widths),
                         dropout_rate=config.get("dropout_rate", 0.5)),
        # v_threshold starts at 1.0 = exactly INT16_MAX after quantization; the
        # sigmoid parametrization keeps it inside (0, 1] if it is learned.
        "neuron": NeuronSpec(neuron_type="LIF", tau=int(round(float(config["tau"]))),
                             v_threshold=1.0, v_reset=0.0,
                             trainable_tau=bool(config.get("trainable_tau", False)),
                             trainable_threshold=bool(config.get("trainable_threshold", False)),
                             integer_leak=bool(config.get("integer_leak", True))),
        "train": TrainSpec(epochs=config["epochs"],
                           optimizer=config.get("optimizer", "adam"),
                           lr=config.get("lr", 1e-3),
                           weight_decay=config.get("weight_decay", 0.0),
                           scheduler=config.get("scheduler", "cosine"),
                           warmup_epochs=config.get("warmup_epochs", 0),
                           label_smoothing=config.get("label_smoothing", 0.0),
                           grad_clip=config.get("grad_clip", 0.0),
                           rate_penalty=config.get("rate_penalty", 0.0),
                           qat_mode=config.get("qat_mode", "inline"),
                           qat_warmup_frac=config.get("qat_warmup_frac", 0.25),
                           qat_epochs=config.get("qat_epochs", 4),
                           qat_lr_scale=config.get("qat_lr_scale", 0.5),
                           fold_bias_mode=config.get("fold_bias_mode", "threshold"),
                           fold_bias_margin=config.get("fold_bias_margin", 0.05)),
    }


# Historical categorical choices, retained for callers that import them.
# The active space below samples channels from 8 through 128 on a log
# scale and odd kernels from 3 through 9. These lists no longer define
# the sampling ranges; the earlier narrowing used a fixed resolution.
CHANNEL_CHOICES = [32, 64]
KERNEL_CHOICES = [5, 7]


def geometry_fits(flat) -> bool:
    """Does the shape sampled so far satisfy the chip's connection limits?

    Only shape decides this: kernel, channels, depth, downsampling, resolution
    and the head. Nothing about the optimizer or the regularizer can move a
    fan-in, which is why this can run before any of them are sampled.

    Returns False on a config that will not even build, since a feature map
    that collapses below the kernel is infeasible in the same practical sense.
    """
    try:
        spec = config_to_specs({"tau": 2, **flat})
        ok, _ = check_feasibility(spec["input"], spec["encoder"],
                                  spec["downsample"], spec["head"], spec["output"])
        return bool(ok)
    except Exception:
        return False


class DefineByRunSpace:
    """
    Optuna define-by-run space, as a MODULE-LEVEL CALLABLE OBJECT rather than a
    closure.

    Why a class: Ray Tune periodically checkpoints the search algorithm, and
    OptunaSearch.save() pickles the space object. A function defined inside
    another function is a local object that pickle cannot serialise
    ("Can't get local object 'make_define_by_run.<locals>.define_by_run'"), which
    aborts the whole study at the first checkpoint. An instance of a top-level
    class pickles fine -- its state is just the plain ints/strs/lists below.

    Define-by-run (vs a flat dict) is what lets the space be honestly
    conditional: a per-layer kernel size k_3 is only suggested on trials whose
    depth >= 4, and fc_width_1 only when there is a 2nd FC layer -- so Optuna's
    surrogate never sees a dimension that had no effect on that trial's score.

    per_layer=True  -> kernel/stride/channels/pool are sampled INDEPENDENTLY per
                       conv layer (answers "is kernel size independently
                       optimised?" -- yes).
    per_layer=False -> one kernel/channels/stride shared by all layers (the
                       older uniform space), useful as a cheaper baseline.
    """

    def __init__(self, batch_size: int, epochs: int, data_dir_abs: str,
                 t_choices: list, per_layer: bool = True, require=()):
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.data_dir_abs = str(data_dir_abs)
        self.t_choices = list(t_choices)
        self.per_layer = bool(per_layer)
        # Knobs whose on/off switch is removed, so every trial gets them.
        # Naming one drops its `use_*` categorical entirely rather than pinning
        # it True, because a dimension with one value teaches TPE nothing and
        # still costs a column.
        self.require = tuple(require or ())

    def __call__(self, trial):
        batch_size = self.batch_size
        epochs = self.epochs
        data_dir_abs = self.data_dir_abs
        t_choices = self.t_choices
        per_layer = self.per_layer
        require = self.require
        # ---- encoder depth ----
        # winners are depth 2-3 (Kruskal-Wallis favoured shallow, q=0.08); depth
        # 5 only ever reached 0.77. Keep 2-4 so depth-3 branches stay in play.
        # GEOMETRY FIRST, AND NOTHING ELSE UNTIL IT PASSES.
        #
        # A configuration rejected by the connection limits used to be scored 0
        # after every knob had been sampled, so Optuna recorded "weight_decay
        # 1e-3 with dropout 0.2 scored zero" for a trial where neither was
        # tested. 82 of 400 trials did that, and the sampler then steered away
        # from regularization values whose only crime was appearing next to an
        # illegal shape.
        #
        # Define-by-run fixes it for free: a dimension a trial never suggests
        # does not exist for that trial, so TPE's model of it never sees the
        # zero. Sample the shape, check it, and return early if it fails. The
        # trial still scores 0, but only against the shape knobs, which is the
        # only thing that was actually wrong with it.
        geom = {}
        depth = geom["depth"] = trial.suggest_int("depth", 2, 4)

        if per_layer:
            # independent geometry per layer
            for i in range(depth):
                geom[f"k_{i}"] = trial.suggest_int(f"k_{i}", 3, 9, step=2)
                geom[f"ch_{i}"] = trial.suggest_int(f"ch_{i}", 8, 128, log=True)
                # each layer independently: downsample by stride-2, by pooling,
                # or not at all (stride 1, no pool -> size-preserving).
                # each layer independently: downsample by stride-2, by pooling,
                # or not at all. The chosen branch sets the flat key
                # config_to_specs reads; the others stay absent and default off.
                ds = geom[f"ds_{i}"] = trial.suggest_categorical(
                    f"ds_{i}", ["stride", "pool", "none"])
                if ds == "stride":
                    geom[f"stride_{i}"] = trial.suggest_int(f"stride_{i}", 2, 2)
                elif ds == "pool":
                    geom[f"pool_{i}"] = trial.suggest_int(f"pool_{i}", 1, 1)
        else:
            # log scale: the useful range spans 8 to 128 and the
            # interesting differences are multiplicative. The paper
            # reached its best with 6 and 16, which the old floor of 32
            # excluded outright.
            geom["channels"] = trial.suggest_int("channels", 8, 128, log=True)
            # odd sizes only, so padding stays symmetric.
            geom["kernel_size"] = trial.suggest_int("kernel_size", 3, 9, step=2)
            mode = geom["downsample_mode"] = trial.suggest_categorical(
                "downsample_mode", ["stride", "pool"])
            if mode == "stride":
                # stride=1 never downsamples -> the flatten explodes and the
                # config is infeasible every time (all the fc_in fan-in busts in
                # the data came from here). Force stride-2.
                geom["stride"] = trial.suggest_categorical("stride", [2])

        # resize_to=0 (native 128x128) is omitted: 128*128*2 = 32,768 axons is
        # far over the limit, so it could never pass feasibility.
        # step=8 keeps the conv arithmetic on friendly sizes; 88 is the largest
        # multiple of 8 under the measured 16,000 axon ceiling (15,488). 89 is
        # the true maximum for two channels but breaks the step.
        geom["resize_to"] = trial.suggest_int("resize_to", 24, 88, step=8)
        geom["T"] = trial.suggest_categorical("T", t_choices)

        # ---- head: GAP vs flatten, then variable-depth FC ----
        # GAP collapses HxW before the head (Q4): far fewer params, less
        # overfitting, and deployable. When GAP is chosen the huge first-FC
        # fan-in disappears, so many more configs pass feasibility.
        # final_reduction is searched again. The old "GAP costs ~40 points"
        # result was measured when resolution never varied, and flatten's
        # fan-in is exactly what makes high resolution infeasible, so the two
        # have to be compared across resolutions.
        # fc_layers 0-1 only: every top-10 model had 0 hidden FC (Kruskal-Wallis
        # q=0.04), so 2-3 hidden layers are pure waste. Keep 1 as a branch.
        n_fc = geom["fc_layers"] = trial.suggest_int("fc_layers", 0, 1)
        for i in range(n_fc):
            geom[f"fc_width_{i}"] = trial.suggest_categorical(
                f"fc_width_{i}", FC_WIDTH_CHOICES)
        # Moved up from the bottom of this function, because the head decides
        # the flatten fan-in and the flatten fan-in is what the connection
        # limits reject. Checking before it is sampled would check the wrong
        # network.
        geom["final_reduction"] = trial.suggest_categorical(
            "final_reduction", ["flatten", "gap"])

        consts = {"N": batch_size, "epochs": epochs, "data_dir": data_dir_abs,
                  "pool_type": "avg"}

        if not geometry_fits({**consts, **geom}):
            # Stop here. Nothing below this line gets sampled, so no training
            # knob is blamed for a shape that will not fit. tau is supplied
            # because config_to_specs reads it directly and the trial still has
            # to build far enough to report itself infeasible.
            return {**consts, "tau": 2, "infeasible_geometry": True}

        # ---- neuron: INTEGER leak, per-layer (Q2, Q3) ------------------------
        # tau was suggest_float(1.5, 2.5). On chip the leak register is an
        # integer and the converter's convention is leak = round(tau), so EVERY
        # value in that interval deployed as leak=2: the search was resolving a
        # difference that does not survive conversion, and the float metric was
        # rewarding it. The old Spearman rho=-0.60 for tau is therefore a
        # correlation with a coordinate the hardware cannot represent -- it does
        # not carry over, which is why the range is reopened rather than
        # narrowed around the old winners.
        trial.suggest_categorical("tau", HW_TAU_CHOICES)
        # Now genuinely A/B-able: both are quantized in the forward pass, and
        # hs_api reads a neuron model per neuron key, so per-layer leaks and
        # thresholds deploy as-is.
        trial.suggest_categorical("trainable_tau", [False, True])
        trial.suggest_categorical("trainable_threshold", [False, True])

        # ---- regularization ----
        # dropout: not significant, but winners live in 0.1-0.45; trim the tails.
        # This one sits in the HEAD, before each linear layer.
        trial.suggest_float("dropout_rate", 0.1, 0.45)
        # Spatiotemporal dropout inside the conv stack, on the spikes leaving
        # each block. With fc_layers capped at 1 the head holds a single dropout
        # in front of the classifier, so nothing regularizes the layers that
        # build the features. Conditional, so "none in the conv stack" stays a
        # reachable baseline rather than a measure-zero point in a range.
        if "conv_dropout" in require or \
                trial.suggest_categorical("use_conv_dropout", [False, True]):
            trial.suggest_float("conv_dropout", 0.05, 0.3)
        # norm=none is GONE: it produced every dead (below-chance) network
        # (pooled Mann-Whitney p=0.0015; learned-vs-dead odds ratio 21.8). Only
        # the two foldable, hardware-legal options remain -- and this lets tdBN
        # get a fair test against plain BN on the good backbone.
        norm = trial.suggest_categorical("norm", ["bn", "tdbn"])
        if norm == "tdbn":
            trial.suggest_float("tdbn_alpha", 0.5, 2.0)
        trial.suggest_float("label_smoothing", 0.0, 0.2)
        trial.suggest_categorical("grad_clip", [0.0, 1.0, 5.0])
        # Global L1 on the mean spatiotemporal firing rate. Conditional rather
        # than a range that includes zero: a continuous range never samples
        # exactly 0, so "off" would never be tested, and off is the baseline
        # every other trial has already been run under.
        if "rate_penalty" in require or \
                trial.suggest_categorical("use_rate_penalty", [False, True]):
            trial.suggest_float("rate_penalty", 1e-4, 1e-1, log=True)

        # ---- optimizer / LR schedule ----
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        # lr: positive correlation (Spearman q=0.04); winners are 1.6e-3..3.5e-3,
        # so drop the under-converging low end (was 1e-4).
        trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        # weight_decay is the main lever on whether FOLDED weights fit inside the
        # representable [-1, 1] band (w_alpha=1). The old ranges were fitted
        # against a float objective that could not see clipping, so their floors
        # (1e-8) are almost certainly too low now that clipping is scored --
        # negligible decay lets the folded distribution sprawl past the grid.
        # Floors raised; the search re-decides the rest against hw_val_accuracy.
        if optimizer == "adamw":
            trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)    # decoupled
        else:
            trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        sched = trial.suggest_categorical("scheduler", ["cosine", "onecycle", "step", "none"])
        if sched in ("cosine", "step"):
            trial.suggest_int("warmup_epochs", 0, 3)

        return consts


def make_define_by_run(batch_size: int, epochs: int, data_dir_abs: str, t_choices: list,
                       per_layer: bool = True, require=()) -> "DefineByRunSpace":
    """Factory kept for API compatibility -- returns a picklable space object."""
    return DefineByRunSpace(batch_size, epochs, data_dir_abs, t_choices, per_layer,
                            require)
