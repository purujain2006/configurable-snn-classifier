"""The searchable space, as a picklable module-level object.

Ray checkpoints the searcher by pickling it, so this cannot be a closure.

Moved verbatim from Practice2.py lines 2360-2578 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

from .config import (InputSpec, ConvLayerSpec, EncoderSpec, OutputSpec,
                     DownsampleSpec, HeadSpec, NeuronSpec, TrainSpec,
                     parse_fc_widths)
from .hardware import HW_TAU_CHOICES


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
                              norm=norm, tdbn_alpha=config.get("tdbn_alpha", 1.0))
    else:                 # uniform
        encoder = EncoderSpec(depth=depth, channels=config["channels"],
                              kernel_size=config["kernel_size"], stride=config.get("stride", 1),
                              padding=0, dilation=1, bias=(norm == "none"),
                              norm=norm, tdbn_alpha=config.get("tdbn_alpha", 1.0))

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
                             trainable_threshold=bool(config.get("trainable_threshold", False))),
        "train": TrainSpec(epochs=config["epochs"],
                           optimizer=config.get("optimizer", "adam"),
                           lr=config.get("lr", 1e-3),
                           weight_decay=config.get("weight_decay", 0.0),
                           scheduler=config.get("scheduler", "cosine"),
                           warmup_epochs=config.get("warmup_epochs", 0),
                           label_smoothing=config.get("label_smoothing", 0.0),
                           grad_clip=config.get("grad_clip", 0.0),
                           qat_mode=config.get("qat_mode", "inline"),
                           qat_warmup_frac=config.get("qat_warmup_frac", 0.25),
                           qat_epochs=config.get("qat_epochs", 4),
                           qat_lr_scale=config.get("qat_lr_scale", 0.5),
                           fold_bias_mode=config.get("fold_bias_mode", "threshold"),
                           fold_bias_margin=config.get("fold_bias_margin", 0.05)),
    }


# NARROWED to the regions the statistics said matter:
#   channels: 128/256 were the dominant INFEASIBILITY driver (chi-square p=6e-4,
#     Cramer's V=0.59: 32ch 88% feasible, 128ch 14%) and never beat 32/64 on
#     accuracy. Dropped.
#   kernel: every top-10 model used k5 or k7; k3 dropped.
CHANNEL_CHOICES = [32, 64]
KERNEL_CHOICES = [5, 7]


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
                 t_choices: list, per_layer: bool = True):
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.data_dir_abs = str(data_dir_abs)
        self.t_choices = list(t_choices)
        self.per_layer = bool(per_layer)

    def __call__(self, trial):
        batch_size = self.batch_size
        epochs = self.epochs
        data_dir_abs = self.data_dir_abs
        t_choices = self.t_choices
        per_layer = self.per_layer
        # ---- encoder depth ----
        # winners are depth 2-3 (Kruskal-Wallis favoured shallow, q=0.08); depth
        # 5 only ever reached 0.77. Keep 2-4 so depth-3 branches stay in play.
        depth = trial.suggest_int("depth", 2, 4)

        if per_layer:
            # independent geometry per layer
            for i in range(depth):
                trial.suggest_categorical(f"k_{i}", KERNEL_CHOICES)
                trial.suggest_categorical(f"ch_{i}", CHANNEL_CHOICES)
                # each layer independently: downsample by stride-2, by pooling,
                # or not at all (stride 1, no pool -> size-preserving).
                # each layer independently: downsample by stride-2, by pooling,
                # or not at all. The chosen branch sets the flat key
                # config_to_specs reads; the others stay absent and default off.
                ds = trial.suggest_categorical(f"ds_{i}", ["stride", "pool", "none"])
                if ds == "stride":
                    trial.suggest_int(f"stride_{i}", 2, 2)
                elif ds == "pool":
                    trial.suggest_int(f"pool_{i}", 1, 1)
        else:
            trial.suggest_categorical("channels", CHANNEL_CHOICES)
            trial.suggest_categorical("kernel_size", KERNEL_CHOICES)
            mode = trial.suggest_categorical("downsample_mode", ["stride", "pool"])
            if mode == "stride":
                # stride=1 never downsamples -> the flatten explodes and the
                # config is infeasible every time (all the fc_in fan-in busts in
                # the data came from here). Force stride-2.
                trial.suggest_categorical("stride", [2])

        # resize_to=0 (native 128x128) is omitted: 128*128*2 = 32,768 axons is
        # over the 16,383 limit, so it could never pass feasibility.
        trial.suggest_categorical("resize_to", [32, 64])
        trial.suggest_categorical("T", t_choices)

        # ---- head: GAP vs flatten, then variable-depth FC ----
        # GAP collapses HxW before the head (Q4): far fewer params, less
        # overfitting, and deployable. When GAP is chosen the huge first-FC
        # fan-in disappears, so many more configs pass feasibility.
        # final_reduction fixed to flatten (set in the constants below): every
        # top model used it, and GAP historically cost ~40 pts on this task.
        # fc_layers 0-1 only: every top-10 model had 0 hidden FC (Kruskal-Wallis
        # q=0.04), so 2-3 hidden layers are pure waste. Keep 1 as a branch.
        n_fc = trial.suggest_int("fc_layers", 0, 1)
        for i in range(n_fc):
            trial.suggest_categorical(f"fc_width_{i}", FC_WIDTH_CHOICES)

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
        trial.suggest_float("dropout_rate", 0.1, 0.45)
        # norm=none is GONE: it produced every dead (below-chance) network
        # (pooled Mann-Whitney p=0.0015; learned-vs-dead odds ratio 21.8). Only
        # the two foldable, hardware-legal options remain -- and this lets tdBN
        # get a fair test against plain BN on the good backbone.
        norm = trial.suggest_categorical("norm", ["bn", "tdbn"])
        if norm == "tdbn":
            trial.suggest_float("tdbn_alpha", 0.5, 2.0)
        trial.suggest_float("label_smoothing", 0.0, 0.2)
        trial.suggest_categorical("grad_clip", [0.0, 1.0, 5.0])

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

        # final_reduction pinned here (not searched) -- see the head note above.
        return {"N": batch_size, "epochs": epochs, "data_dir": data_dir_abs,
                "pool_type": "avg", "final_reduction": "flatten"}


def make_define_by_run(batch_size: int, epochs: int, data_dir_abs: str, t_choices: list,
                       per_layer: bool = True) -> "DefineByRunSpace":
    """Factory kept for API compatibility -- returns a picklable space object."""
    return DefineByRunSpace(batch_size, epochs, data_dir_abs, t_choices, per_layer)
