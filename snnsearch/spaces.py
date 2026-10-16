"""The search space: what the search samples, in what order.

Optuna samples define-by-run, so a trial only has the knobs it actually used.
The order matters: geometry first, then the feasibility check, then everything
else. A shape that does not fit the chip stops there, so no training knob is
blamed for it.

Nothing here is specific to one dataset. The input channels, size and class
count come from the dataset (data_facts), and the resize range is worked out
from them and the chip's axon limit (resize_range). Every trial records the
dataset facts it was sampled under, so a trial record rebuilds on its own.

config_to_specs turns a sampled flat dict into the specs in config.py. The
space is a module-level class because Ray pickles it.
"""

import math

from .config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                     HeadSpec, NeuronSpec, TrainSpec)
from .hardware import AXON_LIMITS, HW_TAU_CHOICES, check_spec_feasibility


# NARROWED after the 65-trial analysis (see DVS_Search_Statistical_Analysis).
# 512 was the only hidden-FC width that showed up in a strong trial; the head is
# rarely used at all now (fc_layers capped at 1), so keep a small set.
FC_WIDTH_CHOICES = [128, 256, 512]

# Codings that can send more than one bit per input value (encoders.py)
MULTI_BIT_CODINGS = ("passthrough", "levels")
MAX_INPUT_BITS = 4

# How the leak is set: one fixed tau, one learned tau per layer, or one
# learned tau per output channel
TAU_MODES = ["fixed", "layer", "channel"]

# The dataset facts every trial carries
DATA_KEYS = ("data_C", "data_H", "data_W", "num_classes", "hflip_ok")
# Trial records written before the dataset facts were recorded. Every one of
# them came from DVS128 Gesture, the only dataset searched until then.
_LEGACY_DATA = {"data_C": 2, "data_H": 128, "data_W": 128, "num_classes": 11,
                "hflip_ok": False}


def data_facts(bundle) -> dict:
    """The dataset facts a trial records, from a DatasetBundle."""
    return {"data_C": int(bundle.C), "data_H": int(bundle.H), "data_W": int(bundle.W),
            "num_classes": int(bundle.num_classes),
            "hflip_ok": bool(getattr(bundle, "hflip_ok", False))}


def resize_range(data: dict) -> tuple:
    """(low, high, step) for the square input size resize_to.

    The top is the native size or the largest square whose single-bit input
    fits the chip's axon limit, whichever is smaller; the bottom is about a
    quarter of the top, but not below 16. Multi-bit input needs B times the
    axons, and the chip check rejects a size that no longer fits.

    DVS128 Gesture: 24 to 88 in steps of 8 (89 x 89 x 2 = 15,842 axons).
    CIFAR-10: 16 to 32 in steps of 4. MNIST: 16 to 28 in steps of 4.
    """
    side = min(int(data["data_H"]), int(data["data_W"]))
    cap = int(math.isqrt(AXON_LIMITS["total_axons"] // int(data["data_C"])))
    hi = min(side, cap)
    step = 8 if hi > 48 else 4
    if hi < side:
        hi -= hi % step                 # a reduced size stays on the step grid
    floor_lo = max(min(16, hi), hi // 4)
    lo = hi - step * ((hi - floor_lo) // step)
    return lo, hi, step


def _data(config: dict) -> dict:
    if "data_C" in config:
        return {k: config[k] for k in DATA_KEYS if k in config}
    return dict(_LEGACY_DATA)


def config_to_specs(config: dict) -> dict:
    """Flat Ray/Optuna config dict -> the structured cfg every other function takes.

    Encoder is assembled per-layer when per-layer keys (k_0, ch_0, ...) are
    present, otherwise from the uniform depth/channels/kernel_size fields."""
    import json
    n_fc = int(config.get("fc_layers", 1))
    widths = [str(config[f"fc_width_{i}"]) for i in range(n_fc) if f"fc_width_{i}" in config]
    norm = config.get("norm", "bn")
    depth = int(config["depth"])
    data = _data(config)

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

    # tau_mode replaced the trainable_tau switch; older records carry the switch
    tau_mode = config.get("tau_mode") or \
        ("layer" if config.get("trainable_tau", False) else "fixed")

    return {
        "input": InputSpec(N=config["N"], C=int(data["data_C"]), H=int(data["data_H"]),
                           W=int(data["data_W"]), T=config["T"],
                           resize_to=config["resize_to"],
                           input_bits=int(config.get("input_bits", 1) or 1),
                           hflip_ok=bool(data.get("hflip_ok", False))),
        "encoder": encoder,
        "output": OutputSpec(num_classes=int(data["num_classes"])),
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
                             trainable_tau=tau_mode != "fixed",
                             tau_per_channel=tau_mode == "channel",
                             trainable_threshold=bool(config.get("trainable_threshold", False)),
                             integer_leak=bool(config.get("integer_leak", True)),
                             signed_spikes=bool(config.get("signed_spikes", False))),
        "train": TrainSpec(epochs=config["epochs"],
                           optimizer=config.get("optimizer", "adam"),
                           lr=config.get("lr", 1e-3),
                           weight_decay=config.get("weight_decay", 0.0),
                           scheduler=config.get("scheduler", "cosine"),
                           warmup_epochs=config.get("warmup_epochs", 0),
                           label_smoothing=config.get("label_smoothing", 0.0),
                           grad_clip=config.get("grad_clip", 0.0),
                           rate_penalty=config.get("rate_penalty", 0.0),
                           loss=config.get("loss", "rate"),
                           augment=bool(config.get("use_augment", False)),
                           aug_shift=config.get("aug_shift", 0.0),
                           aug_cutout=config.get("aug_cutout", 0.0),
                           aug_drop=config.get("aug_drop", 0.0),
                           aug_flip=bool(config.get("aug_flip", False)),
                           seed=config.get("seed"),
                           qat_mode=config.get("qat_mode", "inline"),
                           qat_warmup_frac=config.get("qat_warmup_frac", 0.25),
                           qat_epochs=config.get("qat_epochs", 4),
                           qat_lr_scale=config.get("qat_lr_scale", 0.5),
                           fold_bias_mode=config.get("bias_mode",
                                                     config.get("fold_bias_mode", "threshold")),
                           fold_bias_margin=config.get("fold_bias_margin", 0.05)),
    }


def geometry_fits(flat) -> bool:
    """Does the shape sampled so far satisfy the chip's connection limits?

    Only shape decides this: kernel, channels, depth, downsampling, resolution,
    bits per input value, signed spikes, bias neurons and the head. Nothing
    about the optimizer or the regularizer can move a fan-in, which is why
    this can run before any of them are sampled.

    Returns False on a config that will not even build, since a feature map
    that collapses below the kernel is infeasible in the same practical sense.
    """
    try:
        spec = config_to_specs({"tau": 2, **flat})
        ok, _ = check_spec_feasibility(spec)
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

    `data` holds the dataset facts (data_facts). `max_input_bits` is 1 when
    the input coding sends one bit per value; otherwise the number of bits per
    input value is searched from 1 to it. `seed` is recorded in every trial so
    a replay starts from the same weights.
    """

    def __init__(self, batch_size: int, epochs: int, data_dir_abs: str,
                 t_choices: list, per_layer: bool = True, require=(), *,
                 data: dict, max_input_bits: int = 1, seed: int = None):
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
        self.data = {k: data[k] for k in DATA_KEYS}
        self.max_input_bits = max(1, int(max_input_bits))
        self.seed = seed

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

        # Bits per input value. One axon carries one bit, so B bits cost B
        # axons per value and B times the first layer's fan-in, which is why
        # it is geometry. Only when the coding can send more than one bit.
        if self.max_input_bits > 1:
            geom["input_bits"] = trial.suggest_int("input_bits", 1, self.max_input_bits)

        # Square input size, from the dataset's size and the axon limit
        # (resize_range). Native resolution is in range only when it fits.
        lo, hi, step = resize_range(self.data)
        geom["resize_to"] = trial.suggest_int("resize_to", lo, hi, step=step)
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
        # Where each layer's bias goes (TrainSpec.fold_bias_mode). "threshold"
        # folds it into the per-channel threshold for free; "neuron" deploys a
        # real bias through always-on bias neurons, which costs fan-in, so it is
        # part of the geometry and checked with it.
        geom["bias_mode"] = trial.suggest_categorical("bias_mode", ["threshold", "neuron"])
        # Signed spikes: +1 or -1 per unit, two chip neurons each, so twice the
        # fan-in and fan-out. Geometry for the same reason.
        geom["signed_spikes"] = trial.suggest_categorical("signed_spikes", [False, True])
        # How pooling runs on chip, only when some layer pools. "avg" folds into
        # the next layer (exact, but 4x its fan-in per 2x2 pool); "max" is one
        # OR neuron per window (exact for spikes, no fan-in cost, one extra
        # timestep). Part of the geometry because it changes fan-in.
        if any(geom.get(f"ds_{i}") == "pool" for i in range(depth)) or \
                geom.get("downsample_mode") == "pool":
            geom["pool_type"] = trial.suggest_categorical("pool_type", ["avg", "max"])

        consts = {"N": batch_size, "epochs": epochs, "data_dir": data_dir_abs,
                  **self.data, "seed": self.seed}

        if not geometry_fits({**consts, **geom}):
            # Stop here. Nothing below this line gets sampled, so no training
            # knob is blamed for a shape that will not fit. tau is supplied
            # because config_to_specs reads it directly and the trial still has
            # to build far enough to report itself infeasible.
            return {**consts, "tau": 2, "infeasible_geometry": True}

        # ---- neuron: INTEGER leak ---------------------------------------------
        # Only powers of two: the chip's leak is a shift (V -= V // 2**leak).
        # Searches 01 and 02 also sampled 3, 6 and 63, which do not exist on
        # chip. With a learned tau this is the starting value; the deployed
        # tau is recorded separately as deployed_tau.
        trial.suggest_categorical("tau", HW_TAU_CHOICES)
        # fixed, learned per layer, or learned per channel. The chip stores a
        # neuron model per neuron, so all three deploy as they are.
        trial.suggest_categorical("tau_mode", TAU_MODES)
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
        # Augmentation of the training inputs (augment.py), on or off, then
        # how much of each kind. Flips only where the dataset allows them.
        if "augment" in require or \
                trial.suggest_categorical("use_augment", [False, True]):
            trial.suggest_float("aug_shift", 0.0, 0.2)
            trial.suggest_float("aug_cutout", 0.0, 0.4)
            trial.suggest_float("aug_drop", 0.0, 0.3)
            if self.data.get("hflip_ok"):
                trial.suggest_categorical("aug_flip", [False, True])

        # ---- loss ----
        # "rate" scores the clip's spike count, which is the chip's answer;
        # "tet" scores every timestep (Deng et al., ICLR 2022).
        trial.suggest_categorical("loss", ["rate", "tet"])

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


# The on/off and either/or choices the balanced block covers. Each maps a
# level to the sampled parameters that produce it. pooling and input_bits are
# set while drawing the shape (_random_geometry); the rest are copied as is.
BALANCED_FACTORS = {
    # geometry: drawn with the shape and checked against the chip with it
    "pooling": {
        # no pooling anywhere: the first layer shrinks by stride, the rest
        # by stride or not at all
        "none": {},
        "avg": {"pool_type": "avg"},
        "max": {"pool_type": "max"},
    },
    "bias_mode": {"threshold": {"bias_mode": "threshold"}, "neuron": {"bias_mode": "neuron"}},
    "final_reduction": {"flatten": {"final_reduction": "flatten"},
                        "gap": {"final_reduction": "gap"}},
    "signed_spikes": {"off": {"signed_spikes": False}, "on": {"signed_spikes": True}},
    # "multi" draws 2..max_input_bits; left out when the coding sends one bit
    "input_bits": {"1": {"input_bits": 1}, "multi": {}},
    # training
    "tau_mode": {m: {"tau_mode": m} for m in TAU_MODES},
    "loss": {"rate": {"loss": "rate"}, "tet": {"loss": "tet"}},
    "augment": {"off": {"use_augment": False}, "on": {"use_augment": True}},
    "conv_dropout": {"off": {"use_conv_dropout": False}, "on": {"use_conv_dropout": True}},
    "rate_penalty": {"off": {"use_rate_penalty": False}, "on": {"use_rate_penalty": True}},
}


def balanced_design(factors: dict, n: int, rng, tries: int = 300) -> list:
    """n rows of factor levels, each level of each factor equally often.

    n is rounded up to a multiple of every factor's level count. With ten
    factors a full factorial would be thousands of trials, so instead each
    factor's levels are spread evenly over the rows and shuffled, and of
    `tries` shuffles the one whose pairs of factors are closest to balanced
    is kept: every pair of levels from two factors then appears about
    n / (levels_a * levels_b) times, so no option is mostly tried alongside
    one setting of another.
    """
    import itertools
    names = list(factors)
    counts = [len(factors[k]) for k in names]
    lcm = 1
    for c in counts:
        lcm = lcm * c // math.gcd(lcm, c)
    n = int(math.ceil(n / lcm) * lcm)
    best, best_cost = None, None
    for _ in range(max(1, tries)):
        cols = []
        for name in names:
            col = list(factors[name]) * (n // len(factors[name]))
            rng.shuffle(col)
            cols.append(col)
        cost = 0.0
        for a, b in itertools.combinations(range(len(names)), 2):
            expected = n / (counts[a] * counts[b])
            seen = {}
            for la, lb in zip(cols[a], cols[b]):
                seen[(la, lb)] = seen.get((la, lb), 0) + 1
            cost += sum((seen.get((la, lb), 0) - expected) ** 2
                        for la in factors[names[a]] for lb in factors[names[b]])
        if best_cost is None or cost < best_cost:
            best, best_cost = cols, cost
    return [{name: best[j][i] for j, name in enumerate(names)} for i in range(n)]


def _random_geometry(rng, levels, t_choices, data, max_input_bits):
    """One random network shape with the given balanced-block levels, using
    the same ranges DefineByRunSpace samples from."""
    depth = rng.randint(2, 4)
    g = {"depth": depth}
    for i in range(depth):
        g[f"k_{i}"] = rng.choice([3, 5, 7, 9])
        g[f"ch_{i}"] = int(round(math.exp(rng.uniform(math.log(8), math.log(128)))))
        if levels["pooling"] == "none":
            ds = rng.choice(["stride", "none"]) if i else "stride"
        else:
            ds = "pool" if i == 0 else rng.choice(["stride", "pool", "none"])
        g[f"ds_{i}"] = ds
        if ds == "stride":
            g[f"stride_{i}"] = 2
        elif ds == "pool":
            g[f"pool_{i}"] = 1
    if max_input_bits > 1:
        g["input_bits"] = 1 if levels.get("input_bits", "1") == "1" else \
            rng.randint(2, max_input_bits)
    lo, hi, step = resize_range(data)
    g["resize_to"] = rng.randrange(lo, hi + 1, step)
    g["T"] = rng.choice(list(t_choices))
    g["fc_layers"] = rng.randint(0, 1)
    if g["fc_layers"]:
        g["fc_width_0"] = rng.choice(FC_WIDTH_CHOICES)
    for name in ("bias_mode", "final_reduction", "signed_spikes", "pooling"):
        g.update(BALANCED_FACTORS[name][levels[name]])
    return g


def balanced_points(n: int, seed: int = 0, t_choices=(16,), batch_size: int = 16,
                    epochs: int = 40, *, data: dict, max_input_bits: int = 1,
                    run_seed: int = None) -> list:
    """The balanced block: n trials (rounded up to a multiple of 6) in which
    every level of every BALANCED_FACTORS choice appears equally often.

    Optuna samples whatever has scored well so far, so an option that starts
    badly can end up barely tried: in Search 02, GAP got 32 of 400 trials and
    the firing-rate penalty 26, too few to judge either. Running the block
    first gives each level of each choice the same number of trials, so the
    levels can be compared on equal footing. The search seeds these as its
    first trials; Optuna steers only after them.

    Each point carries a whole network shape, drawn at random from the search
    ranges and redrawn until it fits the chip. Leaving the shape to the
    sampler lost about half the block to the chip check, and with it the
    balance. Amounts (learning rate, dropout rate, augmentation strength and
    so on) are left to the sampler, which samples them at random during the
    block.
    """
    import random
    rng = random.Random(seed)
    factors = dict(BALANCED_FACTORS)
    if max_input_bits <= 1:
        factors.pop("input_bits")
    if int(n) <= 0:
        return []
    consts = {"N": batch_size, "epochs": epochs, "data_dir": ".", **data, "seed": run_seed}
    points = []
    for levels in balanced_design(factors, int(n), rng):
        for _attempt in range(10_000):
            geom = _random_geometry(rng, levels, t_choices, data, max_input_bits)
            if geometry_fits({**consts, **geom}):
                break
        else:
            raise RuntimeError(f"no shape fits the chip for {levels}")
        point = dict(geom)
        for name in ("tau_mode", "loss", "augment", "conv_dropout", "rate_penalty"):
            point.update(BALANCED_FACTORS[name][levels[name]])
        points.append(point)
    return points


def make_define_by_run(batch_size: int, epochs: int, data_dir_abs: str, t_choices: list,
                       per_layer: bool = True, require=(), *, data: dict,
                       max_input_bits: int = 1, seed: int = None) -> "DefineByRunSpace":
    """Factory kept for API compatibility -- returns a picklable space object."""
    return DefineByRunSpace(batch_size, epochs, data_dir_abs, t_choices, per_layer,
                            require, data=data, max_input_bits=max_input_bits, seed=seed)
