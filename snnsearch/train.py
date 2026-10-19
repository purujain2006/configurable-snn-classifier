"""The training loop, and the forward pass every measurement uses.

    forward_over_time    T input steps plus one flush step per spiking layer,
                         as the chip runs a clip
    build_optimizer      weight decay on conv and linear weights only
    train_one_epoch      the rate loss or TET, plus augmentation (augment.py)
    run_training         float warmup, then training on the INT16 grid, then
                         the deployed network, its metrics, and the per-trial
                         simulator check
"""

import math
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict

from .config import TrainSpec
from .neuron import HardwareLIFNode
from .model import build_model
from .quantize import deploy_and_measure, deployment_report, hardware_export
from ._torch import _HAS_TORCH, _require_torch, torch, nn, F, functional


def param_groups(net, weight_decay):
    """Weight decay on conv and linear weights only.

    Every other parameter is a scalar or a per-channel vector: the neuron's
    raw_log2_tau / raw_tau and raw_threshold, and BatchNorm's gamma and beta.
    Decaying them pulls each toward zero for no regularization benefit, and
    it moved real deployment parameters. Measured on the grokking runs, all
    starting at tau 8 for 300 epochs: weight decay 9.6e-5 ended at tau ~5.5,
    1e-3 at ~3 (rank1), 1e-2 at 0, the clamp floor. So weight decay was also
    setting tau, and the two could not be told apart in any result. Decaying
    beta similarly shifts the folded threshold (theta' = theta - b').
    """
    decay, no_decay = [], []
    for p in net.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim > 1 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(net, train_cfg: TrainSpec):
    name = train_cfg.optimizer.lower()
    groups = param_groups(net, train_cfg.weight_decay)
    if name == "adam":
        # NB: weight_decay in Adam is classic L2 added to the gradient, which
        # interacts with the adaptive step. Use adamw for decoupled decay.
        return torch.optim.Adam(groups, lr=train_cfg.lr)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=train_cfg.lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=train_cfg.lr, momentum=train_cfg.momentum,
                               nesterov=True)
    raise ValueError(f"Unknown optimizer {train_cfg.optimizer!r}")


def build_scheduler(optimizer, train_cfg: TrainSpec, steps_per_epoch: int):
    """Returns (scheduler, step_per_batch). `None` scheduler = constant LR."""
    name = train_cfg.scheduler.lower()
    epochs = train_cfg.epochs

    if name == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=train_cfg.lr, epochs=epochs,
            steps_per_epoch=max(1, steps_per_epoch),
        )
        return sched, True

    if name == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - train_cfg.warmup_epochs))
    elif name == "step":
        main = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=train_cfg.step_gamma)
    elif name == "none":
        main = None
    else:
        raise ValueError(f"Unknown scheduler {train_cfg.scheduler!r}")

    if train_cfg.warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=train_cfg.warmup_epochs)
        if main is None:
            return warm, False
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, main], milestones=[train_cfg.warmup_epochs]), False

    return main, False


# =============================================================================
# 8. Train / evaluate -- loop over T manually; filters never see T, only the
#    neuron's internal state does.
# =============================================================================

def hardware_flush_steps(net) -> int:
    """
    Extra zero-input timesteps needed to drain the pipeline, = the number of
    spiking layers.

    HardwareLIFNode fires on the membrane carried in from the previous step
    (the converter's order), so each spiking layer delays its output by one
    timestep. The conversion script does exactly this: T input frames, then
    `num_layers` steps with an empty input list, accumulating output spikes
    throughout and dividing by T. Without the flush, the last frames' evidence
    never reaches the classifier and the deployed accuracy is lower than
    anything measured in training -- for no reason visible in the config.
    """
    # an OR-pool layer is a layer of chip neurons too, so it adds a step
    from .neuron import OrPool
    return sum(1 for m in net.conv_fc if isinstance(m, HardwareLIFNode)
               or (OrPool is not None and isinstance(m, OrPool)))


@contextmanager
def exact_arithmetic():
    """TF32 off inside, so GPU convolutions are true float32.

    On Ampere and newer NVIDIA GPUs PyTorch runs cuDNN convolutions in TF32
    by default, which keeps about 3 significant digits of each weight. A
    deployed weight is an integer times W_DELTA, so TF32 moves a membrane by
    several weight steps and flips spikes the chip would not: the first
    trained network checked on a cluster GPU disagreed with IntegerSim on 1
    of 216 validation clips, and float32 on a CPU agreed on every spike.
    Training tolerates the noise. Measuring the deployed network does not, so
    every measurement of it runs inside this.
    """
    if not _HAS_TORCH:
        yield
        return
    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def forward_over_time(net, x, flush_steps: int = None, per_step: bool = False):
    """
    x: (N, T, C, H, W) -> output spike rate, neuron state reset afterwards.
    With per_step=True, returns (rate, steps) where steps is (T, N, classes):
    the output at each step that carries an input step's evidence, i.e. steps
    flush .. flush+T-1, since every spiking layer delays by one step.

    Mirrors the conversion script's measurement exactly: accumulate output over
    T input steps PLUS `flush_steps` zero-input steps, then divide by T (not by
    T + flush) so the rate stays comparable across depths.

    Single-step nets are driven by an explicit loop; multi-step nets (tdbn) get
    the whole (T, N, ...) tensor at once because tdBN's statistics span T.
    """
    dtype = next(net.parameters()).dtype         # float64 for hs_export.exact_copy
    x = x.to(dtype).transpose(0, 1)  # (T, N, C, H, W)
    T = x.shape[0]
    if flush_steps is None:
        flush_steps = hardware_flush_steps(net)

    if getattr(net, "step_mode", "s") == "m":
        if flush_steps > 0:
            pad = torch.zeros_like(x[:1]).expand(flush_steps, *x.shape[1:])
            x = torch.cat([x, pad], dim=0)
        seq = net(x)
    else:
        outs = [net(x[t]) for t in range(T)]
        if flush_steps > 0:
            zeros = torch.zeros_like(x[0])
            outs += [net(zeros) for _ in range(flush_steps)]
        seq = torch.stack(outs)
    out = seq.sum(0) / T
    functional.reset_net(net)
    if per_step:
        return out, seq[flush_steps:flush_steps + T]
    return out


class _RateProbe:
    """Collects the mean firing rate of every spiking layer, differentiably.

    The penalty has to see the rate as a function of the weights, so this
    cannot run under no_grad and cannot store floats. The spike tensor comes
    out of the surrogate function, so its mean carries a gradient back to the
    membrane and from there to everything upstream.

    In single-step mode the hook fires once per timestep per layer, so the
    collected means already span time; in multi-step mode the spike tensor
    carries the time axis itself. Either way the mean over everything collected
    is the mean spatiotemporal firing rate, which is what a global L1 penalty
    is defined on.
    """

    def __init__(self, net):
        self.rates = []
        self._handles = []
        nodes = [m for m in net.modules() if isinstance(m, HardwareLIFNode)]
        # EVERY SPIKING LAYER EXCEPT THE LAST.
        #
        # The final layer's spikes are the prediction: forward_over_time divides
        # their count by T and that rate is what cross-entropy scores. Including
        # them in a firing-rate penalty asks the network to stop answering, and
        # the two objectives then pull against each other on the same tensor.
        #
        # Measured: trials with the penalty on reached 0.332 at epoch 9 against
        # 0.626 without it, and 92% of them were killed at the first rung
        # against 43%. Not a slow start, a network being told to go quiet.
        self._nodes = nodes[:-1]
        for m in self._nodes:
            if getattr(m, "signed", False):
                m._activity = []
            self._handles.append(m.register_forward_hook(self._hook))

    def _hook(self, mod, _inp, out):
        steps = getattr(mod, "_activity", None)
        if steps:
            # A signed unit outputs spike - spike_neg, so its mean let every
            # -1 spike LOWER the penalty: Search 05's first trials with signed
            # spikes and the penalty on were rewarded for negative spikes. Its
            # rate is the two trains' sum instead, which the neuron records at
            # every step (once per call in single-step mode, T in multi-step).
            self.rates.append(torch.stack(steps).mean())
            steps.clear()
        else:
            self.rates.append(out.mean())

    def mean(self):
        """Scalar mean rate for the batch just run, or None if nothing fired."""
        if not self.rates:
            return None
        return torch.stack(self.rates).mean()

    def clear(self):
        self.rates.clear()
        for m in self._nodes:
            if getattr(m, "_activity", None):
                m._activity.clear()

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        for m in self._nodes:
            if getattr(m, "_activity", None) is not None:
                m._activity = None


def train_one_epoch(net, loader, optimizer, device, criterion=None,
                    grad_clip: float = 0.0, batch_scheduler=None,
                    rate_penalty: float = 0.0, loss_kind: str = "rate",
                    augment=None):
    """Returns (mean loss, accuracy, mean firing rate).

    `rate_penalty` adds lambda * mean spatiotemporal firing rate to the loss.
    Fewer spikes means fewer accumulates on chip, so this is the one
    regularizer here that also buys energy. It is a global L1 on the rate: no
    per-layer targets, no schedule, because nothing yet says what the right
    per-layer rate would be.

    `loss_kind` "rate" scores the output rate over the whole clip, which is
    what the chip's answer is. "tet" averages a cross-entropy at every step
    instead (Deng et al., ICLR 2022), so every step is trained to classify
    rather than only their sum. Accuracy is the rate's either way.

    `augment`, if given, maps a training batch to an augmented one
    (augment.augment_batch).
    """
    net.train()
    criterion = criterion or (lambda o, y: F.cross_entropy(o, y))
    total, correct, loss_sum, rate_sum, rate_n = 0, 0, 0.0, 0.0, 0
    probe = _RateProbe(net) if rate_penalty and rate_penalty > 0 else None
    for x, y, _lengths in loader:  # pad_sequence_collate returns (data, labels, lengths);
                                    # lengths is always == T here since split_by="number"
                                    # gives every sample a fixed frame count -- safe to ignore.
        x, y = x.to(device), y.to(device)
        if augment is not None:
            x = augment(x)
        optimizer.zero_grad()
        if probe is not None:
            probe.clear()
        if loss_kind == "tet":
            out, steps = forward_over_time(net, x, per_step=True)
            loss = torch.stack([criterion(o, y) for o in steps]).mean()
        else:
            out = forward_over_time(net, x)
            loss = criterion(out, y)
        if probe is not None:
            rate = probe.mean()
            if rate is not None:
                loss = loss + rate_penalty * rate
                rate_sum += float(rate.detach().item())
                rate_n += 1
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    if probe is not None:
        probe.detach()
    return (loss_sum / max(1, total), correct / max(1, total),
            rate_sum / rate_n if rate_n else None)


def evaluate(net, loader, device):
    net.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y, _lengths in loader:
            x, y = x.to(device), y.to(device)
            out = forward_over_time(net, x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def run_training(cfg: dict, data_dir: str = None, device=None, report_fn=None,
                 ckpt_path: str = None, loaders=None) -> float:
    """Shared training loop used by both `single` mode and each Ray trial.
    If ckpt_path is given, the best-val model weights are saved there for later
    folding/deployment."""
    _require_torch()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg: TrainSpec = cfg["train"]
    if train_cfg.loss not in ("rate", "tet"):
        raise ValueError(f"train.loss must be 'rate' or 'tet', got {train_cfg.loss!r}")

    # Before the model is built, so the seed fixes the initial weights, the
    # order of the training batches and the augmentation draws.
    if train_cfg.seed is not None:
        import random
        import numpy as np
        random.seed(train_cfg.seed)
        np.random.seed(train_cfg.seed % 2**32)
        torch.manual_seed(train_cfg.seed)

    net = build_model(cfg).to(device)
    if loaders is None:
        raise ValueError(
            "run_training needs `loaders`. Build them with\n"
            "    snnsearch.pipeline.prepare(cfg)\n"
        "which resolves the dataset through the registry rather than "
        "assuming one directory layout.")
    train_loader, val_loader, _ = loaders
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    augment = None
    if train_cfg.augment:
        from .augment import augment_batch
        hflip_ok = bool(getattr(cfg.get("input"), "hflip_ok", False))
        augment = lambda x: augment_batch(x, train_cfg, hflip_ok)     # noqa: E731

    mode = getattr(train_cfg, "qat_mode", "inline")

    # ---- how the epoch budget is split -----------------------------------
    if mode == "inline":
        warmup = max(1, round(train_cfg.epochs * train_cfg.qat_warmup_frac))
        warmup = min(warmup, train_cfg.epochs - 1) if train_cfg.epochs > 1 else 1
        grid_epochs = train_cfg.epochs - warmup
    else:
        warmup, grid_epochs = train_cfg.epochs, 0   # tail/ptq handled below

    # Written by run_phase rather than returned, so the two call sites below
    # keep their two-value unpacking and cannot drift out of step with it.
    #
    # `window` holds the most recent validation accuracies. It is cleared at the
    # start of every phase: inline QAT folds BatchNorm between the warmup and the
    # grid phase, so accuracy either side of that boundary comes from two
    # different models and averaging across it would describe neither.
    last = {"firing_rate": None, "window": [], "smoothed": None}
    smooth_k = max(1, int(getattr(train_cfg, "val_smoothing_epochs", 5) or 1))

    def run_phase(net, optimizer, scheduler, step_per_batch, n_epochs, epoch0,
                  best, best_state, tag):
        """One training phase; reports each epoch into the shared trajectory so
        ASHA still sees a continuous per-epoch curve across warmup + grid.

        `best` now tracks the best SMOOTHED validation accuracy, and the saved
        weights are the ones ending the best recent stretch rather than the ones
        that happened to land on the luckiest single epoch.
        """
        last["window"] = []
        for e in range(n_epochs):
            train_loss, train_acc, rate = train_one_epoch(
                net, train_loader, optimizer, device, criterion,
                grad_clip=train_cfg.grad_clip,
                batch_scheduler=scheduler if step_per_batch else None,
                rate_penalty=getattr(train_cfg, "rate_penalty", 0.0) or 0.0,
                loss_kind=train_cfg.loss, augment=augment)
            if rate is not None:
                last["firing_rate"] = rate
            if scheduler is not None and not step_per_batch:
                scheduler.step()
            val_acc = evaluate(net, val_loader, device)

            # Mean of the last `smooth_k` epochs, or of however many exist so
            # far. Early epochs therefore average fewer points and are noisier,
            # which is correct: there is no more evidence to average.
            last["window"].append(val_acc)
            del last["window"][:-smooth_k]
            smoothed = sum(last["window"]) / len(last["window"])
            last["smoothed"] = smoothed

            if smoothed > best:
                best = smoothed
                best_state = deepcopy(net.state_dict())
            if report_fn is not None:
                report_fn(epoch=epoch0 + e, train_loss=train_loss, train_acc=train_acc,
                          val_acc=val_acc, best_val_acc=best,
                          smoothed_val_acc=smoothed, smoothing_window=len(last["window"]),
                          lr=optimizer.param_groups[0]["lr"], phase=tag,
                          firing_rate=rate)
        return best, best_state

    # ---- phase 1: float warmup (BN active, precise weights) --------------
    optimizer = build_optimizer(net, train_cfg)
    scheduler, step_per_batch = build_scheduler(optimizer, train_cfg, len(train_loader))
    best, best_state = 0.0, None
    best, best_state = run_phase(net, optimizer, scheduler, step_per_batch,
                                 warmup, 0, best, best_state, tag="float")
    float_best = best

    if mode == "inline":
        # ---- phase 2: fold inline, train the rest ON the grid ------------
        if best_state is not None:
            net.load_state_dict(best_state)          # grid-train from best warmup
        net.eval()                                   # fold from running stats
        net.to_qat_folded(bias_mode=train_cfg.fold_bias_mode,
                          fold_bias_margin=getattr(train_cfg, "fold_bias_margin", 0.05),
                          fold_bias_qat_form=getattr(train_cfg, "fold_bias_qat_form", "threshold"))
        net.to(device)

        grid_cfg = deepcopy(train_cfg)
        grid_cfg.lr = train_cfg.lr * train_cfg.qat_lr_scale
        grid_cfg.warmup_epochs = 0
        grid_cfg.scheduler = "cosine"
        grid_cfg.epochs = max(1, grid_epochs)
        _qse = getattr(train_cfg, "qat_schedule_epochs", None)
        if _qse:
            grid_cfg.epochs = max(1, min(int(_qse), grid_epochs))
        grid_cfg.scheduler = getattr(train_cfg, "qat_scheduler", None) \
            or grid_cfg.scheduler
        gopt = build_optimizer(net, grid_cfg)
        gsched, gstep = build_scheduler(gopt, grid_cfg, len(train_loader))

        grid_best, grid_state = 0.0, None
        if grid_epochs > 0:
            grid_best, grid_state = run_phase(net, gopt, gsched, gstep,
                                              grid_epochs, warmup, 0.0, None, tag="grid")
            if grid_state is not None:
                net.load_state_dict(grid_state)

        # ---- freeze into the deployable BN-free model & measure IT -------
        net.eval()
        hw_net = deepcopy(net).export_deployed().to(device)
        from .hs_export import exact_copy
        hw_exact = exact_copy(hw_net)
        with exact_arithmetic():
            hw_acc = evaluate(hw_exact, val_loader, device)
        rep = deployment_report(hw_net)
        hw = {
            "hw_val_accuracy": hw_acc,
            "grid_val_accuracy": grid_best,
            "weight_clip_frac": rep["weight_clip_frac"],
            "max_abs_weight": rep["max_abs_weight"],
            "min_threshold": rep["min_threshold"],
            "deployable": rep["deployable"],
            "deploy_reasons": "; ".join(rep["blocking_reasons"])[:300],
            "deploy_warnings": "; ".join(rep["warnings"])[:300],
            # bits the widest threshold needs as the chip receives it
            "threshold_bits": rep.get("threshold_bits"),
            "chip_threshold_max": rep.get("chip_threshold_max"),
            "flush_steps": hardware_flush_steps(hw_net),
        }
    else:
        # ---- tail / ptq: old behaviour -----------------------------------
        if best_state is not None:
            net.load_state_dict(best_state)
        with exact_arithmetic():
            hw_net, hw = deploy_and_measure(net, cfg, train_loader, val_loader, device)

    hw["float_val_accuracy"] = float_best
    # what each spiking layer deploys with, per layer, e.g. "8,8,4"; a layer
    # with a tau per channel shows its range, e.g. "2-16"
    def _tau_text(leak):
        if isinstance(leak, (list, tuple)):
            lo, hi = min(leak), max(leak)
            return str(lo) if lo == hi else f"{lo}-{hi}"
        return str(leak)
    hw["deployed_tau"] = ",".join(_tau_text(m.hw_leak()) for m in hw_net.conv_fc
                                  if isinstance(m, HardwareLIFNode))

    # Run the validation clips through simpleSim's arithmetic (IntegerSim), so
    # each trial reports what the simulator scores and whether it agrees with
    # PyTorch clip by clip. A network the simulator cannot run is not deployable.
    from .hs_export import verify_deployment, exact_copy
    if mode != "inline":
        hw_exact = exact_copy(hw_net)
    hw.update(verify_deployment(hw_exact, val_loader, device, hardware_flush_steps(hw_net)))
    if hw["sim_reason"]:
        hw["deployable"] = False
        hw["deploy_reasons"] = "; ".join(
            r for r in (hw.get("deploy_reasons"), "simulator: " + hw["sim_reason"]) if r)[:300]

    # What conversion cost: same weights, same point in training,
    # measured before and after export. Positive means export lost
    # accuracy, which is the only direction this can honestly go.
    # inline mode names it grid_val_accuracy, tail/ptq names it
    # qat_val_accuracy; both are the trained net just before export.
    pre_export = hw.get("grid_val_accuracy") or hw.get("qat_val_accuracy")
    hw["quant_gap"] = ((pre_export - hw["hw_val_accuracy"])
                       if pre_export else None)
    hw["pre_export_val_accuracy"] = pre_export

    # What the whole schedule bought, warmup best to deployed. Useful,
    # but it is a training-progress figure and not a conversion cost,
    # so it gets a name that says so.
    hw["end_to_end_gain"] = hw["hw_val_accuracy"] - float_best

    # Only measured when the penalty is on, because the probe holds a graph
    # reference per spiking layer per timestep and this box has run out of GPU
    # memory before. SynOps already reports the energy proxy for every trial.
    hw["firing_rate"] = last["firing_rate"]
    hw["rate_penalty"] = getattr(train_cfg, "rate_penalty", 0.0) or 0.0

    if ckpt_path is not None:
        torch.save({
            "state_dict": net.state_dict(),
            "hw_state_dict": hw_net.state_dict(),
            "hardware_export": hardware_export(hw_net),
            "encoder_layers_json": cfg["encoder"].layers_json,
            "config": {n: asdict(s) for n, s in cfg.items()},
            "metrics": hw,
            "folded": True, "bias_mode": train_cfg.fold_bias_mode,
            # leak floored on each layer's own grid, as the chip does
            "leak_grid": "chip",
            "qat_mode": mode,
        }, ckpt_path)
    return {"float_val_accuracy": float_best, "hw_net": hw_net, **hw}
