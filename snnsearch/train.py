"""Optimizer construction, the training loop, and the pipeline flush.

Moved verbatim from Practice2.py lines 1990-2241 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

import math
from copy import deepcopy
from dataclasses import asdict

from .config import TrainSpec
from .neuron import HardwareLIFNode
from .model import build_model
from .quantize import deploy_and_measure, deployment_report, hardware_export
from ._torch import _HAS_TORCH, _require_torch, torch, nn, F, functional


def build_optimizer(net, train_cfg: TrainSpec):
    name = train_cfg.optimizer.lower()
    if name == "adam":
        # NB: weight_decay in Adam is classic L2 added to the gradient, which
        # interacts with the adaptive step. Use adamw for decoupled decay.
        return torch.optim.Adam(net.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(net.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(net.parameters(), lr=train_cfg.lr, momentum=train_cfg.momentum,
                               weight_decay=train_cfg.weight_decay, nesterov=True)
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
    return sum(1 for m in net.conv_fc if isinstance(m, HardwareLIFNode))


def forward_over_time(net, x, flush_steps: int = None):
    """
    x: (N, T, C, H, W) -> output spike rate, neuron state reset afterwards.

    Mirrors the conversion script's measurement exactly: accumulate output over
    T input steps PLUS `flush_steps` zero-input steps, then divide by T (not by
    T + flush) so the rate stays comparable across depths.

    Single-step nets are driven by an explicit loop; multi-step nets (tdbn) get
    the whole (T, N, ...) tensor at once because tdBN's statistics span T.
    """
    x = x.transpose(0, 1)  # (T, N, C, H, W)
    T = x.shape[0]
    if flush_steps is None:
        flush_steps = hardware_flush_steps(net)

    if getattr(net, "step_mode", "s") == "m":
        if flush_steps > 0:
            pad = torch.zeros_like(x[:1]).expand(flush_steps, *x.shape[1:])
            x = torch.cat([x, pad], dim=0)
        out = net(x).sum(0) / T
    else:
        out_sum = 0.0
        for t in range(T):
            out_sum = out_sum + net(x[t])
        if flush_steps > 0:
            zeros = torch.zeros_like(x[0])
            for _ in range(flush_steps):
                out_sum = out_sum + net(zeros)
        out = out_sum / T
    functional.reset_net(net)
    return out


def train_one_epoch(net, loader, optimizer, device, criterion=None,
                    grad_clip: float = 0.0, batch_scheduler=None):
    net.train()
    criterion = criterion or (lambda o, y: F.cross_entropy(o, y))
    total, correct, loss_sum = 0, 0, 0.0
    for x, y, _lengths in loader:  # pad_sequence_collate returns (data, labels, lengths);
                                    # lengths is always == T here since split_by="number"
                                    # gives every sample a fixed frame count -- safe to ignore.
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = forward_over_time(net, x)
        loss = criterion(out, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


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

    net = build_model(cfg).to(device)
    if loaders is None:
        raise ValueError(
            "run_training needs `loaders`. Build them with\n"
            "    snnsearch.pipeline.prepare(cfg)\n"
        "which resolves the dataset through the registry rather than "
        "assuming one directory layout.")
    train_loader, val_loader, _ = loaders
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

    mode = getattr(train_cfg, "qat_mode", "inline")

    # ---- how the epoch budget is split -----------------------------------
    if mode == "inline":
        warmup = max(1, round(train_cfg.epochs * train_cfg.qat_warmup_frac))
        warmup = min(warmup, train_cfg.epochs - 1) if train_cfg.epochs > 1 else 1
        grid_epochs = train_cfg.epochs - warmup
    else:
        warmup, grid_epochs = train_cfg.epochs, 0   # tail/ptq handled below

    def run_phase(net, optimizer, scheduler, step_per_batch, n_epochs, epoch0,
                  best, best_state, tag):
        """One training phase; reports each epoch into the shared trajectory so
        ASHA still sees a continuous per-epoch curve across warmup + grid."""
        for e in range(n_epochs):
            train_loss, train_acc = train_one_epoch(
                net, train_loader, optimizer, device, criterion,
                grad_clip=train_cfg.grad_clip,
                batch_scheduler=scheduler if step_per_batch else None)
            if scheduler is not None and not step_per_batch:
                scheduler.step()
            val_acc = evaluate(net, val_loader, device)
            if val_acc > best:
                best = val_acc
                best_state = deepcopy(net.state_dict())
            if report_fn is not None:
                report_fn(epoch=epoch0 + e, train_loss=train_loss, train_acc=train_acc,
                          val_acc=val_acc, best_val_acc=best,
                          lr=optimizer.param_groups[0]["lr"], phase=tag)
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
        hw_acc = evaluate(hw_net, val_loader, device)
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
            "flush_steps": hardware_flush_steps(hw_net),
        }
    else:
        # ---- tail / ptq: old behaviour -----------------------------------
        if best_state is not None:
            net.load_state_dict(best_state)
        hw_net, hw = deploy_and_measure(net, cfg, train_loader, val_loader, device)

    hw["float_val_accuracy"] = float_best

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

    if ckpt_path is not None:
        torch.save({
            "state_dict": net.state_dict(),
            "hw_state_dict": hw_net.state_dict(),
            "hardware_export": hardware_export(hw_net),
            "encoder_layers_json": cfg["encoder"].layers_json,
            "config": {n: asdict(s) for n, s in cfg.items()},
            "metrics": hw,
            "folded": True, "bias_mode": train_cfg.fold_bias_mode,
            "qat_mode": mode,
        }, ckpt_path)
    return {"float_val_accuracy": float_best, "hw_net": hw_net, **hw}
