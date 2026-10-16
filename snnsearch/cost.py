"""How big a configuration is: neurons, connections and parameters per layer.

Counted from the plan (planning.py), so it agrees with the feasibility check by
construction. The energy proxy, SynOps, needs a trained network and lives in
synops.py. format_summary() is the readable table `main.py summary` prints.
"""

from .config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                     HeadSpec, NeuronSpec, TrainSpec, effective_hw,
                     parse_fc_widths, resolve_conv_layers)
from .planning import plan_network, plan_encoder, conv_out_size, InfeasibleConfig
from .hardware import (check_spec_feasibility, AXON_LIMITS, NEURON_LIMITS, HW_TAU_CHOICES,
                       W_BITS, W_ALPHA, INT16_MAX, W_DELTA)


def count_neurons_and_synapses(cfg: dict) -> dict:
    """cfg is the same dict used everywhere else: input/encoder/output/downsample/head/neuron.

    Counts chip neurons and synapses. A signed layer has two neurons per unit,
    each reading both neurons of every signed source unit, so its neurons
    double and its connections from a signed layer quadruple. Multi-bit input
    takes input_bits axons per value, each with its own synapse.
    """
    plan = plan_network(cfg["input"], cfg["encoder"], cfg["output"], cfg["downsample"], cfg["head"])
    encoder_cfg, input_cfg = cfg["encoder"], cfg["input"]
    sign = 2 if getattr(cfg.get("neuron"), "signed_spikes", False) else 1
    bits = max(1, int(getattr(input_cfg, "input_bits", 1) or 1))

    rows = []
    for b in plan.blocks:
        oh, ow = b.conv_out_hw
        neurons = b.out_channels * oh * ow
        weights_per_neuron = b.kernel_size * b.kernel_size * b.in_channels
        # sources per synapse: input axons for the first layer, else neurons
        src = bits if b.index == 0 else sign
        rows.append({
            "layer": f"conv{b.index}",
            "detail": (f"{b.in_channels}x{b.in_hw[0]}x{b.in_hw[1]} -> "
                       f"{b.out_channels}x{oh}x{ow}  k={b.kernel_size} s={b.stride} p={b.padding}"),
            "neurons": neurons * sign,
            "connections": neurons * weights_per_neuron * src * sign,
            "params": weights_per_neuron * b.out_channels + (b.out_channels if encoder_cfg.bias else 0),
        })
        if encoder_cfg.norm != "none":
            # tdBN has the SAME parameter count as ordinary BN -- one gamma and
            # one beta per channel. (It is BNTT that would scale as C*T.)
            label = "tdbn" if encoder_cfg.norm == "tdbn" else "bn"
            rows.append({
                "layer": f"{label}{b.index}",
                "detail": (f"tdBN({b.out_channels}) over (T,N,H,W)" if encoder_cfg.norm == "tdbn"
                           else f"BatchNorm2d({b.out_channels})"),
                "neurons": 0, "connections": 0, "params": 2 * b.out_channels,
            })
        if b.pool:
            ph, pw = b.out_hw
            or_pool = cfg["downsample"].pool_type == "max"
            rows.append({
                "layer": f"pool{b.index}",
                # a max pool is a layer of OR neurons on chip; an avg pool has none
                "detail": (f"{'max (OR neurons)' if or_pool else 'avg'} k={b.pool_kernel} "
                           f"s={b.pool_stride} -> {b.out_channels}x{ph}x{pw}"),
                # one OR neuron per window and sign, reading its own sign only
                "neurons": b.out_channels * ph * pw * sign if or_pool else 0,
                "connections": b.out_channels * ph * pw * b.pool_kernel * b.pool_kernel * sign,
                "params": 0,
            })

    if plan.reduction == "gap":
        rows.append({
            "layer": "gap",
            "detail": f"{plan.encoder_out_c}x{plan.encoder_out_hw[0]}x{plan.encoder_out_hw[1]} -> {plan.encoder_out_c}",
            "neurons": 0,
            "connections": plan.encoder_out_c * plan.encoder_out_hw[0] * plan.encoder_out_hw[1],
            "params": 0,
        })

    for lin in plan.linears:
        tag = "classifier" if lin.is_classifier else f"fc{lin.index}"
        rows.append({
            "layer": tag,
            "detail": f"{lin.in_features} -> {lin.out_features}",
            "neurons": lin.out_features * sign,
            "connections": lin.in_features * lin.out_features * sign * sign,
            "params": lin.in_features * lin.out_features,
        })

    totals = {
        "neurons": sum(r["neurons"] for r in rows),
        "connections": sum(r["connections"] for r in rows),
        "params": sum(r["params"] for r in rows),
    }
    in_h, in_w = effective_hw(input_cfg)
    return {
        "rows": rows,
        "totals": totals,
        "input_axons": in_h * in_w * input_cfg.C * bits,
        "timesteps": input_cfg.T,
        "neuron_updates_per_sample": totals["neurons"] * input_cfg.T,
        "plan": plan,
    }


def format_summary(cfg: dict) -> str:
    """Human-readable architecture + feasibility + cost table. This is the thing
    to paste into a write-up."""
    counts = count_neurons_and_synapses(cfg)
    feasible, violations = check_spec_feasibility(cfg)
    in_h, in_w = effective_hw(cfg["input"])

    lines = []
    lines.append("=" * 78)
    lines.append("Spiking CNN -- architecture summary")
    lines.append("=" * 78)
    ec = cfg["encoder"]
    nc = cfg["neuron"]
    layers = resolve_conv_layers(ec)
    mode = "per-layer" if ec.layers_json else "uniform"
    inp = cfg["input"]
    lines.append(f"input        : {inp.C}x{in_h}x{in_w}  (native {inp.H}x{inp.W}, "
                 f"resize_to={inp.resize_to}), T={inp.T}, batch={inp.N}, "
                 f"{getattr(inp, 'input_bits', 1)} bit(s) per value")
    lines.append(f"encoder      : {mode}, depth={len(layers)} bias={ec.bias} norm={ec.norm}"
                 + (f" tdbn_alpha={ec.tdbn_alpha}" if ec.norm == "tdbn" else ""))
    if ec.layers_json:
        for i, L in enumerate(layers):
            ov = ("".join(f" {k}={getattr(L, k)}" for k in ("tau", "v_threshold") if getattr(L, k) is not None))
            lines.append(f"  layer {i}    : ch={L.out_channels} k={L.kernel_size} stride={L.stride} pool={L.pool}{ov}")
    lines.append(f"downsample   : mode={cfg['downsample'].mode} pool={cfg['downsample'].pool_type} "
                 f"k={cfg['downsample'].pool_kernel_size} s={cfg['downsample'].pool_stride}")
    lines.append(f"head         : reduction={cfg['head'].final_reduction} fc_widths=[{cfg['head'].fc_widths}] "
                 f"dropout={cfg['head'].dropout_rate}")
    leak = int(round(float(nc.tau)))
    per = "channel" if getattr(nc, "tau_per_channel", False) else "layer"
    tau_tag = f"learned power of 2 per {per}, init {leak}" if nc.trainable_tau else f"{leak}"
    th_tag = "learned in (0,1]" if nc.trainable_threshold else f"{nc.v_threshold}"
    signed = ", signed spikes (2 neurons per unit)" if getattr(nc, "signed_spikes", False) else ""
    lines.append(f"neuron       : {nc.neuron_type} tau={tau_tag} v_th={th_tag} "
                 f"v_reset={nc.v_reset}{signed}")
    th_int = int(min(max(round(float(nc.v_threshold) / W_DELTA), 1), INT16_MAX))
    # the chip's leak is the shift: V -= V // 2**leak, so leak = log2(tau)
    shift = max(0, leak.bit_length() - 1)
    lines.append(f"  -> hardware: LIF_neuron(threshold={th_int}, nu=-17, leak={shift})"
                 f"   [w_alpha={W_ALPHA:g}, w_bits={W_BITS}, weights clamped to +/-1]")
    if "train" in cfg:
        t = cfg["train"]
        lines.append(f"training     : {t.optimizer} lr={t.lr:g} wd={t.weight_decay:g} sched={t.scheduler} "
                     f"epochs={t.epochs} label_smoothing={t.label_smoothing:g} grad_clip={t.grad_clip:g}")
        aug = (f"shift {t.aug_shift:.2f} cutout {t.aug_cutout:.2f} drop {t.aug_drop:.2f}"
               f"{' flip' if t.aug_flip and inp.hflip_ok else ''}") if t.augment else "off"
        lines.append(f"               loss={t.loss} augment={aug} bias={t.fold_bias_mode} "
                     f"seed={t.seed}")
    lines.append("")

    header = f"{'layer':<12} {'shape / detail':<46} {'neurons':>10} {'connections':>14} {'params':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in counts["rows"]:
        lines.append(f"{r['layer']:<12} {r['detail']:<46} {r['neurons']:>10,} {r['connections']:>14,} {r['params']:>12,}")
    lines.append("-" * len(header))
    t = counts["totals"]
    lines.append(f"{'TOTAL':<12} {'':<46} {t['neurons']:>10,} {t['connections']:>14,} {t['params']:>12,}")
    lines.append("")
    lines.append(f"input axons             : {counts['input_axons']:,}  (limit {AXON_LIMITS['total_axons']:,})")
    lines.append(f"spiking neurons         : {t['neurons']:,}")
    lines.append(f"synaptic connections    : {t['connections']:,}")
    lines.append(f"trainable parameters    : {t['params']:,}")
    lines.append(f"neuron updates / sample : {counts['neuron_updates_per_sample']:,}  ({t['neurons']:,} x T={counts['timesteps']})")
    n_spiking = len(counts["plan"].blocks) + len(counts["plan"].linears)
    if cfg["downsample"].pool_type == "max":       # each OR-pool layer adds a step
        n_spiking += sum(1 for b in counts["plan"].blocks if b.pool)
    lines.append(f"pipeline flush steps    : {n_spiking}  (1 per spiking layer; the converter's "
                 f"post-input zero steps)")
    lines.append("")
    if feasible:
        lines.append("HiAER-Spike feasibility : PASS")
    else:
        lines.append("HiAER-Spike feasibility : FAIL")
        for v in violations:
            lines.append(f"   - {v}")
    lines.append("=" * 78)
    return "\n".join(lines)
