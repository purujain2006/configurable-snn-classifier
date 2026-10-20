"""Per-clip match between the trained model and HiAER-Spike's simpleSim.

Leif's requirement is zero per-clip disagreements, not two accuracies that
round alike. Two models can both score 0.8333 on 288 clips and disagree on
forty of them.

TWO STEPS, RUN IN THIS ORDER

1. Prove the fast integer simulator IS simpleSim.

       python tools/match_simplesim.py --selftest

   Builds a small random network with rank1's layer types (conv, LIF, 2x2 avg
   pool, conv, LIF, 2x2 avg pool, linear, LIF), unrolls it into hs_api's
   axon and neuron dictionaries, runs the real hs_api simple_sim, and requires
   every membrane potential and every spike of every neuron at every timestep
   to equal IntegerSim's. Needs hs_api importable (poetry run, or --hs-path).

   Why not run rank1 itself through hs_api: it unrolls to ~260 million
   synapses, which hs_api holds as Python tuples and then copies twice.

2. Compare the checkpoint against IntegerSim, clip by clip.

       python tools/match_simplesim.py --ckpt rank1.pth --data-root DVS128 --tau 4

   --tau is required when the checkpoint trained a tau that is not a power of
   two. simpleSim's leak is `v // 2**leak`, a shift, so tau = 3 does not
   exist on the chip.

The PyTorch side is the same model with the chip's tau and the chip's leak
grid applied (hs_export.make_chip_faithful). Without --tau the trained model is
scored as-is too, so the cost of the tau change is visible.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Python puts this script's directory first on sys.path, and tools/ holds
# source clones of fxpmath, hs_api and connectome_utils. The fxpmath clone has
# an __init__.py at its top level, so it would shadow the installed fxpmath
# and fail on a circular import.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# 1. selftest
# ---------------------------------------------------------------------------

VARIANTS = {
    "rank1": "conv, LIF, 2x2 avg pool, conv, LIF, 2x2 avg pool, linear, LIF",
    "bias": "the same, with a real bias on every layer (bias neurons); behind a pool "
            "the bias exceeds one INT16 synapse and is split across bias neurons",
    "gap": "conv, LIF, global average pool, linear with bias, LIF",
    "maxpool": "conv, LIF, OR pool, conv, LIF, OR pool, linear, LIF",
    "bits": "rank1's shape reading 3-bit input: 3 axons per value, weights w, 2w, 4w",
    "signed": "rank1's shape with signed spikes and biases: two neurons per unit, "
              "with separate positive and negative thresholds",
    "signed_maxpool": "signed spikes through OR pools: one OR neuron per sign per window",
    "per_channel_tau": "rank1's shape with a different leak in every channel",
}


def _random_layers(rng, leak, small=False, variant="rank1"):
    """Small network of one of the layer types the tool deploys (VARIANTS).

    small=False uses INT16-sized weights, like a real export. small=True uses
    weights of a few units, so membranes land EXACTLY on thresholds and on
    multiples of 2**leak. Those are the cases where `>` versus `>=`, or floor
    versus truncation of a negative membrane, would show up; with large random
    weights they almost never occur and an off-by-one passes unnoticed.
    """
    import numpy as np

    def w(*shape, limit=None):
        if small:
            return rng.integers(-4, 7, size=shape).astype(np.int64)
        hi = 32767 if limit is None else limit
        return rng.integers(-int(hi * 0.8), hi + 1, size=shape).astype(np.int64)

    def th(n, scale):
        t = rng.integers(3, 9, size=n) if small else rng.integers(30000, 90000, size=n)
        return scale * t.astype(np.int64) + scale - 1

    def bias(n, scale):
        # in membrane units, so up to scale * INT16_MAX behind a pool
        lim = 6 if small else scale * 30000
        return rng.integers(-lim, lim + 1, size=n).astype(np.int64)

    def leaks(n):
        if variant == "per_channel_tau":
            return [int(x) for x in rng.integers(1, 4, size=n)]
        return leak

    orpool = {"kind": "orpool", "pool": 2, "scale": 1, "threshold": np.zeros(1, dtype=np.int64),
              "bias": None, "tau": None, "leak": 0, "memoryless": True}

    if variant in ("maxpool", "signed_maxpool"):
        sg = variant == "signed_maxpool"
        layers = [
            {"kind": "conv", "w": w(4, 2, 3, 3), "pool": 1, "scale": 1, "stride": (1, 1),
             "padding": (1, 1), "threshold": th(4, 1), "leak": leak, "signed": sg},
            dict(orpool, signed=sg),
            {"kind": "conv", "w": w(3, 4, 5, 5), "pool": 1, "scale": 1, "stride": (1, 1),
             "padding": (2, 2), "threshold": th(3, 1), "leak": leak, "signed": sg},
            dict(orpool, signed=sg),
            {"kind": "linear", "w": w(5, 3 * 2 * 2), "pool": 1, "scale": 1,
             "threshold": th(5, 1), "leak": leak, "signed": sg},
        ]
        if sg:
            for L in layers:
                if L["kind"] != "orpool":
                    L["threshold_neg"] = th(L["w"].shape[0], 1)
        return layers
    if variant == "gap":
        return [
            {"kind": "conv", "w": w(4, 2, 3, 3), "pool": 1, "scale": 1, "stride": (1, 1),
             "padding": (1, 1), "threshold": th(4, 1), "bias": bias(4, 1), "leak": leak},
            # GAP over the 8x8 map: a pool of 8, membrane 64x finer
            {"kind": "linear", "w": w(5, 4), "pool": 8, "scale": 64,
             "threshold": th(5, 64), "bias": bias(5, 64), "leak": leak},
        ]
    bits = 3 if variant == "bits" else 1
    first_limit = 32767 >> (bits - 1)
    layers = [
        {"kind": "conv", "w": w(4, 2, 3, 3, limit=first_limit), "pool": 1, "scale": 1,
         "stride": (1, 1), "padding": (1, 1), "threshold": th(4, 1), "leak": leaks(4),
         "input_bits": bits},
        {"kind": "conv", "w": w(3, 4, 5, 5), "pool": 2, "scale": 4, "stride": (1, 1),
         "padding": (2, 2), "threshold": th(3, 4), "leak": leaks(3)},
        {"kind": "linear", "w": w(5, 3 * 2 * 2), "pool": 2, "scale": 4,
         "threshold": th(5, 4), "leak": leaks(5)},
    ]
    if variant in ("bias", "signed"):
        for L in layers:
            L["bias"] = bias(L["w"].shape[0], L["scale"])
    if variant == "signed":
        for L in layers:
            L["signed"] = True
            L["threshold_neg"] = th(L["w"].shape[0], L["scale"])
    return layers


def selftest(args):
    import numpy as np
    import torch
    from snnsearch.hs_export import IntegerSim, unroll_to_hs_api, input_axons
    from hs_api.api import CRI_network

    in_shape, T, flush = (2, 8, 8), 10, 5
    all_ok = True
    names = list(VARIANTS)
    for variant, small, leak in [(v, sm, lk) for v in names
                                 for sm in (False, True) for lk in (1, 2, 3)]:
        rng = np.random.default_rng(10 * leak + small + 100 * names.index(variant))
        layers = _random_layers(rng, leak, small, variant)
        bits = layers[0].get("input_bits", 1)
        # hs_lif_neuron sets the noise shift to -17, which is zero noise on
        # every commit; see its docstring
        axons, conns, groups = unroll_to_hs_api(layers, in_shape)
        group_keys = [keys for layer_groups in groups for keys, _sign in layer_groups]
        all_keys = [k for keys in group_keys for k in keys]
        net = CRI_network(axons=axons, connections=conns, outputs=all_keys,
                          target="simpleSim")

        top = 2 ** bits - 1
        active = rng.random((T,) + in_shape) < 0.35
        levels = rng.integers(1, top + 1, size=(T,) + in_shape)
        frames = torch.from_numpy((active * levels).astype(np.float64))
        sim = IntegerSim(layers, in_shape, device="cpu")

        # IntegerSim's membranes and spikes at every step, in unroll's key order
        ref_V, ref_S = [], []
        for grps, _outputs in sim.steps(frames, flush):
            ref_V.append(torch.cat([v.reshape(-1) for v, _s in grps]).cpu().numpy().copy())
            ref_S.append(set(k for (v, s), keys in zip(grps, group_keys)
                             for j, k in enumerate(keys) if s.reshape(-1)[j] > 0))

        bad_steps = 0
        for t in range(T + flush):
            inputs = input_axons(frames[t], bits) if t < T else []
            pots, spk = net.step(inputs, membranePotential=True)
            by_key = {k: float(p) for k, p in pots}
            hs_V = np.array([by_key[k] for k in all_keys])
            hs_S = set(spk)
            if hs_S != ref_S[t] or not np.array_equal(hs_V, ref_V[t]):
                bad_steps += 1
                if bad_steps <= 3:
                    dv = np.flatnonzero(hs_V != ref_V[t])
                    print(f"  {variant} leak={leak} t={t}: spikes only in hs_api "
                          f"{sorted(hs_S - ref_S[t])[:5]}, only in IntegerSim "
                          f"{sorted(ref_S[t] - hs_S)[:5]}, {dv.size} membranes differ")
        ok = bad_steps == 0
        all_ok &= ok
        n_spikes = sum(len(s) for s in ref_S)
        n_bias = sum(1 for k in conns if k.startswith("BiasN"))
        neg = int(sum((v < 0).sum() for v in ref_V))
        print(f"{variant:15s} {'small' if small else 'INT16'} leak {leak}: "
              f"{len(all_keys):4d} neurons, {n_bias:3d} bias neurons, {n_spikes:5d} spikes, "
              f"{neg:5d} negative membranes, "
              f"{'IDENTICAL' if ok else f'{bad_steps} of {T + flush} steps differ'}")
    print("\nSELFTEST PASSED: IntegerSim reproduces hs_api simpleSim exactly"
          if all_ok else "\nSELFTEST FAILED")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# 2. per-clip comparison
# ---------------------------------------------------------------------------

def _clips(spec, args):
    """{split: [(frames uint8 (T, C, H, W), label), ...]} from a clips file or the dataset."""
    import torch
    splits = ["val", "test"] if args.split == "both" else [args.split]
    inp = spec["input"]
    if args.clips:
        saved = torch.load(args.clips, weights_only=False)
        meta = saved["meta"]
        if (meta["T"], meta["resize_to"]) != (inp.T, inp.resize_to):
            raise SystemExit(f"{args.clips} holds T={meta['T']} resize={meta['resize_to']}, "
                             f"the checkpoint wants T={inp.T} resize={inp.resize_to}")
        print(f"clips from {args.clips} ({meta})")
        out = {}
        for sp in splits:
            if sp not in saved["splits"]:
                raise SystemExit(f"{args.clips} has no {sp!r} split")
            fr, lab = saved["splits"][sp]
            out[sp] = list(zip(fr, lab.tolist()))
    else:
        from snnsearch.data.builtin import dvs128_gesture
        from snnsearch.data.loaders import build_dataloaders
        from snnsearch.encoders import build_encoder
        bundle = dvs128_gesture(args.data_root, T=inp.T)
        enc = build_encoder("passthrough", T=inp.T, resize_to=inp.resize_to or None)
        _tr, va, te = build_dataloaders(bundle, batch_size=16, encoder=enc, seed=args.seed,
                                        val_split=args.val_split)
        loaders = {"val": va, "test": te}
        out = {}
        for sp in splits:
            items = []
            for x, y, _ in loaders[sp]:
                items += list(zip(x.to(torch.uint8), y.tolist()))
            out[sp] = items
    if args.n:
        out = {sp: items[:args.n] for sp, items in out.items()}
    return out


def save_clips(clips, spec, args):
    import torch
    payload = {"meta": {"T": spec["input"].T, "resize_to": spec["input"].resize_to,
                        "val_split": args.val_split, "seed": args.seed},
               "splits": {sp: (torch.stack([f for f, _ in items]),
                               torch.tensor([y for _, y in items]))
                          for sp, items in clips.items()}}
    torch.save(payload, args.save_clips)
    mb = os.path.getsize(args.save_clips) / 1e6
    print(f"saved {', '.join(f'{len(v)} {k}' for k, v in clips.items())} clips "
          f"to {args.save_clips} ({mb:.0f} MB)")


def _torch_preds(net, items, device, flush, batch=16):
    import torch
    from snnsearch.train import forward_over_time
    preds = []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            x = torch.stack([f for f, _ in items[i:i + batch]]).float().to(device)
            out = forward_over_time(net, x, flush_steps=flush)
            preds += out.argmax(1).cpu().tolist()
    return preds


def _real_simplesim(layers, in_shape, items, flush, target="simpleSim", core=0):
    """Run clips through hs_api on the full network: its own simple_sim, or
    with target "CRI" the FPGA itself (on a host with hs_bridge and a flashed
    bitstream).

    Slow and memory-hungry by construction: hs_api stores every synapse as a
    Python object. Measured on a smaller net it costs ~600 bytes and ~50 us
    per synapse to build, so rank1 (239M synapses) needs on the order of
    150 GB of RAM and a few hours before the first clip runs.

    Returns [(pred, per-step output rows, mean latency per step or None)].
    On the FPGA, step() returns (spikes, latency, hbm_access); the latency is
    kept, and membranes are cleared between clips with fpga_controller.clear,
    as hs_api's own hardware tests do.
    """
    import torch
    from snnsearch.hs_export import unroll_to_hs_api, input_axons
    from hs_api.api import CRI_network

    t0 = time.time()
    print(f"hs_api {target}: unrolling the network into hs_api dictionaries", flush=True)
    axons, conns, groups = unroll_to_hs_api(layers, in_shape)
    n_syn = sum(len(v) for v in axons.values()) + sum(len(v[0]) for v in conns.values())
    print(f"  {len(axons)} axons, {len(conns)} neurons, {n_syn:,} synapses "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    # output units: positive neurons, and negative ones for signed output
    out_groups = groups[-1]
    out_keys = [k for keys, _sign in out_groups for k in keys]
    n_cls = len(out_groups[0][0])
    n_neurons = len(conns)
    extra = {"coreID": core} if target == "CRI" else {}
    net = CRI_network(axons=axons, connections=conns, outputs=out_keys, target=target, **extra)
    del axons, conns
    print(f"  CRI_network built ({(time.time() - t0) / 60:.1f} min)", flush=True)
    clear = None
    if target == "CRI":
        from hs_bridge.FPGA_Execution import fpga_controller
        # padding to whole groups of 32 per neuron model adds neurons; clear them too
        try:
            n_neurons = sum(1 for n in net.connectome.connectomeDict.values()
                            if n.get_neuron_type() == "neuron")
        except AttributeError:
            pass
        clear = lambda: fpga_controller.clear(n_neurons, False, core)     # noqa: E731

    pos = {k: (i, sign) for keys, sign in out_groups for i, k in enumerate(keys)}
    bits = layers[0].get("input_bits", 1)
    results = []
    for ci, (frames, _label) in enumerate(items):
        t1 = time.time()
        # fresh membranes per clip, as Christopher's DVS example clears the FPGA
        if clear is not None:
            clear()
        else:
            net.simpleSim.initialize_sim_vars(net.simpleSim.numNeurons)
        counts = [0] * n_cls
        per_step, latency = [], []
        for t in range(frames.shape[0] + flush):
            inputs = input_axons(frames[t], bits) if t < frames.shape[0] else []
            row = [0] * n_cls
            out = net.step(inputs)
            if isinstance(out, tuple):          # the FPGA: (spikes, latency, hbm_access)
                out, lat = out[0], out[1]
                latency.append(lat)
            for k in out:
                i, sign = pos[k]
                row[i] += sign
                counts[i] += sign
            per_step.append(row)
        pred = int(torch.tensor(counts).argmax().item())
        mean_lat = (sum(float(x) for x in latency) / len(latency)) if latency else None
        results.append((pred, per_step, mean_lat))
        print(f"  clip {ci}: predicted {pred} ({time.time() - t1:.0f}s"
              + (f", latency {mean_lat:.0f} per step" if mean_lat is not None else "") + ")",
              flush=True)
    return results


def _per_layer_torch(net, frames, device, flush):
    """Spike count per spiking layer per timestep, for one clip."""
    import torch
    from snnsearch.neuron import HardwareLIFNode
    from spikingjelly.activation_based import functional
    nodes = [m for m in net.conv_fc if isinstance(m, HardwareLIFNode)]
    rec = [[] for _ in nodes]
    hooks = [n.register_forward_hook(lambda m, i, o, k=k: rec[k].append(int(o.sum().item())))
             for k, n in enumerate(nodes)]
    T = frames.shape[0]
    with torch.no_grad():
        for t in range(T + flush):
            x = frames[t] if t < T else torch.zeros_like(frames[0])
            net(x.unsqueeze(0).to(device))
    functional.reset_net(net)
    for h in hooks:
        h.remove()
    return rec


def compare(args):
    import torch
    from snnsearch.hs_export import (load_deployed, integer_layers, IntegerSim,
                                     make_chip_faithful)
    from snnsearch.train import hardware_flush_steps

    # TF32 convolutions round the membrane to ~3 significant digits, which is
    # coarser than the INT16 grid. Off, so the model is scored on the grid.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    net, spec, ckpt = load_deployed(args.ckpt, device)
    flush = hardware_flush_steps(net)
    trained = integer_layers(net)
    print(f"checkpoint: {args.ckpt}")
    print(f"trained tau per layer: {[L['tau'] for L in trained]}, flush steps {flush}")
    for i, L in enumerate(trained):
        print(f"  layer {i}: {L['kind']:6s} w{tuple(L['w'].shape)} pool {L['pool']} "
              f"scale {L['scale']} threshold {L['threshold'].min()}..{L['threshold'].max()}")

    bad_tau = [L["tau"] for L in trained if L["leak"] is None]
    if bad_tau and args.tau is None:
        raise SystemExit(
            f"\nThe checkpoint trained tau {sorted(set(bad_tau))}, and simpleSim's "
            f"leak is a shift (v // 2**leak), so those values do not exist on the "
            f"chip. Rerun with --tau 2 or --tau 4.")

    clips = _clips(spec, args)
    splits = list(clips)
    if args.save_clips:
        save_clips(clips, spec, args)
    report = {"ckpt": args.ckpt, "tau": args.tau, "trained_tau": [L["tau"] for L in trained],
              "flush_steps": flush, "splits": {}}

    if bad_tau or args.tau is not None:
        # the trained model as-is, so the cost of changing tau is on record
        for split in splits:
            p = _torch_preds(net, clips[split], device, flush)
            y = [lab for _, lab in clips[split]]
            acc = sum(a == b for a, b in zip(p, y)) / len(y)
            print(f"\n[{split}] trained model, tau {report['trained_tau']}: {acc:.4f} on {len(y)} clips")
            report["splits"].setdefault(split, {})["trained_model_acc"] = acc

    make_chip_faithful(net, args.tau)
    layers = integer_layers(net)
    sim = IntegerSim(layers, (spec["input"].C, spec["input"].resize_to, spec["input"].resize_to),
                     device=device)

    for split in splits:
        t0 = time.time()
        p_torch = _torch_preds(net, clips[split], device, flush)
        labels = [lab for _, lab in clips[split]]
        frames = [f.float() for f, _ in clips[split]]
        p_sim, sim_steps = [], []
        for f in frames:
            counts, rec = sim.run(f, flush, record_output=True)
            p_sim.append(int(counts.argmax().item()))
            sim_steps.append(rec)
        dis = [i for i, (a, b) in enumerate(zip(p_torch, p_sim)) if a != b]
        acc_t = sum(a == b for a, b in zip(p_torch, labels)) / len(labels)
        acc_s = sum(a == b for a, b in zip(p_sim, labels)) / len(labels)
        print(f"\n[{split}] {len(labels)} clips, tau {[L['tau'] for L in layers]}, "
              f"{time.time() - t0:.0f}s")
        print(f"  SpikingJelly (chip-faithful)  accuracy {acc_t:.4f}")
        print(f"  simpleSim (IntegerSim)        accuracy {acc_s:.4f}")
        print(f"  per-clip disagreements        {len(dis)}"
              + (f"  clips {dis[:20]}" if dis else "  PERFECT MATCH"))
        print(f"  largest |membrane|            {sim.max_abs_membrane:.0f} "
              f"(simpleSim's 32-bit membrane wraps at {2 ** 31})")
        report["splits"].setdefault(split, {}).update({
            "n": len(labels), "spikingjelly_acc": acc_t, "simplesim_acc": acc_s,
            "disagreements": dis, "pred_spikingjelly": p_torch, "pred_simplesim": p_sim,
            "labels": labels})

        if args.per_layer:
            f0 = frames[0]
            a = _per_layer_torch(net, f0, device, flush)
            _, b = sim.run(f0, flush, record=True)
            for i, (ra, rb) in enumerate(zip(a, b)):
                same = ra == rb
                print(f"  clip 0 layer {i}: {'same' if same else 'DIFFERENT'} "
                      f"spike counts per step (total {sum(ra)} vs {sum(rb)})")
                if not same:
                    print(f"    torch {ra}\n    sim   {rb}")

        if args.real_simplesim:
            k = min(args.real_simplesim, len(frames))
            in_shape = (spec["input"].C, spec["input"].resize_to, spec["input"].resize_to)
            real = _real_simplesim(layers, in_shape, clips[split][:k], flush,
                                   target=args.hs_target, core=args.core)
            # every output spike at every timestep, not only the argmax
            r_pred = [p for p, _, _ in real]
            dis_r = [i for i in range(k) if r_pred[i] != p_sim[i]]
            exact = [i for i, (_, steps, _) in enumerate(real) if steps != sim_steps[i]]
            acc_r = sum(a == b for a, b in zip(r_pred, labels[:k])) / k
            acc_sk = sum(a == b for a, b in zip(p_sim[:k], labels[:k])) / k
            lats = [lat for _, _, lat in real if lat is not None]
            where = "the FPGA (CRI)" if args.hs_target == "CRI" else "real hs_api simpleSim"
            print(f"\n[{split}] {where} on {k} clips")
            print(f"  accuracy                                {acc_r:.4f} "
                  f"(IntegerSim on the same clips {acc_sk:.4f})")
            print(f"  predictions differing from IntegerSim   {len(dis_r)}"
                  + (f"  clips {dis_r}" if dis_r else ""))
            print(f"  clips whose output spike trains differ  {len(exact)}"
                  + (f"  clips {exact}" if exact else "  (identical at every step)"))
            if lats:
                print(f"  mean latency per step                   {sum(lats) / len(lats):.0f}")
            report["splits"][split]["real_simplesim"] = {
                "target": args.hs_target, "n": k, "pred": r_pred, "accuracy": acc_r,
                "integersim_accuracy_same_clips": acc_sk,
                "pred_disagreements_vs_integersim": dis_r,
                "output_train_mismatch_clips": exact,
                "mean_latency_per_step": (sum(lats) / len(lats)) if lats else None}

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hs-path", nargs="*", default=[],
                    help="directories to put on sys.path for hs_api, connectome_utils, "
                         "fxpmath when they are not installed (not needed under poetry run)")
    ap.add_argument("--ckpt")
    ap.add_argument("--data-root", help="DVS128 Gesture root (holds frames_number_16_split_by_number)")
    ap.add_argument("--save-clips", help="write the encoded clips to this file, so another "
                                         "machine can run without the dataset")
    ap.add_argument("--clips", help="read clips from a --save-clips file instead of --data-root")
    ap.add_argument("--real-simplesim", type=int, default=0, metavar="K",
                    help="also run the first K clips of each split through hs_api's own "
                         "simple_sim on the FULL network (~150 GB RAM, hours)")
    ap.add_argument("--hs-target", default="simpleSim", choices=["simpleSim", "CRI"],
                    help="where --real-simplesim runs its clips: hs_api's simple_sim, or "
                         "CRI, the FPGA (crisdsc0: book it, flash the bitstream first)")
    ap.add_argument("--core", type=int, default=0, help="FPGA core for --hs-target CRI")
    ap.add_argument("--tau", type=int, default=None,
                    help="power-of-two tau to deploy with, on every layer")
    ap.add_argument("--split", default="both", choices=["val", "test", "both"])
    ap.add_argument("--n", type=int, default=0, help="first N clips only (0 = all)")
    ap.add_argument("--seed", type=int, default=1, help="train/val split seed (runs used 1)")
    ap.add_argument("--val-split", default="clip", choices=["clip", "subject"],
                    help="clip (default) re-scores a pre-October-2026 checkpoint on the "
                         "clips it was selected on. Its training saw every subject, "
                         "so a subject split would not be held out for it either. "
                         "Use subject for checkpoints trained with the subject split.")
    ap.add_argument("--per-layer", action="store_true",
                    help="also compare per-layer spike counts on clip 0")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("-o", "--out", default="match_report.json")
    args = ap.parse_args()
    for p in args.hs_path:
        sys.path.insert(0, p)
    if args.selftest:
        return selftest(args)
    if not (args.ckpt and (args.data_root or args.clips)):
        ap.error("--ckpt and one of --data-root / --clips are required unless --selftest")
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
