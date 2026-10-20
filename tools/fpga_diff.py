"""Where the FPGA departs from simpleSim, neuron by neuron and step by step.

Run on crisdsc0 from ~/hs_api with `poetry run python`, the way
tools/match_simplesim.py runs (book the FPGA and check `top` first):

    # one validation clip, every membrane at every step, once on each target
    python ~/configurable-snn-classifier/tools/fpga_diff.py record --ckpt ~/fpga_run/rank2.pth --clips ~/fpga_run/rank2_val_clips.pt --clip 0 --target simpleSim -o ~/fpga_run/rec_sim.npz --hs-path ~/hs_api ~/hs_bridge ~/connectome_utils
    python ~/configurable-snn-classifier/tools/fpga_diff.py record --ckpt ~/fpga_run/rank2.pth --clips ~/fpga_run/rank2_val_clips.pt --clip 0 --target CRI -o ~/fpga_run/rec_fpga.npz --hs-path ~/hs_api ~/hs_bridge ~/connectome_utils

    # the first step and neurons that differ, as text and as an HTML report
    python ~/configurable-snn-classifier/tools/fpga_diff.py compare ~/fpga_run/rec_sim.npz ~/fpga_run/rec_fpga.npz -o ~/fpga_run/fpga_diff.html

    # one record alone, no FPGA needed: which neurons the negative-leak bug can hit
    python ~/configurable-snn-classifier/tools/fpga_diff.py show ~/fpga_run/rec_sim.npz -o ~/fpga_run/rank2_exposure.html

    # tiny networks that each test one suspect, on both targets
    python ~/configurable-snn-classifier/tools/fpga_diff.py micro -o ~/fpga_run/fpga_micro.html --hs-path ~/hs_api ~/hs_bridge ~/connectome_utils

`compare` aligns the two records by hs_api neuron key ("L{layer}_{index}",
"L{layer}n_{index}" for the negative neuron of a signed unit) and reports the
first step where any membrane or output spike differs, the layer nearest the
input that differs then (where the error starts), and for its neurons the
membrane one step before, both values, and a guess at the cause. A negative
membrane whose FPGA value equals a logical (unsigned) shift instead of an
arithmetic one is the L6m negative-leak bug.

`show` reads one record and reports every neuron, by layer and step: spikes
(read off the membranes, since a neuron fires at a step when its membrane
the step before is above threshold), bursts per step, firing rates, silent
and always-on neurons, synaptic events, how close membranes run to threshold,
when the prediction settles, and where membranes go negative. On the L6m
FPGA each negative membrane leaks with a logical shift, so that map is where
the bug can act, predicted from the simulator alone. `compare` adds, per
layer, the spikes the FPGA missed and the ones it added.

`micro` builds networks of a few neurons, each asking one question:
negative membranes leaking at shifts 1 to 7, many spikes in one step (to the
output and between neurons), large fan-out, thresholds above 2**17, and an
always-on bias neuron. It runs each on simpleSim and on the FPGA and reports
pass or fail with the smallest failing case.

The reports are single HTML files with no outside dependencies: copy them
back with scp and open them in a browser.
"""
import argparse
import html
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# tools/ holds source clones of hs_api, fxpmath and connectome_utils; keep
# them from shadowing the installed packages (as match_simplesim.py does)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, ROOT)

WORD = 2 ** 36                      # a neuron's state word in URAM
MP_BITS = 32                        # its membrane potential: bits 31:0
PLUS, MINUS = "+", "−"        # outside f-string expressions: Python 3.10 on crisdsc0


def signed_bits(v, bits):
    """v as a bits-wide two's-complement value."""
    w = 2 ** bits
    v = int(v) % w
    return v - w if v >= w // 2 else v


def signed_mp(v):
    """The membrane potential in a state word, read as hs_api's own
    read_membrane reads it on the L6m testing-suite commits: bits 31:0,
    sign-extended. The FPGA's step(membranePotential=True) hands back the raw
    36-bit word, so a negative membrane read as 36 bits looks like a huge
    positive one. simpleSim hands back the value itself, which this leaves
    as it is (simpleSim's membrane is 32 bits too)."""
    return signed_bits(v, MP_BITS)


def leak_arith(v, s):
    """simpleSim's leak, V -= V >> s with an arithmetic shift (floor)."""
    return v - (v >> s)


def leak_logical(v, s, bits=MP_BITS):
    """The L6m bug: the shift treats a negative membrane as unsigned. bits is
    the width shifted: 32 for the membrane hs_api reads, 36 for the whole
    state word. Nothing documents which, so compare and micro try both."""
    return signed_bits(v - ((v % 2 ** bits) >> s), bits)


# ---------------------------------------------------------------------------
# hs_api, on either target
# ---------------------------------------------------------------------------

def build(axons, conns, out_keys, target, core=0):
    """(CRI_network, clear) for target "simpleSim" or "CRI". clear() resets
    every membrane; on the FPGA through hs_bridge, as hs_api's tests do."""
    from hs_api.api import CRI_network
    extra = {"coreID": core} if target == "CRI" else {}
    try:
        net = CRI_network(axons=axons, connections=conns, outputs=out_keys, target=target,
                          **extra)
    except TypeError as exc:
        if not extra or core != 0 or "coreID" not in str(exc):
            raise
        net = CRI_network(axons=axons, connections=conns, outputs=out_keys, target=target)
    fast_index_lookup(net)
    if target == "CRI":
        from hs_bridge.FPGA_Execution import fpga_controller
        try:
            n = sum(1 for x in net.connectome.connectomeDict.values()
                    if x.get_neuron_type() == "neuron")
        except AttributeError:
            n = len(conns)
        return net, (lambda: fpga_controller.clear(n, False, core))
    return net, (lambda: net.simpleSim.initialize_sim_vars(net.simpleSim.numNeurons))


# connectome_utils' neuron lookups that scan every entry, and the index each
# one matches
INDEX_LOOKUPS = (("get_neuron_by_idx", "get_coreTypeIdx"),
                 ("get_neuron_by_hbmIdx", "get_hbmIdx"))


def fast_index_lookup(net):
    """Replace connectome_utils' neuron lookups by index, which scan every
    entry, with dictionaries built once. hs_api calls one for every neuron at
    every step when membranes are read: get_neuron_by_idx on simpleSim, and
    get_neuron_by_hbmIdx on the FPGA in the L6m testing-suite commits. Rank
    2's 47k neurons made that billions of comparisons a step: the simpleSim
    record sat for hours after the build, and the FPGA one took 8 minutes a
    step. Same answers as the scans: the first neuron (not axon) with that
    index, None when there is none."""
    conn = getattr(net, "connectome", None)
    entries = getattr(conn, "connectomeDict", None)
    if not entries:
        return
    try:
        neurons = [n for n in entries.values() if n.get_neuron_type() == "neuron"]
    except AttributeError:
        return
    for lookup, index in INDEX_LOOKUPS:
        if not hasattr(conn, lookup):
            continue
        try:
            by_idx = {}
            for n in neurons:
                by_idx.setdefault(getattr(n, index)(), n)
        except AttributeError:
            continue
        setattr(conn, lookup, by_idx.get)


def step(net, inputs):
    """(membranes {key: int}, output spike keys, counter or None) for one step."""
    mem, rest = net.step(inputs, membranePotential=True)
    if isinstance(rest, tuple):              # the FPGA: (spikes, counter, hbm_access)
        spikes, counter = rest[0], rest[1]
    else:
        spikes, counter = rest, None
    return {k: int(v) for k, v in mem}, list(spikes), counter


def run(net, clear, inputs_per_step, keys, out_keys, progress=False):
    """Membranes (steps x keys), output spikes (steps x out_keys), counters,
    and on the FPGA how often each value of a state word's top four bits
    (35:32, outside the membrane) came back: all 0 or all F for sign
    extension, anything else means those bits hold something of their own.
    progress prints about ten lines: on the FPGA, reading every membrane back
    takes minutes a step for a network of rank 2's size."""
    t0 = time.time()
    clear()
    idx = {k: i for i, k in enumerate(keys)}
    oidx = {k: i for i, k in enumerate(out_keys)}
    S = len(inputs_per_step)
    M = np.zeros((S, len(keys)), np.int64)
    O = np.zeros((S, len(out_keys)), bool)
    counters = []
    tops = {}
    for t, inp in enumerate(inputs_per_step):
        mem, spikes, counter = step(net, inp)
        fpga = counter is not None
        for k, v in mem.items():
            i = idx.get(k)
            if i is not None:
                M[t, i] = signed_mp(v)
                if fpga:
                    top = (int(v) % WORD) >> MP_BITS
                    tops[top] = tops.get(top, 0) + 1
        for k in spikes:
            j = oidx.get(k)
            if j is not None:
                O[t, j] = True
        counters.append(np.nan if counter is None else float(counter))
        if progress and ((t + 1) % max(1, S // 10) == 0 or t == 0 or t + 1 == S):
            took = (time.time() - t0) / 60
            print(f"  step {t + 1}/{S} ({took:.1f} min, about "
                  f"{took / (t + 1) * (S - t - 1):.0f} min left)", flush=True)
    return M, O, counters, {f"{k:X}": n for k, n in sorted(tops.items())}


# ---------------------------------------------------------------------------
# record: one clip of a trained network
# ---------------------------------------------------------------------------

def neuron_table(layers, in_shape):
    """key -> [layer, sign, channel, position, theta, shift, shape] for every
    layer neuron, mirroring hs_export.unroll_to_hs_api's keys."""
    from snnsearch.hs_export import _layer_shapes, _normalize
    layers = [_normalize(L) for L in layers]
    shapes = _layer_shapes(layers, in_shape)
    table = {}
    for li, L in enumerate(layers):
        shp = [int(s) for s in shapes[li]]
        n = int(np.prod(shp))
        per_ch = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        signs = ((1, ""), (-1, "n")) if L["signed"] else ((1, ""),)
        for sign, tag in signs:
            orpool = L["kind"] == "orpool"
            th = None if orpool else (L["threshold"] if sign > 0 else L["threshold_neg"])
            for j in range(n):
                ch = j // per_ch
                table[f"L{li}{tag}_{j}"] = [li, sign, ch, j % per_ch,
                                            None if orpool else int(th[ch]),
                                            None if orpool else int(L["leak"][ch]), shp]
    return table


def record_network(layers, in_shape, frames, flush, target, core=0):
    """Run one clip on target, step by step, and keep every layer neuron's
    membrane and every output spike, plus each neuron's fan-out so a report
    can count the synaptic events the chip handles."""
    from snnsearch.hs_export import unroll_to_hs_api, input_axons
    t0 = time.time()
    axons, conns, groups = unroll_to_hs_api(layers, in_shape)
    table = neuron_table(layers, in_shape)
    keys = list(table)
    out_keys = [k for ks, _sign in groups[-1] for k in ks]
    print(f"{target}: {len(axons)} axons, {len(conns)} neurons "
          f"({(time.time() - t0) / 60:.1f} min to unroll)", flush=True)
    bits = int(layers[0].get("input_bits", 1) or 1)
    T = int(frames.shape[0])
    inputs = [input_axons(frames[t], bits) for t in range(T)] + [[] for _ in range(flush)]
    fan_out = np.array([len(conns[k][0]) for k in keys], np.int64)
    bias_fan_out = int(sum(len(c[0]) for k, c in conns.items() if k not in table))
    axon_events = int(sum(len(axons[a]) for step_in in inputs for a in step_in))
    net, clear = build(axons, conns, out_keys, target, core)
    del axons, conns
    print(f"  network built ({(time.time() - t0) / 60:.1f} min)", flush=True)
    M, O, counters, tops = run(net, clear, inputs, keys, out_keys, progress=True)
    print(f"  {len(inputs)} steps recorded ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return {"target": target, "keys": keys, "out_keys": out_keys, "membrane": M,
            "out_spikes": O, "counter": counters, "table": table, "T": T, "flush": flush,
            "fan_out": fan_out, "bias_fan_out": bias_fan_out, "axon_events": axon_events,
            "mp_bits": MP_BITS, "word_top_bits": tops}


RECORD_ARRAYS = ("membrane", "out_spikes", "counter", "keys", "out_keys", "fan_out")


def save_record(path, rec):
    arrays = {k: np.asarray(rec[k]) for k in RECORD_ARRAYS if rec.get(k) is not None}
    arrays["counter"] = np.asarray(rec["counter"], float)
    np.savez_compressed(path, meta=np.asarray(json.dumps(
        {k: v for k, v in rec.items() if k not in RECORD_ARRAYS})), **arrays)
    print(f"wrote {path}")


def load_record(path):
    """A saved record. Records from before fan-out was kept have fan_out None."""
    z = np.load(path, allow_pickle=False)
    rec = json.loads(str(z["meta"]))
    rec.update(membrane=z["membrane"], out_spikes=z["out_spikes"], counter=z["counter"],
               keys=[str(k) for k in z["keys"]], out_keys=[str(k) for k in z["out_keys"]],
               fan_out=z["fan_out"] if "fan_out" in z.files else None)
    if rec.get("target") == "CRI" and "mp_bits" not in rec:
        # saved before the 32-bit reading: the 36-bit value kept every bit of
        # the raw word, so it can be read again the way hs_api reads it
        raw = np.asarray(rec["membrane"], np.int64) % WORD
        tops, counts = np.unique(raw >> MP_BITS, return_counts=True)
        mp = raw % 2 ** MP_BITS
        rec["membrane"] = np.where(mp >= 2 ** (MP_BITS - 1), mp - 2 ** MP_BITS, mp)
        rec.update(mp_bits=MP_BITS, redecoded=True,
                   word_top_bits={f"{int(k):X}": int(n) for k, n in zip(tops, counts)})
    return rec


def cmd_record(a):
    import torch
    from snnsearch.hs_export import load_deployed, integer_layers, make_chip_faithful
    from snnsearch.train import hardware_flush_steps
    net, spec, _ = load_deployed(a.ckpt, "cpu")
    flush = hardware_flush_steps(net)
    make_chip_faithful(net, a.tau)
    layers = integer_layers(net)
    saved = torch.load(a.clips, weights_only=False)
    if a.split not in saved["splits"]:
        raise SystemExit(f"{a.clips} has no {a.split!r} split")
    frames, labels = saved["splits"][a.split]
    inp = spec["input"]
    side = inp.resize_to or inp.H
    rec = record_network(layers, (inp.C, side, side), frames[a.clip], flush, a.target, a.core)
    rec.update(clip=a.clip, label=int(labels[a.clip]), split=a.split,
               ckpt=os.path.abspath(a.ckpt))
    save_record(a.out, rec)
    return 0


# ---------------------------------------------------------------------------
# compare: where two records part
# ---------------------------------------------------------------------------

def explains_logical(v_prev, ref, hw, shift):
    """Whether hw is what a logical-shift leak leaves where the simulator's
    arithmetic leak left ref. The hardware's order is not documented, so both
    are tried: the leak before this step's input (same input on both sides),
    and after it (the value the simulator leaked into ref is recovered)."""
    if not shift or hw == ref:
        return False
    for bits in (MP_BITS, 36):             # the shifter's width is not documented
        def logical(v):
            return signed_mp(leak_logical(v, shift, bits))
        if v_prev < 0 and signed_mp(hw - ref - logical(v_prev)
                                    + leak_arith(v_prev, shift)) == 0:
            return True
        if shift > 20:                      # the search below is over 2**shift values
            continue
        k = 2 ** shift
        est = (ref * k) // (k - 1)
        for v in range(est - k - 2, est + k + 3):
            if v < 0 and leak_arith(v, shift) == ref and logical(v) == hw:
                return True
    return False


def classify(v_prev, ref, hw, shift, theta=None):
    """A guess at why one neuron's FPGA membrane differs at its first step.
    Every membrane matched the step before, so both targets fired the same
    neurons; what is left is the leak, the reset, or the input arriving."""
    if explains_logical(v_prev, ref, hw, shift):
        return "negative leak (logical shift)"
    if v_prev < 0 or ref < 0:
        return "negative membrane, not the logical-shift pattern"
    if theta is not None and v_prev > theta:
        return "fired this step: reset or input differs"
    return "input differs (spikes lost or added on the way in)"


def output_classes(out_keys, table):
    """class -> [(output column, sign)]: a class scores its positive
    neuron's spikes minus its negative twin's."""
    cls = {}
    for j, k in enumerate(out_keys):
        _li, sign, ch, pos, *_ = table[k]
        c = ch if len(table[k][6]) == 1 else ch * table[k][6][1] * table[k][6][2] + pos
        cls.setdefault(c, []).append((j, sign))
    return cls


def layer_groups(table, keys):
    """(group of each neuron, group names, labels, index arrays) for each
    layer's positive and negative neuron groups, positive first."""
    group_of = [(table[k][0], table[k][1]) for k in keys]
    names = sorted(set(group_of), key=lambda g: (g[0], -g[1]))
    labels = [f"layer {li} {PLUS if s > 0 else MINUS}" for li, s in names]
    gidx = {g: np.array([i for i, x in enumerate(group_of) if x == g]) for g in names}
    return group_of, names, labels, gidx


def thresholds(table, keys):
    """Each neuron's threshold; an OR-pool neuron fires above 0."""
    return np.array([0 if table[k][4] is None else table[k][4] for k in keys], np.int64)


def spikes_from_membranes(M, th):
    """Every neuron's spikes, read off its recorded membranes. simpleSim, like
    the chip, fires on the membrane carried into a step (phase one) and
    reports the membrane after that step's input (phase two), so a neuron
    fires at step t exactly when its recorded membrane at t - 1 is above its
    threshold. Membranes start at 0."""
    F = np.zeros(M.shape, bool)
    if len(M):
        F[0] = th < 0
        F[1:] = M[:-1] > th[None, :]
    return F


def output_mismatches(keys, out_keys, F, O):
    """Output spikes the membranes imply that the record lacks, or the reverse.
    0 for simpleSim; on the FPGA, a spike its own membrane called for that
    never reached the host, or one that did with no membrane behind it."""
    col = {k: i for i, k in enumerate(keys)}
    idx = [col[k] for k in out_keys if k in col]
    S = min(len(F), len(O))
    return int((F[:S][:, idx] != np.asarray(O[:S], bool)[:, :len(idx)]).sum())


def analyze(ref, hw, n_rows=20):
    """Everything the report shows, as plain data."""
    hidx = {k: i for i, k in enumerate(hw["keys"])}
    keys = [k for k in ref["keys"] if k in hidx]
    ridx = {k: i for i, k in enumerate(ref["keys"])}
    ri = np.array([ridx[k] for k in keys], dtype=int)
    Mr = ref["membrane"][:, ri]
    Mh = hw["membrane"][:, [hidx[k] for k in keys]]
    S = min(len(Mr), len(Mh))
    Mr, Mh = Mr[:S], Mh[:S]
    table = ref["table"]
    out_keys = ref["out_keys"]
    ho = {k: i for i, k in enumerate(hw["out_keys"])}
    Or = ref["out_spikes"][:S]
    Oh = hw["out_spikes"][:S][:, [ho[k] for k in out_keys]]
    D = Mr != Mh
    Ds = (Or != Oh).any(axis=1)
    bad = np.where(D.any(axis=1) | Ds)[0]
    first = int(bad[0]) if len(bad) else None

    group_of, group_names, labels, gidx = layer_groups(table, keys)
    share = np.array([[float(D[t, gidx[g]].mean()) for t in range(S)] for g in group_names])
    # spikes of every neuron on both targets, from the membranes
    th = thresholds(table, keys)
    Fr, Fh = spikes_from_membranes(Mr, th), spikes_from_membranes(Mh, th)
    missed, extra = Fr & ~Fh, Fh & ~Fr
    spikes = {"sim": [int(Fr[:, gidx[g]].sum()) for g in group_names],
              "fpga": [int(Fh[:, gidx[g]].sum()) for g in group_names],
              "missed": [int(missed[:, gidx[g]].sum()) for g in group_names],
              "extra": [int(extra[:, gidx[g]].sum()) for g in group_names],
              "missed_step": missed.sum(axis=1).astype(int).tolist(),
              "extra_step": extra.sum(axis=1).astype(int).tolist(),
              "out_check_ref": output_mismatches(keys, out_keys, Fr, Or),
              "out_check_hw": output_mismatches(keys, out_keys, Fh, Oh)}

    cls = output_classes(out_keys, table)
    n_cls = len(cls)
    tot_r = [int(sum(Or[:, j].sum() * s for j, s in cls[c])) for c in sorted(cls)]
    tot_h = [int(sum(Oh[:, j].sum() * s for j, s in cls[c])) for c in sorted(cls)]

    out = {"steps": S, "first": first, "labels": labels, "share": share.tolist(),
           "n_neurons": len(keys), "class_ref": tot_r, "class_hw": tot_h,
           "pred_ref": int(np.argmax(tot_r)) if n_cls else None,
           "pred_hw": int(np.argmax(tot_h)) if n_cls else None,
           "label": ref.get("label"), "clip": ref.get("clip"),
           "differ_at_end": int(D[-1].sum()) if S else 0,
           "rows": [], "scatter": [], "traces": [], "origin": None, "spikes": spikes}
    if first is None:
        return out

    at = np.where(D[first])[0]
    if len(at) == 0:                     # only an output spike differs
        out["origin"] = "output spikes"
        return out
    layer_at = [group_of[i][0] for i in at]
    origin_layer = min(layer_at)
    origin = [i for i in at if group_of[i][0] == origin_layer]
    origin.sort(key=lambda i: -abs(int(Mh[first, i]) - int(Mr[first, i])))
    out["origin"] = f"layer {origin_layer}"
    neg = match = 0
    for i in origin:
        li, sign, ch, pos, theta, shift, shp = table[keys[i]]
        v_prev = int(Mr[first - 1, i]) if first > 0 else 0
        r, h = int(Mr[first, i]), int(Mh[first, i])
        why = classify(v_prev, r, h, shift, theta)
        neg += v_prev < 0 or r < 0
        match += why == "negative leak (logical shift)"
        if len(out["rows"]) < n_rows:
            where = (f"ch {ch}, y {pos // shp[2]}, x {pos % shp[2]}" if len(shp) == 3
                     else f"unit {ch}")
            out["rows"].append({"key": keys[i], "layer": li, "sign": sign, "where": where,
                                "theta": theta, "shift": shift, "before": v_prev,
                                "sim": r, "fpga": h, "why": why})
    out["origin_count"] = len(origin)
    out["origin_negative_before"] = neg
    out["origin_logical_shift"] = match
    # error against the previous membrane, every neuron of the origin layer
    lay = [i for i, g in enumerate(group_of) if g[0] == origin_layer]
    step_ = max(1, len(lay) // 1500)
    for i in lay[::step_]:
        v_prev = int(Mr[first - 1, i]) if first > 0 else 0
        out["scatter"].append([v_prev, int(Mh[first, i]) - int(Mr[first, i])])
    # membrane over time: the first origin neurons, and the output neuron
    # whose spike total changed most
    for i in origin[:3]:
        out["traces"].append({"key": keys[i], "theta": table[keys[i]][4],
                              "sim": Mr[:, i].tolist(), "fpga": Mh[:, i].tolist()})
    if out_keys:
        j = int(np.argmax(np.abs(Oh.sum(0).astype(int) - Or.sum(0).astype(int))))
        k = out_keys[j]
        if k in hidx:
            i = {kk: n for n, kk in enumerate(keys)}[k]
            out["traces"].append({"key": k + " (output)", "theta": table[k][4],
                                  "sim": Mr[:, i].tolist(), "fpga": Mh[:, i].tolist()})
    return out


def cmd_compare(a):
    ref, hw = load_record(a.ref), load_record(a.hw)
    res = analyze(ref, hw)
    print(text_summary(res, ref, hw))
    write(a.out, report_compare(res, ref, hw))
    return 0


def steps_to_wrap(v, s, limit=200):
    """Steps of logical-shift leaking, with no input, until a negative
    membrane wraps past -2**31 and turns positive (and so can fire)."""
    for k in range(1, limit + 1):
        v = leak_logical(v, s)
        if v > 0:
            return k
    return None


def analyze_one(rec, n_rows=15):
    """One record, every neuron: spikes, activity, headroom and the decision by
    layer and step, and which neurons have a negative membrane, which is where
    the L6m negative-leak bug acts."""
    M, table, keys = np.asarray(rec["membrane"], np.int64), rec["table"], rec["keys"]
    S = len(M)
    group_of, names, labels, gidx = layer_groups(table, keys)
    th = thresholds(table, keys)
    F = spikes_from_membranes(M, th)
    per_step = F.sum(axis=1).astype(int)
    layers = []
    for g, lab in zip(names, labels):
        ix = gidx[g]
        Fg, Mg, tg = F[:, ix], M[:, ix], th[ix]
        counts, per_neuron = Fg.sum(axis=1), Fg.sum(axis=0)
        lif = tg > 0
        near = np.abs(Mg[:, lif] - tg[lif]) <= 0.1 * tg[lif]
        peak_v = int(np.abs(Mg).max()) if Mg.size else 0
        th_max = int(tg.max()) if lif.any() else None
        layers.append({
            "label": lab, "neurons": int(len(ix)), "spikes": int(counts.sum()),
            "rate": float(Fg.mean()) if Fg.size else 0.0,
            "silent": float((per_neuron == 0).mean()) if len(ix) else 0.0,
            "busy": float((per_neuron >= 0.9 * S).mean()) if len(ix) else 0.0,
            "mean_step": float(counts.mean()) if S else 0.0,
            "peak_step": int(counts.max()) if S else 0, "peak_at": int(counts.argmax()) if S else 0,
            "near": float(near.mean()) if near.size else None,
            "peak_v": peak_v, "peak_v_bits": peak_v.bit_length() + 1,
            "th_max": th_max, "th_bits": None if th_max is None else th_max.bit_length() + 1})
    events = None
    if rec.get("fan_out") is not None:
        fo = np.asarray(rec["fan_out"], np.int64)
        neuron_ev = int((F.astype(np.int64) @ fo).sum())
        bias_ev = int(rec.get("bias_fan_out", 0)) * S
        axon_ev = int(rec.get("axon_events", 0))
        events = {"neurons": neuron_ev, "bias": bias_ev, "axons": axon_ev,
                  "total": neuron_ev + bias_ev + axon_ev}

    # the decision: running output totals per class
    cls = output_classes(rec["out_keys"], table)
    O = np.asarray(rec["out_spikes"], np.int64)
    if cls:
        per_cls = np.stack([sum(O[:, j] * s for j, s in cls[c]) for c in sorted(cls)], axis=1)
    else:
        per_cls = np.zeros((S, 0), np.int64)
    cum = np.cumsum(per_cls, axis=0)
    tot = [int(v) for v in cum[-1]] if S and cls else []
    pred = int(np.argmax(tot)) if tot else None
    leads = []
    for row in cum:
        top = np.flatnonzero(row == row.max()) if row.size else []
        leads.append(int(top[0]) if len(top) == 1 else -1)
    lock = None
    if pred is not None and leads[-1] == pred:
        lock = S - 1
        while lock > 0 and leads[lock - 1] == pred:
            lock -= 1
    label = rec.get("label")
    margin = None
    if label is not None and 0 <= label < cum.shape[1] and cum.shape[1] > 1:
        margin = (cum[:, label] - np.delete(cum, label, axis=1).max(axis=1)).astype(int).tolist()

    # negative membranes, where the L6m leak bug acts
    leaks = np.array([bool(table[k][5]) for k in keys])
    neg = (M < 0) & leaks[None, :]
    share = [[float(neg[t, gidx[g]].mean()) for t in range(S)] for g in names]
    ever = [float(neg[:, gidx[g]].any(axis=0).mean()) for g in names]
    hit = np.where(neg.any(axis=0))[0]
    first_t = neg.argmax(axis=0)
    order = sorted(hit, key=lambda i: (int(first_t[i]), group_of[i][0], int(M[first_t[i], i])))
    rows = []
    for i in order[:n_rows]:
        li, sign, ch, pos, theta, shift, shp = table[keys[i]]
        t = int(first_t[i])
        v = int(M[t, i])
        where = (f"ch {ch}, y {pos // shp[2]}, x {pos % shp[2]}" if len(shp) == 3
                 else f"unit {ch}")
        rows.append({"key": keys[i], "where": where, "step": t, "membrane": v, "shift": shift,
                     "theta": theta, "sim_next": leak_arith(v, shift),
                     "bug_next": leak_logical(v, shift), "wrap": steps_to_wrap(v, shift)})
    return {"steps": S, "n_neurons": len(keys), "labels": labels, "share": share,
            "ever": ever, "n_hit": int(len(hit)), "hit_keys": [keys[i] for i in hit],
            "share_steps": float(neg.mean()) if S else 0.0,
            "first_step": int(first_t[hit].min()) if len(hit) else None, "rows": rows,
            "per_step": per_step.tolist(), "spikes": int(F.sum()),
            "burst": int(per_step.max()) if S else 0,
            "burst_at": int(per_step.argmax()) if S else 0, "layers": layers, "events": events,
            "out_check": output_mismatches(keys, rec["out_keys"], F, rec["out_spikes"]),
            "class": tot, "pred": pred, "lock": lock, "margin": margin,
            "label": label, "clip": rec.get("clip"), "split": rec.get("split"),
            "target": rec.get("target"), "word_top_bits": rec.get("word_top_bits")}


def top_bits_note(tops):
    """What the top four bits of the FPGA's state words say about them."""
    total = sum(tops.values()) or 1
    seen = ", ".join(f"{k}: {100 * n / total:.1f}%" for k, n in tops.items())
    if set(tops) <= {"0"}:
        meaning = "the membrane sits in bits 31:0 and the top bits stay clear"
    elif set(tops) <= {"0", "F"}:
        meaning = "the word is sign-extended to 36 bits"
    else:
        meaning = "those bits hold something other than the membrane"
    return (f"FPGA state words, top four bits (35:32): {seen}. So {meaning}; membranes "
            f"here are bits 31:0, sign-extended, as hs_api's read_membrane reads them.")


def report_show(res):
    name = res["target"]
    color = "--fpga" if name == "CRI" else "--sim"
    sub = (f"{name} record of {res['split']} clip {res['clip']} (label {res['label']}), "
           f"{res['steps']} steps, {res['n_neurons']:,} neurons. Every spike here is read off "
           f"the membranes: a neuron fires at a step when its membrane the step before is "
           f"above threshold.")
    pct = 100 * res["n_hit"] / max(1, res["n_neurons"])
    ev = res["events"]
    b = ['<div class="tiles">',
         tile("prediction / label", f"{res['pred']} / {res['label']}"),
         tile("prediction settles", "never" if res["lock"] is None else f"step {res['lock']}"),
         tile("spikes in the clip", f"{res['spikes']:,}"),
         tile("largest burst", f"{res['burst']:,} at step {res['burst_at']}"),
         tile("synaptic events", "not recorded" if ev is None else compact(ev["total"])),
         tile("neurons that go negative", f"{res['n_hit']:,} ({pct:.0f}%)"), "</div>"]
    if res["out_check"] == 0:
        b.append('<p class="note">Check: the output spikes the membranes imply match the '
                 'recorded ones at every step.</p>')
    else:
        b.append(f'<p class="note">Check: {res["out_check"]:,} output spikes differ from what '
                 f'the membranes imply. On simpleSim that means this reading is wrong; on the '
                 f'FPGA, that spikes were lost or added on the way to the host.</p>')
    if res.get("word_top_bits"):
        b.append(f'<p class="note">{esc(top_bits_note(res["word_top_bits"]))}</p>')
    L = res["layers"]

    b.append("<h2>Spikes per step</h2><p class=\"sub\">All layer neurons together. The chip "
             "moves a step's spikes through its spike FIFOs at once, so large bursts are where "
             "dropped spikes would show; the micro report's burst tests find the size where "
             "the FPGA starts dropping them.</p>")
    b.append('<div class="card">' + line_chart([(name, color, res["per_step"])])
             + "<details><summary>By layer</summary>" + table_html(
                 ["layer", "neurons", "spikes per step", "most in one step", "at step"],
                 [(esc(r["label"]), f"{r['neurons']:,}", f"{r['mean_step']:,.0f}",
                   f"{r['peak_step']:,}", r["peak_at"]) for r in L], num=(1, 2, 3, 4))
             + "</details></div>")

    b.append("<h2>Activity by layer</h2><p class=\"sub\">Share of each layer's neurons that "
             "fire at a step, averaged over the clip.</p>")
    b.append('<div class="card">' + grouped_bars(
        [r["label"] for r in L], [(name, color, [round(100 * r["rate"], 1) for r in L])],
        x_label="layer, percent firing per step") + table_html(
        ["layer", "neurons", "spikes in the clip", "firing per step", "never fire",
         "fire on 90% of steps or more"],
        [(esc(r["label"]), f"{r['neurons']:,}", f"{r['spikes']:,}", f"{100 * r['rate']:.1f}%",
          f"{100 * r['silent']:.0f}%", f"{100 * r['busy']:.0f}%") for r in L],
        num=(1, 2, 3, 4, 5)) + "</div>")
    if ev is not None:
        b.append(f'<p class="note">Synaptic events in the clip: {ev["total"]:,}, each one a '
                 f'synapse the chip applies. {ev["neurons"]:,} come from layer spikes, '
                 f'{ev["axons"]:,} from input axons, {ev["bias"]:,} from bias neurons.</p>')
    else:
        b.append('<p class="note">This record has no fan-out, so synaptic events are not '
                 'counted. Records made with this version of record keep it.</p>')

    b.append("<h2>Headroom</h2><p class=\"sub\">How close membranes run to threshold, and how "
             "much of the 32-bit membrane they use. A neuron-step within 10% of "
             "threshold flips with an error that small.</p>")
    b.append('<div class="card">' + table_html(
        ["layer", "largest threshold", "largest membrane", "neuron-steps within 10% of threshold"],
        [(esc(r["label"]), "" if r["th_max"] is None else f"{r['th_max']:,} ({r['th_bits']} bits)",
          f"{r['peak_v']:,} ({r['peak_v_bits']} bits)",
          "" if r["near"] is None else f"{100 * r['near']:.1f}%") for r in L],
        num=(1, 2, 3)) + "</div>")

    if res["margin"] is not None:
        settle = ("" if res["lock"] is None else
                  f" The prediction stops changing at step {res['lock']} of {res['steps']}.")
        b.append("<h2>Decision over time</h2><p class=\"sub\">The label's running output total "
                 "minus the best other class's. Above zero, stopping at that step would answer "
                 f"correctly.{settle}</p>")
        b.append('<div class="card">' + line_chart(
            [(name, color, res["margin"])], hlines=[(0, "tie")]) + "</div>")

    b.append("<h2>Negative membranes, by layer and step</h2><p class=\"sub\">Share of each "
             "layer's leaking neurons whose membrane is negative at that step. Each one is "
             "leaked with a logical shift on the L6m FPGA, the known bug. "
             f"{res['n_hit']:,} neurons ({pct:.0f}%) go negative at some step"
             + ("" if res["first_step"] is None else f", first at step {res['first_step']}")
             + f"; {100 * res['share_steps']:.0f}% of neuron-steps are negative.</p>")
    b.append('<div class="card">' + heat_legend()
             + heatmap(res["labels"], res["share"], what="negative") + "</div>")
    b.append("<h2>Neurons that go negative at least once</h2><p class=\"sub\">Share of each "
             "layer's neurons, over the whole clip.</p>")
    b.append('<div class="card">' + grouped_bars(
        res["labels"], [(name, color, [round(100 * e) for e in res["ever"]])],
        x_label="layer, percent of its neurons") + "</div>")
    if res["rows"]:
        b.append("<h2>What the bug would do to the first ones</h2><p class=\"sub\">The "
                 "membrane the simulator holds, what its arithmetic leak leaves next, what a "
                 "logical shift leaves instead, and how many steps of that it takes to wrap "
                 "past \u22122<sup>31</sup> to a huge positive value, which fires. Until "
                 "then the neuron sits far below threshold and cannot fire.</p>")
        b.append('<div class="card">' + table_html(
            ["neuron", "where", "step", "membrane", "shift", "simulator leaks to",
             "logical shift leaves", "steps until it wraps"],
            [(esc(r["key"]), esc(r["where"]), r["step"], f"{r['membrane']:,}", r["shift"],
              f"{r['sim_next']:,}", f"{r['bug_next']:,}",
              "" if r["wrap"] is None else r["wrap"]) for r in res["rows"]],
            num=(2, 3, 4, 5, 6, 7)) + "</div>")
    if res["class"]:
        cats = [str(i) for i in range(len(res["class"]))]
        b.append("<h2>Output spikes per class</h2><p class=\"sub\">Positive minus negative "
                 "output spikes over the clip; the largest is the prediction.</p>")
        b.append('<div class="card">' + grouped_bars(
            cats, [(name, color, res["class"])], x_label="class") + "</div>")
    return page("One clip, every neuron", sub, "".join(b))


def cmd_show(a):
    res = analyze_one(load_record(a.record))
    pct = 100 * res["n_hit"] / max(1, res["n_neurons"])
    settles = "never settles" if res["lock"] is None else f"settles at step {res['lock']}"
    print(f"prediction {res['pred']} (label {res['label']}), {settles}")
    print(f"{res['spikes']:,} spikes; largest burst {res['burst']:,} at step {res['burst_at']}")
    if res["events"] is not None:
        print(f"{res['events']['total']:,} synaptic events")
    print(f"{res['n_hit']:,} of {res['n_neurons']:,} neurons ({pct:.0f}%) have a negative "
          f"membrane at some step; first at step {res['first_step']}; "
          f"{100 * res['share_steps']:.0f}% of neuron-steps")
    if res["out_check"]:
        print(f"check: {res['out_check']} output spikes differ from what the membranes imply")
    if res.get("word_top_bits"):
        print(top_bits_note(res["word_top_bits"]))
    write(a.out, report_show(res))
    return 0


def text_summary(res, ref, hw):
    lines = [f"{ref['target']} vs {hw['target']}, {ref.get('split')} clip {res['clip']} "
             f"(label {res['label']}), {res['steps']} steps, {res['n_neurons']:,} neurons"]
    if res["first"] is None:
        lines.append("identical: every membrane and every output spike, at every step")
        return "\n".join(lines)
    lines.append(f"first difference at step {res['first']}, starting in {res['origin']}")
    if res.get("origin_count"):
        lines.append(f"  {res['origin_count']} neurons there differ; "
                     f"{res['origin_negative_before']} have a negative membrane in the "
                     f"simulator; {res['origin_logical_shift']} match a logical-shift leak")
        for r in res["rows"][:10]:
            lines.append(f"  {r['key']:14s} {r['where']:22s} shift {r['shift']}: "
                         f"before {r['before']:>12,}  sim {r['sim']:>12,}  "
                         f"fpga {r['fpga']:>14,}  {r['why']}")
    sp = res["spikes"]
    lines.append(f"spikes (read off the membranes): sim {sum(sp['sim']):,}, "
                 f"fpga {sum(sp['fpga']):,}; the FPGA missed {sum(sp['missed']):,} and "
                 f"added {sum(sp['extra']):,}")
    if sp["out_check_hw"]:
        lines.append(f"  {sp['out_check_hw']} FPGA output spikes disagree with its own membranes")
    if hw.get("word_top_bits"):
        lines.append(top_bits_note(hw["word_top_bits"]))
    lines.append(f"output totals per class: sim {res['class_ref']}, fpga {res['class_hw']}")
    lines.append(f"prediction: sim {res['pred_ref']}, fpga {res['pred_hw']}, "
                 f"label {res['label']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# micro: one question per network
# ---------------------------------------------------------------------------

NEG_LEAK_DRIVE = 20000
NEG_LEAK_THETA = 2 ** 30


def leak_models(w, s, n, logical, theta=NEG_LEAK_THETA, bits=MP_BITS):
    """A membrane driven by w at step 0 and then left alone, leaked with an
    arithmetic or a logical shift, firing and resetting to 0 above theta (a
    wrapped negative membrane turns huge and positive, and fires):
    {"before": simpleSim's order, fire on the membrane carried in, reset or
    leak, then add the input; "after": add the input, then fire or leak}."""
    def f(v, s):
        return signed_mp(leak_logical(v, s, bits)) if logical else leak_arith(v, s)
    out = {}
    for order in ("after", "before"):
        v, trace = 0, []
        for t in range(n):
            drive = w if t == 0 else 0
            if order == "before":
                v = f(0 if v > theta else v, s) + drive
            else:
                v += drive
                v = 0 if v > theta else f(v, s)
            trace.append(v)
        out[order] = trace
    return out


def micro_tests():
    """(name, question, axons, connections, outputs, schedule, cases). The
    schedule is the axon keys driven at each step; cases is a list of
    (label, function(rec) -> comparable summary)."""
    from snnsearch.hs_export import hs_lif_neuron, hs_bias_neuron
    # One model object per (threshold, shift), shared by every neuron that
    # uses it, as hs_export.unroll_to_hs_api does. hs_api tells models apart
    # by object, pads each model's neurons to a multiple of 32, and hs_bridge
    # sends the padded neuron count in 17 bits: a model per neuron turned
    # 4096 neurons into 131,072 and the FPGA load failed.
    shared = {}

    def lif(theta, shift):
        key = (int(theta), int(shift))
        if key not in shared:
            shared[key] = hs_lif_neuron(*key)
        return shared[key]

    NO_LEAK = 63               # shift 63: no leak for any positive membrane
    BIG = NEG_LEAK_THETA
    tests = []

    # 1. negative membranes leaking, shifts 1 to 7, with positive controls
    axons = {"A0": []}
    conns = {}
    for s in range(1, 8):
        for tag, w in (("neg", -NEG_LEAK_DRIVE), ("pos", NEG_LEAK_DRIVE)):
            k = f"{tag}{s}"
            axons["A0"].append((k, w))
            conns[k] = ([], lif(BIG, s))
    sched = [["A0"]] + [[] for _ in range(11)]
    cases = []
    for s in range(1, 8):
        for tag in ("neg", "pos"):
            cases.append((f"{'negative' if tag == 'neg' else 'positive'} membrane, shift {s}",
                          ("membrane", f"{tag}{s}")))
    # one declared output, which never fires (hs_api may want at least one)
    tests.append(("negative_leak", "Does a negative membrane leak like a positive one?",
                  axons, conns, ["neg1"], sched, cases))

    # 2 and 3. many spikes in one step: straight to the output, and one hop on
    sizes = [16, 64, 256, 1024, 2048, 4096]
    for name, hop in (("output_burst", False), ("internal_burst", True)):
        n = max(sizes)
        axons = {f"B{i}": [(f"b{i}", 1)] for i in range(n)}
        conns = {f"b{i}": ([(f"c{i}", 1)] if hop else [], lif(0, NO_LEAK))
                 for i in range(n)}
        if hop:
            conns.update({f"c{i}": ([], lif(0, NO_LEAK)) for i in range(n)})
        outs = [f"c{i}" if hop else f"b{i}" for i in range(n)]
        sched, cases, window = [], [], 4
        for N in sizes:
            start = len(sched)
            sched.append([f"B{i}" for i in range(N)])
            sched += [[] for _ in range(window - 1)]
            cases.append((f"{N} spikes in one step", ("spikes", start, start + window)))
        q = ("Do all spikes reach the output when many fire at once?" if not hop else
             "Do all spikes cross one neuron-to-neuron hop when many fire at once?")
        tests.append((name, q, axons, conns, outs, sched, cases))

    # 4. fan-out: one axon to K neurons, and one neuron to K neurons
    ks = [64, 512, 2048, 4095]
    axons, conns, outs, sched, cases = {}, {}, [], [], []
    for K in ks:
        axons[f"F{K}"] = [(f"f{K}_{j}", 1) for j in range(K)]
        for j in range(K):
            conns[f"f{K}_{j}"] = ([], lif(0, NO_LEAK))
            outs.append(f"f{K}_{j}")
        start = len(sched)
        sched += [[f"F{K}"], [], []]
        cases.append((f"axon to {K} neurons", ("spikes", start, start + 3, f"f{K}_")))
    axons["G"] = [("g", 1)]
    conns["g"] = ([(f"h{j}", 1) for j in range(4095)], lif(0, NO_LEAK))
    conns.update({f"h{j}": ([], lif(0, NO_LEAK)) for j in range(4095)})
    outs += [f"h{j}" for j in range(4095)]
    start = len(sched)
    sched += [["G"], [], [], []]
    cases.append(("neuron to 4095 neurons", ("spikes", start, start + 4, "h")))
    tests.append(("fan_out", "Does a spike reach every target of a large fan-out?",
                  axons, conns, outs, sched, cases))

    # 5. thresholds past 16 and 17 bits
    thetas = [32767, 2 ** 17 + 5, 2 ** 20 + 1]
    axons = {f"T{i}": [(f"t{i}", 32767)] for i in range(len(thetas))}
    conns = {f"t{i}": ([], lif(th, NO_LEAK)) for i, th in enumerate(thetas)}
    sched = [[f"T{i}" for i in range(len(thetas))] for _ in range(40)]
    cases = [(f"threshold {th:,}", ("first_spike", f"t{i}")) for i, th in enumerate(thetas)]
    cases += [(f"threshold {th:,}: membrane", ("membrane", f"t{i}"))
              for i, th in enumerate(thetas)]
    tests.append(("thresholds", "Do thresholds wider than 16 bits fire at the right step?",
                  axons, conns, [f"t{i}" for i in range(len(thetas))], sched, cases))

    # 6. an always-on bias neuron driving a counter neuron
    axons = {"Z": []}
    conns = {"bias": ([("u", 1)], hs_bias_neuron()), "u": ([], lif(3, NO_LEAK))}
    sched = [[] for _ in range(20)]
    tests.append(("bias_neuron", "Does an always-on bias neuron fire every step?",
                  axons, conns, ["u"], sched,
                  [("counter neuron spike steps", ("spike_steps", "u")),
                   ("counter neuron membrane", ("membrane", "u"))]))
    return tests


def _summary(rec, spec):
    keys, outs = rec["keys"], rec["out_keys"]
    kind = spec[0]
    if kind == "membrane":
        return rec["membrane"][:, keys.index(spec[1])].tolist()
    if kind == "spikes":
        lo, hi = spec[1], spec[2]
        cols = [j for j, k in enumerate(outs) if len(spec) < 4 or k.startswith(spec[3])]
        return int(rec["out_spikes"][lo:hi][:, cols].sum())
    j = outs.index(spec[1])
    steps = np.where(rec["out_spikes"][:, j])[0].tolist()
    if kind == "first_spike":
        return steps[0] if steps else None
    return steps                          # spike_steps


def run_micro(targets=("simpleSim", "CRI"), core=0):
    results = []
    for name, question, axons, conns, outs, sched, cases in micro_tests():
        keys = list(conns)
        recs = {}
        for target in targets:
            t0 = time.time()
            net, clear = build(axons, conns, outs, target, core)
            M, O, _, _ = run(net, clear, sched, keys, outs)
            recs[target] = {"keys": keys, "out_keys": outs, "membrane": M, "out_spikes": O}
            print(f"{name} on {target}: {len(sched)} steps ({time.time() - t0:.0f}s)",
                  flush=True)
        ref, hw = (recs[t] for t in targets)
        rows = []
        for label, spec in cases:
            r, h = _summary(ref, spec), _summary(hw, spec)
            row = {"case": label, "ref": r, "hw": h, "pass": r == h}
            if spec[0] == "membrane" and name == "negative_leak" and r:
                s = int(label.rsplit(" ", 1)[1])
                w = -NEG_LEAK_DRIVE if label.startswith("negative") else NEG_LEAK_DRIVE
                arith = leak_models(w, s, len(r), False)
                order = next((o for o in arith if arith[o] == r), None)
                row["sim_order"] = order
                row["matches_logical"] = False
                for bits in (MP_BITS, 36):
                    logical = leak_models(w, s, len(r), True, bits=bits)
                    if order and h == logical[order] and r != logical[order]:
                        row["matches_logical"], row["logical_bits"] = True, bits
                        break
            rows.append(row)
        failing = [x["case"] for x in rows if not x["pass"]]
        results.append({"name": name, "question": question, "rows": rows,
                        "pass": not failing, "first_fail": failing[0] if failing else None,
                        "targets": list(targets)})
    return results


def cmd_micro(a):
    results = run_micro(tuple(a.targets), a.core)
    for t in results:
        print(f"{'PASS' if t['pass'] else 'FAIL'}  {t['name']:16s} "
              + ("" if t["pass"] else f"first failing case: {t['first_fail']}"))
    with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
        json.dump(results, fh, indent=1, default=int)
    write(a.out, report_micro(results))
    return 0


# ---------------------------------------------------------------------------
# the HTML reports: inline SVG, light and dark, no outside files
# ---------------------------------------------------------------------------

CSS = """
.viz-root { color-scheme: light;
  --surface-1:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --sim:#2a78d6; --fpga:#eb6834; --zero:#f0efec;
  --seq-1:#b7d3f6; --seq-2:#86b6ef; --seq-3:#5598e7; --seq-4:#2a78d6; --seq-5:#1c5cab; --seq-6:#0d366b;
  --good:#0ca30c; --critical:#d03b3b; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --sim:#3987e5; --fpga:#d95926; --zero:#2c2c2a;
  --seq-1:#184f95; --seq-2:#1c5cab; --seq-3:#256abf; --seq-4:#3987e5; --seq-5:#6da7ec; --seq-6:#b7d3f6; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark;
  --surface-1:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --sim:#3987e5; --fpga:#d95926; --zero:#2c2c2a;
  --seq-1:#184f95; --seq-2:#1c5cab; --seq-3:#256abf; --seq-4:#3987e5; --seq-5:#6da7ec; --seq-6:#b7d3f6; }
html, body { margin:0; }
body { background:var(--page); }
.viz-root { background:var(--page); color:var(--ink); min-height:100vh;
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:980px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; font-weight:600; }
h2 { font-size:16px; margin:32px 0 4px; font-weight:600; }
p.sub, p.note { color:var(--ink-2); margin:0 0 12px; }
.card { background:var(--surface-1); border:1px solid var(--ring); border-radius:10px;
  padding:16px; margin-top:8px; overflow-x:auto; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:8px; margin-top:16px; }
.tile { background:var(--surface-1); border:1px solid var(--ring); border-radius:10px; padding:12px 14px; }
.tile .label { color:var(--ink-2); font-size:12px; }
.tile .value { font-size:22px; font-weight:600; }
.legend { display:flex; gap:16px; flex-wrap:wrap; color:var(--ink-2); font-size:12px; margin:4px 0 8px; }
.key { display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }
.sw { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:middle; margin-right:4px; }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th { color:var(--ink-2); font-weight:600; }
td.num { text-align:right; }
details { margin-top:8px; color:var(--ink-2); }
summary { cursor:pointer; }
.status { font-weight:600; }
.status .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }
svg text { fill:var(--muted); font-size:11px; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
svg .title { fill:var(--ink-2); font-size:12px; }
.multiples { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:12px; }
.multiples.small { grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); }
.card svg { max-width:760px; display:block; }
.card svg.heat { max-width:none; }
@media (max-width:420px) { .multiples { grid-template-columns:1fr; } }
"""


def esc(s):
    return html.escape(str(s))


def compact(v):
    v = float(v)
    a = abs(v)
    for unit, d in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if a >= d:
            x = v / d
            return f"{x:.0f}{unit}" if abs(x) >= 10 else f"{x:.1f}{unit}"
    return f"{v:.0f}" if a >= 1 or v == 0 else f"{v:.2g}"


def ticks(lo, hi, n=5):
    """Round tick values from at or below lo to at or above hi."""
    if hi - lo < max(1.0, abs(hi), abs(lo)) * 1e-9:    # flat, or below float resolution
        lo, hi = lo - 1, hi + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step_ = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    v = math.floor(lo / step_) * step_
    out = [v]
    for _ in range(4 * n + 4):
        if v >= hi - 1e-9 * step_:
            break
        v += step_
        out.append(v)
    return out


def legend(items):
    return ('<div class="legend">' + "".join(
        f'<span><span class="key" style="background:var({c})"></span>{esc(n)}</span>'
        for n, c in items) + "</div>")


def line_chart(series, width=600, height=220, x_label="step", hlines=(), title=""):
    """series: [(name, css var, ys)] sharing x = 0..len-1."""
    ml, mr, mt, mb = 56, 16, 26 if title else 8, 30
    pw, ph = width - ml - mr, height - mt - mb
    allv = [y for _, _, ys in series for y in ys] + [h for h, _ in hlines]
    lo, hi = min(allv + [0]), max(allv + [0])
    yt = ticks(lo, hi)
    lo, hi = min(yt), max(yt)
    n = max(len(ys) for _, _, ys in series)
    X = lambda i: ml + (pw * i / max(1, n - 1))                     # noqa: E731
    Y = lambda v: mt + ph - (ph * (v - lo) / (hi - lo))             # noqa: E731
    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
         f'aria-label="{esc(title or "line chart")}">']
    if title:
        p.append(f'<text class="title" x="{ml}" y="12">{esc(title)}</text>')
    for v in yt:
        p.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{compact(v)}</text>')
    for v in ticks(0, n - 1, 6):
        if 0 <= v <= n - 1:
            p.append(f'<text x="{X(v):.1f}" y="{mt + ph + 16}" text-anchor="middle">{int(v)}</text>')
    p.append(f'<text x="{ml + pw / 2:.1f}" y="{height - 2}" text-anchor="middle">{esc(x_label)}</text>')
    for hv, name in hlines:
        p.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(hv):.1f}" y2="{Y(hv):.1f}" '
                 f'stroke="var(--axis)" stroke-width="1"/>'
                 f'<text x="{ml + pw}" y="{Y(hv) - 4:.1f}" text-anchor="end">{esc(name)}</text>')
    for name, color, ys in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
        p.append(f'<polyline points="{pts}" fill="none" stroke="var({color})" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
    for i in range(n):          # one hover target per step, all series in its tooltip
        tip = f"step {i}: " + ", ".join(f"{nm} {ys[i]:,}" for nm, _, ys in series if i < len(ys))
        x0 = X(i) - pw / max(1, n - 1) / 2
        p.append(f'<rect x="{x0:.1f}" y="{mt}" width="{pw / max(1, n - 1):.1f}" height="{ph}" '
                 f'fill="transparent"><title>{esc(tip)}</title></rect>')
    p.append("</svg>")
    return "".join(p)


def heatmap(row_labels, values, cell=14, x_label="step", what="differ"):
    """values: rows x columns in 0..1, the share of neurons that differ (or,
    with what="negative", that have a negative membrane)."""
    rows, cols = len(values), max((len(r) for r in values), default=0)
    ml, mt, mb = 88, 4, 30
    width, height = ml + cols * cell + 8, mt + rows * cell + mb
    p = [f'<svg class="heat" viewBox="0 0 {width} {height}" width="{width}" role="img" '
         f'aria-label="share of neurons that {what}, by layer and step">']
    for r, lab in enumerate(row_labels):
        p.append(f'<text x="{ml - 6}" y="{mt + r * cell + cell - 2}" text-anchor="end">{esc(lab)}</text>')
        for c, v in enumerate(values[r]):
            fill = "var(--zero)" if v == 0 else f"var(--seq-{bin_of(v)})"
            p.append(f'<rect x="{ml + c * cell}" y="{mt + r * cell}" width="{cell - 1}" '
                     f'height="{cell - 1}" rx="2" fill="{fill}"><title>{esc(lab)}, step {c}: '
                     f'{100 * v:.1f}% {what}</title></rect>')
    for c in ticks(0, cols - 1, 8):
        if 0 <= c <= cols - 1:
            p.append(f'<text x="{ml + c * cell + cell / 2:.1f}" y="{mt + rows * cell + 14}" '
                     f'text-anchor="middle">{int(c)}</text>')
    p.append(f'<text x="{ml + cols * cell / 2:.1f}" y="{height - 2}" text-anchor="middle">{x_label}</text></svg>')
    return "".join(p)


BINS = [0.01, 0.05, 0.2, 0.5, 0.9]


def bin_of(v):
    return 1 + sum(v > b for b in BINS)


def heat_legend():
    labels = ["0", "≤ 1%", "1–5%", "5–20%", "20–50%", "50–90%", "> 90%"]
    sw = ['<span><span class="sw" style="background:var(--zero)"></span>0</span>']
    sw += [f'<span><span class="sw" style="background:var(--seq-{i})"></span>{labels[i]}</span>'
           for i in range(1, 7)]
    return '<div class="legend">' + "".join(sw) + "</div>"


def bar_path(x, y, w, h, r=4):
    """A bar rounded at its data end (top), square at the baseline."""
    if h <= 0:
        return ""
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"L{x + w:.1f},{y + h:.1f} Z")


def grouped_bars(categories, series, width=600, height=220, x_label=""):
    """series: [(name, css var, values)]; values may be negative."""
    ml, mr, mt, mb = 56, 16, 8, 30 + (14 if x_label else 0)
    pw, ph = width - ml - mr, height - mt - mb
    allv = [v for _, _, vs in series for v in vs] + [0]
    yt = ticks(min(allv), max(allv))
    lo, hi = min(yt), max(yt)
    Y = lambda v: mt + ph - ph * (v - lo) / (hi - lo)                # noqa: E731
    band = pw / max(1, len(categories))
    bw = min(24, (band - 8) / len(series) - 2)
    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="grouped bars">']
    for v in yt:
        p.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{compact(v)}</text>')
    base = Y(0)
    for ci, cat in enumerate(categories):
        x0 = ml + ci * band + (band - len(series) * (bw + 2)) / 2
        for si, (name, color, vs) in enumerate(series):
            v = vs[ci]
            x = x0 + si * (bw + 2)
            top, h = (Y(v), base - Y(v)) if v >= 0 else (base, Y(v) - base)
            d = bar_path(x, top, bw, h) if v >= 0 else (
                f"M{x:.1f},{top:.1f} h{bw:.1f} v{max(0, h - 4):.1f} q0,4 -4,4 "
                f"h{-(bw - 8):.1f} q-4,0 -4,-4 Z" if h > 4 else "")
            if d:
                p.append(f'<path d="{d}" fill="var({color})"><title>{esc(cat)}: {esc(name)} '
                         f'{v:,}</title></path>')
        p.append(f'<text x="{ml + ci * band + band / 2:.1f}" y="{mt + ph + 16}" '
                 f'text-anchor="middle">{esc(cat)}</text>')
    p.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{base:.1f}" y2="{base:.1f}" stroke="var(--axis)"/>')
    if x_label:
        p.append(f'<text x="{ml + pw / 2:.1f}" y="{height - 2}" text-anchor="middle">{esc(x_label)}</text>')
    p.append("</svg>")
    return "".join(p)


def scatter(points, width=600, height=240, x_label="", y_label=""):
    ml, mr, mt, mb = 64, 16, 8, 34
    pw, ph = width - ml - mr, height - mt - mb
    xs, ys = [x for x, _ in points] + [0], [y for _, y in points] + [0]
    xt, yt = ticks(min(xs), max(xs)), ticks(min(ys), max(ys))
    X = lambda v: ml + pw * (v - min(xt)) / (max(xt) - min(xt))      # noqa: E731
    Y = lambda v: mt + ph - ph * (v - min(yt)) / (max(yt) - min(yt))  # noqa: E731
    p = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="scatter">']
    for v in yt:
        p.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                 f'stroke="var(--grid)"/><text x="{ml - 6}" y="{Y(v) + 4:.1f}" '
                 f'text-anchor="end">{compact(v)}</text>')
    for v in xt:
        p.append(f'<text x="{X(v):.1f}" y="{mt + ph + 16}" text-anchor="middle">{compact(v)}</text>')
    p.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{mt}" y2="{mt + ph}" stroke="var(--axis)"/>')
    for x, y in points:
        p.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="var(--fpga)" '
                 f'stroke="var(--surface-1)" stroke-width="2"><title>before {x:,}, '
                 f'FPGA minus sim {y:,}</title></circle>')
    p.append(f'<text x="{ml + pw / 2:.1f}" y="{height - 2}" text-anchor="middle">{esc(x_label)}</text>'
             f'<text x="12" y="{mt + ph / 2:.1f}" transform="rotate(-90 12 {mt + ph / 2:.1f})" '
             f'text-anchor="middle">{esc(y_label)}</text></svg>')
    return "".join(p)


def status(ok):
    return (f'<span class="status"><span class="dot" style="background:var(--good)"></span>'
            f'✓ pass</span>' if ok else
            f'<span class="status"><span class="dot" style="background:var(--critical)">'
            f'</span>✕ fail</span>')


def table_html(head, rows, num=()):
    out = ["<table><thead><tr>"] + [f"<th>{esc(h)}</th>" for h in head] + ["</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(
            f'<td class="num">{esc(v)}</td>' if i in num else f"<td>{v}</td>"
            for i, v in enumerate(r)) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def page(title, sub, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(title)}</title><style>{CSS}</style></head>"
            f'<body><div class="viz-root"><main><h1>{esc(title)}</h1>'
            f'<p class="sub">{esc(sub)}</p>{body}</main></div></body></html>')


def tile(label, value):
    return f'<div class="tile"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div></div>'


def report_compare(res, ref, hw):
    sim_name, fpga_name = ref["target"], hw["target"]
    sub = (f"{ref.get('split', '')} clip {res['clip']} (label {res['label']}), "
           f"{res['steps']} steps, {res['n_neurons']:,} neurons. {sim_name} is the reference.")
    if res["first"] is None:
        body = ('<div class="tiles">' + tile("first difference", "none") + "</div>"
                f'<p class="note">Every membrane and every output spike matches at every step.</p>')
        return page("FPGA vs simpleSim", sub, body)
    b = ['<div class="tiles">',
         tile("first difference", f"step {res['first']}"),
         tile("where it starts", res["origin"]),
         tile("neurons differing at the end", f"{res['differ_at_end']:,}"),
         tile("prediction (sim / FPGA / label)",
              f"{res['pred_ref']} / {res['pred_hw']} / {res['label']}"), "</div>"]
    if hw.get("word_top_bits"):
        b.append(f'<p class="note">{esc(top_bits_note(hw["word_top_bits"]))}</p>')
    if res.get("origin_count"):
        b.append(f'<p class="note">At step {res["first"]}, {res["origin_count"]} neurons of '
                 f'{res["origin"]} differ. {res["origin_negative_before"]} of them have a '
                 f'negative membrane in the simulator, and {res["origin_logical_shift"]} match '
                 f'what a logical (unsigned) shift would leave, the negative-leak bug.</p>')
    b.append("<h2>Where the membranes differ</h2><p class=\"sub\">Share of each layer's "
             "neurons whose FPGA membrane differs from the simulator's, at every step.</p>")
    b.append('<div class="card">' + heat_legend()
             + heatmap(res["labels"], res["share"]) + "</div>")
    sp = res["spikes"]
    only_sim, only_fpga = f"only in {sim_name} (missed)", f"only on {fpga_name} (extra)"
    b.append("<h2>Spikes the FPGA missed and added</h2><p class=\"sub\">Every neuron's spikes, "
             "read off the membranes on both targets. Missed spikes are neurons the FPGA kept "
             "below threshold, as the leak bug does to negative membranes; extra spikes are "
             "membranes that ran above it. A spike dropped on its way to other neurons shows "
             "up one layer later, as missed spikes there.</p>")
    b.append('<div class="card">' + legend([(only_sim, "--sim"), (only_fpga, "--fpga")])
             + grouped_bars(res["labels"], [(only_sim, "--sim", sp["missed"]),
                                            (only_fpga, "--fpga", sp["extra"])],
                            x_label="layer, spikes over the clip")
             + line_chart([(only_sim, "--sim", sp["missed_step"]),
                           (only_fpga, "--fpga", sp["extra_step"])])
             + "<details><summary>Table</summary>" + table_html(
                 ["layer", f"{sim_name} spikes", f"{fpga_name} spikes", "missed", "extra"],
                 [(esc(lab), f"{r:,}", f"{h:,}", f"{m:,}", f"{e:,}") for lab, r, h, m, e in
                  zip(res["labels"], sp["sim"], sp["fpga"], sp["missed"], sp["extra"])],
                 num=(1, 2, 3, 4)) + "</details></div>")
    if sp["out_check_hw"]:
        b.append(f'<p class="note">{sp["out_check_hw"]:,} of the FPGA\'s recorded output spikes '
                 f'disagree with what its own output membranes imply: spikes lost or added '
                 f'between the neuron and the host.</p>')
    else:
        b.append('<p class="note">The FPGA\'s recorded output spikes match what its own output '
                 'membranes imply, so its output neurons fire as their membranes say.</p>')
    if res["class_ref"]:
        cats = [str(i) for i in range(len(res["class_ref"]))]
        b.append("<h2>Output spikes per class</h2><p class=\"sub\">Positive minus negative "
                 "output spikes over the clip; the largest is the prediction.</p>")
        b.append('<div class="card">' + legend([(sim_name, "--sim"), (fpga_name, "--fpga")])
                 + grouped_bars(cats, [(sim_name, "--sim", res["class_ref"]),
                                       (fpga_name, "--fpga", res["class_hw"])],
                                x_label="class")
                 + "<details><summary>Table</summary>"
                 + table_html(["class", sim_name, fpga_name],
                              [(c, r, h) for c, r, h in zip(cats, res["class_ref"],
                                                            res["class_hw"])], num=(1, 2))
                 + "</details></div>")
    if res["rows"]:
        b.append(f"<h2>The first neurons to differ</h2><p class=\"sub\">{esc(res['origin'])}, "
                 f"step {res['first']}, largest differences first.</p>")
        b.append('<div class="card">' + table_html(
            ["neuron", "where", "shift", "threshold", "before", "sim", "FPGA", "likely cause"],
            [(esc(r["key"]), esc(r["where"]), r["shift"], f"{r['theta']:,}" if r["theta"]
              is not None else "", f"{r['before']:,}", f"{r['sim']:,}", f"{r['fpga']:,}",
              esc(r["why"])) for r in res["rows"]], num=(2, 3, 4, 5, 6)) + "</div>")
    if res["scatter"]:
        b.append("<h2>Error against the membrane before it</h2><p class=\"sub\">Every neuron "
                 f"of {esc(res['origin'])} at step {res['first']}. A leak bug that only hits "
                 "negative membranes leaves errors only left of the zero line.</p>")
        b.append('<div class="card">' + scatter(
            res["scatter"], x_label="membrane the step before (simulator)",
            y_label="FPGA minus simulator") + "</div>")
    if res["traces"]:
        b.append("<h2>Membranes over the clip</h2>")
        b.append(legend([(sim_name, "--sim"), (fpga_name, "--fpga")]) + '<div class="multiples">')
        for tr in res["traces"]:
            hl = [(tr["theta"], "threshold")] if tr["theta"] is not None and \
                abs(tr["theta"]) < 4 * max(1, max(abs(v) for v in tr["sim"] + tr["fpga"])) else []
            b.append('<div class="card">' + line_chart(
                [(sim_name, "--sim", tr["sim"]), (fpga_name, "--fpga", tr["fpga"])],
                width=460, height=200, hlines=hl, title=tr["key"]) + "</div>")
        b.append("</div>")
    return page("FPGA vs simpleSim", sub, "".join(b))


def report_micro(results):
    targets = results[0]["targets"] if results else ["simpleSim", "CRI"]
    sim_name, fpga_name = targets
    b = ['<div class="tiles">'] + [tile(t["name"].replace("_", " "),
                                        "pass" if t["pass"] else "fail") for t in results]
    b.append("</div>")
    b.append('<div class="card">' + table_html(
        ["test", "question", "result", "smallest failing case"],
        [(esc(t["name"]), esc(t["question"]), status(t["pass"]), esc(t["first_fail"] or ""))
         for t in results]) + "</div>")
    for t in results:
        b.append(f"<h2>{esc(t['name'].replace('_', ' '))}</h2><p class=\"sub\">"
                 f"{esc(t['question'])}</p>")
        if t["name"] == "negative_leak":
            b.append(legend([(sim_name, "--sim"), (fpga_name, "--fpga")])
                     + '<div class="multiples small">')
            for row in t["rows"]:
                if not row["case"].startswith("negative"):
                    continue
                b.append('<div class="card">' + line_chart(
                    [(sim_name, "--sim", row["ref"]), (fpga_name, "--fpga", row["hw"])],
                    width=300, height=170, title=f"{row['case']}: "
                    f"{'pass' if row['pass'] else 'fail'}") + "</div>")
            b.append("</div>")
        elif t["name"] in ("output_burst", "internal_burst", "fan_out"):
            cats = [row["case"].replace(" spikes in one step", "").replace("axon to ", "axon→")
                    .replace("neuron to ", "neuron→").replace(" neurons", "")
                    for row in t["rows"]]
            b.append('<div class="card">' + legend([(sim_name, "--sim"), (fpga_name, "--fpga")])
                     + grouped_bars(cats, [(sim_name, "--sim", [r["ref"] for r in t["rows"]]),
                                           (fpga_name, "--fpga", [r["hw"] for r in t["rows"]])],
                                    x_label="spikes sent at once" if "burst" in t["name"]
                                    else "fan-out") + "</div>")
        rows = []
        for row in t["rows"]:
            r, h = row["ref"], row["hw"]
            show = (lambda v: f"{len(v)} steps, last {v[-1]:,}" if isinstance(v, list) and v
                    and isinstance(v[0], int) and len(v) > 6 else v)
            extra = ""
            if row.get("matches_logical"):
                extra = (f" (matches a {row['logical_bits']}-bit logical shift, leak "
                         f"{row['sim_order']} the step's input)")
            rows.append((esc(row["case"]), status(row["pass"]) + esc(extra),
                         esc(show(r)), esc(show(h))))
        b.append('<details><summary>Every case</summary><div class="card">'
                 + table_html(["case", "result", sim_name, fpga_name], rows)
                 + "</div></details>")
    return page("FPGA micro tests", f"Each network asks one question; {sim_name} is the "
                f"reference and {fpga_name} must match it exactly.", "".join(b))


def write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="one clip, every membrane at every step")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--clips", required=True, help="a match_simplesim.py --save-clips file")
    r.add_argument("--split", default="val")
    r.add_argument("--clip", type=int, default=0)
    r.add_argument("--target", default="simpleSim", choices=["simpleSim", "CRI"])
    r.add_argument("--core", type=int, default=0)
    r.add_argument("--tau", type=int, default=None)
    r.add_argument("-o", "--out", required=True)
    c = sub.add_parser("compare", help="two records -> the first difference and a report")
    c.add_argument("ref", help="the simpleSim record")
    c.add_argument("hw", help="the FPGA record")
    c.add_argument("-o", "--out", default="fpga_diff.html")
    w = sub.add_parser("show", help="one record: which neurons the negative-leak bug can hit")
    w.add_argument("record")
    w.add_argument("-o", "--out", default="fpga_exposure.html")
    m = sub.add_parser("micro", help="tiny networks, one question each, on both targets")
    m.add_argument("--targets", nargs=2, default=["simpleSim", "CRI"])
    m.add_argument("--core", type=int, default=0)
    m.add_argument("-o", "--out", default="fpga_micro.html")
    for p in (r, m):
        p.add_argument("--hs-path", nargs="*", default=[],
                       help="directories to put on sys.path for hs_api, hs_bridge, "
                            "connectome_utils (last on the line)")
    a = ap.parse_args(argv)
    for d in getattr(a, "hs_path", []):
        sys.path.insert(0, d)
    return {"record": cmd_record, "compare": cmd_compare, "show": cmd_show,
            "micro": cmd_micro}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
