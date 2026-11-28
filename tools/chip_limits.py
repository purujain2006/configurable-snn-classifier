"""How large a network one core of the FPGA takes, measured on the card.

Run on the FPGA host from ~/hs_api with `poetry run python`, after booking the
FPGA, checking that nobody else is using it, and flashing the bitstream:

    poetry run python ~/configurable-snn-classifier/tools/chip_limits.py --label v14 -o ~/fpga_run/v14_limits.html --hs-path ~/hs_api ~/hs_bridge ~/connectome_utils

On a host with several cards (crisdsc2) add --card N, as for
tools/match_simplesim.py.

The limits the search designs to (snnsearch/hardware.py) were measured on the
L6m bitstream. The tightest, 16,000 input axons, holds a two-channel input to
89 x 89 pixels. v14's hs_bridge lays out pointer tables for 2**17 axons and
2**17 neurons per core, so v14 may take much more. This finds out.

For each limit it builds networks at the search's current value, just past it
and well past it, runs each on simpleSim and on the FPGA, and compares the
output spikes. A size passes when the FPGA matches simpleSim exactly and
simpleSim matches the answer worked out by hand, which checks the probe
itself. A limit stops at its first size that does not pass. After an FPGA
error it runs a one-neuron health check; if that fails too, it stops
everything, since the card needs a reflash before anything else it reports
can be trusted.

  input_axons     N axons. About a dozen of them, on both sides of 2**8,
                  2**14, 2**15, 2**16 and at the end, each wired to its own
                  neuron and driven one at a time. An axon number cut to fewer
                  bits wakes another axon's neuron, or none.
  neurons         N neurons on one core. About a dozen, spread the same way,
                  are each driven once and must each fire once.
  axon_fan_out    one axon to K neurons: all K fire.
  neuron_fan_out  one neuron to K neurons: all K fire.
  axon_fan_in     K axons to one neuron, 1,024 a step so that no step is a
                  large burst. With weight 1, a neuron with threshold K - 1
                  fires once and one with threshold K never: nothing lost and
                  nothing counted twice.
  neuron_fan_in   the same through K source neurons.

It writes an HTML report and a JSON file next to it, and prints a `limits:`
block for a search config with the largest size that passed for each search
limit. A size that passed is a floor: the true limit can be higher.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# tools/ can hold source clones of hs_api and connectome_utils; keep them from
# shadowing the installed packages, as fpga_diff.py and match_simplesim.py do
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, ROOT)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fd = _load("fpga_diff")           # builds a network on a target; the report's pieces
ms = _load("match_simplesim")     # which card, and input packets built with numpy

CHUNK = 1024       # spikes the fan-in networks send in one step
NO_LEAK = 63       # shift 63: a positive membrane never leaks

FAMILIES = [
    # (name, the search limit it measures, sizes to try, question)
    ("input_axons", "total_axons", [16_000, 16_384, 16_385, 32_768, 65_536, 131_072],
     "How many input axons can one core take?"),
    ("neurons", None, [65_536, 122_880, 131_040, 131_072],
     "How many neurons can one core hold?"),
    ("axon_fan_out", "axon_fan_out", [4_096, 4_097, 8_192],
     "How many neurons can one input axon feed?"),
    ("neuron_fan_out", "neuron_fan_out", [4_095, 4_096, 8_192],
     "How many neurons can one neuron feed?"),
    ("axon_fan_in", "axon_fan_in", [8_191, 8_192, 16_384, 32_768],
     "How many input axons can feed one neuron?"),
    ("neuron_fan_in", "neuron_fan_in", [8_159, 8_160, 16_384, 32_768],
     "How many neurons can feed one neuron?"),
]
NAMES = [f[0] for f in FAMILIES]

STATUS_TEXT = {
    "pass": "matches simpleSim",
    "fail": "runs, but the spikes differ from simpleSim",
    "refused": "raised an error while loading",
    "hang": "a step never finished",
    "error": "raised an error while stepping",
    "probe_error": "simpleSim disagrees with the expected answer (a bug in this probe)",
    "sim_failed": "simpleSim itself could not run it",
    "skipped": "skipped",
    "not_run": "not run",
}
DONE = ("pass", "skipped", "not_run")


class Models:
    """One hs_api model object per threshold, shared by every neuron that
    uses it. hs_api pads each model object's neurons to a multiple of 32, so
    a model per neuron would multiply the neuron count (see fpga_diff)."""

    def __init__(self):
        self.by_theta = {}

    def __call__(self, theta):
        from snnsearch.hs_export import hs_lif_neuron
        theta = int(theta)
        if theta not in self.by_theta:
            self.by_theta[theta] = hs_lif_neuron(theta, NO_LEAK)
        return self.by_theta[theta]


def edges(n):
    """About a dozen indices below n: 0 and 1, both sides of 2**8, 2**14,
    2**15, 2**16 and 2**17, the middle, and the last two."""
    pts = {0, 1, n // 2, n - 2, n - 1}
    for b in (8, 14, 15, 16, 17):
        pts |= {2 ** b - 1, 2 ** b}
    return sorted(p for p in pts if 0 <= p < n)


def _one_at_a_time(points, what):
    """Drive point p's axon a{p} at one step, rest one step: its neuron n{p}
    fires on the step after the drive, and at no other step."""
    sched, cases = [], []
    for p in points:
        t = len(sched)
        sched += [[f"a{p}"], []]
        cases.append((f"{what} {p:,}", ("steps", f"n{p}"), [t + 1]))
    return sched, cases


def net_input_axons(n, lif):
    pts = edges(n)
    axons = {f"a{i}": [] for i in range(n)}
    for p in pts:
        axons[f"a{p}"] = [(f"n{p}", 1)]
    conns = {f"n{p}": ([], lif(0)) for p in pts}
    sched, cases = _one_at_a_time(pts, "axon")
    return dict(axons=axons, conns=conns, outs=list(conns), sched=sched, cases=cases)


def net_neurons(n, lif):
    pts = edges(n)
    conns = {f"n{i}": ([], lif(0)) for i in range(n)}
    axons = {f"a{p}": [(f"n{p}", 1)] for p in pts}
    sched, cases = _one_at_a_time(pts, "neuron")
    return dict(axons=axons, conns=conns, outs=[f"n{p}" for p in pts], sched=sched, cases=cases)


def net_axon_fan_out(k, lif):
    axons = {"a": [(f"t{j}", 1) for j in range(k)]}
    conns = {f"t{j}": ([], lif(0)) for j in range(k)}
    return dict(axons=axons, conns=conns, outs=list(conns), sched=[["a"], [], []],
                cases=[(f"all {k:,} targets fire", ("count", "t", 0, 3), k)])


def net_neuron_fan_out(k, lif):
    axons = {"a": [("s", 1)]}
    conns = {"s": ([(f"t{j}", 1) for j in range(k)], lif(0))}
    conns.update({f"t{j}": ([], lif(0)) for j in range(k)})
    return dict(axons=axons, conns=conns, outs=[f"t{j}" for j in range(k)],
                sched=[["a"], [], [], []],
                cases=[(f"all {k:,} targets fire", ("count", "t", 0, 4), k)])


def _fan_in_cases(k, fires_at):
    return [(f"threshold {k - 1:,} fires once (all {k:,} inputs counted)", ("steps", "lo"),
             [fires_at]),
            (f"threshold {k:,} never fires (no input counted twice)", ("steps", "hi"), [])]


def net_axon_fan_in(k, lif):
    m = math.ceil(k / CHUNK)
    axons = {f"a{i}": [("lo", 1), ("hi", 1)] for i in range(k)}
    conns = {"lo": ([], lif(k - 1)), "hi": ([], lif(k))}
    sched = [[f"a{i}" for i in range(c * CHUNK, min(k, (c + 1) * CHUNK))] for c in range(m)]
    return dict(axons=axons, conns=conns, outs=["lo", "hi"], sched=sched + [[], [], []],
                cases=_fan_in_cases(k, m))


def net_neuron_fan_in(k, lif):
    m = math.ceil(k / CHUNK)
    axons = {f"d{c}": [(f"s{j}", 1) for j in range(c * CHUNK, min(k, (c + 1) * CHUNK))]
             for c in range(m)}
    conns = {f"s{j}": ([("lo", 1), ("hi", 1)], lif(0)) for j in range(k)}
    conns["lo"] = ([], lif(k - 1))
    conns["hi"] = ([], lif(k))
    sched = [[f"d{c}"] for c in range(m)] + [[], [], [], []]
    return dict(axons=axons, conns=conns, outs=["lo", "hi"], sched=sched,
                cases=_fan_in_cases(k, m + 1))


def net_health(lif):
    return dict(axons={"a": [("n", 1)]}, conns={"n": ([], lif(0))}, outs=["n"],
                sched=[["a"], [], []], cases=[("one neuron fires", ("steps", "n"), [1])])


BUILDERS = {"input_axons": net_input_axons, "neurons": net_neurons,
            "axon_fan_out": net_axon_fan_out, "neuron_fan_out": net_neuron_fan_out,
            "axon_fan_in": net_axon_fan_in, "neuron_fan_in": net_neuron_fan_in}
# input axons a network of this family and size needs (others need a few)
AXONS_NEEDED = {"input_axons": lambda n: n, "axon_fan_in": lambda k: k}


def run_spikes(net, clear, sched, outs):
    """Output spikes, steps x outs, for one run from cleared membranes."""
    clear()
    col = {k: j for j, k in enumerate(outs)}
    O = np.zeros((len(sched), len(outs)), bool)
    for t, inputs in enumerate(sched):
        out = net.step(list(inputs))
        for k in (out[0] if isinstance(out, tuple) else out):   # the FPGA adds counters
            j = col.get(k)
            if j is not None:
                O[t, j] = True
    return O


def _err(exc):
    return f"{type(exc).__name__}: {exc}"[:400]


def on_target(spec, target, core=0):
    """Build and run one network on one target: {"status", "O" or "error",
    "s"}. status is ran, refused (an error while loading), hang (hs_bridge's
    step timeout) or error (any other error while stepping)."""
    t0 = time.time()
    try:
        net, clear = fd.build(spec["axons"], spec["conns"], spec["outs"], target, core)
    except Exception as exc:
        return {"status": "refused", "error": _err(exc), "s": time.time() - t0}
    try:
        O = run_spikes(net, clear, spec["sched"], spec["outs"])
    except TimeoutError as exc:
        return {"status": "hang", "error": _err(exc), "s": time.time() - t0}
    except Exception as exc:
        return {"status": "error", "error": _err(exc), "s": time.time() - t0}
    return {"status": "ran", "O": O, "s": time.time() - t0}


def summary(O, outs, spec):
    """("steps", key): the steps key fired at. ("count", prefix, lo, hi):
    spikes of outputs named prefix... in steps lo to hi - 1."""
    if spec[0] == "steps":
        return np.where(O[:, outs.index(spec[1])])[0].tolist()
    _, prefix, lo, hi = spec
    cols = [j for j, k in enumerate(outs) if k.startswith(prefix)]
    return int(O[lo:hi][:, cols].sum())


def judge(spec, results, targets):
    """The row for one network run on the reference and the chip."""
    ref, chip = (results[t] for t in targets)
    cases = []
    for label, what, expected in spec["cases"]:
        r = summary(ref["O"], spec["outs"], what) if ref["status"] == "ran" else None
        c = summary(chip["O"], spec["outs"], what) if chip["status"] == "ran" else None
        cases.append({"case": label, "expected": expected, "sim": r, "hw": c,
                      "pass": r == expected and c == r})
    if ref["status"] != "ran":
        status = "sim_failed"
    elif any(x["sim"] != x["expected"] for x in cases):
        status = "probe_error"
    elif chip["status"] != "ran":
        status = chip["status"]
    else:
        status = "pass" if all(x["pass"] for x in cases) else "fail"
    error = next((results[t].get("error") for t in reversed(targets) if results[t].get("error")),
                 None)
    return {"status": status, "cases": cases, "error": error,
            "seconds": {t: round(results[t]["s"], 1) for t in targets}}


def healthy(target, core, lif):
    spec = net_health(lif)
    r = on_target(spec, target, core)
    return r["status"] == "ran" and summary(r["O"], spec["outs"], ("steps", "n")) == [1]


def probe(targets=("simpleSim", "CRI"), core=0, only=None, log=print):
    """Every limit's networks in order, each size on both targets. Returns
    the rows: family, size, status, cases, error, seconds, note."""
    lif = Models()
    rows, stop, max_axons = [], None, None
    for name, _key, sizes, _question in FAMILIES:
        if only and name not in only:
            continue
        first_bad = None
        for size in sizes:
            row = {"family": name, "size": size}
            need = AXONS_NEEDED.get(name, lambda _s: 0)(size)
            if stop:
                row.update(status="not_run", note=stop)
            elif first_bad is not None:
                row.update(status="skipped", note=f"{first_bad:,} already did not pass")
            elif max_axons is not None and name != "input_axons" and need > max_axons:
                row.update(status="skipped", note=f"needs {need:,} input axons; input_axons "
                                                  f"passed only up to {max_axons:,}")
            else:
                spec = BUILDERS[name](size, lif)
                results = {t: on_target(spec, t, core) for t in targets}
                row.update(judge(spec, results, targets))
                secs = ", ".join(f"{t} {s:.0f}s" for t, s in row["seconds"].items())
                log(f"{name} {size:,}: {row['status']} ({secs})"
                    + (f"  {row['error']}" if row["error"] and row["status"] != "pass" else ""))
                if row["status"] in ("refused", "hang", "error"):
                    row["healthy_after"] = healthy(targets[1], core, lif)
                    log(f"  one-neuron health check afterwards: "
                        f"{'passed' if row['healthy_after'] else 'FAILED'}")
                    if not row["healthy_after"]:
                        stop = (f"the card failed a one-neuron health check after {name} "
                                f"{size:,}; reflash it before trusting anything it reports")
            if row["status"] not in DONE and first_bad is None:
                first_bad = size
            rows.append(row)
        if name == "input_axons":
            passed = [r["size"] for r in rows if r["family"] == name and r["status"] == "pass"]
            max_axons = max(passed) if passed else None
    return rows


def family_summary(rows, name):
    rs = [r for r in rows if r["family"] == name]
    passed = [r["size"] for r in rs if r["status"] == "pass"]
    bad = next((r for r in rs if r["status"] not in DONE), None)
    return {"largest_pass": max(passed) if passed else None, "first_bad": bad,
            "all_passed": bool(rs) and all(r["status"] == "pass" for r in rs), "ran": bool(rs),
            "measured": any(r["status"] not in ("skipped", "not_run") for r in rs),
            "why_not": next((r.get("note") for r in rs if r.get("note")), None)}


def limits_block(rows, label=""):
    """A config `limits:` block: for each search limit, the largest size that
    matched simpleSim. Empty when nothing passed."""
    from snnsearch.hardware import _DEFAULT_LIMITS
    lines = []
    for name, key, _sizes, _q in FAMILIES:
        s = family_summary(rows, name)
        if key is None or s["largest_pass"] is None:
            continue
        if s["all_passed"]:
            why = "every size tried passed; the limit may be higher"
        elif s["first_bad"] is not None:
            why = f"{s['first_bad']['size']:,} did not pass"
        else:
            why = "larger sizes were not run"
        lines.append(f"  {key}: {s['largest_pass']}    # was {_DEFAULT_LIMITS[key]}; {why}")
    if not lines:
        return ""
    head = (f"limits:   # tools/chip_limits.py{', ' + label if label else ''}, "
            f"{time.strftime('%Y-%m-%d')}: the largest size that matched simpleSim")
    return "\n".join([head] + lines)


def text_summary(rows):
    from snnsearch.hardware import _DEFAULT_LIMITS
    out = []
    for name, key, _sizes, _q in FAMILIES:
        s = family_summary(rows, name)
        if not s["ran"]:
            continue
        now = f"{_DEFAULT_LIMITS[key]:,}" if key else "-"
        lp = s["largest_pass"]
        line = f"{name:15s} search uses {now:>7s}   "
        if not s["measured"]:
            out.append(line + f"not measured: {s['why_not']}")
            continue
        if lp is None:
            line += "nothing passed"
        else:
            line += f"passed up to {lp:,}"
            if s["all_passed"]:
                line += " (every size tried; the limit may be higher)"
        b = s["first_bad"]
        if b is not None:
            line += f"; {b['size']:,}: {STATUS_TEXT[b['status']]}"
            if b.get("error"):
                line += f" ({b['error'][:120]})"
        out.append(line)
    return "\n".join(out)


def _status_html(st):
    if st == "pass":
        return fd.status(True)
    if st == "fail":
        return fd.status(False)
    return fd.esc(STATUS_TEXT.get(st, st))


def report(rows, label="", targets=("simpleSim", "CRI")):
    from snnsearch.hardware import _DEFAULT_LIMITS
    sim_name, chip_name = targets
    b = ['<div class="tiles">']
    for name, key, _sizes, _q in FAMILIES:
        s = family_summary(rows, name)
        if not s["ran"]:
            continue
        lp = s["largest_pass"]
        value = ("not measured" if not s["measured"] else "none passed" if lp is None
                 else f"≥ {lp:,}" if s["all_passed"] else f"{lp:,}")
        b.append(fd.tile(name.replace("_", " "), value))
    b.append("</div>")
    summary_rows = []
    for name, key, _sizes, question in FAMILIES:
        s = family_summary(rows, name)
        if not s["ran"]:
            continue
        bad = s["first_bad"]
        summary_rows.append((
            fd.esc(name), fd.esc(question),
            fd.esc(f"{_DEFAULT_LIMITS[key]:,}" if key else "-"),
            fd.esc("none" if s["largest_pass"] is None else f"{s['largest_pass']:,}"),
            fd.esc("" if bad is None else f"{bad['size']:,}"),
            fd.esc("" if bad is None else STATUS_TEXT[bad["status"]])))
    b.append('<div class="card">' + fd.table_html(
        ["limit", "question", "search uses", "largest that passed", "first that did not",
         "what happened"], summary_rows) + "</div>")
    block = limits_block(rows, label)
    if block:
        b.append("<h2>For a search config</h2><p class=\"note\">A size that passed is a floor: "
                 "the real limit can be higher. Paste this into the config's top level.</p>"
                 f'<div class="card"><pre>{fd.esc(block)}</pre></div>')
    for name, _key, _sizes, question in FAMILIES:
        rs = [r for r in rows if r["family"] == name]
        if not rs:
            continue
        b.append(f"<h2>{fd.esc(name.replace('_', ' '))}</h2><p class=\"sub\">{fd.esc(question)}</p>")
        table = []
        for r in rs:
            secs = r.get("seconds") or {}
            table.append((fd.esc(f"{r['size']:,}"), _status_html(r["status"]),
                          fd.esc(" / ".join(f"{secs[t]:.0f}s" for t in targets if t in secs)),
                          fd.esc(r.get("note") or r.get("error") or "")))
        b.append('<div class="card">' + fd.table_html(
            ["size", "result", f"time ({sim_name} / {chip_name})", "note"], table) + "</div>")
        detail = []
        for r in rs:
            for c in r.get("cases", []):
                detail.append((fd.esc(f"{r['size']:,}"), fd.esc(c["case"]),
                               fd.status(c["pass"]), fd.esc(c["expected"]), fd.esc(c["sim"]),
                               fd.esc(c["hw"])))
        if detail:
            b.append('<details><summary>Every case</summary><div class="card">' + fd.table_html(
                ["size", "case", "result", "expected", sim_name, chip_name], detail)
                + "</div></details>")
    title = "Chip limits" + (f": {label}" if label else "")
    return fd.page(title, f"Each network runs on {sim_name} and on {chip_name}. A size passes "
                   f"when {chip_name} matches {sim_name} exactly and {sim_name} matches the "
                   f"answer worked out by hand.", "".join(b))


def prepare_card(card):
    """Point hs_bridge at --card, and build its input packets with numpy
    (checked byte for byte against hs_bridge's own first)."""
    from hs_bridge.FPGA_Execution import fpga_controller
    if card is not None:
        import hs_bridge.wrapped_dmadump.dmadump as dd
        ms.use_card(dd, card)
        print(f"FPGA card {card}: every DMA read and write goes to it", flush=True)
    try:
        import hs_bridge.network as hsb_network
    except ImportError:
        hsb_network = None
    if ms.use_fast_input_packets(fpga_controller, hsb_network):
        print("input packets built with numpy (byte-identical to hs_bridge's input_user, "
              "checked just now)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs=2, default=["simpleSim", "CRI"],
                    help="the reference, then the chip")
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--card", type=int, default=None,
                    help="the FPGA card, N in /dev/adxdmaN; required on a host with several "
                         "cards, such as crisdsc2")
    ap.add_argument("--only", nargs="+", choices=NAMES, help="measure only these limits")
    ap.add_argument("--label", default="", help="the bitstream, for the report, e.g. v14")
    ap.add_argument("-o", "--out", default="chip_limits.html")
    ap.add_argument("--hs-path", nargs="*", default=[],
                    help="directories to put on sys.path for hs_api, hs_bridge, "
                         "connectome_utils (last on the line)")
    a = ap.parse_args(argv)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    if not os.path.isdir(out_dir):
        ap.error(f"-o: the folder {out_dir} does not exist")
    for d in a.hs_path:
        sys.path.insert(0, d)
    try:
        import hs_api.api  # noqa: F401
    except ImportError as exc:
        ap.error(f"hs_api does not import ({exc}). Run from ~/hs_api with poetry run python "
                 f"and pass --hs-path ~/hs_api ~/hs_bridge ~/connectome_utils")
    if "CRI" in a.targets:
        error, warning = ms.card_check(a.card)
        if error:
            ap.error(error)
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
        prepare_card(a.card)
    rows = probe(tuple(a.targets), a.core, a.only,
                 log=lambda s: print(s, flush=True))
    with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
        json.dump({"label": a.label, "targets": a.targets, "core": a.core, "card": a.card,
                   "rows": rows}, fh, indent=1, default=int)
    fd.write(a.out, report(rows, a.label, tuple(a.targets)))
    print()
    print(text_summary(rows))
    block = limits_block(rows, a.label)
    if block:
        print("\nFor a search config:\n" + block)
    stopped = next((r["note"] for r in rows if r["status"] == "not_run"), None)
    if stopped:
        print(f"\nSTOPPED EARLY: {stopped}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
