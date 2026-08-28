"""A self-contained HTML report, written after a run.

No server, no framework, no build step, and no network access at view time:
one file you can email or drop in a shared folder. Charts are inline SVG drawn
here rather than a plotting library, which keeps the dependency list at numpy
and makes the file openable on any machine.

Deliberately not an interactive dashboard. A report answers "what happened in
this run", which is the actual question after a search finishes, and costs days
less to build and maintain.
"""

import html
import json
import os
from datetime import datetime

CSS = """
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:#000;color:#f6f4f1;font:16px/1.7 Inter,system-ui,-apple-system,sans-serif;
     -webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:56px 30px 80px}
h1{font-size:34px;font-weight:650;letter-spacing:-.026em;margin:0 0 6px}
h2{font-size:24px;font-weight:650;letter-spacing:-.02em;margin:56px 0 14px;
   padding-top:24px;border-top:1px solid rgba(255,255,255,.13)}
h3{font-size:18px;font-weight:650;margin:32px 0 8px}
.kicker{font:500 11.5px/1 ui-monospace,monospace;letter-spacing:.22em;text-transform:uppercase;
        color:#e0a94f;margin:0 0 14px}
p{color:#cfcbc4;margin:14px 0}
.sub{color:#a7a29a;font-size:14.5px}
table{width:100%;border-collapse:collapse;margin:20px 0;font-size:14.5px}
th{text-align:left;font:500 11.5px/1 ui-monospace,monospace;letter-spacing:.16em;
   text-transform:uppercase;color:#a7a29a;padding:12px 14px;border-bottom:1px solid rgba(255,255,255,.24)}
td{padding:11px 14px;border-bottom:1px solid rgba(255,255,255,.075);color:#cfcbc4;vertical-align:top}
td.n{text-align:right;font-family:ui-monospace,monospace;font-size:13.5px}
td.k{font-family:ui-monospace,monospace;color:#e0a94f;font-size:13.5px;white-space:nowrap}
tr.best td{background:rgba(224,169,79,.08)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1px;
      background:rgba(255,255,255,.075);border:1px solid rgba(255,255,255,.075);margin:26px 0}
.cell{background:#030303;padding:20px}
.cell .lab{font:500 11px/1 ui-monospace,monospace;letter-spacing:.18em;text-transform:uppercase;color:#a7a29a}
.cell .val{font-size:27px;font-weight:650;margin:8px 0 4px;letter-spacing:-.02em}
.cell .val.acc{color:#e0a94f}
.cell .note{font-size:13px;color:#7d7871;line-height:1.5}
.ok{color:#9bb17a}.bad{color:#cf6b58}.warn{color:#e0a94f}
.tag{font:12px/1 ui-monospace,monospace;padding:3px 9px;border:1px solid;white-space:nowrap}
.tag.ok{border-color:rgba(155,177,122,.45)}.tag.bad{border-color:rgba(207,107,88,.45)}
figure{margin:26px 0;border:1px solid rgba(255,255,255,.13);background:#030303}
figure svg{display:block;width:100%;height:auto}
figcaption{padding:13px 18px;border-top:1px solid rgba(255,255,255,.075);
           font-size:13.5px;color:#a7a29a}
pre{margin:0;padding:18px 20px;overflow-x:auto;font:13px/1.7 ui-monospace,monospace;color:#d5d0c8;
    background:#030303;border:1px solid rgba(255,255,255,.13)}
.note{padding:18px 20px;border:1px solid rgba(255,255,255,.13);border-left:2px solid #e0a94f;
      background:rgba(255,255,255,.018);margin:24px 0;font-size:15px;color:#cfcbc4}
footer{margin-top:64px;padding-top:20px;border-top:1px solid rgba(255,255,255,.13);
       color:#7d7871;font-size:13px}
"""

E = lambda s: html.escape(str(s), quote=True)


# ----------------------------------------------------------------- charts
def _axes(w, h, pad):
    return pad, w - pad, pad, h - pad


def scatter(points, w=880, h=340, xlab="", ylab="", pad=58, highlight=None):
    """points: [(x, y, label)]. Returns inline SVG."""
    pts = [(x, y, l) for x, y, l in points if x is not None and y is not None]
    if not pts:
        return "<p class='sub'>no data to plot</p>"
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = _axes(w, h, pad)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin: xmax += 1
    if ymax == ymin: ymax += 1e-6
    X = lambda v: x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)
    Y = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    out = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<rect width="{w}" height="{h}" fill="#030303"/>')
    for i in range(5):                                   # gridlines + y labels
        gy = y0 + i * (y1 - y0) / 4
        val = ymax - i * (ymax - ymin) / 4
        out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" stroke="#2a2724"/>')
        out.append(f'<text x="{x0-9}" y="{gy+4:.1f}" fill="#7d7871" font-size="11" '
                   f'font-family="ui-monospace,monospace" text-anchor="end">{val:.3g}</text>')
    for i in range(5):
        gx = x0 + i * (x1 - x0) / 4
        val = xmin + i * (xmax - xmin) / 4
        out.append(f'<text x="{gx:.1f}" y="{y1+20}" fill="#7d7871" font-size="11" '
                   f'font-family="ui-monospace,monospace" text-anchor="middle">{val:.3g}</text>')
    for x, y, lab in pts:
        best = highlight is not None and lab == highlight
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{6 if best else 4}" '
                   f'fill="{"#e0a94f" if best else "#8f9e73"}" '
                   f'fill-opacity="{1 if best else .75}"><title>{E(lab)}: '
                   f'{x:.4g}, {y:.4g}</title></circle>')
    out.append(f'<text x="{(x0+x1)/2:.0f}" y="{h-8}" fill="#a7a29a" font-size="12" '
               f'text-anchor="middle">{E(xlab)}</text>')
    out.append(f'<text x="14" y="{(y0+y1)/2:.0f}" fill="#a7a29a" font-size="12" '
               f'text-anchor="middle" transform="rotate(-90 14 {(y0+y1)/2:.0f})">{E(ylab)}</text>')
    out.append("</svg>")
    return "".join(out)


def curves(series, w=880, h=320, pad=58, xlab="epoch", ylab="validation accuracy"):
    """series: {label: [(x, y), ...]}"""
    allpts = [p for v in series.values() for p in v]
    if not allpts:
        return "<p class='sub'>no per-epoch reports recorded</p>"
    x0, x1, y0, y1 = _axes(w, h, pad)
    xmax = max(p[0] for p in allpts) or 1
    ymax = max(p[1] for p in allpts) or 1
    X = lambda v: x0 + v / xmax * (x1 - x0)
    Y = lambda v: y1 - v / ymax * (y1 - y0)
    out = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect width="{w}" height="{h}" fill="#030303"/>']
    for i in range(5):
        gy = y0 + i * (y1 - y0) / 4
        out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" stroke="#2a2724"/>')
        out.append(f'<text x="{x0-9}" y="{gy+4:.1f}" fill="#7d7871" font-size="11" '
                   f'font-family="ui-monospace,monospace" text-anchor="end">'
                   f'{ymax - i*ymax/4:.2f}</text>')
    for lab, pts in series.items():
        if not pts:
            continue
        d = " ".join(("M" if i == 0 else "L") + f"{X(x):.1f},{Y(y):.1f}"
                     for i, (x, y) in enumerate(sorted(pts)))
        out.append(f'<path d="{d}" fill="none" stroke="#8f9e73" stroke-opacity=".55" '
                   f'stroke-width="1.5"><title>{E(lab)}</title></path>')
    out.append(f'<text x="{(x0+x1)/2:.0f}" y="{h-8}" fill="#a7a29a" font-size="12" '
               f'text-anchor="middle">{E(xlab)}</text>')
    out.append("</svg>")
    return "".join(out)


def bars(items, w=880, rowh=34, pad=58):
    """items: [(label, value, max, colour)]"""
    if not items:
        return "<p class='sub'>nothing to show</p>"
    h = rowh * len(items) + 24
    out = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
           f'<rect width="{w}" height="{h}" fill="#030303"/>']
    bx, bw = 210, w - 210 - 90
    for i, (lab, val, mx, col) in enumerate(items):
        y = 12 + i * rowh
        frac = (val / mx) if mx else 0
        out.append(f'<text x="{bx-12}" y="{y+16}" fill="#cfcbc4" font-size="13" '
                   f'font-family="ui-monospace,monospace" text-anchor="end">{E(lab)}</text>')
        out.append(f'<rect x="{bx}" y="{y+4}" width="{bw}" height="16" fill="#161412"/>')
        out.append(f'<rect x="{bx}" y="{y+4}" width="{max(1,bw*min(1,frac)):.1f}" '
                   f'height="16" fill="{col}"/>')
        out.append(f'<text x="{bx+bw+10}" y="{y+16}" fill="#a7a29a" font-size="12" '
                   f'font-family="ui-monospace,monospace">{val:,.0f}</text>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------- report
def build(run_dir, cfg=None, out_path=None):
    """Read a results directory and write report.html beside it."""
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    trials = _read_jsonl(os.path.join(run_dir, "trials.jsonl"))
    infeas = _read_jsonl(os.path.join(run_dir, "infeasible.jsonl"))
    progress = _read_jsonl(os.path.join(run_dir, "trial_progress.jsonl"))

    scored = [t for t in trials if _num(t.get("val_accuracy")) is not None]
    deployed = [t for t in trials if _num(t.get("hw_val_accuracy")) is not None]
    best = max(scored, key=lambda t: t["val_accuracy"], default=None)

    P = []
    P.append(f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
             f'<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>{E(os.path.basename(run_dir))} report</title>'
             f'<link rel="preconnect" href="https://fonts.googleapis.com">'
             f'<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;650&display=swap" rel="stylesheet">'
             f'<style>{CSS}</style></head><body><div class="wrap">')
    P.append(f'<p class="kicker">snnsearch report</p><h1>{E(os.path.basename(run_dir))}</h1>')
    P.append(f'<p class="sub">{len(trials)} trials &#183; generated '
             f'{datetime.now().strftime("%Y-%m-%d %H:%M")}</p>')

    # ---- headline ----
    P.append('<div class="grid">')
    P.append(_cell("Trials recorded", f"{len(trials)}",
                   f"{len(infeas)} rejected before any GPU time"))
    P.append(_cell("Reached conversion", f"{len(deployed)}",
                   "only these say what the chip would do",
                   accent=len(deployed) > 0))
    if best:
        P.append(_cell("Best score", f"{best['val_accuracy']*100:.2f}%",
                       f"trial {E(str(best.get('trial_id','?'))[:8])}", accent=True))
    syn = [_num(t.get("synops_per_sample")) for t in trials]
    syn = [s for s in syn if s]
    if syn:
        P.append(_cell("Best SynOps", f"{min(syn):,.0f}", "synaptic ops per sample"))
    P.append("</div>")

    if not deployed and trials:
        P.append('<div class="note"><strong>No trial reached the deployment stage.</strong> '
                 'Every score here is the floating-point network before conversion, which is '
                 'an upper bound rather than a result. Check that early stopping is not '
                 'cutting trials off before the deploy report lands.</div>')

    # ---- accuracy vs SynOps ----
    if syn:
        P.append("<h2>Accuracy against energy</h2>")
        P.append('<p>Each point is one trial. Up and to the left is better: same accuracy '
                 'for fewer synaptic operations. SynOps counts events, not joules, so it '
                 'compares across platforms but is not a wattage.</p>')
        pts = [(_num(t.get("synops_per_sample")), _num(t.get("val_accuracy")),
                str(t.get("trial_id", ""))[:8]) for t in trials]
        P.append('<figure>' + scatter(pts, xlab="SynOps per sample",
                                      ylab="validation accuracy",
                                      highlight=str(best.get("trial_id",""))[:8] if best else None)
                 + '<figcaption>Amber marks the highest-scoring trial. Hover a point for its id.'
                   '</figcaption></figure>')

    # ---- learning curves ----
    if progress:
        P.append("<h2>Learning curves</h2>")
        series = {}
        for r in progress:
            tid = str(r.get("trial_id", "?"))[:8]
            e, a = _num(r.get("epoch")), _num(r.get("val_accuracy"))
            if e is not None and a is not None:
                series.setdefault(tid, []).append((e, a))
        P.append('<figure>' + curves(series) +
                 f'<figcaption>{len(series)} trials. Short lines were stopped early by the '
                 f'scheduler.</figcaption></figure>')

    # ---- leaderboard ----
    if scored:
        P.append("<h2>Leaderboard</h2>")
        cols = [("trial_id", "trial", "k"), ("val_accuracy", "score", "n"),
                ("hw_val_accuracy", "converted", "n"), ("float_val_accuracy", "float", "n"),
                ("synops_per_sample", "synops", "n"), ("epochs_run", "epochs", "n"),
                ("deployable", "deployable", "")]
        P.append("<table><thead><tr>" +
                 "".join(f"<th>{E(t)}</th>" for _, t, _c in cols) + "</tr></thead><tbody>")
        for t in sorted(scored, key=lambda r: -r["val_accuracy"])[:25]:
            row = ["<tr class='best'>" if t is best else "<tr>"]
            for key, _t, cls in cols:
                v = t.get(key)
                row.append(f'<td class="{cls}">{_fmt(key, v)}</td>')
            P.append("".join(row) + "</tr>")
        P.append("</tbody></table>")

    # ---- why things were rejected ----
    if infeas:
        P.append("<h2>Rejected before training</h2>")
        from collections import Counter
        kinds = Counter()
        for r in infeas:
            for part in str(r.get("violations", "")).split(";"):
                part = part.strip()
                if part:
                    kinds[" ".join(part.split()[:3])] += 1
        P.append("<table><thead><tr><th>count</th><th>violation</th></tr></thead><tbody>")
        for k, c in kinds.most_common():
            P.append(f'<td class="n">{c}</td><td class="k">{E(k)}</td></tr><tr>')
        P.append("</tbody></table>")
        P.append(f'<p class="sub">{len(infeas)} configurations failed the connection check, '
                 f'which costs microseconds and never reaches a GPU.</p>')

    # ---- config ----
    if cfg:
        P.append("<h2>Configuration</h2><pre>" +
                 E(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                              indent=2, default=str)) + "</pre>")

    P.append('<footer>Written by snnsearch.report from the raw run files. '
             'SynOps counts synaptic operations per sample and is a proxy for energy, '
             'not a measurement of it.</footer>')
    P.append("</div></body></html>")

    out_path = out_path or os.path.join(run_dir, "report.html")
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join(P))
    return out_path


def _cell(lab, val, note, accent=False):
    return (f'<div class="cell"><div class="lab">{E(lab)}</div>'
            f'<div class="val{" acc" if accent else ""}">{E(val)}</div>'
            f'<div class="note">{E(note)}</div></div>')


def _fmt(key, v):
    if v is None or v == "":
        return '<span class="sub">&#8212;</span>'
    if key == "deployable":
        ok = str(v).lower() in ("true", "1")
        return f'<span class="tag {"ok" if ok else "bad"}">{"yes" if ok else "no"}</span>'
    if key in ("val_accuracy", "hw_val_accuracy", "float_val_accuracy"):
        n = _num(v)
        return f"{n*100:.2f}%" if n is not None else E(v)
    if key == "synops_per_sample":
        n = _num(v)
        return f"{n:,.0f}" if n is not None else E(v)
    if key == "trial_id":
        return E(str(v)[:8])
    return E(v)


def _num(v):
    try:
        f = float(v)
        return None if f != f else f          # NaN is not a number here either
    except (TypeError, ValueError):
        return None


def _read_jsonl(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                      # a torn last line from a killed run
    return out
