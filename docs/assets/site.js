/* site.js — shared chrome: nav, table of contents, reveal animations,
   and small canvas plotting helpers used by the interactive demos. */
(function () {
  "use strict";

  /* ---------- navigation ---------- */
  const PAGES = [
    ["index.html", "01", "Big picture"],
    ["neurons.html", "02", "Spiking neurons"],
    ["architecture.html", "03", "Architecture"],
    ["hardware.html", "04", "Hardware & quantization"],
    ["training.html", "05", "Data & training"],
    ["search.html", "06", "The search"],
    ["reference.html", "07", "Code reference"]
  ];

  function currentPage() {
    const p = location.pathname.split("/").pop();
    return p === "" ? "index.html" : p;
  }

  function buildNav() {
    const nav = document.createElement("nav");
    nav.className = "nav";
    const cur = currentPage();
    const links = PAGES.map(([href, num, label]) =>
      `<a href="${href}" class="${href === cur ? "active" : ""}"><span class="num">${num}</span>${label}</a>`
    ).join("");
    nav.innerHTML = `
      <a class="nav-logo" href="index.html">
        <svg class="mark" width="20" height="20" viewBox="0 0 32 32" aria-hidden="true"><path d="M2 22h5l3-13 4 19 3-11h4l2 5h7" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/></svg><span>SNN <span style="color:var(--faint);">/</span> Silicon</span>
      </a>
      <div class="nav-links">${links}</div>
      <a class="nav-gh" href="https://github.com/purujain2006/configurable-snn-classifier" aria-label="GitHub" title="View the code on GitHub">
        <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
        </svg>
      </a>`;
    document.body.prepend(nav);
  }

  /* ---------- table of contents (from h2[id]) ---------- */
  function buildTOC() {
    const toc = document.querySelector(".toc");
    if (!toc) return;
    const heads = document.querySelectorAll("main h2[id]");
    if (!heads.length) { toc.remove(); return; }
    let html = '<div class="toc-title">On this page</div>';
    heads.forEach(h => {
      const txt = h.textContent.replace(/^\s*\d+[\.\)]?\s*/, "");
      html += `<a href="#${h.id}" data-target="${h.id}">${txt}</a>`;
    });
    toc.innerHTML = html;

    const links = toc.querySelectorAll("a");
    const map = new Map();
    links.forEach(a => map.set(a.dataset.target, a));
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          links.forEach(a => a.classList.remove("active"));
          const a = map.get(e.target.id);
          if (a) a.classList.add("active");
        }
      });
    }, { rootMargin: "-15% 0px -70% 0px" });
    heads.forEach(h => obs.observe(h));
  }

  /* ---------- reveal on scroll ---------- */
  function reveals() {
    const els = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window)) { els.forEach(e => e.classList.add("vis")); return; }
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("vis"); obs.unobserve(e.target); } });
    }, { threshold: 0.06 });
    els.forEach(e => obs.observe(e));
  }

  /* ---------- canvas helpers ---------- */
  // Prepare a canvas for crisp drawing at devicePixelRatio; returns ctx and CSS-pixel size.
  function setupCanvas(canvas, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    const h = cssHeight || canvas.clientHeight || 300;
    canvas.style.height = h + "px";
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  const PALETTE = {
    bg: "#050505", grid: "#2a272455", axis: "#3a3733",
    text: "#a7a29a", accent: "#ece7de", accent2: "#cf8763",
    spike: "#e0a94f", good: "#9bb17a", bad: "#cf6b58", warn: "#e0a94f"
  };

  // Minimal line plot. series: [{xs, ys, color, width, dash, label, step}]
  function linePlot(canvas, series, opts) {
    opts = opts || {};
    const { ctx, w, h } = setupCanvas(canvas, opts.height || 305);
    const padL = opts.padL ?? 58, padR = opts.padR ?? 18, padT = opts.padT ?? 18, padB = opts.padB ?? 40;
    const pw = w - padL - padR, ph = h - padT - padB;

    let xmin = opts.xmin, xmax = opts.xmax, ymin = opts.ymin, ymax = opts.ymax;
    const allX = [], allY = [];
    series.forEach(s => { allX.push(...s.xs); allY.push(...s.ys); });
    if (xmin === undefined) xmin = Math.min(...allX);
    if (xmax === undefined) xmax = Math.max(...allX);
    if (ymin === undefined) ymin = Math.min(...allY);
    if (ymax === undefined) ymax = Math.max(...allY);
    if (ymax === ymin) { ymax += 1; ymin -= 1; }
    if (xmax === xmin) { xmax += 1; }
    const X = x => padL + (x - xmin) / (xmax - xmin) * pw;
    const Y = y => padT + (1 - (y - ymin) / (ymax - ymin)) * ph;

    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, w, h);

    // grid + ticks
    ctx.strokeStyle = PALETTE.grid; ctx.fillStyle = PALETTE.text;
    ctx.font = "13px 'JetBrains Mono', monospace"; ctx.lineWidth = 1;
    const yticks = opts.yticks || niceTicks(ymin, ymax, 5);
    yticks.forEach(t => {
      if (t < ymin - 1e-9 || t > ymax + 1e-9) return;
      ctx.beginPath(); ctx.moveTo(padL, Y(t)); ctx.lineTo(w - padR, Y(t)); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(fmtTick(t), padL - 7, Y(t));
    });
    const xticks = opts.xticks || niceTicks(xmin, xmax, 7);
    xticks.forEach(t => {
      if (t < xmin - 1e-9 || t > xmax + 1e-9) return;
      ctx.beginPath(); ctx.moveTo(X(t), padT); ctx.lineTo(X(t), h - padB); ctx.stroke();
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(fmtTick(t), X(t), h - padB + 6);
    });

    // axis labels
    if (opts.xlabel) { ctx.textAlign = "center"; ctx.fillText(opts.xlabel, padL + pw / 2, h - 10); }
    if (opts.ylabel) {
      ctx.save(); ctx.translate(15, padT + ph / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center"; ctx.fillText(opts.ylabel, 0, 0); ctx.restore();
    }

    // horizontal reference lines
    (opts.hlines || []).forEach(hl => {
      ctx.strokeStyle = hl.color || PALETTE.warn;
      ctx.setLineDash(hl.dash || [5, 4]); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(padL, Y(hl.y)); ctx.lineTo(w - padR, Y(hl.y)); ctx.stroke();
      ctx.setLineDash([]);
      if (hl.label) {
        ctx.fillStyle = hl.color || PALETTE.warn;
        ctx.textAlign = "left"; ctx.textBaseline = "bottom";
        ctx.fillText(hl.label, padL + 5, Y(hl.y) - 3);
      }
    });

    // series
    series.forEach(s => {
      ctx.strokeStyle = s.color || PALETTE.accent;
      ctx.lineWidth = s.width || 2;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      for (let i = 0; i < s.xs.length; i++) {
        const px = X(s.xs[i]), py = Y(s.ys[i]);
        if (i === 0) ctx.moveTo(px, py);
        else if (s.step) { ctx.lineTo(px, Y(s.ys[i - 1])); ctx.lineTo(px, py); }
        else ctx.lineTo(px, py);
      }
      ctx.stroke(); ctx.setLineDash([]);
      if (s.dots) {
        ctx.fillStyle = s.color || PALETTE.accent;
        for (let i = 0; i < s.xs.length; i++) {
          ctx.beginPath(); ctx.arc(X(s.xs[i]), Y(s.ys[i]), s.dotR || 2.5, 0, Math.PI * 2); ctx.fill();
        }
      }
    });

    // legend
    if (opts.legend !== false) {
      let lx = padL + 10, ly = padT + 8;
      series.filter(s => s.label).forEach(s => {
        ctx.fillStyle = s.color || PALETTE.accent;
        ctx.fillRect(lx, ly + 2, 14, 3);
        ctx.fillStyle = PALETTE.text; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(s.label, lx + 22, ly + 4);
        ly += 19;
      });
    }
    return { X, Y, ctx, w, h, padL, padR, padT, padB };
  }

  function niceTicks(lo, hi, n) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const step0 = span / Math.max(1, n - 1);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    let step = mag;
    for (const m of [1, 2, 2.5, 5, 10]) { if (step0 <= m * mag) { step = m * mag; break; } }
    const ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
    return ticks;
  }
  function fmtTick(t) {
    if (Math.abs(t) >= 1e6) return (t / 1e6) + "M";
    if (Math.abs(t) >= 1e4) return (t / 1e3) + "k";
    if (Math.abs(t) < 1e-3 && t !== 0) return t.toExponential(0);
    return +t.toFixed(4) + "";
  }

  // horizontal bar chart into a container div (DOM-based; accessible + crisp)
  function barChart(el, items, opts) {
    opts = opts || {};
    const max = opts.max || Math.max(...items.map(i => i.value), 1e-9);
    el.innerHTML = items.map(i => {
      const pct = Math.max(0.5, i.value / max * 100);
      const col = i.color || "var(--accent)";
      return `<div style="display:flex;align-items:center;gap:10px;margin:7px 0;">
        <div style="flex:0 0 ${opts.labelW || 130}px;font-family:var(--mono);font-size:13.5px;color:var(--muted);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${i.label}</div>
        <div style="flex:1;height:${opts.barH || 20}px;background:var(--bg-3);border-radius:5px;overflow:hidden;">
          <div style="width:${pct}%;height:100%;background:${col};border-radius:5px;transition:width .35s ease;"></div>
        </div>
        <div style="flex:0 0 ${opts.valueW || 100}px;font-family:var(--mono);font-size:13.5px;color:var(--text);">${i.display !== undefined ? i.display : i.value}</div>
      </div>`;
    }).join("");
  }

  /* ---------- boot ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    buildNav();
    buildTOC();
    reveals();
  });

  window.Site = { setupCanvas, linePlot, barChart, PALETTE, niceTicks };
})();
