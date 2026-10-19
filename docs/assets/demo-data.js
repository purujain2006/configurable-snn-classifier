/* demo-data.js: visuals for page B (event camera, frames, splits, noise, augmentation). */
(function () {
  "use strict";
  const L = window.Learn, G = window.Gesture;
  const { S, T, C } = L;
  const RES = G.RES;
  const listeners = [];
  let kind = "rwave";

  /* one real training run: training and validation accuracy per epoch */
  const TRAIN = [0.483, 0.652, 0.699, 0.719, 0.732, 0.773, 0.764, 0.796, 0.805, 0.82, 0.826, 0.832, 0.836, 0.855, 0.858, 0.889, 0.911, 0.934, 0.921, 0.918, 0.917, 0.925, 0.931, 0.941, 0.944, 0.941, 0.946, 0.94, 0.951, 0.945, 0.947, 0.959, 0.95, 0.953, 0.952, 0.961, 0.968, 0.963, 0.97, 0.979, 0.984, 0.979, 0.982, 0.982, 0.985, 0.988, 0.992, 0.988, 0.991, 0.991, 0.992, 0.994, 0.993, 0.992, 0.995, 0.992, 0.991, 0.992, 0.995, 0.99, 0.997, 0.995, 0.994, 0.992, 0.992, 0.992, 0.989, 0.995, 0.993, 0.992, 0.992, 0.995, 0.993, 0.995, 0.991, 0.997, 0.993, 0.995, 0.997, 0.997];
  const VAL = [0.431, 0.514, 0.542, 0.486, 0.514, 0.602, 0.505, 0.528, 0.431, 0.556, 0.588, 0.519, 0.569, 0.579, 0.63, 0.699, 0.671, 0.708, 0.63, 0.685, 0.694, 0.718, 0.713, 0.69, 0.713, 0.69, 0.727, 0.731, 0.713, 0.731, 0.769, 0.685, 0.731, 0.745, 0.764, 0.755, 0.736, 0.736, 0.75, 0.727, 0.718, 0.741, 0.731, 0.745, 0.764, 0.764, 0.736, 0.727, 0.741, 0.736, 0.741, 0.727, 0.745, 0.75, 0.745, 0.745, 0.745, 0.741, 0.755, 0.745, 0.759, 0.745, 0.736, 0.75, 0.745, 0.745, 0.736, 0.745, 0.736, 0.75, 0.755, 0.759, 0.755, 0.75, 0.745, 0.75, 0.731, 0.745, 0.75, 0.741];

  L.onReady(() => {
    [dvs, frames, resize, split, overfit, noise, augment]
      .forEach(f => { try { f(); } catch (e) { console.error(f.name, e); } });
  });
  function setKind(k) { kind = k; listeners.forEach(f => f()); }

  /* ---------- B.1 event camera ---------- */
  function dvs() {
    const canvas = document.getElementById("dvs-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(RES, RES);
    const onB = new Float32Array(RES * RES), offB = new Float32Array(RES * RES);
    let clip = G.makeClip(kind), simT = 0, idx = 0;
    const readout = document.getElementById("dvs-readout");
    let winCount = [], lastShown = 0;
    function render() {
      for (let i = 0; i < RES * RES; i++) {
        const a = Math.min(1, onB[i]), b = Math.min(1, offB[i]), k = i * 4;
        img.data[k] = 6 + 218 * a + 186 * b; img.data[k + 1] = 6 + 163 * a + 116 * b;
        img.data[k + 2] = 6 + 73 * a + 86 * b; img.data[k + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }
    function advance(dsim) {
      const decay = Math.exp(-dsim / 0.03);
      for (let i = 0; i < RES * RES; i++) { onB[i] *= decay; offB[i] *= decay; }
      let end = simT + dsim, n = 0;
      for (;;) {
        while (idx < clip.n && clip.t[idx] < end) {
          const j = clip.y[idx] * RES + clip.x[idx];
          if (clip.p[idx] > 0) onB[j] = 1; else offB[j] = 1;
          idx++; n++;
        }
        if (end < clip.dur) break;
        end -= clip.dur; idx = 0;     // the recording loops
      }
      simT = end;
      winCount.push([dsim, n]);
      let tt = 0, nn = 0;
      for (let k = winCount.length - 1; k >= 0 && tt < 0.05; k--) { tt += winCount[k][0]; nn += winCount[k][1]; }
      if (winCount.length > 200) winCount.shift();
      return tt ? nn / tt : 0;
    }
    function step(dt) {
      const rate = advance(dt * 0.35);
      render();
      lastShown += dt;
      if (lastShown > 0.25) {
        lastShown = 0;
        let lit = 0;
        for (let i = 0; i < RES * RES; i++) if (onB[i] > 0.2 || offB[i] > 0.2) lit++;
        readout.innerHTML = `about <b>${L.fmt(Math.round(rate / 1000) * 1000)}</b> events per second · ${(100 - 100 * lit / (RES * RES)).toFixed(1)}% of pixels silent · shown at 0.35× speed`;
      }
      return true;
    }
    L.loop(canvas, step, { button: document.getElementById("dvs-play") });
    function load() { clip = G.makeClip(kind); simT = 0; idx = 0; onB.fill(0); offB.fill(0); for (let k = 0; k < 12; k++) advance(0.01); render(); }
    listeners.push(load);
    L.seg(document.getElementById("dvs-kind"), v => setKind(v));
    load();
  }

  /* ---------- B.2 frames ---------- */
  function frames() {
    const strip = document.getElementById("frames-strip"), thumbs = document.getElementById("frames-thumbs");
    if (!strip) return;
    let Tn = 16;
    function draw() {
      const clip = G.makeClip(kind), sl = G.slices(clip, Tn);
      const { ctx, w, h } = Site.setupCanvas(strip, 70);
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
      const B = 140, hist = new Array(B).fill(0);
      for (let i = 0; i < clip.n; i++) hist[Math.min(B - 1, Math.floor(clip.t[i] / clip.dur * B))]++;
      const mx = Math.max(...hist), bw = (w - 20) / B;
      ctx.fillStyle = "rgba(236,231,222,.45)";
      hist.forEach((v, k) => { const bh = v / mx * (h - 26); ctx.fillRect(10 + k * bw, h - 18 - bh, Math.max(1, bw - 0.5), bh); });
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1;
      for (let k = 1; k < Tn; k++) {
        const x = 10 + sl[k].t0 / clip.dur * (w - 20);
        ctx.beginPath(); ctx.moveTo(x, 4); ctx.lineTo(x, h - 18); ctx.stroke();
      }
      ctx.fillStyle = C.muted; ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillText("0 s", 10, h - 4); ctx.fillText(`${clip.dur} s`, w - 44, h - 4);
      ctx.fillText(`${L.fmt(clip.n)} events, ${Tn} slices of about ${L.fmt(Math.round(clip.n / Tn))} each`, 60, h - 4);
      thumbs.innerHTML = "";
      sl.forEach((f, k) => {
        const wrap = document.createElement("div");
        const c = document.createElement("canvas");
        const on = G.resample(f.on, 32), off = G.resample(f.off, 32);
        G.paint(c, on.map(v => (v > 0 ? 1 : 0)), off.map(v => (v > 0 ? 1 : 0)), 32, 1);
        c.title = `frame ${k + 1}`;
        wrap.appendChild(c);
        const lab = document.createElement("div"); lab.className = "th"; lab.textContent = k + 1;
        wrap.appendChild(lab);
        thumbs.appendChild(wrap);
      });
    }
    L.seg(document.getElementById("frames-T"), v => { Tn = +v; draw(); });
    listeners.push(draw);
    window.addEventListener("resize", draw);
  }

  /* ---------- B.2 resize ---------- */
  function resize() {
    const big = document.getElementById("rs-big"), small = document.getElementById("rs-small"), bar = document.getElementById("rs-bar");
    if (!big) return;
    let N = 40, bits = 1;
    function draw() {
      const clip = G.makeClip(kind), f = G.slices(clip, 16)[7];
      G.paint(big, f.on.map(v => (v > 0 ? 1 : 0)), f.off.map(v => (v > 0 ? 1 : 0)), RES, 1);
      const top = 2 ** bits - 1, on = G.resample(f.on, N), off = G.resample(f.off, N);
      G.paint(small, on.map(v => Math.min(top, v)), off.map(v => Math.min(top, v)), N, top);
      document.getElementById("rs-small-cap").textContent = `after shrinking: ${N} × ${N}${bits > 1 ? `, counts up to ${top}` : ""}`;
      const axons = N * N * 2 * bits, lim = 16000, fits = axons <= lim;
      const largest = Math.floor(Math.sqrt(lim / (2 * bits)));
      const s = L.svg(bar, 760, 82, "Input axons used against the chip's limit");
      const X = v => 10 + Math.min(1, v / 34000) * 740;
      S("rect", { x: 10, y: 18, width: 740, height: 18, fill: "rgba(255,255,255,.05)" }, s);
      S("rect", { x: 10, y: 18, width: X(axons) - 10, height: 18, fill: fits ? C.green : C.red }, s);
      S("line", { x1: X(lim), y1: 10, x2: X(lim), y2: 44, stroke: C.accent, "stroke-width": 2 }, s);
      T(s, X(lim) + 4, 14, "limit 16,000", "lt-a");
      T(s, 10, 60, `${N} × ${N} pixels × 2 channels × ${bits} bit${bits > 1 ? "s" : ""} = ${L.fmt(axons)} axons: ${fits ? "fits" : "does not fit"}`, fits ? "lt-g" : "lt-bad");
      T(s, 10, 76, `largest square that fits with ${bits} bit${bits > 1 ? "s" : ""} per pixel: ${largest} × ${largest}`, "lt-ms");
    }
    L.range("rs-size", v => { N = v; draw(); });
    L.seg(document.getElementById("rs-bits"), v => { bits = +v; draw(); });
    listeners.push(draw);
  }

  /* ---------- B.3 split ---------- */
  function split() {
    const box = document.getElementById("split-vis");
    if (!box) return;
    const r = L.rng(29), people = [...Array(29)].map((_, i) => i);
    const pool = people.slice(0, 23), test = new Set(people.slice(23));
    const order = pool.slice().sort(() => r() - 0.5), val = new Set(order.slice(0, 4));
    const dotsVal = [...Array(29)].map(() => [...Array(12)].map(() => r() < 216 / 1176));
    const NOTE = {
      person: "Whole people are held out. Every validation score asks the question the test set asks: how well does the network handle someone it has never seen?",
      clip: "Split by clip instead, and every training person lands on both sides. The network can score well on validation by recognizing each person's style rather than the gesture, and the validation score stops predicting the test score."
    };
    function draw(mode) {
      const s = L.svg(box, 760, 300, "Twenty-nine people split into training, validation and test");
      const col = role => role === "test" ? C.rust : role === "val" ? C.accent : C.s4;
      people.forEach(i => {
        const row = Math.floor(i / 10), c = i % 10, x = 30 + c * 72, y = 30 + row * 90;
        const role = test.has(i) ? "test" : (mode === "person" && val.has(i)) ? "val" : "train";
        const icon = mode === "clip" && !test.has(i) ? C.muted : col(role);
        S("circle", { cx: x + 18, cy: y + 8, r: 7, fill: icon }, s);
        S("path", { d: `M${x + 6},${y + 36} Q${x + 6},${y + 18} ${x + 18},${y + 18} Q${x + 30},${y + 18} ${x + 30},${y + 36} Z`, fill: icon }, s);
        for (let d = 0; d < 12; d++) {
          const dr = role === "test" ? "test" : mode === "clip" ? (dotsVal[i][d] ? "val" : "train") : role;
          S("circle", { cx: x + 4 + (d % 6) * 5.6, cy: y + 46 + Math.floor(d / 6) * 6, r: 2, fill: col(dr) }, s);
        }
      });
      const ly = 290;
      [["training", C.s4], ["validation", C.accent], ["test", C.rust]].forEach(([t, c], k) => {
        S("rect", { x: 30 + k * 150, y: ly - 9, width: 10, height: 10, fill: c }, s);
        T(s, 46 + k * 150, ly, t, "lt");
      });
      document.getElementById("split-readout").textContent = mode === "person"
        ? "training: 19 people, 960 clips · validation: 4 people, 216 clips · test: 6 people, 288 clips"
        : "training and validation: the same 23 people · test: 6 people, 288 clips";
      document.getElementById("split-note").textContent = NOTE[mode];
    }
    L.seg(document.getElementById("split-mode"), draw);
  }

  /* ---------- B.4 overfitting ---------- */
  function overfit() {
    const canvas = document.getElementById("overfit-canvas");
    if (!canvas) return;
    function draw() {
      const xs = TRAIN.map((_, i) => i + 1);
      const p = Site.linePlot(canvas, [
        { xs, ys: TRAIN, color: C.s1, label: "training accuracy", width: 2 },
        { xs, ys: VAL, color: C.accent, label: "validation accuracy", width: 2 }
      ], { height: 300, ymin: 0.4, ymax: 1.0, xmin: 0, xmax: 80, xlabel: "epoch", ylabel: "accuracy", padL: 58, legend: false });
      const { ctx, X, Y } = p;
      ctx.font = "12px 'JetBrains Mono', monospace"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
      [["training accuracy", C.s1, 0.62], ["validation accuracy", C.accent, 0.56]].forEach(([t, c, y]) => {
        ctx.fillStyle = c; ctx.fillRect(X(46), Y(y) - 1.5, 14, 3);
        ctx.fillStyle = C.muted; ctx.fillText(t, X(46) + 22, Y(y));
      });
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.muted; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(20.5), Y(1.0)); ctx.lineTo(X(20.5), Y(0.4)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.muted; ctx.font = "12px 'JetBrains Mono', monospace"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      ctx.fillText("integer weights from here", X(21.5), Y(0.43));
      const x = X(78);
      ctx.strokeStyle = C.red; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x - 6, Y(0.997)); ctx.lineTo(x, Y(0.997)); ctx.lineTo(x, Y(0.745)); ctx.lineTo(x - 6, Y(0.745)); ctx.stroke();
      ctx.fillStyle = C.red; ctx.textAlign = "right";
      ctx.fillText("gap: about 25 points", x - 10, Y(0.87));
    }
    draw();
    window.addEventListener("resize", draw);
  }

  /* ---------- B.5 noise ---------- */
  function noise() {
    const canvas = document.getElementById("nz-canvas");
    if (!canvas) return;
    let p = 0.75, n = 216, hist = {}, total = 0, last = null;
    const r = L.rng(17);
    function drawOne() {
      let k = 0;
      for (let i = 0; i < n; i++) if (r() < p) k++;
      last = k / n;
      const key = Math.round(last * 1000) / 10;
      hist[key] = (hist[key] || 0) + 1; total++;
    }
    function draw() {
      const { ctx, w, h } = Site.setupCanvas(canvas, 240);
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
      const lo = 40, hi = 100, padL = 40, padR = 14, padB = 30, ph = h - padB - 14;
      const X = v => padL + (v - lo) / (hi - lo) * (w - padL - padR);
      const mx = Math.max(1, ...Object.values(hist));
      ctx.fillStyle = C.s4;
      Object.entries(hist).forEach(([k, c]) => { const bh = c / mx * ph; ctx.fillRect(X(+k) - 2, 14 + ph - bh, 4, bh); });
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.accent; ctx.beginPath(); ctx.moveTo(X(p * 100), 6); ctx.lineTo(X(p * 100), 14 + ph); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.muted; ctx.font = "12px 'JetBrains Mono', monospace"; ctx.textAlign = "center";
      for (let v = lo; v <= hi; v += 10) ctx.fillText(`${v}%`, X(v), h - 10);
      ctx.textAlign = "left"; ctx.fillText(`${total} sets drawn`, padL, 12);
      const sd = Math.sqrt(p * (1 - p) / n) * 100;
      document.getElementById("nz-last").textContent = last === null ? "-" : `${(last * 100).toFixed(1)}%`;
      document.getElementById("nz-range").textContent = `± ${(1.96 * sd).toFixed(1)} points`;
      document.getElementById("nz-clip").textContent = `${(100 / n).toFixed(2)} points`;
    }
    function reset() { hist = {}; total = 0; last = null; draw(); }
    L.range("nz-p", v => { p = v; reset(); }, v => v / 100);
    L.seg(document.getElementById("nz-n"), v => { n = +v; reset(); });
    document.getElementById("nz-one").addEventListener("click", () => { drawOne(); draw(); });
    document.getElementById("nz-many").addEventListener("click", () => { for (let i = 0; i < 200; i++) drawOne(); draw(); });
    for (let i = 0; i < 60; i++) drawOne();
    draw();
    window.addEventListener("resize", draw);
  }

  /* ---------- B.6 augmentation ---------- */
  function augment() {
    const grid = document.getElementById("aug-grid");
    if (!grid) return;
    const N = 48, CROP = [16, 30, 96];          // the square where the arms move
    let shift = 0.1, cut = 0.2, drop = 0.1, seed = 1;
    function base(k) {
      const sl = G.slices(G.makeClip(k), 16);
      return [6, 7, 8].map(i => ({
        on: G.resample(sl[i].on, N, CROP).map(v => (v > 0 ? 1 : 0)),
        off: G.resample(sl[i].off, N, CROP).map(v => (v > 0 ? 1 : 0))
      }));
    }
    function apply(fr, r) {
      const s = Math.round(shift * N), dx = s ? Math.round((r() * 2 - 1) * s) : 0, dy = s ? Math.round((r() * 2 - 1) * s) : 0;
      const side = Math.round(cut * N), cy = Math.floor(r() * N), cx = Math.floor(r() * N), half = Math.floor(side / 2);
      const out = fr.map(f => {
        const on = new Float32Array(N * N), off = new Float32Array(N * N);
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
          const sy = y - dy, sx = x - dx;
          if (sy < 0 || sx < 0 || sy >= N || sx >= N) continue;
          if (side && Math.abs(y - cy) <= half && Math.abs(x - cx) <= half) continue;
          const i = y * N + x, j = sy * N + sx;
          if (f.on[j] && r() >= drop) on[i] = 1;
          if (f.off[j] && r() >= drop) off[i] = 1;
        }
        return { on, off };
      });
      const parts = [];
      if (s) parts.push(`shift (${dx >= 0 ? "+" : ""}${dx}, ${dy >= 0 ? "+" : ""}${dy}) px`);
      if (side) parts.push(`cutout ${side} × ${side}`);
      if (drop) parts.push(`drop ${Math.round(drop * 100)}%`);
      return { frames: out, caption: parts.join(" · ") || "no change" };
    }
    function strip(frs) {
      const c = document.createElement("canvas"), gap = 3, W = 3 * N + 2 * gap;
      c.width = W; c.height = N;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, N);
      frs.forEach((f, k) => {
        const t = document.createElement("canvas");
        G.paint(t, f.on, f.off, N, 1);
        ctx.drawImage(t, k * (N + gap), 0);
      });
      c.style.aspectRatio = `${W} / ${N}`;
      return c;
    }
    function fig(parent, canvas, caption) {
      const f = document.createElement("figure");
      f.appendChild(canvas);
      const cap = document.createElement("figcaption"); cap.textContent = caption;
      f.appendChild(cap);
      parent.appendChild(f);
    }
    function draw() {
      const fr = base(kind), r = L.rng(seed);
      grid.innerHTML = "";
      fig(grid, strip(fr), `original: ${G.NAMES[kind]}`);
      for (let k = 0; k < 5; k++) { const a = apply(fr, r); fig(grid, strip(a.frames), a.caption); }
    }
    function flips() {
      const fl = document.getElementById("flip-grid");
      if (!fl) return;
      const f = base("rwave")[1];
      const mir = a => { const o = new Float32Array(N * N); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) o[y * N + x] = a[y * N + (N - 1 - x)]; return o; };
      fl.innerHTML = "";
      const c1 = document.createElement("canvas"), c2 = document.createElement("canvas");
      G.paint(c1, f.on, f.off, N, 1); G.paint(c2, mir(f.on), mir(f.off), N, 1);
      fig(fl, c1, "right hand wave");
      fig(fl, c2, "mirrored: now a left hand wave");
    }
    L.range("aug-shift", v => { shift = v; draw(); }, v => v / 100);
    L.range("aug-cut", v => { cut = v; draw(); }, v => v / 100);
    L.range("aug-drop", v => { drop = v; draw(); }, v => v / 100);
    document.getElementById("aug-new").addEventListener("click", () => { seed++; draw(); });
    listeners.push(draw);
    flips();
  }
})();
