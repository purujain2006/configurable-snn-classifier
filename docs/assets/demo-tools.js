/* demo-tools.js: page A's training tools (A.12 to A.15). Batch norm against
   drift, dropout on a small network, three optimizers in one valley, and what
   the rate loss and TET each score. */
(function () {
  "use strict";
  const L = window.Learn;
  const { S, T, C } = L;

  L.onReady(() => {
    [batchnorm, dropout, optimizers, losses].forEach(f => { try { f(); } catch (e) { console.error(f.name, e); } });
  });

  /* ---------- A.12 batch norm and tdBN ---------- */
  function batchnorm() {
    const box = document.getElementById("bn-vis");
    if (!box) return;
    const r = L.rng(12), Z = [...Array(400)].map(() => L.gauss(r));
    let drift = 2, theta = 1, mode = "none";
    const W = 760, H = 250, x0 = 40, x1 = 720, y0 = 20, y1 = 200, LO = -4, HI = 4, BINS = 40;
    const X = v => x0 + (x1 - x0) * (v - LO) / (HI - LO);
    function values() {
      if (mode === "bn") return Z.slice();                      // average 0, spread 1, whatever the drift
      if (mode === "tdbn") return Z.map(z => z * theta);        // spread follows the threshold
      const m = 0.8 * drift, sd = Math.exp(0.35 * drift);
      return Z.map(z => m + sd * z);
    }
    function draw() {
      const v = values(), bins = Array(BINS).fill(0);
      v.forEach(x => { bins[L.clamp(Math.floor((x - LO) / (HI - LO) * BINS), 0, BINS - 1)]++; });
      const top = Math.max(30, ...bins), bw = (x1 - x0) / BINS;
      const s = L.svg(box, W, H, "How one channel's values are spread, against the threshold");
      S("line", { x1: x0, x2: x1, y1: y1, y2: y1, stroke: C.lineStrong }, s);
      bins.forEach((n, i) => {
        if (!n) return;
        const h = (y1 - y0) * n / top, mid = LO + (i + 0.5) * (HI - LO) / BINS;
        S("rect", { x: x0 + i * bw + 1, y: y1 - h, width: bw - 2, height: h, fill: mid > theta ? C.accent : C.s4, rx: 1.5 }, s);
      });
      [-4, -2, 0, 2, 4].forEach(t => T(s, X(t), y1 + 18, t === -4 ? "−4 or less" : t === 4 ? "4 or more" : String(t).replace("-", "−"), "lt-ms mid"));
      T(s, (x0 + x1) / 2, H - 6, "value arriving at the neuron", "lt-ms mid");
      S("line", { x1: X(theta), x2: X(theta), y1: y0 - 6, y2: y1, stroke: C.accent, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, s);
      L.plate(T(s, X(theta) + 6, y0 + 4, `θ = ${theta.toFixed(2)}`, "lt-a"));
      const above = v.filter(x => x > theta).length / v.length;
      const what = above < 0.02 ? "the neuron almost never fires" : above > 0.6 ? "the neuron fires on most steps" : "the neuron fires some of the time";
      document.getElementById("bn-readout").textContent = `${Math.round(100 * above)}% of values above θ, so ${what}`;
    }
    L.range("bn-drift", d => { drift = d; draw(); });
    L.range("bn-theta", t => { theta = t; draw(); });
    L.seg(document.getElementById("bn-mode"), m => { mode = m; draw(); });
  }

  /* ---------- A.13 dropout ---------- */
  function dropout() {
    const box = document.getElementById("dropout-vis");
    if (!box) return;
    const sizes = [5, 7, 7, 3], W = 760, H = 260;
    const xs = sizes.map((_, i) => 110 + i * 180);
    const pos = sizes.map((n, i) => [...Array(n)].map((_, j) => [xs[i], H / 2 + (j - (n - 1) / 2) * 32]));
    const r = L.rng(7);
    let mode = "train", step = 0, off = new Set();
    function pick() {
      off = new Set();
      [1, 2].forEach(l => sizes[l] && [...Array(sizes[l])].forEach((_, j) => { if (r() < 0.3) off.add(`${l},${j}`); }));
    }
    function draw() {
      const s = L.svg(box, W, H, "A small network with some hidden units switched off");
      const isOff = (l, j) => mode === "train" && off.has(`${l},${j}`);
      for (let l = 0; l < sizes.length - 1; l++) pos[l].forEach((a, i) => pos[l + 1].forEach((b, j) => {
        const dead = isOff(l, i) || isOff(l + 1, j);
        S("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: dead ? C.lineSoft : Site.a(C.accent, 0.45), "stroke-width": dead ? 0.6 : 1.1 }, s);
      }));
      pos.forEach((col, l) => col.forEach(([x, y], j) => {
        const dead = isOff(l, j);
        S("circle", { cx: x, cy: y, r: 11, fill: dead ? C.bg : l === 0 ? C.s4 : l === 3 ? C.accent : Site.a(C.accent, 0.35), stroke: dead ? C.lineStrong : C.accent, "stroke-width": 1.2 }, s);
        if (dead) {
          S("line", { x1: x - 6, y1: y - 6, x2: x + 6, y2: y + 6, stroke: C.red, "stroke-width": 1.6 }, s);
          S("line", { x1: x + 6, y1: y - 6, x2: x - 6, y2: y + 6, stroke: C.red, "stroke-width": 1.6 }, s);
        }
      }));
      ["input", "hidden", "hidden", "output"].forEach((t, i) => T(s, xs[i], 22, t, "lt-ms mid"));
      const n = off.size;
      document.getElementById("dropout-readout").textContent = mode === "train"
        ? `training step ${step}: ${n} of 14 hidden units off`
        : "after training: all 14 hidden units on";
    }
    let acc = 0;
    const ctl = L.loop(box, dt => {
      if (mode !== "train") return false;
      acc += dt;
      if (acc > 0.9) { acc = 0; step++; pick(); draw(); }
    }, { button: document.getElementById("dropout-play"), autoplay: false });
    L.seg(document.getElementById("dropout-mode"), m => { mode = m; if (m === "train") { if (!step) { step = 1; pick(); } } else ctl.pause(); draw(); });
  }

  /* ---------- A.14 three optimizers ----------
     loss = (a·u² + b·v²) / 2 in valley coordinates (u along the valley, v
     across it), turned 20 degrees on screen. Steep across, gentle along. */
  function optimizers() {
    const box = document.getElementById("optim-vis");
    if (!box) return;
    const A = 0.3, B = 25, STEPS = 40, ROT = 20 * Math.PI / 180;
    const start = [-3.2, 0.55];
    const loss = ([u, v]) => 0.5 * (A * u * u + B * v * v);
    const grad = ([u, v]) => [A * u, B * v];
    function run(kind) {
      let p = start.slice(), m = [0, 0], q = [0, 0];
      const path = [p.slice()];
      for (let t = 1; t <= STEPS; t++) {
        const g = grad(p);
        if (kind === "gd") p = [p[0] - 0.075 * g[0], p[1] - 0.075 * g[1]];
        else if (kind === "mom") { m = [0.9 * m[0] + g[0], 0.9 * m[1] + g[1]]; p = [p[0] - 0.02 * m[0], p[1] - 0.02 * m[1]]; }
        else {
          m = [0.9 * m[0] + 0.1 * g[0], 0.9 * m[1] + 0.1 * g[1]];
          q = [0.999 * q[0] + 0.001 * g[0] ** 2, 0.999 * q[1] + 0.001 * g[1] ** 2];
          const mh = m.map(x => x / (1 - 0.9 ** t)), qh = q.map(x => x / (1 - 0.999 ** t));
          p = [p[0] - 0.15 * mh[0] / (Math.sqrt(qh[0]) + 1e-8), p[1] - 0.15 * mh[1] / (Math.sqrt(qh[1]) + 1e-8)];
        }
        path.push(p.slice());
      }
      return path;
    }
    const RUNS = [["gd", "gradient descent", C.s1], ["mom", "momentum", C.rust], ["adam", "Adam", C.accent]].map(([k, n, c]) => ({ k, n, c, path: run(k) }));
    const W = 760, H = 300, cx = 400, cy = 150, SC = 64;
    const toScreen = ([u, v]) => [cx + SC * (u * Math.cos(ROT) - v * Math.sin(ROT)), cy + SC * (u * Math.sin(ROT) + v * Math.cos(ROT))];
    let shown = 0;
    function draw() {
      const s = L.svg(box, W, H, "Three optimizers stepping down a narrow valley");
      // shading: equal-loss ellipses, stronger toward the bottom
      [24, 12, 6, 3, 1.5, 0.75, 0.3].forEach((lv, i) => {
        const ru = Math.sqrt(2 * lv / A) * SC, rv = Math.sqrt(2 * lv / B) * SC;
        S("ellipse", { cx, cy, rx: ru, ry: rv, transform: `rotate(${ROT * 180 / Math.PI} ${cx} ${cy})`, fill: Site.a(C.ink, 0.03 + 0.012 * i), stroke: C.lineSoft }, s);
      });
      S("circle", { cx, cy, r: 4, fill: C.green }, s);
      RUNS.forEach(run => {
        const pts = run.path.slice(0, shown + 1).map(toScreen);
        S("polyline", { points: pts.map(p => p.join(",")).join(" "), fill: "none", stroke: run.c, "stroke-width": 2 }, s);
        pts.forEach((p, i) => S("circle", { cx: p[0], cy: p[1], r: i === pts.length - 1 ? 4.5 : 2.2, fill: run.c }, s));
      });
      const [sx, sy] = toScreen(start);
      L.plate(T(s, sx - 10, sy - 12, "start", "lt-ms end"));
      RUNS.forEach((run, i) => {
        S("rect", { x: 24, y: 18 + i * 20, width: 14, height: 4, fill: run.c }, s);
        T(s, 46, 24 + i * 20, run.n, "lt");
      });
      document.getElementById("optim-readout").textContent = `step ${shown} of ${STEPS} · loss: ` +
        RUNS.map(run => `${run.n} ${loss(run.path[shown]).toFixed(2)}`).join(" · ");
    }
    let acc = 0;
    const ctl = L.loop(box, dt => {
      acc += dt * 10;
      while (acc >= 1 && shown < STEPS) { acc -= 1; shown++; }
      draw();
      return shown < STEPS;
    }, { button: document.getElementById("optim-play"), autoplay: false });
    document.getElementById("optim-restart").addEventListener("click", () => { shown = 0; acc = 0; draw(); ctl.play(); });
    draw();
  }

  /* ---------- A.15 rate loss and TET ---------- */
  function losses() {
    const box = document.getElementById("tet-vis");
    if (!box) return;
    const STEPS = 12, NAMES = ["gesture 1", "gesture 2", "gesture 3"];
    let seed = 4, mode = "rate", spikes;
    function clip() {
      const r = L.rng(seed);
      // the right neuron (gesture 1) wakes up late; gesture 2 fires early
      spikes = [0, 1, 2].map(c => [...Array(STEPS)].map((_, t) => {
        const p = c === 0 ? (t < 5 ? 0.15 : 0.85) : c === 1 ? (t < 5 ? 0.75 : 0.15) : 0.12;
        return r() < p ? 1 : 0;
      }));
    }
    const softmax = xs => { const m = Math.max(...xs), e = xs.map(x => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0); return e.map(x => x / z); };
    function draw() {
      const W = 760, H = 250, x0 = 120, cw = 34, y0 = 34, rh = 34;
      const s = L.svg(box, W, H, "Three output neurons over twelve steps, and what each loss scores");
      const counts = spikes.map(row => row.reduce((a, b) => a + b, 0));
      spikes.forEach((row, c) => {
        T(s, x0 - 12, y0 + c * rh + 20, NAMES[c], c === 0 ? "lt-b end" : "lt end");
        row.forEach((sp, t) => S("rect", { x: x0 + t * cw + 2, y: y0 + c * rh + 4, width: cw - 4, height: rh - 8, rx: 2, fill: sp ? (c === 0 ? C.accent : C.s4) : C.cell }, s));
        const tx = x0 + STEPS * cw + 30;
        S("rect", { x: tx, y: y0 + c * rh + 8, width: counts[c] * 12, height: rh - 16, fill: c === 0 ? C.accent : C.s4, rx: 2 }, s);
        T(s, tx + counts[c] * 12 + 8, y0 + c * rh + 22, String(counts[c]), "lt-m");
      });
      for (let t = 0; t < STEPS; t++) T(s, x0 + t * cw + cw / 2, y0 - 10, String(t + 1), "lt-ms mid");
      T(s, x0 - 12, y0 - 10, "step", "lt-ms end");
      T(s, x0 + STEPS * cw + 30, y0 - 10, "total spikes", "lt-ms");
      const by = y0 + 3 * rh + 18;
      if (mode === "rate") {
        S("rect", { x: x0 + STEPS * cw + 22, y: y0 - 2, width: 150, height: 3 * rh + 4, fill: "none", stroke: C.accent, "stroke-width": 1.6, rx: 3 }, s);
        const p = softmax(counts.map(n => n * 0.6)), l = -Math.log(p[0]);
        T(s, x0, by + 26, `Only the totals are scored. Gesture 1 gets probability ${p[0].toFixed(2)}, so the loss is ${l.toFixed(2)}.`, "lt");
        document.getElementById("tet-readout").textContent = `rate loss ${l.toFixed(2)} · the early steps, where gesture 2 leads, cost nothing as long as the total is right`;
      } else {
        const per = [...Array(STEPS)].map((_, t) => -Math.log(softmax(spikes.map(row => row[t] * 2.5))[0]));
        const mx = Math.max(...per, 1);
        per.forEach((l, t) => {
          S("rect", { x: x0 + t * cw + 2, y: y0 - 2, width: cw - 4, height: 3 * rh + 4, fill: "none", stroke: Site.a(C.accent, 0.7), rx: 3 }, s);
          const h = 46 * l / mx;
          S("rect", { x: x0 + t * cw + 6, y: by + 50 - h, width: cw - 12, height: h, fill: l > 1 ? C.red : Site.a(C.accent, 0.6), rx: 1.5 }, s);
        });
        const avg = per.reduce((a, b) => a + b, 0) / STEPS;
        T(s, x0 - 12, by + 46, "loss per step", "lt-ms end");
        document.getElementById("tet-readout").textContent = `TET loss ${avg.toFixed(2)}, the average of 12 step losses · the early steps, where gesture 2 leads, are what it pushes on`;
      }
    }
    clip();
    L.seg(document.getElementById("tet-mode"), m => { mode = m; draw(); });
    document.getElementById("tet-new").addEventListener("click", () => { seed++; clip(); draw(); });
  }
})();
