/* demo-pipeline.js: visuals for page C (the pipeline and the search). */
(function () {
  "use strict";
  const L = window.Learn;
  const { S, T, C } = L;

  L.onReady(() => {
    [pipe, phases, sim, funnel, block, tpe, asha, finalists, timeline]
      .forEach(f => { try { f(); } catch (e) { console.error(f.name, e); } });
  });

  function arrow(g, x1, y1, x2, y2, color, w, dash) {
    S("line", { x1, y1, x2, y2, stroke: color || C.faint, "stroke-width": w || 1.4, "stroke-dasharray": dash || null }, g);
    const a = Math.atan2(y2 - y1, x2 - x1), s = 7;
    S("path", { d: `M${x2},${y2} L${x2 - s * Math.cos(a - 0.4)},${y2 - s * Math.sin(a - 0.4)} L${x2 - s * Math.cos(a + 0.4)},${y2 - s * Math.sin(a + 0.4)} Z`, fill: color || C.faint }, g);
  }
  function box(g, x, y, w, h, lines, opts) {
    opts = opts || {};
    const r = S("rect", { x, y, width: w, height: h, fill: opts.fill || C.panel, stroke: opts.stroke || C.line, "stroke-width": opts.sw || 1 }, g);
    const n = lines.length, lh = 17;
    lines.forEach((t, i) => T(g, x + w / 2, y + h / 2 - (n - 1) * lh / 2 + i * lh + 5, t, (opts.cls || "lt-b") + " mid"));
    return r;
  }

  /* ---------- C.1 the pipeline ---------- */
  const STAGES = [
    { t: ["Propose a", "configuration"], title: "Propose a configuration",
      text: "The search picks one value for every setting: how many layers and how wide, how each layer shrinks the map, how the input is encoded, how the neurons behave, and how training runs. The first 96 trials are a designed, balanced set; after that, a sampler proposes each one from what has scored well.",
      why: "Settings change how much the others matter, so they are chosen together as one configuration." },
    { t: ["Check it fits", "the chip"], title: "Check that it fits the chip",
      text: "Before anything is trained, the network's sizes are worked out and compared with the chip's limits: 16,000 input axons, and caps on how many connections each neuron may receive and send. A shape that fails ends the trial with a score of zero.",
      why: "A network the chip cannot hold is useless however accurate it is, and this check takes microseconds." },
    { t: ["Recordings", "to frames"], title: "Turn recordings into frames",
      text: "Every recording is cut into T frames with equal numbers of events, shrunk to the chosen size, and encoded with the chosen number of bits per pixel.",
      why: "The network reads one frame per timestep, and the frame size and bits decide how many of the chip's input axons are used." },
    { t: ["Warm up with", "ordinary weights"], title: "Warm up with ordinary weights",
      text: "For the first quarter of training, 20 of 80 epochs, the weights are ordinary floating-point numbers and batch normalization keeps each layer's activity in a healthy range.",
      why: "Batch normalization needs real statistics of the data before it can be folded into the weights." },
    { t: ["Fold and round", "to integers"], title: "Fold and round to the chip's integers",
      text: "Batch normalization is folded into the convolution weights and the neurons' thresholds, and every weight is rounded to one of the 16-bit integer values the chip stores.",
      why: "The chip has no batch normalization and no floating point. From here on, the network being trained is the network that would deploy." },
    { t: ["Train on the", "integer grid"], title: "Keep training on the integer grid",
      text: "The remaining 60 epochs train with the integer weights in place, at half the learning rate. The network learns to work with exactly the weights it will deploy with.",
      why: "Rounding only at the end would break the precise balances training relies on; training on the grid avoids that." },
    { t: ["Checkpoint", "at epoch 30"], title: "Early-stopping checkpoint at epoch 30",
      text: "At epoch 30 each trial is ranked against every other trial that has reached epoch 30. Only the best third continue to epoch 80; the rest stop and free their slot. Trials in the designed block always run to the end.",
      why: "Most trials will not win, and it is usually clear by then. The checkpoint comes ten epochs after the switch to integers, once accuracy has recovered from the switch." },
    { t: ["Export to the", "chip's format"], title: "Export to the chip's format",
      text: "Each layer is written out as the chip stores it: neurons, connections, integer weights, integer thresholds and leak settings.",
      why: "This is the network the chip would load, so it is what gets scored." },
    { t: ["Check on a", "chip simulator"], title: "Check it on a simulator of the chip",
      text: "Every validation clip runs through a simulator that uses the chip's exact integer arithmetic, and its answers are compared with the trained network's. A network the simulator cannot run is marked not deployable.",
      why: "It confirms that what was trained is what the chip would actually do." },
    { t: ["Score on held-", "out people"], title: "Score the deployed network",
      text: "The trial's score is the deployed network's accuracy on the validation people, who were never trained on. The search records it, learns from it, and starts the next trial.",
      why: "Scoring the deployed network rather than the training-time one means the search optimizes exactly what runs on the chip." },
    { t: ["Finalists:", "retrain the top 5"], title: "After the last trial: finalists",
      text: "The five best configurations are trained again from scratch with three new random seeds each. The configuration with the best average wins, and only then is its accuracy on the test people measured, once.",
      why: "One lucky run should not decide the winner, and the test set stays an honest final check." }
  ];
  function pipe() {
    const boxEl = document.getElementById("pipe-vis");
    if (!boxEl) return;
    const W = 180, Hh = 64;
    const P = [[40, 40], [260, 40], [480, 40], [700, 40], [700, 180], [480, 180], [260, 180], [40, 180], [40, 320], [260, 320], [700, 320]];
    const s = L.svg(boxEl, 920, 430, "The stages one trial goes through");
    const gA = S("g", {}, s), gB = S("g", {}, s);
    const ctr = i => [P[i][0] + W / 2, P[i][1] + Hh / 2];
    // arrows between stages
    const pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]];
    pairs.forEach(([a, b]) => {
      const [ax, ay] = P[a], [bx, by] = P[b];
      if (ay === by) {
        if (bx > ax) arrow(gA, ax + W, ay + Hh / 2, bx - 2, by + Hh / 2);
        else arrow(gA, ax, ay + Hh / 2, bx + W + 2, by + Hh / 2);
      } else arrow(gA, ax + W / 2, ay + Hh, bx + W / 2, by - 2);
    });
    // loop back to the start
    S("path", { d: `M${P[9][0] + W / 2},${P[9][1] + Hh} V412 H14 V${P[0][1] + Hh / 2} H${P[0][0] - 4}`, fill: "none", stroke: C.accent, "stroke-opacity": .55, "stroke-width": 1.4 }, gA);
    arrow(gA, 30, P[0][1] + Hh / 2, P[0][0] - 2, P[0][1] + Hh / 2, C.accent, 1.4);
    T(gA, 26, 250, "next trial", "lt-a", { transform: "rotate(-90 26 250)", "text-anchor": "middle" });
    // finalists branch
    arrow(gA, P[9][0] + W, P[9][1] + Hh / 2, P[10][0] - 2, P[10][1] + Hh / 2, C.faint, 1.2, "5 4");
    T(gA, (P[9][0] + W + P[10][0]) / 2, P[9][1] + Hh / 2 - 8, "after the last trial", "lt-s mid");
    // failure exits
    S("line", { x1: P[1][0] + W / 2, y1: P[1][1] + Hh, x2: P[1][0] + W / 2, y2: 122, stroke: C.red }, gA);
    S("rect", { x: P[1][0] + 10, y: 122, width: W - 20, height: 26, fill: "rgba(207,107,88,.12)", stroke: C.red }, gA);
    T(gA, P[1][0] + W / 2, 139, "fails: score 0", "lt-bad mid");
    S("line", { x1: P[6][0] + W / 2, y1: P[6][1] + Hh, x2: P[6][0] + W / 2, y2: 262, stroke: C.red }, gA);
    S("rect", { x: P[6][0] + 10, y: 262, width: W - 20, height: 26, fill: "rgba(207,107,88,.12)", stroke: C.red }, gA);
    T(gA, P[6][0] + W / 2, 279, "not top third: stop", "lt-bad mid");
    // stage boxes
    const rects = STAGES.map((st, i) => {
      const g = S("g", { style: "cursor:pointer" }, gB);
      const r = box(g, P[i][0], P[i][1], W, Hh, st.t);
      T(g, P[i][0] + 8, P[i][1] + 14, String(i + 1).padStart(2, "0"), "lt-ms");
      g.addEventListener("click", () => { ctrl.pause(); go(i); });
      return r;
    });
    const token = S("circle", { r: 7, fill: C.accent, opacity: 0.9 }, s);
    const cap = document.getElementById("pipe-cap");
    let cur = 0, from = 0, tAnim = 1, hold = 0;
    function show() {
      rects.forEach((r, i) => {
        r.setAttribute("stroke", i === cur ? C.accent : C.line);
        r.setAttribute("stroke-width", i === cur ? 2 : 1);
        r.setAttribute("fill", i === cur ? "rgba(224,169,79,.12)" : C.panel);
      });
      const st = STAGES[cur];
      cap.innerHTML = `<div class="sc-num">STAGE ${String(cur + 1).padStart(2, "0")} OF ${STAGES.length}</div><div class="sc-title">${st.title}</div><p>${st.text}</p><p class="sc-why"><b>Why:</b> ${st.why}</p>`;
    }
    function place() {
      const [x1, y1] = ctr(from), [x2, y2] = ctr(cur), k = Math.min(1, tAnim);
      const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      token.setAttribute("cx", L.lerp(x1, x2, e) + 70);
      token.setAttribute("cy", L.lerp(y1, y2, e) - 20);
    }
    function go(i) {
      from = cur; cur = (i + STAGES.length) % STAGES.length; hold = 0;
      tAnim = ctrl && ctrl.playing ? 0 : 1;      // jump straight there when paused
      show(); place();
    }
    const ctrl = L.loop(boxEl, dt => {
      if (tAnim < 1) { tAnim += dt / 0.6; place(); return true; }
      hold += dt;
      if (hold > 4.2) go(cur === 9 ? 10 : cur === 10 ? 0 : cur + 1);
      return true;
    }, { button: document.getElementById("pipe-play"), autoplay: false });
    document.getElementById("pipe-next").addEventListener("click", () => { ctrl.pause(); go(cur + 1); });
    document.getElementById("pipe-prev").addEventListener("click", () => { ctrl.pause(); go(cur - 1); });
    show(); place();
  }

  /* ---------- C.3 phases ---------- */
  function phases() {
    const boxEl = document.getElementById("phases-vis");
    if (!boxEl) return;
    const s = L.svg(boxEl, 780, 250, "Weights before and after being snapped to the chip's integer grid");
    const x0 = 50, x1 = 740, X = w => x0 + (w + 1) / 2 * (x1 - x0);
    const E = 80, SW = 20, GRID = 8;
    const TX = e => x0 + e / E * (x1 - x0);
    S("rect", { x: TX(0), y: 20, width: TX(SW) - TX(0), height: 16, fill: "rgba(236,231,222,.12)" }, s);
    S("rect", { x: TX(SW), y: 20, width: TX(E) - TX(SW), height: 16, fill: "rgba(224,169,79,.18)" }, s);
    T(s, TX(SW / 2), 52, "ordinary weights", "lt-s mid");
    T(s, TX((SW + E) / 2), 52, "integer weights", "lt-a mid");
    T(s, x0 - 8, 33, "epoch", "lt-s end");
    const cursor = S("line", { y1: 14, y2: 42, stroke: C.text, "stroke-width": 2 }, s);
    const gGrid = S("g", {}, s);
    for (let k = -GRID; k <= GRID; k++) S("line", { x1: X(k / GRID), x2: X(k / GRID), y1: 76, y2: 222, stroke: C.accent, "stroke-opacity": 0.12 }, gGrid);
    S("line", { x1: x0, x2: x1, y1: 228, y2: 228, stroke: C.line }, s);
    [-1, -0.5, 0, 0.5, 1].forEach(v => T(s, X(v), 244, v === 0 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v), "lt-ms mid"));
    const r = L.rng(3), n = 90;
    let w = [], ys = [], dots = [], epoch = 0, snapK = 0, snapped = [];
    function init() {
      w = [...Array(n)].map(() => L.clamp(L.gauss(r) * 0.32, -0.95, 0.95));
      ys = [...Array(n)].map(() => 84 + r() * 132);
      dots.forEach(d => d.remove());
      dots = w.map((v, i) => S("circle", { cx: X(v), cy: ys[i], r: 3.4, fill: C.s1, "fill-opacity": .85 }, s));
      epoch = 0; snapK = 0; snapped = [];
    }
    const q = v => Math.round(v * GRID) / GRID;
    function render() {
      cursor.setAttribute("x1", TX(epoch)); cursor.setAttribute("x2", TX(epoch));
      const on = epoch >= SW;
      gGrid.querySelectorAll("line").forEach(l => l.setAttribute("stroke-opacity", on ? 0.45 : 0.12));
      dots.forEach((d, i) => {
        const x = on ? L.lerp(w[i], snapped[i], Math.min(1, snapK)) : w[i];
        d.setAttribute("cx", X(x));
        d.setAttribute("fill", on && snapK >= 1 ? C.accent : C.s1);
      });
      document.getElementById("phases-readout").innerHTML = `epoch <b>${Math.floor(epoch)}</b> of 80 · ${on ? "integer weights: every weight sits on a storable value" : "ordinary weights: any value allowed"}`;
    }
    let acc = 0;
    const ctrl = L.loop(boxEl, dt => {
      if (epoch >= E) return false;
      if (epoch >= SW && snapK < 1) { snapK += dt / 1.0; render(); return true; }
      acc += dt * 8;
      while (acc >= 1) {
        acc -= 1; epoch++;
        if (epoch <= SW) w = w.map(v => L.clamp(v + 0.025 * L.gauss(r), -0.98, 0.98));
        if (epoch === SW) snapped = w.map(q);
        if (epoch > SW) snapped = snapped.map(v => r() < 0.04 ? L.clamp(v + (r() < 0.5 ? -1 : 1) / GRID, -1, 1) : v);
        if (epoch >= E) break;
      }
      render();
      return true;
    }, { button: document.getElementById("phases-play"), autoplay: false });
    document.getElementById("phases-restart").addEventListener("click", () => { init(); render(); ctrl.play(); });
    init(); render();
  }

  /* ---------- C.4 simulator (static) ---------- */
  function sim() {
    const boxEl = document.getElementById("sim-vis");
    if (!boxEl) return;
    const s = L.svg(boxEl, 840, 250, "Trained network and chip simulator produce the same spikes");
    const r = L.rng(8), n = 8, steps = 40;
    const sp = [...Array(n)].map((_, i) => [...Array(steps)].map((_, t) => r() < 0.1 + 0.25 * Math.sin((t + i * 3) / 5) ** 2));
    function raster(y0, label, color) {
      T(s, 20, y0 + 10, label, "lt-b");
      for (let i = 0; i < n; i++) for (let t = 0; t < steps; t++) {
        S("rect", { x: 200 + t * 14, y: y0 + 22 + i * 10, width: 12, height: 8, fill: sp[i][t] ? color : "rgba(255,255,255,.04)" }, s);
      }
      T(s, 20, y0 + 34, "8 neurons, 40 steps", "lt-s");
    }
    raster(10, "trained network", C.s1);
    raster(130, "chip simulator", C.accent);
    T(s, 20, 70, "answer: wave", "lt-m");
    T(s, 20, 190, "answer: wave", "lt-a");
  }

  /* ---------- C.6 shape first (static) ---------- */
  function funnel() {
    const boxEl = document.getElementById("funnel-vis");
    if (!boxEl) return;
    const s = L.svg(boxEl, 840, 210, "Shape is chosen and checked before training settings");
    box(s, 20, 30, 190, 70, ["1. choose the shape", "depth, widths, kernels,", "size, bits, head"], { cls: "lt" });
    arrow(s, 210, 65, 258, 65);
    S("path", { d: "M330,20 L400,65 L330,110 L260,65 Z", fill: C.panel, stroke: C.accent }, s);
    T(s, 330, 62, "fits the", "lt-b mid"); T(s, 330, 78, "chip?", "lt-b mid");
    arrow(s, 400, 65, 450, 65, C.green);
    T(s, 425, 56, "yes", "lt-g mid");
    box(s, 450, 30, 170, 70, ["2. choose training", "settings"]);
    arrow(s, 620, 65, 668, 65);
    box(s, 668, 30, 152, 70, ["3. train", "and score"]);
    arrow(s, 330, 110, 330, 150, C.red);
    T(s, 342, 134, "no", "lt-bad");
    S("rect", { x: 200, y: 150, width: 260, height: 44, fill: "rgba(207,107,88,.1)", stroke: C.red }, s);
    T(s, 330, 168, "score 0, charged to the shape only", "lt-bad mid");
    T(s, 330, 185, "training settings never chosen", "lt-s mid");
  }

  /* ---------- C.7 balanced block ---------- */
  function block() {
    const boxEl = document.getElementById("block-vis");
    if (!boxEl) return;
    const F = [["pooling", ["none", "avg", "max"]], ["bias", ["threshold", "neuron"]], ["head", ["flatten", "GAP"]],
      ["signed spikes", ["off", "on"]], ["input bits", ["1", "more"]], ["leak τ", ["fixed", "per layer", "per channel"]],
      ["loss", ["rate", "TET"]], ["augmentation", ["off", "on"]], ["conv dropout", ["off", "on"]], ["rate penalty", ["off", "on"]]];
    const cols = ["rgba(167,162,154,.55)", C.accent, C.rust];
    const r = L.rng(96), n = 96;
    const s = L.svg(boxEl, 900, 20 + F.length * 26 + 30, "Ninety-six designed trials with every option balanced");
    const lx = 250, cw = (900 - lx - 10) / n;
    F.forEach(([name, lv], row) => {
      const arr = [];
      for (let i = 0; i < n; i++) arr.push(i % lv.length);
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
      const y = 14 + row * 26;
      const t = S("text", { x: 10, y: y + 13, class: "lt" }, s);
      const a = S("tspan", { fill: C.text }, t); a.textContent = name + "  ";
      lv.forEach((v, k) => { const sp = S("tspan", { fill: k === 0 ? C.s4 : cols[k] }, t); sp.textContent = v + (k < lv.length - 1 ? " / " : ""); });
      arr.forEach((v, i) => S("rect", { x: lx + i * cw, y, width: Math.max(1, cw - 1), height: 18, fill: cols[v] }, s));
    });
    T(s, lx, 14 + F.length * 26 + 18, "trial 1", "lt-s");
    T(s, 890, 14 + F.length * 26 + 18, "trial 96", "lt-s end");
  }

  /* ---------- C.8 TPE ---------- */
  function tpe() {
    const canvas = document.getElementById("tpe-canvas");
    if (!canvas) return;
    const lo = Math.log10(5e-4), hi = Math.log10(5e-3), u0 = Math.log10(1.6e-3);
    const truth = u => 0.72 - 0.9 * (u - u0) * (u - u0);
    const BW = 0.07;
    let r, trials, next;
    function kde(pts, u) { if (!pts.length) return 1e-6; let s = 0; for (const p of pts) s += Math.exp(-0.5 * ((u - p) / BW) ** 2); return s / pts.length / (BW * 2.5066) + 0.05; }
    function split() {
      const sorted = trials.slice().sort((a, b) => b.s - a.s);
      const k = Math.min(25, Math.ceil(0.1 * trials.length));
      return { good: sorted.slice(0, k), rest: sorted.slice(k) };
    }
    function propose() {
      const { good, rest } = split(), gp = good.map(t => t.u), rp = rest.map(t => t.u);
      let best = null, bestv = -1;
      for (let i = 0; i < 24; i++) {
        const c = L.clamp(gp[Math.floor(r() * gp.length)] + L.gauss(r) * BW, lo, hi);
        const v = kde(gp, c) / kde(rp, c);
        if (v > bestv) { bestv = v; best = c; }
      }
      return best;
    }
    function run() { trials.push({ u: next, s: truth(next) + 0.025 * L.gauss(r) }); next = propose(); }
    function reset() {
      r = L.rng(42); trials = [];
      for (let i = 0; i < 12; i++) { const u = lo + r() * (hi - lo); trials.push({ u, s: truth(u) + 0.025 * L.gauss(r) }); }
      next = propose(); draw();
    }
    function draw() {
      const { ctx, w, h } = Site.setupCanvas(canvas, 320);
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
      const padL = 56, padR = 16, X = u => padL + (u - lo) / (hi - lo) * (w - padL - padR);
      const sy0 = 16, sh = 170, Y = v => sy0 + sh - (v - 0.4) / 0.4 * sh;
      ctx.strokeStyle = C.lineSoft; ctx.strokeRect(padL, sy0, w - padL - padR, sh);
      ctx.font = "11px 'JetBrains Mono', monospace"; ctx.fillStyle = C.muted; ctx.textAlign = "right";
      [0.4, 0.5, 0.6, 0.7, 0.8].forEach(v => ctx.fillText(v.toFixed(1), padL - 6, Y(v) + 4));
      ctx.save(); ctx.translate(14, sy0 + sh / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText("score", 0, 0); ctx.restore();
      ctx.setLineDash([2, 4]); ctx.strokeStyle = "rgba(236,231,222,.3)"; ctx.beginPath();
      for (let i = 0; i <= 100; i++) { const u = lo + i / 100 * (hi - lo); i ? ctx.lineTo(X(u), Y(truth(u))) : ctx.moveTo(X(u), Y(truth(u))); }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "rgba(236,231,222,.45)"; ctx.textAlign = "left"; ctx.fillText("hidden: true average score", X(lo) + 6, Y(0.43));
      const { good, rest } = split();
      rest.forEach(t => { ctx.fillStyle = C.s4; ctx.beginPath(); ctx.arc(X(t.u), Y(t.s), 3.5, 0, 7); ctx.fill(); });
      good.forEach(t => { ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(X(t.u), Y(t.s), 5, 0, 7); ctx.fill(); });
      const dy0 = sy0 + sh + 26, dh = 70;
      const gp = good.map(t => t.u), rp = rest.map(t => t.u);
      let mx = 0; const lv = [], gv = [];
      for (let i = 0; i <= 160; i++) { const u = lo + i / 160 * (hi - lo); lv.push(kde(gp, u)); gv.push(kde(rp, u)); mx = Math.max(mx, lv[i], gv[i]); }
      [[gv, C.s4], [lv, C.accent]].forEach(([arr, c]) => {
        ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath();
        arr.forEach((v, i) => { const x = X(lo + i / 160 * (hi - lo)), y = dy0 + dh - v / mx * dh; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.stroke();
      });
      ctx.lineWidth = 1;
      ctx.fillStyle = C.accent; ctx.fillText("where good trials sit", padL + 4, dy0 - 6);
      ctx.fillStyle = C.s4; ctx.fillText("where the rest sit", padL + 170, dy0 - 6);
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.text; ctx.beginPath(); ctx.moveTo(X(next), sy0); ctx.lineTo(X(next), dy0 + dh); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.text; ctx.textAlign = X(next) > w - 160 ? "right" : "left";
      ctx.fillText("next try", X(next) + (X(next) > w - 160 ? -6 : 6), sy0 + 12);
      ctx.fillStyle = C.muted; ctx.textAlign = "center";
      [5e-4, 1e-3, 2e-3, 5e-3].forEach(v => ctx.fillText(v.toExponential(0).replace("e-", "e−"), X(Math.log10(v)), h - 8));
      ctx.textAlign = "left"; ctx.fillText("learning rate (log scale)", padL, h - 22);
      const best = trials.reduce((a, b) => (b.s > a.s ? b : a));
      document.getElementById("tpe-readout").innerHTML = `${trials.length} trials · best <b>${best.s.toFixed(3)}</b> at ${Math.pow(10, best.u).toExponential(1)}`;
    }
    document.getElementById("tpe-one").addEventListener("click", () => { run(); draw(); });
    document.getElementById("tpe-ten").addEventListener("click", () => { for (let i = 0; i < 10; i++) run(); draw(); });
    document.getElementById("tpe-reset").addEventListener("click", reset);
    window.addEventListener("resize", draw);
    reset();
  }

  /* ---------- C.9 ASHA ---------- */
  function asha() {
    const canvas = document.getElementById("asha-canvas");
    if (!canvas) return;
    const n = 12, E = 80, RUNG = 30;
    const r = L.rng(30);
    const tr = [...Array(n)].map((_, i) => {
      const a = 0.45 + r() * 0.3, k = 6 + r() * 12;
      const ys = [];
      for (let e = 0; e <= E; e++) {
        let v = a * (1 - Math.exp(-e / k)) + 0.1;
        if (e > 20 && e < 28) v -= 0.05 * (1 - (e - 20) / 8);
        ys.push(L.clamp(v + 0.012 * L.gauss(r), 0.05, 0.95));
      }
      return { ys, block: i < 3 };
    });
    const at = tr.map(t => t.ys[RUNG]).sort((a, b) => b - a), cut = at[Math.ceil(n / 3) - 1];
    tr.forEach(t => { t.stop = !t.block && t.ys[RUNG] < cut; });
    let e = 0, geo = null;
    function draw() {
      if (!geo) geo = Site.setupCanvas(canvas, 300);
      const { ctx, w, h } = geo;
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
      const padL = 50, padB = 30, X = x => padL + x / E * (w - padL - 14), Y = v => 14 + (1 - v / 0.95) * (h - padB - 14);
      ctx.strokeStyle = C.lineSoft; ctx.strokeRect(padL, 14, w - padL - 14, h - padB - 14);
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.muted;
      ctx.beginPath(); ctx.moveTo(X(RUNG), 14); ctx.lineTo(X(RUNG), h - padB); ctx.stroke();
      ctx.strokeStyle = "rgba(236,231,222,.25)"; ctx.beginPath(); ctx.moveTo(X(20), 14); ctx.lineTo(X(20), h - padB); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "11px 'JetBrains Mono', monospace"; ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("checkpoint", X(RUNG) + 4, 26); ctx.fillText("switch to integers", X(20) - 128, 26);
      ctx.textAlign = "center";
      [0, 20, 40, 60, 80].forEach(v => ctx.fillText(v, X(v), h - 10));
      ctx.textAlign = "right"; [0.2, 0.4, 0.6, 0.8].forEach(v => ctx.fillText(v.toFixed(1), padL - 6, Y(v) + 4));
      tr.forEach(t => {
        const last = t.stop ? Math.min(e, RUNG) : e;
        ctx.strokeStyle = t.block ? "rgba(167,162,154,.8)" : (t.stop && e >= RUNG ? C.red : (e >= RUNG ? C.accent : C.s1));
        ctx.lineWidth = t.block ? 1.4 : 1.8; ctx.beginPath();
        for (let x = 0; x <= last; x++) { const px = X(x), py = Y(t.ys[x]); x ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
        ctx.stroke();
        if (t.stop && e >= RUNG) { const px = X(RUNG), py = Y(t.ys[RUNG]); ctx.strokeStyle = C.red; ctx.beginPath(); ctx.moveTo(px - 4, py - 4); ctx.lineTo(px + 4, py + 4); ctx.moveTo(px + 4, py - 4); ctx.lineTo(px - 4, py + 4); ctx.stroke(); }
      });
      const stopped = tr.filter(t => t.stop).length;
      document.getElementById("asha-readout").innerHTML = e < RUNG ? `epoch <b>${Math.floor(e)}</b>: every trial still training`
        : `epoch <b>${Math.floor(e)}</b>: ${stopped} of ${n - 3} sampled trials stopped at the checkpoint; their slots go to new trials`;
    }
    let acc = 0;
    const ctrl = L.loop(canvas, dt => { acc += dt * 12; while (acc >= 1 && e < E) { acc--; e++; } draw(); return e < E; },
      { button: document.getElementById("asha-play"), autoplay: false });
    document.getElementById("asha-restart").addEventListener("click", () => { e = 0; draw(); ctrl.play(); });
    window.addEventListener("resize", () => { geo = null; draw(); });
    e = E; draw();
  }

  /* ---------- C.10 finalists (static) ---------- */
  function finalists() {
    const boxEl = document.getElementById("final-vis");
    if (!boxEl) return;
    const D = [[0.745, [0.71, 0.722, 0.704]], [0.736, [0.731, 0.745, 0.727]], [0.722, [0.718, 0.713, 0.731]], [0.722, [0.70, 0.722, 0.708]], [0.722, [0.713, 0.727, 0.704]]];
    const s = L.svg(boxEl, 840, 270, "Finalists retrained with three seeds each");
    const Y = v => 220 - (v - 0.68) / 0.08 * 190, X = i => 140 + i * 150;
    [0.68, 0.70, 0.72, 0.74, 0.76].forEach(v => { S("line", { x1: 90, x2: 820, y1: Y(v), y2: Y(v), stroke: C.lineSoft }, s); T(s, 82, Y(v) + 4, `${Math.round(v * 100)}%`, "lt-ms end"); });
    const means = D.map(([, sd]) => sd.reduce((a, b) => a + b) / 3), best = means.indexOf(Math.max(...means));
    D.forEach(([srch, sd], i) => {
      const x = X(i), m = means[i];
      S("path", { d: `M${x - 30},${Y(srch)} l6,-6 l6,6 l-6,6 Z`, fill: "none", stroke: C.text, "stroke-width": 1.4 }, s);
      sd.forEach((v, k) => S("circle", { cx: x + 2 + k * 12, cy: Y(v), r: 4.5, fill: i === best ? C.accent : C.s4 }, s));
      S("line", { x1: x - 4, x2: x + 32, y1: Y(m), y2: Y(m), stroke: i === best ? C.accent : C.s1, "stroke-width": 2.5 }, s);
      T(s, x + 8, 246, `search rank ${i + 1}`, "lt-s mid");
      if (i === best) T(s, x + 8, 262, "winner: best average", "lt-a mid");
    });
    T(s, 100, 18, "◇ score in the search    ● retrained with a new seed    ━ average of the three", "lt-s");
  }

  /* ---------- C.11 timeline ---------- */
  function timeline() {
    const canvas = document.getElementById("tl-canvas");
    if (!canvas) return;
    const r = L.rng(5);
    const trials = [], rejects = [];
    const slot = [0, 0];
    let best = 0;
    for (let i = 0; i < 400; i++) {
      const isBlock = i < 96;
      if (!isBlock && r() < 0.3) { rejects.push(Math.min(slot[0], slot[1])); continue; }
      const k = slot[0] <= slot[1] ? 0 : 1, start = slot[k];
      const cutT = !isBlock && r() < 0.5;
      const dur = isBlock ? 0.15 + r() * 0.3 : (cutT ? 0.16 : 0.42);
      const score = isBlock ? L.clamp(0.53 + 0.12 * L.gauss(r), 0.08, 0.71) : (cutT ? null : L.clamp(0.68 + 0.03 * L.gauss(r), 0.5, 0.77));
      slot[k] = start + dur;
      trials.push({ k, start, end: start + dur, isBlock, cut: cutT, score });
    }
    const Tend = Math.max(slot[0], slot[1]);
    const events = trials.filter(t => t.score !== null).map(t => [t.end, t.score]).sort((a, b) => a[0] - b[0]);
    let now = 0, geo = null;
    function draw() {
      if (!geo) geo = Site.setupCanvas(canvas, 260);
      const { ctx, w, h } = geo;
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
      const padL = 92, X = t => padL + t / Tend * (w - padL - 12);
      const lanes = [[118, "GPU slot 1"], [150, "GPU slot 2"], [190, "rejected"]];
      ctx.font = "11px 'JetBrains Mono', monospace"; ctx.fillStyle = C.muted; ctx.textAlign = "right";
      lanes.forEach(([y, l]) => ctx.fillText(l, padL - 8, y + 12));
      ctx.fillText("best", padL - 8, 40);
      trials.forEach(t => {
        if (t.start > now) return;
        const e = Math.min(t.end, now);
        ctx.fillStyle = t.isBlock ? "rgba(167,162,154,.7)" : t.cut ? C.red : C.accent;
        ctx.fillRect(X(t.start), lanes[t.k][0], Math.max(1, X(e) - X(t.start) - 0.6), 18);
      });
      ctx.fillStyle = C.red;
      rejects.forEach(t => { if (t <= now) ctx.fillRect(X(t), lanes[2][0], 1.2, 16); });
      ctx.strokeStyle = C.lineSoft; ctx.strokeRect(padL, 10, w - padL - 12, 80);
      const Yb = v => 90 - (v - 0.4) / 0.4 * 80;
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.beginPath();
      let b = 0, started = false;
      events.forEach(([t, s]) => {
        if (t > now) return;
        const nb = Math.max(b, s);
        if (!started) { ctx.moveTo(X(t), Yb(nb)); started = true; } else { ctx.lineTo(X(t), Yb(b)); ctx.lineTo(X(t), Yb(nb)); }
        b = nb;
      });
      if (started) ctx.lineTo(X(now), Yb(b));
      ctx.stroke(); ctx.lineWidth = 1;
      best = b;
      ctx.fillStyle = C.muted; ctx.textAlign = "center";
      for (let hr = 0; hr <= Tend; hr += 12) ctx.fillText(`${hr} h`, X(hr), h - 8);
      const blockEnd = trials[95] ? trials[95].end : 0;
      if (now >= blockEnd) { ctx.setLineDash([4, 4]); ctx.strokeStyle = C.muted; ctx.beginPath(); ctx.moveTo(X(blockEnd), 100); ctx.lineTo(X(blockEnd), 214); ctx.stroke(); ctx.setLineDash([]); ctx.textAlign = "left"; ctx.fillText("designed block ends", X(blockEnd) + 4, 108); }
      const done = trials.filter(t => t.end <= now).length;
      document.getElementById("tl-readout").innerHTML = `hour <b>${Math.floor(now)}</b> · ${done} trials trained · ${rejects.filter(t => t <= now).length} rejected · best so far <b>${b ? b.toFixed(3) : "-"}</b>`;
    }
    const ctrl = L.loop(canvas, dt => { now = Math.min(Tend, now + dt * Tend / 14); draw(); return now < Tend; },
      { button: document.getElementById("tl-play"), autoplay: false });
    document.getElementById("tl-restart").addEventListener("click", () => { now = 0; draw(); ctrl.play(); });
    window.addEventListener("resize", () => { geo = null; draw(); });
    now = Tend; draw();
  }
})();
