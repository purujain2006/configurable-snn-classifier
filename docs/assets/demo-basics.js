/* demo-basics.js: visuals for page A (neurons, networks, convolutions, spikes). */
(function () {
  "use strict";
  const L = window.Learn;
  const { S, T, C } = L;

  L.onReady(() => {
    [neuron, forward, descent, conv, channels, shrink, head, lif, snnNet, work, surrogate, signed]
      .forEach(f => { try { f(); } catch (e) { console.error(f.name, e); } });
  });

  function arrow(g, x1, y1, x2, y2, color, w) {
    S("line", { x1, y1, x2, y2, stroke: color || C.faint, "stroke-width": w || 1.4 }, g);
    const a = Math.atan2(y2 - y1, x2 - x1), s = 7;
    S("path", {
      d: `M${x2},${y2} L${x2 - s * Math.cos(a - 0.4)},${y2 - s * Math.sin(a - 0.4)} L${x2 - s * Math.cos(a + 0.4)},${y2 - s * Math.sin(a + 0.4)} Z`,
      fill: color || C.faint
    }, g);
  }

  /* ---------- A.1 one neuron ---------- */
  function neuron() {
    const box = document.getElementById("neuron-vis");
    if (!box) return;
    const st = { x: [0, 0, 0], w: [0, 0, 0], b: 0 };
    function draw() {
      const s = L.svg(box, 640, 250, "A neuron with three weighted inputs");
      const ys = [55, 125, 195], nx = 350, ny = 125;
      const sum = st.x.reduce((a, x, i) => a + x * st.w[i], st.b);
      const out = Math.max(0, sum);
      ys.forEach((y, i) => {
        const w = st.w[i];
        S("line", { x1: 82, y1: y, x2: nx - 44, y2: ny, stroke: w >= 0 ? C.accent : C.rust,
          "stroke-width": 1 + 7 * Math.abs(w), "stroke-opacity": 0.85 }, s);
        const mx = (82 + nx - 44) / 2, my = (y + ny) / 2 - 10;
        T(s, mx, my, `× ${w.toFixed(2)}`, w >= 0 ? "lt-a mid" : "lt-r mid");
        S("circle", { cx: 60, cy: y, r: 22, fill: `rgba(236,231,222,${0.06 + 0.3 * st.x[i]})`, stroke: C.lineStrong }, s);
        T(s, 60, y + 4, st.x[i].toFixed(2), "lt-m mid");
        T(s, 60, y - 28, `input ${i + 1}`, "lt-s mid");
      });
      S("line", { x1: nx, y1: 32, x2: nx, y2: ny - 44, stroke: st.b >= 0 ? C.accent : C.rust, "stroke-width": 1.5, "stroke-dasharray": "3 3" }, s);
      T(s, nx, 24, `bias ${st.b >= 0 ? "+" : "−"}${Math.abs(st.b).toFixed(2)}`, "lt-m mid");
      S("circle", { cx: nx, cy: ny, r: 44, fill: C.panel, stroke: C.accent, "stroke-width": 1.5 }, s);
      T(s, nx, ny - 6, "Σ", "lt-h mid");
      T(s, nx, ny + 16, sum.toFixed(2), "lt-m mid");
      arrow(s, nx + 46, ny, 432, ny);
      // ReLU panel
      const bx = 440, by = 70, bw = 100, bh = 110;
      S("rect", { x: bx, y: by, width: bw, height: bh, fill: "none", stroke: C.line }, s);
      T(s, bx + bw / 2, by - 8, "ReLU", "lt-s mid");
      const X = v => bx + (v + 2) / 4 * bw, Y = v => by + bh - v / 2 * bh;
      S("line", { x1: X(0), y1: by, x2: X(0), y2: by + bh, stroke: C.lineSoft }, s);
      S("polyline", { points: `${X(-2)},${Y(0)} ${X(0)},${Y(0)} ${X(2)},${Y(2)}`, fill: "none", stroke: C.s1, "stroke-width": 1.6 }, s);
      const cs = L.clamp(sum, -2, 2);
      S("circle", { cx: X(cs), cy: Y(Math.max(0, cs)), r: 5, fill: C.accent }, s);
      arrow(s, bx + bw + 2, ny, 566, ny);
      S("circle", { cx: 595, cy: ny, r: 26, fill: `rgba(224,169,79,${0.08 + 0.6 * Math.min(1, out)})`, stroke: C.accent }, s);
      T(s, 595, ny + 4, out.toFixed(2), "lt-m mid");
      T(s, 595, ny - 36, "output", "lt-s mid");
      document.getElementById("neuron-sum").textContent = sum.toFixed(2);
      document.getElementById("neuron-out").textContent = out.toFixed(2);
    }
    [1, 2, 3].forEach(i => {
      L.range("nx" + i, v => { st.x[i - 1] = v; draw(); }, v => v / 100);
      L.range("nw" + i, v => { st.w[i - 1] = v; draw(); }, v => v / 100);
    });
    L.range("nb", v => { st.b = v; draw(); }, v => v / 100);
    draw();
  }

  /* ---------- A.2 forward pass ---------- */
  function forward() {
    const box = document.getElementById("fwd-vis");
    if (!box) return;
    const sizes = [4, 6, 5, 3], labels = ["clap", "wave", "roll"];
    const r = L.rng(7);
    const W = [], B = [];
    for (let l = 0; l < 3; l++) {
      W.push([...Array(sizes[l + 1])].map(() => [...Array(sizes[l])].map(() => L.gauss(r) * 1.1 / Math.sqrt(sizes[l]))));
      B.push([...Array(sizes[l + 1])].map(() => 0.1 * L.gauss(r)));
    }
    let x = [], acts = [], probs = [];
    let inputSeed = 3;
    function compute() {
      const ri = L.rng(inputSeed);
      x = [...Array(4)].map(() => Math.round(ri() * 100) / 100);
      acts = [x];
      for (let l = 0; l < 3; l++) {
        const a = W[l].map((row, j) => row.reduce((s, w, i) => s + w * acts[l][i], B[l][j]));
        acts.push(l < 2 ? a.map(v => Math.max(0, v)) : a);
      }
      const z = acts[3], m = Math.max(...z), e = z.map(v => Math.exp(v - m)), se = e.reduce((a, b) => a + b);
      probs = e.map(v => v / se);
    }
    const s = L.svg(box, 760, 290, "A four-layer network running a forward pass");
    const xs = [70, 250, 430, 560];
    const pos = sizes.map((n, l) => [...Array(n)].map((_, i) => [xs[l], 145 + (i - (n - 1) / 2) * 44]));
    const gE = S("g", {}, s), gP = S("g", {}, s), gN = S("g", {}, s), gB = S("g", {}, s);
    const edges = [];
    for (let l = 0; l < 3; l++) for (let j = 0; j < sizes[l + 1]; j++) for (let i = 0; i < sizes[l]; i++) {
      const [x1, y1] = pos[l][i], [x2, y2] = pos[l + 1][j], w = W[l][j][i];
      const ln = S("line", { x1, y1, x2, y2, stroke: w >= 0 ? C.accent : C.rust, "stroke-width": 0.6 + 2.2 * Math.min(1.5, Math.abs(w)), "stroke-opacity": 0.12 }, gE);
      const p = S("circle", { r: 3, fill: w >= 0 ? C.accent : C.rust, opacity: 0 }, gP);
      edges.push({ l, i, j, w, ln, p, x1, y1, x2, y2 });
    }
    const nodes = pos.map((col, l) => col.map(([cx, cy]) => {
      const c = S("circle", { cx, cy, r: 15, fill: C.cell, stroke: C.lineStrong }, gN);
      const t = T(gN, cx, cy + 4, "", "lt-ms mid");
      return { c, t };
    }));
    ["input", "hidden layer 1", "hidden layer 2", "output"].forEach((n, l) => T(s, xs[l], 280, n, "lt-s mid"));
    const bars = labels.map((lab, j) => {
      const y = pos[3][j][1];
      T(gB, 590, y + 4, lab, "lt-m");
      S("rect", { x: 632, y: y - 7, width: 110, height: 14, fill: "rgba(255,255,255,.05)" }, gB);
      const bar = S("rect", { x: 632, y: y - 7, width: 0, height: 14, fill: C.s4 }, gB);
      const pt = T(gB, 742, y - 11, "", "lt-ms end");
      return { bar, pt };
    });
    let phase = 0;
    function render() {
      const done = Math.floor(phase), frac = phase - done;
      sizes.forEach((n, l) => {
        const a = acts[l], mx = Math.max(1e-6, ...a.map(Math.abs));
        nodes[l].forEach((nd, i) => {
          const on = l <= done || (l === done + 1 && frac > 0.85);
          const v = l === 3 ? probs[i] : a[i] / mx;
          nd.c.setAttribute("fill", on ? `rgba(224,169,79,${0.1 + 0.8 * Math.max(0, v)})` : C.cell);
          nd.t.textContent = on && l === 0 ? a[i].toFixed(2) : "";
        });
      });
      edges.forEach(e => {
        const strength = Math.min(1, Math.abs(e.w * acts[e.l][e.i]) / 0.8);
        const active = e.l === done && phase < 3;
        const lit = e.l < done || active;
        e.ln.setAttribute("stroke-opacity", lit ? 0.12 + 0.6 * strength : 0.08);
        if (active && strength > 0.02) {
          e.p.setAttribute("cx", L.lerp(e.x1, e.x2, frac));
          e.p.setAttribute("cy", L.lerp(e.y1, e.y2, frac));
          e.p.setAttribute("opacity", 0.25 + 0.75 * strength);
        } else e.p.setAttribute("opacity", 0);
      });
      const best = probs.indexOf(Math.max(...probs));
      bars.forEach((b, j) => {
        const show = phase >= 3;
        b.bar.setAttribute("width", show ? 110 * probs[j] : 0);
        b.bar.setAttribute("fill", j === best ? C.accent : C.s4);
        b.pt.textContent = show ? `${Math.round(probs[j] * 100)}%` : "";
      });
      document.getElementById("fwd-answer").innerHTML = phase >= 3
        ? `answer: <b>${labels[best]}</b> (${Math.round(probs[best] * 100)}%)` : "running…";
    }
    const ctrl = L.loop(box, dt => { phase = Math.min(3, phase + dt / 0.9); render(); return phase < 3; });
    function run() { phase = 0; render(); ctrl.play(); }
    document.getElementById("fwd-new").addEventListener("click", () => { inputSeed++; compute(); run(); });
    document.getElementById("fwd-run").addEventListener("click", run);
    compute();
    if (L.reduced) phase = 3;
    render();
  }

  /* ---------- A.3 gradient descent ---------- */
  function descent() {
    const canvas = document.getElementById("gd-canvas");
    if (!canvas) return;
    const LRS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.62, 0.7];
    const th = 0.5, ct = Math.cos(th), sn = Math.sin(th), A = 3.0, Bc = 0.25;
    const f = (x, y) => { const u = x * ct + y * sn, v = -x * sn + y * ct; return 0.5 * (A * u * u + Bc * v * v); };
    const grad = (x, y) => {
      const u = x * ct + y * sn, v = -x * sn + y * ct, gu = A * u, gv = Bc * v;
      return [gu * ct - gv * sn, gu * sn + gv * ct];
    };
    const start = [-3.4, 2.0];
    let lr = 0.2, path = [start.slice()], status = "ready";
    let bg = null, geo = null;
    function layout() {
      const g = Site.setupCanvas(canvas, 320);
      const lw = Math.min(g.w * 0.62, g.w - 200);
      geo = { ...g, lw, X: x => (x + 4.5) / 9 * lw, Y: y => g.h / 2 - y / 9 * lw * (g.h / g.h) };
      geo.Y = y => g.h / 2 - y * (lw / 9);
      // background bands
      const dpr = window.devicePixelRatio || 1;
      const W = Math.round(lw * dpr), H = Math.round(g.h * dpr);
      const img = g.ctx.createImageData(W, H);
      for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
        const x = (px / dpr) / lw * 9 - 4.5, y = (g.h / 2 - py / dpr) / (lw / 9);
        const v = f(x, y), band = Math.floor(Math.log(v + 0.02) * 2.2);
        const edge = Math.floor(Math.log(f(x + 0.03, y) + 0.02) * 2.2) !== band;
        const lum = 10 + Math.min(40, Math.max(0, band + 9) * 3.2);
        const k = (py * W + px) * 4;
        img.data[k] = edge ? 120 : lum; img.data[k + 1] = edge ? 100 : lum * 0.95; img.data[k + 2] = edge ? 70 : lum * 0.88; img.data[k + 3] = 255;
      }
      bg = img;
    }
    function draw() {
      if (!geo) layout();
      const { ctx, w, h, lw, X, Y } = geo;
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
      const dpr = window.devicePixelRatio || 1;
      ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.putImageData(bg, 0, 0); ctx.restore();
      ctx.fillStyle = C.green; ctx.beginPath(); ctx.arc(X(0), Y(0), 4, 0, 7); ctx.fill();
      ctx.font = "12px 'JetBrains Mono', monospace"; ctx.fillStyle = C.green; ctx.fillText("lowest loss", X(0) + 8, Y(0) - 8);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1.4; ctx.beginPath();
      path.forEach(([x, y], i) => { const px = L.clamp(X(x), -50, lw + 50), py = L.clamp(Y(y), -50, h + 50); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
      ctx.fillStyle = C.accent;
      path.forEach(([x, y], i) => { ctx.beginPath(); ctx.arc(X(x), Y(y), i === path.length - 1 ? 5 : 2.2, 0, 7); ctx.fill(); });
      ctx.fillStyle = C.s1; ctx.fillText("start", X(start[0]) - 14, Y(start[1]) - 10);
      // loss panel
      const px0 = lw + 50, pw = w - px0 - 14, py0 = 22, ph = h - 64;
      ctx.strokeStyle = C.line; ctx.strokeRect(px0, py0, pw, ph);
      const lo = -3, hi = 2, NX = 60;
      const LY = v => py0 + ph - (Math.log10(Math.max(1e-3, v)) - lo) / (hi - lo) * ph;
      ctx.fillStyle = C.muted;
      [-3, -2, -1, 0, 1, 2].forEach(e => { ctx.fillText(e === 0 ? "1" : `1e${e}`, px0 - 38, LY(Math.pow(10, e)) + 4); });
      ctx.fillText("loss", px0, py0 - 7); ctx.fillText(`step (0 to ${NX})`, px0 + pw - 110, py0 + ph + 18);
      ctx.strokeStyle = C.accent; ctx.beginPath();
      path.slice(0, NX + 1).forEach(([x, y], i) => { const qx = px0 + i / NX * pw, qy = L.clamp(LY(f(x, y)), py0, py0 + ph); i ? ctx.lineTo(qx, qy) : ctx.moveTo(qx, qy); });
      ctx.stroke();
      const [cx, cy] = path[path.length - 1];
      document.getElementById("gd-step").textContent = path.length - 1;
      document.getElementById("gd-loss").textContent = f(cx, cy) > 999 ? "huge" : f(cx, cy).toFixed(3);
      document.getElementById("gd-status").textContent = status;
    }
    function stepOnce() {
      const [x, y] = path[path.length - 1], [gx, gy] = grad(x, y);
      const nx = x - lr * gx, ny = y - lr * gy;
      path.push([nx, ny]);
      const v = f(nx, ny);
      if (!isFinite(v) || Math.abs(nx) > 12 || Math.abs(ny) > 12) { status = "flew out of the valley"; return false; }
      if (v < 2e-3) { status = "reached the bottom"; return false; }
      if (path.length > 60) { status = lr < 0.1 ? "still crawling" : "nearly there"; return false; }
      const prev = path[path.length - 2];
      status = Math.sign((nx * ct + ny * sn)) !== Math.sign((prev[0] * ct + prev[1] * sn)) && lr > 0.4 ? "bouncing across the valley" : "rolling downhill";
      return true;
    }
    const tk = L.ticker(6);
    let finished = false;
    const playBtn = document.getElementById("gd-play");
    playBtn.addEventListener("click", () => { if (finished) reset(); }, true);
    const ctrl = L.loop(canvas, dt => {
      let alive = true;
      for (let n = tk.ticks(dt); n > 0 && alive; n--) alive = stepOnce();
      draw();
      if (!alive) finished = true;
      return alive;
    }, { button: playBtn, autoplay: false });
    function reset() { path = [start.slice()]; status = "ready"; finished = false; draw(); }
    L.range("gd-lr", v => { lr = v; reset(); }, i => LRS[i]);
    document.getElementById("gd-reset").addEventListener("click", () => { reset(); ctrl.play(); });
    window.addEventListener("resize", () => { geo = null; draw(); });
    draw();
  }

  /* ---------- A.4 convolution ---------- */
  function conv() {
    const box = document.getElementById("conv-vis");
    if (!box) return;
    const N = 10, K = 3, M = N - K + 1;
    const KERNELS = {
      vertical: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
      horizontal: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
      blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map(r => r.map(v => v / 9)),
      outline: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]].map(r => r.map(v => v / 4))
    };
    let img = [...Array(N)].map(() => Array(N).fill(0));
    function defaultImg() {
      img = [...Array(N)].map(() => Array(N).fill(0));
      for (let r = 1; r <= 8; r++) { img[r][2] = 1; img[r][3] = 1; }
      for (let c = 2; c <= 8; c++) { img[7][c] = 1; img[8][c] = 1; }
      for (let r = 1; r <= 3; r++) for (let c = 6; c <= 8; c++) img[r][c] = 1;
    }
    defaultImg();
    let kname = "vertical", pos = -1;
    const cs = 22, ix = 16, iy = 40, kx = 292, ky = 106, kc = 30, ox = 450, oy = 62;
    const s = L.svg(box, 680, 300, "A kernel sliding across a picture to make a feature map");
    T(s, ix, 26, "picture (click to draw)", "lt-s");
    T(s, kx, ky - 14, "kernel", "lt-s");
    T(s, ox, oy - 14, "feature map", "lt-s");
    const inCells = [], outCells = [], kCells = [], kTxt = [], oTxt = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const rect = S("rect", { x: ix + c * cs, y: iy + r * cs, width: cs - 2, height: cs - 2, style: "cursor:pointer" }, s);
      rect.addEventListener("click", () => { img[r][c] = img[r][c] ? 0 : 1; paint(); });
      inCells.push(rect);
    }
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
      kCells.push(S("rect", { x: kx + j * kc, y: ky + i * kc, width: kc - 2, height: kc - 2, fill: C.cell, stroke: C.line }, s));
      kTxt.push(T(s, kx + j * kc + kc / 2 - 1, ky + i * kc + kc / 2 + 3, "", "lt-ms mid"));
    }
    for (let r = 0; r < M; r++) for (let c = 0; c < M; c++) {
      outCells.push(S("rect", { x: ox + c * cs, y: oy + r * cs, width: cs - 2, height: cs - 2, fill: C.cell }, s));
    }
    const win = S("rect", { width: K * cs - 2, height: K * cs - 2, fill: "none", stroke: C.accent, "stroke-width": 2, opacity: 0 }, s);
    const oHi = S("rect", { width: cs - 2, height: cs - 2, fill: "none", stroke: C.accent, "stroke-width": 2, opacity: 0 }, s);
    const l1 = S("line", { stroke: C.accent, "stroke-opacity": 0.5, "stroke-dasharray": "3 3" }, s);
    const l2 = S("line", { stroke: C.accent, "stroke-opacity": 0.5, "stroke-dasharray": "3 3" }, s);
    const readout = document.getElementById("conv-readout");
    function out(r, c) {
      const k = KERNELS[kname];
      let v = 0;
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) v += k[i][j] * img[r + i][c + j];
      return v;
    }
    function maxAbs() {
      let m = 1e-6;
      for (let r = 0; r < M; r++) for (let c = 0; c < M; c++) m = Math.max(m, Math.abs(out(r, c)));
      return m;
    }
    function paint() {
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) inCells[r * N + c].setAttribute("fill", img[r][c] ? C.s1 : C.cell);
      const k = KERNELS[kname], km = Math.max(...k.flat().map(Math.abs));
      k.flat().forEach((v, i) => {
        kCells[i].setAttribute("fill", L.signedFill(v / km));
        kTxt[i].textContent = Math.abs(v) < 0.2 && v !== 0 ? v.toFixed(2) : String(+v.toFixed(2));
        kTxt[i].setAttribute("class", Math.abs(v / km) > 0.45 ? "lt-dark mid" : "lt-ms mid");
      });
      const m = maxAbs();
      for (let r = 0; r < M; r++) for (let c = 0; c < M; c++) {
        const idx = r * M + c;
        outCells[idx].setAttribute("fill", idx <= pos ? L.signedFill(out(r, c) / m) : C.cell);
        outCells[idx].setAttribute("stroke", idx <= pos ? "none" : C.lineSoft);
      }
      if (pos >= 0 && pos < M * M) {
        const r = Math.floor(pos / M), c = pos % M;
        win.setAttribute("x", ix + c * cs); win.setAttribute("y", iy + r * cs); win.setAttribute("opacity", 1);
        oHi.setAttribute("x", ox + c * cs); oHi.setAttribute("y", oy + r * cs); oHi.setAttribute("opacity", 1);
        l1.setAttribute("x1", ix + (c + K) * cs); l1.setAttribute("y1", iy + (r + 1.5) * cs);
        l1.setAttribute("x2", kx); l1.setAttribute("y2", ky + 1.5 * kc);
        l2.setAttribute("x1", kx + K * kc); l2.setAttribute("y1", ky + 1.5 * kc);
        l2.setAttribute("x2", ox + c * cs); l2.setAttribute("y2", oy + (r + 0.5) * cs);
        [l1, l2].forEach(l => l.setAttribute("opacity", 1));
        readout.innerHTML = `row ${r + 1}, column ${c + 1}: sum of 9 products = <b>${+out(r, c).toFixed(2)}</b>`;
      } else {
        [win, oHi, l1, l2].forEach(e => e.setAttribute("opacity", 0));
        readout.textContent = pos >= M * M - 1 ? "feature map complete" : "";
      }
    }
    function advance() { pos++; if (pos >= M * M) { pos = M * M - 1; return false; } paint(); return true; }
    const tk = L.ticker(7);
    const ctrl = L.loop(box, dt => {
      let alive = true;
      for (let n = tk.ticks(dt); n > 0 && alive; n--) alive = advance();
      if (!alive) { pos = M * M; paint(); }
      return alive;
    }, { button: document.getElementById("conv-play"), autoplay: false });
    document.getElementById("conv-play").addEventListener("click", () => { if (pos >= M * M - 1) { pos = -1; paint(); } }, true);
    document.getElementById("conv-step").addEventListener("click", () => { ctrl.pause(); if (pos >= M * M - 1) pos = -1; advance(); });
    document.getElementById("conv-clear").addEventListener("click", () => { img = img.map(r => r.map(() => 0)); paint(); });
    L.seg(document.getElementById("conv-kernel"), v => { kname = v; paint(); });
    pos = M * M; paint();
  }

  /* ---------- A.4 channels (static) ---------- */
  function slab(g, x, y, size, depth, fill, stroke, step) {
    step = step || 9;
    for (let d = depth - 1; d >= 0; d--) {
      S("rect", { x: x + d * step, y: y - d * step, width: size, height: size, fill, stroke: stroke || C.lineStrong }, g);
    }
  }
  function channels() {
    const box = document.getElementById("channels-vis");
    if (!box) return;
    const s = L.svg(box, 820, 230, "Channels: kernels make one feature map each");
    slab(s, 20, 80, 110, 2, "rgba(236,231,222,.06)");
    T(s, 20, 215, "input: 2 channels", "lt-b");
    T(s, 20, 228, "(brighter and darker events)", "lt-s");
    const cols = [C.accent, C.rust, C.green];
    [0, 1, 2].forEach(i => {
      const y = 40 + i * 58;
      S("rect", { x: 230, y, width: 26, height: 26, fill: "none", stroke: cols[i], "stroke-width": 1.6 }, s);
      S("rect", { x: 235, y: y - 5, width: 26, height: 26, fill: "none", stroke: cols[i], "stroke-opacity": 0.6 }, s);
      T(s, 272, y + 18, `kernel ${i + 1}`, "lt-s");
      arrow(s, 160, 135, 226, y + 13, C.faint, 1);
      arrow(s, 330, y + 13, 404, 120 - i * 9 + 20, cols[i], 1.1);
    });
    for (let d = 2; d >= 0; d--) {
      S("rect", { x: 420 + d * 12, y: 70 - d * 12, width: 100, height: 100, fill: "rgba(0,0,0,.6)", stroke: cols[d], "stroke-width": 1.4 }, s);
    }
    T(s, 420, 215, "layer 1: 3 channels", "lt-b");
    T(s, 420, 228, "3 kernels make 3 maps", "lt-s");
    S("rect", { x: 610, y: 100, width: 28, height: 28, fill: "none", stroke: C.s1, "stroke-width": 1.4 }, s);
    S("rect", { x: 616, y: 94, width: 28, height: 28, fill: "none", stroke: C.s1, "stroke-opacity": .7 }, s);
    S("rect", { x: 622, y: 88, width: 28, height: 28, fill: "none", stroke: C.s1, "stroke-opacity": .45 }, s);
    T(s, 600, 160, "a next-layer kernel", "lt-s");
    T(s, 600, 173, "spans all 3 maps", "lt-s");
    arrow(s, 560, 110, 604, 112);
    arrow(s, 660, 110, 720, 110);
    S("rect", { x: 726, y: 72, width: 76, height: 76, fill: "rgba(224,169,79,.12)", stroke: C.accent }, s);
    T(s, 812, 215, "one map per kernel", "lt-b end");
    T(s, 812, 228, "in the next layer", "lt-s end");
  }

  /* ---------- A.4 shrinking ---------- */
  function shrink() {
    const box = document.getElementById("shrink-vis");
    if (!box) return;
    const r = L.rng(11), N = 8;
    const g = [...Array(N)].map(() => [...Array(N)].map(() => (r() < 0.33 ? 1 : 0)));
    const NOTES = {
      stride: "A 3 × 3 kernel that moves two cells at a time (here every weight is 1, so each output counts the spikes in its window). The windows overlap by one cell, and the 8 × 8 map becomes 3 × 3 without padding.",
      avg: "Each output is the average of a 2 × 2 block, so a block holding one spike gives 0.25. The chip has no pooling unit: the average is folded into the next layer's weights, so that layer reads all four cells and has four times the connections.",
      max: "Each output is the largest value in its block. For spikes that means: did any of the four fire? On the chip this is one extra neuron per block that fires if any input fired. It adds one step of delay, but the next layer only reads the pooled map."
    };
    let mode = "stride", idx = 0;
    const s = L.svg(box, 700, 250, "Shrinking a map of spikes by stride or pooling");
    const cs = 24, ix = 24, iy = 30;
    const gIn = S("g", {}, s), gOut = S("g", {}, s), gHi = S("g", {}, s);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      S("rect", { x: ix + j * cs, y: iy + i * cs, width: cs - 2, height: cs - 2, fill: g[i][j] ? C.accent : C.cell }, gIn);
    }
    T(s, ix, 20, "8 × 8 spikes", "lt-s");
    const titleOut = T(s, 340, 20, "", "lt-s");
    function outShape() { return mode === "stride" ? 3 : 4; }
    function value(oi, oj) {
      if (mode === "stride") { let v = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) v += g[oi * 2 + a][oj * 2 + b]; return v; }
      const vals = [g[oi * 2][oj * 2], g[oi * 2][oj * 2 + 1], g[oi * 2 + 1][oj * 2], g[oi * 2 + 1][oj * 2 + 1]];
      return mode === "avg" ? vals.reduce((a, b) => a + b) / 4 : Math.max(...vals);
    }
    function build() {
      L.clear(gOut);
      const M = outShape(), oc = mode === "stride" ? 48 : 38, ox = 340, oy = 30;
      titleOut.textContent = `${M} × ${M} out`;
      const mx = mode === "stride" ? 9 : 1;
      for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
        const v = value(i, j);
        S("rect", { x: ox + j * oc, y: oy + i * oc, width: oc - 3, height: oc - 3, fill: v > 0 ? `rgba(224,169,79,${0.15 + 0.85 * v / mx})` : C.cell, "data-k": `${i},${j}` }, gOut);
        T(gOut, ox + j * oc + oc / 2 - 1, oy + i * oc + oc / 2 + 4, mode === "avg" ? (v === 0 ? "0" : v.toFixed(2)) : String(v), v / mx > 0.3 ? "lt-dark mid" : "lt-ms mid");
      }
      document.getElementById("shrink-note").textContent = NOTES[mode];
    }
    function highlight() {
      L.clear(gHi);
      const M = outShape(), oc = mode === "stride" ? 48 : 38, ox = 340, oy = 30;
      const oi = Math.floor(idx / M) % M, oj = idx % M;
      const k = mode === "stride" ? 3 : 2, st = 2;
      S("rect", { x: ix + oj * st * cs - 1, y: iy + oi * st * cs - 1, width: k * cs, height: k * cs, fill: "none", stroke: C.s1, "stroke-width": 2 }, gHi);
      S("rect", { x: ox + oj * oc - 1, y: oy + oi * oc - 1, width: oc - 1, height: oc - 1, fill: "none", stroke: C.s1, "stroke-width": 2 }, gHi);
      S("line", { x1: ix + (oj * st + k) * cs, y1: iy + (oi * st + k / 2) * cs, x2: ox + oj * oc, y2: oy + (oi + 0.5) * oc, stroke: C.s1, "stroke-opacity": .4, "stroke-dasharray": "3 3" }, gHi);
    }
    const tk = L.ticker(1.6);
    L.loop(box, dt => { const n = tk.ticks(dt); if (n) { idx = (idx + n) % (outShape() ** 2); highlight(); } return true; });
    L.seg(document.getElementById("shrink-mode"), v => { mode = v; idx = 0; build(); highlight(); });
  }

  /* ---------- A.4 head (static) ---------- */
  function head() {
    const box = document.getElementById("head-vis");
    if (!box) return;
    const s = L.svg(box, 840, 280, "Flatten versus global average pooling");
    const r = L.rng(5), cols = [C.accent, C.rust, C.green];
    const maps = [0, 1, 2].map(() => [...Array(16)].map(() => r()));
    function drawMaps(x, y) {
      for (let d = 2; d >= 0; d--) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
        const v = maps[d][i * 4 + j];
        S("rect", { x: x + d * 14 + j * 15, y: y - d * 14 + i * 15, width: 14, height: 14, fill: `rgba(${d === 0 ? "224,169,79" : d === 1 ? "192,122,92" : "155,177,122"},${0.15 + 0.6 * v})`, stroke: "#000", "stroke-width": 0.5 }, s);
      }
    }
    function classes(x) {
      const pts = [];
      for (let k = 0; k < 11; k++) { const y = 40 + k * 20; pts.push(y); S("circle", { cx: x, cy: y, r: 6, fill: C.cell, stroke: C.lineStrong }, s); }
      T(s, x, 270, "11 classes", "lt-s mid");
      return pts;
    }
    // flatten
    T(s, 20, 20, "Flatten: keeps where", "lt-b");
    drawMaps(20, 120);
    arrow(s, 120, 140, 160, 140);
    const vx = 170, cyc = [];
    for (let k = 0; k < 48; k++) {
      const d = Math.floor(k / 16), y = 22 + k * 4.9;
      cyc.push(y);
      S("rect", { x: vx, y, width: 14, height: 4.2, fill: `rgba(${d === 0 ? "224,169,79" : d === 1 ? "192,122,92" : "155,177,122"},${0.15 + 0.6 * maps[d][k % 16]})` }, s);
    }
    T(s, vx + 7, 270, "48 inputs", "lt-s mid");
    const c1 = classes(330);
    cyc.forEach((y, k) => { if (k % 2 === 0) c1.forEach(cy => S("line", { x1: vx + 14, y1: y + 2, x2: 324, y2: cy, stroke: C.faint, "stroke-opacity": 0.06 }, s)); });
    // GAP
    T(s, 450, 20, "GAP: keeps how much, not where", "lt-b");
    drawMaps(450, 120);
    arrow(s, 550, 140, 600, 140);
    T(s, 575, 128, "average", "lt-s mid");
    const gy = [110, 140, 170];
    gy.forEach((y, d) => {
      const m = maps[d].reduce((a, b) => a + b) / 16;
      S("rect", { x: 615, y: y - 9, width: 18, height: 18, fill: `rgba(${d === 0 ? "224,169,79" : d === 1 ? "192,122,92" : "155,177,122"},${0.15 + 0.6 * m})` }, s);
    });
    T(s, 624, 270, "3 inputs", "lt-s mid");
    const c2 = classes(790);
    gy.forEach(y => c2.forEach(cy => S("line", { x1: 633, y1: y, x2: 784, y2: cy, stroke: C.faint, "stroke-opacity": 0.18 }, s)));
  }

  /* ---------- A.5 LIF neuron ---------- */
  function lif() {
    const canvas = document.getElementById("lif-canvas");
    if (!canvas) return;
    const Wt = [0.45, 0.35, 0.25], VIS = 90;
    let tau = 4, th = 1.0, rate = 0.3, v = 0, n = 0;
    const r = L.rng(21);
    let hist = [];
    let lastTxt = "-";
    function stepOnce() {
      const ins = Wt.map(() => (r() < rate ? 1 : 0));
      const carried = v;
      const fired = carried > th;
      let vv = fired ? 0 : carried;
      vv = vv - vv / tau;
      const leaked = vv;
      const drive = ins.reduce((a, s, i) => a + s * Wt[i], 0);
      vv += drive;
      v = vv; if (fired) n++;
      hist.push({ ins, v: carried, fired, after: v });
      if (hist.length > VIS) hist.shift();
      lastTxt = `${carried.toFixed(2)} ${fired ? "fired, reset" : "no spike"}, leak to ${leaked.toFixed(2)}, +${drive.toFixed(2)} = ${v.toFixed(2)}`;
    }
    let geo = null;
    function draw() {
      if (!geo) geo = Site.setupCanvas(canvas, 300);
      const { ctx, w, h } = geo;
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, h);
      const x0 = 70, pw = w - x0 - 12, dx = pw / VIS;
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillStyle = C.muted;
      ["in 1", "in 2", "in 3"].forEach((t, i) => ctx.fillText(`${t} ×${Wt[i]}`, 4, 22 + i * 16));
      const vy0 = 74, vh = 170, vmax = Math.max(1.6, th * 1.4);
      const Y = val => vy0 + vh - val / vmax * vh;
      ctx.strokeStyle = C.lineSoft; ctx.strokeRect(x0, vy0, pw, vh);
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.accent; ctx.beginPath(); ctx.moveTo(x0, Y(th)); ctx.lineTo(x0 + pw, Y(th)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C.accent; ctx.fillText("threshold", 4, Y(th) + 4);
      ctx.fillStyle = C.muted; ctx.fillText("potential", 4, vy0 + vh - 4);
      ctx.fillText("out", 4, h - 22);
      hist.forEach((e, k) => {
        const x = x0 + k * dx;
        e.ins.forEach((s, i) => { if (s) { ctx.fillStyle = C.s1; ctx.fillRect(x + 1, 12 + i * 16, Math.max(1.5, dx - 2), 11); } });
        if (e.fired) { ctx.fillStyle = C.accent; ctx.fillRect(x + 1, h - 34, Math.max(2, dx - 2), 18); }
      });
      ctx.strokeStyle = C.s1; ctx.lineWidth = 1.5; ctx.beginPath();
      hist.forEach((e, k) => { const x = x0 + k * dx; k ? ctx.lineTo(x, Y(Math.min(vmax, e.v))) : ctx.moveTo(x, Y(Math.min(vmax, e.v))); ctx.lineTo(x + dx, Y(Math.min(vmax, e.v))); });
      ctx.stroke();
      document.getElementById("lif-v").textContent = v.toFixed(2);
      document.getElementById("lif-n").textContent = n;
      document.getElementById("lif-last").textContent = lastTxt;
    }
    const tk = L.ticker(8);
    const ctrl = L.loop(canvas, dt => { for (let k = tk.ticks(dt); k > 0; k--) stepOnce(); draw(); return true; },
      { button: document.getElementById("lif-play") });
    document.getElementById("lif-step").addEventListener("click", () => { ctrl.pause(); stepOnce(); draw(); });
    L.seg(document.getElementById("lif-tau"), val => { tau = +val; });
    L.range("lif-th", val => { th = val; draw(); }, x => x / 100);
    L.range("lif-rate", val => { rate = val; }, x => x / 100);
    for (let k = 0; k < 40; k++) stepOnce();
    window.addEventListener("resize", () => { geo = null; draw(); });
    draw();
  }

  /* ---------- A.6 spiking network ---------- */
  function snnNet() {
    const box = document.getElementById("snn-vis");
    if (!box) return;
    const N = 8, H = 8, O = 3, labels = ["clap", "wave", "roll"], FLUSH = 2;
    const wIn = (j, c) => (c === j ? 0.2 : -0.015);
    const wOut = [[0, 0, 0, 0.42, 0.42, 0, 0, 0], [0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34], [0.5, 0, 0, 0, 0, 0, 0, 0.5]];
    let Tn = 24, t = 0, r = L.rng(4);
    let vh = Array(H).fill(0), vo = Array(O).fill(0), sh = Array(H).fill(0), so = Array(O).fill(0), cnt = Array(O).fill(0), sin = [];
    const s = L.svg(box, 780, 330, "A spiking network reading a moving bar");
    const cs = 20, gx = 20, gy = 40;
    T(s, gx, 26, "input frame", "lt-s");
    const inCells = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) inCells.push(S("rect", { x: gx + j * cs, y: gy + i * cs, width: cs - 2, height: cs - 2, fill: C.cell }, s));
    const hx = 290, ox = 470;
    const hy = j => 52 + j * 21, oy = k => 90 + k * 50;
    const eIn = [], eOut = [];
    for (let j = 0; j < H; j++) eIn.push(S("line", { x1: gx + j * cs + cs / 2, y1: gy + N * cs, x2: hx - 9, y2: hy(j), stroke: C.faint, "stroke-opacity": .15 }, s));
    for (let k = 0; k < O; k++) for (let j = 0; j < H; j++) if (wOut[k][j]) eOut.push({ k, j, ln: S("line", { x1: hx + 9, y1: hy(j), x2: ox - 13, y2: oy(k), stroke: C.accent, "stroke-opacity": .12, "stroke-width": 1.2 }, s) });
    const hN = [...Array(H)].map((_, j) => S("circle", { cx: hx, cy: hy(j), r: 8, fill: C.cell, stroke: C.lineStrong }, s));
    T(s, hx, 26, "hidden", "lt-s mid");
    const oN = [...Array(O)].map((_, k) => S("circle", { cx: ox, cy: oy(k), r: 12, fill: C.cell, stroke: C.lineStrong }, s));
    T(s, ox, 26, "output", "lt-s mid");
    const bars = labels.map((lab, k) => {
      T(s, ox + 26, oy(k) + 4, lab, "lt-m");
      S("rect", { x: ox + 70, y: oy(k) - 8, width: 200, height: 16, fill: "rgba(255,255,255,.05)" }, s);
      return { b: S("rect", { x: ox + 70, y: oy(k) - 8, width: 0, height: 16, fill: C.s4 }, s), t: T(s, ox + 276, oy(k) + 4, "0", "lt-ms") };
    });
    T(s, ox + 70, 26, "spike count", "lt-s");
    const strip = S("g", {}, s);
    function buildStrip() {
      L.clear(strip);
      const total = Tn + FLUSH, w = 740 / total;
      for (let k = 0; k < total; k++) S("rect", { x: 20 + k * w, y: 290, width: w - 1, height: 14, fill: k >= Tn ? "rgba(255,255,255,.1)" : C.cell, "data-k": k }, strip);
      T(strip, 20, 322, `T = ${Tn} frames`, "lt-s");
      T(strip, 760, 322, `+ ${FLUSH} flush steps`, "lt-s end");
    }
    function barCol(tt) {
      const period = 14, ph = tt % period, x = ph < period / 2 ? ph : period - ph;
      return Math.round(x / (period / 2) * (N - 2));
    }
    function stepOnce() {
      if (t >= Tn + FLUSH) return false;
      const c0 = barCol(t);
      sin = [];
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const inBar = t < Tn && (j === c0 || j === c0 + 1);
        sin.push(t < Tn && r() < (inBar ? 0.8 : 0.03) ? 1 : 0);
      }
      // chip order: fire on carried-in potential, reset, leak, integrate
      const fh = vh.map(v => v > 1.0), fo = vo.map(v => v > 1.0);
      const tau = 4;
      vh = vh.map((v, j) => { let x = fh[j] ? 0 : v; x -= x / tau; let d = 0; for (let i = 0; i < N; i++) for (let c = 0; c < N; c++) d += sin[i * N + c] * wIn(j, c); return x + d; });
      vo = vo.map((v, k) => { let x = fo[k] ? 0 : v; x -= x / tau; let d = 0; for (let j = 0; j < H; j++) d += sh[j] * wOut[k][j]; return x + d; });
      sh = fh.map(Number); so = fo.map(Number);
      so.forEach((f, k) => { cnt[k] += f; });
      t++;
      return true;
    }
    function paint() {
      sin.forEach((v, i) => inCells[i].setAttribute("fill", v ? C.accent : C.cell));
      if (!sin.length) inCells.forEach(c => c.setAttribute("fill", C.cell));
      hN.forEach((c, j) => c.setAttribute("fill", sh[j] ? C.accent : C.cell));
      oN.forEach((c, k) => c.setAttribute("fill", so[k] ? C.accent : C.cell));
      eIn.forEach((l, j) => l.setAttribute("stroke-opacity", sh[j] ? .5 : .12));
      eOut.forEach(e => e.ln.setAttribute("stroke-opacity", sh[e.j] ? .7 : .1));
      const mx = Math.max(1, ...cnt), best = cnt.indexOf(Math.max(...cnt));
      bars.forEach((b, k) => { b.b.setAttribute("width", 200 * cnt[k] / Math.max(mx, 10)); b.b.setAttribute("fill", cnt[k] && k === best ? C.accent : C.s4); b.t.textContent = cnt[k]; });
      strip.querySelectorAll("rect").forEach(rc => { const k = +rc.dataset.k; rc.setAttribute("stroke", k === t - 1 ? C.accent : "none"); if (k < t && k < Tn) rc.setAttribute("fill", "rgba(224,169,79,.3)"); });
      document.getElementById("snn-t").textContent = `${t} of ${Tn + FLUSH}`;
      document.getElementById("snn-pred").textContent = Math.max(...cnt) ? `${labels[best]}${t >= Tn + FLUSH ? " (final)" : ""}` : "-";
    }
    function restart() { t = 0; r = L.rng(4); vh = Array(H).fill(0); vo = Array(O).fill(0); sh = Array(H).fill(0); so = Array(O).fill(0); cnt = Array(O).fill(0); sin = []; buildStrip(); paint(); }
    const tk = L.ticker(5);
    const ctrl = L.loop(box, dt => {
      let alive = true;
      for (let k = tk.ticks(dt); k > 0 && alive; k--) alive = stepOnce();
      paint();
      return alive;
    }, { button: document.getElementById("snn-play"), autoplay: false });
    document.getElementById("snn-restart").addEventListener("click", () => { restart(); ctrl.play(); });
    L.range("snn-T", v => { Tn = v; restart(); });
  }

  /* ---------- A.7 work ---------- */
  function work() {
    const box = document.getElementById("work-vis");
    if (!box) return;
    const s = L.svg(box, 780, 270, "An ordinary layer and a spiking layer doing work");
    const r = L.rng(9);
    const panels = [{ x: 30, title: "Ordinary layer: every connection, every step" }, { x: 420, title: "Spiking layer: only connections from neurons that fired" }];
    const P = panels.map(p => {
      T(s, p.x, 20, p.title, "lt-b");
      const ins = [...Array(8)].map((_, i) => [p.x + 30, 45 + i * 26]);
      const outs = [...Array(6)].map((_, j) => [p.x + 290, 62 + j * 30]);
      const edges = [];
      ins.forEach(([x1, y1], i) => outs.forEach(([x2, y2]) => edges.push({ i, ln: S("line", { x1: x1 + 9, y1, x2: x2 - 9, y2, stroke: C.accent, "stroke-opacity": .08 }, s) })));
      const inC = ins.map(([cx, cy]) => S("circle", { cx, cy, r: 8, fill: C.cell, stroke: C.lineStrong }, s));
      outs.forEach(([cx, cy]) => S("circle", { cx, cy, r: 8, fill: C.cell, stroke: C.lineStrong }, s));
      const label = T(s, p.x, 262, "", "lt-m");
      return { edges, inC, label, total: 0 };
    });
    let steps = 0;
    function stepOnce() {
      steps++;
      const vals = [...Array(8)].map(() => r());
      P[0].inC.forEach((c, i) => c.setAttribute("fill", `rgba(236,231,222,${0.1 + 0.5 * vals[i]})`));
      P[0].edges.forEach(e => e.ln.setAttribute("stroke-opacity", .55));
      P[0].total += 48;
      P[0].label.textContent = `this step: 48 multiply-adds   total: ${L.fmt(P[0].total)}`;
      const spk = vals.map(v => v < 0.22);
      const k = spk.filter(Boolean).length * 6;
      P[1].inC.forEach((c, i) => c.setAttribute("fill", spk[i] ? C.accent : C.cell));
      P[1].edges.forEach(e => e.ln.setAttribute("stroke-opacity", spk[e.i] ? .8 : .05));
      P[1].total += k;
      P[1].label.textContent = `this step: ${k} additions   total: ${L.fmt(P[1].total)}`;
      document.getElementById("work-readout").innerHTML = `after ${steps} steps the spiking layer did <b>${Math.round(100 * P[1].total / P[0].total)}%</b> of the work`;
    }
    const tk = L.ticker(2.5);
    L.loop(box, dt => { for (let k = tk.ticks(dt); k > 0; k--) stepOnce(); return true; }, { button: document.getElementById("work-play") });
    stepOnce();
  }

  /* ---------- A.8 surrogate (static) ---------- */
  function surrogate() {
    const box = document.getElementById("surrogate-vis");
    if (!box) return;
    const s = L.svg(box, 820, 250, "A spike's step and the smooth stand-in used for learning");
    function plot(x0, title, fn, smooth, ymax, yLabel) {
      const pw = 330, ph = 160, y0 = 40;
      T(s, x0, 22, title, "lt-b");
      S("rect", { x: x0, y: y0, width: pw, height: ph, fill: "none", stroke: C.lineSoft }, s);
      const X = v => x0 + (v + 3) / 6 * pw, Y = v => y0 + ph - v / ymax * ph;
      S("line", { x1: X(0), y1: y0, x2: X(0), y2: y0 + ph, stroke: C.line, "stroke-dasharray": "3 3" }, s);
      T(s, X(0), y0 + ph + 16, "threshold", "lt-s mid");
      T(s, x0, y0 + ph + 34, "membrane potential, relative to threshold", "lt-s");
      T(s, x0 - 6, y0 + 10, yLabel, "lt-s end");
      const pts = (f) => [...Array(121)].map((_, i) => { const v = -3 + i * 0.05; return `${X(v)},${Y(Math.min(ymax, f(v)))}`; }).join(" ");
      if (fn) S("polyline", { points: pts(fn), fill: "none", stroke: C.accent, "stroke-width": 2.4 }, s);
      S("polyline", { points: pts(smooth), fill: "none", stroke: C.s1, "stroke-width": 1.6, "stroke-dasharray": fn ? "5 4" : "none" }, s);
      return { X, Y, y0, ph };
    }
    const a = 2;
    const sig = v => 0.5 + Math.atan(Math.PI / 2 * a * v) / Math.PI;
    const dsig = v => (a / 2) / (1 + Math.pow(Math.PI / 2 * a * v, 2));
    plot(60, "Forward: what the neuron outputs", v => (v >= 0 ? 1 : 0), sig, 1.1, "1");
    const p2 = plot(470, "Backward: the slope used for learning", null, dsig, 1.15, "");
    S("line", { x1: p2.X(-3), y1: p2.Y(0) - 1, x2: p2.X(3), y2: p2.Y(0) - 1, stroke: C.accent, "stroke-width": 2.4 }, s);
    S("line", { x1: p2.X(0), y1: p2.Y(0), x2: p2.X(0), y2: p2.y0 + 4, stroke: C.accent, "stroke-width": 2.4 }, s);
    T(s, p2.X(0) + 8, p2.y0 + 14, "true slope: infinite", "lt-a");
    T(s, p2.X(0) + 8, p2.y0 + 28, "here, zero elsewhere", "lt-a");
    T(s, p2.X(-2.9), p2.Y(0.55), "stand-in slope", "lt-m");
  }

  /* ---------- A.9 signed (static) ---------- */
  function signed() {
    const box = document.getElementById("signed-vis");
    if (!box) return;
    const s = L.svg(box, 820, 260, "A signed unit is two neurons");
    const n = 40, drive = [...Array(n)].map((_, t) => 0.55 * Math.sin(t / 5.5) + 0.12 * Math.sin(t * 1.7));
    let vp = 0, vn = 0;
    const sp = [], sm = [];
    drive.forEach(d => {
      const fp = vp > 1, fn = vn > 1;
      vp = (fp ? 0 : vp); vp -= vp / 4; vp += d;
      vn = (fn ? 0 : vn); vn -= vn / 4; vn -= d;
      sp.push(fp); sm.push(fn);
    });
    const x0 = 150, w = 640, dx = w / n;
    T(s, 20, 58, "drive into the unit", "lt-s");
    S("line", { x1: x0, y1: 55, x2: x0 + w, y2: 55, stroke: C.line }, s);
    S("polyline", { points: drive.map((d, t) => `${x0 + t * dx + dx / 2},${55 - d * 40}`).join(" "), fill: "none", stroke: C.s1, "stroke-width": 1.5 }, s);
    T(s, 20, 128, "positive neuron", "lt-s");
    T(s, 20, 168, "negative neuron", "lt-s");
    T(s, 20, 222, "unit output", "lt-b");
    S("line", { x1: x0, y1: 220, x2: x0 + w, y2: 220, stroke: C.line }, s);
    for (let t = 0; t < n; t++) {
      const x = x0 + t * dx;
      S("rect", { x: x + 1, y: 115, width: dx - 2, height: 16, fill: sp[t] ? C.accent : C.cell }, s);
      S("rect", { x: x + 1, y: 155, width: dx - 2, height: 16, fill: sm[t] ? C.rust : C.cell }, s);
      if (sp[t]) S("line", { x1: x + dx / 2, y1: 220, x2: x + dx / 2, y2: 196, stroke: C.accent, "stroke-width": 2.5 }, s);
      if (sm[t]) S("line", { x1: x + dx / 2, y1: 220, x2: x + dx / 2, y2: 244, stroke: C.rust, "stroke-width": 2.5 }, s);
    }
    T(s, x0 + w + 6, 200, "+1", "lt-a");
    T(s, x0 + w + 6, 246, "−1", "lt-r");
  }
})();
