/* demo-bits.js: visuals for page E (multi-bit input). */
(function () {
  "use strict";
  const L = window.Learn;
  const { S, T, C } = L;

  L.onReady(() => {
    [animation, cost, popcount].forEach(f => { try { f(); } catch (e) { console.error(f.name, e); } });
  });

  /* ---------- E.2 the main animation ---------- */
  function animation() {
    const box = document.getElementById("bits-vis");
    if (!box) return;
    const COUNTS = [0, 2, 5, 7, 3, 1, 9, 4, 6, 2, 0, 12];
    const w = 0.2, TH = 1.0, TAU = 4;
    let B = 3, speed = 1, step = -1, phase = 0, v = 0, total = 0, out = [], frameDone = false;
    const s = L.svg(box, 900, 400, "A pixel's event count sent to a neuron one bit per axon");
    // static frame
    T(s, 40, 26, "the pixel, this frame", "lt-s");
    S("rect", { x: 40, y: 36, width: 130, height: 130, fill: C.panel, stroke: C.lineStrong }, s);
    const gDots = S("g", {}, s);
    const countT = T(s, 105, 190, "", "lt-m mid");
    T(s, 210, 26, "cap and write in binary", "lt-s");
    S("rect", { x: 210, y: 70, width: 120, height: 60, fill: C.panel, stroke: C.line }, s);
    const valT = T(s, 270, 108, "", "lt-h mid");
    const clipT = T(s, 270, 150, "", "lt-r mid");
    const gBits = S("g", {}, s), gAx = S("g", {}, s), gPulse = S("g", {}, s);
    const NX = 720, NY = 150;
    S("circle", { cx: NX, cy: NY, r: 38, fill: C.panel, stroke: C.lineStrong, "stroke-width": 1.4 }, s);
    T(s, NX, NY - 50, "first-layer neuron", "lt-s mid");
    const vT = T(s, NX, NY + 5, "", "lt-m mid");
    // potential bar
    const bx = 800, by0 = 60, bh = 180;
    S("rect", { x: bx, y: by0, width: 18, height: bh, fill: Site.a(C.ink, .04), stroke: C.line }, s);
    const vmax = 2.2, BY = val => by0 + bh - Math.min(vmax, val) / vmax * bh;
    S("line", { x1: bx - 6, x2: bx + 24, y1: BY(TH), y2: BY(TH), stroke: C.accent, "stroke-dasharray": "4 3" }, s);
    T(s, bx + 28, BY(TH) + 4, "threshold", "lt-a");
    T(s, bx + 9, by0 + bh + 16, "potential", "lt-s mid");
    const vBar = S("rect", { x: bx + 2, width: 14, fill: C.s1, "fill-opacity": .6 }, s);
    const spikeMark = S("text", { x: NX + 48, y: NY - 30, class: "lt-a", opacity: 0 }, s); spikeMark.textContent = "spike!";
    // timeline
    const gTL = S("g", {}, s);
    T(s, 40, 300, "frames", "lt-s");
    const cells = COUNTS.map((c, i) => {
      const r = S("rect", { x: 100 + i * 52, y: 288, width: 46, height: 26, fill: C.panel, stroke: C.line }, gTL);
      const t = T(gTL, 123 + i * 52, 306, String(c), "lt-ms mid");
      const o = S("rect", { x: 100 + i * 52, y: 320, width: 46, height: 6, fill: "none" }, gTL);
      return { r, t, o };
    });
    T(s, 40, 330, "neuron", "lt-s");
    T(s, 40, 360, "events per frame, then whether the neuron fired", "lt-s");

    let axons = [];
    function layout() {
      L.clear(gBits); L.clear(gAx);
      axons = [];
      for (let k = 0; k < B; k++) {
        const yy = B === 1 ? 150 : 150 - (B - 1) * 30 + (B - 1 - k) * 60;
        const rect = S("rect", { x: 380, y: yy - 18, width: 40, height: 36, fill: C.cell, stroke: C.lineStrong }, gBits);
        const bt = T(gBits, 400, yy + 5, "0", "lt-h mid");
        T(gBits, 372, yy + 5, `${2 ** k}s`, "lt-ms end");
        const line = S("line", { x1: 420, y1: yy, x2: NX - 38, y2: NY + (yy - NY) * 0.25, stroke: C.lineStrong, "stroke-width": 1.5 }, gAx);
        L.plate(T(gAx, 470, yy - 6, `axon ${k + 1}, weight ${(w * 2 ** k).toFixed(1)}`, "lt-ms"));
        axons.push({ k, yy, rect, bt, line, x1: 420, y1: yy, x2: NX - 38, y2: NY + (yy - NY) * 0.25 });
      }
    }
    function bitsOf(val) { return [...Array(B)].map((_, k) => (val >> k) & 1); }
    function frameValues() {
      const c = COUNTS[step], cap = 2 ** B - 1, val = Math.min(c, cap);
      return { c, cap, val, bits: bitsOf(val) };
    }
    function render() {
      if (step < 0) { countT.textContent = "press Play"; valT.textContent = ""; clipT.textContent = ""; return; }
      const { c, cap, val, bits } = frameValues();
      const shown = phase < 0.3 ? Math.round(c * phase / 0.3) : c;
      if (gDots.childElementCount !== shown) { L.clear(gDots); const r = L.rng(17 + step * 7); for (let i = 0; i < shown; i++) S("circle", { cx: 52 + r() * 106, cy: 48 + r() * 106, r: 5, fill: r() < 0.5 ? C.accent : C.rust }, gDots); }
      countT.textContent = `${c} event${c === 1 ? "" : "s"}`;
      const p1 = phase >= 0.3;
      valT.textContent = p1 ? String(val) : "";
      clipT.textContent = p1 && c > cap ? `capped at ${cap}` : (p1 ? `${bits.slice().reverse().join("")} in binary` : "");
      clipT.setAttribute("class", c > cap ? "lt-r mid" : "lt-ms mid");
      axons.forEach(a => {
        const on = p1 && bits[a.k];
        a.bt.textContent = p1 ? String(bits[a.k]) : "0";
        a.rect.setAttribute("fill", on ? C.accent : C.cell);
        a.bt.setAttribute("class", on ? "lt-dark mid" : "lt-h mid");
        a.line.setAttribute("stroke", on && phase >= 0.5 ? C.accent : C.lineStrong);
      });
      L.clear(gPulse);
      if (phase >= 0.5 && phase < 1.0) {
        const f = (phase - 0.5) / 0.5;
        axons.forEach(a => { if (bits[a.k]) S("circle", { cx: L.lerp(a.x1, a.x2, f), cy: L.lerp(a.y1, a.y2, f), r: 6, fill: C.accent }, gPulse); });
      }
      const recv = val * w;
      const terms = axons.filter(a => bits[a.k]).map(a => `${(w * 2 ** a.k).toFixed(1)}`);
      document.getElementById("bits-recv").textContent = phase >= 1.0 ? (terms.length ? `${terms.join(" + ")} = ${recv.toFixed(1)} = ${val} × ${w}` : "nothing") : "…";
      document.getElementById("bits-ops").textContent = phase >= 0.5 ? bits.reduce((a, b) => a + b, 0) : "…";
      document.getElementById("bits-total").textContent = total;
      vT.textContent = v.toFixed(2);
      vBar.setAttribute("y", BY(v)); vBar.setAttribute("height", by0 + bh - BY(v));
      cells.forEach((cl, i) => {
        cl.r.setAttribute("stroke", i === step ? C.accent : C.line);
        cl.r.setAttribute("stroke-width", i === step ? 2 : 1);
        cl.o.setAttribute("fill", out[i] === undefined ? "none" : out[i] ? C.accent : Site.a(C.ink, .08));
      });
    }
    // chip order for the neuron: fire on the carried-in potential, reset, leak, then add the input
    function finishFrame() {
      if (frameDone || step < 0) return;
      frameDone = true;
      const { val, bits } = frameValues();
      const fired = v > TH;
      out[step] = fired;
      v = fired ? 0 : v;
      v -= v / TAU;
      v += val * w;
      total += bits.reduce((a, b) => a + b, 0);
      spikeMark.setAttribute("opacity", fired ? 1 : 0);
    }
    function nextFrame() {
      finishFrame();
      step = (step + 1) % COUNTS.length;
      if (step === 0) { v = 0; total = 0; out = []; }
      phase = 0; frameDone = false;
    }
    const ctrl = L.loop(box, dt => {
      if (step < 0) nextFrame();
      const before = phase;
      phase += dt * speed / 1.0;
      if (before < 1.0 && phase >= 1.0) finishFrame();
      if (phase >= 1.4) nextFrame();
      render();
      return true;
    }, { button: document.getElementById("bits-play"), autoplay: false });
    document.getElementById("bits-step").addEventListener("click", () => {
      ctrl.pause();
      nextFrame();
      phase = 1.01; finishFrame();
      render();
    });
    L.seg(document.getElementById("bits-B"), val => { B = +val; step = -1; phase = 0; v = 0; total = 0; out = []; frameDone = false; layout(); L.clear(gDots); render(); });
    L.seg(document.getElementById("bits-speed"), val => { speed = +val; });
  }

  /* ---------- E.4 costs (static) ---------- */
  function cost() {
    const box = document.getElementById("cost-vis");
    if (!box) return;
    const s = L.svg(box, 860, 260, "Costs of extra bits");
    const panels = [
      ["largest square frame", [89, 63, 51, 44], v => `${v} × ${v}`, 89],
      ["first-layer weight steps", [32767, 16383, 8191, 4095], v => L.fmt(v), 32767],
      ["inputs per first-layer neuron", [50, 100, 150, 200], v => String(v), 200]
    ];
    panels.forEach(([title, vals, f, mx], p) => {
      const x0 = 20 + p * 285;
      T(s, x0, 22, title, "lt-b");
      vals.forEach((v, i) => {
        const y = 44 + i * 48, wd = 150 * v / mx;
        T(s, x0, y + 18, `${i + 1} bit${i ? "s" : ""}`, "lt-ms");
        S("rect", { x: x0 + 50, y: y + 4, width: wd, height: 20, fill: i === 0 ? C.accent : Site.a(C.accent, .45) }, s);
        T(s, x0 + 56 + wd, y + 19, f(v), "lt-ms");
      });
    });
  }

  /* ---------- E.5 popcount table (static) ---------- */
  function popcount() {
    const box = document.getElementById("pop-vis");
    if (!box) return;
    const s = L.svg(box, 860, 210, "Counts, their bits, and the axon spikes they cost");
    T(s, 20, 54, "count", "lt-s");
    T(s, 20, 104, "bits (4s 2s 1s)", "lt-s");
    T(s, 20, 168, "axon spikes", "lt-s");
    for (let v = 0; v < 8; v++) {
      const x = 150 + v * 88;
      T(s, x + 30, 56, String(v), "lt-h mid");
      [2, 1, 0].forEach((k, j) => {
        const on = (v >> k) & 1;
        S("rect", { x: x + j * 21, y: 86, width: 18, height: 26, fill: on ? C.accent : C.cell, stroke: C.line }, s);
        T(s, x + j * 21 + 9, 104, String(on), on ? "lt-dark mid" : "lt-ms mid");
      });
      const pc = ((v >> 2) & 1) + ((v >> 1) & 1) + (v & 1);
      for (let i = 0; i < pc; i++) S("line", { x1: x + 12 + i * 14, y1: 178, x2: x + 12 + i * 14, y2: 150, stroke: C.accent, "stroke-width": 3 }, s);
      T(s, x + 30, 200, String(pc), "lt-m mid");
    }
  }
})();
