/* demo-why.js: pictures for page 0 (why this project exists). */
(function () {
  "use strict";
  const L = window.Learn;
  const { S, T, C } = L;

  L.onReady(() => { [vonNeumann, energy, arch].forEach(f => { try { f(); } catch (e) { console.error(f.name, e); } }); });

  function box(g, x, y, w, h, title, stroke) {
    S("rect", { x, y, width: w, height: h, fill: C.panel, stroke: stroke || C.lineStrong }, g);
    T(g, x + w / 2, y + 20, title, "lt-b mid");
  }

  /* weights crossing the bus, with two energy meters */
  function vonNeumann() {
    const el = document.getElementById("vn-vis");
    if (!el) return;
    const s = L.svg(el, 760, 230, "Weights travelling from memory to the processor");
    box(s, 20, 30, 180, 150, "memory");
    box(s, 560, 30, 180, 150, "processor");
    for (let i = 0; i < 24; i++) S("rect", { x: 40 + (i % 6) * 24, y: 62 + Math.floor(i / 6) * 24, width: 18, height: 18, fill: "rgba(224,169,79,.35)" }, s);
    T(s, 650, 110, "× and +", "lt-h mid");
    S("line", { x1: 200, y1: 105, x2: 560, y2: 105, stroke: C.line, "stroke-width": 10 }, s);
    T(s, 380, 90, "the bus: every weight crosses it", "lt-s mid");
    const pk = S("rect", { x: 200, y: 99, width: 12, height: 12, fill: C.accent }, s);
    const mv = S("rect", { x: 20, y: 200, width: 0, height: 10, fill: C.red }, s);
    const cp = S("rect", { x: 20, y: 216, width: 0, height: 10, fill: C.green }, s);
    const mvT = T(s, 740, 209, "", "lt-ms end"), cpT = T(s, 740, 225, "", "lt-ms end");
    let t = 0, n = 0;
    const MOVE = 640, COMP = 4.6, scale = 600 / (MOVE * 12);
    function draw() {
      pk.setAttribute("x", 200 + (t % 1) * 348);
      mv.setAttribute("width", Math.min(600, n * MOVE * scale));
      cp.setAttribute("width", Math.max(1, n * COMP * scale));
      mvT.textContent = `moving: ${L.fmt(n * MOVE)} pJ`;
      cpT.textContent = `computing: ${L.fmt(n * COMP, 1)} pJ`;
      const ro = document.getElementById("vn-readout");
      if (ro) ro.innerHTML = `${n} weights used · <b>${n ? Math.round(100 * MOVE / (MOVE + COMP)) : 0}%</b> of the energy went into moving them`;
    }
    const ctl = L.loop(el, dt => { t += dt * 1.2; if (t >= n + 1) n = Math.floor(t); draw(); return n < 12; },
      { button: document.getElementById("vn-play"), autoplay: false });
    document.getElementById("vn-restart").addEventListener("click", () => { t = 0; n = 0; draw(); ctl.play(); });
    draw();
  }

  /* energy per operation, log scale (Horowitz, ISSCC 2014, 45 nm) */
  function energy() {
    const el = document.getElementById("energy-vis");
    if (!el) return;
    const rows = [["read 32 bits from main memory (DRAM)", 640, C.red], ["read 32 bits from on-chip memory (SRAM)", 5, C.rust],
      ["multiply two 32-bit decimals", 3.7, C.accent], ["add two 32-bit decimals", 0.9, C.accent], ["add two 32-bit whole numbers", 0.1, C.green]];
    const s = L.svg(el, 760, 40 + rows.length * 36, "Energy per operation, log scale");
    const X = v => 330 + (Math.log10(v) + 1) / 4 * 400;
    [0.1, 1, 10, 100, 1000].forEach(v => { S("line", { x1: X(v), x2: X(v), y1: 10, y2: 20 + rows.length * 36, stroke: C.lineSoft }, s); T(s, X(v), 34 + rows.length * 36, `${v} pJ`, "lt-ms mid"); });
    rows.forEach(([lab, v, col], i) => {
      const y = 16 + i * 36;
      T(s, 320, y + 15, lab, "lt end");
      S("rect", { x: X(0.1), y, width: X(v) - X(0.1) + 2, height: 20, fill: col, opacity: 0.85 }, s);
      L.plate(T(s, X(v) + 6, y + 15, `${v} pJ`, "lt-m"));
    });
  }

  /* an ordinary computer against a neuromorphic chip */
  function arch() {
    const el = document.getElementById("arch-vis");
    if (!el) return;
    const s = L.svg(el, 760, 250, "An ordinary computer and a neuromorphic chip");
    T(s, 170, 20, "ordinary computer", "lt-b mid");
    box(s, 20, 40, 110, 150, "memory");
    box(s, 210, 40, 110, 150, "processor");
    S("line", { x1: 130, y1: 115, x2: 210, y2: 115, stroke: C.red, "stroke-width": 8 }, s);
    T(s, 170, 220, "every weight crosses the bus, every time", "lt-s mid");
    T(s, 570, 20, "neuromorphic chip", "lt-b mid");
    const r = L.rng(3);
    for (let i = 0; i < 6; i++) {
      const x = 400 + (i % 3) * 120, y = 40 + Math.floor(i / 3) * 80;
      S("rect", { x, y, width: 100, height: 64, fill: C.panel, stroke: C.lineStrong }, s);
      T(s, x + 50, y + 16, "neurons + memory", "lt-ms mid");
      for (let k = 0; k < 8; k++) S("circle", { cx: x + 14 + k * 10, cy: y + 40, r: 3.5, fill: r() < 0.2 ? C.accent : C.s4, opacity: 0.85 }, s);
    }
    S("path", { d: "M500,104 C520,120 540,120 560,112", fill: "none", stroke: C.accent, "stroke-width": 2 }, s);
    S("path", { d: "M460,104 C470,118 480,124 490,120", fill: "none", stroke: C.accent, "stroke-width": 2 }, s);
    T(s, 570, 220, "weights stay put; only spikes travel, as short address messages", "lt-s mid");
  }
})();
