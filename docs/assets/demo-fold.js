/* demo-fold.js: what folding batch normalization does, drawn (page G). */
(function () {
  "use strict";
  const L = window.Learn;
  const { S, T, C } = L;

  L.onReady(() => { try { fold(); } catch (e) { console.error("fold", e); } });

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

  /* ---------- folding (static) ---------- */
  function fold() {
    const boxEl = document.getElementById("fold-vis");
    if (!boxEl) return;
    const s = L.svg(boxEl, 840, 270, "Folding batch normalization into the weights and threshold");
    T(s, 20, 24, "Trained", "lt-a");
    box(s, 20, 36, 220, 64, ["convolution", "weights w"]);
    arrow(s, 240, 68, 290, 68);
    box(s, 290, 36, 260, 64, ["batch normalization", "× scale, + shift, per channel"], { stroke: C.rust });
    arrow(s, 550, 68, 600, 68);
    box(s, 600, 36, 220, 64, ["spiking neuron", "fires above threshold θ"]);
    T(s, 20, 160, "Deployed", "lt-a");
    box(s, 20, 172, 220, 64, ["convolution", "weights w × scale"], { stroke: C.accent });
    arrow(s, 240, 204, 600, 204);
    box(s, 600, 172, 220, 64, ["spiking neuron", "threshold θ − shift"], { stroke: C.accent });
    arrow(s, 130, 104, 130, 166, C.accent, 1.2, "4 4");
    arrow(s, 420, 104, 420, 196, C.accent, 1.2, "4 4");
    arrow(s, 470, 104, 690, 166, C.accent, 1.2, "4 4");
    T(s, 140, 140, "scale folds into the weights", "lt-s");
    T(s, 830, 128, "shift becomes a threshold change", "lt-s end");
    T(s, 830, 142, "(or a bias neuron)", "lt-s end");
    T(s, 420, 258, "no normalization step left on the chip", "lt-s mid");
  }

})();
