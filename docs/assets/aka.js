/* aka.js: "In the search records" blocks.
   A block names a column of the search's records (val_accuracy, k_1, T, ...),
   gives a worked example, and draws where that name lives: a part of the
   network or a training curve.

   Markup:
     <aside class="aka" data-viz="net" data-hl="k_1">
       <div class="aka-head"><span class="aka-k">In the search records</span>
         <button class="aka-name" data-hl="k_0">k_0</button> ...</div>
       <p>worked example</p>
       <div class="aka-viz"></div>
     </aside>
   data-viz: net | curve | none. Hovering, focusing or clicking a name button
   with data-hl moves the highlight. Needs learn.js. */
(function () {
  "use strict";
  const L = window.Learn;
  const { S, T, C } = L;

  /* The network every thumbnail draws: the best network of the balanced block,
     a real trial of the DVS128 Gesture search (page C). */
  const BEST = {
    size: 40, T: 48, bits: 1, classes: 11, head: "flatten", fc: [128], signed: true,
    bias: "threshold", pool: null, norm: "BN",
    layers: [{ k: 5, ch: 71, ds: "stride" }, { k: 5, ch: 18, ds: "none" }, { k: 3, ch: 41, ds: "stride" }]
  };
  /* its validation and training accuracy after every epoch */
  const CURVE = {
    val: [0.431, 0.514, 0.542, 0.486, 0.514, 0.602, 0.505, 0.528, 0.431, 0.556, 0.588, 0.519, 0.569, 0.579, 0.63, 0.699, 0.671, 0.708, 0.63, 0.685, 0.694, 0.718, 0.713, 0.69, 0.713, 0.69, 0.727, 0.731, 0.713, 0.731, 0.769, 0.685, 0.731, 0.745, 0.764, 0.755, 0.736, 0.736, 0.75, 0.727, 0.718, 0.741, 0.731, 0.745, 0.764, 0.764, 0.736, 0.727, 0.741, 0.736, 0.741, 0.727, 0.745, 0.75, 0.745, 0.745, 0.745, 0.741, 0.755, 0.745, 0.759, 0.745, 0.736, 0.75, 0.745, 0.745, 0.736, 0.745, 0.736, 0.75, 0.755, 0.759, 0.755, 0.75, 0.745, 0.75, 0.731, 0.745, 0.75, 0.741],
    train: [0.483, 0.652, 0.699, 0.719, 0.732, 0.773, 0.764, 0.796, 0.805, 0.82, 0.826, 0.832, 0.836, 0.855, 0.858, 0.889, 0.911, 0.934, 0.921, 0.918, 0.917, 0.925, 0.931, 0.941, 0.944, 0.941, 0.946, 0.94, 0.951, 0.945, 0.947, 0.959, 0.95, 0.953, 0.952, 0.961, 0.968, 0.963, 0.97, 0.979, 0.984, 0.979, 0.982, 0.982, 0.985, 0.988, 0.992, 0.988, 0.991, 0.991, 0.992, 0.994, 0.993, 0.992, 0.995, 0.992, 0.991, 0.992, 0.995, 0.99, 0.997, 0.995, 0.994, 0.992, 0.992, 0.992, 0.989, 0.995, 0.993, 0.992, 0.992, 0.995, 0.993, 0.995, 0.991, 0.997, 0.993, 0.995, 0.997, 0.997],
    deployed: 0.745, floatBest: 0.679
  };

  /* map sizes, the way the tool plans them */
  function sizes(cfg) {
    let H = cfg.size;
    return cfg.layers.map(ly => {
      const s = ly.ds === "stride" ? 2 : 1, pad = ly.ds === "pool" ? (ly.k - 1) / 2 : 0;
      const conv = Math.floor((H + 2 * pad - ly.k) / s) + 1;
      const out = ly.ds === "pool" ? Math.floor(conv / 2) : conv;
      const r = { inH: H, conv, out };
      H = out;
      return r;
    });
  }

  /* which drawn parts each name lights up */
  function partsFor(hl, n) {
    const all = (p) => [...Array(n)].map((_, i) => p + i);
    const m = /^(k|ch|ds|stride|pool)_(\d+)$/.exec(hl || "");
    if (m) {
      const i = +m[2];
      return { k: ["kernel" + i], ch: ["slab" + i], ds: ["shrink" + i], stride: ["shrink" + i], pool: ["shrink" + i] }[m[1]];
    }
    const map = {
      resize_to: ["inputSize"], T: ["frames"], input_bits: ["bits"], data_C: ["frames"],
      depth: all("slab"), final_reduction: ["head"], fc_layers: ["fc"], fc_width_0: ["fc"],
      num_classes: ["out"], signed_spikes: ["sign"], pool_type: all("shrink"),
      tau: ["lif"], tau_mode: ["lif"], trainable_threshold: ["lif"], deployed_tau: ["lif", "fcLif"],
      norm: ["norm"], tdbn_alpha: ["norm"], bias_mode: ["bias"], dropout_rate: ["drop"],
      use_conv_dropout: ["cdrop"], conv_dropout: ["cdrop"],
      epochs: ["trainBar"], lr: ["lrCurve"], scheduler: ["lrCurve"], warmup_epochs: ["lrRamp"],
      phase: ["gridPhase"], grace_period: ["grace"],
      training: ["trainBar"], float_val_accuracy: ["floatPhase"], epochs_run: ["trainBar"], stopped_early: ["grace"],
      weight_decay: ["trainBar"], label_smoothing: ["out"], grad_clip: ["trainBar"], optimizer: ["trainBar"],
      loss: ["out"], use_augment: ["frames"], aug_shift: ["frames"], aug_cutout: ["frames"], aug_drop: ["frames"],
      use_rate_penalty: ["lif"], rate_penalty: ["lif"],
      val_accuracy: ["out"], hw_val_accuracy: ["out"], sim_val_accuracy: ["out"], pre_export_val_accuracy: ["out"],
      sim_disagreements: ["out"], quant_gap: ["out"], hw_test_accuracy: ["out"],
      deployable: all("slab").concat(["head", "fc", "out"]), feasible: all("slab").concat(["head", "fc", "out"]),
      synops_per_sample: all("slab").concat(["fc", "out"])
    };
    return map[hl] || [];
  }

  function arrow(g, x1, y1, x2, y2, color) {
    S("line", { x1, y1, x2: x2 - 5, y2, stroke: color || C.faint, "stroke-width": 1.3 }, g);
    S("path", { d: `M${x2},${y2} l-7,-4 l0,8 Z`, fill: color || C.faint }, g);
  }

  /* draw(el, cfg, opts) -> { set(hl) }. opts.mini: small, no labels; opts.train: add the training strip */
  function net(el, cfg, opts) {
    opts = opts || {};
    const mini = !!opts.mini;
    const n = cfg.layers.length, sz = sizes(cfg);
    const W = mini ? 360 : 900, Hn = mini ? 132 : 250, Ht = opts.train && !mini ? 120 : 0;
    const s = L.svg(el, W, Hn + Ht, opts.label || "Where this setting lives in the network");
    const parts = {};
    const reg = (key, node, base) => { (parts[key] = parts[key] || []).push({ node, base }); return node; };
    const cy = mini ? 70 : 112;
    const unit = mini ? 66 : 120;                      // pixel height of the input frame
    const hOf = h => Math.max(mini ? 16 : 22, unit * h / cfg.size);
    const dOf = ch => (mini ? 4 : 6) + (mini ? 2.4 : 4.5) * Math.log2(Math.max(2, ch));

    // input: a stack of frames
    const fx = mini ? 8 : 30, fw = unit;
    const nf = mini ? 4 : 6;
    for (let i = nf - 1; i >= 0; i--) {
      const o = i * (mini ? 3 : 5);
      reg("frames", S("rect", { x: fx + o, y: cy - fw / 2 - o, width: fw, height: fw, fill: i ? C.panel : "#151311", stroke: C.line }, s), "frame");
    }
    // a few event pixels on the front frame
    const r = L.rng(4);
    for (let i = 0; i < (mini ? 10 : 26); i++) {
      const px = fx + 4 + r() * (fw - 10), py = cy - fw / 2 + 4 + r() * (fw - 10);
      S("rect", { x: px, y: py, width: mini ? 3 : 4, height: mini ? 3 : 4, fill: r() < 0.5 ? C.accent : C.rust, opacity: 0.85 }, s);
    }
    reg("inputSize", S("rect", { x: fx, y: cy - fw / 2, width: fw, height: fw, fill: "none", stroke: "transparent", "stroke-width": 2.5 }, s), "outline");
    // bits: small axon ticks under the frame
    const bitsG = S("g", {}, s);
    for (let b = 0; b < 4; b++) {
      const on = b < cfg.bits;
      S("line", { x1: fx + 8 + b * (mini ? 7 : 12), y1: cy + fw / 2 + 4, x2: fx + 8 + b * (mini ? 7 : 12), y2: cy + fw / 2 + (mini ? 10 : 16), stroke: on ? C.muted : "rgba(255,255,255,.08)", "stroke-width": mini ? 2 : 3 }, bitsG);
    }
    reg("bits", bitsG, "group");
    if (!mini) {
      T(s, fx + fw / 2, cy + fw / 2 + 34, `${cfg.size} × ${cfg.size} input`, "lt-s mid");
      T(s, fx + fw / 2, cy + fw / 2 + 48, `${cfg.T} frames, ${cfg.bits} bit${cfg.bits > 1 ? "s" : ""}`, "lt-s mid");
    }

    // layers
    let x = fx + fw + (mini ? 22 : 46) + nf * (mini ? 3 : 5);
    let prev = { x: fx, y: cy - fw / 2, w: fw, h: fw };
    const gap = mini ? 24 : 62;
    cfg.layers.forEach((ly, i) => {
      const h = hOf(sz[i].out), d = dOf(ly.ch), w = Math.max(mini ? 10 : 14, h * 0.24);
      // kernel window on the previous map, with a cone to this layer
      const kh = Math.max(mini ? 5 : 8, (i === 0 ? fw : prev.h) * ly.k / (i === 0 ? cfg.size : sz[i - 1].out));
      const kx = prev.x + prev.w * 0.55 - kh / 2, ky = prev.y + prev.h * 0.35 - kh / 2;
      const kg = S("g", {}, s);
      S("rect", { x: kx, y: ky, width: kh, height: kh, fill: "rgba(255,255,255,.05)", stroke: C.faint }, kg);
      S("line", { x1: kx + kh, y1: ky, x2: x, y2: cy - h / 4, stroke: C.faint, "stroke-width": 0.8, "stroke-dasharray": "3 3" }, kg);
      S("line", { x1: kx + kh, y1: ky + kh, x2: x, y2: cy - h / 4 + 2, stroke: C.faint, "stroke-width": 0.8, "stroke-dasharray": "3 3" }, kg);
      reg("kernel" + i, kg, "kernel");
      // the layer: a slab of channels
      const sg = S("g", {}, s);
      const layersN = mini ? 3 : 4;
      for (let j = layersN - 1; j >= 0; j--) {
        const o = j * d / layersN;
        S("rect", { x: x + o, y: cy - h / 2 - o, width: w, height: h, fill: j ? C.panel : "#1a1714", stroke: C.lineStrong }, sg);
      }
      reg("slab" + i, sg, "slab");
      // the neurons: norm then LIF, as a small two-part tag under the slab
      if (!mini) {
        const ty = cy + h / 2 + 14;
        const nb = S("g", {}, s);
        S("rect", { x: x - 6, y: ty, width: 34, height: 16, fill: C.panel, stroke: C.line }, nb);
        T(nb, x + 11, ty + 12, "norm", "lt-ms mid");
        reg("norm", nb, "group");
        const lb = S("g", {}, s);
        S("rect", { x: x + 30, y: ty, width: 30, height: 16, fill: C.panel, stroke: C.line }, lb);
        T(lb, x + 45, ty + 12, "LIF", "lt-ms mid");
        reg("lif", lb, "group");
        if (cfg.signed) reg("sign", T(s, x + w + d + 4, cy - h / 2 - d + 4, "±", "lt-a"), "text");
        T(s, x + w / 2 + d / 2, cy - h / 2 - d - 10, `layer ${i + 1}`, "lt-b mid");
        T(s, x + w / 2 + d / 2, ty + 34, `${ly.ch} ch, ${sz[i].out}×${sz[i].out}`, "lt-s mid");
      } else {
        T(s, x + w / 2 + d / 2, cy + h / 2 + 14, String(i + 1), "lt-ms mid");
        reg("lif", S("circle", { cx: x + w / 2 + d / 2, cy: cy - h / 2 - d - 5, r: 3, fill: C.faint }, s), "dot");
        reg("norm", S("rect", { x: x + w / 2 + d / 2 - 4, y: cy + h / 2 + 18, width: 8, height: 4, fill: C.faint }, s), "dot");
        if (cfg.signed) reg("sign", S("circle", { cx: x + w + d, cy: cy - h / 2 - d, r: 2.5, fill: C.faint }, s), "dot");
      }
      // how the map shrinks, on the way out
      const ax1 = x + w + d + (mini ? 3 : 6), ax2 = ax1 + gap - (mini ? 6 : 12);
      arrow(s, ax1, cy, ax2, cy);
      const shG = S("g", {}, s);
      const lab = ly.ds === "stride" ? "stride 2" : ly.ds === "pool" ? (cfg.pool === "max" ? "max pool" : "pool") : "no shrink";
      if (!mini) {
        T(shG, (ax1 + ax2) / 2, cy - 18, lab, "lt-ms mid");
      } else {
        S("rect", { x: (ax1 + ax2) / 2 - 5, y: cy - 12, width: 10, height: 6, fill: ly.ds === "none" ? "rgba(255,255,255,.08)" : C.faint }, shG);
      }
      reg("shrink" + i, shG, "group");
      if (i < n - 1) {
        const cd = S("g", {}, s);
        for (let q = 0; q < 2; q++) T(cd, (ax1 + ax2) / 2 - 6 + q * 12, cy + (mini ? 16 : 20), "×", mini ? "lt-ms mid" : "lt-s mid", { opacity: 0 });
        reg("cdrop", cd, "ghost");
      }
      prev = { x, y: cy - h / 2, w, h };
      x = ax2 + 6;
    });

    // head
    const hw = mini ? 22 : 64, hh = mini ? 40 : 86;
    const hg = S("g", {}, s);
    S("rect", { x, y: cy - hh / 2, width: hw, height: hh, fill: C.panel, stroke: C.lineStrong }, hg);
    if (!mini) {
      T(hg, x + hw / 2, cy - 4, cfg.head === "gap" ? "GAP" : "flatten", "lt-b mid");
      const last = sz[n - 1].out, chL = cfg.layers[n - 1].ch;
      T(hg, x + hw / 2, cy + 14, cfg.head === "gap" ? `${chL}` : `${(chL * last * last).toLocaleString("en-US")}`, "lt-ms mid");
      T(s, x + hw / 2, cy + hh / 2 + 18, "head", "lt-s mid");
    }
    reg("head", hg, "group");
    const dg = S("g", {}, s);
    for (let q = 0; q < 3; q++) T(dg, x + hw + (mini ? 6 : 12), cy - 14 + q * 14, "×", "lt-ms mid", { opacity: 0 });
    reg("drop", dg, "ghost");
    x += hw + (mini ? 14 : 28);
    // hidden layer
    if (cfg.fc && cfg.fc.length) {
      const fg = S("g", {}, s), k = mini ? 5 : 8;
      for (let q = 0; q < k; q++) S("circle", { cx: x + 6, cy: cy - (k - 1) * (mini ? 7 : 10) / 2 + q * (mini ? 7 : 10), r: mini ? 2.6 : 3.6, fill: C.s4 }, fg);
      reg("fc", fg, "group");
      if (!mini) { T(s, x + 6, cy + 58, `${cfg.fc[0]}`, "lt-s mid"); T(s, x + 6, cy + 72, "hidden", "lt-s mid"); reg("fcLif", T(s, x + 6, cy - 50, "LIF", "lt-ms mid"), "text"); }
      x += mini ? 26 : 54;
    }
    // outputs
    const og = S("g", {}, s), k = cfg.classes;
    for (let q = 0; q < k; q++) S("circle", { cx: x + 6, cy: cy - (k - 1) * (mini ? 4.6 : 8) / 2 + q * (mini ? 4.6 : 8), r: mini ? 1.9 : 3, fill: q === 3 ? C.accent : C.s4 }, og);
    reg("out", og, "group");
    if (!mini) { T(s, x + 6, cy + 58, `${k}`, "lt-s mid"); T(s, x + 6, cy + 72, "classes", "lt-s mid"); }
    // a mini drawing is cropped to what it drew, so it fills the width it gets
    const Wv = mini ? Math.ceil(x + 18) : W;
    if (mini) s.setAttribute("viewBox", `0 0 ${Wv} ${Hn}`);
    // bias: a small always-on node above the layers
    const bg = S("g", {}, s);
    if (!mini) {
      T(bg, fx + fw + 70, 22, cfg.bias === "neuron" ? "bias neurons, one connection to every neuron" : "bias folded into each threshold (θ − b)", "lt-ms", { opacity: 0 });
    }
    reg("bias", bg, "ghost");

    // training strip
    if (Ht) {
      const y0 = Hn + 10, x0 = 70, x1 = W - 40, E = 80, X = e => x0 + (x1 - x0) * e / E;
      T(s, 20, y0 + 14, "training", "lt-b");
      const bar = S("g", {}, s);
      S("rect", { x: X(0), y: y0 + 74, width: X(20) - X(0), height: 12, fill: "rgba(236,231,222,.25)" }, bar);
      const gp = S("rect", { x: X(20), y: y0 + 74, width: X(80) - X(20), height: 12, fill: "rgba(224,169,79,.25)" }, bar);
      reg("trainBar", bar, "group");
      reg("floatPhase", bar.firstChild, "fillF");
      reg("gridPhase", gp, "fillA");
      T(s, X(10), y0 + 104, "ordinary weights", "lt-s mid");
      T(s, X(50), y0 + 104, "integer weights", "lt-s mid");
      [0, 20, 80].forEach(e => T(s, X(e), y0 + 68, String(e), "lt-ms mid"));
      reg("grace", T(s, X(30), y0 + 68, "30", "lt-ms mid", { opacity: 0 }), "textGhost");
      // learning rate: flat for this trial, halved on the integer weights
      const lr = S("polyline", { points: `${X(0)},${y0 + 30} ${X(20)},${y0 + 30} ${X(20)},${y0 + 46} ${X(80)},${y0 + 46}`, fill: "none", stroke: C.faint, "stroke-width": 2 }, s);
      reg("lrCurve", lr, "line");
      reg("lrRamp", S("line", { x1: X(0), y1: y0 + 56, x2: X(2), y2: y0 + 30, stroke: "transparent", "stroke-width": 2 }, s), "lineGhost");
      T(s, X(50), y0 + 40, "learning rate, halved at the switch", "lt-ms mid");
    }

    function paint(on, p) {
      const col = C.accent;
      if (p.base === "slab") p.node.querySelectorAll("rect").forEach((r, i) => { r.setAttribute("stroke", on ? col : C.lineStrong); r.setAttribute("stroke-width", on ? 2 : 1); });
      else if (p.base === "kernel") p.node.querySelectorAll("rect,line").forEach(r => { r.setAttribute("stroke", on ? col : C.faint); if (r.tagName === "rect") r.setAttribute("fill", on ? "rgba(224,169,79,.35)" : "rgba(255,255,255,.05)"); });
      else if (p.base === "frame") { p.node.setAttribute("stroke", on ? col : C.line); }
      else if (p.base === "outline") p.node.setAttribute("stroke", on ? col : "transparent");
      else if (p.base === "group") p.node.querySelectorAll("rect,circle,line,text").forEach(e => {
        if (e.tagName === "text") e.style.fill = on ? col : "";
        else if (e.tagName === "line") e.setAttribute("stroke", on ? col : (e.dataset.base || e.getAttribute("stroke")));
        else if (e.tagName === "circle") e.setAttribute("fill", on ? col : (e.dataset.base || e.getAttribute("fill")));
        else e.setAttribute("stroke", on ? col : (e.dataset.base || "none"));
      });
      else if (p.base === "ghost") p.node.querySelectorAll("text").forEach(e => { e.setAttribute("opacity", on ? 1 : 0); e.style.fill = on ? col : ""; });
      else if (p.base === "text") p.node.style.fill = on ? col : "";
      else if (p.base === "textGhost") { p.node.setAttribute("opacity", on ? 1 : 0); p.node.style.fill = on ? col : ""; }
      else if (p.base === "dot") p.node.setAttribute("fill", on ? col : C.faint);
      else if (p.base === "fillA") p.node.setAttribute("fill", on ? "rgba(224,169,79,.7)" : "rgba(224,169,79,.25)");
      else if (p.base === "fillF") p.node.setAttribute("fill", on ? "rgba(155,177,122,.75)" : "rgba(236,231,222,.25)");
      else if (p.base === "line") { p.node.setAttribute("stroke", on ? col : C.faint); p.node.setAttribute("stroke-width", on ? 3 : 2); }
      else if (p.base === "lineGhost") p.node.setAttribute("stroke", on ? col : "transparent");
    }
    // remember each element's resting color so a group can be restored
    s.querySelectorAll("line,circle,rect").forEach(e => {
      const a = e.tagName === "circle" ? "fill" : "stroke";
      // an element drawn without an outline goes back to none, not to a default
      e.dataset.base = e.getAttribute(a) || "none";
    });
    let lit = [];
    const tag = T(s, 0, 0, "", mini ? "lt-a" : "lt-a", { opacity: 0 });
    tag.style.fontWeight = "600";
    function set(hl) {
      lit.forEach(k => (parts[k] || []).forEach(p => paint(false, p)));
      lit = partsFor(hl, n);
      lit.forEach(k => (parts[k] || []).forEach(p => paint(true, p)));
      const first = lit.length && parts[lit[0]] && parts[lit[0]][0];
      tag.setAttribute("opacity", 0);
      if (!first || !hl) return;
      try {
        const b = first.node.getBBox();
        if (!b.width && !b.height) return;
        tag.textContent = hl;
        tag.setAttribute("opacity", 1);
        place(b);
      } catch (e) { /* not rendered yet */ }
    }
    /* the label goes beside the lit part, in the first spot where it covers no
       other label, crosses no line and does not cut a box edge */
    function obstacles() {
      const texts = [...s.querySelectorAll("text")].filter(t => t !== tag && t.textContent.trim() &&
        t.getAttribute("opacity") !== "0").map(t => t.getBBox());
      const pts = [];
      s.querySelectorAll("line, polyline").forEach(e => {
        if (e.getAttribute("stroke") === "transparent") return;
        let p = [];
        if (e.tagName === "line") p = [[+e.getAttribute("x1"), +e.getAttribute("y1")], [+e.getAttribute("x2"), +e.getAttribute("y2")]];
        else p = (e.getAttribute("points") || "").trim().split(/\s+/).map(q => q.split(",").map(Number));
        for (let i = 0; i + 1 < p.length; i++) for (let t = 0; t <= 1; t += 0.1) pts.push([p[i][0] + (p[i + 1][0] - p[i][0]) * t, p[i][1] + (p[i + 1][1] - p[i][1]) * t]);
      });
      const rects = [...s.querySelectorAll("rect")].filter(r => {
        const st = r.getAttribute("stroke");
        return st && st !== "none" && st !== "transparent";
      }).map(r => r.getBBox()).filter(r => r.width > 12 && r.height > 10);
      return { texts, pts, rects };
    }
    function place(b) {
      const ob = obstacles();
      const cx = b.x + b.width / 2, cy = b.y + b.height / 2;
      const top = mini ? 15 : 17;                       // the strip above every drawing is empty
      const spots = [[cx, b.y - 6, "middle"], [cx, top, "middle"], [cx, b.y + b.height + 13, "middle"],
                     [b.x + b.width + 5, cy + 4, "start"], [b.x - 5, cy + 4, "end"],
                     [cx, b.y - 20, "middle"], [cx, b.y + b.height + 27, "middle"],
                     [cx - 30, top, "middle"], [cx + 30, top, "middle"]];
      const over = (a, c) => a.x < c.x + c.width && c.x < a.x + a.width && a.y < c.y + c.height && c.y < a.y + a.height;
      const inside = (o, i) => i.x >= o.x && i.y >= o.y && i.x + i.width <= o.x + o.width && i.y + i.height <= o.y + o.height;
      let best = null;
      for (const [x, y, anchor] of spots) {
        tag.setAttribute("x", x); tag.setAttribute("y", y); tag.setAttribute("text-anchor", anchor);
        const t = tag.getBBox();
        if (t.x < 0 || t.y < 0 || t.x + t.width > Wv || t.y + t.height > Hn + Ht) continue;
        const core = { x: t.x + 1, y: t.y + 2, width: t.width - 2, height: t.height - 4 };
        let bad = ob.texts.filter(o => over(core, o)).length;
        bad += ob.pts.filter(([px, py]) => px > core.x && px < core.x + core.width && py > core.y && py < core.y + core.height).length ? 1 : 0;
        bad += ob.rects.filter(r => over(core, r) && !inside(r, core)).length;
        if (!best || bad < best.bad) best = { x, y, anchor, bad };
        if (bad === 0) break;
      }
      if (!best) best = { x: Math.max(4, Math.min(Wv - 4, cx)), y: Math.max(12, b.y - 6), anchor: "middle" };
      tag.setAttribute("x", best.x); tag.setAttribute("y", best.y); tag.setAttribute("text-anchor", best.anchor);
    }
    set(opts.hl);
    return { set };
  }

  /* curve(el): the best trial's validation (amber) and training (pale) accuracy */
  function curve(el, spec) {
    spec = spec || {};
    const W = 640, H = 236, x0 = 46, x1 = W - 20, y0 = 34, y1 = H - 40;
    const s = L.svg(el, W, H, "Training and validation accuracy of one trial, epoch by epoch");
    const X = e => x0 + (x1 - x0) * e / 80, Y = v => y1 - (y1 - y0) * (v - 0.4) / 0.6;
    [0.4, 0.6, 0.8, 1.0].forEach(v => { S("line", { x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: C.lineSoft }, s); T(s, x0 - 6, Y(v) + 4, (v * 100) + "%", "lt-ms end"); });
    [0, 20, 40, 60, 80].forEach(e => T(s, X(e), y1 + 16, String(e), "lt-ms mid"));
    T(s, (x0 + x1) / 2, H - 6, "epoch", "lt-ms mid");
    S("rect", { x: X(0), y: y0, width: X(20) - X(0), height: y1 - y0, fill: "rgba(236,231,222,.04)" }, s);
    T(s, X(10), y0 - 10, "ordinary weights", "lt-ms mid");
    T(s, X(50), y0 - 10, "integer weights", "lt-ms mid");
    const line = (arr, col, w) => S("polyline", { points: arr.map((v, e) => `${X(e + 0.5)},${Y(v)}`).join(" "), fill: "none", stroke: col, "stroke-width": w }, s);
    if (spec.train !== false) line(CURVE.train, C.s1, 1.6);
    line(CURVE.val, C.accent, 2);
    if (spec.mark === "float") {
      S("line", { x1: X(0), x2: X(20), y1: Y(CURVE.floatBest), y2: Y(CURVE.floatBest), stroke: C.green, "stroke-dasharray": "4 3", "stroke-width": 1.5 }, s);
      L.plate(T(s, X(20) + 4, Y(CURVE.floatBest) + 14, "warm-up score 0.679", "lt-g"));
    }
    if (spec.mark !== "none") {
      S("circle", { cx: X(80), cy: Y(CURVE.deployed), r: 4, fill: C.accent }, s);
      L.plate(T(s, X(80) - 6, Y(CURVE.deployed) + 18, "deployed 0.745", "lt-a end"));
    }
    L.plate(T(s, X(62), Y(0.985) + 16, "training", "lt-ms"));
    L.plate(T(s, X(40), Y(0.66), "validation", "lt-a"));
  }

  function init(root) {
    (root || document).querySelectorAll(".aka").forEach(box => {
      if (box.dataset.ready) return;
      box.dataset.ready = "1";
      const viz = box.querySelector(".aka-viz");
      if (!viz) return;
      const kind = box.dataset.viz;
      try {
        if (kind === "net") {
          const ctl = net(viz, BEST, { mini: box.dataset.size !== "full", hl: box.dataset.hl, train: box.dataset.train === "1" });
          const names = [...box.querySelectorAll("[data-hl]")].filter(b => b !== box);
          names.forEach(b => {
            const desc = box.querySelector(".aka-desc");
            const go = () => {
              names.forEach(o => o.classList.toggle("on", o === b));
              ctl.set(b.dataset.hl);
              if (desc && b.dataset.desc) desc.innerHTML = `<code class="aka-name">${b.textContent}</code> ${b.dataset.desc}`;
            };
            b.addEventListener("mouseenter", go); b.addEventListener("focus", go); b.addEventListener("click", go);
          });
          const first = names.find(b => b.dataset.hl === box.dataset.hl);
          if (first) { first.classList.add("on"); first.dispatchEvent(new Event("focus")); }
        } else if (kind === "curve") {
          curve(viz, { mark: box.dataset.mark, train: box.dataset.trainline !== "0" });
        }
      } catch (e) { console.error("aka", e); }
    });
  }

  window.Aka = { net, curve, BEST, CURVE, sizes, init };
  L.onReady(() => init());
})();
