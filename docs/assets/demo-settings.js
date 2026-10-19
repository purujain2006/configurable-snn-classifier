/* demo-settings.js: visuals for page D (the network builder and one card per setting). */
(function () {
  "use strict";
  const L = window.Learn;
  const { S, T, C } = L;

  L.onReady(() => {
    [builder, cards, ranked].forEach(f => { try { f(); } catch (e) { console.error(f.name, e); } });
  });

  function arrow(g, x1, y1, x2, y2, color, w) {
    S("line", { x1, y1, x2, y2, stroke: color || C.faint, "stroke-width": w || 1.2 }, g);
    const a = Math.atan2(y2 - y1, x2 - x1), s = 6;
    S("path", { d: `M${x2},${y2} L${x2 - s * Math.cos(a - 0.4)},${y2 - s * Math.sin(a - 0.4)} L${x2 - s * Math.cos(a + 0.4)},${y2 - s * Math.sin(a + 0.4)} Z`, fill: color || C.faint }, g);
  }

  /* ---------- sizes and the chip check, as every trial computes them ---------- */
  const LIM = { axons: 16000, axFanOut: 4096, axFanIn: 8191, nFanOut: 4095, nFanIn: 8159 };
  function plan(c) {
    const sign = c.signed ? 2 : 1, folds = c.pool !== "max", biasN = c.bias === "neuron";
    let H = c.size, Cin = 2;
    const blocks = [];
    for (let i = 0; i < c.layers.length; i++) {
      const ly = c.layers[i], k = ly.k;
      let s = 1, pad = 0, pool = false;
      if (ly.ds === "stride") s = 2;
      else if (ly.ds === "pool") { pad = (k - 1) / 2; pool = true; }
      const conv = Math.floor((H + 2 * pad - k) / s) + 1;
      const out = pool ? Math.floor(conv / 2) : conv;
      if (conv < 1 || out < 1) return { error: `Layer ${i + 1} shrinks a ${H} × ${H} map to nothing with a ${k} × ${k} kernel. Use a larger frame, a smaller kernel, or less shrinking.` };
      blocks.push({ k, s, pool, Cin, Cout: ly.ch, inH: H, conv, out, ds: ly.ds });
      H = out; Cin = ly.ch;
    }
    const checks = [];
    const add = (label, v, lim) => checks.push({ label, v, lim, ok: v <= lim });
    let prevPool = 1;
    blocks.forEach((b, i) => {
      const scale = prevPool * prevPool, biasIn = biasN ? scale : 0;
      const fanOut = Math.ceil(b.k / b.s) ** 2 * b.Cout * sign;
      if (i === 0) {
        add("input axons", b.inH * b.inH * 2 * c.bits, LIM.axons);
        add("layer 1: input axons per neuron", b.k * b.k * 2 * c.bits, LIM.axFanIn);
        add("each input axon reaches", fanOut, LIM.axFanOut);
      } else {
        add(`layer ${i}: neurons each one reaches`, fanOut, LIM.nFanOut);
        add(`layer ${i + 1}: inputs per neuron`, b.k * b.k * b.Cin * scale * sign + biasIn, LIM.nFanIn);
      }
      prevPool = b.pool && folds ? 2 : 1;
    });
    const last = blocks[blocks.length - 1];
    add(`layer ${blocks.length}: neurons each one reaches`, (c.fc || 11) * sign, LIM.nFanOut);
    let fanIn, pool;
    if (last.pool && !folds) { fanIn = last.Cout * last.out * last.out; pool = 1; }
    else { fanIn = last.Cout * last.conv * last.conv; pool = last.pool ? Math.floor(last.conv / last.out) : 1; }
    if (c.head === "gap") pool *= last.out;
    add(c.fc ? "hidden layer: inputs per neuron" : "classifier: inputs per neuron", fanIn * sign + (biasN ? pool * pool : 0), LIM.nFanIn);
    if (c.fc) {
      add("hidden layer: neurons each one reaches", 11 * sign, LIM.nFanOut);
      add("classifier: inputs per neuron", c.fc * sign + (biasN ? 1 : 0), LIM.nFanIn);
    }
    // size, counted the way the chip holds it
    let neurons = 0, conns = 0, params = 0;
    blocks.forEach((b, i) => {
      const n = b.Cout * b.conv * b.conv;
      neurons += n * sign;
      conns += n * b.k * b.k * b.Cin * (i === 0 ? c.bits : sign) * sign;
      params += b.k * b.k * b.Cin * b.Cout;
      if (b.pool) {
        if (!folds) neurons += b.Cout * b.out * b.out * sign;
        conns += b.Cout * b.out * b.out * 4 * sign;
      }
    });
    if (c.head === "gap") conns += last.Cout * last.out * last.out;
    const inF = c.head === "gap" ? last.Cout : last.Cout * last.out * last.out;
    const lins = c.fc ? [[inF, c.fc], [c.fc, 11]] : [[inF, 11]];
    lins.forEach(([a, b]) => { neurons += b * sign; conns += a * b * sign * sign; params += a * b; });
    return { blocks, checks, ok: checks.every(x => x.ok), neurons, conns, params, inF };
  }

  /* ---------- drawing a network ---------- */
  const side = n => 12 + Math.sqrt(n) * 11;
  function slab(g, x, cy, s, depth, opts) {
    opts = opts || {};
    const off = 5, n = Math.max(1, depth);
    for (let d = n - 1; d >= 0; d--) {
      S("rect", { x: x + d * off, y: cy - s / 2 - d * off, width: s, height: s, fill: d === 0 ? (opts.fill || "rgba(236,231,222,.08)") : "rgba(0,0,0,.7)", stroke: opts.stroke || C.lineStrong, "stroke-width": d === 0 ? 1.2 : 0.8 }, g);
    }
    return { x, y: cy - s / 2, s, w: s + (n - 1) * off };
  }
  const depthOf = ch => L.clamp(Math.round(Math.log2(ch)) - 1, 1, 7);
  function drawNet(boxEl, c, P) {
    const H = 310, cy = 150;
    let x = 24;
    const parts = [];
    // input
    parts.push({ kind: "in", s: side(c.size), depth: 2, label: [`input ${c.size} × ${c.size}`, `2 channels${c.bits > 1 ? ` × ${c.bits} bits` : ""}`, `T = ${c.T} frames`] });
    if (!P.error) P.blocks.forEach((b, i) => parts.push({ kind: "conv", b, i, s: side(b.out), pre: b.pool ? side(b.conv) : 0, depth: depthOf(b.Cout),
      label: [`${b.Cout} × ${b.out} × ${b.out}`, `k${b.k} ${b.ds === "stride" ? "stride 2" : b.ds === "pool" ? (c.pool === "max" ? "+ max pool" : "+ avg pool") : "no shrink"}`] }));
    let total = 24;
    parts.forEach(p => { p.w = Math.max(p.s + (p.depth - 1) * 5, p.pre || 0); total += p.w + 56; });
    total += 260;
    const W = Math.max(880, total);
    const s = L.svg(boxEl, W, H, "The network as built");
    if (P.error) { T(s, 24, 140, P.error, "lt-bad"); return; }
    let prev = null;
    parts.forEach((p, k) => {
      const top = cy - p.s / 2;
      if (p.pre) S("rect", { x, y: cy - p.pre / 2, width: p.pre, height: p.pre, fill: "none", stroke: C.faint, "stroke-dasharray": "3 3" }, s);
      const fill = p.kind === "in" ? "rgba(224,169,79,.1)" : "rgba(236,231,222,.08)";
      const r = slab(s, x + (p.pre ? (p.pre - p.s) / 2 : 0), cy, p.s, p.depth, { fill, stroke: p.kind === "in" ? C.accent : C.lineStrong });
      if (c.signed && p.kind === "conv") T(s, r.x + r.w - 2, top - (p.depth - 1) * 5 - 5, "±", "lt-a end");
      if (c.bias === "neuron" && p.kind === "conv") {
        S("circle", { cx: r.x + p.s / 2, cy: 26, r: 7, fill: C.panel, stroke: C.rust }, s);
        T(s, r.x + p.s / 2, 30, "b", "lt-r mid");
        arrow(s, r.x + p.s / 2, 33, r.x + p.s / 2, top - (p.depth - 1) * 5 - 3, C.rust, 1);
      }
      p.label.forEach((t, j) => T(s, r.x + p.s / 2, cy + Math.max(p.s, p.pre || 0) / 2 + 18 + j * 14, t, j === 0 ? "lt-m mid" : "lt-ms mid"));
      if (prev) {
        // kernel window on the previous block, with lines to this block
        const b = p.b, ks = Math.max(3, prev.r.s * b.k / prev.H);
        const kx = prev.r.x + prev.r.s * 0.55 - ks / 2, ky = cy - ks / 2 - prev.r.s * 0.12;
        S("rect", { x: kx, y: ky, width: ks, height: ks, fill: "rgba(224,169,79,.25)", stroke: C.accent }, s);
        const tx = r.x + p.s * 0.45, ty = cy - p.s * 0.1;
        S("line", { x1: kx + ks, y1: ky, x2: tx, y2: ty, stroke: C.accent, "stroke-opacity": .5 }, s);
        S("line", { x1: kx + ks, y1: ky + ks, x2: tx, y2: ty, stroke: C.accent, "stroke-opacity": .5 }, s);
        S("circle", { cx: tx, cy: ty, r: 2.2, fill: C.accent }, s);
      }
      prev = { r, H: p.kind === "in" ? c.size : p.b.out };
      x += p.w + 56;
    });
    // head
    const last = P.blocks[P.blocks.length - 1];
    const hx = x - 20, len = c.head === "gap" ? 20 + 8 * Math.log2(last.Cout) : Math.min(180, 30 + 16 * Math.log2(P.inF));
    S("rect", { x: hx, y: cy - len / 2, width: 12, height: len, fill: "rgba(236,231,222,.15)", stroke: C.lineStrong }, s);
    T(s, hx + 6, cy + len / 2 + 18, c.head === "gap" ? "GAP" : "flatten", "lt-m mid");
    T(s, hx + 6, cy + len / 2 + 32, L.fmt(P.inF), "lt-ms mid");
    let fx = hx + 12;
    function column(xc, n, label, sub) {
      const k = Math.min(n, 11), ys = [];
      for (let i = 0; i < k; i++) { const y = cy + (i - (k - 1) / 2) * 13; ys.push(y); S("circle", { cx: xc, cy: y, r: 4.5, fill: C.panel, stroke: c.signed ? C.accent : C.lineStrong }, s); }
      T(s, xc, cy + (k - 1) * 6.5 + 22, label, "lt-m mid");
      if (sub) T(s, xc, cy + (k - 1) * 6.5 + 36, sub, "lt-ms mid");
      return ys;
    }
    if (c.fc) {
      const xc = fx + 70, ys = column(xc, 8, `${c.fc}`, "hidden");
      ys.forEach(y => S("line", { x1: fx, y1: cy, x2: xc - 5, y2: y, stroke: C.faint, "stroke-opacity": .3 }, s));
      fx = xc + 5;
    }
    const xo = fx + 80, yo = column(xo, 11, "11 classes");
    yo.forEach(y => S("line", { x1: fx, y1: cy, x2: xo - 5, y2: y, stroke: C.faint, "stroke-opacity": .25 }, s));
  }

  function builder() {
    const vis = document.getElementById("b-vis");
    if (!vis) return;
    const CH = [8, 12, 16, 24, 32, 48, 64, 96, 128];
    const cfg = { size: 40, T: 48, bits: 1, depth: 3, pool: "avg", head: "flatten", fc: 128, signed: true, bias: "threshold",
      layers: [{ ch: 64, k: 5, ds: "stride" }, { ch: 16, k: 5, ds: "none" }, { ch: 48, k: 3, ds: "stride" }, { ch: 32, k: 3, ds: "none" }] };
    const wrap = document.getElementById("b-layers");
    const rows = cfg.layers.map((ly, i) => {
      const row = document.createElement("div");
      row.className = "builder-layer";
      const sel = (lab, opts, val, key) => {
        const d = document.createElement("div"); d.className = "ctl";
        const id = `b-l${i}-${key}`;
        d.innerHTML = `<label for="${id}">${lab}</label><select id="${id}">${opts.map(o => `<option value="${o[0]}"${String(o[0]) === String(val) ? " selected" : ""}>${o[1]}</option>`).join("")}</select>`;
        d.querySelector("select").addEventListener("change", e => { ly[key] = key === "ds" ? e.target.value : +e.target.value; render(); });
        return d;
      };
      const name = document.createElement("span"); name.className = "bl-name"; name.textContent = `layer ${i + 1}`;
      row.append(name, sel("channels", CH.map(v => [v, v]), ly.ch, "ch"), sel("kernel", [3, 5, 7, 9].map(v => [v, `${v} × ${v}`]), ly.k, "k"),
        sel("shrink by", [["stride", "stride 2"], ["pool", "pooling"], ["none", "none"]], ly.ds, "ds"));
      wrap.appendChild(row);
      return row;
    });
    function render() {
      rows.forEach((r, i) => { r.style.display = i < cfg.depth ? "" : "none"; });
      const c = { ...cfg, layers: cfg.layers.slice(0, cfg.depth) };
      const P = plan(c);
      drawNet(vis, c, P);
      const list = document.getElementById("b-checks");
      if (P.error) {
        list.innerHTML = "";
        ["b-neurons", "b-conns", "b-params"].forEach(id => { document.getElementById(id).textContent = "-"; });
        document.getElementById("b-pass").innerHTML = `<span style="color:var(--bad)">fails</span>`;
        return;
      }
      document.getElementById("b-neurons").textContent = L.fmt(P.neurons);
      document.getElementById("b-conns").textContent = P.conns >= 1e6 ? `${(P.conns / 1e6).toFixed(1)} M` : L.fmt(P.conns);
      document.getElementById("b-params").textContent = L.fmt(P.params);
      document.getElementById("b-pass").innerHTML = P.ok ? `<span style="color:var(--good)">fits</span>` : `<span style="color:var(--bad)">does not fit</span>`;
      list.innerHTML = P.checks.map(x => `<li><span>${x.label}</span><span class="${x.ok ? "under" : "over"}">${L.fmt(x.v)} ${x.ok ? "≤" : ">"} ${L.fmt(x.lim)}</span></li>`).join("");
    }
    L.range("b-size", v => { cfg.size = v; render(); });
    L.seg(document.getElementById("b-T"), v => { cfg.T = +v; render(); });
    L.seg(document.getElementById("b-bits"), v => { cfg.bits = +v; render(); });
    L.seg(document.getElementById("b-depth"), v => { cfg.depth = +v; render(); });
    L.seg(document.getElementById("b-pool"), v => { cfg.pool = v; render(); });
    L.seg(document.getElementById("b-head"), v => { cfg.head = v; render(); });
    L.seg(document.getElementById("b-fc"), v => { cfg.fc = +v; render(); });
    L.seg(document.getElementById("b-signed"), v => { cfg.signed = v === "1"; render(); });
    L.seg(document.getElementById("b-bias"), v => { cfg.bias = v; render(); });
    render();
  }

  /* ---------- mini pictures for the cards ---------- */
  const W = 320, Hm = 128;
  const lab = (s, x, y, t, cls) => T(s, x, y, t, cls || "lt-s");
  function px(s, x, y, n, cell, fn, color) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = fn((j + 0.5) / n, (i + 0.5) / n);
      S("rect", { x: x + j * cell, y: y + i * cell, width: cell - (cell > 3 ? 0.6 : 0), height: cell - (cell > 3 ? 0.6 : 0), fill: v ? (color || C.accent) : "rgba(255,255,255,.04)" }, s);
    }
  }
  const handShape = (u, v) => ((u - 0.35) ** 2 / 0.02 + (v - 0.3) ** 2 / 0.03 < 1) || (Math.abs((u - 0.35) * 0.9 + (v - 0.3) * -0.44 - 0) < 0.06 && v > 0.3 && v < 0.85);
  const V = {
    T(s) {
      const r = L.rng(2);
      [[16, 30], [48, 82]].forEach(([n, y]) => {
        lab(s, 10, y + 4, `T = ${n}`, "lt-m");
        const w = 240 / n;
        for (let i = 0; i < n; i++) S("rect", { x: 70 + i * w, y: y - 12, width: Math.max(1, w - 1), height: 26, fill: `rgba(224,169,79,${0.15 + 0.6 * r()})` }, s);
      });
      lab(s, 70, 122, "same clip: more, thinner slices");
    },
    size(s) {
      px(s, 30, 14, 8, 11, handShape); lab(s, 30, 118, "24 × 24: coarse");
      px(s, 190, 14, 32, 2.75, handShape); lab(s, 190, 118, "88 × 88: 13× the axons");
    },
    bits(s) {
      lab(s, 10, 22, "a pixel that saw 5 events", "lt-m");
      lab(s, 10, 52, "1 bit"); S("line", { x1: 80, y1: 48, x2: 300, y2: 48, stroke: C.accent, "stroke-width": 3 }, s); lab(s, 80, 42, "1: something fired", "lt-a");
      lab(s, 10, 86, "3 bits");
      [[0, 1], [1, 0], [2, 1]].forEach(([k, on]) => { S("line", { x1: 80, y1: 76 + k * 14, x2: 300, y2: 76 + k * 14, stroke: on ? C.accent : C.lineStrong, "stroke-width": on ? 3 : 1.5 }, s); lab(s, 306, 80 + k * 14, `×${2 ** k}`, "lt-ms"); });
      lab(s, 80, 124, "1 + 4 = 5: the count arrives, on 3 axons");
    },
    depth(s) {
      [[2, 18], [4, 74]].forEach(([n, y]) => {
        lab(s, 10, y + 20, `${n} layers`, "lt-m");
        let x = 90;
        for (let i = 0; i <= n; i++) { const sz = 36 - i * 6; S("rect", { x, y: y + 20 - sz / 2, width: sz, height: sz, fill: i ? "rgba(236,231,222,.08)" : "rgba(224,169,79,.12)", stroke: i ? C.lineStrong : C.accent }, s); if (i < n) arrow(s, x + sz + 2, y + 20, x + sz + 16, y + 20); x += sz + 22; }
      });
    },
    channels(s) {
      [[3, 40, "8 channels"], [12, 190, "128 channels"]].forEach(([n, x, t]) => {
        for (let d = n - 1; d >= 0; d--) S("rect", { x: x + d * 4, y: 30 - d * 4 + 20, width: 60, height: 60, fill: d ? "rgba(0,0,0,.6)" : "rgba(236,231,222,.08)", stroke: C.lineStrong, "stroke-width": .8 }, s);
        lab(s, x, 124, t);
      });
    },
    kernel(s) {
      [[3, 30, "3 × 3: fine detail"], [9, 190, "9 × 9: 9× the inputs"]].forEach(([k, x, t]) => {
        const n = 12, c = 7.5;
        px(s, x, 14, n, c, () => 0);
        S("rect", { x: x + (n - k) / 2 * c, y: 14 + (n - k) / 2 * c, width: k * c, height: k * c, fill: "rgba(224,169,79,.25)", stroke: C.accent }, s);
        lab(s, x, 124, t);
      });
    },
    shrink(s) {
      const pairs = [["stride 2", 12, 8, 3], ["pooling", 112, 8, 4], ["none", 212, 8, 6]];
      pairs.forEach(([t, x, a, b]) => {
        const c = 5;
        S("rect", { x, y: 20, width: a * c, height: a * c, fill: "rgba(236,231,222,.08)", stroke: C.lineStrong }, s);
        arrow(s, x + a * c + 3, 40, x + a * c + 15, 40);
        S("rect", { x: x + a * c + 18, y: 40 - b * c / 2, width: b * c, height: b * c, fill: "rgba(224,169,79,.15)", stroke: C.accent }, s);
        lab(s, x, 86, t, "lt-m"); lab(s, x, 102, `8 → ${b}`);
      });
      lab(s, 12, 124, "a 3 × 3 kernel; pooling keeps size, then halves");
    },
    poolType(s) {
      const g = [1, 0, 1, 0], c = 22;
      [[20, "average: 0.5"], [180, "max: 1 (any fired)"]].forEach(([x, t]) => {
        g.forEach((v, i) => S("rect", { x: x + (i % 2) * c, y: 30 + Math.floor(i / 2) * c, width: c - 2, height: c - 2, fill: v ? C.accent : "rgba(255,255,255,.05)" }, s));
        arrow(s, x + 48, 52, x + 68, 52);
        S("rect", { x: x + 72, y: 41, width: c, height: c, fill: x < 100 ? "rgba(224,169,79,.5)" : C.accent }, s);
        lab(s, x, 104, t, "lt-m");
      });
      lab(s, 20, 124, "avg folds into the next layer; max is an OR neuron");
    },
    head(s) {
      for (let d = 2; d >= 0; d--) S("rect", { x: 14 + d * 5, y: 40 - d * 5, width: 40, height: 40, fill: d ? "rgba(0,0,0,.6)" : "rgba(236,231,222,.08)", stroke: C.lineStrong }, s);
      arrow(s, 70, 46, 96, 30); arrow(s, 70, 66, 96, 86);
      S("rect", { x: 100, y: 10, width: 170, height: 10, fill: "rgba(236,231,222,.25)" }, s); lab(s, 100, 36, "flatten: every cell, keeps where");
      S("rect", { x: 100, y: 82, width: 18, height: 10, fill: "rgba(224,169,79,.5)" }, s); lab(s, 124, 92, "GAP: one per channel");
      lab(s, 14, 124, "both feed the 11-class classifier");
    },
    fc(s) {
      [[24, 0], [84, 1]].forEach(([y, hid]) => {
        S("rect", { x: 10, y: y - 6, width: 50, height: 12, fill: "rgba(236,231,222,.2)" }, s);
        let x = 60;
        if (hid) { arrow(s, x + 2, y, x + 40, y); S("rect", { x: x + 44, y: y - 10, width: 12, height: 20, fill: "rgba(224,169,79,.35)", stroke: C.accent }, s); lab(s, x + 34, y + 26, "hidden 128 to 512"); x += 56; }
        arrow(s, x + 2, y, x + 40, y);
        for (let i = 0; i < 5; i++) S("circle", { cx: x + 50, cy: y - 12 + i * 6, r: 2.4, fill: C.s1 }, s);
        lab(s, x + 60, y + 4, "11");
      });
    },
    tau(s) {
      const X = t => 30 + t * 18, Y = v => 108 - v * 90;
      S("line", { x1: X(0), y1: Y(0), x2: X(15), y2: Y(0), stroke: C.line }, s);
      [[2, C.red], [4, C.accent], [8, C.s1], [16, C.green]].forEach(([tau, col], k) => {
        const pts = []; let v = 1;
        for (let t = 0; t <= 15; t++) { pts.push(`${X(t)},${Y(v)}`); v -= v / tau; }
        S("polyline", { points: pts.join(" "), fill: "none", stroke: col, "stroke-width": 1.6 }, s);
        T(s, 310, 22 + k * 14, `τ ${tau}`, "lt-ms end", { fill: col });
      });
      lab(s, 30, 124, "potential after one input, fading step by step");
    },
    tauMode(s) {
      const rows = [["fixed", () => C.s4], ["per layer", (l) => [C.accent, C.rust, C.green][l]], ["per channel", (l, ch) => [C.accent, C.rust, C.green, C.s1][(l + ch) % 4]]];
      rows.forEach(([t, f], r) => {
        lab(s, 10, 26 + r * 36, t, "lt-m");
        for (let l = 0; l < 3; l++) for (let ch = 0; ch < 4; ch++) S("rect", { x: 110 + l * 66 + ch * 13, y: 14 + r * 36, width: 11, height: 18, fill: f(l, ch), "fill-opacity": .7 }, s);
      });
    },
    threshold(s) {
      const X = e => 20 + e * 2.9, Y = v => 110 - v * 80;
      S("line", { x1: X(0), y1: Y(0.6), x2: X(80), y2: Y(0.6), stroke: C.s4, "stroke-dasharray": "5 4" }, s);
      lab(s, X(81), Y(0.6) + 4, "fixed");
      const pts = []; for (let e = 0; e <= 80; e++) pts.push(`${X(e)},${Y(0.6 + 0.25 * (1 - Math.exp(-e / 25)) + 0.02 * Math.sin(e / 3))}`);
      S("polyline", { points: pts.join(" "), fill: "none", stroke: C.accent, "stroke-width": 1.6 }, s);
      lab(s, X(81), Y(0.85) + 4, "learned", "lt-a");
      lab(s, 20, 124, "threshold over 80 epochs of training");
    },
    signed(s) {
      lab(s, 10, 30, "plain", "lt-m"); S("circle", { cx: 100, cy: 26, r: 10, fill: C.panel, stroke: C.lineStrong }, s);
      [130, 160, 205, 250].forEach(x => S("line", { x1: x, y1: 34, x2: x, y2: 14, stroke: C.accent, "stroke-width": 2 }, s));
      lab(s, 10, 88, "signed", "lt-m");
      S("circle", { cx: 92, cy: 72, r: 9, fill: C.panel, stroke: C.accent }, s); T(s, 92, 76, "+", "lt-a mid");
      S("circle", { cx: 92, cy: 100, r: 9, fill: C.panel, stroke: C.rust }, s); T(s, 92, 104, "−", "lt-r mid");
      S("line", { x1: 120, y1: 86, x2: 310, y2: 86, stroke: C.line }, s);
      [[135, 1], [170, -1], [210, 1], [250, -1], [285, -1]].forEach(([x, d]) => S("line", { x1: x, y1: 86, x2: x, y2: 86 - d * 18, stroke: d > 0 ? C.accent : C.rust, "stroke-width": 2 }, s));
      lab(s, 120, 124, "two chip neurons per unit");
    },
    bias(s) {
      lab(s, 10, 22, "in the threshold", "lt-m");
      S("line", { x1: 30, y1: 48, x2: 140, y2: 48, stroke: C.s4, "stroke-dasharray": "4 3" }, s);
      S("line", { x1: 30, y1: 60, x2: 140, y2: 60, stroke: C.accent, "stroke-width": 1.6 }, s);
      lab(s, 30, 80, "θ moves down by the bias: free");
      lab(s, 180, 22, "bias neurons", "lt-m");
      S("circle", { cx: 200, cy: 52, r: 9, fill: C.panel, stroke: C.rust }, s); T(s, 200, 56, "b", "lt-r mid");
      arrow(s, 210, 52, 268, 52, C.rust);
      S("circle", { cx: 280, cy: 52, r: 11, fill: C.panel, stroke: C.lineStrong }, s);
      lab(s, 180, 80, "always on, weight = bias");
      lab(s, 10, 120, "the neuron option costs one extra input per neuron");
    },
    norm(s) {
      const cols = 8, rows = 4, c = 14;
      [[20, "BN: one step", 1], [180, "tdBN: all steps", 0]].forEach(([x, t, one]) => {
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
          const lit = one ? j === 3 : true;
          S("rect", { x: x + j * c, y: 16 + i * c, width: c - 2, height: c - 2, fill: lit ? "rgba(224,169,79,.45)" : "rgba(255,255,255,.05)" }, s);
        }
        lab(s, x, 90, t, "lt-m");
        lab(s, x, 106, "columns: timesteps");
        lab(s, x, 120, "rows: clips in the batch");
      });
    },
    augment(s) {
      const f = handShape;
      [[10, "original", (u, v) => f(u, v)], [90, "shift", (u, v) => f(u - 0.15, v - 0.08)], [170, "cutout", (u, v) => f(u, v) && !(Math.abs(u - 0.35) < 0.12 && Math.abs(v - 0.35) < 0.12)], [250, "drop", (u, v) => f(u, v) && ((Math.floor(u * 40) * 7 + Math.floor(v * 40) * 3) % 5 !== 0)]]
        .forEach(([x, t, fn]) => { px(s, x, 18, 24, 2.6, fn); lab(s, x, 100, t, "lt-m"); });
      lab(s, 10, 122, "applied identically to every frame of a clip");
    },
    convDrop(s) {
      const r = L.rng(4), n = 10, c = 10;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const sp = r() < 0.3, dropped = sp && r() < 0.3;
        S("rect", { x: 40 + j * c, y: 10 + i * c, width: c - 1.5, height: c - 1.5, fill: sp && !dropped ? C.accent : "rgba(255,255,255,.05)" }, s);
        if (dropped) { S("line", { x1: 40 + j * c, y1: 10 + i * c, x2: 40 + j * c + 8, y2: 10 + i * c + 8, stroke: C.red }, s); S("line", { x1: 40 + j * c + 8, y1: 10 + i * c, x2: 40 + j * c, y2: 10 + i * c + 8, stroke: C.red }, s); }
      }
      lab(s, 160, 40, "spikes leaving a layer,", "lt"); lab(s, 160, 56, "some silenced at random", "lt");
      lab(s, 160, 72, "(the same ones for the", "lt"); lab(s, 160, 88, "whole clip)", "lt");
    },
    headDrop(s) {
      const r = L.rng(6);
      for (let i = 0; i < 24; i++) {
        const dropped = r() < 0.3, x = 20 + i * 12;
        S("rect", { x, y: 40, width: 10, height: 26, fill: dropped ? "rgba(255,255,255,.05)" : "rgba(236,231,222,.4)" }, s);
        if (dropped) S("line", { x1: x, y1: 40, x2: x + 10, y2: 66, stroke: C.red }, s);
      }
      lab(s, 20, 30, "inputs to the classifier", "lt-m");
      lab(s, 20, 96, "a different random set is silenced");
      lab(s, 20, 112, "on every training step");
    },
    rate(s) {
      const r = L.rng(8);
      [[14, 0.35, "no penalty"], [70, 0.1, "with a rate penalty"]].forEach(([y, p, t]) => {
        lab(s, 10, y + 4, t, "lt-m");
        for (let row = 0; row < 3; row++) for (let i = 0; i < 40; i++) if (r() < p) S("rect", { x: 10 + i * 7.5, y: y + 10 + row * 10, width: 3, height: 8, fill: C.accent }, s);
      });
    },
    wd(s) {
      const bars = (x, sd, t) => {
        for (let i = -7; i <= 7; i++) { const h = 70 * Math.exp(-0.5 * (i / sd) ** 2); S("rect", { x: x + (i + 7) * 9, y: 96 - h, width: 7, height: h, fill: "rgba(236,231,222,.4)" }, s); }
        lab(s, x, 116, t, "lt-m");
      };
      bars(10, 3.2, "little decay"); bars(170, 1.6, "strong decay");
    },
    ls(s) {
      [[10, false, "plain target"], [170, true, "smoothed target"]].forEach(([x, sm, t]) => {
        for (let i = 0; i < 11; i++) { const v = i === 4 ? (sm ? 0.82 : 1) : (sm ? 0.018 : 0); S("rect", { x: x + i * 12, y: 96 - v * 76, width: 9, height: Math.max(1, v * 76), fill: i === 4 ? C.accent : C.s4 }, s); }
        lab(s, x, 116, t, "lt-m");
      });
    },
    clip(s) {
      S("circle", { cx: 90, cy: 64, r: 40, fill: "none", stroke: C.faint, "stroke-dasharray": "4 4" }, s);
      arrow(s, 90, 64, 210, 24, C.red, 1.6);
      arrow(s, 90, 64, 90 + 40 * Math.cos(-0.32), 64 + 40 * Math.sin(-0.32), C.accent, 2.4);
      lab(s, 140, 14, "an unusually large step", "lt-bad");
      lab(s, 136, 78, "cut to the limit", "lt-a");
      lab(s, 20, 122, "keeps one bad batch from wrecking the weights");
    },
    opt(s) {
      const row = (y, a, b, t) => {
        lab(s, 10, y - 10, t, "lt-m");
        [["gradient", 10], [a, 110], [b, 210]].forEach(([u, x], i) => { S("rect", { x, y, width: 88, height: 24, fill: C.panel, stroke: i === 1 ? C.accent : C.line }, s); T(s, x + 44, y + 16, u, "lt-ms mid"); if (i < 2) arrow(s, x + 90, y + 12, x + 108, y + 12); });
      };
      row(24, "+ decay, scale", "step", "Adam");
      row(84, "scale", "step, − decay", "AdamW");
    },
    lr(s) {
      const X = u => 160 + u * 140, Y = u => 110 - u * u * 80;
      const pts = []; for (let u = -1; u <= 1.001; u += 0.05) pts.push(`${X(u)},${Y(u)}`);
      S("polyline", { points: pts.join(" "), fill: "none", stroke: C.line }, s);
      let u = -0.95; const small = [];
      for (let k = 0; k < 8; k++) { small.push([X(u), Y(u)]); u -= 0.12 * 2 * u; }
      S("polyline", { points: small.map(p => p.join(",")).join(" "), fill: "none", stroke: C.green, "stroke-width": 1.5 }, s);
      u = 0.95; const big = [];
      for (let k = 0; k < 6; k++) { big.push([X(u), Y(u)]); u -= 0.9 * 2 * u; }
      S("polyline", { points: big.map(p => p.join(",")).join(" "), fill: "none", stroke: C.red, "stroke-width": 1.5 }, s);
      lab(s, 10, 30, "small: slow but steady", "lt-g"); lab(s, 10, 50, "large: bounces", "lt-bad");
    },
    sched(s) {
      const X = e => 20 + e * 3.4, Y = v => 108 - v * 84;
      const curves = [["cosine", C.accent, e => 0.5 * (1 + Math.cos(Math.PI * e / 80))], ["one-cycle", C.rust, e => e < 24 ? e / 24 : 0.5 * (1 + Math.cos(Math.PI * (e - 24) / 56))],
        ["step", C.green, e => Math.pow(0.2, Math.floor(e / 27))], ["none", C.s4, () => 0.6]];
      curves.forEach(([t, col, f], k) => {
        const pts = []; for (let e = 0; e <= 80; e++) pts.push(`${X(e)},${Y(f(e))}`);
        S("polyline", { points: pts.join(" "), fill: "none", stroke: col, "stroke-width": 1.6 }, s);
        T(s, 310, 20 + k * 14, t, "lt-ms end", { fill: col });
      });
      lab(s, 20, 124, "learning rate across the epochs");
    },
    warm(s) {
      const X = e => 20 + e * 3.4, Y = v => 108 - v * 84;
      const pts = []; for (let e = 0; e <= 80; e++) { const v = e < 3 ? (e + 1) / 4 : 0.5 * (1 + Math.cos(Math.PI * (e - 3) / 77)); pts.push(`${X(e)},${Y(v)}`); }
      S("polyline", { points: pts.join(" "), fill: "none", stroke: C.accent, "stroke-width": 1.6 }, s);
      S("rect", { x: X(0), y: 20, width: X(3) - X(0), height: 88, fill: "rgba(224,169,79,.1)" }, s);
      lab(s, X(4), 34, "ramp up over the first few epochs", "lt-a");
      lab(s, 20, 124, "then the chosen schedule");
    },
    loss(s) {
      const steps = 10;
      [[20, "rate: count over the clip, one loss"], [80, "TET: a loss at every step, averaged"]].forEach(([y, t], r) => {
        lab(s, 10, y - 4, t, "lt-m");
        for (let i = 0; i < steps; i++) S("rect", { x: 10 + i * 22, y, width: 18, height: 16, fill: "rgba(236,231,222,.12)", stroke: C.line }, s);
        if (r === 0) { arrow(s, 230, y + 8, 260, y + 8); S("circle", { cx: 272, cy: y + 8, r: 8, fill: C.accent }, s); }
        else for (let i = 0; i < steps; i++) S("circle", { cx: 19 + i * 22, cy: y + 26, r: 3.5, fill: C.accent }, s);
      });
    }
  };

  /* ---------- the cards ---------- */
  const CARDS = [
    { g: "input", v: "T", name: "Frames per clip (T)", tried: "16, 24, 32, 48",
      what: "How many slices each recording is cut into, so how many steps the network runs. More frames give finer timing and more chances to accumulate evidence; each one also costs a step of time and work on the chip.",
      eff: { rows: [["16", .517, 15], ["24", .506, 33], ["32", .527, 15], ["48", .565, 33]], verdict: "moderate", note: "Every one of the twelve best networks used 48, the largest value tried." } },
    { g: "input", v: "size", name: "Frame size", tried: "24 to 88, in steps of 8",
      what: "The width and height each frame is shrunk to. Larger frames keep more detail but use more of the chip's 16,000 input axons and make every layer bigger.",
      eff: { rows: [["24", .503, 22], ["32", .518, 14], ["40", .541, 21], ["48", .581, 18], ["56", .546, 8]], verdict: "unclear", note: "A slight rise up to 48; sizes above 56 were tried too rarely to say." } },
    { g: "input", v: "bits", name: "Bits per pixel", tried: "1 to 4", link: "bits.html",
      what: "How many events per pixel the input can tell apart. One bit says only whether anything happened; more bits carry the count, at the cost of one axon per bit.",
      eff: { rows: [["1", .547, 48], ["2", .508, 15], ["3", .477, 13], ["4", .549, 20]], verdict: "unclear", note: "One bit did as well as any; all twelve best networks used it. More bits force a smaller frame." } },
    { g: "shape", v: "depth", name: "Number of layers", tried: "2, 3, 4",
      what: "How many convolution layers the network stacks. Each layer combines the previous layer's patterns into larger ones, and each adds one step of delay on the chip.",
      eff: { rows: [["2", .506, 32], ["3", .549, 33], ["4", .540, 31]], verdict: "moderate", note: "Two layers was too few; three and four were about equal." } },
    { g: "shape", v: "channels", name: "Channels per layer", tried: "8 to 128, chosen separately for each layer",
      what: "How many different patterns each layer looks for, so how many feature maps it makes. Wider layers can represent more, and cost more neurons and connections.",
      eff: { rows: [["22-57 total", .495, 32], ["61-105", .521, 32], ["108-293", .579, 32]], verdict: "moderate", note: "Networks with more channels in total scored higher." } },
    { g: "shape", v: "kernel", name: "Kernel size", tried: "3, 5, 7, 9, chosen separately for each layer",
      what: "How large a patch each neuron reads. A larger kernel sees more context in one step, and multiplies the number of inputs per neuron.",
      eff: { rows: [["3", .529, 87], ["5", .535, 80], ["7", .550, 77], ["9", .522, 43]], verdict: "unclear", note: "Counted per layer. No size stood out." } },
    { g: "shape", v: "shrink", name: "How each layer shrinks the map", tried: "stride 2, pooling, or none, for each layer",
      what: "Stride moves the kernel two cells at a time; pooling runs the convolution normally and then halves the map; none keeps the map almost the same size. Shrinking keeps later layers affordable and widens what each neuron sees.",
      eff: { rows: [["none first", .499, 24], ["pool first", .547, 30], ["stride first", .539, 42]], verdict: "moderate", note: "Shown for the first layer, where it matters most: not shrinking the full-size input first did worst." } },
    { g: "shape", v: "poolType", name: "How pooling runs on the chip", tried: "average or max, when some layer pools",
      what: "The chip has no pooling unit. Average pooling is folded into the next layer's weights, which then reads four times as many inputs. Max pooling of spikes is an OR, done by one extra neuron per 2 × 2 block, which adds a step of delay but keeps the next layer's inputs small.",
      eff: { rows: [["no pooling", .517, 32], ["average", .543, 32], ["max", .534, 32]], verdict: "unclear" } },
    { g: "shape", v: "head", name: "Head: flatten or GAP", tried: "flatten, global average pooling",
      what: "How the last maps become the classifier's input. Flatten keeps every cell, so the classifier knows where each feature appeared. Global average pooling keeps one average per channel: far fewer weights, but no positions.",
      eff: { rows: [["flatten", .595, 48], ["GAP", .468, 48]], verdict: "large", note: "The biggest effect of any setting. Where a gesture happens in the frame carries much of its meaning." } },
    { g: "shape", v: "fc", name: "Hidden layer before the classifier", tried: "none, or one layer of 128, 256 or 512",
      what: "An optional fully connected layer of spiking neurons between the head and the 11 output neurons.",
      eff: { rows: [["none", .526, 50], ["one", .538, 46]], verdict: "unclear" } },
    { g: "neuron", v: "tau", name: "Leak τ", tried: "2, 4, 8, 16 (starting value)",
      what: "How fast the membrane potential fades: each step it loses 1/τ of itself. The chip computes the leak as a bit shift, so τ must be a power of two. A large τ remembers input longer.",
      eff: { rows: [["2", .522, 27], ["4", .540, 29], ["8", .532, 14], ["16", .532, 26]], verdict: "unclear", note: "Where τ starts matters little when it is learned." } },
    { g: "neuron", v: "tauMode", name: "How τ is set", tried: "fixed, learned per layer, learned per channel",
      what: "τ can stay at its starting value, be learned once per layer, or be learned separately for every channel. Learned values are rounded to powers of two, so every option deploys as trained.",
      eff: { rows: [["fixed", .511, 32], ["per layer", .546, 32], ["per channel", .537, 32]], verdict: "unclear", note: "Learning τ looked slightly better than fixing it." } },
    { g: "neuron", v: "threshold", name: "Learned threshold", tried: "off, on",
      what: "Whether each layer's firing threshold is learned during training or kept fixed. Either way it is stored as an integer on the chip.",
      eff: { rows: [["fixed", .553, 52], ["learned", .506, 44]], verdict: "moderate", note: "Learning the threshold did worse." } },
    { g: "neuron", v: "signed", name: "Signed spikes", tried: "off, on", link: "basics.html#signed",
      what: "Whether a unit can also fire a negative spike. Each signed unit is two chip neurons, one for each sign, so it doubles neurons and roughly doubles to quadruples connections, and can push a network past the chip's limits.",
      eff: { rows: [["off", .528, 48], ["on", .535, 48]], verdict: "unclear", note: "No average difference, yet every one of the twelve best networks used signed spikes." } },
    { g: "neuron", v: "bias", name: "Where the bias goes", tried: "in the threshold, or bias neurons",
      what: "The chip has no bias term. The bias can be folded into each channel's threshold, which is free but only equivalent at one membrane value, or delivered by always-on bias neurons, which is exact but costs one input per neuron.",
      eff: { rows: [["threshold", .553, 48], ["bias neurons", .510, 48]], verdict: "moderate", note: "Folding it into the threshold did better." } },
    { g: "norm", v: "norm", name: "BN or tdBN", tried: "batch norm, or tdBN with a scale from 0.5 to 2",
      what: "Batch normalization rescales each channel using statistics from the current batch, one timestep at a time. tdBN uses statistics across all timesteps together and starts its scale relative to the threshold, which suits spiking networks. Both fold into the weights and thresholds at deployment.",
      eff: { rows: [["BN", .511, 51], ["tdBN", .555, 45]], verdict: "moderate", note: "tdBN did better, and within tdBN, a larger starting scale did better." } },
    { g: "reg", v: "augment", name: "Augmentation", tried: "off or on; shift up to 0.2, cutout up to 0.4, event drop up to 0.3", link: "data.html#augment",
      what: "Random shifts, blanked squares and dropped events applied to training clips only, so the network cannot memorize exact clips.",
      eff: { rows: [["off", .563, 48], ["on", .500, 48]], verdict: "moderate", note: "At 80 epochs it hurt, more so at higher strengths: cutout above 0.27 averaged 44%. It makes training harder, so it may need longer runs to pay off." } },
    { g: "reg", v: "convDrop", name: "Dropout between layers", tried: "off, or 5% to 30%",
      what: "Silences a random set of units on the spikes leaving each layer, the same units for a whole clip, so no layer can rely on any single unit.",
      eff: { rows: [["off", .553, 48], ["on", .510, 48]], verdict: "moderate" } },
    { g: "reg", v: "headDrop", name: "Dropout in the head", tried: "10% to 45%",
      what: "Silences a random set of the classifier's inputs on every training step.",
      eff: { rows: [["10-21%", .549, 32], ["21-32%", .535, 32], ["33-45%", .511, 32]], verdict: "unclear", note: "Less looked slightly better." } },
    { g: "reg", v: "rate", name: "Firing-rate penalty", tried: "off, or a weight from 0.0001 to 0.1",
      what: "Adds the average firing rate of the hidden layers to the loss, so the network learns to fire less. Fewer spikes means fewer synaptic operations and less energy on the chip.",
      eff: { rows: [["off", .547, 48], ["on", .516, 48]], verdict: "unclear", note: "A small cost in accuracy for a network that does less work." } },
    { g: "reg", v: "wd", name: "Weight decay", tried: "0.0000001 to 0.01, depending on the optimizer",
      what: "Pulls every weight a little toward zero on each step. Besides fighting overfitting, it keeps folded weights inside the range the chip's 16-bit weights can store.",
      eff: { rows: [["low", .536, 32], ["middle", .550, 32], ["high", .509, 32]], verdict: "unclear" } },
    { g: "reg", v: "ls", name: "Label smoothing", tried: "0 to 0.2",
      what: "Trains toward a target that gives the right class slightly less than 100% and spreads the rest over the others, which discourages overconfident answers.",
      eff: { rows: [["0-0.07", .519, 32], ["0.07-0.13", .552, 32], ["0.14-0.19", .524, 32]], verdict: "unclear" } },
    { g: "reg", v: "clip", name: "Gradient clipping", tried: "off, 1, 5",
      what: "Caps how large a single update can be, so one unusual batch cannot throw the weights far off.",
      eff: { rows: [["off", .533, 32], ["1", .508, 29], ["5", .550, 35]], verdict: "unclear" } },
    { g: "opt", v: "opt", name: "Optimizer", tried: "Adam, AdamW",
      what: "Both adapt each weight's step size to its own gradient history. They differ in weight decay: Adam mixes it into the gradient, AdamW applies it separately, which is usually considered cleaner.",
      eff: { rows: [["Adam", .554, 50], ["AdamW", .508, 46]], verdict: "moderate" } },
    { g: "opt", v: "lr", name: "Learning rate", tried: "0.0005 to 0.005", link: "basics.html#learning",
      what: "The size of each step. The phase on integer weights uses half of it.",
      eff: { rows: [["low", .509, 32], ["middle", .535, 32], ["high", .551, 32]], verdict: "unclear", note: "Within this range, higher looked slightly better." } },
    { g: "opt", v: "sched", name: "Learning-rate schedule", tried: "cosine, one-cycle, step, none",
      what: "How the learning rate changes over the run: cosine glides down, one-cycle rises then falls, step drops it in stages, none keeps it constant.",
      eff: { rows: [["step", .563, 21], ["cosine", .552, 20], ["none", .531, 29], ["one-cycle", .491, 26]], verdict: "moderate", note: "One-cycle did worst." } },
    { g: "opt", v: "warm", name: "Warm-up epochs", tried: "0 to 3, with cosine or step",
      what: "Starts the learning rate low and ramps it up over the first few epochs, so early, poorly scaled gradients do not make large jumps." },
    { g: "opt", v: "loss", name: "Loss", tried: "rate, TET", link: "pipeline.html#step",
      what: "Rate scores the total output spikes over the clip, exactly the chip's answer. TET scores the output at every step and averages, pushing every step to be right on its own.",
      eff: { rows: [["rate", .526, 48], ["TET", .537, 48]], verdict: "unclear" } }
  ];


  /* each card's names in the search records, and the example network's values
     (the best of the balanced block, page C). net: draw where it lives. */
  const NAMES = {
    T: { n: ["T"], ex: "48", net: 1 },
    size: { n: ["resize_to"], ex: "40", net: 1 },
    bits: { n: ["input_bits"], ex: "1", net: 1 },
    depth: { n: ["depth"], ex: "3", net: 1 },
    channels: { n: ["ch_0", "ch_1", "ch_2"], ex: "71, 18, 41", net: 1 },
    kernel: { n: ["k_0", "k_1", "k_2"], ex: "5, 5, 3", net: 1 },
    shrink: { n: ["ds_0", "ds_1", "ds_2"], ex: "stride, none, stride", net: 1 },
    poolType: { n: ["pool_type"], ex: "empty (no layer pools)", net: 1 },
    head: { n: ["final_reduction"], ex: "flatten", net: 1 },
    fc: { n: ["fc_layers", "fc_width_0"], ex: "1, 128", net: 1 },
    tau: { n: ["tau"], ex: "4", net: 1 },
    tauMode: { n: ["tau_mode"], ex: "fixed", net: 1 },
    threshold: { n: ["trainable_threshold"], ex: "False", net: 1 },
    signed: { n: ["signed_spikes"], ex: "True", net: 1 },
    bias: { n: ["bias_mode"], ex: "threshold", net: 1 },
    norm: { n: ["norm", "tdbn_alpha"], ex: "bn (tdbn_alpha only applies to tdBN)", net: 1 },
    augment: { n: ["use_augment", "aug_shift", "aug_cutout", "aug_drop"], ex: "False" },
    convDrop: { n: ["use_conv_dropout", "conv_dropout"], ex: "False", net: 1 },
    headDrop: { n: ["dropout_rate"], ex: "0.153", net: 1 },
    rate: { n: ["use_rate_penalty", "rate_penalty"], ex: "True, 0.0036" },
    wd: { n: ["weight_decay"], ex: "0.00031" },
    ls: { n: ["label_smoothing"], ex: "0.131" },
    clip: { n: ["grad_clip"], ex: "1" },
    opt: { n: ["optimizer"], ex: "adam" },
    lr: { n: ["lr"], ex: "0.00106" },
    sched: { n: ["scheduler"], ex: "none" },
    warm: { n: ["warmup_epochs"], ex: "unused with a constant rate" },
    loss: { n: ["loss"], ex: "tet" }
  };
  function namesHTML(v) {
    const e = NAMES[v];
    if (!e) return "";
    const net = !!e.net;
    const chips = e.n.map(n => net ? `<button class="aka-name" data-hl="${n}" type="button">${n}</button>` : `<code class="aka-name">${n}</code>`).join("");
    return `<aside class="aka"${net ? ` data-viz="net" data-hl="${e.n[0]}"` : ""}><div class="aka-head"><span class="aka-k">In the search records</span>${chips}</div><p class="aka-sub">example network: ${e.ex}</p>${net ? `<div class="aka-viz"></div>` : ""}</aside>`;
  }

  function effectHTML(e) {
    if (!e) return `<div class="sc-effect">Not measured separately.</div>`;
    const lo = 0.40, hi = 0.62, best = Math.max(...e.rows.map(r => r[1]));
    const rows = e.rows.map(([l, m, n]) => `<div class="eb-row${m === best ? " best" : ""}"><span>${l}</span><span class="eb-track"><span class="eb-fill" style="width:${L.clamp((m - lo) / (hi - lo), 0.02, 1) * 100}%"></span></span><span>${(m * 100).toFixed(1)}%</span></div>`).join("");
    return `<div class="sc-effect"><b>Measured</b><span class="verdict-tag ${e.verdict === "large" ? "strong" : e.verdict === "moderate" ? "mild" : "none"}">${e.verdict}</span><div class="effect-bars">${rows}</div>${e.note ? `<div>${e.note}</div>` : ""}</div>`;
  }
  function cards() {
    CARDS.forEach(cd => {
      const grid = document.querySelector(`.set-grid[data-group="${cd.g}"]`);
      if (!grid) return;
      const el = document.createElement("div");
      el.className = "set-card";
      el.innerHTML = `<h4>${cd.name}</h4><div class="sc-range">tried: ${cd.tried}</div>${namesHTML(cd.v)}<div class="sc-vis"></div><p>${cd.what}${cd.link ? ` <a href="${cd.link}">More</a>.` : ""}</p>${effectHTML(cd.eff)}`;
      grid.appendChild(el);
      const s = L.svg(el.querySelector(".sc-vis"), W, Hm, cd.name);
      try { V[cd.v](s); } catch (e) { console.error(cd.v, e); }
      if (window.Aka) window.Aka.init(el);
    });
  }

  function ranked() {
    const box = document.getElementById("ranked-vis");
    if (!box) return;
    const items = CARDS.filter(c => c.eff && ["T", "bits", "depth", "shrink", "poolType", "head", "fc", "tau", "tauMode", "threshold", "signed", "bias", "norm", "augment", "convDrop", "rate", "clip", "opt", "sched", "loss"].includes(c.v))
      .map(c => { const v = c.eff.rows.map(r => r[1]); return { name: c.name, gap: Math.max(...v) - Math.min(...v) }; })
      .sort((a, b) => b.gap - a.gap);
    const rowH = 22, s = L.svg(box, 760, items.length * rowH + 40, "Settings ranked by the gap between their best and worst option");
    const X = g => 270 + g / 0.14 * 460;
    items.forEach((it, i) => {
      const y = 10 + i * rowH;
      T(s, 262, y + 13, it.name, "lt end");
      S("rect", { x: 270, y: y + 3, width: X(it.gap) - 270, height: 13, fill: it.gap >= 0.08 ? C.accent : it.gap >= 0.054 ? "rgba(224,169,79,.55)" : "rgba(167,162,154,.5)" }, s);
      T(s, X(it.gap) + 6, y + 14, `${(it.gap * 100).toFixed(1)} pts`, "lt-ms");
    });
    const yb = 10 + items.length * rowH;
    S("line", { x1: X(0.054), y1: 4, x2: X(0.054), y2: yb, stroke: C.text, "stroke-dasharray": "4 4", "stroke-opacity": .6 }, s);
    T(s, X(0.054) + 4, yb + 18, "luck alone: about 5 points", "lt-s");
  }
})();
