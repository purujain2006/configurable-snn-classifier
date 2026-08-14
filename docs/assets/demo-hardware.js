(function () {
  "use strict";
  const $ = id => document.getElementById(id);

  /* ---------- 4.1 fan budget calculator ---------- */
  (function fanCalc() {
    if (!$("fb-bars")) return;
    function update() {
      const k = +$("fb-k").value, cin = +$("fb-cin").value, cout = +$("fb-cout").value,
            s = +$("fb-s").value, inHW = +$("fb-hw").value;

      // the head's fan-in is not a free choice: it is this layer's output volume
      const oh = SNN.convOutSize(inHW, k, s, 0, 1);
      const ow = oh;
      const collapsed = oh < 1 || ow < 1;
      const fcIn = collapsed ? 0 : cout * oh * ow;   // flatten head
      const gapIn = cout;                            // GAP head

      const fanIn = k * k * cin;
      const fanOut = Math.pow(Math.ceil(k / s), 2) * cout;
      const L = SNN.NEURON_LIMITS;

      $("fb-derived").innerHTML = collapsed
        ? `<span style="color:var(--bad);">the ${inHW}×${inHW} map is smaller than a ${k}×${k} kernel, so this layer cannot be built</span>`
        : `this layer: <b>${cin}×${inHW}×${inHW}</b> &nbsp;&#8594;&nbsp; <b>${cout}×${oh}×${ow}</b>
           &nbsp;·&nbsp; flatten head reads <b>${cout}×${oh}×${ow} = ${SNN.fmt(fcIn)}</b> features
           &nbsp;·&nbsp; GAP head would read <b>${SNN.fmt(gapIn)}</b>`;

      const items = [
        { label: `conv fan-in  k²·Cin = ${SNN.fmt(fanIn)}`, value: fanIn, max: L.fan_in * 1.6,
          display: `limit ${SNN.fmt(L.fan_in)}`, color: fanIn > L.fan_in ? "var(--bad)" : "var(--good)" },
        { label: `conv fan-out  ⌈k/s⌉²·Cout = ${SNN.fmt(fanOut)}`, value: fanOut, max: L.fan_in * 1.6,
          display: `limit ${SNN.fmt(L.fan_out)}`, color: fanOut > L.fan_out ? "var(--bad)" : "var(--good)" },
        { label: `head fan-in  Cout·H·W = ${SNN.fmt(fcIn)}`, value: fcIn, max: L.fan_in * 1.6,
          display: `limit ${SNN.fmt(L.fan_in)}`, color: fcIn > L.fan_in ? "var(--bad)" : "var(--good)" }
      ];
      window.Site.barChart($("fb-bars"), items, { max: L.fan_in * 1.6, labelW: 250, valueW: 95 });

      const bust = [];
      if (fanIn > L.fan_in) bust.push(`conv fan-in is ${SNN.fmt(fanIn)} against a limit of ${SNN.fmt(L.fan_in)}`);
      if (fanOut > L.fan_out) bust.push(`conv fan-out is ${SNN.fmt(fanOut)} against a limit of ${SNN.fmt(L.fan_out)}`);
      if (fcIn > L.fan_in) bust.push(`the flatten produces ${SNN.fmt(fcIn)} features, so every neuron in the first head layer would need that many inputs against a limit of ${SNN.fmt(L.fan_in)}. A GAP head reads ${SNN.fmt(gapIn)} instead, at the cost described in section 3.5`);
      $("fb-note").innerHTML = bust.length
        ? `OVER BUDGET &nbsp;·&nbsp; ${bust.join(". ")}.`
        : `WITHIN BUDGET &nbsp;·&nbsp; Every quantity fits. Raising output channels multiplies the head fan-in directly, since it is Cout·H·W, which is how most wide configurations failed.`;
    }
    ["fb-k", "fb-cin", "fb-cout", "fb-s", "fb-hw"].forEach(id => $(id).addEventListener("change", update));
    update();
  })();

  /* ---------- 4.2 INT16 playground ---------- */
  (function int16() {
    const cv = $("q-canvas");
    if (!cv) return;
    function draw() {
      const w = (+$("q-w").value) / 100;
      const q = SNN.fakeQuantizeWeight(w);
      const code = SNN.weightIntCode(w);
      const { ctx, w: W, h } = window.Site.setupCanvas(cv, 190);
      ctx.fillStyle = window.Site.PALETTE.bg; ctx.fillRect(0, 0, W, h);
      const padL = 30, padR = 30, mid = h / 2 + 8;
      const X = v => padL + (v + 1.7) / 3.4 * (W - padL - padR);
      // representable band
      ctx.fillStyle = "rgba(155,177,122,.08)";
      ctx.fillRect(X(-1), mid - 34, X(1) - X(-1), 68);
      // axis
      ctx.strokeStyle = "#3a3733"; ctx.beginPath(); ctx.moveTo(X(-1.7), mid); ctx.lineTo(X(1.7), mid); ctx.stroke();
      ctx.font = "12.5px 'JetBrains Mono', monospace"; ctx.fillStyle = "#a7a29a";
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      [-1.5, -1, -0.5, 0, 0.5, 1, 1.5].forEach(t => {
        ctx.fillStyle = Math.abs(t) === 1 ? "#9bb17a" : "#a7a29a";
        ctx.beginPath(); ctx.moveTo(X(t), mid - 5); ctx.lineTo(X(t), mid + 5);
        ctx.strokeStyle = Math.abs(t) === 1 ? "#9bb17a" : "#3a3733"; ctx.stroke();
        ctx.fillText(t.toFixed(1), X(t), mid + 9);
      });
      ctx.fillStyle = "#9bb17a"; ctx.textBaseline = "bottom";
      ctx.fillText("representable [−1, +1]", X(0), mid - 38);
      // float marker
      ctx.fillStyle = "#cf8763";
      ctx.beginPath(); ctx.arc(X(w), mid - 20, 6, 0, Math.PI * 2); ctx.fill();
      ctx.textBaseline = "bottom"; ctx.fillText("float " + w.toFixed(2), X(w), mid - 30);
      // arrow to quantized
      ctx.strokeStyle = "#e0a94f"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(X(w), mid - 14); ctx.lineTo(X(q), mid + 18); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#e0a94f";
      ctx.beginPath(); ctx.arc(X(q), mid + 22, 6, 0, Math.PI * 2); ctx.fill();
      ctx.textBaseline = "top"; ctx.fillText("chip " + q.toFixed(3), X(q), mid + 32);

      $("q-w-out").textContent = w.toFixed(3);
      $("q-float").textContent = w.toFixed(3);
      $("q-quant").textContent = q.toFixed(5);
      $("q-int").textContent = code;
      const loss = Math.abs(w - q);
      $("q-loss").textContent = loss.toFixed(5);
      $("q-loss").className = "val " + (loss > 0.01 ? "bad" : "good");
    }
    $("q-w").addEventListener("input", draw);
    window.addEventListener("resize", draw);
    draw();
  })();

  /* ---------- 4.3 fold one channel ---------- */
  (function foldOne() {
    if (!$("f-w")) return;
    function update() {
      const w = (+$("f-w").value) / 100, g = (+$("f-g").value) / 100, v = (+$("f-v").value) / 100;
      const m = (+$("f-m").value) / 100, b = (+$("f-b").value) / 100;
      const { wPrime, bPrime, scale } = SNN.foldConvBn(w, 0, g, b, m, v);
      $("f-w-out").textContent = w.toFixed(2); $("f-g-out").textContent = g.toFixed(2);
      $("f-v-out").textContent = v.toFixed(2); $("f-m-out").textContent = m.toFixed(2);
      $("f-b-out").textContent = b.toFixed(2);
      $("f-scale").textContent = scale.toFixed(3);
      $("f-wp").textContent = wPrime.toFixed(3);
      const clipped = Math.abs(wPrime) > 1;
      $("f-wp").className = "val " + (clipped ? "bad" : "good");
      $("f-clip").textContent = clipped ? "CLIPPED" : "fits";
      $("f-bp").textContent = bPrime.toFixed(3);
      const thp = 1.0 - bPrime;
      $("f-thp").textContent = thp.toFixed(3);
      let warn = "";
      if (thp <= 0) {
        $("f-thp").className = "val bad";
        warn = `<div class="verdict fail">BLOCKING &nbsp;·&nbsp; folded threshold θ′ = ${thp.toFixed(3)} ≤ 0 — this neuron would fire unconditionally on chip. deployment_report scores this config 0.</div>`;
      } else if (thp > 1) {
        $("f-thp").className = "val warn";
        warn = `<div class="verdict pass" style="border-color:rgba(224,169,79,.4);color:var(--warn);">NOTE &nbsp;·&nbsp; θ′ > 1 — legal to compute with, but note it exceeds the (0,1] grid the base θ obeys; the fold’s legality check watches the unclamped value.</div>`;
      } else {
        $("f-thp").className = "val acc";
      }
      if (clipped) warn += `<div class="verdict fail" style="margin-top:8px;">CLIPPED &nbsp;·&nbsp; |W′| > 1 — the INT16 grid will silently truncate this weight. In a real layer this is counted by weight_clip_fraction.</div>`;
      $("f-warn").innerHTML = warn;
    }
    ["f-w", "f-g", "f-v", "f-m", "f-b"].forEach(id => $(id).addEventListener("input", update));
    update();
  })();

  /* ---------- 4.3b clip histogram ---------- */
  (function clipHist() {
    const cv = $("h-canvas");
    if (!cv) return;
    const N = 4000, CH = 16;
    function update() {
      const std = (+$("h-std").value) / 100;
      const scSpread = (+$("h-sc").value) / 100;
      const doFold = $("h-fold").checked;
      $("h-std-out").textContent = std.toFixed(2);
      $("h-sc-out").textContent = scSpread.toFixed(1) + "×";
      const rand = SNN.rng(77);
      // per-channel scales: log-uniform in [1/scSpread, scSpread]
      const scales = Array.from({ length: CH }, () => Math.exp((rand() * 2 - 1) * Math.log(scSpread)));
      const ws = [];
      for (let i = 0; i < N; i++) {
        let w = SNN.gaussian(rand) * std;
        if (doFold) w *= scales[i % CH];
        ws.push(w);
      }
      // histogram
      const BINS = 80, lo = -2.2, hi = 2.2;
      const bins = new Array(BINS).fill(0);
      let clipped = 0, maxAbs = 0;
      for (const w of ws) {
        maxAbs = Math.max(maxAbs, Math.abs(w));
        if (Math.abs(w) > 1) clipped++;
        const bi = Math.floor((SNN.clamp(w, lo, hi - 1e-9) - lo) / (hi - lo) * BINS);
        bins[SNN.clamp(bi, 0, BINS - 1)]++;
      }
      const { ctx, w: W, h } = window.Site.setupCanvas(cv, 275);
      ctx.fillStyle = window.Site.PALETTE.bg; ctx.fillRect(0, 0, W, h);
      const padL = 12, padB = 24, bw = (W - padL * 2) / BINS;
      const peak = Math.max(...bins, 1);
      const X = v => padL + (v - lo) / (hi - lo) * (W - padL * 2);
      for (let i = 0; i < BINS; i++) {
        const x0 = lo + (i + 0.5) / BINS * (hi - lo);
        const bh = bins[i] / peak * (h - padB - 26);
        ctx.fillStyle = Math.abs(x0) > 1 ? "#cf6b58" : "#ece7de";
        ctx.fillRect(padL + i * bw, h - padB - bh, Math.max(1, bw - 1), bh);
      }
      // ±1 markers
      ctx.strokeStyle = "#9bb17a"; ctx.setLineDash([5, 4]);
      [-1, 1].forEach(t => { ctx.beginPath(); ctx.moveTo(X(t), 8); ctx.lineTo(X(t), h - padB); ctx.stroke(); });
      ctx.setLineDash([]);
      ctx.font = "12.5px 'JetBrains Mono', monospace"; ctx.fillStyle = "#9bb17a";
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText("−1", X(-1), h - padB + 6); ctx.fillText("+1", X(1), h - padB + 6);
      ctx.fillStyle = "#a7a29a";
      [-2, 0, 2].forEach(t => ctx.fillText(String(t), X(t), h - padB + 6));

      const frac = clipped / N;
      $("h-clip").textContent = (frac * 100).toFixed(1) + "%";
      $("h-clip").className = "val " + (frac > 0.02 ? "bad" : "good");
      $("h-budget").textContent = frac > 0.02 ? "EXCEEDED" : "OK";
      $("h-budget").className = "val " + (frac > 0.02 ? "warn" : "good");
      $("h-max").textContent = maxAbs.toFixed(2);
    }
    ["h-std", "h-sc"].forEach(id => $(id).addEventListener("input", update));
    $("h-fold").addEventListener("change", update);
    window.addEventListener("resize", update);
    update();
  })();

  /* ---------- 4.5 QAT epoch split ---------- */
  (function qatSplit() {
    if (!$("qt-ep")) return;
    function update() {
      const epochs = +$("qt-ep").value;
      const frac = (+$("qt-fr").value) / 100;
      // exact formula from run_training (inline mode)
      let warmup = Math.max(1, Math.round(epochs * frac));
      warmup = epochs > 1 ? Math.min(warmup, epochs - 1) : 1;
      const grid = epochs - warmup;
      $("qt-ep-out").textContent = epochs;
      $("qt-fr-out").textContent = frac.toFixed(2);
      const wp = warmup / epochs * 100, gp = grid / epochs * 100;
      $("qt-bar").innerHTML =
        `<div style="width:${wp}%;background:#ece7de;display:grid;place-items:center;color:#050505;font-weight:700;min-width:52px;">float ×${warmup}</div>` +
        `<div style="width:${gp}%;background:#cf8763;display:grid;place-items:center;color:#050505;font-weight:700;min-width:52px;">on-grid ×${grid}</div>`;
    }
    ["qt-ep", "qt-fr"].forEach(id => $(id).addEventListener("input", update));
    update();
  })();

  /* ---------- 4.3 bias into input vs theta - b' ---------- */
  (function foldBiasForm() {
    if (!$("fx-canvas")) return;
    const TAUS = SNN.HW_TAU_CHOICES;      // [2,3,4,6,8,16,32,63]
    const STEPS = 19;                     // T=16 plus 3 flush steps, as deployed
    const THETA = 1.0, W = 0.30;
    let train = [];

    function newTrain() {
      const r = +$("fx-rate").value / 100;
      train = Array.from({ length: STEPS }, () => (Math.random() < r ? 1 : 0));
    }

    // Runs the verified chip-order neuron. `bias` adds to the input each step;
    // `thShift` lowers the threshold instead. Same intent, different operation.
    function run(tau, bias, thShift) {
      const st = { v: 0 }, p = { tau, v_threshold: THETA - thShift, v_reset: 0 };
      const spikes = [], vs = [];
      for (let t = 0; t < STEPS; t++) {
        spikes.push(SNN.hardwareLIFStep(st, train[t] * W + bias, p));
        vs.push(st.v);
      }
      return { spikes, vs };
    }

    function draw() {
      const tau = TAUS[+$("fx-tau").value], b = +$("fx-b").value / 100;
      $("fx-tau-out").textContent = tau;
      $("fx-b-out").textContent = b.toFixed(2);
      $("fx-rate-out").textContent = $("fx-rate").value + "%";

      const A = run(tau, b, 0);        // bias into the input
      const B = run(tau, 0, b);        // theta - b'
      const nA = A.spikes.reduce((s, x) => s + x, 0);
      const nB = B.spikes.reduce((s, x) => s + x, 0);
      const agree = A.spikes.filter((s, i) => s === B.spikes[i]).length;

      $("fx-na").textContent = nA;
      $("fx-nb").textContent = nB;
      const ag = $("fx-agree");
      ag.textContent = `${agree}/${STEPS}`;
      ag.className = "val " + (agree === STEPS ? "good" : agree < STEPS - 2 ? "bad" : "warn");
      $("fx-gap").textContent = (b * tau).toFixed(2);

      const P = window.Site.PALETTE;
      const { ctx, w, h } = window.Site.setupCanvas($("fx-canvas"), 340);
      ctx.fillStyle = P.bg; ctx.fillRect(0, 0, w, h);

      const pad = 12, half = (w - pad * 3) / 2;
      const vmax = Math.max(THETA * 1.6, ...A.vs, ...B.vs) * 1.05;

      [[A, 0, "bias added to input", "what QAT trained", THETA],
       [B, 1, "threshold lowered to θ − b′", "what the chip runs", THETA - b]
      ].forEach(([R, side, title, sub, thLine]) => {
        const x0 = pad + side * (half + pad), top = 46, ph = 150;
        ctx.fillStyle = P.text; ctx.font = "600 14px Inter, sans-serif";
        ctx.fillText(title, x0, 20);
        ctx.fillStyle = P.axis; ctx.font = "12px Inter, sans-serif";
        ctx.fillText(sub, x0, 36);

        const cell = half / STEPS;
        const Y = v => top + ph - Math.max(0, Math.min(1, v / vmax)) * ph;

        // threshold line
        ctx.strokeStyle = P.spike; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x0, Y(thLine)); ctx.lineTo(x0 + half, Y(thLine)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = P.spike; ctx.font = "11px JetBrains Mono, monospace";
        ctx.fillText("θ = " + thLine.toFixed(2), x0 + half - 62, Y(thLine) - 5);

        // input ticks
        for (let t = 0; t < STEPS; t++) {
          if (!train[t]) continue;
          ctx.fillStyle = "rgba(224,169,79,.30)";
          ctx.fillRect(x0 + t * cell + cell * 0.3, top + ph + 6, cell * 0.4, 10);
        }
        ctx.fillStyle = P.axis; ctx.font = "11px Inter, sans-serif";
        ctx.fillText("input", x0, top + ph + 30);

        // membrane trace
        ctx.strokeStyle = P.accent; ctx.lineWidth = 1.8; ctx.beginPath();
        R.vs.forEach((v, t) => {
          const px = x0 + t * cell + cell / 2, py = Y(v);
          t ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        });
        ctx.stroke();

        // spike band
        for (let t = 0; t < STEPS; t++) {
          const differ = A.spikes[t] !== B.spikes[t];
          if (!R.spikes[t]) continue;
          ctx.fillStyle = differ ? P.bad : P.good;
          ctx.fillRect(x0 + t * cell + cell * 0.2, top + ph + 44, cell * 0.6, 14);
        }
        ctx.fillStyle = P.axis; ctx.font = "11px Inter, sans-serif";
        ctx.fillText("output spikes", x0, top + ph + 74);
      });

      // accumulation curve d_t = b*tau*(1-(1-1/tau)^t)
      const cy = 268, ch = 54, cw = w - pad * 2;
      ctx.fillStyle = P.text; ctx.font = "600 13px Inter, sans-serif";
      ctx.fillText("membrane offset the bias creates, dₜ = b′·τ·(1 − (1 − 1/τ)ᵗ)", pad, cy - 10);
      const dmax = Math.max(b * tau, b * 1.2, 0.01);
      ctx.strokeStyle = P.accent2; ctx.lineWidth = 2; ctx.beginPath();
      for (let t = 0; t < STEPS; t++) {
        const d = b * tau * (1 - Math.pow(1 - 1 / tau, t + 1));
        const px = pad + (t / (STEPS - 1)) * cw, py = cy + ch - (d / dmax) * ch;
        t ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
      // what the threshold shift is worth, for comparison
      const by = cy + ch - (b / dmax) * ch;
      ctx.strokeStyle = P.spike; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(pad, by); ctx.lineTo(pad + cw, by); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = P.spike; ctx.font = "11px JetBrains Mono, monospace";
      ctx.fillText("threshold shift is worth b′ = " + b.toFixed(2) + ", flat", pad + 4, by - 5);
      ctx.fillStyle = P.accent2;
      ctx.fillText("reaches " + (b * tau).toFixed(2), pad + cw - 96, cy + 12);
    }

    ["fx-tau", "fx-b"].forEach(id => $(id).addEventListener("input", draw));
    $("fx-rate").addEventListener("input", () => { newTrain(); draw(); });
    $("fx-new").addEventListener("click", () => { newTrain(); draw(); });
    window.addEventListener("resize", draw);
    newTrain(); draw();
  })();
})();
