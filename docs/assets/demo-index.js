(function () {
  "use strict";
  const P = () => window.Site.PALETTE;

  /* ---------------- hero: ambient spike raster ---------------- */
  (function heroRaster() {
    const cv = document.getElementById("hero-raster");
    if (!cv) return;
    let ctx, W, H;
    const ROWS = 26;
    let spikes = []; // {row, x}
    const rand = SNN.rng(7);
    function resize() {
      const r = window.Site.setupCanvas(cv, cv.parentElement.clientHeight);
      ctx = r.ctx; W = r.w; H = r.h;
    }
    resize();
    window.addEventListener("resize", resize);
    function tick() {
      ctx.clearRect(0, 0, W, H);
      const rowH = H / ROWS;
      // spawn
      if (rand() < 0.9) spikes.push({ row: Math.floor(rand() * ROWS), x: W + 4 });
      spikes.forEach(s => s.x -= 1.15);
      spikes = spikes.filter(s => s.x > -6);
      for (const s of spikes) {
        const y = s.row * rowH + rowH / 2;
        const g = ctx.createLinearGradient(s.x - 26, 0, s.x, 0);
        g.addColorStop(0, "rgba(236,231,222,0)");
        g.addColorStop(1, "rgba(236,231,222,.35)");
        ctx.fillStyle = g;
        ctx.fillRect(s.x - 26, y - 0.6, 26, 1.2);
        ctx.fillStyle = "rgba(224,169,79,.8)";
        ctx.fillRect(s.x, y - 2.6, 1.8, 5.2);
      }
      requestAnimationFrame(tick);
    }
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) tick();
  })();

  /* ---------------- demo: ANN vs SNN ops ---------------- */
  (function annVsSnn() {
    const cv = document.getElementById("avs-canvas");
    if (!cv) return;
    const N_IN = 6, T = 16;
    const slider = document.getElementById("avs-sparsity");
    const out = document.getElementById("avs-sparsity-out");
    const runBtn = document.getElementById("avs-run");
    const annOpsEl = document.getElementById("avs-ann-ops");
    const snnOpsEl = document.getElementById("avs-snn-ops");
    const savedEl = document.getElementById("avs-saved");
    let anim = null;

    function makeInput(p) {
      const rand = SNN.rng(Math.floor(Math.random() * 1e9));
      const grid = [];
      for (let i = 0; i < N_IN; i++) {
        grid.push(Array.from({ length: T }, () => rand() < p ? 1 : 0));
      }
      return grid;
    }

    function draw(grid, step, annOps, snnOps, spikesOut) {
      const { ctx, w, h } = window.Site.setupCanvas(cv, 285);
      ctx.fillStyle = P().bg; ctx.fillRect(0, 0, w, h);
      const half = w / 2;
      const cell = Math.min((half - 90) / T, 16);
      const rowH = 20;
      const topY = 46;
      ctx.font = "13.5px 'JetBrains Mono', monospace";

      function panel(x0, title, isSNN) {
        ctx.fillStyle = "#cfcbc4"; ctx.textAlign = "left"; ctx.textBaseline = "top";
        ctx.font = "600 14px 'JetBrains Mono', monospace";
        ctx.fillText(title, x0 + 12, 12);
        ctx.font = "12.5px 'JetBrains Mono', monospace";
        for (let i = 0; i < N_IN; i++) {
          const y = topY + i * rowH;
          ctx.fillStyle = "#7d7871";
          ctx.fillText("in" + i, x0 + 12, y + 3);
          for (let t = 0; t < T; t++) {
            const x = x0 + 44 + t * cell;
            const active = grid[i][t] === 1;
            const done = t < step;
            if (isSNN) {
              ctx.fillStyle = active ? (done ? "#e0a94f" : "#3a3025") : (done ? "#161514" : "#0a0a0a");
            } else {
              // ANN: every cell is "a number", always processed
              ctx.fillStyle = done ? "rgba(236,231,222,.75)" : "#1e1c1a";
            }
            ctx.fillRect(x, y, cell - 2.5, 12);
          }
        }
        // step cursor
        if (step > 0 && step <= T) {
          const x = x0 + 44 + (step - 1) * cell;
          ctx.strokeStyle = "#ffffff2e";
          ctx.strokeRect(x - 1, topY - 3, cell - 0.5, N_IN * rowH + 2);
        }
      }
      panel(0, "ANN — dense numbers, all processed", false);
      panel(half, "SNN — spikes, silence is skipped", true);
      // divider
      ctx.strokeStyle = "#2a2724"; ctx.beginPath(); ctx.moveTo(half, 8); ctx.lineTo(half, h - 8); ctx.stroke();
      // output spikes for SNN
      ctx.fillStyle = "#a7a29a"; ctx.font = "12.5px 'JetBrains Mono', monospace";
      ctx.fillText("neuron output:", half + 12, topY + N_IN * rowH + 14);
      for (let t = 0; t < Math.min(step, T); t++) {
        if (spikesOut[t]) {
          ctx.fillStyle = "#9bb17a";
          ctx.fillRect(half + 118 + t * cell, topY + N_IN * rowH + 10, cell - 2.5, 12);
        } else {
          ctx.fillStyle = "#161514";
          ctx.fillRect(half + 118 + t * cell, topY + N_IN * rowH + 10, cell - 2.5, 12);
        }
      }
    }

    function run() {
      if (anim) cancelAnimationFrame(anim);
      const p = parseInt(slider.value, 10) / 100;
      const grid = makeInput(p);
      // simulate the SNN neuron (hardware order) with weight .45 per input
      const state = { v: 0 };
      const spikesOut = [];
      let step = 0, annOps = 0, snnOps = 0, last = 0;
      function frame(ts) {
        if (ts - last > 240) {
          last = ts;
          if (step < T) {
            let x = 0, events = 0;
            for (let i = 0; i < N_IN; i++) { if (grid[i][step]) { events++; x += 0.45; } }
            annOps += N_IN;          // dense: one MAC per input, every step
            snnOps += events;        // event-driven: one ADD per spike
            spikesOut.push(SNN.hardwareLIFStep(state, x, { tau: 4, v_threshold: 1.0 }));
            step++;
            annOpsEl.textContent = annOps;
            snnOpsEl.textContent = snnOps;
            savedEl.textContent = annOps ? Math.round((1 - snnOps / annOps) * 100) + "%" : "—";
          }
          draw(grid, step, annOps, snnOps, spikesOut);
          if (step >= T) return;
        }
        anim = requestAnimationFrame(frame);
      }
      anim = requestAnimationFrame(frame);
    }
    slider.addEventListener("input", () => { out.textContent = slider.value + "%"; });
    runBtn.addEventListener("click", run);
    draw(makeInput(0.2), 0, 0, 0, []);
  })();

  /* ---------------- demo: poke the neuron ---------------- */
  (function poke() {
    const cv = document.getElementById("poke-canvas");
    if (!cv) return;
    const TAUS = SNN.HW_TAU_CHOICES;
    const tauSlider = document.getElementById("poke-tau");
    const tauOut = document.getElementById("poke-tau-out");
    const autoBox = document.getElementById("poke-auto");
    const btn = document.getElementById("poke-btn");
    const W_IN = 0.45, THETA = 1.0, HIST = 160;
    const state = { v: 0 };
    const vHist = new Array(HIST).fill(0);
    const inHist = new Array(HIST).fill(0);
    const spkHist = new Array(HIST).fill(0);
    let pending = 0;
    const rand = SNN.rng(42);

    function stepSim() {
      let x = pending * W_IN;
      pending = 0;
      if (autoBox.checked && rand() < 0.28) x += W_IN;
      const tau = TAUS[parseInt(tauSlider.value, 10)];
      const s = SNN.hardwareLIFStep(state, x, { tau, v_threshold: THETA });
      vHist.push(state.v); vHist.shift();
      inHist.push(x); inHist.shift();
      spkHist.push(s); spkHist.shift();
    }

    function draw() {
      const { ctx, w, h } = window.Site.setupCanvas(cv, 275);
      ctx.fillStyle = P().bg; ctx.fillRect(0, 0, w, h);
      const padL = 8, plotTop = 34, plotBot = h - 26;
      const dx = (w - padL - 8) / HIST;
      const vmax = Math.max(1.6, ...vHist) * 1.05;
      const Y = v => plotBot - (v / vmax) * (plotBot - plotTop);
      // threshold line
      ctx.strokeStyle = "#e0a94f"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(padL, Y(THETA)); ctx.lineTo(w - 8, Y(THETA)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#e0a94f"; ctx.font = "12.5px 'JetBrains Mono', monospace";
      ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText("threshold θ = 1.0", padL + 4, Y(THETA) - 3);
      // input bars
      for (let i = 0; i < HIST; i++) {
        if (inHist[i] > 0) {
          ctx.fillStyle = "rgba(224,169,79,.5)";
          const bh = (inHist[i] / vmax) * (plotBot - plotTop);
          ctx.fillRect(padL + i * dx, plotBot - bh, Math.max(1.5, dx * 0.6), bh);
        }
      }
      // membrane trace
      ctx.strokeStyle = "#ece7de"; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < HIST; i++) {
        const x = padL + i * dx, y = Y(vHist[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      // output spikes
      for (let i = 0; i < HIST; i++) {
        if (spkHist[i]) {
          ctx.fillStyle = "#9bb17a";
          ctx.fillRect(padL + i * dx, 12, Math.max(2, dx * 0.6), 14);
        }
      }
      ctx.fillStyle = "#9bb17a"; ctx.textBaseline = "top";
      ctx.fillText("output spikes", padL, 0);
      ctx.fillStyle = "#a7a29a"; ctx.textAlign = "right";
      ctx.fillText("v = " + state.v.toFixed(3), w - 10, 0);
    }

    btn.addEventListener("click", () => { pending += 1; });
    tauSlider.addEventListener("input", () => { tauOut.textContent = TAUS[parseInt(tauSlider.value, 10)]; });
    tauOut.textContent = TAUS[parseInt(tauSlider.value, 10)];
    setInterval(stepSim, 90);
    (function loop() { draw(); requestAnimationFrame(loop); })();
  })();
})();
