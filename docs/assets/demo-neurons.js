(function () {
  "use strict";
  const TAUS = SNN.HW_TAU_CHOICES;
  const P = () => window.Site.PALETTE;

  /* ============ 2.2 full simulator ============ */
  (function fullSim() {
    const cv = document.getElementById("sim-canvas");
    if (!cv) return;
    const HIST = 160;
    const els = {
      pattern: document.getElementById("sim-pattern"),
      rate: document.getElementById("sim-rate"), rateOut: document.getElementById("sim-rate-out"),
      w: document.getElementById("sim-w"), wOut: document.getElementById("sim-w-out"),
      tau: document.getElementById("sim-tau"), tauOut: document.getElementById("sim-tau-out"),
      theta: document.getElementById("sim-theta"), thetaOut: document.getElementById("sim-theta-out"),
      pause: document.getElementById("sim-pause"),
      v: document.getElementById("sim-v"), fr: document.getElementById("sim-fr"),
      thint: document.getElementById("sim-thint")
    };
    const state = { v: 0 };
    const vH = new Array(HIST).fill(0), inH = new Array(HIST).fill(0), spH = new Array(HIST).fill(0);
    let t = 0, paused = false;
    const rand = SNN.rng(1234);

    function inputAt(t, p) {
      const kind = els.pattern.value;
      if (kind === "poisson") return rand() < p ? 1 : 0;
      if (kind === "regular") { const period = Math.max(1, Math.round(1 / Math.max(p, 0.02))); return t % period === 0 ? 1 : 0; }
      if (kind === "burst") { const phase = t % 40; return phase < 10 ? (rand() < Math.min(1, p * 2.6) ? 1 : 0) : 0; }
      if (kind === "step") return 1; // constant drive scaled by w & p
      return 0;
    }

    function step() {
      if (paused) return;
      const p = parseInt(els.rate.value, 10) / 100;
      const w = parseInt(els.w.value, 10) / 100;
      const tau = TAUS[parseInt(els.tau.value, 10)];
      const theta = parseInt(els.theta.value, 10) / 100;
      const ev = inputAt(t, p);
      const x = els.pattern.value === "step" ? w * p * 2 : ev * w;
      const s = SNN.hardwareLIFStep(state, x, { tau, v_threshold: theta });
      vH.push(state.v); vH.shift();
      inH.push(x); inH.shift();
      spH.push(s); spH.shift();
      t++;
      els.v.textContent = state.v.toFixed(3);
      els.fr.textContent = (spH.reduce((a, b) => a + b, 0) / HIST).toFixed(2);
      els.thint.textContent = SNN.quantizedThresholdInt(theta);
    }

    function draw() {
      const { ctx, w, h } = window.Site.setupCanvas(cv, 345);
      ctx.fillStyle = P().bg; ctx.fillRect(0, 0, w, h);
      const theta = parseInt(els.theta.value, 10) / 100;
      const padL = 8, spikeBand = 30, inBand = 26;
      const plotTop = spikeBand + 12, plotBot = h - inBand - 12;
      const dx = (w - padL - 8) / HIST;
      const vmax = Math.max(theta * 1.5, 1.4, ...vH) * 1.04;
      const Y = v => plotBot - (Math.max(v, 0) / vmax) * (plotBot - plotTop);
      ctx.font = "12.5px 'JetBrains Mono', monospace";
      // output band
      ctx.fillStyle = "#9bb17a"; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText("output spikes", padL, 2);
      for (let i = 0; i < HIST; i++) if (spH[i]) {
        ctx.fillRect(padL + i * dx, 14, Math.max(2, dx * 0.55), 13);
      }
      // input band
      ctx.fillStyle = "#e0a94f";
      ctx.fillText("input events", padL, h - inBand + 12);
      for (let i = 0; i < HIST; i++) if (inH[i] > 0) {
        ctx.globalAlpha = 0.85;
        ctx.fillRect(padL + i * dx, h - inBand, Math.max(2, dx * 0.55), 12);
        ctx.globalAlpha = 1;
      }
      // threshold
      ctx.strokeStyle = "#e0a94f"; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(padL, Y(theta)); ctx.lineTo(w - 8, Y(theta)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#e0a94f"; ctx.textBaseline = "bottom";
      ctx.fillText("θ = " + theta.toFixed(2), padL + 4, Y(theta) - 2);
      // membrane
      ctx.strokeStyle = "#ece7de"; ctx.lineWidth = 2; ctx.beginPath();
      for (let i = 0; i < HIST; i++) {
        const x = padL + i * dx, y = Y(vH[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    els.rate.addEventListener("input", () => els.rateOut.textContent = els.rate.value + "%");
    els.w.addEventListener("input", () => els.wOut.textContent = (els.w.value / 100).toFixed(2));
    els.tau.addEventListener("input", () => els.tauOut.textContent = TAUS[parseInt(els.tau.value, 10)]);
    els.theta.addEventListener("input", () => els.thetaOut.textContent = (els.theta.value / 100).toFixed(2));
    els.pause.addEventListener("click", () => {
      paused = !paused;
      els.pause.textContent = paused ? "Resume" : "Pause";
    });
    setInterval(step, 70);
    (function loop() { draw(); requestAnimationFrame(loop); })();
  })();

  /* ============ 2.3 order comparison ============ */
  (function orderCmp() {
    const cv = document.getElementById("ord-canvas");
    if (!cv) return;
    const T = 44;
    const els = {
      tau: document.getElementById("ord-tau"), tauOut: document.getElementById("ord-tau-out"),
      w: document.getElementById("ord-w"), wOut: document.getElementById("ord-w-out"),
      btn: document.getElementById("ord-new"),
      sjN: document.getElementById("ord-sj-n"), hwN: document.getElementById("ord-hw-n"),
      agree: document.getElementById("ord-agree")
    };
    let seed = 99;

    function simulate() {
      const tau = TAUS[parseInt(els.tau.value, 10)];
      const w = parseInt(els.w.value, 10) / 100;
      const rand = SNN.rng(seed);
      const input = Array.from({ length: T }, () => rand() < 0.30 ? 1 : 0);
      const sj = { v: 0 }, hw = { v: 0 };
      const sjV = [], hwV = [], sjS = [], hwS = [];
      for (let t = 0; t < T; t++) {
        const x = input[t] * w;
        sjS.push(SNN.spikingjellyLIFStep(sj, x, { tau, v_threshold: 1 }));
        sjV.push(sj.v);
        hwS.push(SNN.hardwareLIFStep(hw, x, { tau, v_threshold: 1 }));
        hwV.push(hw.v);
      }
      return { input, sjV, hwV, sjS, hwS };
    }

    function draw() {
      const { input, sjV, hwV, sjS, hwS } = simulate();
      const { ctx, w, h } = window.Site.setupCanvas(cv, 375);
      ctx.fillStyle = P().bg; ctx.fillRect(0, 0, w, h);
      const padL = 30, dx = (w - padL - 12) / T;
      ctx.font = "12.5px 'JetBrains Mono', monospace";

      // shared input row
      ctx.fillStyle = "#e0a94f"; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText("shared input", padL, 2);
      for (let t = 0; t < T; t++) if (input[t]) ctx.fillRect(padL + t * dx, 14, Math.max(2, dx * 0.5), 11);

      function tracePanel(y0, y1, vs, ss, color, label) {
        const vmax = Math.max(1.35, ...vs) * 1.05;
        const Y = v => y1 - (Math.max(v, 0) / vmax) * (y1 - y0);
        ctx.strokeStyle = "#e0a94f55"; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(padL, Y(1)); ctx.lineTo(w - 12, Y(1)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath();
        for (let t = 0; t < T; t++) {
          const x = padL + t * dx + dx / 2, y = Y(vs[t]);
          t === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        for (let t = 0; t < T; t++) if (ss[t]) {
          ctx.fillStyle = color;
          ctx.fillRect(padL + t * dx, y0 - 16, Math.max(2, dx * 0.5), 11);
        }
        ctx.fillStyle = color; ctx.textBaseline = "bottom";
        ctx.fillText(label, padL, y0 - 18);
      }
      tracePanel(70, 175, sjV, sjS, "#ece7de", "textbook (charge→fire) — v and spikes");
      tracePanel(225, 322, hwV, hwS, "#e0a94f", "chip (fire→reset→leak→integrate) — v and spikes");

      let agree = 0;
      for (let t = 0; t < T; t++) if (sjS[t] === hwS[t]) agree++;
      els.sjN.textContent = sjS.reduce((a, b) => a + b, 0);
      els.hwN.textContent = hwS.reduce((a, b) => a + b, 0);
      els.agree.textContent = Math.round(agree / T * 100) + "%";
    }
    els.tau.addEventListener("input", () => { els.tauOut.textContent = TAUS[parseInt(els.tau.value, 10)]; draw(); });
    els.w.addEventListener("input", () => { els.wOut.textContent = (els.w.value / 100).toFixed(2); draw(); });
    els.btn.addEventListener("click", () => { seed = Math.floor(Math.random() * 1e9); draw(); });
    window.addEventListener("resize", draw);
    draw();
  })();

  /* ============ 2.4 integer leak ============ */
  (function intLeak() {
    const cv = document.getElementById("il-canvas");
    if (!cv) return;
    const els = {
      tau: document.getElementById("il-tau"), out: document.getElementById("il-tau-out"),
      f: document.getElementById("il-float"), i: document.getElementById("il-int"),
      keep: document.getElementById("il-keep")
    };
    function draw() {
      const tauF = parseInt(els.tau.value, 10) / 100;
      const leak = Math.round(tauF);
      const T = 24;
      const dec = tau => { const ys = [1]; for (let t = 1; t < T; t++) ys.push(ys[t - 1] * (1 - 1 / tau)); return ys; };
      const xs = Array.from({ length: T }, (_, i) => i);
      window.Site.linePlot(cv, [
        { xs, ys: dec(tauF), color: "#cf8763", width: 2, label: `float τ=${tauF.toFixed(2)} (training illusion)`, dots: true, dotR: 2 },
        { xs, ys: dec(leak), color: "#e0a94f", width: 2, dash: [6, 4], label: `chip leak=${leak} (reality)`, dots: true, dotR: 2 }
      ], { height: 285, xlabel: "timestep", ylabel: "membrane (no input)", ymin: 0, ymax: 1 });
      els.out.textContent = tauF.toFixed(2);
      els.f.textContent = tauF.toFixed(2);
      els.i.textContent = leak;
      els.keep.textContent = Math.round((1 - 1 / leak) * 100) + "%";
    }
    els.tau.addEventListener("input", draw);
    window.addEventListener("resize", draw);
    draw();
  })();

  /* ============ 2.5 threshold sigmoid + quantize ============ */
  (function thres() {
    const cv = document.getElementById("th-canvas");
    if (!cv) return;
    const els = {
      raw: document.getElementById("th-raw"), rawOut: document.getElementById("th-raw-out"),
      theta: document.getElementById("th-theta"), int: document.getElementById("th-int"),
      legal: document.getElementById("th-legal")
    };
    function draw() {
      const raw = parseInt(els.raw.value, 10) / 10;
      const theta = SNN.sigmoid(raw);
      const xs = [], ys = [];
      for (let r = -6; r <= 6.001; r += 0.1) { xs.push(r); ys.push(SNN.sigmoid(r)); }
      const plt = window.Site.linePlot(cv, [
        { xs, ys, color: "#ece7de", width: 2, label: "θ = sigmoid(raw_threshold)" }
      ], {
        height: 305, xlabel: "raw_threshold (unconstrained parameter)", ylabel: "θ",
        ymin: 0, ymax: 1.08,
        hlines: [{ y: 1.0, color: "#cf6b58", label: "θ=1.0 → int 32767 (ceiling)" }]
      });
      // marker
      plt.ctx.fillStyle = "#e0a94f";
      plt.ctx.beginPath(); plt.ctx.arc(plt.X(raw), plt.Y(theta), 5, 0, Math.PI * 2); plt.ctx.fill();
      els.rawOut.textContent = raw.toFixed(1);
      els.theta.textContent = theta.toFixed(5);
      els.int.textContent = SNN.quantizedThresholdInt(theta);
      els.legal.textContent = "YES";
    }
    els.raw.addEventListener("input", draw);
    window.addEventListener("resize", draw);
    draw();
  })();

  /* ============ 2.6 surrogate gradient ============ */
  (function surr() {
    const cv = document.getElementById("sg-canvas");
    if (!cv) return;
    const els = {
      x: document.getElementById("sg-x"), xOut: document.getElementById("sg-x-out"),
      fwd: document.getElementById("sg-fwd"), grad: document.getElementById("sg-grad")
    };
    function draw() {
      const xv = parseInt(els.x.value, 10) / 100;
      const xs = [], step = [], grad = [];
      for (let x = -2; x <= 2.001; x += 0.02) {
        xs.push(x);
        step.push(SNN.heaviside(x));
        grad.push(SNN.atanSurrogateGrad(x));
      }
      const plt = window.Site.linePlot(cv, [
        { xs, ys: step, color: "#e0a94f", width: 2, label: "forward: real spike (Heaviside)", step: true },
        { xs, ys: grad, color: "#cf8763", width: 2, label: "backward: ATan surrogate gradient" }
      ], { height: 295, xlabel: "v − θ (distance from threshold)", ylabel: "", ymin: -0.05, ymax: 1.12 });
      plt.ctx.fillStyle = "#ece7de";
      plt.ctx.beginPath(); plt.ctx.arc(plt.X(xv), plt.Y(SNN.atanSurrogateGrad(xv)), 5, 0, Math.PI * 2); plt.ctx.fill();
      els.xOut.textContent = xv.toFixed(2);
      els.fwd.textContent = SNN.heaviside(xv);
      els.grad.textContent = SNN.atanSurrogateGrad(xv).toFixed(3);
    }
    els.x.addEventListener("input", draw);
    window.addEventListener("resize", draw);
    draw();
  })();
})();
