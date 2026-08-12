(function () {
  "use strict";
  const $ = id => document.getElementById(id);

  /* ---------- 5.2 binarize demo ---------- */
  (function binarize() {
    const cv = $("bz-canvas");
    if (!cv) return;
    const SRC = 16, DST = 8;
    let grid = new Array(SRC * SRC).fill(0);
    let mode = "or";
    let geom = null;

    function randomize() {
      const rand = SNN.rng(Math.floor(Math.random() * 1e9));
      grid = grid.map(() => rand() < 0.09 ? 1 : 0);
      draw();
    }
    // bilinear 2x downsample of a grid == average of each 2x2 block for aligned corners=False halving
    function downAvg() {
      const out = new Array(DST * DST).fill(0);
      for (let y = 0; y < DST; y++) for (let x = 0; x < DST; x++) {
        let s = 0;
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++)
          s += grid[(y * 2 + dy) * SRC + (x * 2 + dx)];
        out[y * DST + x] = s / 4;
      }
      return out;
    }

    function draw() {
      const r = window.Site.setupCanvas(cv, 305);
      const ctx = r.ctx, W = r.w, H = r.h;
      ctx.fillStyle = window.Site.PALETTE.bg; ctx.fillRect(0, 0, W, H);
      const cell = Math.min((W / 2 - 70) / SRC, (H - 60) / SRC);
      const dcell = cell * 2;
      const x0 = 24, y0 = 44;
      const x1 = W / 2 + 40;
      ctx.font = "600 13.5px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#cfcbc4"; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText("input events 16×16 (click to paint)", x0, 16);
      ctx.fillText(mode === "or" ? "output 8×8 — bilinear > 0 (the code)" : "output 8×8 — plain average", x1, 16);
      // src grid
      for (let y = 0; y < SRC; y++) for (let x = 0; x < SRC; x++) {
        ctx.fillStyle = grid[y * SRC + x] ? "#e0a94f" : "#111110";
        ctx.fillRect(x0 + x * cell, y0 + y * cell, cell - 1, cell - 1);
      }
      // dst grid
      const avg = downAvg();
      let active = 0;
      for (let y = 0; y < DST; y++) for (let x = 0; x < DST; x++) {
        const a = avg[y * DST + x];
        if (mode === "or") {
          const v = a > 0 ? 1 : 0;
          if (v) active++;
          ctx.fillStyle = v ? "#e0a94f" : "#111110";
        } else {
          if (a > 0) active++;
          const g = Math.round(a * 255);
          ctx.fillStyle = a > 0 ? `rgba(224,169,79,${0.15 + a * 0.85})` : "#111110";
        }
        ctx.fillRect(x1 + x * dcell, y0 + y * dcell, dcell - 2, dcell - 2);
        if (mode === "avg" && a > 0 && a < 1) {
          ctx.fillStyle = "#050505"; ctx.font = "11.5px 'JetBrains Mono', monospace";
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(a.toFixed(2), x1 + x * dcell + dcell / 2, y0 + y * dcell + dcell / 2);
          ctx.font = "600 13.5px 'JetBrains Mono', monospace"; ctx.textAlign = "left"; ctx.textBaseline = "top";
        }
      }
      geom = { x0, y0, cell };
      const nIn = grid.reduce((a, b) => a + b, 0);
      const nOut = mode === "or" ? avg.filter(a => a > 0).length : avg.filter(a => a > 0).length;
      $("bz-in").textContent = nIn;
      $("bz-ind").textContent = Math.round(nIn / (SRC * SRC) * 100) + "% density";
      $("bz-out").textContent = mode === "or" ? avg.filter(a => a > 0).length : "—";
      $("bz-outd").textContent = mode === "or"
        ? Math.round(avg.filter(a => a > 0).length / (DST * DST) * 100) + "% density (rises!)"
        : "fractional grays — not spikes";
      $("bz-lost").textContent = mode === "or" ? 0 : "n/a";
    }

    cv.addEventListener("click", e => {
      if (!geom) return;
      const rect = cv.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const gx = Math.floor((mx - geom.x0) / geom.cell), gy = Math.floor((my - geom.y0) / geom.cell);
      if (gx >= 0 && gx < SRC && gy >= 0 && gy < SRC) {
        grid[gy * SRC + gx] ^= 1;
        draw();
      }
    });
    $("bz-rand").addEventListener("click", randomize);
    $("bz-clear").addEventListener("click", () => { grid.fill(0); draw(); });
    document.querySelectorAll("#bz-mode button").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll("#bz-mode button").forEach(x => x.classList.remove("on"));
      b.classList.add("on"); mode = b.dataset.m; draw();
    }));
    window.addEventListener("resize", draw);
    randomize();
  })();

  /* ---------- 5.4 flush pipeline ---------- */
  (function flush() {
    const cv = $("fl-canvas");
    if (!cv) return;
    const T = 8;
    let playT = -1, timer = null;
    const COLORS = ["#cf6b58", "#c98a4e", "#e0a94f", "#a3a878", "#9bb17a", "#7f9c8b", "#ece7de", "#cf8763"];

    function draw() {
      const L = +$("fl-l").value;
      const doFlush = $("fl-flush").checked;
      const totalSteps = T + (doFlush ? L : 0);
      const rows = L + 2; // input, L layers, output accumulator
      const r = window.Site.setupCanvas(cv, 295);
      const ctx = r.ctx, W = r.w, H = r.h;
      ctx.fillStyle = window.Site.PALETTE.bg; ctx.fillRect(0, 0, W, H);
      const padL = 84, padT = 26;
      const cw = Math.min((W - padL - 14) / (T + 6), 40);
      const rh = (H - padT - 14) / rows;
      ctx.font = "12.5px 'JetBrains Mono', monospace";

      // row labels
      const labels = ["input frame"].concat(Array.from({ length: L }, (_, i) => "layer " + i)).concat(["output count"]);
      ctx.fillStyle = "#a7a29a"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      labels.forEach((lb, i) => ctx.fillText(lb, padL - 8, padT + i * rh + rh / 2));

      let reached = 0;
      for (let t = 0; t < totalSteps; t++) {
        const x = padL + t * cw;
        const inFlush = t >= T;
        // column header
        ctx.fillStyle = inFlush ? "#e0a94f" : "#7d7871";
        ctx.textAlign = "center"; ctx.textBaseline = "bottom";
        ctx.fillText(inFlush ? "∅" : "t" + t, x + cw / 2, padT - 4);
        const visible = playT < 0 || t <= playT;
        for (let rI = 0; rI < rows; rI++) {
          const y = padT + rI * rh;
          // stage rI at time t carries frame (t - rI) if 0 <= t-rI < T
          let frame = t - rI;
          let fill = "#111110";
          if (rI === 0) { frame = t; fill = (t < T) ? COLORS[t % COLORS.length] : "#111110"; }
          else if (rI <= L) { fill = (frame >= 0 && frame < T) ? COLORS[frame % COLORS.length] : "#111110"; }
          else { // output row: frame arriving = t - L
            const fa = t - L;
            fill = (fa >= 0 && fa < T) ? COLORS[fa % COLORS.length] : "#111110";
            if (fa >= 0 && fa < T && visible) reached = Math.max(reached, fa + 1);
          }
          ctx.globalAlpha = visible ? 1 : 0.13;
          ctx.fillStyle = fill;
          ctx.fillRect(x + 1, y + 2, cw - 3, rh - 5);
          ctx.globalAlpha = 1;
        }
        if (inFlush) {
          ctx.strokeStyle = "#e0a94f40";
          ctx.strokeRect(x, padT, cw, rows * rh);
        }
      }
      // flush divider
      ctx.strokeStyle = "#e0a94f"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(padL + T * cw, padT - 14); ctx.lineTo(padL + T * cw, padT + rows * rh); ctx.stroke();
      ctx.setLineDash([]);
      if (doFlush) {
        ctx.fillStyle = "#e0a94f"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
        ctx.fillText("← input ends · flush →", padL + T * cw + 4, padT - 14);
      }

      const total = playT < 0 ? totalSteps : Math.min(playT + 1, totalSteps);
      const fullReach = doFlush ? T : Math.max(0, T - L);
      const shown = playT < 0 ? fullReach : Math.min(reached, fullReach);
      $("fl-fs").textContent = doFlush ? L : 0;
      $("fl-l-out").textContent = L;
      $("fl-reach").textContent = `${shown} / ${T}`;
      const lost = Math.round((T - fullReach) / T * 100);
      $("fl-lost").textContent = lost + "%";
      $("fl-lost").className = "val " + (lost > 0 ? "bad" : "good");
    }

    $("fl-play").addEventListener("click", () => {
      if (timer) clearInterval(timer);
      playT = -1;
      const L = +$("fl-l").value;
      const totalSteps = T + ($("fl-flush").checked ? L : 0);
      timer = setInterval(() => {
        playT++;
        if (playT >= totalSteps) { clearInterval(timer); timer = null; playT = -1; }
        draw();
      }, 330);
    });
    ["fl-l"].forEach(id => $(id).addEventListener("input", () => { playT = -1; draw(); }));
    $("fl-flush").addEventListener("change", () => { playT = -1; draw(); });
    window.addEventListener("resize", draw);
    draw();
  })();

  /* ---------- 5.5 LR schedules ---------- */
  (function lrviz() {
    const cv = $("lr-canvas");
    if (!cv) return;
    function schedule(kind, epochs, warmup, lr0) {
      const ys = [];
      const main = epochs - (kind === "onecycle" ? 0 : warmup);
      for (let e = 0; e < epochs; e++) {
        let lr;
        if (kind === "onecycle") {
          const pct = e / Math.max(1, epochs - 1);
          if (pct < 0.3) lr = lr0 / 25 + (lr0 - lr0 / 25) * (pct / 0.3);
          else lr = lr0 * (0.5 * (1 + Math.cos(Math.PI * (pct - 0.3) / 0.7)));
        } else if (e < warmup && kind !== "none") {
          lr = lr0 * (0.1 + 0.9 * (e / Math.max(1, warmup)));
        } else {
          const t = e - warmup;
          if (kind === "cosine") lr = lr0 * 0.5 * (1 + Math.cos(Math.PI * t / Math.max(1, main)));
          else if (kind === "step") lr = lr0 * Math.pow(0.2, Math.floor(e / Math.max(1, Math.floor(epochs / 3))));
          else lr = lr0;
        }
        ys.push(lr);
      }
      return ys;
    }
    function draw() {
      const kind = $("lr-kind").value;
      const epochs = +$("lr-ep").value;
      const warmup = +$("lr-wu").value;
      $("lr-ep-out").textContent = epochs;
      $("lr-wu-out").textContent = warmup;
      const lr0 = 0.003;
      const ys = schedule(kind, epochs, warmup, lr0);
      const xs = ys.map((_, i) => i);
      window.Site.linePlot(cv, [
        { xs, ys, color: "#ece7de", width: 2.2, label: kind + (warmup && kind !== "onecycle" && kind !== "none" ? ` + ${warmup} warmup` : ""), dots: epochs <= 40, dotR: 2 }
      ], { height: 275, xlabel: "epoch", ylabel: "learning rate", ymin: 0, ymax: lr0 * 1.12 });
    }
    ["lr-kind"].forEach(id => $(id).addEventListener("change", draw));
    ["lr-ep", "lr-wu"].forEach(id => $(id).addEventListener("input", draw));
    window.addEventListener("resize", draw);
    draw();
  })();
})();
