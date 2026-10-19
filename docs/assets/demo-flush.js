/* demo-flush.js: how much evidence flush steps save (page A). */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);

  /* ---------- evidence moving through the layers, with and without flush steps (page A) ---------- */
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
})();
