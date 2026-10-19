/* demo-downsample.js: why a shrunk event frame uses OR instead of an average (page B). */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);

  /* ---------- shrinking a sparse frame: average against OR (page B) ---------- */
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
      ctx.fillText(mode === "or" ? "output 8×8: bilinear interpolation, then > 0" : "output 8×8: average pooling", x1, 16);
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
      $("bz-out").textContent = mode === "or" ? avg.filter(a => a > 0).length : "-";
      $("bz-outd").textContent = mode === "or"
        ? Math.round(avg.filter(a => a > 0).length / (DST * DST) * 100) + "% density (rises!)"
        : "fractional values require binarization";
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
})();
