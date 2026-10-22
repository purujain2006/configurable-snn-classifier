(function () {
  "use strict";
  const C = window.Site.C;
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
        g.addColorStop(0, Site.a(C.s1, 0));
        g.addColorStop(1, Site.a(C.s1, .35));
        ctx.fillStyle = g;
        ctx.fillRect(s.x - 26, y - 0.6, 26, 1.2);
        ctx.fillStyle = Site.a(C.accent, .8);
        ctx.fillRect(s.x, y - 2.6, 1.8, 5.2);
      }
      requestAnimationFrame(tick);
    }
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) tick();
  })();
})();
