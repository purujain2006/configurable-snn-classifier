/* demo-lr.js: the learning-rate schedules a run can use (page D). */
(function () {
  "use strict";
  const C = window.Site.C;
  const $ = id => document.getElementById(id);

  /* ---------- learning-rate schedules (page D) ---------- */
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
        { xs, ys, color: C.s1, width: 2.2, label: kind + (warmup && kind !== "onecycle" && kind !== "none" ? ` + ${warmup} warmup` : ""), dots: epochs <= 40, dotR: 2 }
      ], { height: 275, xlabel: "epoch", ylabel: "learning rate", ymin: 0, ymax: lr0 * 1.12 });
    }
    ["lr-kind"].forEach(id => $(id).addEventListener("change", draw));
    ["lr-ep", "lr-wu"].forEach(id => $(id).addEventListener("input", draw));
    window.addEventListener("resize", draw);
    draw();
  })();
})();
