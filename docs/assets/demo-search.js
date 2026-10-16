(function () {
  "use strict";
  const $ = id => document.getElementById(id);

  /* ---------- 6.3 ASHA simulator ---------- */
  (function asha() {
    const cv = $("as-canvas");
    if (!cv) return;
    const NT = 24;
    let seed = 5;

    function rungsFor(grace, rf, epochs) {
      const out = [];
      let r = Math.min(grace, epochs);
      while (r < epochs) { out.push(r); r *= rf; }
      return out;
    }

    function simulate() {
      const grace = +$("as-g").value, rf = +$("as-rf").value, epochs = +$("as-ep").value;
      const rand = SNN.rng(seed);
      // trial curves: saturating exponentials; some "late bloomers" (cosine-ish)
      const trials = [];
      for (let i = 0; i < NT; i++) {
        const cap = 0.45 + rand() * 0.5;             // final accuracy
        const speed = 0.06 + rand() * 0.25;          // convergence rate
        const late = rand() < 0.3;                   // late-blooming schedule
        const ys = [];
        for (let e = 0; e < epochs; e++) {
          let frac = 1 - Math.exp(-speed * (e + 1));
          if (late) frac = Math.pow((e + 1) / epochs, 1.6); // slow start, strong finish
          ys.push(Math.max(0.05, cap * frac + (rand() - 0.5) * 0.04));
        }
        trials.push({ ys, aliveUntil: epochs, cap });
      }
      // apply ASHA
      const rungs = rungsFor(grace, rf, epochs);
      for (const rung of rungs) {
        const arrived = trials.filter(t => t.aliveUntil >= rung);
        if (!arrived.length) continue;
        const scored = arrived.map(t => ({ t, s: t.ys[rung - 1] })).sort((a, b) => b.s - a.s);
        const keep = Math.max(1, Math.floor(scored.length / rf));
        scored.forEach((o, idx) => { if (idx >= keep && o.t.aliveUntil > rung) o.t.aliveUntil = rung; });
      }
      return { trials, rungs, epochs };
    }

    function draw() {
      const { trials, rungs, epochs } = simulate();
      const r = window.Site.setupCanvas(cv, 325);
      const ctx = r.ctx, W = r.w, H = r.h;
      ctx.fillStyle = window.Site.PALETTE.bg; ctx.fillRect(0, 0, W, H);
      const padL = 40, padB = 26, padT = 12;
      const X = e => padL + e / (epochs - 1) * (W - padL - 12);
      const Y = a => padT + (1 - a) * (H - padT - padB);
      // axes
      ctx.strokeStyle = "#2a272455"; ctx.fillStyle = "#a7a29a";
      ctx.font = "12.5px 'JetBrains Mono', monospace";
      [0, 0.25, 0.5, 0.75, 1].forEach(t => {
        ctx.beginPath(); ctx.moveTo(padL, Y(t)); ctx.lineTo(W - 12, Y(t)); ctx.stroke();
        ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(t.toFixed(2), padL - 6, Y(t));
      });
      // rungs
      rungs.forEach(rg => {
        ctx.strokeStyle = "#e0a94f"; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(X(rg - 1), padT); ctx.lineTo(X(rg - 1), H - padB); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#e0a94f"; ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText("rung@" + rg, X(rg - 1), H - padB + 6);
      });
      // curves
      let cost = 0, finished = 0;
      trials.forEach(t => {
        cost += t.aliveUntil;
        const dead = t.aliveUntil < epochs;
        if (!dead) finished++;
        ctx.strokeStyle = dead ? "#33302c55" : (t.cap > 0.85 ? "#9bb17a" : "#ece7de");
        ctx.lineWidth = dead ? 1 : 1.6;
        ctx.beginPath();
        for (let e = 0; e < t.aliveUntil; e++) {
          const x = X(e), y = Y(t.ys[e]);
          e === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        if (dead) {
          ctx.fillStyle = "#cf6b58";
          ctx.beginPath(); ctx.arc(X(t.aliveUntil - 1), Y(t.ys[t.aliveUntil - 1]), 2.6, 0, Math.PI * 2); ctx.fill();
        }
      });
      $("as-rungs").textContent = rungs.length ? rungs.join(", ") : "(none)";
      $("as-fin").textContent = finished + " / " + NT;
      $("as-cost").textContent = SNN.fmt(cost);
      $("as-save").textContent = "vs. " + SNN.fmt(NT * epochs) + " without ASHA (" + Math.round((1 - cost / (NT * epochs)) * 100) + "% saved)";
      $("as-g-out").textContent = $("as-g").value;
      $("as-rf-out").textContent = $("as-rf").value;
      $("as-ep-out").textContent = $("as-ep").value;
    }
    ["as-g", "as-rf", "as-ep"].forEach(id => $(id).addEventListener("input", draw));
    $("as-run").addEventListener("click", () => { seed = Math.floor(Math.random() * 1e9); draw(); });
    window.addEventListener("resize", draw);
    draw();
  })();

  /* ---------- 6.4 define-by-run ---------- */
  (function dbr() {
    if (!$("db-params")) return;
    const state = { space: "per_layer", norm: "bn", opt: "adam", sch: "cosine", aug: "off" };
    function segWire(id, key) {
      document.querySelectorAll("#" + id + " button").forEach(b => b.addEventListener("click", () => {
        document.querySelectorAll("#" + id + " button").forEach(x => x.classList.remove("on"));
        b.classList.add("on");
        state[key] = b.dataset.v;
        update();
      }));
    }
    segWire("db-space", "space"); segWire("db-norm", "norm");
    segWire("db-opt", "opt"); segWire("db-sch", "sch"); segWire("db-aug", "aug");

    function update() {
      const depth = +$("db-d").value, nfc = +$("db-fc").value;
      $("db-d-out").textContent = depth;
      $("db-fc-out").textContent = nfc;
      const params = [["depth", "2–4"]];
      if (state.space === "per_layer") {
        for (let i = 0; i < depth; i++) {
          params.push([`k_${i}`, "3–9 odd"], [`ch_${i}`, "8–128 log"], [`ds_${i}`, "{stride,pool,none}"]);
        }
      } else {
        params.push(["channels", "8–128 log"], ["kernel_size", "3–9 odd"], ["downsample_mode", "{stride,pool}"]);
      }
      params.push(["input_bits", "1–4"], ["resize_to", "from dataset"], ["T", "from config"],
                  ["fc_layers", "0–1"]);
      for (let i = 0; i < nfc; i++) params.push([`fc_width_${i}`, "{128,256,512}"]);
      params.push(["final_reduction", "{flatten,gap}"], ["bias_mode", "{threshold,neuron}"],
                  ["signed_spikes", "{F,T}"], ["pool_type", "{avg,max} if a layer pools"]);
      params.push(["tau", "{2,4,8,16}"], ["tau_mode", "{fixed,layer,channel}"],
                  ["trainable_threshold", "{F,T}"], ["dropout_rate", "0.1–0.45"],
                  ["use_conv_dropout", "{F,T}"], ["norm", "{bn,tdbn}"]);
      if (state.norm === "tdbn") params.push(["tdbn_alpha", "0.5–2.0"]);
      params.push(["label_smoothing", "0–0.2"], ["grad_clip", "{0,1,5}"],
                  ["use_rate_penalty", "{F,T}"], ["use_augment", "{F,T}"]);
      if (state.aug === "on") {
        params.push(["aug_shift", "0–0.2"], ["aug_cutout", "0–0.4"], ["aug_drop", "0–0.3"],
                    ["aug_flip", "{F,T} if allowed"]);
      }
      params.push(["loss", "{rate,tet}"], ["optimizer", "{adam,adamw}"]);
      params.push(state.opt === "adamw" ? ["weight_decay", "1e-5–1e-2 log"] : ["weight_decay", "1e-7–1e-3 log"]);
      params.push(["lr", "5e-4–5e-3 log"], ["scheduler", "{cos,1cyc,step,none}"]);
      if (state.sch === "cosine" || state.sch === "step") params.push(["warmup_epochs", "0–3"]);
      $("db-params").innerHTML = params.map(([n, r]) => `<span class="pill"><b>${n}</b> ${r}</span>`).join("");
      $("db-count").textContent = params.length;
    }
    ["db-d", "db-fc"].forEach(id => $(id).addEventListener("input", update));
    update();
  })();
})();
