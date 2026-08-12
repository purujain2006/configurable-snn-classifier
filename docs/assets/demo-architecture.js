(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const fmt = SNN.fmt;

  /* ---------------- network builder ---------------- */
  const PRESETS = {
    seed1: { resize: 64, T: 16, ds: "stride", norm: "bn", red: "flatten", fc: "",
             layers: [{ ch: 32, k: 7, s: 2, pool: false }, { ch: 32, k: 7, s: 2, pool: false }] },
    seed3: { resize: 64, T: 16, ds: "stride", norm: "bn", red: "flatten", fc: "",
             layers: [{ ch: 32, k: 7, s: 2, pool: false }, { ch: 32, k: 7, s: 2, pool: false }, { ch: 32, k: 7, s: 2, pool: false }] },
    default: { resize: 0, T: 16, ds: "stride", norm: "bn", red: "flatten", fc: "512",
             layers: [{ ch: 128, k: 3, s: 2, pool: false }, { ch: 128, k: 3, s: 2, pool: false }, { ch: 128, k: 3, s: 2, pool: false }] },
    perlayer: { resize: 64, T: 16, ds: "stride", norm: "bn", red: "flatten", fc: "256",
             layers: [{ ch: 32, k: 7, s: 2, pool: false }, { ch: 64, k: 5, s: 1, pool: true }, { ch: 32, k: 5, s: 2, pool: false }] },
    gap: { resize: 64, T: 16, ds: "stride", norm: "bn", red: "gap", fc: "512",
             layers: [{ ch: 64, k: 5, s: 2, pool: false }, { ch: 64, k: 5, s: 2, pool: false }, { ch: 64, k: 5, s: 2, pool: false }] },
    collapse: { resize: 32, T: 8, ds: "stride", norm: "bn", red: "flatten", fc: "",
             layers: [{ ch: 32, k: 7, s: 2, pool: false }, { ch: 32, k: 7, s: 2, pool: false }, { ch: 32, k: 7, s: 2, pool: false }, { ch: 32, k: 7, s: 2, pool: false }, { ch: 32, k: 7, s: 2, pool: false }] }
  };
  let layers = JSON.parse(JSON.stringify(PRESETS.seed1.layers));

  const CH_OPTS = [16, 32, 64, 128, 256];
  const K_OPTS = [3, 5, 7];
  const S_OPTS = [1, 2];

  function renderLayerTable() {
    const tb = $("nb-layers").querySelector("tbody");
    tb.innerHTML = layers.map((L, i) => `
      <tr>
        <td class="layer-name">conv${i}</td>
        <td><select data-i="${i}" data-f="ch">${CH_OPTS.map(c => `<option ${c === L.ch ? "selected" : ""}>${c}</option>`).join("")}</select></td>
        <td><select data-i="${i}" data-f="k">${K_OPTS.map(c => `<option ${c === L.k ? "selected" : ""}>${c}</option>`).join("")}</select></td>
        <td><select data-i="${i}" data-f="s">${S_OPTS.map(c => `<option ${c === L.s ? "selected" : ""}>${c}</option>`).join("")}</select></td>
        <td><input type="checkbox" data-i="${i}" data-f="pool" ${L.pool ? "checked" : ""} style="accent-color:var(--accent);width:15px;height:15px;"></td>
        <td>${layers.length > 1 ? `<button class="btn small" data-del="${i}">×</button>` : ""}</td>
      </tr>`).join("");
    tb.querySelectorAll("select,input").forEach(el => {
      el.addEventListener("change", () => {
        const i = +el.dataset.i, f = el.dataset.f;
        if (f === "pool") layers[i].pool = el.checked;
        else layers[i][f] = parseInt(el.value, 10);
        update();
      });
      // style the inline selects
      if (el.tagName === "SELECT") {
        el.style.cssText = "background:var(--bg-3);color:var(--text);border:1px solid var(--border-2);border-radius:6px;padding:4px 7px;font-family:var(--mono);font-size:12px;";
      }
    });
    tb.querySelectorAll("[data-del]").forEach(b => b.addEventListener("click", () => {
      layers.splice(+b.dataset.del, 1); renderLayerTable(); update();
    }));
  }

  function currentCfg() {
    const resize = parseInt($("nb-resize").value, 10);
    return {
      input: SNN.inputSpec({ resize_to: resize, T: parseInt($("nb-T").value, 10) }),
      encoder: SNN.encoderSpec({
        layers: layers.map(L => ({ out_channels: L.ch, kernel_size: L.k, stride: L.s, pool: L.pool })),
        norm: $("nb-norm").value, bias: $("nb-norm").value === "none"
      }),
      output: SNN.outputSpec(),
      downsample: SNN.downsampleSpec({ mode: $("nb-ds").value }),
      head: SNN.headSpec({ final_reduction: $("nb-red").value, fc_widths: $("nb-fc").value }),
      neuron: SNN.neuronSpec()
    };
  }

  function update() {
    const cfg = currentCfg();
    const [feasible, violations] = SNN.checkFeasibility(cfg.input, cfg.encoder, cfg.downsample, cfg.head, cfg.output);

    let counts = null, planErr = null;
    try { counts = SNN.countNeuronsAndSynapses(cfg); }
    catch (e) { planErr = e.message; }

    // ---- flow diagram ----
    const flow = $("nb-flow");
    const [ih, iw] = SNN.effectiveHW(cfg.input);
    let fhtml = `<div class="flow-box input"><div class="fb-kind">input</div><div class="fb-shape">${cfg.input.C}×${ih}×${iw}</div><div class="fb-meta">T=${cfg.input.T} · ${fmt(ih * iw * cfg.input.C)} axons</div></div>`;
    if (counts) {
      for (const b of counts.plan.blocks) {
        fhtml += `<div class="flow-arrow">→</div>
          <div class="flow-box"><div class="fb-kind">conv${b.index}${b.pool ? "+pool" : ""}</div>
          <div class="fb-shape">${b.out_channels}×${b.out_hw[0]}×${b.out_hw[1]}</div>
          <div class="fb-meta">k=${b.kernel_size} s=${b.stride}${b.pool ? " → pool/2" : ""}</div></div>`;
      }
      fhtml += `<div class="flow-arrow">→</div>
        <div class="flow-box head"><div class="fb-kind">${counts.plan.reduction}</div>
        <div class="fb-shape">${fmt(counts.plan.fc_in_features)}</div><div class="fb-meta">features to head</div></div>`;
      for (const l of counts.plan.linears) {
        fhtml += `<div class="flow-arrow">→</div>
          <div class="flow-box head"><div class="fb-kind">${l.is_classifier ? "classifier" : "fc" + l.index}</div>
          <div class="fb-shape">${fmt(l.out_features)}</div><div class="fb-meta">${l.is_classifier ? "gesture scores" : "hidden"}</div></div>`;
      }
    } else {
      fhtml += `<div class="flow-arrow">→</div><div class="flow-box" style="border-color:var(--bad);"><div class="fb-kind" style="color:var(--bad);">plan failed</div><div class="fb-meta">${planErr}</div></div>`;
    }
    flow.innerHTML = fhtml;

    // ---- cost table ----
    const tb = $("nb-cost");
    if (counts) {
      tb.innerHTML = counts.rows.map(r =>
        `<tr><td class="layer-name">${r.layer}</td><td style="font-family:var(--mono);font-size:12px;">${r.detail}</td>
         <td class="num">${fmt(r.neurons)}</td><td class="num">${fmt(r.connections)}</td><td class="num">${fmt(r.params)}</td></tr>`).join("") +
        `<tr class="total"><td>TOTAL</td><td></td><td class="num">${fmt(counts.totals.neurons)}</td><td class="num">${fmt(counts.totals.connections)}</td><td class="num">${fmt(counts.totals.params)}</td></tr>`;
      $("nb-axons").textContent = fmt(counts.input_axons);
      $("nb-axons").className = "val " + (counts.input_axons > SNN.AXON_LIMITS.total_axons ? "bad" : "good");
      $("nb-neurons").textContent = fmt(counts.totals.neurons);
      $("nb-conns").textContent = fmt(counts.totals.connections);
      $("nb-params").textContent = fmt(counts.totals.params);
      $("nb-updates").textContent = fmt(counts.neuron_updates_per_sample);
      $("nb-flush").textContent = SNN.flushSteps(counts.plan);
    } else {
      tb.innerHTML = `<tr><td colspan="5" style="color:var(--bad);font-family:var(--mono);font-size:12.5px;">${planErr}</td></tr>`;
      ["nb-axons", "nb-neurons", "nb-conns", "nb-params", "nb-updates", "nb-flush"].forEach(id => { $(id).textContent = "—"; $(id).className = "val"; });
    }

    // ---- verdict ----
    $("nb-verdict").innerHTML = feasible
      ? `<div class="verdict pass">PASS &nbsp;·&nbsp; HiAER-Spike feasibility <span class="why">every layer fits the chip's wiring budgets</span></div>`
      : `<div class="verdict fail">FAIL &nbsp;·&nbsp; HiAER-Spike feasibility</div>
         <ul class="violations">${violations.map(v => `<li>${v}</li>`).join("")}</ul>`;

    // ---- CLI ----
    const enc = layers.every(L => L.ch === layers[0].ch && L.k === layers[0].k && L.s === layers[0].s && !L.pool)
      ? `--encoder depth=${layers.length} channels=${layers[0].ch} kernel_size=${layers[0].k} stride=${layers[0].s} norm=${$("nb-norm").value}`
      : `--encoder norm=${$("nb-norm").value} layers_json='${JSON.stringify(layers.map(L => {
          const d = { out_channels: L.ch, kernel_size: L.k, stride: L.s };
          if (L.pool) d.pool = true;
          return d;
        }))}'`;
    $("nb-cli").textContent =
      `python Practice.py summary \\\n  --input resize_to=${$("nb-resize").value} T=${$("nb-T").value} \\\n  ${enc} \\\n  --downsample mode=${$("nb-ds").value} \\\n  --head final_reduction=${$("nb-red").value} fc_widths=${$("nb-fc").value || ""}`;
  }

  function loadPreset(name) {
    const p = PRESETS[name];
    $("nb-resize").value = String(p.resize);
    $("nb-T").value = String(p.T);
    $("nb-ds").value = p.ds;
    $("nb-norm").value = p.norm;
    $("nb-red").value = p.red;
    $("nb-fc").value = p.fc;
    layers = JSON.parse(JSON.stringify(p.layers));
    renderLayerTable();
    update();
  }

  $("nb-preset").addEventListener("change", e => loadPreset(e.target.value));
  ["nb-resize", "nb-T", "nb-ds", "nb-norm", "nb-red"].forEach(id => $(id).addEventListener("change", update));
  $("nb-fc").addEventListener("input", update);
  $("nb-add").addEventListener("click", () => {
    if (layers.length >= 6) return;
    layers.push({ ch: 32, k: 5, s: 2, pool: false });
    renderLayerTable(); update();
  });
  loadPreset("seed1");
})();
