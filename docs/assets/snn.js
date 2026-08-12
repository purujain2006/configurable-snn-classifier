/* ============================================================
   snn.js — faithful JavaScript ports of the torch-free math in
   Practice2.py (sections 1–4): shape planning, HiAER-Spike
   feasibility, the cost model, and the INT16 deployment grid.

   Every function mirrors its Python counterpart line-for-line so
   the interactive demos on this site compute the SAME numbers the
   real code does. Verified against Practice2.py outputs (see
   tools/test_snn_port.js in the repo).
   ============================================================ */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SNN = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /* ---------------- hardware constants (section 3 + 4b) ---------------- */
  const AXON_LIMITS = { total_axons: 16383, fan_out: 4096, fan_in: 8191 };
  const NEURON_LIMITS = { fan_out: 4095, fan_in: 8159 };

  const W_BITS = 16;
  const W_ALPHA = 1.0;
  const INT16_MAX = Math.pow(2, W_BITS - 1) - 1; // 32767
  const W_DELTA = W_ALPHA / INT16_MAX;           // 1/32767
  const HW_TAU_CHOICES = [2, 3, 4, 6, 8, 16, 32, 63];
  const HW_TAU_MIN = 2, HW_TAU_MAX = 128;

  /* ---------------- config defaults (section 1 dataclasses) ------------- */
  function inputSpec(o) {
    return Object.assign({ N: 16, C: 2, H: 128, W: 128, T: 16, resize_to: 0 }, o);
  }
  function convLayerSpec(o) {
    return Object.assign({
      out_channels: 128, kernel_size: 3, stride: 2, padding: 0,
      dilation: 1, pool: false, tau: null, v_threshold: null
    }, o);
  }
  function encoderSpec(o) {
    return Object.assign({
      depth: 3, channels: 128, kernel_size: 3, stride: 2, padding: 0,
      dilation: 1, layers: null,          // JS analogue of layers_json (array of dicts)
      bias: false, norm: "bn", tdbn_alpha: 1.0
    }, o);
  }
  function outputSpec(o) { return Object.assign({ num_classes: 11 }, o); }
  function downsampleSpec(o) {
    return Object.assign({ mode: "stride", pool_type: "avg", pool_kernel_size: 2, pool_stride: 2 }, o);
  }
  function headSpec(o) {
    return Object.assign({ final_reduction: "flatten", fc_widths: "512", dropout_rate: 0.5 }, o);
  }
  function neuronSpec(o) {
    return Object.assign({
      neuron_type: "LIF", tau: 2, v_threshold: 1.0, v_reset: 0.0,
      trainable_tau: false, trainable_threshold: false
    }, o);
  }

  function effectiveHW(input) {
    if (input.resize_to) return [input.resize_to, input.resize_to];
    return [input.H, input.W];
  }

  function resolveConvLayers(encoder) {
    if (encoder.layers && encoder.layers.length) {
      return encoder.layers.map(d => convLayerSpec(d));
    }
    const out = [];
    for (let i = 0; i < encoder.depth; i++) {
      out.push(convLayerSpec({
        out_channels: encoder.channels, kernel_size: encoder.kernel_size,
        stride: encoder.stride, padding: encoder.padding,
        dilation: encoder.dilation, pool: false
      }));
    }
    return out;
  }

  function parseFcWidths(spec) {
    if (spec === null || spec === undefined) return [];
    spec = String(spec).trim();
    if (spec === "" || spec.toLowerCase() === "none" || spec === "0") return [];
    const widths = [];
    for (let chunk of spec.split(",")) {
      chunk = chunk.trim();
      if (!chunk) continue;
      const v = parseInt(chunk, 10);
      if (!(v > 0)) throw new Error(`FC width must be positive, got ${chunk}`);
      widths.push(v);
    }
    return widths;
  }

  /* ---------------- section 2: shape planning -------------------------- */
  class InfeasibleConfig extends Error {}

  function convOutSize(size, k, s, p, d) {
    d = d === undefined ? 1 : d;
    return Math.floor((size + 2 * p - d * (k - 1) - 1) / s) + 1;
  }

  function planEncoder(input, encoder, downsample) {
    let [curH, curW] = effectiveHW(input);
    let inC = input.C;
    const layers = resolveConvLayers(encoder);
    const plans = [];

    layers.forEach((L, i) => {
      const canShrink = curH > 3 && curW > 3;
      const wantPool = L.pool || downsample.mode === "pool";
      let convStride, convPadding;
      if (canShrink && !wantPool) {
        convStride = L.stride;
        convPadding = L.padding;
      } else {
        convStride = 1;
        convPadding = Math.floor((L.dilation * (L.kernel_size - 1)) / 2); // size-preserving
      }

      const oh = convOutSize(curH, L.kernel_size, convStride, convPadding, L.dilation);
      const ow = convOutSize(curW, L.kernel_size, convStride, convPadding, L.dilation);
      if (oh < 1 || ow < 1) {
        throw new InfeasibleConfig(
          `block${i}: spatial size collapses to ${oh}x${ow} ` +
          `(in ${curH}x${curW}, k=${L.kernel_size}, s=${convStride}, p=${convPadding}). ` +
          `Kernel is larger than the feature map.`);
      }

      const doPool = canShrink && wantPool;
      let ph, pw;
      if (doPool) {
        ph = Math.floor(oh / downsample.pool_stride);
        pw = Math.floor(ow / downsample.pool_stride);
        if (ph < 1 || pw < 1) {
          throw new InfeasibleConfig(
            `block${i}: pool collapses ${oh}x${ow} to ${ph}x${pw} ` +
            `(pool_stride=${downsample.pool_stride}).`);
        }
      } else { ph = oh; pw = ow; }

      plans.push({
        index: i, in_channels: inC, out_channels: L.out_channels,
        kernel_size: L.kernel_size, stride: convStride, padding: convPadding,
        dilation: L.dilation, in_hw: [curH, curW], conv_out_hw: [oh, ow],
        pool: doPool,
        pool_kernel: doPool ? downsample.pool_kernel_size : 0,
        pool_stride: doPool ? downsample.pool_stride : 0,
        out_hw: [ph, pw], tau: L.tau, v_threshold: L.v_threshold
      });
      curH = ph; curW = pw; inC = L.out_channels;
    });
    return plans;
  }

  function planNetwork(input, encoder, output, downsample, head) {
    const blocks = planEncoder(input, encoder, downsample);
    let outC, outHW;
    if (blocks.length) {
      outC = blocks[blocks.length - 1].out_channels;
      outHW = blocks[blocks.length - 1].out_hw;
    } else {
      outC = input.C; outHW = effectiveHW(input);
    }
    let fcIn;
    if (head.final_reduction === "gap") fcIn = outC;
    else if (head.final_reduction === "flatten") fcIn = outC * outHW[0] * outHW[1];
    else throw new Error(`Unknown final_reduction ${head.final_reduction}`);

    const widths = parseFcWidths(head.fc_widths);
    const dims = [fcIn].concat(widths).concat([output.num_classes]);
    const linears = [];
    for (let i = 0; i < dims.length - 1; i++) {
      linears.push({
        index: i, in_features: dims[i], out_features: dims[i + 1],
        is_classifier: i === dims.length - 2
      });
    }
    return {
      blocks, linears, reduction: head.final_reduction,
      encoder_out_c: outC, encoder_out_hw: outHW, fc_in_features: fcIn
    };
  }

  /* ---------------- section 3: HiAER-Spike feasibility ------------------ */
  function checkFeasibility(input, encoder, downsample, head, output) {
    head = head || headSpec();
    output = output || outputSpec();
    const violations = [];
    let plan;
    try {
      plan = planNetwork(input, encoder, output, downsample, head);
    } catch (e) {
      if (e instanceof InfeasibleConfig) return [false, [e.message]];
      throw e;
    }

    const firstFcWidth = plan.linears.length ? plan.linears[0].out_features : 0;

    for (const b of plan.blocks) {
      const k = b.kernel_size, s = b.stride;
      const fanIn = k * k * b.in_channels;
      const isLast = b.index === plan.blocks.length - 1;
      let fanOut, fanOutLabel;
      if (isLast) { fanOut = firstFcWidth; fanOutLabel = "fan_out(->fc)"; }
      else { fanOut = Math.pow(Math.ceil(k / s), 2) * b.out_channels; fanOutLabel = "fan_out(->conv)"; }

      if (b.index === 0) {
        const totalAxons = b.in_hw[0] * b.in_hw[1] * b.in_channels;
        if (totalAxons > AXON_LIMITS.total_axons)
          violations.push(`block0: total_axons ${totalAxons} > ${AXON_LIMITS.total_axons}`);
        if (fanOut > AXON_LIMITS.fan_out)
          violations.push(`block0: axonal_${fanOutLabel} ${fanOut} > ${AXON_LIMITS.fan_out}`);
        if (fanIn > AXON_LIMITS.fan_in)
          violations.push(`block0: axonal_fan_in ${fanIn} > ${AXON_LIMITS.fan_in}`);
      } else {
        if (fanOut > NEURON_LIMITS.fan_out)
          violations.push(`block${b.index}: neuron_${fanOutLabel} ${fanOut} > ${NEURON_LIMITS.fan_out}`);
        if (fanIn > NEURON_LIMITS.fan_in)
          violations.push(`block${b.index}: neuron_fan_in ${fanIn} > ${NEURON_LIMITS.fan_in}`);
      }
    }

    for (const lin of plan.linears) {
      const tag = lin.is_classifier ? "classifier" : `fc${lin.index}`;
      if (lin.in_features > NEURON_LIMITS.fan_in)
        violations.push(`${tag}: neuron_fan_in ${lin.in_features} > ${NEURON_LIMITS.fan_in}`);
      if (!lin.is_classifier && lin.out_features > NEURON_LIMITS.fan_out)
        violations.push(`${tag}: neuron_fan_out ${lin.out_features} > ${NEURON_LIMITS.fan_out}`);
    }
    return [violations.length === 0, violations];
  }

  /* ---------------- section 4: cost model ------------------------------- */
  function countNeuronsAndSynapses(cfg) {
    const plan = planNetwork(cfg.input, cfg.encoder, cfg.output, cfg.downsample, cfg.head);
    const encoder = cfg.encoder, input = cfg.input;
    const rows = [];

    for (const b of plan.blocks) {
      const [oh, ow] = b.conv_out_hw;
      const neurons = b.out_channels * oh * ow;
      const wpn = b.kernel_size * b.kernel_size * b.in_channels;
      rows.push({
        layer: `conv${b.index}`,
        detail: `${b.in_channels}x${b.in_hw[0]}x${b.in_hw[1]} -> ` +
                `${b.out_channels}x${oh}x${ow}  k=${b.kernel_size} s=${b.stride} p=${b.padding}`,
        neurons, connections: neurons * wpn,
        params: wpn * b.out_channels + (encoder.bias ? b.out_channels : 0)
      });
      if (encoder.norm !== "none") {
        const label = encoder.norm === "tdbn" ? "tdbn" : "bn";
        rows.push({
          layer: `${label}${b.index}`,
          detail: encoder.norm === "tdbn"
            ? `tdBN(${b.out_channels}) over (T,N,H,W)` : `BatchNorm2d(${b.out_channels})`,
          neurons: 0, connections: 0, params: 2 * b.out_channels
        });
      }
      if (b.pool) {
        const [ph, pw] = b.out_hw;
        rows.push({
          layer: `pool${b.index}`,
          detail: `avg k=${b.pool_kernel} s=${b.pool_stride} -> ${b.out_channels}x${ph}x${pw}`,
          neurons: 0,
          connections: b.out_channels * ph * pw * b.pool_kernel * b.pool_kernel,
          params: 0
        });
      }
    }

    if (plan.reduction === "gap") {
      rows.push({
        layer: "gap",
        detail: `${plan.encoder_out_c}x${plan.encoder_out_hw[0]}x${plan.encoder_out_hw[1]} -> ${plan.encoder_out_c}`,
        neurons: 0,
        connections: plan.encoder_out_c * plan.encoder_out_hw[0] * plan.encoder_out_hw[1],
        params: 0
      });
    }

    for (const lin of plan.linears) {
      rows.push({
        layer: lin.is_classifier ? "classifier" : `fc${lin.index}`,
        detail: `${lin.in_features} -> ${lin.out_features}`,
        neurons: lin.out_features,
        connections: lin.in_features * lin.out_features,
        params: lin.in_features * lin.out_features
      });
    }

    const totals = {
      neurons: rows.reduce((a, r) => a + r.neurons, 0),
      connections: rows.reduce((a, r) => a + r.connections, 0),
      params: rows.reduce((a, r) => a + r.params, 0)
    };
    const [inH, inW] = effectiveHW(input);
    return {
      rows, totals,
      input_axons: inH * inW * input.C,
      timesteps: input.T,
      neuron_updates_per_sample: totals.neurons * input.T,
      plan
    };
  }

  /* number of spiking layers = pipeline flush steps */
  function flushSteps(plan) { return plan.blocks.length + plan.linears.length; }

  /* ---------------- section 4b: INT16 deployment grid ------------------- */
  function clamp(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }

  // mirror of fake_quantize_weight (forward pass only)
  function fakeQuantizeWeight(w) {
    const levels = Math.pow(2, W_BITS - 1) - 1;   // 32767
    const wc = clamp(w / W_ALPHA, -1, 1);
    return Math.round(Math.abs(wc) * levels) / levels * Math.sign(wc || 1) * W_ALPHA * (wc === 0 ? 0 : 1);
  }
  function weightIntCode(w) {                      // the stored INT16 value
    const wc = clamp(w / W_ALPHA, -1, 1);
    return Math.round(wc * INT16_MAX);
  }
  // mirror of quantized_threshold_int
  function quantizedThresholdInt(th) {
    return Math.trunc(clamp(Math.round(th / W_DELTA), 1, INT16_MAX));
  }
  // mirror of fake_quantize_threshold (forward)
  function fakeQuantizeThreshold(th) {
    const thc = clamp(th, W_DELTA, W_ALPHA);
    return Math.round(thc / W_DELTA) * W_DELTA;
  }

  // mirror of _fold_conv_bn_params for scalars
  function foldConvBn(w, convB, gamma, beta, mean, variance, eps) {
    eps = eps === undefined ? 1e-5 : eps;
    const scale = gamma / Math.sqrt(variance + eps);
    return { wPrime: w * scale, bPrime: beta + ((convB || 0) - mean) * scale, scale };
  }

  /* ---------------- neuron dynamics (section 5) ------------------------- */
  // HardwareLIFNode.single_step_forward — the converter's order:
  // 1. fire on membrane carried in from PREVIOUS step (strict >)
  // 2. hard reset  3. integer leak toward v_reset  4. integrate input
  function hardwareLIFStep(state, x, p) {
    const vReset = p.v_reset || 0;
    const spike = state.v > p.v_threshold ? 1 : 0;        // v > theta (strict)
    let v = spike ? vReset : state.v;                     // hard reset
    v = v - (v - vReset) / p.tau;                         // integer leak
    v = v + x;                                            // undecayed input
    state.v = v;
    return spike;
  }

  // SpikingJelly LIFNode (decay_input=True), charge -> fire -> reset:
  function spikingjellyLIFStep(state, x, p) {
    const vReset = p.v_reset || 0;
    let v = state.v + (x - (state.v - vReset)) / p.tau;   // decayed input charge
    const spike = v >= p.v_threshold ? 1 : 0;             // fire same step
    if (spike) v = vReset;                                // hard reset
    state.v = v;
    return spike;
  }

  // surrogate gradient: ATan (spikingjelly's default, alpha=2)
  function atanSurrogateGrad(x, alpha) {
    alpha = alpha || 2.0;
    return alpha / 2 / (1 + Math.pow(Math.PI / 2 * alpha * x, 2));
  }
  function heaviside(x) { return x >= 0 ? 1 : 0; }
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  /* ---------------- misc utilities -------------------------------------- */
  function fmt(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  function fmtCompact(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
    return String(Math.round(n));
  }

  // tiny seeded RNG (mulberry32) so demos are reproducible
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // Box–Muller gaussian
  function gaussian(rand) {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  return {
    AXON_LIMITS, NEURON_LIMITS, W_BITS, W_ALPHA, INT16_MAX, W_DELTA,
    HW_TAU_CHOICES, HW_TAU_MIN, HW_TAU_MAX,
    inputSpec, convLayerSpec, encoderSpec, outputSpec, downsampleSpec, headSpec, neuronSpec,
    effectiveHW, resolveConvLayers, parseFcWidths,
    InfeasibleConfig, convOutSize, planEncoder, planNetwork,
    checkFeasibility, countNeuronsAndSynapses, flushSteps,
    clamp, fakeQuantizeWeight, weightIntCode, quantizedThresholdInt, fakeQuantizeThreshold,
    foldConvBn, hardwareLIFStep, spikingjellyLIFStep,
    atanSurrogateGrad, heaviside, sigmoid,
    fmt, fmtCompact, rng, gaussian
  };
});
