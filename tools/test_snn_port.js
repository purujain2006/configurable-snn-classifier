/* Validates docs/assets/snn.js against ground-truth JSON produced by the
   torch-free sections of Practice2.py. Run: node tools/test_snn_port.js <gt.json> */
const fs = require("fs");
const path = require("path");
const SNN = require(path.join(__dirname, "..", "docs", "assets", "snn.js"));

const gtPath = process.argv[2] || path.join(__dirname, "ground_truth.json");
const GT = JSON.parse(fs.readFileSync(gtPath, "utf8"));

function makeCfg(name) {
  const S = SNN;
  const C = {
    seed_d2_k7: () => ({
      input: S.inputSpec({ resize_to: 64, T: 16 }),
      encoder: S.encoderSpec({ depth: 2, channels: 32, kernel_size: 7, stride: 2 }),
      output: S.outputSpec(), downsample: S.downsampleSpec(),
      head: S.headSpec({ fc_widths: "", dropout_rate: 0.28 }), neuron: S.neuronSpec()
    }),
    seed_d3_k7: () => ({
      input: S.inputSpec({ resize_to: 64, T: 16 }),
      encoder: S.encoderSpec({ depth: 3, channels: 32, kernel_size: 7, stride: 2 }),
      output: S.outputSpec(), downsample: S.downsampleSpec(),
      head: S.headSpec({ fc_widths: "", dropout_rate: 0.35 }), neuron: S.neuronSpec()
    }),
    default_fail: () => ({
      input: S.inputSpec(), encoder: S.encoderSpec(), output: S.outputSpec(),
      downsample: S.downsampleSpec(), head: S.headSpec(), neuron: S.neuronSpec()
    }),
    gap_fc512: () => ({
      input: S.inputSpec({ resize_to: 64, T: 16 }),
      encoder: S.encoderSpec({ depth: 3, channels: 64, kernel_size: 5, stride: 2 }),
      output: S.outputSpec(), downsample: S.downsampleSpec(),
      head: S.headSpec({ final_reduction: "gap", fc_widths: "512" }), neuron: S.neuronSpec()
    }),
    pool_mode: () => ({
      input: S.inputSpec({ resize_to: 64, T: 8 }),
      encoder: S.encoderSpec({ depth: 2, channels: 32, kernel_size: 5, stride: 1 }),
      output: S.outputSpec(),
      downsample: S.downsampleSpec({ mode: "pool", pool_type: "avg" }),
      head: S.headSpec(), neuron: S.neuronSpec()
    }),
    per_layer: () => ({
      input: S.inputSpec({ resize_to: 64, T: 16 }),
      encoder: S.encoderSpec({
        layers: [
          { out_channels: 32, kernel_size: 7, stride: 2 },
          { out_channels: 64, kernel_size: 5, stride: 1, pool: true },
          { out_channels: 32, kernel_size: 5, stride: 2, tau: 4, v_threshold: 0.8 }
        ]
      }),
      output: S.outputSpec(), downsample: S.downsampleSpec(),
      head: S.headSpec({ fc_widths: "256" }), neuron: S.neuronSpec()
    }),
    tiny_maps: () => ({
      input: S.inputSpec({ resize_to: 32, T: 8 }),
      encoder: S.encoderSpec({ depth: 4, channels: 32, kernel_size: 5, stride: 2 }),
      output: S.outputSpec(), downsample: S.downsampleSpec(),
      head: S.headSpec({ fc_widths: "" }), neuron: S.neuronSpec()
    }),
    collapse: () => ({
      input: S.inputSpec({ resize_to: 32, T: 8 }),
      encoder: S.encoderSpec({ depth: 5, channels: 32, kernel_size: 7, stride: 2 }),
      output: S.outputSpec(), downsample: S.downsampleSpec(),
      head: S.headSpec({ fc_widths: "" }), neuron: S.neuronSpec()
    })
  };
  return C[name]();
}

let pass = 0, fail = 0;
function check(label, got, want) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g === w) { pass++; }
  else { fail++; console.log(`FAIL ${label}\n  got:  ${g}\n  want: ${w}`); }
}

for (const [name, gt] of Object.entries(GT)) {
  if (name.startsWith("_")) continue;
  const cfg = makeCfg(name);
  const [feasible, violations] = SNN.checkFeasibility(
    cfg.input, cfg.encoder, cfg.downsample, cfg.head, cfg.output);
  check(`${name}.feasible`, feasible, gt.feasible);
  check(`${name}.violations`, violations, gt.violations);

  if (gt.infeasible_error) {
    let threw = null;
    try { SNN.countNeuronsAndSynapses(cfg); }
    catch (e) { threw = e.message; }
    check(`${name}.infeasible_error`, threw, gt.infeasible_error);
    continue;
  }
  const counts = SNN.countNeuronsAndSynapses(cfg);
  check(`${name}.totals`, counts.totals, gt.totals);
  check(`${name}.input_axons`, counts.input_axons, gt.input_axons);
  check(`${name}.updates`, counts.neuron_updates_per_sample, gt.neuron_updates_per_sample);
  check(`${name}.rows`, counts.rows, gt.rows);
  check(`${name}.fc_in`, counts.plan.fc_in_features, gt.fc_in_features);
  check(`${name}.blocks`, counts.plan.blocks.map(b => ({
    index: b.index, in_channels: b.in_channels, out_channels: b.out_channels,
    kernel_size: b.kernel_size, stride: b.stride, padding: b.padding,
    in_hw: b.in_hw, conv_out_hw: b.conv_out_hw, pool: b.pool, out_hw: b.out_hw
  })), gt.blocks);
  check(`${name}.linears`, counts.plan.linears, gt.linears);
}

// quantization constants + threshold ints
const C = GT._constants;
check("W_DELTA", SNN.W_DELTA, C.W_DELTA);
check("INT16_MAX", SNN.INT16_MAX, C.INT16_MAX);
check("HW_TAU_CHOICES", SNN.HW_TAU_CHOICES, C.HW_TAU_CHOICES);
for (const [th, want] of Object.entries(GT._threshold_int_examples)) {
  check(`thresholdInt(${th})`, SNN.quantizedThresholdInt(parseFloat(th)), want);
}

console.log(`\n${pass} checks passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
