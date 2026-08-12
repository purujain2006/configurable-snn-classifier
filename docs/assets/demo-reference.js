(function () {
"use strict";
const R = [
{g:"§1 · Configuration dataclasses"},
{n:"InputSpec", k:"dataclass", s:"N=16, C=2, H=128, W=128, T=16, resize_to=0", o:"Shape of the input tensor: batch size, polarity channels, sensor resolution, timesteps, optional square resize.",
 b:"resize_to=0 feeds the native 128×128 through unchanged, which needs 32,768 input axons against a limit of 16,383, so working configurations set 32 or 64. T is the number of frames each recording is divided into. See <a href='architecture.html#configs'>3.1</a>."},
{n:"effective_hw", k:"function", s:"effective_hw(input_cfg) -> (H, W)", o:"Resolves the native or resized question into one height and width.",
 b:"Small, but it removes the possibility of one part of the program reading H and W while another reads resize_to."},
{n:"ConvLayerSpec", k:"dataclass", s:"out_channels, kernel_size, stride, padding, dilation, pool, tau, v_threshold", o:"Geometry of one convolution, with optional per-layer neuron parameters.",
 b:"tau and v_threshold default to None, meaning the global NeuronSpec applies. Per-layer neuron parameters deploy at no cost because the chip reads a neuron model per neuron. Per-channel parameters were rejected as too many degrees of freedom for this dataset."},
{n:"EncoderSpec", k:"dataclass", s:"depth, channels, kernel_size, stride, layers_json, bias, norm, tdbn_alpha", o:"The convolution stack, given either as uniform fields or as a per-layer JSON list.",
 b:"norm selects none, bn or tdbn. The model enables bias when norm is none, since a convolution with neither normalisation nor bias holds every output channel at zero mean. See <a href='architecture.html#perlayer'>3.6</a>."},
{n:"resolve_conv_layers", k:"function", s:"resolve_conv_layers(encoder_cfg) -> list[ConvLayerSpec]", o:"Reduces either encoder form to one canonical list.",
 b:"Validates layers_json strictly and reports unknown keys with the list of valid ones. The planner, checker, cost model and builder then handle a single representation."},
{n:"OutputSpec", k:"dataclass", s:"num_classes=11", o:"Width of the classifier.", b:"DVS128 Gesture defines eleven classes, including an other category."},
{n:"DownsampleSpec", k:"dataclass", s:"mode='stride'|'pool', pool_type, pool_kernel_size=2, pool_stride=2", o:"How feature maps lose resolution between layers.",
 b:"Stride mode reduces inside the convolution. Pool mode uses stride-1 convolutions followed by pooling. A per-layer pool flag overrides the global setting. See <a href='architecture.html#planning'>3.2</a>."},
{n:"HeadSpec", k:"dataclass", s:"final_reduction='flatten'|'gap', fc_widths='512', dropout_rate=0.5", o:"How the convolution output becomes class scores.",
 b:"fc_widths is a comma-separated string of hidden widths; empty means the classifier follows the reduction directly, which is what the strongest configurations used. See <a href='architecture.html#head'>3.5</a>."},
{n:"NeuronSpec", k:"dataclass", s:"neuron_type='LIF', tau=2, v_threshold=1.0, v_reset=0.0, trainable_tau, trainable_threshold", o:"Global neuron parameters.",
 b:"tau is an integer because the chip holds the leak in an integer register. A continuous value in [1.5, 2.5] deploys as 2 for every value in the interval. See <a href='neurons.html#leak'>2.3</a>."},
{n:"TrainSpec", k:"dataclass", s:"epochs, optimizer, lr, weight_decay, scheduler, warmup_epochs, label_smoothing, grad_clip, qat_mode, qat_warmup_frac, qat_epochs, qat_lr_scale, fold_bias_mode", o:"The optimisation recipe, covering everything that affects training but not shape.",
 b:"qat_mode selects inline, tail or ptq. fold_bias_mode='threshold' is the only route the current converter deploys. See <a href='hardware.html#qat'>4.5</a>."},
{n:"parse_fc_widths", k:"function", s:"parse_fc_widths('512,256') -> [512, 256]", o:"Parses the hidden width string; empty, none and 0 all give an empty list.", b:"Rejects non-positive widths with a specific message."},
{n:"fill_from_kv", k:"function", s:"fill_from_kv(schema_cls, ['T=16', 'resize_to=64']) -> instance", o:"Fills any dataclass from key=value strings, which is what gives every field a command-line option.",
 b:"Reads the dataclass field list for valid names and types, supports aliases for renamed fields, and reports unknown keys with the valid set. Adding a field adds its option. See <a href='architecture.html#configs'>3.1</a>."},
{n:"build_config_parser / parse_config", k:"function", s:"parse_config(argv) -> {'input': InputSpec, 'encoder': ..., ...}", o:"Defines the seven configuration flags and assembles the dictionary the rest of the program takes.",
 b:"Uses parse_known_args so the configuration parser can serve as a parent parser for each subcommand."},

{g:"§2 · Shape planning"},
{n:"InfeasibleConfig", k:"exception", s:"class InfeasibleConfig(ValueError)", o:"Raised when a configuration cannot be built, such as a kernel larger than the remaining feature map.",
 b:"Representing an impossible network as a typed exception lets the search record a score of zero instead of failing the trial."},
{n:"conv_out_size", k:"function", s:"conv_out_size(size, k, s, p, d=1) -> int", o:"The convolution output size formula.",
 b:"Used interactively in the builder on <a href='architecture.html#builder'>3.4</a>."},
{n:"ConvBlockPlan / LinearPlan / NetPlan", k:"dataclass", s:"NetPlan(blocks, linears, reduction, encoder_out_c, encoder_out_hw, fc_in_features)", o:"The derived geometry of every layer, computed before any module is constructed.",
 b:"conv_out_hw is the post-convolution size, where the neurons sit. out_hw is the post-pooling size that feeds the next block. The builder, the feasibility check and the cost model all read this object."},
{n:"plan_encoder", k:"function", s:"plan_encoder(input, encoder, downsample) -> list[ConvBlockPlan]", o:"Walks the convolution stack computing sizes, applying the small-map guard and the downsampling precedence.",
 b:"A map of 3×3 or smaller forces stride 1 with size-preserving padding. A per-layer pool flag takes precedence over the global mode. A kernel larger than the map raises InfeasibleConfig with the sizes involved. See <a href='architecture.html#planning'>3.2</a>."},
{n:"plan_network", k:"function", s:"plan_network(input, encoder, output, downsample, head) -> NetPlan", o:"Adds the head to the encoder plan and computes the feature count entering the first linear layer.",
 b:"Flatten gives C×H×W features and GAP gives C. That number determines most feasibility outcomes."},

{g:"§3 · Routing limits"},
{n:"AXON_LIMITS / NEURON_LIMITS", k:"constant", s:"16,383 axons · axon fan-out 4,096, fan-in 8,191 · neuron fan-out 4,095, fan-in 8,159", o:"The chip's routing budgets.",
 b:"The first block draws on the axon budgets because it reads the input. Later layers use the neuron budgets. See <a href='hardware.html#limits'>4.1</a>."},
{n:"check_feasibility", k:"function", s:"check_feasibility(input, encoder, downsample, head, output) -> (bool, [violations])", o:"Applies the routing formulas to every convolution block and linear layer in the plan.",
 b:"Fan-in is k²·C_in and fan-out is ceil(k/s)²·C_out, except for the last convolution block, whose fan-out is the width of the first linear layer. An earlier version omitted the head from the check entirely."},

{g:"§4 · Cost model"},
{n:"count_neurons_and_synapses", k:"function", s:"count_neurons_and_synapses(cfg) -> {rows, totals, input_axons, neuron_updates_per_sample, plan}", o:"Counts spiking neurons, realised synapses and trainable parameters per layer and in total.",
 b:"Neurons are counted once rather than per timestep. Connections exceed parameters because one filter is reused at every position. See <a href='architecture.html#cost'>3.3</a>."},
{n:"format_summary", k:"function", s:"format_summary(cfg) -> str", o:"Renders the architecture table, the counts and the feasibility verdict as text.",
 b:"Includes the hardware neuron line with its integer threshold and leak, and the flush step count. This is the output of summary mode."},

{g:"§4b · The INT16 grid"},
{n:"W_BITS / W_ALPHA / INT16_MAX / W_DELTA", k:"constant", s:"16 · 1.0 · 32767 · 1/32767", o:"The numeric grid, taken from the conversion library rather than chosen here.",
 b:"Weights are clamped to [-1, 1] with no rescaling, thresholds must fall in (0, 1] to encode, and the leak is an integer. See <a href='hardware.html#int16'>4.2</a>."},
{n:"HW_TAU_CHOICES", k:"constant", s:"[2, 3, 4, 6, 8, 16, 32, 63]", o:"The integer leak values the search samples.",
 b:"Spaced closely at the low end, where consecutive integers behave differently, and sparsely at the high end, where they do not. 63 matches a known-working conversion. See <a href='neurons.html#leak'>2.3</a>."},
{n:"_ste", k:"function", s:"_ste(quantized, original) = original + (quantized - original).detach()", o:"Straight-through estimator: the quantised value forward, the gradient to the float parameter backward.",
 b:"Used for weight quantisation, threshold quantisation and the learned integer leak. See <a href='hardware.html#ste'>4.4</a>."},
{n:"fake_quantize_weight", k:"function", s:"fake_quantize_weight(w) -> tensor", o:"Reproduces the converter's weight quantisation in a differentiable form: clamp to [-1, 1], round to 32,767 steps per side.",
 b:"Clamping inside the forward pass makes an out-of-range folded weight raise the training loss, rather than appearing only after conversion."},
{n:"fake_quantize_threshold / quantized_threshold_int", k:"function", s:"θ -> grid value in (0, 1] · θ -> integer in [1, 32767]", o:"The two representations of a threshold: the float the stored integer denotes, and the integer itself.",
 b:"Both clamp so that the encoded value is always legal."},
{n:"weight_clip_fraction", k:"function", s:"weight_clip_fraction(net) -> float", o:"Proportion of convolution and linear weights outside the representable range.",
 b:"Meaningful only on the folded network, since folding is what moves weights out of range. Compared against the 2% budget in DEPLOY_LIMITS."},

{g:"§5 · The model"},
{n:"TdBatchNorm2d", k:"class", s:"TdBatchNorm2d(num_features, alpha, v_threshold)", o:"Batch normalisation whose statistics span the time axis as well as the batch, with gamma initialised relative to the firing threshold.",
 b:"SpikingJelly's multi-step mode already flattens time into the batch axis before calling BatchNorm2d, which is the reduction tdBN specifies, so only the initialisation differs. Folds at inference like ordinary batch normalisation."},
{n:"HardwareLIFNode", k:"class", s:"HardwareLIFNode(tau=2, v_threshold=1.0, v_reset=0.0, learn_tau, learn_threshold, ...)", o:"The neuron this project trains and deploys, matching the chip in update order, integer leak, quantised threshold and undecayed input.",
 b:"hw_tau rounds a learned leak with a straight-through gradient. v_threshold is parametrised through a sigmoid when learned. raw_v_threshold() exposes the unclamped value so the deployment check cannot be satisfied by a safety clamp. Covered throughout <a href='neurons.html'>page 2</a>."},
{n:"_WeightFakeQuant", k:"class", s:"parametrisation on module.weight", o:"Makes every read of a weight return its quantised value while gradients reach the float parameter.",
 b:"Registered on every convolution and linear layer by enable_weight_fake_quant."},
{n:"ConvBNFoldQuant", k:"class", s:"ConvBNFoldQuant(conv, bn, bias_mode='threshold', quantize=True)", o:"Replaces a convolution and batch normalisation pair with one module that folds and quantises on every forward pass.",
 b:"Folds from running statistics rather than batch statistics, since running statistics are what deploy. The affine parameters and running statistics keep updating from a separate pass over the raw convolution output. export_folded_conv() freezes the result. See <a href='hardware.html#qat'>4.5</a>."},
{n:"_LegacyLearnableThresholdLIF", k:"class", s:"retained for older checkpoints", o:"The previous trainable-threshold neuron, which is not deployable.",
 b:"Its continuous tau rounds at conversion and its update order does not match the chip. Its docstring records the reasoning behind learnable thresholds: gradient reaches the threshold only through the surrogate, positivity needs a parametrisation rather than a clamp, and weight scale and threshold are jointly under-determined without normalisation."},
{n:"build_neuron", k:"function", s:"build_neuron(neuron_cfg, tau, v_threshold) -> HardwareLIFNode", o:"Returns the deployable neuron on every path, with trainability as a flag rather than a different class.",
 b:"Any other neuron type raises NotImplementedError with the export work that would be required first."},
{n:"DVSGesturePuru", k:"class", s:"DVSGesturePuru(input, encoder, output, downsample, head, neuron)", o:"The network, constructed from the plan so that its shapes cannot diverge from the ones checked.",
 b:"Per block: convolution, normalisation, neuron, optional pooling. Then flatten or global average pooling, then dropout, a bias-free linear layer and a neuron per head layer. to_qat_folded() fuses the pairs in place and export_deployed() freezes the result."},

{g:"§5b · Folding"},
{n:"_fold_conv_bn_params", k:"function", s:"(W, b, gamma, beta, mu, var, eps) -> (W', b')", o:"The fold arithmetic, written without depending on a specific array library.",
 b:"Usable from numpy for reference checks. Shown interactively on <a href='hardware.html#folding'>4.3</a>."},
{n:"fold_bn", k:"function", s:"fold_bn(net, bias_mode='conv'|'threshold') -> folded copy", o:"Returns a batch-norm-free copy, with the folded bias either on the convolution or in per-channel thresholds.",
 b:"Refuses to run on a network in training mode, since that would fold batch statistics instead of running statistics. Freezes a learned threshold after folding, so the sigmoid parameter cannot overwrite the folded value."},
{n:"fold_bias_report", k:"function", s:"fold_bias_report(net) -> [{conv_index, mean and max |b'|/theta}]", o:"Reports how large the folded bias is relative to the threshold, per convolution.",
 b:"A ratio near 0.1 folds into the threshold cleanly. Near 0.5 it does not. Call before folding, on the evaluation-mode network."},
{n:"verify_fold", k:"function", s:"verify_fold(bn_net, folded_net, loader, device) -> {pred_agreement, max_abs_logit_diff, acc_bn, acc_folded, acc_delta}", o:"Compares a network against its folded copy at three levels of sensitivity.",
 b:"Prediction agreement should be exactly 1.0 for a bias-on-convolution fold, and the largest logit difference should be around 1e-5. Accuracy alone is too coarse to detect a subtly incorrect fold. See <a href='hardware.html#verify'>4.6</a>."},

{g:"§5c · Deployment"},
{n:"DEPLOY_LIMITS", k:"constant", s:"min_threshold=0.0 (blocking) · max_weight_clip_frac=0.02 (warning)", o:"The two conditions the deployment audit checks.",
 b:"A folded threshold at or below zero makes the neuron fire unconditionally, so the network is unrepresentable and scores zero. Heavy clipping is reported but allowed, since quantisation-aware training has already had the opportunity to work around it."},
{n:"enable_weight_fake_quant / bake_weight_fake_quant", k:"function", s:"register and then collapse the weight parametrisation", o:"Turns quantised weight reads on during training and writes the quantised values into the state dictionary at the end.",
 b:"Built on torch.nn.utils.parametrize, with leave_parametrized=True when baking."},
{n:"deployment_report", k:"function", s:"deployment_report(net) -> {neurons, weight_clip_frac, min_threshold, deployable, blocking_reasons, warnings}", o:"Audits the folded and quantised network for legality and reports per-layer integer parameters.",
 b:"Reads the unclamped threshold, because a clamp applied for numerical safety would let an illegal fold pass the check that exists to catch it."},
{n:"hardware_export", k:"function", s:"hardware_export(net) -> {w_alpha, w_bits, w_delta, int16_max, flush_steps, neuron_layers}", o:"Writes the integer threshold and leak for each spiking layer.",
 b:"Replaces the single hard-coded threshold and leak in the conversion script, which prevented per-layer neuron parameters from being deployed at all."},
{n:"deploy_and_measure", k:"function", s:"deploy_and_measure(net, cfg, train_loader, val_loader, device) -> (hw_net, metrics)", o:"The tail and ptq path: fold, measure, quantise, measure, optionally fine-tune, then report.",
 b:"Produces hw_val_accuracy, the value the search optimises. The inline path reaches the same endpoint inside run_training."},

{g:"§6–7 · Data and optimisation"},
{n:"DVSResizeAndBinarize", k:"class", s:"(T, C, H, W) frames -> resized, binary", o:"Interpolates each frame and thresholds at zero.",
 b:"Any non-zero contribution becomes a full spike, so no event is lost and density per pixel rises as resolution falls. See <a href='training.html#binarize'>5.2</a>."},
{n:"build_dataloaders", k:"function", s:"build_dataloaders(cfg, data_dir, val_fraction=0.15) -> (train, val, test)", o:"Loads DVS128 Gesture in frame mode with a seeded validation split.",
 b:"The seed keeps every trial scored against the same clips. drop_last is enabled on training only. num_workers is 0 because each trial is already a separate process. See <a href='training.html#dataset'>5.1</a>."},
{n:"build_optimizer", k:"function", s:"build_optimizer(net, train_cfg) -> Adam | AdamW | SGD", o:"Constructs the optimiser, noting that Adam adds weight decay to the gradient while AdamW applies it separately.",
 b:"The distinction matters because weight decay controls how far folded weights spread beyond the representable range."},
{n:"build_scheduler", k:"function", s:"build_scheduler(optimizer, train_cfg, steps_per_epoch) -> (scheduler, step_per_batch)", o:"Cosine, step, onecycle or constant, with optional linear warm-up.",
 b:"Returns whether the schedule advances per batch or per epoch. Shown on <a href='training.html#loop'>5.5</a>."},

{g:"§8 · Training and evaluation"},
{n:"hardware_flush_steps", k:"function", s:"hardware_flush_steps(net) -> int", o:"The number of zero-input timesteps needed to drain the network, equal to the number of spiking layers.",
 b:"Without them the final frames of every clip never reach the classifier, and deployed accuracy falls below anything measured in training. See <a href='training.html#flush'>5.4</a>."},
{n:"forward_over_time", k:"function", s:"forward_over_time(net, x, flush_steps=None) -> rates", o:"Runs T input timesteps and the flush steps, accumulates output spikes, divides by T and resets neuron state.",
 b:"Matches what the conversion script measures. Single-step networks are driven by an explicit loop; tdBN networks receive the whole sequence at once because their statistics span time."},
{n:"train_one_epoch / evaluate", k:"function", s:"one epoch of training · accuracy over a loader", o:"The standard loops, built on forward_over_time.",
 b:"The sequence lengths returned by the collate function are ignored, since split_by='number' gives every clip exactly T frames."},
{n:"run_training", k:"function", s:"run_training(cfg, data_dir, device, report_fn, ckpt_path) -> metrics", o:"The full run used by both single mode and every search trial: warm-up, inline fold, on-grid training, export and measurement.",
 b:"Splits the epoch budget by qat_warmup_frac, keeps the best weights from each phase, and reports every epoch so the early-stopping curve is continuous. The checkpoint holds both networks, the configuration, the metrics and the hardware table. See <a href='training.html#phases'>5.6</a>."},

{g:"§9 · Dataset cache"},
{n:"_quiet_with_heartbeat", k:"context manager", s:"with _quiet_with_heartbeat('building cache'):", o:"Replaces the library's per-file output with one line every twenty seconds, restoring the full log on failure.",
 b:"Captures the real stdout before redirecting so the heartbeat thread can still write."},
{n:"_class_dirs_have_npz and related probes", k:"function", s:"cache completeness checks", o:"Distinguish a finished cache from one that is present but incomplete.",
 b:"Every class directory must exist and contain samples. Incomplete caches are rebuilt rather than trusted."},
{n:"warmup_dataset_cache", k:"function", s:"warmup_dataset_cache(data_dir, T_values)", o:"Builds the frame cache for every T the search will sample, before any parallel trial starts.",
 b:"Two trials building the same directory can corrupt it. Caching depends only on the frame count and split mode, so warming T is sufficient. See <a href='training.html#dataset'>5.1</a>."},

{g:"§10 · The search"},
{n:"config_to_specs", k:"function", s:"config_to_specs(flat_config) -> structured cfg", o:"Converts the sampler's flat parameter dictionary into the dataclass dictionary the rest of the program takes.",
 b:"Builds a per-layer encoder when per-layer keys are present, otherwise a uniform one. Also fixes the search-time invariants: the threshold starts at 1.0 and tau is rounded to an integer."},
{n:"CHANNEL_CHOICES / KERNEL_CHOICES / FC_WIDTH_CHOICES", k:"constant", s:"[32, 64] · [5, 7] · [128, 256, 512]", o:"The narrowed sampling sets.",
 b:"Each carries its supporting statistic in a comment. Collected in the table on <a href='search.html#narrow'>6.6</a>."},
{n:"DefineByRunSpace", k:"class", s:"DefineByRunSpace(batch_size, epochs, data_dir, t_choices, per_layer)(trial)", o:"The conditional search space, written as a picklable class.",
 b:"A nested function cannot be pickled, and Ray checkpoints the sampler by pickling it. Sampling only the parameters that apply to a trial keeps the sampler's model from fitting parameters that had no effect. See <a href='search.html#dbr'>6.5</a>."},
{n:"SEED_CONFIGS", k:"constant", s:"three known configurations evaluated first", o:"Starting points for the sampler, taken from an earlier study.",
 b:"Their geometry transfers. Their recorded accuracies were measured under the older floating-point objective and will not reproduce, which the comment states directly."},
{n:"ResultsWriter", k:"class", s:"append_jsonl / write_json / write_text / append_csv", o:"Writes results to disk as they are produced, with fsync after each record and atomic replacement for multi-line files.",
 b:"An interrupted search keeps everything written so far. See <a href='search.html#results'>6.7</a>."},
{n:"validate_data_dir", k:"function", s:"validate_data_dir(path) -> absolute path", o:"Checks the dataset directory before anything else runs and reports the expected layout.",
 b:"Also detects the case where the intended root is one directory deeper."},
{n:"export_trial_records", k:"function", s:"export_trial_records(results, out_dir) -> csv path", o:"Flattens every trial into configuration, metrics and architecture counts.",
 b:"Recomputes feasibility and counts even for trials that never trained, so an analysis covers the whole sample."},
{n:"trainable", k:"function", s:"trainable(config)", o:"What each search process runs: the feasibility check, floating-point training with per-epoch reports, then the conversion measurement.",
 b:"An infeasible or undeployable configuration scores zero regardless of its training accuracy. See <a href='search.html#funnel'>6.2</a>."},
{n:"_make_streaming_callback", k:"function", s:"driver-side callback bound to a ResultsWriter", o:"The single process that writes shared files, recording progress, new best results, leaderboard rows and rejections.",
 b:"A pruned trial is recorded as pruned rather than as zero, which would be indistinguishable from a configuration that converted badly. Logging errors are caught so they cannot end a search."},
{n:"run_search", k:"function", s:"run_search(args)", o:"Validates the dataset, warms the cache, starts Ray, checks that the search space pickles, wires the sampler and scheduler, runs, and exports.",
 b:"The sampler and the scheduler use different metrics on purpose, so the shared configuration is left without one. Selection uses scope='last', which is the conversion report. See <a href='search.html#metrics'>6.3</a>."},

{g:"§11 · Entry points"},
{n:"run_fold", k:"function", s:"python Practice2.py fold --ckpt ... --data-dir ...", o:"Loads a checkpoint, reports the folded bias ratios, folds, verifies, quantises and writes the deployment artefacts.",
 b:"Saves the folded weights, a numpy archive of every convolution and linear array for use without PyTorch, and the per-layer neuron table with the required flush step count."},
{n:"main", k:"function", s:"summary | single | fold | search", o:"Argument definitions for the four modes and the console output of single mode.",
 b:"Silences one deprecation warning raised by PyTorch's own scheduler composition, with a comment explaining why the fallback path is correct."}
];

const list = document.getElementById("ref-list");
let html = "";
for (const e of R) {
  if (e.g) { html += `<div class="ref-group-title">${e.g}</div>`; continue; }
  html += `<details class="ref" data-search="${(e.n + " " + e.o + " " + e.b).toLowerCase().replace(/<[^>]+>/g, "")}">
    <summary><span class="ref-name">${e.n}</span><span class="ref-kind">${e.k}</span><span class="ref-one">${e.o}</span></summary>
    <div class="ref-body"><div class="ref-sig">${e.s}</div><p>${e.b}</p></div>
  </details>`;
}
list.innerHTML = html;

const box = document.getElementById("ref-search");
box.addEventListener("input", () => {
  const q = box.value.trim().toLowerCase();
  document.querySelectorAll("#ref-list .ref").forEach(d => {
    d.style.display = !q || d.dataset.search.includes(q) ? "" : "none";
    if (q && d.dataset.search.includes(q)) d.open = q.length > 2;
  });
  document.querySelectorAll("#ref-list .ref-group-title").forEach(t => {
    let el = t.nextElementSibling, any = false;
    while (el && !el.classList.contains("ref-group-title")) {
      if (el.classList.contains("ref") && el.style.display !== "none") { any = true; break; }
      el = el.nextElementSibling;
    }
    t.style.display = any || !q ? "" : "none";
  });
});
})();
