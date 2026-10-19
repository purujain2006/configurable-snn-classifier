(function () {
"use strict";
const R = [
{g:"snnsearch/config.py: what a configuration is"},
{n:"InputSpec", k:"dataclass", s:"N=16, C=2, H=128, W=128, T=16, resize_to=0", o:"The input: batch size, channels, image size, timesteps, and the square size to shrink frames to.",
 b:"resize_to=0 keeps the camera's 128×128, which needs 32,768 input axons against the chip's 16,000, so DVS128 configurations shrink it. See <a href='settings.html#anatomy'>D.2</a>."},
{n:"effective_hw", k:"function", s:"effective_hw(input_cfg) -> (H, W)", o:"The input height and width after any resize.", b:"The planner and the model both read this, so they agree on the input size."},
{n:"ConvLayerSpec", k:"dataclass", s:"out_channels, kernel_size, stride, padding, dilation, pool, tau, v_threshold", o:"One convolution layer, with optional per-layer τ and θ.",
 b:"tau and v_threshold default to None, which means the global NeuronSpec applies."},
{n:"EncoderSpec", k:"dataclass", s:"depth, channels, kernel_size, stride, layers_json, bias, norm, tdbn_alpha, dropout_rate", o:"The convolution stack, as identical layers or as a per-layer JSON list.",
 b:"norm is bn or tdbn. dropout_rate here is dropout inside the convolution layers; the head's dropout lives in HeadSpec. See <a href='settings.html#anatomy'>D.2</a>."},
{n:"resolve_conv_layers", k:"function", s:"resolve_conv_layers(encoder_cfg) -> list[ConvLayerSpec]", o:"Turns either encoder form into one list of layers.",
 b:"Rejects unknown keys in layers_json with the list of valid ones."},
{n:"DownsampleSpec", k:"dataclass", s:"mode='stride'|'pool', pool_type, pool_kernel_size=2, pool_stride=2", o:"How images shrink between layers.",
 b:"A per-layer pool flag overrides the global mode. See <a href='hardware.html#planning'>G.1</a>."},
{n:"HeadSpec", k:"dataclass", s:"final_reduction='flatten'|'gap', fc_widths, dropout_rate", o:"How the convolution output becomes class scores.",
 b:"fc_widths is a comma-separated list of hidden layer sizes; empty means the class layer follows directly. See <a href='basics.html#cnn'>A.4</a>."},
{n:"OutputSpec", k:"dataclass", s:"num_classes=11", o:"The number of classes.", b:"DVS128 Gesture has 11."},
{n:"NeuronSpec", k:"dataclass", s:"neuron_type='LIF', tau=2, v_threshold=1.0, v_reset=0.0, trainable_tau, trainable_threshold, integer_leak", o:"The global neuron settings.",
 b:"τ must be a power of two, because the chip leaks by a bit shift. A learned τ is rounded to one in log scale. See <a href='neurons.html#leak'>F.3</a>."},
{n:"TrainSpec", k:"dataclass", s:"epochs, optimizer, lr, weight_decay, scheduler, label_smoothing, grad_clip, rate_penalty, qat_mode, fold_bias_mode, val_smoothing_epochs, ...", o:"The training recipe.",
 b:"qat_mode is inline, tail or ptq (<a href='pipeline.html#phases'>C.3</a>). fold_bias_mode is threshold or neuron, where the folded bias goes (<a href='hardware.html#folding'>G.5</a>). val_smoothing_epochs sets how many epochs are averaged when picking the checkpoint."},
{n:"parse_fc_widths", k:"function", s:"parse_fc_widths('512,256') -> [512, 256]", o:"Reads the hidden-width string.", b:"Empty, none and 0 all mean no hidden layers."},
{n:"fill_from_kv", k:"function", s:"fill_from_kv(schema_cls, values) -> instance", o:"Fills a dataclass from key=value strings.",
 b:"Reads the dataclass's own fields, so every new field gets a command-line option. Unknown keys fail with the list of valid ones. See <a href='settings.html#anatomy'>D.2</a>."},

{g:"snnsearch/planning.py: layer sizes, worked out once"},
{n:"InfeasibleConfig", k:"exception", s:"class InfeasibleConfig(ValueError)", o:"Raised when a network cannot be built, such as a kernel larger than the image.",
 b:"The search records it as a rejected trial instead of crashing."},
{n:"conv_out_size", k:"function", s:"conv_out_size(size, kernel_size, stride, padding, dilation=1) -> int", o:"The convolution output size formula.", b:"See <a href='hardware.html#planning'>G.1</a>."},
{n:"ConvBlockPlan / LinearPlan / NetPlan", k:"dataclass", s:"NetPlan(blocks, linears, reduction, encoder_out_c, encoder_out_hw, fc_in_features)", o:"Every layer's sizes, before any layer is built.",
 b:"conv_out_hw is the size after the convolution, where the neurons sit; out_hw is the size after pooling. The builder, the chip check and the counts all read this."},
{n:"plan_encoder / plan_network", k:"function", s:"plan_network(input_cfg, encoder_cfg, output_cfg, downsample_cfg, head_cfg) -> NetPlan", o:"Walk the layers and record their sizes, then add the head.",
 b:"fc_in_features is the head's parameter count per class. Its fan-in on the chip can be larger (see check_feasibility)."},

{g:"snnsearch/hardware.py: the chip's limits"},
{n:"AXON_LIMITS / NEURON_LIMITS", k:"constant", s:"16,000 axons · axon fan-out 4,096, fan-in 8,191 · neuron fan-out 4,095, fan-in 8,159", o:"The chip's connection limits.",
 b:"16,000 is the most input axons the current FPGA design has run. See <a href='hardware.html#limits'>G.3</a>."},
{n:"HW_TAU_CHOICES", k:"constant", s:"[2, 4, 8, 16]", o:"The τ values a search samples.", b:"Powers of two only. See <a href='neurons.html#leak'>F.3</a>."},
{n:"W_BITS / W_ALPHA / INT16_MAX / W_DELTA", k:"constant", s:"16 · 1.0 · 32767 · 1/32767", o:"The integer grid weights and thresholds live on.", b:"See <a href='hardware.html#int16'>G.4</a>."},
{n:"check_feasibility", k:"function", s:"check_feasibility(input_cfg, encoder_cfg, downsample_cfg, head_cfg=None, output_cfg=None, bias_neurons=False) -> (bool, [violations])", o:"Checks axon count, fan-in and fan-out at every layer.",
 b:"Counts the neurons that really connect: a 2×2 average pool multiplies the next layer's fan-in by 4, and flatten and GAP heads both connect to every neuron of the last convolution layer. With bias_neurons, adds their share of fan-in. See <a href='hardware.html#limits'>G.3</a>."},
{n:"check_spec_feasibility", k:"function", s:"check_spec_feasibility(spec) -> (bool, [violations])", o:"check_feasibility on a whole spec dictionary, reading the bias mode from it.", b:"Every caller in the package uses this one."},

{g:"snnsearch/cost.py: how big a network is"},
{n:"count_neurons_and_synapses", k:"function", s:"count_neurons_and_synapses(cfg) -> {rows, totals, input_axons, neuron_updates_per_sample, plan}", o:"Neurons, connections and parameters per layer and in total.", b:"See <a href='hardware.html#cost'>G.2</a>."},
{n:"format_summary", k:"function", s:"format_summary(cfg) -> str", o:"The layer table, the counts and the chip verdict as text.", b:"What <code>main.py summary</code> prints."},

{g:"snnsearch/neuron.py: the chip's neuron"},
{n:"HardwareLIFNode", k:"class", s:"HardwareLIFNode(tau=2, v_threshold=1.0, v_reset=0.0, learn_tau=False, learn_threshold=False, integer_leak=True, ...)", o:"The neuron the tool trains and deploys: fire, reset, leak, integrate, as the chip does.",
 b:"A learned τ is stored as log2(τ) and rounded, so it is always a power of two. A learned θ is stored through a sigmoid, so it stays in (0, 1]. The leak rounds down on the layer's own integer grid. See <a href='neurons.html'>page 2</a>."},
{n:"ConvBNFoldQuant", k:"class", s:"ConvBNFoldQuant(conv, bn, bias_mode='threshold', quantize=True, fold_bias_margin=0.05, fold_bias_qat_form='threshold')", o:"A convolution with its batch normalization folded in and its weights rounded on every forward pass.",
 b:"In threshold mode the folded bias goes into the next neuron's threshold; in neuron mode it stays on the convolution, rounded to the grid, for bias neurons. See <a href='pipeline.html#phases'>C.3</a>."},
{n:"TdBatchNorm2d", k:"class", s:"TdBatchNorm2d(num_features, alpha=1.0, v_threshold=1.0)", o:"Batch normalization whose averages span time as well as the batch.", b:"Folds like ordinary batch normalization."},
{n:"build_neuron", k:"function", s:"build_neuron(neuron_cfg, tau, v_threshold) -> HardwareLIFNode", o:"The one place a neuron is made.", b:"Any neuron type other than LIF raises an error, since nothing else deploys."},

{g:"snnsearch/quantgrid.py: the integer grid, differentiable"},
{n:"fake_quantize_weight / fake_quantize_bias", k:"function", s:"fake_quantize_weight(w) · fake_quantize_bias(b)", o:"Clip to [−1, 1] and round to the grid in the forward pass; pass the gradient straight through.",
 b:"Training computes with the values the chip will store. See <a href='hardware.html#ste'>G.6</a>."},
{n:"fake_quantize_threshold / quantized_threshold_int", k:"function", s:"fake_quantize_threshold(th) · quantized_threshold_int(th)", o:"A threshold as the value the chip stores, and as the integer itself.", b:"Both keep θ in (0, 1]."},
{n:"fold_bias_band / constrain_fold_bias", k:"function", s:"constrain_fold_bias(b, v_th, margin=0.05)", o:"Keeps a folded bias where θ − b′ is still a legal threshold.",
 b:"Applied during training in threshold mode, so the network learns a bias the chip can store. See <a href='hardware.html#folding'>G.5</a>."},
{n:"weight_clip_fraction", k:"function", s:"weight_clip_fraction(net) -> float", o:"The share of weights outside [−1, 1], read before clipping.", b:"Reading the clipped values would always give zero."},

{g:"snnsearch/model.py: the network"},
{n:"DVSGesturePuru", k:"class", s:"DVSGesturePuru(input_cfg, encoder_cfg, output_cfg, downsample_cfg, head_cfg, neuron_cfg, bias_mode='threshold')", o:"The network, built from the plan.",
 b:"Each layer is convolution, normalization, neuron and optional pool; the head is flatten or GAP, then linear layers and neurons. With bias neurons the head layers get a bias. to_qat_folded() switches to training on the grid; export_deployed() freezes the deployable network."},
{n:"build_model", k:"function", s:"build_model(cfg) -> DVSGesturePuru", o:"Builds the network from a spec dictionary.", b:"Reads the bias mode from the TrainSpec."},

{g:"snnsearch/encoders.py and data/: from recordings to spikes"},
{n:"build_encoder", k:"function", s:"build_encoder(coding, T, resize_to=None)", o:"Picks the input coding: direct, poisson, temporal or passthrough.", b:"See <a href='data.html#frames'>B.2</a>."},
{n:"PassthroughEncoder", k:"class", s:"PassthroughEncoder(T, size=None, binarize=True)", o:"For event data: shrink each frame, then turn any nonzero pixel into a spike.", b:"No event disappears when frames shrink."},
{n:"build_dataloaders", k:"function", s:"build_dataloaders(bundle, batch_size, encoder=None, val_fraction=0.15, num_workers=0, seed=1, val_split='subject') -> (train, val, test)", o:"The three loaders.",
 b:"val_split='subject' holds out whole people for validation, like the test set. 'clip' reproduces the old random-clip split. See <a href='data.html#split'>page B</a>."},
{n:"split_indices / subject_ids", k:"function", s:"split_indices(dataset, val_fraction=0.15, seed=1, val_split='subject')", o:"Chooses the validation clips.", b:"Reads each clip's person from its file name."},
{n:"dvs128_gesture / warmup_frame_cache", k:"function", s:"dvs128_gesture(root, T=16) · warmup_frame_cache(root, T_values)", o:"Loads DVS128 Gesture, and builds its frame cache before a search starts.", b:"Two trials building the same cache at once can corrupt it."},
{n:"build_dataset / register / load_from_file", k:"function", s:"build_dataset(spec)", o:"How a dataset plugs in: by built-in name, or by pointing at your own Python function.", b:"See <a href='guide.html#scope'>the guide</a>."},

{g:"snnsearch/train.py: training"},
{n:"forward_over_time", k:"function", s:"forward_over_time(net, x, flush_steps=None) -> rates", o:"Runs T input steps plus one flush step per spiking layer and returns spike rates.", b:"Every accuracy number comes from this. See <a href='basics.html#snn-net'>A.7</a>."},
{n:"hardware_flush_steps", k:"function", s:"hardware_flush_steps(net) -> int", o:"The number of flush steps: one per spiking layer.", b:"See <a href='basics.html#snn-net'>A.7</a>."},
{n:"param_groups / build_optimizer", k:"function", s:"build_optimizer(net, train_cfg) -> Adam | AdamW | SGD", o:"Builds the optimizer, with weight decay on convolution and linear weights only.", b:"Decay on τ used to pull it toward zero. See <a href='neurons.html#leak'>F.3</a>."},
{n:"build_scheduler", k:"function", s:"build_scheduler(optimizer, train_cfg, steps_per_epoch)", o:"Cosine, step, onecycle or constant learning rate, with optional warm-up.", b:"See <a href='pipeline.html#step'>C.2</a>."},
{n:"train_one_epoch / evaluate", k:"function", s:"train_one_epoch(net, loader, optimizer, device, ...) · evaluate(net, loader, device)", o:"One epoch of training, and accuracy over a loader.", b:"Both use forward_over_time."},
{n:"run_training", k:"function", s:"run_training(cfg, data_dir=None, device=None, report_fn=None, ckpt_path=None, loaders=None) -> metrics", o:"One full run: warm-up, fold, training on the grid, export, then the simulator check.", b:"Used by both <code>single</code> and every search trial. See <a href='pipeline.html#phases'>C.3</a>."},

{g:"snnsearch/folding.py and quantize.py: the deployable network"},
{n:"fold_bn", k:"function", s:"fold_bn(net, bias_mode='conv') -> folded copy", o:"Removes batch normalization by folding it into the weights.", b:"Refuses a network in training mode. See <a href='hardware.html#folding'>G.5</a>."},
{n:"verify_fold", k:"function", s:"verify_fold(bn_net, folded_net, loader, device, max_batches=None)", o:"Compares a network with its folded copy: same predictions, how far the scores differ, and accuracy.", b:"See <a href='pipeline.html#sim'>C.4</a>."},
{n:"deployment_report", k:"function", s:"deployment_report(net)", o:"Checks every threshold is legal and reports clipping.", b:"Reads thresholds before any clamp."},
{n:"hardware_export", k:"function", s:"hardware_export(net)", o:"Each spiking layer's integer thresholds, τ and leak shift.", b:"leak_shift is the number the chip's neuron takes."},
{n:"deploy_and_measure", k:"function", s:"deploy_and_measure(net, cfg, train_loader, val_loader, device, report_fn=None)", o:"The tail and ptq path: fold, round, optionally fine-tune, measure.", b:"The inline path does the same inside run_training."},

{g:"snnsearch/hs_export.py: the chip simulator check"},
{n:"IntegerSim", k:"class", s:"IntegerSim(layers, in_shape, device='cpu').run(frames, flush_steps)", o:"The chip simulator's arithmetic, using convolutions instead of a list of connections.",
 b:"Matches hs_api's simpleSim on every potential and spike in the self-test. Raises if a potential nears the simulator's 32-bit range. See <a href='pipeline.html#sim'>C.4</a>."},
{n:"integer_layers", k:"function", s:"integer_layers(net, tau_override=None) -> [layer dicts]", o:"The deployed network as integers: weights, thresholds, biases, τ and leak shift per layer.", b:"Folds average pooling and GAP into the next layer."},
{n:"verify_deployment", k:"function", s:"verify_deployment(hw_net, loader, device, flush_steps, max_clips=None)", o:"Runs every clip through PyTorch and IntegerSim and counts disagreements.", b:"Run on every search trial."},
{n:"load_deployed", k:"function", s:"load_deployed(ckpt_path, device='cpu') -> (hw_net, spec, ckpt)", o:"Rebuilds the deployable network from a checkpoint.", b:"Checks the rebuilt thresholds against the ones saved at training time."},
{n:"split_bias", k:"function", s:"split_bias(value, n_targets) -> [(weight, targets)]", o:"Spreads one channel's bias over as many bias neurons as the chip needs.", b:"Each connection holds at most 32,767 and each neuron feeds at most 4,095 others."},
{n:"unroll_to_hs_api", k:"function", s:"unroll_to_hs_api(layers, in_shape) -> (axons, connections, keys)", o:"Writes a network out as hs_api's lists of axons, neurons and connections.", b:"Used by the self-test, and to run small networks on the real simulator."},
{n:"make_chip_faithful", k:"function", s:"make_chip_faithful(net, tau=None)", o:"Makes an older checkpoint compute what the chip computes.", b:"Sets a power-of-two τ and the chip's leak rounding."},

{g:"snnsearch/synops.py: activity"},
{n:"measure_synops", k:"function", s:"measure_synops(net, loader, device, spec=None, max_batches=None)", o:"Spike-triggered additions per clip, measured on real data.", b:"See <a href='hardware.html#cost'>G.2</a>."},
{n:"score_with_energy", k:"function", s:"score_with_energy(accuracy, synops, mode='accuracy', ...)", o:"Combines accuracy and SynOps by the chosen objective.", b:"accuracy, constrained, weighted or pareto."},

{g:"snnsearch/spaces.py and search.py: the search"},
{n:"DefineByRunSpace", k:"class", s:"DefineByRunSpace(batch_size, epochs, data_dir_abs, t_choices, per_layer=True, require=())", o:"What a trial samples, shape first.",
 b:"A shape that breaks a chip limit stops sampling there, so training settings are never blamed for it. See <a href='pipeline.html#shape-first'>C.6</a>."},
{n:"config_to_specs", k:"function", s:"config_to_specs(config) -> spec dictionary", o:"Turns a sampled trial into dataclasses.", b:"Reads bias_mode into TrainSpec.fold_bias_mode."},
{n:"geometry_fits", k:"function", s:"geometry_fits(flat) -> bool", o:"The chip check on the shape sampled so far.", b:"Runs before any training setting is sampled."},
{n:"run_search", k:"function", s:"run_search(cfg, out_dir)", o:"Runs the whole search with Optuna and Ray.", b:"ASHA stops trials on the per-epoch score; Optuna ranks them on the deployed one. See <a href='settings.html#anatomy'>D.2</a>."},
{n:"make_streaming_callback", k:"function", s:"make_streaming_callback(writer, target=0.975)", o:"Writes each trial's result the moment it arrives.", b:"See <a href='pipeline.html#timeline'>C.11</a>."},
{n:"ResultsWriter", k:"class", s:"append_jsonl / write_json / append_csv", o:"Append-only result files, flushed on every write.", b:"An interrupted search keeps everything written so far."},

{g:"snnsearch/pipeline.py and cli.py: running it"},
{n:"resolve_specs / make_loaders / prepare", k:"function", s:"prepare(cfg, flat_config=None) -> (bundle, encoder, loaders, spec)", o:"From a run configuration to datasets, loaders and specs.", b:"single and search share this path."},
{n:"run_single", k:"function", s:"run_single(cfg, ckpt='best.pth', from_best=None, epochs=None, ...)", o:"Trains one configuration and scores it on test once.", b:"The only place a test score is computed."},
{n:"main", k:"function", s:"python main.py check | summary | single | search | report", o:"The command line.", b:"The five modes are described in the user guide in the repository."}
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
const countEl = document.getElementById("ref-count");
const emptyEl = document.getElementById("ref-empty");
const TOTAL = document.querySelectorAll("#ref-list .ref").length;
const GROUPS = document.querySelectorAll("#ref-list .ref-group-title").length;

function applyFilter() {
  const q = box.value.trim().toLowerCase();
  let shown = 0;
  document.querySelectorAll("#ref-list .ref").forEach(d => {
    const hit = !q || d.dataset.search.includes(q);
    d.style.display = hit ? "" : "none";
    if (hit) shown++;
    if (q && hit) d.open = q.length > 2;
    if (!q) d.open = false;
  });
  document.querySelectorAll("#ref-list .ref-group-title").forEach(t => {
    let el = t.nextElementSibling, any = false;
    while (el && !el.classList.contains("ref-group-title")) {
      if (el.classList.contains("ref") && el.style.display !== "none") { any = true; break; }
      el = el.nextElementSibling;
    }
    t.style.display = any || !q ? "" : "none";
  });
  if (emptyEl) emptyEl.hidden = shown !== 0;
  if (countEl) {
    countEl.textContent = q
      ? `${shown} of ${TOTAL} entries match "${box.value.trim()}"`
      : `${TOTAL} entries across ${GROUPS} modules`;
  }
}

box.addEventListener("input", applyFilter);
applyFilter();
})();
