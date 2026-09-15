# Recruiter and lab-member review

The site should establish what the lab needed to decide, show the tool making that decision, and then explain the implementation. It previously asked readers to learn the chip and library vocabulary before making the tool's purpose clear.

## Recruiter lens

Useful evidence includes the configurable model builder, checks performed before training, consistent evaluation after hardware-specific transformations, saved experiment records, and numerical tests of the browser demonstrations. The recorded trial outcomes and candidate comparisons make these capabilities inspectable.

The strongest opening is a concrete failure a researcher can encounter: a two-channel 128×128 input requires 32,768 input axons, exceeding the package's tested 16,000 limit. The new input-budget demonstration shows how changing resolution affects this check. It also states that fitting the input does not establish that the entire network fits.

The original headline emphasis on a 14.6-point margin over a published reference implied more comparability than the protocols support. The reference remains in the results discussion, but its margin no longer occupies a headline metric card.

Still worth adding when evidence is available:

- A precise account of personal ownership and collaborators' contributions. The repository identifies implementation structure, but it does not establish who designed each part. Do not invent attribution.
- An example of another lab member using a saved configuration or exported model, with the resulting experiment. This would support a claim of adoption.
- A fixed-budget comparison against manual selection or random search. Trial counts demonstrate filtering, but do not establish hours saved or superiority of the search algorithm.
- Physical-device measurements tied to the reported checkpoint and configuration. Software evaluation of the transformed model does not establish board execution, latency, or energy consumption.

## Lab-member lens

The reader needs to know what input is accepted, which models can be built, how trials are selected, what each metric measures, which artifacts to retain, and what a successful conversion check actually establishes.

| Where the reader loses context | Why it matters | Revision or next action |
| --- | --- | --- |
| Intro begins with device-specific terminology | The lab task is unclear | Describe the experiment workflow, then show the input-budget failure. |
| LIF and membrane potential appear in the first demo | Controls require a neuron model the reader has not learned | Define potential, leak, threshold, and LIF before interaction. |
| Input axons and connection counts | A small parameter count can still exceed routing capacity | Explain an axon in the demo and define fan-in and fan-out on Hardware. |
| The site says 16,383 but Python uses 16,000 | The demo can approve a configuration the package rejects | Synchronize the demo and displayed limits. Preserve the distinction between register capacity and the tested L6m ceiling. |
| N, C, H, W, T | Readers cannot interpret tensor shapes or tell batch size from hardware input size | Define each dimension on Architecture and Training; distinguish batch size from per-example connections. |
| Encoder has two meanings | Feature extraction is confused with input coding | Explain EncoderSpec as the convolution stack and input encoding as the time-dependent input rule. |
| layers_json and dataclass names | Implementation names precede their purpose | Put the model description first and make the field table optional. Define JSON and the override behavior. |
| tau, theta, x, and v | The update equation lacks a key | Define the symbols immediately above the rule. |
| w_alpha, w_delta, and INT16 | Readers cannot interpret the quantization example | Explain scale, spacing, and integer representation before the formula. |
| PyTorch, SpikingJelly, hs_api | Library responsibility is unclear | Add a training-library guide and distinguish the device interface from software evaluation. |
| Ray, Optuna, and ASHA | An unfamiliar stack obscures the experiment logic | Introduce trial and hyperparameter, then explain proposing settings, scheduling experiments, and allocating training time. |
| grace_period and reduction_factor | Early-stopping behavior is impossible to predict | Give an 8/24/72-report example and explain asynchronous comparison and late-improvement risk. |
| float_val_accuracy | The name can be read as proof of floating-point-only training | Explain that the historical intermediate metric can span quantization-aware phases. |
| hw_val_accuracy | The name can be mistaken for a physical-chip measurement | Explicitly identify software evaluation of the transformed model. |
| val_accuracy | Readers may assume it is interchangeable with either score above | Explain the final objective and rejected-candidate zero. |
| “Configurable” sounds like arbitrary model support | A lab member may try to supply an unsupported network | Add a model-support matrix and identify the fixed builder. |
| A custom dataset loader sounds sufficient for any task | Detection and regression also need different heads, losses, and metrics | State the current classification contract and additional task work. |
| “Deployable” sounds like a complete hardware certification | The existing checks cover particular constraints | Distinguish shape checks, parameter checks, transformed software evaluation, and physical validation. |
| Results list scores without a decision | The reader cannot tell which experiment to run next | Explain the accuracy/operation choice and propose repeated runs plus matched hardware measurement. |

## Can it operate on any model?

No. `snnsearch/model.py::build_model` constructs `DVSGesturePuru`, a configurable sequential convolutional spiking classifier. The planner and transformation code assume that family. The historical class name does not mean all inputs must be DVS Gesture, but the dataset interface still expects image-shaped classification examples and class-count metadata.

There are two different extension goals:

1. **Run searches for other models.** Introduce a model/task adapter with construction, input shape, state reset, loss, metrics, and checkpoint behavior. Experiment scheduling can be reused without promising hardware compatibility.
2. **Deploy more model families.** Introduce graph planning and explicit operation support, including shape, connection, timing, quantization, and conversion rules. Branches and recurrent state need temporal semantics. Validate each operation and then complete networks against the device runtime.

A separate hardware backend would be needed for another device. Changing numeric connection limits does not change neuron behavior or conversion. Unsupported operations may need substitution or a supported student model trained to imitate the original; these are research directions requiring new evaluation, not existing capabilities.

## Keeping the explanation understandable

- Put the question and observed example before the API name.
- Define necessary symbols locally even when a glossary exists elsewhere. Readers arrive through direct links.
- Keep the main explanation visible. Put source code, field tables, and longer derivations in optional sections.
- Follow each experiment with the decision its evidence supports and the specific uncertainty that remains.
- State which figures belong to the recorded study and which rules describe the current implementation.
- Avoid claiming complete novice comprehension without testing it. Ask a new reader to explain why a trial stops, what the final score measures, and whether their own model is supported. Their answers reveal missing transitions.

Library role descriptions were checked against [Ray Tune's concepts guide](https://docs.ray.io/en/latest/tune/key-concepts.html) and [Optuna's official overview](https://optuna.org/). Support and hardware claims were checked against the local model, planner, data, search, training, and quantization modules.
