# Website editorial and visual audit

Reviewed 13 September 2026. This report covers all eight documentation pages, their shared styles and navigation, and visible reference/demo text. It records editorial changes to the working files as they existed when review began, including pre-existing edits. Source-code behavior and new training experiments are outside this revision.

“LLM-indicative” is used here as an editorial description, not an authorship determination. Semicolons, hyphens, first person, and lists of three are not defects by themselves. The useful test is whether a passage is precise, proportionate to the evidence, and easy to read. Technical compounds such as “leaky integrate-and-fire” and “multiply-accumulate” should remain.

Mercor’s public [Writing great instructions](https://www.mercor.com/docs/writing-great-instructions/) discusses clear annotation instructions and consistency with a specified style. It does not provide a public prohibition matching every item in the requested checklist. This audit applies the user’s criteria and distinguishes editorial judgments from verified research claims.

The published reference is [HiAER-Spike, npj Unconventional Computing](https://www.nature.com/articles/s44335-026-00062-8). It supports the 68.75% comparison value. It does not establish that this project reproduced the same preprocessing, execution environment, or energy measurement.

## Priority findings in results.html

1. **Introduction:** “retrained long,” “data nothing had selected against,” and “the numbers around it” obscure the procedure. Replace with the number of candidates, training budget, test range, and observed computation tradeoff.
2. **Test protocol:** “evaluated exactly once at the end” conflicts with five candidate evaluations and a test-scored decay sweep. Describe each evaluation and disclose whether test results influenced further choices. Evaluation once per fitted model is different from a test set used once for a whole study.
3. **Baseline:** “same task, same test set” is insufficient for an equivalent comparison. Cite the paper, describe its different input resolutions, and call the margin descriptive.
4. **Significance:** “about four standard errors,” “noise floor of 1.3 clips,” and “scores closer than 0.015 are indistinguishable” do not establish paired test significance. Remove those claims. Obtain paired predictions and repeated runs before making stronger statements.
5. **Reproduction:** “which rules out ordinary run-to-run variation” is stronger than two agreeing replays justify. State that the mismatch prompted input-pipeline inspection and that the inspection found different inputs.
6. **Counting:** “Three consequences” introduces a four-row table. Replace the count with a descriptive introduction and distinguish the separate SynOps bug.
7. **Failure rhetoric:** “Dying there costs one trial. Not checking cost four hundred.” turns a useful assertion into a slogan. Explain what the assertion checks and when it stops a trial.
8. **Architecture causation:** “pooling…buys 3 to 4.5 points,” “preserves evidence,” and “the second-block choice is doing the work rather than the width” are unisolated causal claims. Channel counts also differ. Present the observed scores and require a matched ablation.
9. **Metric confusion:** “SynOps per sample is what the chip spends” and “only one…is about energy” mislabel an operation estimate. Define SynOps and dense MACs. Neither measures energy in these runs.
10. **Tradeoff arithmetic:** rank 1 versus rank 3 differs by 2.78 percentage points, not the loose “3.5 points” used in the open question. Replace with exact candidate values.
11. **Decay effect:** “buys accuracy and energy at the same time” conflates a three-clip accuracy difference, operation counts, and energy. Report 37.5% fewer counted operations and a three-clip increase, with one run per setting.
12. **False equality:** 68.40% is not “exactly” the 68.75% baseline. State both values.
13. **Optimum:** three decay settings do not establish the optimum. Say 0.001 performed best among the three values tested.
14. **Replication:** the search correlation and follow-up sweep are not established as independent replication. Their sampling, shared data, and experiment selection must be considered.
15. **Grokking:** define delayed generalization, bound the observation to 300 epochs, and remove unsupported general timing ratios. A longer-training test should retain other settings rather than changing smoothing and folding simultaneously.
16. **GAP:** “loses on merit,” “all 30…died,” and “not a slow starter” overstate early-stop data. Report the checkpoint scores and leave final accuracy unknown. Resolve “all 30” versus the table’s 97% with exact counts.
17. **Dropout:** “unmeasured” is inaccurate when 12 final scores exist. Use “insufficient evidence.” A p-value of 0.50 is not evidence of no effect.
18. **Penalty:** “instructing the network to stop answering” personifies an objective conflict. Explain that a rate penalty also suppressed the output spike rates used as class scores.
19. **Selection analysis:** “the p-values are real; the attribution is not” is a slogan that ignores test assumptions. Distinguish multiplicity adjustment from confounding, adaptive sampling, and early stopping.
20. **Uncertainty:** epoch variation is not a proven lower bound on all uncertainty. Five architectures are not five seeds of one architecture. The 78.8–83.3% range is descriptive, not a confidence interval for the method.
21. **Provenance:** the local text analysis describes 82 older trials, not the 400-trial study. Retain reported values with an explicit provenance note. Do not relabel the older report as validation of the current estimates.
22. **Runtime estimates:** the changed dropout proposal no longer has the same cost as the original forced-on search. Mark its runtime as needing estimation. Label other retained costs as estimates.
23. **Search shares:** 21% + 52% + 28% totals 101% due to rounding. Show the exact two-decimal shares, 20.50%, 51.75%, and 27.75%.

## Website style and visual changes

| ID | Instance / location | Problem | Change implemented |
|---|---|---|---|
| V01 | Long introductory paragraph on index | Several purposes and future ambitions compete with the immediate explanation | Three sentences describing the package, configuration, and evaluation |
| V02 | Results hero | Dense narrative preamble delays the finding | Lead with test range and computation tradeoff |
| V03 | Shared `.hero` | Oversized title and large blank area resemble a promotional landing page | Smaller title, reduced padding, readable lead width |
| V04 | Results `.verdict` blocks | Repeated bordered verdict panels give ordinary observations excessive authority | Convert to simple subsections with short labels |
| V05 | Duplicate shared `.verdict` rules | Later flex and monospace styling can override prose-callout styling | Results findings use a distinct `.finding` class; interactive pass/fail controls retain their layout |
| V06 | Results sentence-length claims | Long bold headings duplicate following paragraphs | Use concise descriptive headings |
| V07 | Shared `.teach` | Dashed borders, capitals, and wide letter spacing overdecorate explanations | Solid subtle border, sentence case, ordinary type |
| V08 | Dense explanations on technical pages | Readers lack a quick orientation before implementation details | Add one compact page recap |
| V09 | Results decay/statistics sections | Readers must read several paragraphs to recover the main implication | Add short section recaps |
| V10 | Results search-space table | Secondary configuration detail interrupts the findings | Put ranges in a native disclosure |
| V11 | Results statistical estimates | Four prominent metrics imply unwarranted certainty | Put detailed estimates and methods behind an explicit disclosure with a visible limitation recap |
| V12 | Results missing TOC | Script selects `h2[id]`, but IDs are on sections | Resolve headings to their containing section IDs |
| V13 | TOC below desktop breakpoint | Long pages have no section navigation on narrower screens | Show a compact two-column contents list above the article |
| V14 | Dense wide tables | Large padding and all-caps headers force awkward wrapping | Smaller padding, sentence-case headers, tabular numeric values |
| V15 | Results rank column | Rows sorted by test score resemble a broken ranking | Label “Search rank” and explain row order |
| V16 | Results fractions versus percentages | 0.8333 and 83.3% require mental conversion | Percentages in summary metrics; explicit fraction notes beside detailed tables |
| V17 | Results red/green data | Value judgments depend on color, including opposite judgments for different cost metrics | Neutral table values, explicit units, subtle selected-row highlight |
| V18 | Voided result table | Struck-through multi-line rows are difficult to read | Retain explicit status while removing strike-through |
| V19 | Metric cards | Four narrow cards wrap labels excessively | Two-column grid and larger numeric readouts |
| V20 | Low-contrast supporting text | Important qualifications are visually faint | Raise the muted-text contrast |
| V21 | Reveal animation | Sections are invisible until JavaScript changes opacity | Content is visible by default, including without animation execution |
| V22 | Mobile navigation | Single desktop-height row competes with article width | Separate identity and horizontally scrollable page links into two rows |
| V23 | Wide mobile tables | Risk of horizontal overflow beyond the page | Contain scrolling within focusable, labeled table regions |
| V24 | Keyboard navigation | No explicit skip path or uniform visible focus | Add a skip link and focus outlines |
| V25 | Dash bullet markers | Every list visually repeats an em dash | Restore ordinary list markers |
| V26 | Results terminology | Undefined abbreviations increase paragraph rereading | Add a compact glossary with normal definition-list semantics |
| V27 | Reference empty state | “Nothing matches that” is vague and wordy | “No matching entries. Search by function name, module, or a broader term.” |
| V29 | Results external scripts | SNN simulation, source viewers, and syntax highlighting are unused on this page | Load only the shared navigation script |
| V30 | Long inline identifiers and demo labels | Code spans and fixed-width chart labels overflow narrow pages | Wrap inline identifiers, allow grid items to shrink, and stack chart labels on mobile |
| V28 | Interactive hardware warnings | Dramatic text and unconditional-firing claims exceed what the check establishes | State the violated range and required pre-clamp check |

## Redundant, irrelevant, or misplaced information

| ID | Instance | Recommendation / outcome |
|---|---|---|
| R01 | Results intro promises “the numbers around it” and a list of unsupported claims | Remove metacommentary; the headings provide navigation |
| R02 | Results “Recording them is the point of this section” | Remove the sentence; describe what was tested |
| R03 | Results “Leif’s second suggestion” | Remove personal attribution with no explanatory value; retain the experimental question |
| R04 | Results repeated “best…still first…both axes” statements | One ranking observation and one uncertainty subsection |
| R05 | Pooling accuracy and SynOps values repeated across adjacent boxes | Consolidate explanation under the comparison and metric interpretation |
| R06 | Within-run uncertainty repeated as “largest known gap,” “floor,” and “honest form” | Retain one visible limitation and one final next-step explanation without repeated slogans |
| R07 | “The fix that matters more than the bug” | Preserve the assertion’s behavior, remove editorial ranking of lessons |
| R08 | Detailed search ranges in the middle of results | Retain as optional detail rather than delete reproducibility information |
| R09 | Analysis correction history | Keep the conclusion and relevant count discrepancy; compress the debugging chronology |
| R10 | Three-phase declarations of why the project exists on index | Replace with the actual package behavior |
| R11 | Index promise to explain flushing later after already explaining it | Keep a short definition and direct section reference |
| R12 | Index contrast between two event channels and three image channels | Remove; it introduces an irrelevant comparison |
| R13 | Hardware extended imagined weight journey | Replace with the retained float parameter and computed quantized value |
| R14 | Hardware “two jobs” for weight decay repeated after folding explanation | One statement of regularization and folded-value clipping |
| R15 | Reference explanations repeat disproved overview claims | Align reference descriptions with the revised caveats |
| R16 | Old 65-trial narrowing notes versus current per-layer search | Keep historical context explicitly historical; verify current configuration before treating those ranges as current |

## Terms to define at first use

| Page / location | Terms | Suggested definition / treatment |
|---|---|---|
| Results summary | SynOps | Estimated spike-triggered additions per clip; defined at first prominent use |
| Results protocol | trial, epoch, checkpoint | One sampled training run; one pass through training data; saved state or scheduled comparison |
| Results protocol | validation, test, held out | Selection data versus evaluation data separated from training; clarify participant overlap and reuse |
| Results tables | fraction, percentage point, M | 0.8333 = 83.33%; subtract percentages for point differences; M = million |
| Results tables | MAC, ANN, dense reference | Multiply-accumulate; conventional artificial neural network; same-shape comparison denominator |
| Results search ranges | convolution block, kernel, channel | Filter plus neighboring operations; spatial filter window; feature-map count |
| Results search ranges | stride, pooling, downsampling | Spacing between convolution positions; combining outputs; reducing spatial dimensions |
| Results search ranges | head, flatten, GAP | Classifier portion; vector preserving positions; global average pooling over each channel |
| Results search ranges | fan-in, fan-out | Incoming and outgoing connections at one neuron |
| Results search ranges | τ, θ, membrane | Leak constant, firing threshold, neuron's accumulated state |
| Results training | optimizer, learning rate, schedule | Update algorithm, update scale, rule for changing that scale |
| Results training | Adam, AdamW | Adaptive optimizers; AdamW applies weight decay separately from the gradient update. Link training explanation |
| Results training | weight decay, dropout, smoothing, clipping | Weight penalty/shrinkage; suppressed activations; softened targets; capped gradient magnitude |
| Results conversion | INT16, quantization, folding | Signed 16-bit format; finite numeric grid; absorption of normalization into neighboring parameters |
| Results experiment selection | Pareto | Candidates where one objective cannot improve without another worsening |
| Results negative results | grokking | Delayed improvement on unseen data after fitting training data |
| Results early stopping | ASHA, pruning, rung | Successive-halving scheduler; stopped training; scheduled comparison checkpoint |
| Results statistics | p-value, q, multiple comparisons | Null-model tail probability; multiplicity-adjusted value as used here; adjustment across the tested settings |
| Results statistics | ρ / Spearman correlation | Rank association from −1 to +1 |
| Results statistics | permutation test | Compare observed statistic against values from reshuffled labels under stated assumptions |
| Results statistics | selection optimism | Upward bias introduced by choosing the largest noisy score |
| Results statistics | confounding, adaptive sampling | Settings changing together; future trials depend on earlier results |
| Results uncertainty | seed, paired prediction, confidence interval | Randomization control; same-clip predictions from both models; uncertainty procedure distinct from observed range |
| Index | neuromorphic, event camera, polarity | Event-driven neuron hardware; sensor reporting brightness changes; positive/negative change channels |
| Neurons | LIF, reset, leak, integrate | Stateful neuron with decay, threshold firing, reset, and accumulated input; existing numbered steps retained |
| Neurons | surrogate gradient, forward/backward pass | Approximate derivative for the backward pass while keeping binary spikes forward |
| Architecture | dataclass, plan, receptive field, dilation | Named Python fields; derived layer dimensions; input region affecting an output; spaced filter positions |
| Hardware | BN / tdBN, fold, bias, affine | Batch normalization / temporal variant; parameter combination; additive offset; scale plus offset |
| Hardware | QAT, PTQ, STE, raw parameter | Quantization-aware training; post-training quantization; straight-through estimator; optimizer's float value |
| Training | tensor axes N,T,C,H,W, batch, loss, cross-entropy | Batch, time, channels, height, width; group of examples; training objective; classification loss |
| Training | coding, passthrough, binarization, flush | Input representation over time; preserve event-frame timing; map to 0/1; zero-input steps for delayed output |
| Search | Optuna, Ray, ASHA | Configuration proposal, execution, and early-stopping components; expanded in section 6.1 |
| Search | conditional space, pickling, JSONL, atomic write, fsync | Settings sampled only when relevant; serialization; one JSON record per line; complete replacement; flush to storage |
| Reference | signature, internal helper, state dictionary | Function calling form; implementation-only function; saved mapping of model parameters |

The results glossary implements the core definitions above. Mathematical definitions already explained in earlier pages are retained. Specialized software terms can remain linked to those explanations rather than being redefined in every table cell.

## Evidence still needed

These are verification tasks, not claims that the style revision has completed:

1. Current 400-trial run manifest and exact configuration IDs for all five retrainings.
2. Per-clip predictions, split files, participant assignments, seeds, and test-evaluation chronology.
3. Statistical output underlying 0.0073, 96.6%, 0.0176, 0.0092, and the corrected q-values.
4. Exact GAP trial counts resolving “all 30” versus 97%.
5. Repeated-seed results for accuracy uncertainty and the three-clip decay difference.
6. Matched channel-count ablation for pooling versus stride.
7. Hardware execution, latency, and energy measurements. Exported-model evaluation cannot substitute for these.
8. Updated range documentation separating older 65-trial settings, the local 82-trial analysis, and the current 400-trial space.
9. Full provenance for the historical hardware-probe numbers and approximate runtime estimates.
10. Confirm reference signatures/defaults against the current package before treating this explanatory index as an exhaustive API reference.

## Itemized language changes

Each numbered entry gives the reviewed text, the issue, and the implemented example. Some short labels were edited in a second pass, so the reviewed text may be an intermediate wording. The same underlying passage can therefore have more than one entry. Links target the revised file. Original source-line references are included when the exact passage could be located in the initial snapshot.

### L001 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:169)

Original line 20.

**Reviewed:** A 400-trial search picked an architecture, five candidates were retrained long and scored once on data nothing had selected against, and the best reached 83.3% test accuracy against the 68.75% the HiAER-Spike paper reports for its best DVS128 model. This page gives that number, the numbers around it, the bug that voided the previous version of this page, and an explicit list of what the data does not support.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** A search of 400 configurations selected five candidates for longer training. Their reported test accuracies range from 78.8% to 83.3% on DVS128 Gesture. The sections below describe the corrected search, comparisons between candidates, and experiments needed to resolve the remaining uncertainty.

### L002 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:62)

Original line 61.

**Reviewed:** Three sets exist. Training fits the weights. Validation, 176 clips carved at random out of the training set, chooses the epoch and chooses the winning configuration. Test, 288 clips from six people who appear nowhere in training, is evaluated exactly once at the end.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Training data fits the weights. A validation set of 176 clips, sampled from the training partition, selects configurations and checkpoints. The test set contains 288 clips from six participants excluded from training.

### L003 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:63)

Original line 66.

**Reviewed:** Validation cannot be compared against published work for two reasons. Its clips come from people the network has already seen, which makes it an easier exam than test. And it was optimized against four hundred times, so the highest value it produced is inflated by whichever configuration got the luckiest draw on those particular 176 clips.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Validation clips can share participants with training clips. Repeatedly selecting configurations against the same validation set can also favor those that happen to perform well on that set. Use the test scores for the external comparison, with the protocol limitations noted below.

### L004 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:64)

Original line 72.

**Reviewed:** Validation accuracy for the winning run was 0.9205. That number appears on this page only as bookkeeping. It is not the result.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The leading candidate reached 92.05% validation accuracy and 83.33% test accuracy. Each of the five retrained candidates was evaluated on test. Section 7.5 also reports test results for a weight-decay sweep, so the page does not describe a single test evaluation across the entire study.

### L005 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:89)

Original line 81.

**Reviewed:** An earlier search reported 0.8977 validation accuracy. Replaying that configuration produced 0.7500. Two replays agreed with each other and disagreed with the trial from the first epoch, which rules out ordinary run-to-run variation.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The earlier search recorded 89.77% validation accuracy, but replaying its saved configuration produced 75.00%. Two replays agreed and diverged from the original trial from the first epoch. This prompted a check of the recorded settings against the actual training inputs.

### L006 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:92)

Original line 90.

**Reviewed:** Each trial sampled an input resolution and a frame count, wrote them to the leaderboard, and then built its dataloaders from the config file instead. Every trial trained at 64×64 with 16 frames regardless of what the sampler chose. The winner's record said 32×32 with 8 frames, and the replay honoured the record, so it trained a smaller and weaker network.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The search recorded sampled input dimensions and frame counts, but built its data loaders from the configuration file. Every trial therefore used 64×64 inputs and 16 frames. The saved candidate specified 32×32 and 8 frames, which the replay used. The two runs evaluated different input configurations.

### L007 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:95)

Original line 98.

**Reviewed:** Three consequences, all of which invalidate previously reported numbers.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The mismatch invalidated the configuration claims below. A separate counting error invalidated the reported operation count.

### L008 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:111)

Original line 114.

**Reviewed:** Correcting the pipeline was mechanical. The useful change is an assertion that runs once per trial and kills it if the configuration being recorded disagrees with the configuration being trained. Dying there costs one trial. Not checking cost four hundred.

**Fix:** Remove personification or dramatic framing. Name the operation, observation, and consequence.

**Revised example:** An assertion now checks that the recorded input settings match those used for training. A mismatch stops the affected trial before its score can enter the search results.

### L009 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:1)

Original line 119.

**Reviewed:** The same failure shape appeared three more times while auditing: a scheduler counter that read an absent field as a value and reported zero pruned trials out of 207, a weight-clipping metric that read the tensor after the clamp and could only ever return zero, and a feasibility table that read a blank column as a rejection and reported 100% for every setting of every knob. A field that is missing is not a field that is false.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The audit also found metrics that treated missing values as measured outcomes and a clipping metric that inspected weights after clamping. Missing data must remain marked as unavailable, and clipping must be measured before the clamp.

### L010 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:118)

Original line 131.

**Reviewed:** 400 trials over a per-layer space, ranked on the accuracy of the exported INT16 network rather than the floating-point one, with early stopping at epochs 10, 30 and 60.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The corrected search sampled layer settings independently. Completed trials were ranked by validation accuracy after export to signed 16-bit integer (INT16) values. The early-stopping scheduler compared intermediate scores at epochs 10 and 30, with a maximum training budget of 60 epochs.

### L011 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:152)

Original line 167.

**Reviewed:** Per-layer geometry mattered immediately. The winning network uses a 3×3 kernel on the first block and 5×5 on the second, with 81 channels then 56. A space that shares one kernel and one channel count across all layers cannot express that, and the previous search could not.

**Fix:** Describe the observed configuration comparison without assigning causation to one setting or treating early stopping as a final outcome.

**Revised example:** The leading candidate uses a 3×3 kernel with 81 channels in its first block and a 5×5 kernel with 56 channels in its second. Allowing different settings in each layer makes this configuration possible. Its result alone does not establish that this search space outperforms a uniform one.

### L012 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:157)

Original line 176.

**Reviewed:** The search ranks on 60-epoch validation. Those five were then retrained for 300 epochs at weight decay 1e-3 and evaluated once on the held-out test set. 107 minutes total.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The five candidates with the highest validation scores after the 60-epoch search were retrained for 300 epochs with weight decay set to 0.001. Total retraining time was reported as 107 minutes. Rows below are ordered by test accuracy, while the first column retains the original search rank.

### L013 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:1)

Original line 225.

**Reviewed:** The search's first choice is still first after 300 epochs on held-out data.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The candidate ranked first during search also had the highest test accuracy after retraining.

### L014 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:203)

Original line 226.

**Reviewed:** Ranks 2 through 5 reshuffled among themselves, which is expected when they sit inside the noise floor of 1.3 clips. Rank 1 stayed on top on both axes at once.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** The other four candidates changed order. These are five different configurations, each evaluated in one retraining run. They do not measure how consistently any one configuration performs across random seeds.

### L015 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:1)

Original line 234.

**Reviewed:** Rank 1 is not statistically separated from the rest.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** The observed ranking does not establish a statistically reliable difference between candidates.

### L016 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:207)

Original line 235.

**Reviewed:** 0.8333 against 0.8056 is eight clips out of 288, roughly 0.9 standard errors. Report it as the best of five, not as significantly better than the others. The defensible claim is the range: this method produces 0.79 to 0.83 on test.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** The gap between ranks 1 and 3 is eight correct predictions out of 288 clips (2.78 percentage points). Establishing a difference requires paired predictions on the same clips and repeated training runs. The observed 78.8% to 83.3% range describes these five candidates, not a confidence interval for the method.

### L017 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:210)

Original line 242.

**Reviewed:** All five converged on nearly the same network

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Shared architecture settings

### L018 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:211)

Original line 243.

**Reviewed:** Every one of the five is two convolution blocks, 3×3 then 5×5, at 48×48 input with 16 frames, no hidden fully-connected layer, and average pooling on the first block. The only structural difference in the whole set is what happens on the second block.

**Fix:** Describe the observed configuration comparison without assigning causation to one setting or treating early stopping as a final outcome.

**Revised example:** All five candidates use two convolution blocks, 3×3 then 5×5 kernels, 48×48 inputs, 16 frames, no hidden fully connected layer, and average pooling after the first block. They differ in channel counts and in how the second block reduces spatial resolution.

### L019 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:169)

Original line 251.

**Reviewed:** Pooling the second block instead of striding it buys 3 to 4.5 points of test accuracy, and costs 2.5 times the synaptic operations.

**Fix:** Describe the observed configuration comparison without assigning causation to one setting or treating early stopping as a final outcome.

**Revised example:** The candidate with second-block average pooling had the highest accuracy and operation count.

### L020 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:214)

Original line 252.

**Reviewed:** Rank 1 is the only network of the five that pools on the second block. It has the highest test accuracy, and it also does the most work: 50.1M synaptic operations against 18 to 23M for the other four.

**Fix:** Describe the observed configuration comparison without assigning causation to one setting or treating early stopping as a final outcome.

**Revised example:** Rank 1 uses average pooling after the second convolution and reports 83.33% test accuracy at 50.1 million synaptic operations per clip. The four candidates using stride-2 convolution report 78.82% to 80.56% at 18.5 to 22.6 million operations.

### L021 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:215)

Original line 257.

**Reviewed:** Stride-2 skips positions before the convolution ever computes them, which is why the striding networks are so much cheaper. Average pooling computes every position and then combines them, which preserves evidence and costs arithmetic. The four striding networks land within 1.7 points of each other on test regardless of their channel counts, so the second-block choice is doing the work rather than the width.

**Fix:** Describe the observed configuration comparison without assigning causation to one setting or treating early stopping as a final outcome.

**Revised example:** Stride-2 convolution evaluates fewer spatial positions. Average pooling combines outputs after the convolution has evaluated those positions, which increases computation. Because channel counts also differ between these candidates, a matched experiment is needed to isolate the effect of pooling.

### L022 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:1)

Original line 268.

**Reviewed:** The two SynOps columns answer different questions and only one of them is about energy.

**Fix:** Separate counted operations from measured energy. State the metric actually observed.

**Revised example:** Operation count and the ratio to a dense network describe different comparisons.

### L023 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:219)

Original line 269.

**Reviewed:** SynOps per sample is what the chip spends. On that measure rank 1 is the most expensive of the five, not the cheapest.

**Fix:** Separate counted operations from measured energy. State the metric actually observed.

**Revised example:** Synaptic operations (SynOps) estimate spike-triggered additions to downstream neurons per input clip. They are a measure of computational activity. Energy and latency were not measured for these candidates.

### L024 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:220)

Original line 273.

**Reviewed:** The ratio against dense asks a different question: for this particular architecture, how much does spiking sparsity save compared with running the same shape as a conventional network? Rank 1 scores 0.73× there because pooling keeps a large feature map that fires sparsely. The striding networks are small and dense, so spiking saves them nothing and they land above 1×.

**Fix:** Describe the observed configuration comparison without assigning causation to one setting or treating early stopping as a final outcome.

**Revised example:** The dense ratio divides each candidate’s SynOps by the multiply-accumulate count for a conventional artificial neural network (ANN) of the same shape. Rank 1 has a ratio of 0.73, so its counted operations are 27% fewer than its own dense reference.

### L025 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:221)

Original line 280.

**Reviewed:** The ratio cannot be compared across architectures, because each row is measured against a different denominator. Rank 1's dense equivalent is 69.0M operations; rank 4's is 12.5M.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Each ratio uses a different reference: 69.0 million dense operations for rank 1 and 12.5 million for rank 4. A lower ratio does not imply fewer absolute operations across architectures. Additions and multiply-accumulates also have different energy costs.

### L026 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:317)

Original line 288.

**Reviewed:** Nobody has decided what 3.5 points of accuracy is worth in synaptic operations.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Selecting a candidate requires an accuracy target or an operation budget.

### L027 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:225)

Original line 289.

**Reviewed:** Rank 1 at 0.8333 and 50.1M, or rank 3 at 0.8056 and 20.3M, is a real choice and this project has no stated budget to resolve it with. The pareto objective exists in the search for exactly this and has never been run.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Rank 3 uses 20.3 million SynOps at 80.56% accuracy, compared with rank 1’s 50.1 million at 83.33%. The project has not set a budget that determines which is preferable. Its untested pareto search objective would retain configurations for which improving one objective requires worsening another.

### L028 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:231)

Original line 299.

**Reviewed:** One architecture, rank 1, trained three times for 300 epochs with nothing changed but the weight decay. This is the only controlled single-variable experiment in the project.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** To examine weight decay, the rank-1 architecture was trained for 300 epochs at each of three values, with other settings reported as unchanged. Weight decay discourages large weights. Each setting has one run, so the comparison does not estimate variation across random seeds.

### L029 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:169)

Original line 332.

**Reviewed:** At the sampled weight decay this spiking network does more arithmetic than an equivalent dense ANN. At 1e-3 it does less, and scores higher.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Increasing weight decay from 0.000094 to 0.001 reduced SynOps by 37.5% in these runs.

### L030 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:261)

Original line 333.

**Reviewed:** 80.1M accumulates against a dense network's 69.0M multiply-accumulates is a ratio of 1.16. An accumulate is cheaper than a multiply-accumulate, so this is not fatal, but an operation-count advantage cannot be claimed while performing 16% more operations.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** SynOps fell from 80.1 million to 50.1 million per clip, while test accuracy increased from 82.29% to 83.33%, a difference of three correctly classified clips. The operation-count change is substantial. The accuracy difference needs repeated runs before it can be attributed reliably to weight decay.

### L031 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:262)

Original line 338.

**Reviewed:** Raising decay to 1e-3 crosses that line to 0.73× and raises test accuracy at the same time. Nothing in the loss asked for fewer spikes. Smaller weights mean smaller input currents, which means membranes cross threshold less often, so the firing rate falls as a side effect.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Spike counts also fell from 263,034 to 141,940 per clip. Smaller weights can reduce input currents and threshold crossings, which is consistent with this observation. The sweep did not directly isolate that mechanism.

### L032 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:169)

Original line 347.

**Reviewed:** Past 1e-3 the trade stops being free.

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** At weight decay 0.01, activity fell further and accuracy declined.

### L033 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:266)

Original line 348.

**Reviewed:** At 1e-2 activity keeps falling and training accuracy collapses to 0.80, taking test down to 0.6840, which lands exactly on the published baseline. The optimum is bracketed on both sides rather than assumed.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Training accuracy fell to 80.0% and test accuracy to 68.40%, compared with the published reference of 68.75%. Of the three settings tested, 0.001 had the highest test accuracy. Three values do not locate the optimum.

### L034 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:270)

Original line 355.

**Reviewed:** Two independent measurements agree here. This controlled sweep says more decay helps, and the observational analysis over 111 scored trials found weight decay correlated with accuracy at ρ = +0.227, q = 0.036. An experiment and a correlation pointing the same way is worth more than either alone.

**Fix:** Shorten a dense paragraph, remove repeated framing, and put the observation before its interpretation.

**Revised example:** Across the 111 scored search trials, weight decay had a positive rank correlation with validation accuracy (ρ = 0.227, adjusted p-value = 0.0364). This association and the sweep concern the same project and are not independent replications. The benefit also did not extend to the largest decay value tested.

### L035 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:275)

Original line 365.

**Reviewed:** Four things were tried on the strength of a suggestion or a hypothesis and did not deliver. Recording them is the point of this section.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** These experiments distinguish an observed lack of improvement from insufficient sampling and an implementation error.

### L036 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:175)

Original line 372.

**Reviewed:** Grokking does not occur within 300 epochs, at any of three weight decays.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** No delayed improvement in validation accuracy was observed within 300 epochs.

### L037 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:278)

Original line 373.

**Reviewed:** Training accuracy saturates around epoch 80 and validation is flat from there to 300. At the best decay, 216 epochs of the run, 72% of it, bought six tenths of a percentage point.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Training accuracy plateaued near epoch 80. At the best tested decay, the final 216 epochs improved validation accuracy by about 0.6 percentage points.

### L038 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:279)

Original line 377.

**Reviewed:** Stated as a bound rather than a refutation. 300 epochs is roughly three times past memorization, and the effect is normally reported at a hundred times or more. Testing it properly would need 3,000 epochs, no label smoothing, and no quantization fold partway through to interrupt the dynamics.

**Fix:** Shorten a dense paragraph, remove repeated framing, and put the observation before its interpretation.

**Revised example:** Grokking describes a delayed improvement on unseen data after a model has already fitted its training data. These runs did not show that pattern within their training budget. A longer run would test whether improvement occurs later, but changing label smoothing or quantization at the same time would confound that comparison.

### L039 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:169)

Original line 387.

**Reviewed:** Global average pooling is far more deployable and still worse.

**Fix:** Describe the observed configuration comparison without assigning causation to one setting or treating early stopping as a final outcome.

**Revised example:** Global average pooling reduced connection-limit rejections but had lower early accuracy.

### L040 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:283)

Original line 388.

**Reviewed:** Replacing the flatten with global average pooling makes the classifier's fan-in equal to the channel count at any resolution, so its rejection rate is 6% against flatten's 22%. It solves the constraint that keeps high resolution out of the space.

**Fix:** Describe the observed configuration comparison without assigning causation to one setting or treating early stopping as a final outcome.

**Revised example:** A flatten head retains each spatial position as a separate input to the classifier. Global average pooling (GAP) averages each channel over space, reducing classifier fan-in to the channel count. Reported rejection rates were 6% for GAP and 22% for flatten heads.

### L041 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:345)

Original line 393.

**Reviewed:** It also loses. All 30 of its trials died at the first checkpoint. Comparing only trials cut at that same checkpoint, so they have had identical training, GAP sat at 0.296 accuracy against flatten's 0.651 and was climbing more slowly, +0.0043 per epoch against +0.0063. Behind and falling further behind is not a slow starter.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** At the first stopping checkpoint, the comparison reports 29.6% accuracy for GAP and 65.1% for flatten heads, with respective gains of 0.43 and 0.63 percentage points per epoch. GAP was weaker at this checkpoint. Since those runs stopped early, their eventual accuracy remains unknown. The original report’s “all 30” wording conflicts with its 97% stopping rate, so an exact count needs verification.

### L042 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:1)

Original line 403.

**Reviewed:** Spatiotemporal dropout in the convolution stack was never given enough trials to judge.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The search provides insufficient evidence about convolutional dropout.

### L043 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:288)

Original line 404.

**Reviewed:** 12 scored trials, p = 0.50. With 12 trials nothing smaller than about three clips is detectable. The sampler drew it 44 times out of 318 rather than the expected 160, having seen it marginally behind at the first checkpoint and moved on.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Spatiotemporal dropout suppresses selected convolutional activations during training to reduce reliance on them. It appeared in 44 of 318 trials that passed the hardware checks, and only 12 reached a final score. The reported p-value of 0.50 does not establish either a benefit or no effect.

### L044 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:289)

Original line 409.

**Reviewed:** Its rung-1 death rate was 43% against 47% for trials without it, so the scheduler was not singling it out. It is simply undersampled. It is now forced on for every trial in the next search.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** At the first checkpoint, 43% of trials with dropout stopped, compared with 47% without it. This does not suggest a large difference in early stopping. A balanced comparison with dropout enabled and disabled is more informative than comparing a new forced-on search with this adaptive search.

### L045 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:1)

Original line 418.

**Reviewed:** The firing-rate penalty was penalizing the output layer.

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** The firing-rate penalty included the output layer.

### L046 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:293)

Original line 419.

**Reviewed:** The probe hooked every spiking layer including the classifier, whose spike counts divided by T are the prediction that cross-entropy scores. So the penalty was instructing the network to stop answering, and the two objectives fought over the same tensor.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The penalty discouraged spikes in every spiking layer, including the classifier. Since output spike rates are also the class scores used by the classification loss, this added an objective that could suppress the prediction signal.

### L047 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:294)

Original line 424.

**Reviewed:** Measured cost: 92% of trials with the penalty died at the first checkpoint against 43% without, at 0.332 accuracy against 0.626. The output layer is now excluded and the fix is not yet verified by a run.

**Fix:** Remove personification or dramatic framing. Name the operation, observation, and consequence.

**Revised example:** Trials with the penalty stopped at the first checkpoint at a rate of 92%, compared with 43% without it. Their reported early accuracies were 33.2% and 62.6%. The output layer has since been excluded, but the effect of that fix still requires an experiment.

### L048 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:300)

Original line 434.

**Reviewed:** Computed from 111 scored trials and 1,549 late-training epochs, using permutation tests rather than parametric ones because the sample is small and the distribution is not normal.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The analysis reports estimates from 111 scored trials and 1,549 late-training epochs. It uses permutation tests, which compare the observed statistic with statistics obtained by reshuffling labels. The estimates below describe this analysis, not uncertainty across repeated training runs.

### L049 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:312)

Original line 446.

**Reviewed:** Knobs that survived correction

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Associations after multiple-comparison adjustment

### L050 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:313)

Original line 447.

**Reviewed:** q is the false-discovery-corrected p-value across all thirteen knobs tested together, so it already accounts for testing many at once.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** q denotes the p-value adjusted for false-discovery rate across 13 tested settings. A p-value measures how unusual the observed statistic is under the test’s null model. Adjustment addresses multiple comparisons, but cannot remove confounding or selection effects. ρ (Spearman’s rho) measures rank association from −1 to +1.

### L051 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:331)

Original line 468.

**Reviewed:** Optuna does not sample randomly. Once it finds a promising region it draws more configurations from it, so the good trials resemble each other and several knobs moved together. In this run scheduler correlates with dropout_rate at −0.57, with tau at +0.46 and with weight_decay at +0.42.

**Fix:** Shorten a dense paragraph, remove repeated framing, and put the observation before its interpretation.

**Revised example:** Optuna adapts its sampling to earlier results, so promising settings appear together more often. In this run, the learning-rate schedule correlates with dropout at −0.57, leak constant at +0.46, and weight decay at +0.42.

### L052 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:332)

Original line 474.

**Reviewed:** When knobs co-vary, a one-at-a-time test credits each of them with the same shared change. The p-values are real; the attribution is not. Separating them needs an experiment that holds the others fixed, which is what section 7.5 does for weight decay and nothing yet does for the other five.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** When settings change together, separate tests cannot identify which setting caused a score difference. Adaptive sampling and early stopping also limit the interpretation of permutation p-values. Matched experiments are needed to test the individual effects.

### L053 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:333)

Original line 480.

**Reviewed:** A second caution applies to any knob the sampler converged on. Depth was 2 in 93% of trials, the head had no hidden layer in 95%, and 16 frames were used in 93%. Those are not findings about what is best; they are the sampler having stopped exploring. A column that is 93% one value cannot support an effect estimate whatever its q-value says.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** The sampled settings were unbalanced: 93% of trials used depth 2, 95% had no hidden head layer, and 93% used 16 frames. These frequencies describe the search’s allocation of trials. They provide limited evidence for comparison with rarely sampled alternatives.

### L054 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:338)

Original line 488.

**Reviewed:** The scheduler decides more than it appears to

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** What early stopping can establish

### L055 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:339)

Original line 489.

**Reviewed:** Early stopping cuts trials at epoch 10 based on where they are at epoch 10. Configurations that start slowly and finish well are indistinguishable, at that moment, from configurations that start slowly and stay bad. Comparing only trials cut at that same checkpoint, so every one has had identical training, separates the two:

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** To compare early performance, the analysis groups trials stopped at the same checkpoint. This avoids comparing scores after different training budgets. Early slopes can describe progress up to that checkpoint, but cannot determine how a stopped trial would perform after full training.

### L056 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:353)

Original line 508.

**Reviewed:** The first version of this comparison averaged slope across trials cut at different epochs, which measures when each was cut rather than how promising it was, because slope falls as training proceeds. GAP looked like a slow starter purely because every GAP trial died at the first checkpoint while flatten trials survived to the third. Restricting to a single checkpoint reversed the conclusion.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** An earlier comparison mixed trials stopped at different epochs. Restricting the analysis to the first checkpoint changed the interpretation of GAP’s early performance. The table above uses the matched checkpoint.

### L057 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:405)

Original line 566.

**Reviewed:** No configuration has been run twice under identical settings. Every uncertainty quoted here is measured from epoch-to-epoch wobble within a single run, which is a floor on the real uncertainty and probably well below it. Two trials are two separate runs, and training varies run to run by more than a nearly-frozen model's score does.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** The five candidate scores come from different architectures. The study reports no repeated final evaluations of one configuration across random seeds. Epoch-to-epoch score variation cannot substitute for that measurement or provide a calibrated confidence interval for a new run.

### L058 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:406)

Original line 572.

**Reviewed:** Until that measurement exists, the honest form of the headline is a range rather than a point: this method produces 0.79 to 0.83 test accuracy on DVS128 Gesture, against a published 0.6875.

**Fix:** Remove personification or dramatic framing. Name the operation, observation, and consequence.

**Revised example:** The current evidence is an observed range of 78.8% to 83.3% across five candidates. Repeated runs and a fully documented evaluation protocol are needed before making a broader performance claim.

### L059 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:61)

Original line 454.

**Reviewed:** Why the test number is the only one worth quoting

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Evaluation protocol

### L060 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:88)

Original line 80.

**Reviewed:** What this page used to say, and why it was wrong

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Reproducing the earlier result

### L061 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:110)

Original line 113.

**Reviewed:** The fix that matters more than the bug

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Validation added to the pipeline

### L062 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:156)

Original line 175.

**Reviewed:** The top five, retrained and scored on test

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Five candidates after retraining

### L063 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:230)

Original line 298.

**Reviewed:** Weight decay buys accuracy and energy at the same time

**Fix:** Separate counted operations from measured energy. State the metric actually observed.

**Revised example:** Weight decay: accuracy and activity

### L064 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:274)

Original line 364.

**Reviewed:** What did not work

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Follow-up experiments

### L065 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:299)

Original line 433.

**Reviewed:** What the statistics license

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** Statistical analysis and limitations

### L066 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:330)

Original line 467.

**Reviewed:** Why five of those six are weaker than their q-value suggests

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** Confounding and adaptive sampling

### L067 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:404)

Original line 565.

**Reviewed:** The largest known gap

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Repeated runs are still needed

### L068 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:357)

Original line 454.

**Reviewed:** Directions

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Next experiments

### L069 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:131)

Original line 147.

**Reviewed:** The space

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Search parameters

### L070 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:202)

Original line 224.

**Reviewed:** Supported

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Observation

### L071 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:206)

Original line 233.

**Reviewed:** Not supported

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Limitation

### L072 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:218)

Original line 267.

**Reviewed:** Read carefully

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Interpretation

### L073 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:265)

Original line 346.

**Reviewed:** Bounded above

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Higher decay

### L074 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:282)

Original line 386.

**Reviewed:** Loses on merit

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Early performance

### L075 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:277)

Original line 371.

**Reviewed:** No effect

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Within this training budget

### L076 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:287)

Original line 402.

**Reviewed:** Unmeasured

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Insufficient evidence

### L077 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:122)

Original line 138.

**Reviewed:** Fate

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Outcome

### L078 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:134)

Original line 150.

**Reviewed:** Knob

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Parameter

### L079 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:125)

Original line 141.

**Reviewed:** Cut by the scheduler

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Stopped early

### L080 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:126)

Original line 142.

**Reviewed:** Scored

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Completed and scored

### L081 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:1)

Original line 140.

**Reviewed:** Arithmetic, no GPU time spent.

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Rejected by the pre-training feasibility check.

### L082 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:125)

Original line 141.

**Reviewed:** Trained, then stopped early for being behind at a checkpoint.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Below the scheduler’s threshold at a checkpoint.

### L083 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:126)

Original line 142.

**Reviewed:** Ran to 60 epochs, converted, and produced a hardware number.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Completed 60 epochs and evaluated the exported model.

### L084 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:163)

Original line 184.

**Reviewed:** Rank

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Search rank

### L085 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:163)

Original line 184.

**Reviewed:** Search val

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Search validation

### L086 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:164)

Original line 185.

**Reviewed:** Val, 300 ep

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Validation, 300 epochs

### L087 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:72)

Original line 186.

**Reviewed:** vs its own dense

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** SynOps / dense MACs

### L088 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:72)

Original line 308.

**Reviewed:** vs dense ANN

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** SynOps / dense MACs

### L089 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:237)

Original line 307.

**Reviewed:** Train acc

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Training accuracy

### L090 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:237)

Original line 307.

**Reviewed:** Test acc

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Test accuracy

### L091 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:343)

Original line 498.

**Reviewed:** Died at first cut

**Fix:** Remove personification or dramatic framing. Name the operation, observation, and consequence.

**Revised example:** Stopped at first checkpoint

### L092 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:343)

Original line 498.

**Reviewed:** Slope / epoch

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Accuracy gain / epoch

### L093 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:218)

Original line 498.

**Reviewed:** Verdict

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Interpretation

### L094 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:346)

Original line 501.

**Reviewed:** behind and slower, cut correctly

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Lower accuracy and slower early improvement

### L095 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:347)

Original line 502.

**Reviewed:** not singled out, just rare

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Similar stopping rate, few completed trials

### L096 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:348)

Original line 503.

**Reviewed:** crippled by the bug in 7.6

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Output-layer penalty included (section 7.6)

### L097 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:324)

Original line 461.

**Reviewed:** confirmed by controlled sweep, 7.5

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Three-value sweep in section 7.5; no repeated seeds

### L098 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:1)

Original line 461.

**Reviewed:** more is better

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Positive association in this search

### L099 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:1)

Original line 458.

**Reviewed:** less is better

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Negative association in this search

### L100 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:323)

Original line 460.

**Reviewed:** co-varies

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** varies together

### L101 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:32)

Original line 307.

**Reviewed:** Test accuracy, best

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Highest test accuracy

### L102 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:37)

Original line 307.

**Reviewed:** Test accuracy, mean of 5

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Mean across five candidates

### L103 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:181)

Original line 214.

**Reviewed:** 288 held-out clips from six people the network never trained on. Measured once.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** 240 of 288 test clips classified correctly by the leading candidate.

### L104 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:175)

Original line 196.

**Reviewed:** Five different architectures from the same search, each retrained to 300 epochs.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Different architectures, each retrained for 300 epochs. This is not a mean across random seeds.

### L105 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:44)

Original line 190.

**Reviewed:** The HiAER-Spike paper's best DVS128 Gesture result, same task, same test set.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Highest DVS Gesture accuracy reported in the HiAER-Spike paper. See the protocol note below.

### L106 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:345)

Original line 190.

**Reviewed:** About four standard errors on 288 clips. The worst of the five still leads by 10.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** Percentage-point difference from the published reference; not a significance estimate.

### L107 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:175)

Original line 192.

**Reviewed:** 0.8333

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** 83.33%

### L108 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:181)

Original line 323.

**Reviewed:** 0.8049

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** 80.49%

### L109 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:187)

Original line 208.

**Reviewed:** 0.6875

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** 68.75%

### L110 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:305)

Original line 440.

**Reviewed:** Noise floor

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** Within-run score variation

### L111 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:305)

Original line 440.

**Reviewed:** 1.3 clips. Scores closer than 0.015 are indistinguishable.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** 0.73 percentage points on validation, about 1.3 clips. Not a significance threshold.

### L112 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:306)

Original line 441.

**Reviewed:** Signal share

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** Estimated between-trial share

### L113 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:306)

Original line 441.

**Reviewed:** Of the spread between trials, this much is real difference rather than noise.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Model-based decomposition of observed score variance; does not isolate causal effects.

### L114 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:307)

Original line 442.

**Reviewed:** How much the best of 111 noisy scores overstates. 3.1 clips.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Estimated selection optimism: 1.76 percentage points. Not a correction to the test score.

### L115 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:308)

Original line 443.

**Reviewed:** At 10 trials per group. Smaller effects need more trials, not better statistics.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** Estimated under the analysis assumptions for 10 trials per group. Not established for repeated runs.

### L116 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:361)

Original line 522.

**Reviewed:** Cost

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Estimated runtime

### L117 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:361)

Original line 522.

**Reviewed:** Decides

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Purpose

### L118 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:367)

Original line 528.

**Reviewed:** Whether the next search can require it. Epoch 9 near 0.6 rather than 0.33.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Compare accuracy and hidden-layer activity with the penalty disabled.

### L119 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:371)

Original line 532.

**Reviewed:** Rerun the search with it forced on for every trial

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Matched runs with dropout on and off, holding other settings fixed

### L120 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:373)

Original line 534.

**Reviewed:** Leif's second suggestion, currently unmeasured at 12 trials.

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Estimate the effect of dropout without adaptive sampling imbalance.

### L121 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:376)

Original line 537.

**Reviewed:** Is pool-pool really better?

**Fix:** Describe the observed configuration comparison without assigning causation to one setting or treating early stopping as a final outcome.

**Revised example:** Does second-block pooling improve accuracy?

### L122 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:379)

Original line 540.

**Reviewed:** Turns an accidental four-case comparison into a controlled one.

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Isolate the second-block operation while retaining the same channel counts.

### L123 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:385)

Original line 546.

**Reviewed:** Locates the accuracy-versus-SynOps frontier instead of three points on it.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Refine the tested decay range, then repeat promising settings across seeds.

### L124 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:389)

Original line 550.

**Reviewed:** Five runs of one configuration, identical settings

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Five runs of one configuration with different random seeds

### L125 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:391)

Original line 552.

**Reviewed:** Every uncertainty on this page is a floor until this exists.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** Estimate variation attributable to training randomness.

### L126 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:394)

Original line 555.

**Reviewed:** Does grokking exist here at all?

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Does longer training improve generalization?

### L127 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:395)

Original line 556.

**Reviewed:** 3,000 epochs, no label smoothing, no quantization fold

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** 3,000 epochs with other settings fixed; test schedule changes separately

### L128 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:397)

Original line 558.

**Reviewed:** Whether the negative result in 7.6 is a bound or an answer.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Measure whether improvement occurs beyond the current 300-epoch budget.

### L129 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:21)

Original line 21.

**Reviewed:** This project is a Python program that finds the best performing configuration of a user-specificed spiking neural network under the constraints of HiAER-Spike, a neuromorphic platform. The platform limits how a network may be wired, supports only a narrow set of operations, and works in limited precision. These constraints drastically alter how a network is constructed, trained, and evaluated. The program is built for reuse. The network and the hardware limits are both supplied as configuration, so the user can point it at a new architecture, or raise the limits as HiAER-Spike develops, without rewriting the search. The immediate goal is to find parameters that increase accuracy for a specified network while the long-term goal is an analysis of which parameters are worth focusing on and expanding capabilities in.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** This Python package searches for spiking neural network (SNN) configurations that fit HiAER-Spike’s connection limits and numeric formats. Architecture and training settings are configurable. Each candidate is evaluated after conversion to the form intended for deployment.

### L130 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:38)

Original line 38.

**Reviewed:** The appeal of such a framework is power consumption. A conventional GPU evaluates every unit of a network on every input. This platform performs work only where an event actually arrives, so a workload with little activity costs little energy.

**Fix:** Separate counted operations from measured energy. State the metric actually observed.

**Revised example:** Event-driven computation can reduce work when spike activity is sparse. Actual energy use also depends on memory traffic, routing, and static power.

### L131 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:52)

Original line 56.

**Reviewed:** The platform defines networks in terms of indvidual neurons, int16 values, and messages passing between neuron. There are no broader categories such as fully connected and convolutional layers.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The platform represents individual neurons, signed 16-bit integer values, and connections between neurons. The converter expands convolutional and fully connected layers into this representation.

### L132 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:64)

Original line 70.

**Reviewed:** The converter performs this expansion. However, more relevant to us, our snnsearch/quantgrid.py mirrors its arithmetic exactly, so that a network is trained against the numbers the converter will produce rather than adjusted to them afterwards. This prevents massive losses in accuracy due to conversion and discourages dependence on high-precision weight values for storing information.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The converter expands the network and maps its values to the hardware format. snnsearch/quantgrid.py mirrors the numeric mapping so training can account for quantization. This reduces one source of conversion mismatch, but does not guarantee unchanged accuracy.

### L133 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:83)

Original line 93.

**Reviewed:** Each of the last three has limit on how large they can be, because each one counts rows in the platform's routing table, and that table is a fixed size. These connection/routing limits directly determine whether or a not a network can be constructed.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Axon count, fan-in, and fan-out have fixed hardware limits. A configuration that exceeds them cannot be represented by the platform’s routing tables.

### L134 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:236)

Original line 99.

**Reviewed:** The platform's memory holds 16-bit signed integers, so the values that exist on it discretely range from −32767 to +32767. Weights can possess values awkward values such as 0.7, which needs to be represented by one of such integer values. To do this, the converter defines w_alpha and fixes it at 1. The constant dictates that the largest integer (+32767) represents the value 1.0.

**Fix:** Shorten a dense paragraph, remove repeated framing, and put the observation before its interpretation.

**Revised example:** The converter uses a symmetric range from −32767 to +32767. With w_alpha = 1 , the integer 32767 represents the real value 1.0. A floating-point weight such as 0.7 is rounded to the nearest value on this grid.

### L135 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:93)

Original line 112.

**Reviewed:** Firing thresholds and the decay constant are mapped the same way.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Firing thresholds use the weight scale. The leak constant is rounded separately to an integer register value, as described in section 2.3.

### L136 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:146)

Original line 173.

**Reviewed:** Quantizing to int16 and the methodology of update order heavily impact the network and should be paid attention to.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Quantization changes representable values. The update order changes spike timing. Both need to match during training and evaluation.

### L137 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:157)

Original line 186.

**Reviewed:** A convolutional NN applies the same small filter across an image and stacks layers of these filters. The design suits a GPU, which multiplies large dense matrices quickly. HiAER-Spike cannot run it. Its neurons exchange binary events rather than continuous values, and they do work only where an event arrives, so there is no dense matrix for the hardware to multiply.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** A convolutional neural network applies shared filters across an image. GPUs evaluate these filters using dense arithmetic. For HiAER-Spike, the connections must be expanded explicitly and the network must use the supported spiking neuron update.

### L138 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:184)

Original line 218.

**Reviewed:** Because a single spike carries almost no information, an SNN is run for T timesteps and its answer is read from how often each output neuron fired. The simulator below runs the same update rule the chip uses.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The classifier accumulates output spikes over T timesteps and chooses the class with the highest count. The simulator below uses the chip’s neuron update rule.

### L139 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:218)

Original line 256.

**Reviewed:** Event camera recordings are mostly silent, so this difference is large in practice. The comparison below runs one neuron with six inputs for sixteen timesteps and counts the operations each model performs.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Sparse inputs can reduce spike-triggered additions. The comparison below counts operations for one neuron with six inputs over sixteen timesteps. It illustrates activity dependence rather than measuring hardware energy.

### L140 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:249)

Original line 291.

**Reviewed:** The networks in this project realize roughly ten million connections, so the ratio shown here scales up accordingly. The cost of the SNN is that time becomes part of the computation: the network runs T times per sample instead of once.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** A full network’s operation count depends on activity in every layer and on the number of timesteps. The single-neuron example does not predict that count directly.

### L141 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:262)

Original line 314.

**Reviewed:** A still background produces no events at all, while a moving hand throws off a dense stream of them along its edges. What the sensor records is a sparse sequence of changes, which is the kind of input a spiking network consumes.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** An event camera records brightness changes, so moving edges generate events while a stable background produces few. This provides a sparse input stream for a spiking network.

### L142 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:297)

Original line 353.

**Reviewed:** Accuracy measured during ordinary training overstates what the deployed network achieves, so it is the wrong quantity to optimize. Any search guided by it will prefer configurations that happen to degrade badly on conversion.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Floating-point accuracy can differ from accuracy after conversion. Ranking by the converted result directly measures the form the project intends to deploy.

### L143 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:1)

Label or passage revised during the review.

**Reviewed:** The implementation is a Python package, snnsearch , with one launcher beside it. Modules are layered so the cheap parts stay cheap: the four that describe and check a configuration import no deep learning library at all, so planning and the connection check run on a laptop with nothing installed.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The implementation is a Python package, snnsearch , with a separate launcher. Configuration, planning, and connection checks do not import a deep learning library, so those checks can run without PyTorch or a GPU.

### L144 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:459)

Original line 524.

**Reviewed:** Values live in the file. A dataset is code rather than values, so the file points at a Python callable for that and carries everything else.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The configuration file holds settings and names the Python callable that loads the dataset.

### L145 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:479)

Original line 547.

**Reviewed:** The format is YAML rather than JSON because every narrowed range in this project carries the statistic that narrowed it, on the same line. JSON has no comments, so it would strip exactly the property that makes the search space checkable.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** YAML supports comments alongside settings. The configuration uses them to record why particular search ranges were chosen.

### L146 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:42)

Original line 46.

**Reviewed:** hardware-executable one

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** a hardware representation

### L147 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:20)

Original line 20.

**Reviewed:** The goal in section 1.8 is to search over network designs. A search can only vary what the program treats as data, and it can only reject a design cheaply if the layer shapes are known before anything is built. This page covers how a configuration is described, how its shapes are derived, and how its size on chip is counted.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The architecture is stored as configuration so the search can vary it. A shared plan computes layer shapes before model construction, allowing connection limits and network size to be checked without training.

### L148 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:190)

Original line 178.

**Reviewed:** The three counts above describe what a network is . None of them describe what it does . A spiking network performs arithmetic only where a spike arrives, so two networks with identical connection counts can do very different amounts of work depending on how often their neurons fire.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Parameter, neuron, and connection counts describe network structure. Computational activity also depends on how many spikes pass through those connections.

### L149 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:194)

Original line 210.

**Reviewed:** Work has to be measured on real data rather than derived from the shape. The accepted measure is the synaptic operation: one accumulate at one downstream neuron, caused by one arriving spike. Summed over a sample it is the quantity a neuromorphic chip spends energy on, and it counts events rather than seconds, so it compares across platforms.

**Fix:** Separate counted operations from measured energy. State the metric actually observed.

**Revised example:** A synaptic operation is one addition to a downstream neuron caused by an incoming spike. Summing these operations over a clip estimates activity for that input. The estimate does not include all hardware costs.

### L150 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:211)

Original line 232.

**Reviewed:** The name matters. SynOps counts events, not joules. Real energy also includes memory traffic, data movement between cores and static power, and on an FPGA static power may dominate all of them. The column is called synops for that reason, and calling it energy would be a claim the measurement does not support.

**Fix:** Separate counted operations from measured energy. State the metric actually observed.

**Revised example:** SynOps count operations. Energy measurements would also need to account for memory traffic, routing between cores, and static power.

### L151 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:214)

Original line 240.

**Reviewed:** Reporting SynOps is free and always happens. Trading accuracy against it is a decision the tool should not make on your behalf, so there are four modes:

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** SynOps are reported during evaluation. The search offers four ways to use that count when selecting configurations:

### L152 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:256)

Original line 285.

**Reviewed:** A preference like "97.5% at lower energy beats 97.7% at higher, but not 98.2%" is a Pareto preference. Collapsing it into a single weighted score forces a number to be invented before the data exists to choose it, which is why pareto is the honest default.

**Fix:** Separate counted operations from measured energy. State the metric actually observed.

**Revised example:** A Pareto comparison retains candidates for which neither accuracy nor operation count can improve without worsening the other. A weighted objective instead requires choosing the relative importance of those metrics in advance.

### L153 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:399)

Original line 443.

**Reviewed:** Spatial position is preserved, which this dataset rewards. Every strong configuration found by the search used it.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Flatten retains spatial position. The leading candidates in the reported search used this head, although that observation does not isolate its effect.

### L154 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:408)

Original line 452.

**Reviewed:** Spatial position is discarded, which cost roughly 40 accuracy points on this task.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** GAP discards spatial position. In the comparison at the first stopping checkpoint, it had lower accuracy. Its performance after full training was not measured.

### L155 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:20)

Original line 20.

**Reviewed:** Section 1.3 listed the connection limits and the numeric formats HiAER-Spike imposes. This page turns them into code by running checks that rejects unbuildable configurations, and a training procedure that keeps the network inside the INT16 grid instead of quantizing it once training is over.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** This page implements HiAER-Spike’s connection limits and signed 16-bit numeric format. It explains how to reject infeasible configurations before training and how to account for conversion during training.

### L156 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:35)

Original line 38.

**Reviewed:** Every connection on this chip takes up one row in a stored table, and that table holds a fixed number of rows. A configuration needing more rows than exist cannot be wired at all, and training it teaches nothing.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The hardware stores connections in tables with fixed capacity. A configuration that exceeds a connection limit must be rejected before deployment.

### L157 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:39)

Original line 46.

**Reviewed:** The limits can be evaluated from the layer shapes alone, which section 3.2 already computes without a GPU. Checking them before training costs microseconds and removes a large fraction of the search space at no cost.

**Fix:** Remove personification or dramatic framing. Name the operation, observation, and consequence.

**Revised example:** The layer plan from section 3.2 provides the dimensions needed to check connection limits before training. This avoids spending training time on infeasible configurations.

### L158 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:184)

Original line 199.

**Reviewed:** Within the range, the grid is fine enough that rounding error is around 10 −5 , which is negligible for a network. Truncation at the edge is the part that costs something, and a trained weight rarely lands out there on its own. Something has to push it.

**Fix:** Remove personification or dramatic framing. Name the operation, observation, and consequence.

**Revised example:** Within the representable range, rounding error is on the order of 10 −5 . Clipping values outside that range can cause larger changes, especially after batch-normalization folding.

### L159 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:192)

Original line 207.

**Reviewed:** Batch normalization rescales each channel using running statistics and two learned parameters. Networks here depend on it. An earlier iteration of this search found that every configuration built without normalization failed to learn at all. The chip provides no normalization layer, so it has to be removed before deployment by folding it into the weights and thresholds.

**Fix:** Qualify the statistical claim and distinguish within-run variation from evidence across independent runs.

**Revised example:** Batch normalization rescales each channel using running statistics and learned parameters. The chip has no normalization layer, so conversion folds this transformation into neighboring parameters. Earlier search comments claimed normalization was essential, but the local 82-trial analysis did not support that general claim.

### L160 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:245)

Original line 270.

**Reviewed:** Adding a constant to a neuron's input every timestep has the same effect as lowering its threshold by that constant. Since the chip stores a threshold per neuron, the bias can be expressed as a per-channel threshold instead, with no extra hardware support.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** A per-channel threshold can store the bias adjustment produced by folding. This keeps the representation compatible with the chip, but is not generally equivalent to injecting that bias on every timestep. The comparison below explains the temporal difference.

### L161 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:376)

Original line 413.

**Reviewed:** The swap looks like an even trade. A neuron that needs to reach 1.0 and receives a free +0.15 each step seems no different from one that only has to reach 0.85. That reasoning holds when there is no leak. With a leak it breaks, and the size of the break is set by τ.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** A bias adds current on every timestep. A threshold shift changes the firing boundary. These operations generally produce different state trajectories, including when repeated input accumulates without leak.

### L162 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:381)

Original line 486.

**Reviewed:** A bias arrives every timestep and accumulates. The leak drains v /τ per step, so the pile grows until draining balances arriving, which happens at b′·τ. A threshold shift is worth b′ once and never compounds. At τ=63 a bias of 0.15 lifts the membrane by 9.45 while the matching threshold shift is worth 0.15.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** During a period without spikes or resets, a constant input bias contributes a membrane offset that approaches b′·τ for the update rule used here. A fixed threshold shift does not reproduce that changing offset.

### L163 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:385)

Original line 432.

**Reviewed:** No single stored threshold can imitate the bias, because the offset is not constant. It restarts at zero after every spike and climbs toward b′·τ during silence. The fix is to stop imitating. Training should apply the bias the way the chip will, so the optimizer works against the deployed behavior rather than a version of it that only exists during training.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Because the bias contribution varies over time and resets after spikes, a fixed threshold adjustment cannot generally reproduce it. Training should therefore use the same threshold form that export writes.

### L164 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:517)

Original line 570.

**Reviewed:** Sections 4.2 and 4.3 together mean that quantization applied after training is destructive in a way training cannot anticipate. Rounding also has a zero derivative, so it cannot simply be inserted into the forward pass and trained through.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Quantization and folding can change the network’s outputs. Applying them during training allows the optimizer to respond to these changes. Since rounding has zero derivative almost everywhere, the backward pass uses an approximate gradient.

### L165 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:558)

Original line 615.

**Reviewed:** That arrangement means a weight is two numbers rather than one, and separating them answers a question the sections above leave open: whether training itself truncates.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Quantization-aware training retains a floating-point parameter for optimization and computes a quantized value for the forward pass.

### L166 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:648)

Original line 724.

**Reviewed:** The deployed network computes something the validated one did not. The difference of 0.26 is damage, and no training metric predicted it.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The deployed weight differs by 0.26 from the value used during floating-point validation. Its effect on accuracy must be evaluated after conversion.

### L167 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:655)

Original line 731.

**Reviewed:** Exceeding the boundary buys no reduction in loss, so gradient descent settles somewhere inside the range instead.

**Fix:** Remove personification or dramatic framing. Name the operation, observation, and consequence.

**Revised example:** The forward value remains clamped at 1.0. The approximate gradient may still move the underlying parameter outside the range, so this does not guarantee an in-range floating-point weight.

### L168 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:656)

Original line 732.

**Reviewed:** Export changes nothing. The deployed network is the validated one, and the truncated weights are not an error relative to what was trained.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Export retains the quantized weights used during training. Other aspects of conversion, including neuron state updates and threshold folding, still require verification.

### L169 · [neurons.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/neurons.html:137)

Original line 135.

**Reviewed:** The substitution has to be removed. Training and deployment need to use one neuron, which means implementing the chip's update order directly rather than converting into it later.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Training uses a custom neuron that implements the chip’s update order. This aligns input scaling and spike timing before conversion.

### L170 · [neurons.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/neurons.html:188)

Original line 189.

**Reviewed:** An earlier study sampled τ as a continuous value between 1.5 and 2.5 and reported a correlation between τ and accuracy. Every value in that interval rounds to 2 on the chip, so all of those configurations deploy as one and the same network. Whatever that correlation measured, the hardware cannot act on it.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** An earlier study sampled τ continuously between 1.5 and 2.5. Those values map to the same integer leak value, 2, after conversion. A relationship measured before rounding does not establish a difference between deployed leak settings.

### L171 · [neurons.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/neurons.html:41)

Original line 39.

**Reviewed:** The chip stores the threshold as int(θ / w_delta) with w_delta = 1/32767 . Two values of θ break that encoding. Above 1.0 the integer exceeds the field. At or below 0 the comparison v > θ is always true, and the neuron fires on every timestep regardless of input.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The chip stores the threshold as int(θ / w_delta) , where w_delta = 1/32767 . The project requires a positive threshold no greater than 1.0. Larger values exceed the selected encoding range. Nonpositive thresholds can cause firing at rest and are rejected.

### L172 · [neurons.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/neurons.html:348)

Original line 366.

**Reviewed:** The four decisions above appear together in about fifteen lines.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The implementation below combines the update order, integer leak, representable threshold, and surrogate gradient.

### L173 · [search.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/search.html:20)

Original line 20.

**Reviewed:** Pages 2 to 5 built one configurable network and a way to measure what it achieves after conversion. This page covers the process that searches the configuration space on a fixed hardware budget, and the metric that decides which configuration wins.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The search proposes configurations, rejects those that exceed hardware limits, and trains the remaining candidates. Early validation scores control stopping, while accuracy after conversion determines the final ranking.

### L174 · [search.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/search.html:31)

Original line 33.

**Reviewed:** Three libraries divide the work, and the file wires them together with settings that matter enough to be defended in comments.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Optuna proposes configurations, Ray runs trials, and ASHA stops trials with low intermediate scores. ASHA is an asynchronous successive-halving scheduler: it allocates more training time to candidates that perform well at scheduled checkpoints.

### L175 · [search.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/search.html:56)

Original line 65.

**Reviewed:** Spending equal time on every proposal wastes most of the budget. Evaluation should proceed in stages, with each stage cheap enough to justify the population that reaches it.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Evaluation proceeds in stages so configurations that fail a hardware check or perform poorly early do not consume the full training budget.

### L176 · [search.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/search.html:221)

Original line 237.

**Reviewed:** Two of these entries are reopenings rather than narrowings. A result that describes a coordinate the hardware cannot represent, or that was measured under an objective now known to be wrong, carries no information about the current objective.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Two ranges were reopened because their earlier measurements used floating-point settings or objectives that did not match the deployed model. Those measurements need reevaluation under the current procedure.

### L177 · [search.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/search.html:230)

Original line 250.

**Reviewed:** Ray writes its consolidated export after the search returns. A search that runs for thirty hours and is then interrupted leaves nothing behind.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The project’s consolidated export runs after the search returns. Incremental project reports also preserve results when a run is interrupted. Ray’s own trial logs and checkpoints are separate artifacts.

### L178 · [training.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/training.html:108)

Original line 106.

**Reviewed:** Frames have to be reduced to 64×64 or 32×32 and converted to binary. The choice of how to reduce them matters: averaging a sparse binary frame produces small fractional values that a spiking input cannot represent, and isolated events disappear.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Frames must be resized to fit the input-axon limit and converted to binary spikes. The corrected search in section 7.3 considers resolutions from 24 to 88 pixels. Averaging can produce fractional values, so the encoding must specify how they become spikes.

### L179 · [training.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/training.html:216)

Original line 218.

**Reviewed:** Stopping the simulation at T throws away the last frames of every clip, and how many grows with depth. A network measured that way scores lower on chip than it did in training. Nothing in the configuration says why.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Stopping after T steps omits responses still propagating through the layers. The number of omitted response steps grows with the number of spiking layers.

### L180 · [reference.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/reference.html:33)

Original line 33.

**Reviewed:** Nothing matches that. The index covers names, one-line descriptions and the notes inside each entry, so a broader word usually finds it.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** No matching entries. Search by function name, module, or a broader term.

### L181 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:96)

Label or passage revised during the review.

**Reviewed:** Once the network is expanded and mapped, the converter feeds the input as spikes for T timesteps, runs a few further timesteps with no input at all, and counts the spikes each output neuron produced. The extra timesteps are there because each layer holds its output back by one step, so the last frame of input needs a moment to reach the far end. Those trailing timesteps are called the flush . The reason behind flushing will be discussed later.

**Fix:** Remove the redundant forward promise and define flush at first use.

**Revised example:** After conversion, the network receives T input frames and then runs additional timesteps with zero input. These flush steps allow delayed spikes to reach the classifier. Section 5.4 explains how the implementation chooses the number of steps.

### L182 · [index.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/index.html:261)

Original line 307.

**Reviewed:** The recordings come from a dynamic vision sensor, an event camera with a 128×128 array. A conventional camera reports every pixel's brightness on a fixed clock. An event camera has no clock. Each pixel reports on its own, only when its own brightness changes, with increases and decreases arriving on separate channels. That last detail is why the data has two channels rather than three.

**Fix:** Replace the absolute “no clock” claim and unnecessary comparison with three channels.

**Revised example:** The recordings come from a dynamic vision sensor with a 128×128 pixel array. Pixels report changes in brightness asynchronously. Increases and decreases form the two polarity channels.

### L183 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:1)

Original line 111.

**Reviewed:** Two fields above need a word of explanation. norm chooses the normalization layer: bn is the ordinary batch normalization of section 1.3, tdbn is a variant that also normalizes across the timesteps rather than the batch alone, and none switches it off. Turning it off makes the model enable a bias on each convolution, since a convolution with neither would hold every output channel at zero mean. dilation spaces a kernel's taps apart so it covers a wider area without more weights. It stays at 1 throughout this project.

**Fix:** Remove the false implication that a convolution without bias necessarily has zero-mean outputs.

**Revised example:** norm selects batch normalization ( bn ), a variant that also normalizes over timesteps ( tdbn ), or no normalization ( none ). With no normalization, the model enables convolution bias. dilation spaces filter positions apart to increase the receptive field without adding weights. It remains 1 in this project.

### L184 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:258)

Original line 291.

**Reviewed:** In the two-layer configuration below, the second convolution holds 50,176 parameters and realizes 7.2 million connections, because the same filter is applied at 144 positions. Only the parameter count ever reaches a GPU's memory. All 7.2 million have to be stored individually here, which is why the chip's limits are written as fan-in and fan-out rather than as a parameter budget.

**Fix:** Remove the absolute claim that only parameters consume GPU memory.

**Revised example:** In the example below, the second convolution has 50,176 parameters and 7.2 million connections. The filter weights are shared across 144 output positions in the model, while the hardware represents the connections individually. Parameter storage therefore differs from connection storage.

### L185 · [architecture.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/architecture.html:298)

Original line 337.

**Reviewed:** The panel below runs the planning, costing and feasibility code from the file, ported to JavaScript and checked against the Python implementation. A few things are worth trying: 128 channels breaks the connection limits, a GAP head moves the parameter count by an order of magnitude, and enough layers will eventually trip the small-map guard from section 3.2.

**Fix:** Replace an informal three-item suggestion with concise interaction guidance.

**Revised example:** The panel runs JavaScript versions of the planning, counting, and feasibility checks, compared against the Python implementation. Change the channel count or classifier head to inspect the connection limits. Increasing depth also demonstrates the small-map guard from section 3.2.

### L186 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:233)

Original line 253.

**Reviewed:** That gives weight decay two jobs at once. Alongside its usual regularization role it now also holds the folded distribution inside the representable range, since a folded weight is the trained weight times the channel scale. The search tunes one number against both. Whether the second job binds is measurable, and in the runs recorded further down it does not.

**Fix:** Replace personified “two jobs” and repeated explanation with the measurable condition.

**Revised example:** Weight decay can limit the magnitude of folded weights as well as regularize training. Whether clipping remains a problem must be checked on the folded values before clamping.

### L187 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:275)

Original line 358.

**Reviewed:** Nothing in the float model constrains b′. It is β + (conv_b − μ)·scale, and β is a free learned parameter, so training can place it anywhere. The result is discovered at export, after the full epoch budget has been spent, and the trial is scored zero.

**Fix:** Remove absolute and dramatic framing about discovering failure after an entire budget.

**Revised example:** In the floating-point model, the folded bias is β + (conv_b − μ) · scale . Without an explicit constraint, learned β can make this bias incompatible with the threshold encoding. The deployment check then rejects the result.

### L188 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:375)

Original line 409.

**Reviewed:** Everything above keeps θ′ inside the range the chip can store. A second question sits underneath it: whether swapping the bias for a threshold reproduces the behavior at all.

**Fix:** Replace the strained spatial metaphor “a second question sits underneath it”.

**Revised example:** A legal stored threshold does not by itself establish equivalent neuron dynamics. The following comparison tests the bias and threshold forms.

### L189 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:563)

Original line 623.

**Reviewed:** Truncation applies to the used value alone. If the optimizer sets the stored value to 1.3, the stored value stays 1.3 and the used value becomes 1.0. So weights do pass the boundary during training. What cannot pass it is the number the network computes with.

**Fix:** Clarify the confusing use of “stored value” for both raw and deployed weights.

**Revised example:** Clamping affects the value used in the forward pass. If the underlying floating-point parameter is 1.3, it remains 1.3 while the network computes with 1.0. Export saves the quantized value.

### L190 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:568)

Original line 632.

**Reviewed:** Clamping the stored value directly would leave the optimizer with nowhere to record small changes. A weight held at 1.0 would receive an update wanting 1.25, stay at 1.0, then receive one wanting 0.9, with no memory of the intervening movement.

**Fix:** Remove the extended imagined sequence of weight movements.

**Revised example:** Keeping the underlying floating-point parameter allows small optimizer updates to accumulate even when rounding leaves the forward value unchanged.

### L191 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:572)

Original line 640.

**Reviewed:** The unconstrained value has to be kept somewhere, so that a weight can travel from 1.3 down to 0.9 continuously while the network sees 1.0 throughout and then 0.9.

**Fix:** Correct the suggestion that the forward value jumps directly from 1.0 to 0.9.

**Revised example:** As the floating-point parameter moves from 1.3 toward 0.9, the forward value remains clamped at 1.0 until the parameter reenters the representable range.

### L192 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:585)

Original line 656.

**Reviewed:** Three things keep the stored value from drifting far past the boundary.

**Fix:** Replace the staged triad with a descriptive lead-in.

**Revised example:** The boundary behavior depends on weight scale, weight decay, and the approximate gradient.

### L193 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:587)

Original line 658.

**Reviewed:** It rarely approaches the boundary at all. Trained convolution weights sit around 0.1 to 0.6, and only folding lifts them toward 1.0.

**Fix:** Limit a sweeping frequency claim to the recorded runs.

**Revised example:** In the recorded runs, unfolded convolution weights typically remained below the clipping boundary. Folding can increase their magnitude.

### L194 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:589)

Original line 660.

**Reviewed:** Nothing is gained by traveling further, since the used value is already pinned at 1.0 and the network's output stops responding.

**Fix:** Replace the slogan-like fragment with the specific mechanism.

**Revised example:** Once the forward value is clamped, further increases in the raw parameter do not change that forward value.

### L195 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:608)

Original line 679.

**Reviewed:** max_abs_weight reads the same tensor and is informative anyway. A weight that had been clamped would appear there as exactly 1.0. The runs recorded in section 4.3 report 0.665, which is what establishes that no truncation occurred. Pointing the fraction at module.parametrizations.weight.original would make it measure something.

**Fix:** Remove “would make it measure something” and bound the diagnostic inference.

**Revised example:** max_abs_weight also reads the clamped tensor. A value of exactly 1.0 can indicate boundary contact, while the recorded maximum of 0.665 indicates no clipping in that inspected tensor. To measure the fraction requiring clipping, inspect module.parametrizations.weight.original before clamping.

### L196 · [hardware.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/hardware.html:660)

Original line 736.

**Reviewed:** The distinction changes how the range itself should be read. A ptq boundary costs accuracy that could in principle be recovered. With inline the boundary was a constraint the optimizer already worked within, so widening it restores nothing. What widening does is enlarge the set of solutions the optimizer can reach, which may or may not contain a better one.

**Fix:** Shorten an extended contrast without claiming widening can never help.

**Revised example:** With post-training quantization, the converted network may differ from the one selected during training. With inline quantization, the optimizer already sees the constrained forward values. Widening the range changes the available solutions, but does not guarantee an accuracy improvement.

### L197 · [search.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/search.html:52)

Original line 57.

**Reviewed:** Configurations differ enormously in how much they cost to evaluate. Section 4.1 showed that a large fraction violate the connection limits, which arithmetic detects in microseconds. Among the rest, many are clearly weak within a few epochs.

**Fix:** Remove intensifiers and unsupported timing precision.

**Revised example:** Connection checks can reject a configuration before training. Among configurations that fit, intermediate validation scores identify candidates for early stopping.

### L198 · [search.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/search.html:89)

Original line 101.

**Reviewed:** The two roles need separate keys, and the scheduler and the sampler must be pointed at different ones. The final report of a trial has to be the converted measurement, so that a sampler reading the last report reads the right number.

**Fix:** Replace repetitive necessity framing with actual behavior.

**Revised example:** The early-stopping scheduler reads intermediate validation scores. The sampler reads each completed trial’s final converted score. Separate metric keys prevent these two measurements from being mixed.

### L199 · [neurons.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/neurons.html:247)

Original line 258.

**Reviewed:** The constraint has to be built into the parameterization rather than checked afterwards, so that no optimizer step can produce an illegal threshold. One other thing moves the threshold: folding a batch normalization bias into it, covered in section 4.3. That shifted value is allowed to sit outside the legal range mid-run, as long as it lands inside by the end. Deployment checks therefore read the threshold through their own accessor. A clamp applied for numerical safety during training would otherwise hide the illegal value from the one check that exists to catch it.

**Fix:** Shorten the dense paragraph and avoid asserting that an illegal threshold is permitted throughout training.

**Revised example:** The base threshold uses a constrained parameterization so optimizer updates remain within the supported range. Folding can shift that threshold further. Deployment checks inspect the raw folded value to avoid hiding an illegal result behind a safety clamp.

### L200 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:20)

Original line 190.

**Reviewed:** A search of 400 configurations selected five candidates for longer training. Their reported test accuracies range from 78.8% to 83.3% on DVS128 Gesture. The sections below describe the corrected search, comparisons between candidates, and experiments needed to resolve the remaining uncertainty.

**Fix:** Remove navigation metacommentary and lead with the observed tradeoff.

**Revised example:** A search of 400 configurations selected five candidates for longer training. Their reported test accuracies range from 78.8% to 83.3% on DVS128 Gesture. The leading candidate uses more synaptic operations than the other four.

### L201 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:112)

Label or passage revised during the review.

**Reviewed:** The audit also found metrics that treated missing values as measured outcomes and a clipping metric that inspected weights after clamping. Missing data must remain marked as unavailable, and clipping must be measured before the clamp.

**Fix:** Keep distinct debugging findings while removing their repeated moral.

**Revised example:** Other checks found three reporting errors: a missing scheduler field counted as zero pruned trials, an absent feasibility field counted as rejection, and a clipping metric that inspected weights after clamping. Missing values must be reported as unavailable. Clipping must be measured before the clamp.

### L202 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:284)

Original line 500.

**Reviewed:** At the first stopping checkpoint, the comparison reports 29.6% accuracy for GAP and 65.1% for flatten heads, with respective gains of 0.43 and 0.63 percentage points per epoch. GAP was weaker at this checkpoint. Since those runs stopped early, their eventual accuracy remains unknown. The original report’s “all 30” wording conflicts with its 97% stopping rate, so an exact count needs verification.

**Fix:** Keep the numerical observation here and move the count discrepancy to the source note.

**Revised example:** At the first stopping checkpoint, GAP reports 29.6% accuracy and a gain of 0.43 percentage points per epoch. Flatten heads report 65.1% and 0.63 points per epoch. Their eventual performance cannot be inferred from these stopped runs.

### L203 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:56)

Original line 190.

**Reviewed:** The leading candidate reports 83.33% test accuracy at 50.1 million SynOps per clip. A candidate at 80.56% uses 20.3 million SynOps. The available runs show an accuracy–computation tradeoff, with no energy measurement or repeated-seed estimate.

**Fix:** Define SynOps at its first prominent appearance.

**Revised example:** The leading candidate reports 83.33% test accuracy at 50.1 million synaptic operations (SynOps) per clip. A candidate at 80.56% uses 20.3 million. This comparison measures computational activity. Energy and variation across repeated training runs remain unmeasured.

### L204 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:91)

Original line 89.

**Reviewed:** The search recorded one configuration and trained a different one.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Recorded inputs differed from training inputs

### L205 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:202)

Label or passage revised during the review.

**Reviewed:** The candidate ranked first during search also had the highest test accuracy after retraining.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Search rank 1 remained first after retraining

### L206 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:206)

Label or passage revised during the review.

**Reviewed:** The observed ranking does not establish a statistically reliable difference between candidates.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Uncertainty in the ranking

### L207 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:213)

Original line 190.

**Reviewed:** The candidate with second-block average pooling had the highest accuracy and operation count.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Average pooling and stride-2 convolution

### L208 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:218)

Label or passage revised during the review.

**Reviewed:** Operation count and the ratio to a dense network describe different comparisons.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Absolute and relative operation counts

### L209 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:224)

Original line 454.

**Reviewed:** Selecting a candidate requires an accuracy target or an operation budget.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Choosing an accuracy–computation tradeoff

### L210 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:260)

Original line 190.

**Reviewed:** Increasing weight decay from 0.000094 to 0.001 reduced SynOps by 37.5% in these runs.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Fewer operations at weight decay 0.001

### L211 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:265)

Original line 190.

**Reviewed:** At weight decay 0.01, activity fell further and accuracy declined.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Accuracy declined at weight decay 0.01

### L212 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:277)

Original line 196.

**Reviewed:** No delayed improvement in validation accuracy was observed within 300 epochs.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** No delayed improvement within 300 epochs

### L213 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:282)

Original line 190.

**Reviewed:** Global average pooling reduced connection-limit rejections but had lower early accuracy.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Global average pooling: lower early accuracy

### L214 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:287)

Label or passage revised during the review.

**Reviewed:** The search provides insufficient evidence about convolutional dropout.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Convolutional dropout needs more trials

### L215 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:292)

Label or passage revised during the review.

**Reviewed:** The firing-rate penalty included the output layer.

**Fix:** Shorten a sentence-length heading; keep evidence in the paragraph below.

**Revised example:** Output spikes were included in the activity penalty

### L216 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:102)

Original line 105.

**Reviewed:** Both were recorded and discarded; every trial used one value

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Both sampled values were ignored during training

### L217 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:103)

Original line 106.

**Reviewed:** Counter paired layers by list position and exceeded the physical ceiling

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Counter paired layers incorrectly and exceeded the maximum possible count

### L218 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:104)

Original line 107.

**Reviewed:** Winner passes the connection limits

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** The selected configuration passes connection limits

### L219 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:104)

Original line 107.

**Reviewed:** Checked at 32×32, trained at 64×64; fan-in 9,216 against a limit of 8,159

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Checked at 32×32 but trained at 64×64. Fan-in was 9,216, above the 8,159 limit.

### L220 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:124)

Label or passage revised during the review.

**Reviewed:** Exceeded a hardware connection limit. Rejected by the pre-training feasibility check.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Exceeded a hardware connection limit

### L221 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:145)

Original line 161.

**Reviewed:** optimizer, lr, schedule

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** optimizer, learning rate, schedule

### L222 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:319)

Label or passage revised during the review.

**Reviewed:** varies together with three other knobs

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Varies together with three other parameters

### L223 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:321)

Label or passage revised during the review.

**Reviewed:** Negative association in this search

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** negative association in this search

### L224 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:324)

Label or passage revised during the review.

**Reviewed:** Positive association in this search

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** positive association in this search

### L225 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:322)

Original line 459.

**Reviewed:** 3 best, 4 worst

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Highest observed group score at 3, lowest at 4

### L226 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:319)

Original line 456.

**Reviewed:** step schedule best

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Highest observed group score with a step schedule

### L227 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:320)

Original line 457.

**Reviewed:** 48 px best

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Highest observed group score at 48 px

### L228 · [results.html](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/results.html:206)

Original line 454.

**Reviewed:** Read with caution because

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** Limitation

### L229 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:6)

Original line 6.

**Reviewed:** so working configurations set 32 or 64

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** so configurations must select a smaller resolution, such as the 48×48 used by the current leading candidate

### L230 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:7)

Original line 7.

**Reviewed:** Resolves the native or resized question into one height and width.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Returns the effective input height and width after resizing.

### L231 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:8)

Original line 8.

**Reviewed:** Small, but it removes the possibility of one part of the program reading H and W while another reads resize_to.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Provides a shared input shape for planning and model construction.

### L232 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:10)

Original line 10.

**Reviewed:** Per-layer neuron parameters deploy at no cost because the chip reads a neuron model per neuron.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The hardware already stores neuron parameters individually, so layers can use different settings.

### L233 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:12)

Original line 12.

**Reviewed:** The model enables bias when norm is none, since a convolution with neither normalization nor bias holds every output channel at zero mean.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** The model enables convolution bias when norm is none.

### L234 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:22)

Original line 22.

**Reviewed:** The optimization recipe, covering everything that affects training but not shape.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Training settings, including optimizer, learning-rate schedule, and quantization mode.

### L235 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:58)

Original line 58.

**Reviewed:** 63 matches a known-working conversion.

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** 63 was used in a recorded conversion.

### L236 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:62)

Original line 62.

**Reviewed:** Two-sided, because both ends fail without raising: theta' at or below 0 makes the neuron fire on every timestep, and theta' above w_alpha saturates at INT16_MAX so the deployed threshold is not the trained one.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Both bounds matter: nonpositive thresholds are rejected, and values above w_alpha saturate at INT16_MAX.

### L237 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:64)

Original line 64.

**Reviewed:** Reports pressure rather than damage. A block sitting hard against the band is a sign the layer wants a bias the threshold cannot express, which is the case for fold_bias_mode='conv'.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Reports how many unconstrained bias values exceed the threshold-compatible range. Frequent boundary contact indicates that the constraint affects the layer.

### L238 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:68)

Original line 68.

**Reviewed:** Clamping inside the forward pass makes an out-of-range folded weight raise the training loss, rather than appearing only after conversion.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Clamping during the forward pass lets training respond to the values that will be exported. It does not guarantee a particular change in loss.

### L239 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:72)

Original line 72.

**Reviewed:** Meaningful only on the folded network, since folding is what moves weights out of range. Compared against the 2% budget in DEPLOY_LIMITS.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** To estimate clipping, inspect the folded weights before clamping. Reading an already clamped tensor makes this fraction zero. The warning threshold in DEPLOY_LIMITS is 2%.

### L240 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:96)

Original line 96.

**Reviewed:** A ratio near 0.1 folds into the threshold cleanly. Near 0.5 it does not.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The ratio is a diagnostic, not a guarantee of equivalent dynamics. Accumulation over time also depends on the leak constant.

### L241 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:102)

Original line 102.

**Reviewed:** A folded threshold at or below zero makes the neuron fire unconditionally, so the network is unrepresentable and scores zero.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** A folded threshold at or below zero violates the supported range and causes deployment rejection.

### L242 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:114)

Original line 114.

**Reviewed:** Any non-zero contribution becomes a full spike, so no event is lost and density per pixel rises as resolution falls.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Positive interpolated values become spikes. Resizing changes spatial information and can change spike density.

### L243 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:124)

Original line 124.

**Reviewed:** Without them the final frames of every clip never reach the classifier, and deployed accuracy falls below anything measured in training.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Without flush steps, some responses to the final frames may not reach the classifier before evaluation stops.

### L244 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:148)

Original line 148.

**Reviewed:** Their geometry transfers. Their recorded accuracies were measured under the older floating-point objective and will not reproduce, which the comment states directly.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** The configurations provide initial candidates. Their earlier floating-point scores are not directly comparable to the converted objective.

### L245 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:158)

Original line 158.

**Reviewed:** Logging errors are caught so they cannot end a search.

**Fix:** Replace absolute or exclusive wording with a statement limited to the reported run, implementation, or tested values.

**Revised example:** Logging exceptions are caught to allow the search to continue.

### L246 · [assets/demo-reference.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-reference.js:159)

Original line 159.

**Reviewed:** Validates the dataset, warms the cache, starts Ray, checks that the search space pickles, wires the sampler and scheduler, runs, and exports.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** Coordinates dataset checks, trial execution, and result export.

### L247 · [assets/demo-hardware.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-hardware.js:124)

Original line 124.

**Reviewed:** ≤ 0 — this neuron would fire unconditionally on chip. deployment_report scores this config 0.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** ≤ 0. This is outside the supported range. deployment_report rejects the configuration.

### L248 · [assets/demo-hardware.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-hardware.js:127)

Original line 127.

**Reviewed:** θ′ > 1 — legal to compute with, but note it exceeds the (0,1] grid the base θ obeys; the fold’s legality check watches the unclamped value.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** θ′ > 1. This exceeds the supported threshold grid. Check the value before clamping.

### L249 · [assets/demo-hardware.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-hardware.js:131)

Original line 131.

**Reviewed:** |W′| > 1 — the INT16 grid will silently truncate this weight. In a real layer this is counted by weight_clip_fraction.

**Fix:** Replace informal, awkward, or repetitive wording with a concrete technical explanation.

**Revised example:** |W′| > 1. Quantization clips this weight. Count affected weights before clamping.

### L250 · [assets/demo-training.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-training.js:42)

Original line 42.

**Reviewed:** output 8×8 — bilinear > 0 (the code)

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** output 8×8: bilinear interpolation, then > 0

### L251 · [assets/demo-training.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-training.js:42)

Original line 42.

**Reviewed:** output 8×8 — plain average

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** output 8×8: average pooling

### L252 · [assets/demo-training.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-training.js:78)

Original line 78.

**Reviewed:** fractional grays — not spikes

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** fractional values require binarization

### L253 · [assets/demo-neurons.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-neurons.js:166)

Original line 166.

**Reviewed:** textbook (charge→fire) — v and spikes

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** library (charge→fire): v and spikes

### L254 · [assets/demo-neurons.js](C:/Users/Puru/Downloads/configurable-snn-classifier/docs/assets/demo-neurons.js:167)

Original line 167.

**Reviewed:** chip (fire→reset→leak→integrate) — v and spikes

**Fix:** Use a descriptive label that identifies the metric, experiment, or limitation without a slogan.

**Revised example:** chip (fire→reset→leak→integrate): v and spikes


## Punctuation and technical-list inventory

The companion [pattern inventory](C:/Users/Puru/Downloads/configurable-snn-classifier/EDITORIAL_PATTERN_INVENTORY.csv) enumerates 322 lexical candidates from the initial HTML prose. It records the source file, original line, exact match, context, and related revision entries. This is a screening inventory, not 322 proven writing defects. HTML entities are decoded before scanning, so their terminator semicolons are not mistaken for punctuation. Program syntax, source excerpts, scripts, and SVG internals are excluded. Visible JavaScript reference and demo copy was reviewed separately in the entries above.

Retained examples include “leaky integrate-and-fire,” “multiply-accumulate,” “per-layer,” “run-to-run,” and lists naming actual model components. Em dashes used only as missing-value symbols are also appropriate. Sentences with similar lengths were not rewritten merely to defeat a heuristic. The existing prose-audit script can flag “enter the search results” and ordinary technical definitions despite their clear meaning.

## Verification

- All eight HTML pages pass `node tools/validate_pages.js`, including element lookups, tag balance, section references, and internal HTML links.
- Shared navigation and modified demo/reference scripts pass Node syntax checks.
- Browser checks cover working results anchors, the glossary disclosure, desktop table rendering, and mobile containment of wide tables. At 390 px, all eight pages have no page-level horizontal overflow and all 49 generated section anchors resolve. Results were also inspected at 1280 and 1440 px. Wide tables remain locally scrollable. The glossary expands correctly, and browser error logs were empty.
- No training experiment was rerun and no benchmark accuracy was independently reproduced as part of this editorial task.

## Follow-up from browser comments

The reproduction-history section was removed at the user's request. The remaining results sections were renumbered, and the training-page reference was updated. Titles now include “Performance of the top five configurations,” “Effects of weight decay,” and “Separating hyperparameter effects from limited sampling and implementation errors.” Results headings do not use colons. The ranking explanation now describes the eight-clip difference, why repeating training matters, and why the observed range does not guarantee future scores. The weight-decay section now explains weights, shrinkage toward zero, overfitting, and possible changes in spike activity before presenting the experiment.

The audit entries above document the initial review. Their removed-section examples and earlier titles are historical rather than instructions to restore that content.
