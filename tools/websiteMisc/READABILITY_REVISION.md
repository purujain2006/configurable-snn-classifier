# Website readability revision

This pass addresses the loss of useful explanation in the earlier editorial revision. It preserves the interactive models and reorganizes the supporting material. The earlier itemized audit and pattern inventory remain in this directory.

## Editorial references

- [Wikipedia's signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) describes recurring patterns such as inflated significance, promotional phrasing, formulaic contrasts, and excessive formatting. These are editing prompts, not reliable proof of authorship.
- [GOV.UK clear language guidance](https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-language/) supports concrete wording and explanations readers can follow.
- [GOV.UK clear title guidance](https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-titles/) informs descriptive headings.

Punctuation and lists are not errors by themselves. Keep them when they make the technical explanation clearer. Edit the underlying vagueness, repetition, or unsupported emphasis.

## Patterns addressed in this pass

| Pattern or problem | Change and example |
| --- | --- |
| Broad statements about research code | Replace generalization with a concrete condition: “If layer dimensions are hardcoded in a model class, changing them requires editing code.” |
| Metaphorical descriptions of quantization | Explain the two representations and their roles. A forward value can remain clipped at 1.0 while the optimizer moves the raw weight from 1.3 toward the valid range. |
| An unexplained claim that update order matters | Work through three timesteps with explicit potential, input, leak, and threshold values. |
| Treating network size as one quantity | Define parameters, neurons, and connections through a numerical convolution example, then distinguish these structural counts from activity. |
| Formulaic Observation / Consequence / In code blocks | Retain the explanation in a disclosure with a specific title. Remove repeated category labels. |
| Repeated introductory recaps | Keep one page overview. Place implementation detail beside its relevant experiment or demo. |
| Source excerpts interrupting the explanation | Make the excerpts expandable while retaining their content and anchor links. |
| An introduction that reads like a full chapter | Start with the project, task, reported result, and navigation. Keep both introductory demos visible; group background sections below them. |
| Results without enough explanation of units | Add a worked comparison of absolute SynOps and ratios to dense MAC counts. Explain why different reference denominators change the comparison. |
| Too much visual emphasis | Remove repeated result-category labels and unnecessary separators. Use ordinary headings and restrained disclosure borders. |
| Simulator plots that are hard to inspect | Add pause, manual step, clear-history controls, and visible state outputs to the introductory neuron demo. Add manual step and clear-history controls to the full simulator. |
| Control changes that are awkward to undo | Add a reset button for each demo's input and select controls. |

## Detail retained or restored

The original baseline is commit `3a0d114`. All 75 original ID-bearing inputs, selects, buttons, and canvases remain. The existing simulation algorithms were already present; this pass restores explanatory depth and improves presentation rather than replacing those algorithms.

Added optional worked explanations cover neuron timing, convolution counts, quantization and raw optimizer parameters, final-frame propagation, search metrics, and operation-count comparisons. Existing diagrams, code excerpts, and implementation paragraphs remain accessible. Links to content inside a disclosure automatically open it.

The explicitly unwanted results reproduction section remains removed. The requested results headings and weight-decay explanation remain in place.

## Verification

- All eight pages pass the structural validator, including script element lookups and cross-references.
- The numerical simulation suite passes 77 checks with no failures.
- Browser checks verify manual stepping, clearing history, resetting controls, and links into collapsed details.
- The introduction and results page were inspected at a 390-pixel mobile viewport.

Preview is served locally on port 8766 because the existing server on port 8765 returned incomplete asset responses.
