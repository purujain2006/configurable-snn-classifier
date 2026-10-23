"""Early stopping that leaves the balanced block alone.

The balanced block exists so every option gets the same number of trials and
a fair comparison. ASHA undid that: it stops a trial at its first checkpoint
if it trails the others then, and options that learn slowly at first
(augmentation, TET, dropout, GAP, signed spikes) trail early. In Search 04, 77
of the 96 block trials were stopped at epoch 20, the end of the float
warm-up, before any of them had trained on integer weights. Augmentation kept
5 of 48 trials and GAP 8.

BalancedBlockASHA never stops the first `protect` trials it is given, which
are the balanced block, because the searcher hands those out first. They
still report to ASHA, so their scores set the cut-offs for the trials after
them. Everything after the block is stopped as usual.

Module level and importing ray at the top, so Ray can pickle it by reference
when it checkpoints the experiment.
"""

from ray.tune.schedulers import ASHAScheduler, TrialScheduler


class BalancedBlockASHA(ASHAScheduler):
    def __init__(self, protect: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._protect = int(protect)
        self._added = 0
        self._protected = set()

    def on_trial_add(self, tune_controller, trial):
        if self._added < self._protect:
            self._protected.add(trial.trial_id)
        self._added += 1
        super().on_trial_add(tune_controller, trial)

    def on_trial_result(self, tune_controller, trial, result):
        decision = super().on_trial_result(tune_controller, trial, result)
        # a stop at max_t is the trial finishing, not being cut
        finished = result.get(self._time_attr, 0) >= self._max_t
        if trial.trial_id in self._protected and decision == TrialScheduler.STOP \
                and not finished:
            return TrialScheduler.CONTINUE
        return decision

    def is_protected(self, trial_id) -> bool:
        return trial_id in self._protected
