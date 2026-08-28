"""Ray Tune + Optuna, wired for a hardware objective.

Three things here are easy to get wrong and were each learned the hard way in
the original run record. They are preserved deliberately.

METRIC SPLIT. Early stopping needs a value every epoch and only floating-point
training provides one. Selection needs the converted accuracy, which exists
once per trial. So both keys are reported every time, the scheduler watches
`float_val_accuracy`, and the sampler reads `val_accuracy` with scope="last",
which is the deploy-phase report.

max_t = epochs + 1. ASHA stops a trial once it has reported max_t times and
every report counts, including the deployment one. With max_t = epochs the
trial was killed on its final epoch, the deploy report never landed, and
selection silently fell back to floating-point accuracy -- exactly what the
metric split exists to prevent. Nothing errored; two trials in the recorded
data show the symptom.

METRIC PLACEMENT. OptunaSearch must be given metric/mode directly, because its
set_search_properties returns early when a space is already set. ASHAScheduler
must NOT also receive them from TuneConfig, or Ray raises. Verified against
ray 2.56; do not "simplify".
"""

import json
import os

from .runconfig import describe
from .synops import score_with_energy


def run_search(cfg, out_dir):
    """Run the search described by `cfg`, writing into `out_dir`."""
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    import torch

    from .spaces import make_define_by_run
    from .results import ResultsWriter

    s = cfg["search"]
    obj = cfg["objective"]
    writer = ResultsWriter(out_dir)
    writer.log(f"[stream] {describe(cfg)}")

    # "uniform" shares one kernel/channel/stride across every layer; anything
    # else lets the sampler choose them per layer.
    space = make_define_by_run(
        batch_size=s.get("batch_size", 16),
        epochs=s["epochs"],
        data_dir_abs=os.path.abspath(cfg["dataset"].get("root") or "."),
        t_choices=s.get("T_choices") or [cfg["encoding"].get("T", 16)],
        per_layer=(s.get("space", "uniform") != "uniform"),
    )
    _assert_picklable(space)

    algo = OptunaSearch(space=space, metric="val_accuracy", mode="max")

    scheduler = None
    if s.get("scheduler", "asha") == "asha":
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=s["epochs"] + 1,            # the +1 buys the deployment report
            metric="float_val_accuracy",      # dense per-epoch trajectory
            mode="max",
            grace_period=min(s.get("grace_period", 8), s["epochs"]),
            reduction_factor=s.get("reduction_factor", 3),
            brackets=s.get("brackets", 1),
        )

    gpu = s.get("gpu_fraction", 1.0) if torch.cuda.is_available() else 0
    if gpu and gpu < 1:
        writer.log(f"[stream] gpu_fraction={gpu} -> up to {int(1/gpu)} trials share one card")

    trainable = _make_trainable(cfg, out_dir)
    tuner = tune.Tuner(
        tune.with_resources(trainable,
                            resources={"cpu": s.get("cpu_per_trial", 1), "gpu": gpu}),
        tune_config=tune.TuneConfig(
            num_samples=s["trials"],
            search_alg=algo,
            scheduler=scheduler,
            # metric/mode deliberately omitted: the searcher already has them
            # and passing them here as well makes Ray reject the scheduler.
        ),
        run_config=ray.train.RunConfig(storage_path=os.path.join(out_dir, "ray"),
                                       name=cfg["run"].get("name", "search")),
    )
    results = tuner.fit()

    # scope="last": the final report of each trial is the deploy-phase one
    best = results.get_best_result(metric="val_accuracy", mode="max", scope="last")
    with open(os.path.join(out_dir, "best.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": best.config, "metrics": dict(best.metrics)},
                  fh, indent=2, default=str)
    writer.log(f"[stream] best val_accuracy={best.metrics.get('val_accuracy')}")
    return results


def _make_trainable(cfg, out_dir):
    """Build the per-trial function. Closes over plain data only, so it pickles."""
    obj = dict(cfg["objective"])
    run_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}

    def trainable(trial_cfg):
        from ray import tune
        from .spaces import config_to_specs
        from .hardware import check_feasibility
        from .pipeline import prepare
        from .train import run_training

        spec = config_to_specs(trial_cfg)

        # tier 1: arithmetic. Costs microseconds and never reaches a GPU.
        feasible, violations = check_feasibility(
            spec["input"], spec["encoder"], spec["downsample"],
            spec["head"], spec["output"])
        if not feasible:
            # Ray requires the key ASHA prunes on to be present on EVERY report,
            # so the reject path must carry float_val_accuracy too.
            tune.report({"val_accuracy": 0.0, "float_val_accuracy": 0.0,
                         "feasible": False, "deployable": False, "phase": "deploy",
                         "violations": "; ".join(violations)[:400]})
            return

        _bundle, _enc, loaders, base = prepare(run_cfg)
        for k in ("input", "output"):
            spec[k] = base[k]                # dataset decides shape, not the sampler

        def report_fn(**kw):
            """Two callers with different keywords report through here.

            The training loop sends a full epoch record; the QAT loop inside
            deploy_and_measure sends only phase, epoch and hw_val_acc. Naming
            the arguments explicitly would make one of the two a TypeError
            twenty minutes into a trial, so this takes whatever arrives and
            fills the rest.

            float_val_accuracy must appear on EVERY report, because it is the
            metric ASHA prunes on and Ray requires the pruning key to be
            present each time.
            """
            acc = kw.get("val_acc", kw.get("hw_val_acc", 0.0))
            tune.report({
                "val_accuracy": acc,
                "float_val_accuracy": kw.get("val_acc", acc),
                "best_val_accuracy": kw.get("best_val_acc", acc),
                "phase": kw.get("phase", "float"),
                "train_accuracy": kw.get("train_acc", 0.0),
                "train_loss": kw.get("train_loss", 0.0),
                "lr": kw.get("lr", 0.0),
                "epoch": kw.get("epoch", 0),
            })

        res = run_training(spec, loaders=loaders, report_fn=report_fn)

        # tier 3: the objective is the converted network, not the float one.
        hw = res.get("hw_val_accuracy")
        deployable = res.get("deployable", False)
        acc = hw if (hw is not None and deployable) else 0.0
        synops = res.get("synops_per_sample")
        score = score_with_energy(
            acc, synops, mode=obj.get("mode", "accuracy"),
            synops_budget=obj.get("synops_budget"),
            synops_reference=obj.get("synops_reference"),
            weight=obj.get("weight", 0.0))
        if isinstance(score, tuple):          # pareto mode returns both
            score = score[0]

        tune.report({
            "val_accuracy": score,            # <- selection metric
            "hw_val_accuracy": hw,
            "float_val_accuracy": res.get("float_val_accuracy"),
            "synops_per_sample": synops,
            "quant_gap": res.get("quant_gap"),
            "weight_clip_frac": res.get("weight_clip_frac"),
            "min_threshold": res.get("min_threshold"),
            "deployable": deployable,
            "deploy_reasons": res.get("deploy_reasons"),
            "phase": "deploy",
        })

    return trainable


def _assert_picklable(space):
    """Ray checkpoints the searcher by pickling it. Fail now, not 30 hours in."""
    import pickle
    try:
        pickle.dumps(space)
    except Exception as exc:
        raise SystemExit(
            f"The search space is not picklable ({exc}).\n"
            "  Ray checkpoints the searcher, so the space must be a module-level\n"
            "  object -- not a closure, a lambda, or a nested function.")
