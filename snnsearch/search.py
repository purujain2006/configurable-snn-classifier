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


# AF_UNIX caps a socket path at 107 bytes. Ray appends roughly
#     /session_2026-08-27_20-48-43_793189_2023486/sockets/plasma_store
# to whatever temp dir it is given, which is about 64 characters, so the
# directory itself has to fit in what is left.
_SOCKET_PATH_MAX = 107
_RAY_SUFFIX_BUDGET = 64


def _ray_temp_dir(explicit, writer):
    """A scratch directory for Ray, satisfying two constraints that conflict.

    Ray's session directory, object store and spill files all go here, so it
    must sit on a volume with room. But it also holds a unix domain socket, and
    that path cannot exceed 107 bytes. A project directory on a lab filesystem
    is usually long enough on its own to blow the limit once Ray adds its
    session suffix, so "next to the results" is not a safe default.

    Returns a path, or None to let Ray choose (which means /tmp).
    """
    room = _SOCKET_PATH_MAX - _RAY_SUFFIX_BUDGET
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "ray"

    candidates = []
    if explicit:
        candidates.append(os.path.abspath(os.path.expanduser(explicit)))
    # /local_disk/<user>/ray is 20-30 characters and is where the frame cache
    # already lives on this cluster, so it has both the room and the length.
    if os.path.isdir("/local_disk"):
        candidates.append(f"/local_disk/{user}/ray")
    candidates.append(os.path.expanduser("~/.snnsearch-ray"))

    for path in candidates:
        if len(path) <= room:
            writer.log(f"[stream] ray scratch: {path}  ({len(path)}/{room} chars)")
            return path
        writer.log(f"[stream] ray scratch: skipping {path}, "
                   f"{len(path)} chars leaves no room for Ray's socket path "
                   f"(limit {room})")

    writer.log("[stream] ray scratch: falling back to /tmp. If /tmp is small, "
               "set search.ray_temp_dir to a SHORT path on a large volume.")
    return None


def _run_config_cls():
    """The RunConfig a Tuner accepts, across Ray versions."""
    import ray
    from ray import tune
    return getattr(tune, "RunConfig", None) or ray.train.RunConfig


def run_search(cfg, out_dir):
    """Run the search described by `cfg`, writing into `out_dir`."""
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    import torch

    from .spaces import make_define_by_run
    from .results import ResultsWriter
    from .streaming import make_streaming_callback

    s = cfg["search"]
    obj = cfg["objective"]
    writer = ResultsWriter(out_dir)
    writer.log(f"[stream] {describe(cfg)}")

    # Ray puts its session directory, object store and spill files under /tmp.
    # On a shared cluster /tmp is usually a small partition, and Ray warns at
    # 95% then fails object creation once spilling is needed -- hours in, with
    # every trial lost. The results directory already lives somewhere with room
    # for the frame cache, so put Ray's scratch beside it.
    if not ray.is_initialized():
        tmp = _ray_temp_dir(s.get("ray_temp_dir"), writer)
        kwargs = {"include_dashboard": False, "log_to_driver": True}
        if tmp:
            os.makedirs(tmp, exist_ok=True)
            kwargs["_temp_dir"] = tmp
        ray.init(**kwargs)

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

    # One core for the trial plus one per DataLoader worker. Reserving only 1
    # while asking for 4 workers would let Ray schedule more trials than there
    # are cores to run them, which is the oversubscription the loaders module
    # was written to avoid.
    workers = s.get("num_workers", 0)
    cpus = s.get("cpu_per_trial") or (workers + 1)
    writer.log(f"[stream] {cpus} CPU per trial ({workers} loader workers)")

    trainable = _make_trainable(cfg, out_dir)
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": cpus, "gpu": gpu}),
        tune_config=tune.TuneConfig(
            num_samples=s["trials"],
            search_alg=algo,
            scheduler=scheduler,
            # metric/mode deliberately omitted: the searcher already has them
            # and passing them here as well makes Ray reject the scheduler.
        ),
        # Streams every result to disk as it arrives. Without it the only
        # durable output is best.json at the very end, so a search that dies
        # part-way leaves nothing to analyse and report.html has nothing to read.
        # ray.tune.RunConfig, NOT ray.train.RunConfig. From Ray 2.49 the Train
        # V2 API is on by default, so ray.train.RunConfig became a different
        # class whose `verbose` field holds the string "_DEPRECATED". Tune reads
        # that field and calls .value on it, so a Train RunConfig reaches
        # tuner.fit() and dies with
        #     AttributeError: 'str' object has no attribute 'value'
        # which names neither Ray version nor RunConfig. getattr keeps this
        # working on Ray before 2.43, where tune.RunConfig did not exist yet.
        run_config=_run_config_cls()(storage_path=os.path.join(out_dir, "ray"),
                                     name=cfg["run"].get("name", "search"),
                                     callbacks=[make_streaming_callback(
                                         writer, target=s.get("target", 0.975))]),
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

        # Bounded: the spike rate settles within a few hundred samples, and
        # this runs once per trial.
        from .pipeline import measure_run_synops
        res.update(measure_run_synops(res, loaders, spec, max_batches=8))

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
