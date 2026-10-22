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

BEFORE ANYTHING STARTS. run_search refuses a config that contradicts itself
(preflight.check_search_config) and, unless told otherwise, a config that has
not passed `main.py preflight` with the same code. Both used to surface hours
into a run.

BALANCED BLOCK. Early stopping never cuts the balanced block's trials
(scheduling.BalancedBlockASHA), so every option is judged on a full run.

FINALISTS. The best trial of a search is partly the luckiest one: its score is
one training run, and the top of several hundred noisy scores sits above its
true level. After the search, the top `finalists` trials are trained again
from scratch with `finalist_seeds` new seeds each, and the winner is the one
with the best MEAN validation score (finalists.csv, best_finalist.json). The
test set is still never looked at.
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


def _failure_config_cls():
    """FailureConfig from the same module as the RunConfig that will hold it."""
    import ray
    from ray import tune
    return getattr(tune, "FailureConfig", None) or ray.train.FailureConfig


def _warm_frame_caches(cfg, t_choices, writer):
    """Build every frame cache the search can sample, before Ray starts.

    Now that the sampled T reaches the dataset rather than only the encoder, a
    search over several T values can have eight trial processes race to build
    the same cache folder. Practice2.py warmed the cache for exactly this
    reason and the refactor dropped it along the way.
    """
    ds = cfg.get("dataset") or {}
    if ds.get("module") or ds.get("name") != "dvs128" or not ds.get("root"):
        return                              # only DVS128 caches frames this way
    from .data.builtin import warmup_frame_cache
    warmup_frame_cache(ds["root"], t_choices, log=writer.log)


def run_search(cfg, out_dir):
    """Run the search described by `cfg`, writing into `out_dir`."""
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    import torch

    from .spaces import make_define_by_run
    from .results import ResultsWriter
    from .streaming import make_streaming_callback

    s = cfg["search"]
    writer = ResultsWriter(out_dir)
    writer.log(f"[stream] {describe(cfg)}")

    from .preflight import check_search_config, preflight_gate, n_balanced
    errors, warnings = check_search_config(cfg)
    for w in warnings:
        writer.log(f"[stream] warning: {w}")
    if errors:
        raise SystemExit("The search config has problems, so nothing was started:\n  "
                         + "\n  ".join(errors))
    blocked = preflight_gate(cfg, out_dir)
    if blocked:
        raise SystemExit(blocked)

    t_choices = s.get("T_choices") or [cfg["encoding"].get("T", 16)]
    _warm_frame_caches(cfg, t_choices, writer)

    # The dataset's own facts (channels, size, classes, whether a flip keeps
    # the label) shape the space, so it is built once here rather than assumed.
    from .data import build_dataset
    from .spaces import data_facts, MULTI_BIT_CODINGS, MAX_INPUT_BITS
    bundle = build_dataset({**cfg["dataset"], "T": int(t_choices[0])})
    data = data_facts(bundle)
    coding = cfg["encoding"].get("coding") or ("passthrough" if bundle.is_event else "levels")
    max_bits = int(s.get("max_input_bits") or MAX_INPUT_BITS)
    if coding not in MULTI_BIT_CODINGS:
        max_bits = 1
    run_seed = cfg["run"].get("seed", 1)
    writer.log(f"[stream] dataset {bundle.name}: {data}; coding {coding}, "
               f"input bits searched 1..{max_bits}")

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
        t_choices=t_choices,
        per_layer=(s.get("space", "uniform") != "uniform"),
        require=s.get("require") or [],
        data=data, max_input_bits=max_bits, seed=run_seed,
    )
    if s.get("require"):
        writer.log(f"[stream] forced on for every trial: {list(s['require'])}")
    _assert_picklable(space)

    # A balanced block first (spaces.balanced_points), then Optuna's model-based
    # sampler. TPE samples at random until it has n_startup_trials results, so
    # the balanced trials count as its random start and steering begins after.
    from optuna.samplers import TPESampler
    from .spaces import balanced_points, block_factors
    n_block = n_balanced(s)
    if n_block and not space.per_layer:
        writer.log("[stream] balanced block skipped: it draws per-layer shapes, "
                   "and this search uses space: uniform")
        n_block = 0
    points = balanced_points(n_block, seed=run_seed, t_choices=t_choices,
                             batch_size=s.get("batch_size", 16), epochs=s["epochs"],
                             data=data, max_input_bits=max_bits, run_seed=run_seed)
    startup = len(points) + int(s.get("startup_trials", 10))
    algo = OptunaSearch(space=space, metric="val_accuracy", mode="max",
                        points_to_evaluate=points or None,
                        sampler=TPESampler(n_startup_trials=startup, seed=run_seed))
    if points:
        names = list(block_factors(t_choices, max_bits))
        writer.log(f"[stream] balanced block: {len(points)} trials in which every "
                   f"option of {', '.join(names)} appears equally often, each with a "
                   f"random shape that fits the chip; Optuna steers after {startup} trials")

    scheduler = None
    if s.get("scheduler", "asha") == "asha":
        from .scheduling import BalancedBlockASHA
        protect = len(points) if s.get("protect_balanced", True) else 0
        if protect:
            writer.log(f"[stream] early stopping never cuts the {protect} balanced-block "
                       f"trials; it starts with trial {protect + 1}")
        scheduler = BalancedBlockASHA(
            protect=protect,
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
    resources = {"cpu": cpus, "gpu": gpu}
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=resources),
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
        run_config=_run_config_cls()(
            storage_path=os.path.join(out_dir, "ray"),
            name=cfg["run"].get("name", "search"),
            callbacks=[make_streaming_callback(writer, target=s.get("target", 0.975))],
            # A trial that hits CUDA OOM is retried rather than lost. Memory
            # pressure is a property of the moment, so the same configuration
            # usually succeeds once its neighbours have finished. Without this,
            # transient pressure permanently removes configurations from the
            # search for a reason unrelated to their quality.
            failure_config=_failure_config_cls()(
                max_failures=s.get("max_failures", 3)),
        ),
    )
    results = tuner.fit()

    # scope="last": the final report of each trial is the deploy-phase one
    best = results.get_best_result(metric="val_accuracy", mode="max", scope="last")
    with open(os.path.join(out_dir, "best.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": best.config, "metrics": dict(best.metrics)},
                  fh, indent=2, default=str)
    writer.log(f"[stream] best val_accuracy={best.metrics.get('val_accuracy')}")

    _rescore_finalists(cfg, out_dir, results, writer, resources, run_seed)
    return results


def _finalist_configs(results, k):
    """The top k finished, deployable trials by their deploy-phase score."""
    done = []
    for r in results:
        m = r.metrics or {}
        if r.error is None and m.get("phase") == "deploy" and m.get("deployable"):
            done.append((float(m.get("val_accuracy") or 0.0), m.get("trial_id"), dict(r.config)))
    done.sort(key=lambda t: t[0], reverse=True)
    return done[:k]


def run_jobs(cfg, out_dir, flats, name, resources=None):
    """Train each flat config in `flats` once, in parallel through Ray, with
    no early stopping. Returns one dict per job, in order: the deploy-phase
    metrics, or an error. Shared by the finalists and the option study."""
    import ray
    import torch
    from ray import tune

    s = cfg["search"]
    if resources is None:
        gpu = s.get("gpu_fraction", 1.0) if torch.cuda.is_available() else 0
        resources = {"cpu": s.get("cpu_per_trial") or (int(s.get("num_workers", 0)) + 1),
                     "gpu": gpu}
    if not ray.is_initialized():
        tmp = _ray_temp_dir(s.get("ray_temp_dir"), _PrintWriter())
        kwargs = {"include_dashboard": False, "log_to_driver": True}
        if tmp:
            os.makedirs(tmp, exist_ok=True)
            kwargs["_temp_dir"] = tmp
        ray.init(**kwargs)
    base = _make_trainable(cfg, out_dir)
    flats = [dict(f) for f in flats]

    def job(trial_cfg):
        return base(flats[int(trial_cfg["job"])])

    grid = tune.Tuner(
        tune.with_resources(job, resources=resources),
        param_space={"job": tune.grid_search(list(range(len(flats))))},
        run_config=_run_config_cls()(
            storage_path=os.path.join(out_dir, "ray"), name=name,
            failure_config=_failure_config_cls()(max_failures=s.get("max_failures", 3)),
        ),
    ).fit()
    out = [None] * len(flats)
    for r in grid:
        m = dict(r.metrics or {})
        deployed = m.get("phase") == "deploy"
        out[int(r.config["job"])] = {
            "val_accuracy": m.get("val_accuracy") if deployed else 0.0,
            "hw_val_accuracy": m.get("hw_val_accuracy"),
            "sim_val_accuracy": m.get("sim_val_accuracy"),
            "sim_disagreements": m.get("sim_disagreements"),
            "deployable": m.get("deployable"),
            "deployed_tau": m.get("deployed_tau"),
            "synops_per_sample": m.get("synops_per_sample"),
            "error": str(r.error)[:200] if r.error else ""}
    return [o or {"val_accuracy": 0.0, "error": "no result"} for o in out]


class _PrintWriter:
    def log(self, msg):
        print(msg, flush=True)


def finalist_jobs(top, seeds, epochs=None):
    """One job per (finalist, seed): the trial's own config with a new seed."""
    jobs = []
    for rank, (score, trial_id, config) in enumerate(top):
        for seed in seeds:
            flat = {k: v for k, v in config.items() if not k.startswith("_")}
            flat["seed"] = seed
            if epochs:
                flat["epochs"] = int(epochs)
            jobs.append({"rank": rank, "trial_id": trial_id, "search_score": score,
                         "seed": seed, "config": flat})
    return jobs


def summarize_finalists(top, rows):
    """Mean and spread of each finalist's re-scored validation accuracy, and
    the one with the best mean."""
    import statistics
    summary = []
    for rank, (score, trial_id, config) in enumerate(top):
        vals = [float(r["val_accuracy"] or 0.0) for r in rows if r["rank"] == rank]
        summary.append({
            "rank_in_search": rank + 1, "search_trial_id": trial_id,
            "search_val_accuracy": score,
            "mean_val_accuracy": statistics.mean(vals) if vals else 0.0,
            "std_val_accuracy": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "seeds": len(vals),
            "per_seed": [r for r in rows if r["rank"] == rank],
            "config": config})
    return summary, max(summary, key=lambda d: d["mean_val_accuracy"])


def _rescore_finalists(cfg, out_dir, results, writer, resources, run_seed):
    """The search's own finalist stage: search.finalists (default 3) trials,
    search.finalist_seeds (default 3) seeds each."""
    s = cfg["search"]
    k, n_seeds = int(s.get("finalists") or 0), int(s.get("finalist_seeds") or 0)
    if k <= 0 or n_seeds <= 0:
        return None
    return rescore_finalists(cfg, out_dir, _finalist_configs(results, k), writer,
                             resources, run_seed, n_seeds, s.get("finalist_epochs"))


def finalists_from_records(out_dir, k):
    """The top k finished, deployable trials in trials.jsonl, for a search
    that was stopped or crashed before its own finalist stage."""
    path = os.path.join(out_dir, "trials.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"no trials.jsonl in {out_dir}: is that the search's results folder?")
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    done = [(float(r.get("val_accuracy") or 0.0), r["trial_id"], dict(r.get("config") or {}))
            for r in rows if r.get("status") == "complete" and r.get("feasible")
            and r.get("deployable") and r.get("hw_val_accuracy") is not None]
    done.sort(key=lambda t: t[0], reverse=True)
    return done[:k]


def run_finalists(cfg, top=None, seeds=None, epochs=None):
    """`main.py finalists`: retrain the best trials of a finished, stopped or
    crashed search with new seeds and pick the best mean, exactly as a search
    does at its end. Writes finalists.csv and best_finalist.json."""
    import ray
    from .pipeline import results_dir
    out_dir = results_dir(cfg)
    s = cfg["search"]
    k = int(top or s.get("finalists") or 3)
    n_seeds = int(seeds or s.get("finalist_seeds") or 3)
    picks = finalists_from_records(out_dir, k)
    if not picks:
        raise SystemExit(f"no finished, deployable trial in {out_dir}/trials.jsonl")
    try:
        rescore_finalists(cfg, out_dir, picks, _PrintWriter(), None,
                          cfg["run"].get("seed", 1), n_seeds,
                          epochs or s.get("finalist_epochs"))
    finally:
        if ray.is_initialized():
            ray.shutdown()
    return 0


def rescore_finalists(cfg, out_dir, top, writer, resources, run_seed, n_seeds, epochs=None):
    """Train the `top` trials again with fresh seeds and pick by the mean.

    No early stopping. The seeds differ from the one every search trial used,
    so no finalist reuses the run it was selected on.
    """
    import csv

    if not top:
        writer.log("[stream] finalists: no deployable trial finished; nothing to re-score")
        return None
    seeds = [int(run_seed) + 1000 * (j + 1) for j in range(n_seeds)]
    jobs = finalist_jobs(top, seeds, epochs)
    writer.log(f"[stream] finalists: re-training the top {len(top)} trials with seeds "
               f"{seeds}, {len(jobs)} runs")

    results = run_jobs(cfg, out_dir, [j["config"] for j in jobs],
                       cfg["run"].get("name", "search") + "_finalists", resources)
    rows = []
    for job, m in zip(jobs, results):
        rows.append({"rank": job["rank"], "search_trial_id": job["trial_id"],
                     "search_val_accuracy": job["search_score"], "seed": job["seed"],
                     "val_accuracy": m.get("val_accuracy"),
                     "hw_val_accuracy": m.get("hw_val_accuracy"),
                     "sim_val_accuracy": m.get("sim_val_accuracy"),
                     "deployable": m.get("deployable"),
                     "error": m.get("error", "")})
    rows.sort(key=lambda row: (row["rank"], row["seed"]))
    with open(os.path.join(out_dir, "finalists.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary, best = summarize_finalists(top, rows)
    for d in summary:
        writer.log(f"[stream] finalist {d['rank_in_search']} ({d['search_trial_id']}): "
                   f"search {d['search_val_accuracy']:.4f}, re-scored "
                   f"{d['mean_val_accuracy']:.4f} +- {d['std_val_accuracy']:.4f} "
                   f"over {d['seeds']} seeds")
    with open(os.path.join(out_dir, "best_finalist.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": best["config"], "best": best, "finalists": summary},
                  fh, indent=2, default=str)
    writer.log(f"[stream] best finalist: {best['search_trial_id']}, mean val "
               f"{best['mean_val_accuracy']:.4f} (best_finalist.json)")
    return best


def _assert_records_what_ran(trial_cfg, spec, encoder):
    """The config a trial gets recorded under must be the one it trained on.

    A search records the sampled config and reports a score, and nothing in
    between checks that the two belong together. When they came apart here,
    every trial trained at the config file's input size while the leaderboard
    named the sampled one, so the winner could not be reproduced and two knobs
    were searched in name only. The failure was silent for four hundred trials.

    Cheap invariant, checked once per trial: whatever the sampler chose for the
    input has to appear in the spec that builds the model and in the encoder
    that feeds it. Dying here costs one trial. Not checking cost a search.
    """
    inp = spec["input"]
    want_T = trial_cfg.get("T")
    want_resize = trial_cfg.get("resize_to")
    problems = []
    if want_T is not None and int(want_T) != int(inp.T):
        problems.append(f"T: sampled {want_T}, spec has {inp.T}")
    if want_T is not None and int(want_T) != int(getattr(encoder, "T", want_T)):
        problems.append(f"T: sampled {want_T}, encoder has {encoder.T}")
    if want_resize is not None and int(want_resize) != int(inp.resize_to):
        problems.append(f"resize_to: sampled {want_resize}, spec has {inp.resize_to}")
    if problems:
        raise RuntimeError(
            "the trial would be recorded under a config it is not running:\n  "
            + "\n  ".join(problems))


def _make_trainable(cfg, out_dir):
    """Build the per-trial function. Closes over plain data only, so it pickles."""
    obj = dict(cfg["objective"])
    run_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    gpu_frac = cfg["search"].get("gpu_fraction", 1.0)

    def trainable(trial_cfg):
        import torch
        from ray import tune
        from .hardware import check_spec_feasibility, set_limits
        from .pipeline import resolve_specs, make_loaders
        from .train import run_training

        set_limits(run_cfg.get("limits"))     # a fresh process has the defaults

        # The trial config decides the input size and the time depth, not just
        # the architecture, so it has to reach the data pipeline. An earlier
        # version built the spec here and the loaders from the YAML, then
        # overwrote the sampled T and resize_to with the file's. Every trial
        # then trained at the file's input size whatever the sampler chose, the
        # recorded config described a run that never happened, and replaying a
        # winner produced a different and weaker network. Resolve once, from
        # one source.
        _bundle, _enc, spec = resolve_specs(run_cfg, trial_cfg, quiet=True)
        _assert_records_what_ran(trial_cfg, spec, _enc)

        # Make gpu_fraction mean something. Ray's fraction is pure bookkeeping:
        # it decides how many trials to schedule and then lets every one of them
        # allocate the whole card. So the trials collectively overcommit, and
        # the process that happens to ask next dies of OOM -- which is usually
        # not the process that took too much. Capping each one at its declared
        # share makes an over-large configuration fail on its own budget instead
        # of starving its neighbours.
        if gpu_frac and torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(float(gpu_frac))
            except Exception:
                pass                      # older torch: fall back to no cap

        # tier 1: arithmetic. Never reaches a GPU, and now runs against the
        # input spec that will actually train. Checking the axon budget at one
        # resolution and training at another passes configurations that do not
        # fit the chip.
        feasible, violations = check_spec_feasibility(spec)
        if not feasible:
            # Ray requires the key ASHA prunes on to be present on EVERY report,
            # so the reject path must carry float_val_accuracy too.
            tune.report({"val_accuracy": 0.0, "float_val_accuracy": 0.0,
                         "feasible": False, "deployable": False, "phase": "deploy",
                         "violations": "; ".join(violations)[:400]})
            return

        loaders = make_loaders(run_cfg, _bundle, _enc, spec, quiet=True)

        # The deploy report below carries no epoch of its own, and it is the
        # LAST result Ray keeps, so anything derived from `epoch` at trial
        # completion read the deploy record and got nothing. That is why
        # `epochs_run` was 0 for every trial that finished training and
        # `stopped_early` was never true: the pruning statistics described the
        # deploy report rather than the run.
        seen = {"epoch": -1}

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
            seen["epoch"] = max(seen["epoch"], int(kw.get("epoch", 0)))
            tune.report({
                "val_accuracy": acc,
                # Present on EVERY report, for the same reason
                # float_val_accuracy is: a key the callback reads must never be
                # absent, or its absence gets read as a value.
                "feasible": True,
                "float_val_accuracy": kw.get("val_acc", acc),
                # best_val_accuracy is now the best SMOOTHED accuracy, so it is
                # no longer the peak of the curve and will read slightly lower
                # than it did in earlier runs. That is the point.
                "best_val_accuracy": kw.get("best_val_acc", acc),
                # The current window mean. ASHA still prunes on
                # float_val_accuracy, the raw single epoch. To prune on the
                # smoothed value instead, set float_val_accuracy to this key --
                # but change one thing at a time, because it alters which trials
                # survive rung 1 and therefore the whole search.
                "smoothed_val_accuracy": kw.get("smoothed_val_acc"),
                "smoothing_window": kw.get("smoothing_window"),
                "phase": kw.get("phase", "float"),
                "train_accuracy": kw.get("train_acc", 0.0),
                "train_loss": kw.get("train_loss", 0.0),
                "lr": kw.get("lr", 0.0),
                "epoch": kw.get("epoch", 0),
                "firing_rate": kw.get("firing_rate"),
            })

        try:
            res = run_training(spec, loaders=loaders, report_fn=report_fn)
        except torch.cuda.OutOfMemoryError:
            # Deliberately NOT scored zero, and deliberately re-raised.
            #
            # Two very different things can stop a configuration, and only one
            # of them is a fact about the configuration. Failing the connection
            # limits means it cannot exist on HiAER-Spike: permanent, scored
            # zero at tier 1 above. Running out of GPU memory means it did not
            # fit in the share we gave it while N other trials happened to be
            # resident: an artifact of how densely this run packs the card,
            # which would not occur at lower concurrency and says nothing about
            # the network.
            #
            # Scoring it zero would teach Optuna that a good architecture is bad
            # because of a scheduling decision. Raising it instead lets Ray
            # retry the trial (see FailureConfig in run_search), usually once
            # neighbours have finished and memory has freed.
            torch.cuda.empty_cache()
            raise

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
            "pre_export_val_accuracy": res.get("pre_export_val_accuracy"),
            "synops_per_sample": synops,
            "synops_reason": res.get("synops_reason"),
            "synops_layers": res.get("synops_layers"),
            "chip_synapses": res.get("chip_synapses"),
            "firing_rate": res.get("firing_rate"),
            "rate_penalty": res.get("rate_penalty"),
            # what export cost, not what the schedule gained
            "quant_gap": res.get("quant_gap"),
            "end_to_end_gain": res.get("end_to_end_gain"),
            "weight_clip_frac": res.get("weight_clip_frac"),
            "min_threshold": res.get("min_threshold"),
            # the `tau` column is the sampled START; trainable tau moves
            "deployed_tau": res.get("deployed_tau"),
            # the same validation clips through simpleSim's arithmetic
            "sim_val_accuracy": res.get("sim_val_accuracy"),
            "sim_disagreements": res.get("sim_disagreements"),
            "deployable": deployable,
            "deploy_reasons": res.get("deploy_reasons"),
            "deploy_warnings": res.get("deploy_warnings"),
            # bits the widest threshold needs as the chip receives it
            "threshold_bits": res.get("threshold_bits"),
            "phase": "deploy",
            "feasible": True,
            # Carried explicitly rather than derived from `epoch`, which this
            # report does not have. A trial that ASHA cut at rung 1 and one
            # that ran the full budget are different facts, and the leaderboard
            # could not tell them apart.
            "epochs_run": seen["epoch"] + 1,
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
