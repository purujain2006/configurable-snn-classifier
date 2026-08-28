"""Write results as the search runs, not only when it finishes.

WHY THIS IS A CALLBACK

Trials are separate processes. Several writing to one file would interleave
partial lines, so the trial function cannot record shared results itself. A Ray
Callback runs in the DRIVER process and sees every tune.report() as it arrives,
which makes it the one safe place to append.

WHY IT MATTERS MORE THAN IT LOOKS

Without it, a search that dies at trial 180 of 200 leaves nothing behind: Ray's
own directories hold checkpoints rather than the flat records the report and the
statistics read. With it, every completed trial is already on disk, so an
interrupted run keeps its work and a running one can be inspected from another
terminal.

WHAT LANDS WHERE

    trial_progress.jsonl   one line per epoch, per trial: the learning curves
    trials.jsonl           one line per finished trial, with its full config
    infeasible.jsonl       configs rejected by the connection limits
    undeployable.jsonl     configs that trained but could not be converted
    leaderboard.csv        flat table, one row per trial, for a spreadsheet
    best.json              rewritten whenever the hardware accuracy improves
    best_summary.txt       the architecture table for that best configuration
    progress.json          counters, for watching without parsing anything
"""

from dataclasses import asdict

# Flat config columns pulled into the leaderboard, in a fixed order so the file
# stays diffable across runs.
LEADERBOARD_COLS = [
    "trial_id", "status", "val_accuracy", "hw_val_accuracy", "float_val_accuracy",
    "quant_gap", "deployable", "weight_clip_frac", "min_threshold", "feasible",
    "epochs_run", "stopped_early", "synops_per_sample", "neurons", "connections",
    "params",
    "depth", "channels", "kernel_size", "stride", "downsample_mode",
    "resize_to", "T", "tau", "trainable_tau", "trainable_threshold",
    "fc_layers", "final_reduction", "dropout_rate", "norm", "tdbn_alpha",
    "optimizer", "lr", "weight_decay", "scheduler", "label_smoothing", "grad_clip",
]


def cfg_to_dict_safe(flat_config):
    """Structured dump of a flat trial config, for best.json.

    Best effort on purpose: this exists to make a record readable, so a config
    that will not convert should be reported as such rather than ending the
    search that produced it.
    """
    from .spaces import config_to_specs
    try:
        return {name: asdict(spec) for name, spec in config_to_specs(flat_config).items()}
    except Exception as exc:
        return {"error": str(exc)}


def make_streaming_callback(writer, target=0.975):
    """A tune.Callback that streams every result to `writer`."""
    from ray import tune

    def _clean(cfg):
        # Absolute paths say nothing about the configuration and make records
        # from different machines look different.
        return {k: v for k, v in (cfg or {}).items()
                if k not in ("data_dir", "results_dir", "data_dir_abs")}

    class StreamingResults(tune.Callback):
        def __init__(self):
            self.best = -1.0
            self.best_trial = None
            self.n_done = self.n_infeasible = self.n_pruned = 0

        def on_trial_result(self, iteration, trials, trial, result, **info):
            writer.append_jsonl("trial_progress.jsonl", {
                "trial_id": trial.trial_id, "epoch": result.get("epoch"),
                "val_accuracy": result.get("val_accuracy"),
                "train_accuracy": result.get("train_accuracy"),
                "train_loss": result.get("train_loss"),
                "lr": result.get("lr"), "phase": result.get("phase"),
                "feasible": result.get("feasible"),
            })

            if not result.get("feasible", True):
                self.n_infeasible += 1
                writer.append_jsonl("infeasible.jsonl", {
                    "trial_id": trial.trial_id,
                    "violations": result.get("violations"),
                    "config": _clean(trial.config)})
                return

            # Only the deploy-phase report can set a new best. A float epoch is
            # progress rather than a result, and ranking on those is what lets a
            # configuration win the search and then lose accuracy on chip.
            if result.get("phase") != "deploy":
                return

            if not result.get("deployable", True):
                writer.append_jsonl("undeployable.jsonl", {
                    "trial_id": trial.trial_id,
                    "reasons": result.get("deploy_reasons"),
                    "float_val_accuracy": result.get("float_val_accuracy"),
                    "config": _clean(trial.config)})
                return

            acc = result.get("hw_val_accuracy") or 0.0
            if acc > self.best:
                self.best, self.best_trial = acc, trial.trial_id
                writer.write_json("best.json", {
                    "trial_id": trial.trial_id,
                    "hw_val_accuracy": result.get("hw_val_accuracy"),
                    "float_val_accuracy": result.get("float_val_accuracy"),
                    "quant_gap": result.get("quant_gap"),
                    "synops_per_sample": result.get("synops_per_sample"),
                    "weight_clip_frac": result.get("weight_clip_frac"),
                    "min_threshold": result.get("min_threshold"),
                    "flat_config": _clean(trial.config),
                    "config": cfg_to_dict_safe(trial.config)})
                try:
                    from .cost import format_summary
                    from .spaces import config_to_specs
                    writer.write_text("best_summary.txt",
                                      format_summary(config_to_specs(trial.config)))
                except Exception as exc:      # never let logging end a search
                    writer.write_text("best_summary.txt", f"(summary failed: {exc})")
                writer.log(
                    f"[stream] NEW BEST on hardware {acc:.4f}  "
                    f"(float {result.get('float_val_accuracy') or 0.0:.4f}, "
                    f"gap {result.get('quant_gap') or 0.0:+.4f}, {trial.trial_id})"
                    + ("   TARGET REACHED" if acc >= target else ""))

        def on_trial_complete(self, iteration, trials, trial, **info):
            self.n_done += 1
            r = trial.last_result or {}
            flat = _clean(trial.config)
            epochs_run = (r.get("epoch") if r.get("epoch") is not None else -1) + 1
            requested = (trial.config or {}).get("epochs")
            stopped_early = bool(r.get("feasible")) and requested is not None \
                and epochs_run < requested
            if stopped_early:
                self.n_pruned += 1
            deployed = r.get("phase") == "deploy"

            writer.append_jsonl("trials.jsonl", {
                "trial_id": trial.trial_id, "status": "complete",
                "feasible": r.get("feasible"),
                "val_accuracy": r.get("val_accuracy"),
                "hw_val_accuracy": r.get("hw_val_accuracy"),
                "float_val_accuracy": r.get("float_val_accuracy"),
                "quant_gap": r.get("quant_gap"),
                "synops_per_sample": r.get("synops_per_sample"),
                "deployable": r.get("deployable"),
                "deploy_reasons": r.get("deploy_reasons"),
                "weight_clip_frac": r.get("weight_clip_frac"),
                "best_val_accuracy": r.get("best_val_accuracy"),
                "epochs_run": epochs_run, "epochs_requested": requested,
                "stopped_early": stopped_early,
                "violations": r.get("violations"), "config": flat})

            writer.append_csv("leaderboard.csv", {
                "trial_id": trial.trial_id,
                "status": "deployed" if deployed else "pruned_before_deploy",
                # A trial pruned by ASHA never reaches the deploy phase, so it
                # has no hardware number. Leaving it blank keeps it distinct
                # from a configuration that deployed and scored zero.
                "val_accuracy": (r.get("hw_val_accuracy") if deployed else None),
                "hw_val_accuracy": r.get("hw_val_accuracy"),
                "float_val_accuracy": r.get("float_val_accuracy"),
                "quant_gap": r.get("quant_gap"),
                "deployable": r.get("deployable"),
                "weight_clip_frac": r.get("weight_clip_frac"),
                "min_threshold": r.get("min_threshold"),
                "synops_per_sample": r.get("synops_per_sample"),
                "feasible": r.get("feasible"), "epochs_run": epochs_run,
                "stopped_early": stopped_early, **flat,
            }, header_order=LEADERBOARD_COLS)

            writer.write_json("progress.json", {
                "trials_completed": self.n_done,
                "infeasible_reports": self.n_infeasible,
                "stopped_early_by_scheduler": self.n_pruned,
                "best_hw_val_accuracy": self.best,
                "best_trial": self.best_trial,
                "target": target, "target_reached": self.best >= target})

        def on_trial_error(self, iteration, trials, trial, **info):
            writer.append_jsonl("trials.jsonl", {
                "trial_id": trial.trial_id, "status": "error",
                "config": _clean(trial.config)})

    return StreamingResults()
