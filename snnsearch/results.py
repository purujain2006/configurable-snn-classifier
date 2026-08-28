"""Incremental result writing, so an interrupted search keeps its work.

Moved verbatim from Practice2.py lines 2633-2776 by build_from_practice2.py.
Edit the behaviour here, not in the original.
"""

import csv
import json
import os
import shutil
import threading
import time
from datetime import datetime

from .planning import InfeasibleConfig
from .hardware import check_feasibility
from .cost import count_neurons_and_synapses
from .spaces import config_to_specs


class ResultsWriter:
    """Append-only, flush-on-every-write result sink rooted at a directory."""

    def __init__(self, root: str, echo: bool = True):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.echo = echo
        self.started = time.time()

    def path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def append_jsonl(self, name: str, record: dict) -> dict:
        record = {"wall_time": round(time.time() - self.started, 3), **record}
        with open(self.path(name), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    def write_json(self, name: str, obj):
        tmp = self.path(name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path(name))     # atomic: never a half-written best.json

    def write_text(self, name: str, text: str):
        tmp = self.path(name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path(name))

    def append_csv(self, name: str, row: dict, header_order=None):
        p = self.path(name)
        new = not os.path.exists(p)
        with open(p, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header_order or list(row), extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
            fh.flush()

    def log(self, msg: str):
        if self.echo:
            print(msg, flush=True)


def default_results_dir(mode: str, explicit: str = None) -> str:
    if explicit:
        return explicit
    return os.path.join("results", f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def validate_data_dir(data_dir: str) -> str:
    """
    Fail early and legibly on a bad --data-dir.

    spikingjelly calls os.mkdir(root/'download') without creating parents, so a
    root that does not exist surfaces as a bare `FileNotFoundError: [WinError 3]
    ... '<root>\\download'` from deep in the library, which does not point at the
    path you typed. Check it here instead. Returns the absolute path.
    """
    if not data_dir or data_dir.strip(".") == "":
        raise SystemExit(f"--data-dir is a placeholder, not a path: {data_dir!r}\n"
                         "  Pass the real DVS128 Gesture root folder.")
    abs_dir = os.path.abspath(os.path.expanduser(data_dir))
    if not os.path.isdir(abs_dir):
        raise SystemExit(
            f"--data-dir does not exist:\n    {abs_dir}\n\n"
            "  It must be the DVS128 Gesture ROOT folder, containing:\n"
            "      <root>/download/DvsGesture.tar.gz\n"
            "      <root>/download/gesture_mapping.csv\n"
            "  spikingjelly builds extract/, events_np/ and frames_number_*/ beside them.")
    # already-built caches mean the raw archive is no longer needed
    if os.path.isdir(os.path.join(abs_dir, "extract")) or os.path.isdir(os.path.join(abs_dir, "events_np")):
        return abs_dir
    tarball = os.path.join(abs_dir, "download", "DvsGesture.tar.gz")
    if not os.path.isfile(tarball):
        try:
            found = ", ".join(sorted(os.listdir(abs_dir))[:12]) or "(empty)"
        except OSError:
            found = "(unreadable)"
        raise SystemExit(
            f"--data-dir exists but has no dataset in it:\n    {abs_dir}\n\n"
            f"  Expected:\n      {tarball}\n  Found instead: {found}\n\n"
            "  If your root is one level deeper (a folder of the same name inside "
            "itself), point --data-dir at the inner one.")
    return abs_dir


def export_trial_records(results, out_dir: str) -> str:
    """
    Flatten every trial into one self-contained table for statistical analysis
    (variance across seeds, which knobs move accuracy, feasibility rate, the
    accuracy/neuron-count Pareto front). Ray keeps its own per-trial logs, but a
    single flat file with {full config} x {final metrics} x {arch counts} is what
    you actually load into pandas.

    Writes both trial_records.jsonl (loss-less) and trial_records.csv (convenient).
    Returns the CSV path.
    """
    import csv, json as _json
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "trial_records.jsonl")
    csv_path = os.path.join(out_dir, "trial_records.csv")

    rows = []
    for r in results:
        cfg_flat = dict(r.config)
        metrics = dict(r.metrics) if r.metrics else {}
        # derive architecture counts even for trials that never trained, so
        # feasibility analysis covers the whole sample, not just trained ones.
        arch = {}
        try:
            specs = config_to_specs(cfg_flat)
            feasible, violations = check_feasibility(
                specs["input"], specs["encoder"], specs["downsample"], specs["head"], specs["output"])
            arch["feasible"] = feasible
            arch["violations"] = "; ".join(violations)[:300]
            if feasible:
                c = count_neurons_and_synapses(specs)["totals"]
                arch.update(neurons=c["neurons"], connections=c["connections"], params=c["params"])
        except Exception as e:
            arch["arch_error"] = str(e)[:200]
        rows.append({"trial_id": getattr(r, "trial_id", None),
                     **{f"cfg.{k}": v for k, v in cfg_flat.items()},
                     **{f"metric.{k}": v for k, v in metrics.items()},
                     **arch})

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(_json.dumps(row, default=str) + "\n")
    keys = sorted({k for row in rows for k in row})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[records] wrote {len(rows)} trials -> {csv_path}")
    return csv_path
