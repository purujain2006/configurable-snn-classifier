"""tools/analysis/recount_synops.py on synthetic data: checkpoint in, corrected count out.

WHY THIS EXISTS

The recount is only worth something if it counts the network that was scored,
the same way a run would have counted it with today's counter. So this builds
a small signed network, saves it the way run_training saves a checkpoint, and
checks that the tool's figure equals measure_synops run directly on the
rebuilt network, that it stays under the dense ceiling, and that the old
figure is read from the single.json beside the checkpoint. No dataset, no GPU.

    python tests/test_recount_synops.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "analysis"))

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ok    {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def save_checkpoint(path, signed=True):
    """A small inline-QAT network, saved with the keys run_training writes."""
    from dataclasses import asdict
    import torch
    from snnsearch.config import (InputSpec, EncoderSpec, OutputSpec, DownsampleSpec,
                                  HeadSpec, NeuronSpec, TrainSpec)
    from snnsearch.model import build_model
    from snnsearch.quantize import hardware_export

    torch.manual_seed(0)
    layers = [{"out_channels": 6, "kernel_size": 3, "stride": 2, "pool": False},
              {"out_channels": 5, "kernel_size": 3, "stride": 1, "pool": True}]
    spec = {"input": InputSpec(C=2, H=16, W=16, T=6, resize_to=16, N=2, input_bits=1),
            "encoder": EncoderSpec(layers_json=json.dumps(layers), norm="bn"),
            "output": OutputSpec(num_classes=4),
            "downsample": DownsampleSpec(mode="stride", pool_type="avg"),
            "head": HeadSpec(final_reduction="flatten", fc_widths="", dropout_rate=0.0),
            "neuron": NeuronSpec(tau=4, signed_spikes=signed),
            "train": TrainSpec(fold_bias_mode="threshold")}
    net = build_model(spec).eval()
    net.to_qat_folded(bias_mode="threshold")
    # strong weights, so the layers past the first fire and the count is not trivial
    for m in net.modules():
        if hasattr(m, "bn"):
            m.conv.weight.data.mul_(4)
    # export_deployed converts in place, so keep the trained form's state first,
    # as run_training's checkpoint does
    trained_state = {k: v.clone() for k, v in net.state_dict().items()}
    hw = net.export_deployed().eval()
    torch.save({"state_dict": trained_state,
                "hw_state_dict": hw.state_dict(),
                "hardware_export": hardware_export(hw),
                "config": {n: asdict(s) for n, s in spec.items()},
                "metrics": {},
                "folded": True, "bias_mode": "threshold",
                "leak_grid": "chip", "qat_mode": "inline"}, path)
    return spec


def synthetic_loader(spec, batches=3, n=4):
    """Batches shaped like the real loader's: (clips (N, T, C, H, W), labels, index)."""
    import torch
    g = torch.Generator().manual_seed(1)
    inp = spec["input"]
    out = []
    for b in range(batches):
        x = (torch.rand(n, inp.T, inp.C, inp.resize_to, inp.resize_to, generator=g) < 0.25).float()
        out.append((x, torch.zeros(n, dtype=torch.long), torch.arange(n) + b * n))
    return out


def main():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is not installed; nothing to check here")
        return 0
    import torch
    import recount_synops as rs
    from snnsearch.hs_export import load_deployed
    from snnsearch.synops import measure_synops
    from snnsearch.train import exact_arithmetic

    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        print("\n-- a signed network, saved and recounted --")
        run = os.path.join(tmp, "rank1")
        os.makedirs(run)
        ckpt = os.path.join(run, "rank1.pth")
        spec = save_checkpoint(ckpt, signed=True)
        loader = synthetic_loader(spec)
        with open(os.path.join(run, "single.json"), "w") as fh:
            json.dump({"synops_per_sample": 1000.0, "synops_over_dense": 0.5}, fh)
        with open(os.path.join(run, "trial.json"), "w") as fh:
            json.dump({"trial_id": "abc123"}, fh)

        row = rs.recount(ckpt, lambda s: loader, device)
        net, spec2, _ = load_deployed(ckpt, device)
        with exact_arithmetic():
            direct = measure_synops(net, loader, device, spec=spec2)

        syn = row["synops_per_sample"]
        check("the count is measured", syn is not None and syn > 0, row.get("reason", ""))
        check("equal to measure_synops on the rebuilt network",
              syn is not None and abs(syn - direct["synops_per_sample"]) <= 1e-9 * max(1.0, syn),
              f"{syn} vs {direct.get('synops_per_sample')}")
        check("within the dense ceiling", syn is not None and syn <= row["synops_ceiling_per_sample"])
        check("layers past the first fire", (row.get("firing_rate") or 0) > 0)
        check("dense comparison present", row.get("synops_over_dense") is not None)
        check("old figure read from single.json", row["old_synops_per_sample"] == 1000.0)
        check("new over old computed", row["new_over_old"] == syn / 1000.0)
        check("trial id read from trial.json", row["trial"] == "abc123")

        print("\n-- finding checkpoints and writing the table --")
        found = rs.find_checkpoints([tmp])
        check("folder search finds the checkpoint", found == [os.path.normpath(ckpt)], str(found))
        out = os.path.join(tmp, "table", "recount.csv")
        rs.write_csv([row], out)
        with open(out, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        check("CSV has a header and one row", len(lines) == 2 and lines[0].startswith("checkpoint,"))

        print("\n-- no single.json: the checkpoint's own metrics are the old figure --")
        os.remove(os.path.join(run, "single.json"))
        check("no old figure means no ratio", rs.recount(ckpt, lambda s: loader, device)["new_over_old"] is None)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
