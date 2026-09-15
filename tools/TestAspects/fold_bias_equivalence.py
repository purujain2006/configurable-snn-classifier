"""Does theta' = theta - b' reproduce what QAT trained?

Inline QAT adds the folded bias to the neuron's INPUT (Practice2.py:995,
`x = x + fb`). export_deployed instead subtracts it from the THRESHOLD
(Practice2.py:1418 and 1529, `new_th = base_th - b_prime`) and clears
_fold_bias. Under the chip's fire -> reset -> leak -> integrate order those are
not the same operation once tau > 1, because a bias arriving every timestep
accumulates against the leak while a threshold shift does not.

This measures the gap on a real checkpoint: real BN statistics, real b', real
theta, real tau, real T and flush count. No torch required.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_ckpt import load, walk

BN_EPS = 1e-5


def folded_bias(g, b, mu, var, eps=BN_EPS):
    """b'_c = beta_c - mu_c * gamma_c / sqrt(var_c + eps)   (Practice2.py:1446)"""
    return b - mu * g / np.sqrt(var + eps)


def folded_scale(g, var, eps=BN_EPS):
    return g / np.sqrt(var + eps)


def run_neuron(xs, tau, theta, bias=0.0):
    """Chip order, one neuron. Practice2.py:895
       spike = v > theta ; v = reset ; v -= (v - v_reset)/tau ; v += x"""
    v = 0.0
    out = np.empty(len(xs), dtype=np.int8)
    for i, x in enumerate(xs):
        s = 1 if v > theta else 0
        out[i] = s
        if s:
            v = 0.0
        v = v - v / tau
        v = v + x + bias
    return out


def main(path):
    ck = load(path)
    sd = {k: a for k, a in walk(ck.get("state_dict", {}))}
    cfg = ck.get("config", {})
    T = int(cfg.get("input", {}).get("T", 16))
    theta = float(cfg.get("neuron", {}).get("v_threshold", 1.0))
    tau = float(cfg.get("neuron", {}).get("tau", 63))
    flush = int(ck.get("hardware_export", {}).get("flush_steps", 3))
    steps = T + flush

    print(f"checkpoint : {path}")
    print(f"tau={tau:g}  theta={theta:g}  T={T}  flush={flush}  total steps={steps}")
    print(f"qat_mode={ck.get('qat_mode')}  bias_mode={ck.get('bias_mode')}\n")

    # ---- recover every conv+BN pair's folded bias ----
    convs = {}
    for k, a in sd.items():
        if ".bn." in k:
            idx = k.split("conv_fc.")[1].split(".")[0]
            convs.setdefault(idx, {})[k.rsplit(".", 1)[1]] = a
        if k.endswith("conv.parametrizations.weight.original"):
            idx = k.split("conv_fc.")[1].split(".")[0]
            convs.setdefault(idx, {})["w"] = a

    export = ck.get("hardware_export", {}).get("neuron_layers", [])
    W_DELTA = 1.0 / 32767

    layers = []
    for li, (idx, d) in enumerate(sorted(convs.items(), key=lambda kv: int(kv[0]))):
        if not {"weight", "bias", "running_mean", "running_var"} <= set(d):
            continue
        bp = folded_bias(d["weight"], d["bias"], d["running_mean"], d["running_var"])
        sc = folded_scale(d["weight"], d["running_var"])
        wp = d["w"] * sc.reshape(-1, *([1] * (d["w"].ndim - 1)))
        layers.append({"idx": idx, "b": bp, "scale": sc, "wp": wp})

        th_p = theta - bp
        print(f"=== conv_fc.{idx}  ({bp.size} channels) ===")
        print(f"  folded bias b'      min {bp.min():+.4f}  max {bp.max():+.4f}  mean|b'| {np.abs(bp).mean():.4f}")
        print(f"  |b'|/theta          mean {np.abs(bp).mean()/theta:.3f}  max {np.abs(bp).max()/theta:.3f}")
        print(f"  theta' = theta - b' min {th_p.min():+.4f}  max {th_p.max():+.4f}"
              f"   channels <= 0: {(th_p <= 0).sum()}/{th_p.size}")
        print(f"  folded weight W'    max|W'| {np.abs(wp).max():.4f}")

        # cross-check against what the run actually exported
        if li < len(export) and isinstance(export[li].get("threshold_int"), list):
            got = np.array(export[li]["threshold_int"], dtype=np.int64)
            want = np.clip(np.round(th_p / W_DELTA), 1, 32767).astype(np.int64)
            agree = int((got == want).sum())
            print(f"  export cross-check  {agree}/{got.size} thresholds match "
                  f"clamp(round((theta-b')/w_delta), 1, 32767)"
                  f"{'  <- confirms theta - b'  if agree == got.size else ''}")
        print()

    # ---- the actual comparison ----
    rng = np.random.default_rng(0)
    print("=" * 78)
    print("SPIKE AGREEMENT: trained behaviour (bias into input) vs deployed (theta - b')")
    print("=" * 78)

    for L in layers:
        bp, wp = L["b"], L["wp"]
        # Drive each channel with its own realistic pre-neuron current: sum of
        # folded weights over the inputs that spiked, at a given input density.
        fan_in = int(np.prod(wp.shape[1:]))
        print(f"\nconv_fc.{L['idx']}   fan-in {fan_in}")
        print(f"  {'density':>8} {'A spikes':>9} {'B spikes':>9} {'chan-steps disagreeing':>24} {'channels differing':>19}")
        for dens in (0.05, 0.10, 0.20, 0.40):
            totA = totB = dis = 0
            chan_diff = 0
            for c in range(bp.size):
                w = wp[c].ravel()
                xs = np.array([w[rng.random(fan_in) < dens].sum() for _ in range(steps)])
                A = run_neuron(xs, tau, theta, bias=float(bp[c]))     # what QAT trained
                B = run_neuron(xs, tau, theta - float(bp[c]))          # what deploys
                totA += int(A.sum()); totB += int(B.sum())
                d = int((A != B).sum()); dis += d
                chan_diff += (d > 0)
            n = bp.size * steps
            print(f"  {dens:>8.0%} {totA:>9} {totB:>9} {dis:>15}/{n:<8} {chan_diff:>15}/{bp.size}")

    print("\nNote: theta - b' is what the chip runs. 'A' is the behaviour the")
    print("training loss was computed against. Divergence means the validated")
    print("network and the deployed network are not the same network.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/probe.pth")
