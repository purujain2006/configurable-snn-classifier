"""tools/chip_limits.py finds a chip's limits, against a stand-in with planted ones.

    python tests/test_chip_limits.py

The stand-in's "simpleSim" is a correct LIF simulator; its "CRI" target cuts
axon numbers to 14 bits, drops an axon's synapses past 4,096, hangs when a
neuron with more than 4,095 synapses fires, and refuses 2**17 neurons. The
probe has to report exactly those limits, skip what depends on them, and
stop everything when the card then fails its health check.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# load torch (through hs_export) before any test patches sys.modules: a module
# first imported under the patch is dropped afterwards, and torch cannot load twice
import snnsearch.hardware  # noqa: E402,F401
import snnsearch.hs_export  # noqa: E402,F401


def planted_limits_hs_api(wedge=False):
    """Stand-in hs_api and hs_bridge modules. wedge: after a hang, every
    later FPGA step hangs too, as a card that needs a reflash would."""
    state = {"wedged": False}

    class Model:
        def __init__(self, kind, theta, _nu, leak):
            self.kind, self.theta, self.leak = kind, int(theta), int(leak)

    class FakeNet:
        def __init__(self, axons, connections, outputs, target, **kw):
            self.axons, self.conns, self.outputs = axons, connections, set(outputs)
            self.target = target
            self.axon_keys = list(axons)
            self.axon_index = {k: i for i, k in enumerate(self.axon_keys)}
            if target == "CRI":
                per_model = {}
                for _syn, m in connections.values():
                    per_model[id(m)] = per_model.get(id(m), 0) + 1
                padded = sum(-(-n // 32) * 32 for n in per_model.values())
                if padded >= 2 ** 17:
                    raise ValueError(f"Insufficient bit width=17: {padded} neurons")
            self.reset()
            self.simpleSim = types.SimpleNamespace(
                numNeurons=len(connections), initialize_sim_vars=lambda n: self.reset())
            state["net"] = self

        def reset(self):
            self.v = {}                    # only membranes that are not 0

        def add(self, syn):
            for post, w in syn:
                self.v[post] = self.v.get(post, 0) + w

        def step(self, inputs, membranePotential=False):
            chip = self.target == "CRI"
            if chip and state["wedged"]:
                raise TimeoutError("step did not complete within 10.0 s")
            # fire on the membrane carried in, reset what fired (no leak)
            spikes = [k for k, v in self.v.items() if v > self.conns[k][1].theta]
            for k in spikes:
                del self.v[k]
            for a in inputs:
                if chip:                   # planted: axon numbers cut to 14 bits
                    a = self.axon_keys[self.axon_index[a] % 2 ** 14]
                syn = self.axons[a]
                self.add(syn[:4096] if chip else syn)   # planted: fan-out cut at 4,096
            for n in spikes:
                syn = self.conns[n][0]
                if chip and len(syn) > 4095:            # planted: a long fan-out hangs
                    state["wedged"] = wedge
                    raise TimeoutError("step did not complete within 10.0 s")
                self.add(syn)
            out = [k for k in spikes if k in self.outputs]
            return (out, 0, 0) if chip else out

    api = types.ModuleType("hs_api.api")
    api.CRI_network = FakeNet
    models = types.ModuleType("hs_api.neuron_models")
    models.LIF_neuron = lambda *a: Model("LIF", *a)
    models.ANN_neuron = lambda *a: Model("ANN", *a)
    ctrl = types.ModuleType("hs_bridge.FPGA_Execution.fpga_controller")
    ctrl.clear = lambda n, dump, core: state["net"].reset()
    fe = types.ModuleType("hs_bridge.FPGA_Execution")
    fe.fpga_controller = ctrl
    return {"hs_api": types.ModuleType("hs_api"), "hs_api.api": api,
            "hs_api.neuron_models": models, "hs_bridge": types.ModuleType("hs_bridge"),
            "hs_bridge.FPGA_Execution": fe, "hs_bridge.FPGA_Execution.fpga_controller": ctrl}


def load_chip_limits():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "chip_limits", os.path.join(ROOT, "tools", "chip_limits.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ChipLimits(unittest.TestCase):
    def test_finds_the_planted_limits(self):
        cl = load_chip_limits()
        lines = []
        with mock.patch.dict(sys.modules, planted_limits_hs_api()):
            rows = cl.probe(log=lines.append)
        got = {(r["family"], r["size"]): r["status"] for r in rows}
        self.assertEqual(got, {
            ("input_axons", 16_000): "pass", ("input_axons", 16_384): "pass",
            ("input_axons", 16_385): "fail",          # axon 16,384 wakes axon 0's neuron
            ("input_axons", 32_768): "skipped", ("input_axons", 65_536): "skipped",
            ("input_axons", 131_072): "skipped",
            ("neurons", 65_536): "pass", ("neurons", 122_880): "pass",
            ("neurons", 131_040): "pass", ("neurons", 131_072): "refused",
            ("axon_fan_out", 4_096): "pass", ("axon_fan_out", 4_097): "fail",
            ("axon_fan_out", 8_192): "skipped",
            ("neuron_fan_out", 4_095): "pass", ("neuron_fan_out", 4_096): "hang",
            ("neuron_fan_out", 8_192): "skipped",
            ("axon_fan_in", 8_191): "pass", ("axon_fan_in", 8_192): "pass",
            ("axon_fan_in", 16_384): "pass",
            ("axon_fan_in", 32_768): "skipped",       # more axons than input_axons passed
            ("neuron_fan_in", 8_159): "pass", ("neuron_fan_in", 8_160): "pass",
            ("neuron_fan_in", 16_384): "pass", ("neuron_fan_in", 32_768): "pass"})
        by = {(r["family"], r["size"]): r for r in rows}
        alias = {c["case"]: c for c in by[("input_axons", 16_385)]["cases"]}
        self.assertEqual(alias["axon 16,384"]["hw"], [])
        self.assertNotEqual(alias["axon 0"]["hw"], alias["axon 0"]["sim"])
        self.assertTrue(alias["axon 0"]["sim"] == alias["axon 0"]["expected"])
        lost = by[("axon_fan_out", 4_097)]["cases"][0]
        self.assertEqual((lost["expected"], lost["sim"], lost["hw"]), (4_097, 4_097, 4_096))
        self.assertIn("Insufficient bit width", by[("neurons", 131_072)]["error"])
        self.assertTrue(by[("neurons", 131_072)]["healthy_after"])
        self.assertTrue(by[("neuron_fan_out", 4_096)]["healthy_after"])
        self.assertIn("32,768 input axons", by[("axon_fan_in", 32_768)]["note"])
        self.assertTrue(any("health check afterwards: passed" in s for s in lines))

        block = cl.limits_block(rows, "planted")
        self.assertIn("limits:", block)
        values = dict(ln.split("#")[0].strip().split(": ") for ln in block.splitlines()[1:])
        self.assertEqual(values, {"total_axons": "16384", "axon_fan_out": "4096",
                                  "neuron_fan_out": "4095", "axon_fan_in": "16384",
                                  "neuron_fan_in": "32768"})
        self.assertIn("every size tried passed", block)        # neuron_fan_in
        self.assertIn("16,385 did not pass", block)
        text = cl.text_summary(rows)
        self.assertIn("passed up to 131,040; 131,072: raised an error while loading", text)
        page = cl.report(rows, "planted")
        for part in ("Chip limits: planted", "For a search config", "neuron fan in",
                     "Every case", "prefers-color-scheme: dark"):
            self.assertIn(part, page)

    def test_stops_when_the_card_fails_its_health_check(self):
        """A hang that leaves the card broken: the health check fails, and
        nothing after it runs. The command line writes the report and the
        JSON anyway and exits 2."""
        cl = load_chip_limits()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(sys.modules, planted_limits_hs_api(wedge=True)):
            out = os.path.join(d, "limits.html")
            code = cl.main(["--only", "neuron_fan_out", "axon_fan_in", "--label", "test",
                            "-o", out])
            with open(os.path.join(d, "limits.json")) as fh:
                saved = json.load(fh)
            with open(out, encoding="utf-8") as fh:
                page = fh.read()
        self.assertEqual(code, 2)
        got = [(r["family"], r["size"], r["status"]) for r in saved["rows"]]
        self.assertEqual(got, [("neuron_fan_out", 4_095, "pass"),
                               ("neuron_fan_out", 4_096, "hang"),
                               ("neuron_fan_out", 8_192, "not_run"),
                               ("axon_fan_in", 8_191, "not_run"),
                               ("axon_fan_in", 8_192, "not_run"),
                               ("axon_fan_in", 16_384, "not_run"),
                               ("axon_fan_in", 32_768, "not_run")])
        self.assertFalse(saved["rows"][1]["healthy_after"])
        self.assertIn("reflash", saved["rows"][2]["note"])
        self.assertIn("Chip limits: test", page)
        self.assertIn("axon_fan_in     search uses   8,191   not measured: the card failed",
                      cl.text_summary(saved["rows"]))

    def test_probe_points_straddle_the_address_widths(self):
        cl = load_chip_limits()
        self.assertEqual(cl.edges(16_385), [0, 1, 255, 256, 8_192, 16_383, 16_384])
        self.assertEqual(cl.edges(131_072), [0, 1, 255, 256, 16_383, 16_384, 32_767, 32_768,
                                             65_535, 65_536, 131_070, 131_071])
        self.assertEqual(cl.edges(3), [0, 1, 2])

    def test_card_check(self):
        """match_simplesim.card_check, shared by both tools: several cards and
        none named, a card that is not there, and a locked card file stop the
        run; a locked channel file only warns."""
        cl = load_chip_limits()
        ms = cl.ms
        with tempfile.TemporaryDirectory() as dev:
            sys_class = os.path.join(dev, "no_sysfs")
            for name in ["adxdma0", "adxdma0ch0", "adxdma2", "adxdma2ch0"]:
                open(os.path.join(dev, name), "w").close()
            err, warn = ms.card_check(None, dev, sys_class)
            self.assertIn("pass --card N", err)
            err, _ = ms.card_check(5, dev, sys_class)
            self.assertIn("no /dev/adxdma5", err)
            self.assertEqual(ms.card_check(2, dev, sys_class), (None, None))
            os.chmod(os.path.join(dev, "adxdma2ch0"), 0o444)
            err, warn = ms.card_check(2, dev, sys_class)
            self.assertIsNone(err)
            self.assertIn("adxdma2ch0", warn)
            os.chmod(os.path.join(dev, "adxdma2"), 0o444)
            err, _ = ms.card_check(2, dev, sys_class)
            self.assertIn("no read/write permission on card 2", err)
            for name in ["adxdma2", "adxdma2ch0"]:
                os.chmod(os.path.join(dev, name), 0o666)


if __name__ == "__main__":
    unittest.main(verbosity=2)
