"""Augmentation, the TET loss and seeding, on a tiny synthetic dataset (CPU).

    python tests/test_training_options.py

Augmented inputs must still be inputs the chip can receive. A seeded run must
repeat exactly. And a network trained with every option the search can pick
must still agree with the simulator clip for clip.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from snnsearch import runconfig
from snnsearch.augment import augment_batch
from snnsearch.pipeline import prepare
from snnsearch.train import run_training, forward_over_time


def aug(**kw):
    base = dict(augment=True, aug_shift=0.0, aug_cutout=0.0, aug_drop=0.0, aug_flip=False)
    return SimpleNamespace(**{**base, **kw})


class AugmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # integer levels of a 2-bit input, the same image at every step
        img = torch.randint(0, 4, (6, 1, 2, 12, 12)).float()
        self.x = img.expand(6, 5, 2, 12, 12).clone()

    def test_off_is_identity(self):
        self.assertTrue(torch.equal(augment_batch(self.x, aug(augment=False, aug_drop=0.5)), self.x))

    def test_outputs_stay_chip_inputs(self):
        y = augment_batch(self.x, aug(aug_shift=0.2, aug_cutout=0.3, aug_drop=0.2, aug_flip=True),
                          hflip_ok=True)
        self.assertEqual(y.shape, self.x.shape)
        self.assertTrue(torch.equal(y, y.round()))
        self.assertTrue(bool(((y >= 0) & (y <= 3)).all()))
        self.assertLessEqual(float(y.sum()), float(self.x.sum()))

    def test_same_transform_at_every_step(self):
        y = augment_batch(self.x, aug(aug_shift=0.25, aug_cutout=0.3, aug_flip=True), hflip_ok=True)
        self.assertTrue(torch.equal(y, y[:, :1].expand_as(y)))

    def test_flip_only_where_the_dataset_allows_it(self):
        self.assertTrue(torch.equal(augment_batch(self.x, aug(aug_flip=True), hflip_ok=False), self.x))
        torch.manual_seed(1)
        y = augment_batch(self.x, aug(aug_flip=True), hflip_ok=True)
        flipped = [not torch.equal(y[i], self.x[i]) for i in range(len(y))]
        self.assertTrue(any(flipped))
        for i, f in enumerate(flipped):
            if f:
                self.assertTrue(torch.equal(y[i], self.x[i].flip(-1)))

    def test_drop_keeps_spikes_binary(self):
        spikes = (torch.rand(4, 5, 2, 8, 8) < 0.3).float()
        y = augment_batch(spikes, aug(aug_drop=0.5))
        self.assertTrue(bool(((y == 0) | (y == 1)).all()))
        self.assertTrue(bool((y <= spikes).all()))


def config():
    return runconfig.load(overrides={
        "dataset": {"module": os.path.join(ROOT, "examples", "synthetic_data.py")},
        "encoding": {"coding": "levels"},
        "search": {"batch_size": 8, "num_workers": 0},
        "report": {"html": False},
    })


EVERYTHING = {
    "depth": 2, "k_0": 3, "ch_0": 6, "ds_0": "pool", "pool_0": 1,
    "k_1": 3, "ch_1": 6, "ds_1": "stride", "stride_1": 2,
    "fc_layers": 0, "final_reduction": "gap", "T": 4, "resize_to": 16, "epochs": 2,
    "input_bits": 2, "signed_spikes": True, "tau_mode": "channel", "tau": 4,
    "trainable_threshold": True, "bias_mode": "neuron", "pool_type": "max",
    "loss": "tet", "use_augment": True, "aug_shift": 0.1, "aug_cutout": 0.2,
    "aug_drop": 0.1, "dropout_rate": 0.1, "optimizer": "adam", "lr": 3e-3,
    "scheduler": "cosine", "seed": 5,
}


class ExactArithmeticTests(unittest.TestCase):
    def test_tf32_off_inside_and_restored_after(self):
        """GPU convolutions in TF32 flip spikes the chip would not, so the
        deployed network is measured with TF32 off."""
        from snnsearch.train import exact_arithmetic
        b = torch.backends
        before = (b.cudnn.allow_tf32, b.cuda.matmul.allow_tf32)
        b.cudnn.allow_tf32 = b.cuda.matmul.allow_tf32 = True
        try:
            with exact_arithmetic():
                self.assertFalse(b.cudnn.allow_tf32 or b.cuda.matmul.allow_tf32)
            self.assertTrue(b.cudnn.allow_tf32 and b.cuda.matmul.allow_tf32)
        finally:
            b.cudnn.allow_tf32, b.cuda.matmul.allow_tf32 = before


class RatePenaltyTests(unittest.TestCase):
    def rate(self, drive, step_mode):
        from snnsearch.neuron import HardwareLIFNode
        from snnsearch.train import _RateProbe
        x = torch.full((4, 1, 2), float(drive)).requires_grad_()
        # the last spiking layer is the prediction and is never penalized
        net = torch.nn.Sequential(HardwareLIFNode(signed=True, step_mode=step_mode),
                                  HardwareLIFNode(step_mode=step_mode))
        probe = _RateProbe(net)
        if step_mode == "m":
            net(x)
        else:
            for t in range(x.shape[0]):
                net(x[t])
        rate = probe.mean()
        rate.backward()
        probe.detach()
        self.assertIsNone(net[0]._activity)
        return float(rate), float(x.grad.sum())

    def test_a_negative_spike_costs_what_a_positive_one_does(self):
        """A signed unit sends +1 or -1, and both are a spike on chip. The
        penalty used to average the output, so a -1 lowered it and the
        network was rewarded for negative spikes."""
        for step_mode in ("s", "m"):
            with self.subTest(step_mode=step_mode):
                up, grad_up = self.rate(2.0, step_mode)
                down, grad_down = self.rate(-2.0, step_mode)
                self.assertGreater(up, 0)
                self.assertAlmostEqual(down, up, places=6)
                # descent moves the input toward zero either way: fewer spikes
                self.assertGreater(grad_up, 0)
                self.assertLess(grad_down, 0)


class TrainingTests(unittest.TestCase):
    def train(self, flat):
        with redirect_stdout(io.StringIO()):
            _b, _e, loaders, spec = prepare(config(), flat)
            res = run_training(spec, loaders=loaders, device=torch.device("cpu"))
        return spec, res

    def test_tet_steps_line_up_with_the_rate(self):
        _b, _e, loaders, spec = None, None, None, None
        with redirect_stdout(io.StringIO()):
            _b, _e, loaders, spec = prepare(config(), EVERYTHING)
        from snnsearch.model import build_model
        net = build_model(spec).eval()
        x, _y, _l = next(iter(loaders[0]))
        with torch.no_grad():
            rate, steps = forward_over_time(net, x, per_step=True)
            again = forward_over_time(net, x)
        self.assertEqual(tuple(steps.shape), (4, x.shape[0], 2))
        self.assertTrue(torch.equal(rate, again))

    def test_every_option_trains_repeats_and_matches_the_simulator(self):
        spec, a = self.train(EVERYTHING)
        self.assertEqual(spec["train"].loss, "tet")
        self.assertTrue(spec["train"].augment and spec["neuron"].signed_spikes)
        self.assertEqual(spec["input"].input_bits, 2)
        self.assertEqual(a["sim_disagreements"], 0, a.get("sim_reason"))
        self.assertFalse(a["sim_reason"])
        _, b = self.train(EVERYTHING)
        self.assertEqual(a["hw_val_accuracy"], b["hw_val_accuracy"])
        sa, sb = a["hw_net"].state_dict(), b["hw_net"].state_dict()
        self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa))
        _, c = self.train({**EVERYTHING, "seed": 6})
        sc = c["hw_net"].state_dict()
        self.assertFalse(all(torch.equal(sa[k], sc[k]) for k in sa))


if __name__ == "__main__":
    unittest.main(verbosity=2)
