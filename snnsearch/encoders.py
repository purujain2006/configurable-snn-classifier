"""Turning an input into spikes.

This is the piece that decides whether the tool handles anything other than an
event camera. DVS data already arrives as events, so the original code needed
only to resize and binarize. A CIFAR-10 image is a static array of floats and
has to be given a time dimension before a spiking network can read it.

The three standard schemes, and when each is worth using:

  DIRECT      Present the same analog value at every timestep and let the first
              layer's weights convert it into current. No stochasticity, no
              information lost at the input, converges fastest. This is what
              most modern directly-trained SNN work uses and it is the default.
              The input layer is not itself spiking, which purists dislike. On
              HiAER-Spike it means the first layer's input arrives over axons
              carrying a value rather than an event.

  POISSON     Emit a spike at each timestep with probability proportional to
              intensity. It spikes end to end and is closest to biology, but it
              adds sampling noise that costs accuracy at
              small T, and two runs of the same image differ.

  TEMPORAL    Brighter pixel fires earlier, once. Extremely sparse, one spike
              per pixel per sample, so it is the cheapest by SynOps, but it
              is fragile to train and needs T large enough to resolve the
              latency code.

  PASSTHROUGH The input is already a spike train, as with DVS. Optionally
              resize and binarize.

Which one wins is task-dependent and is worth searching rather than assuming,
so `coding` is a config field and can be swept like any other.
"""

from ._torch import _HAS_TORCH, torch, F


class Encoder:
    """Maps a batch to (N, T, C, H, W) spikes or currents.

    Subclasses implement `encode`. Input is (N, C, H, W) for static data or
    already (N, T, C, H, W) for event data.
    """

    name = "base"
    #: True when the output carries analog values rather than 0/1 events.
    analog = False

    def __init__(self, T, size=None):
        self.T = int(T)
        self.size = size          # (H, W) to resize to, or None to leave alone

    def _resize(self, x):
        """x is (..., C, H, W). Bilinear, then callers decide about binarizing."""
        if self.size is None:
            return x
        lead, chw = x.shape[:-3], x.shape[-3:]
        flat = x.reshape(-1, *chw)
        out = F.interpolate(flat, size=self.size, mode="bilinear", align_corners=False)
        return out.reshape(*lead, *out.shape[-3:])

    def encode(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.encode(x)

    def describe(self):
        return {"coding": self.name, "T": self.T, "analog": self.analog,
                "resize_to": self.size}


class DirectEncoder(Encoder):
    """Repeat the analog input at every timestep.

    Nothing is discretized at the input, so accuracy is limited by the network
    rather than by the code. Costs nothing in SynOps terms at the input layer
    because there are no input spikes to count.
    """

    name = "direct"
    analog = True

    def encode(self, x):
        x = self._resize(x)
        if x.dim() == 5:                     # already has time
            return x
        return x.unsqueeze(1).expand(-1, self.T, *x.shape[1:]).contiguous()


class PoissonEncoder(Encoder):
    """Sample a spike per timestep with probability equal to intensity.

    Input is expected in [0, 1]. Values outside are clamped, since a
    probability above 1 would silently saturate instead of erroring.
    """

    name = "poisson"
    analog = False

    def encode(self, x):
        x = self._resize(x).clamp(0.0, 1.0)
        if x.dim() == 5:
            return (torch.rand_like(x) < x).float()
        p = x.unsqueeze(1).expand(-1, self.T, *x.shape[1:])
        return (torch.rand_like(p) < p).float()


class TemporalEncoder(Encoder):
    """Time-to-first-spike: brighter fires earlier, exactly once.

    Intensity 1 fires at t=0, intensity just above 0 fires at t=T-1, and exact
    zero never fires. One spike per pixel per sample makes this the sparsest
    of the three by a wide margin.
    """

    name = "temporal"
    analog = False

    def encode(self, x):
        x = self._resize(x).clamp(0.0, 1.0)
        if x.dim() == 5:                     # collapse time we did not create
            x = x.mean(dim=1)
        # index of the timestep at which each unit fires
        t_fire = ((1.0 - x) * (self.T - 1)).round().long()
        out = torch.zeros(x.shape[0], self.T, *x.shape[1:], device=x.device, dtype=x.dtype)
        out.scatter_(1, t_fire.unsqueeze(1), 1.0)
        out = out * (x > 0).unsqueeze(1).to(out.dtype)      # true zero stays silent
        return out


class PassthroughEncoder(Encoder):
    """The input is already events. Optionally resize, then turn counts into
    values a chip axon can carry.

    input_bits = 1: anything non-zero after interpolation becomes a full
    spike. Averaging a sparse binary frame during resize produces small
    fractions a spike cannot represent, and isolated events would vanish. This
    way no event is lost.

    input_bits = B > 1: each pixel carries its event count, clipped to
    2**B - 1, so a busy pixel and a quiet one are told apart. On chip the count
    travels on B axons, one per bit (hs_export.py). Shrinking sums the counts
    of the pixels merged together, and any non-zero sum counts at least 1.
    """

    name = "passthrough"
    analog = False

    def __init__(self, T, size=None, binarize=True, input_bits=1):
        super().__init__(T, size)
        self.binarize = binarize
        self.input_bits = max(1, int(input_bits))

    def encode(self, x):
        if self.input_bits <= 1:
            x = self._resize(x)
            return (x > 0).to(x.dtype) if self.binarize else x
        top = 2 ** self.input_bits - 1
        if self.size is not None:
            lead, (c, h, w) = x.shape[:-3], x.shape[-3:]
            flat = x.reshape(-1, c, h, w)
            out = F.interpolate(flat, size=self.size, mode="area")
            out = out * (h * w) / (self.size[0] * self.size[1])   # mean -> sum of counts
            x = out.reshape(*lead, *out.shape[-3:])
        return torch.clamp(torch.ceil(x - 1e-6), 0, top)

    def describe(self):
        d = super().describe()
        d["binarize"] = self.binarize
        d["input_bits"] = self.input_bits
        return d


class LevelsEncoder(Encoder):
    """A still image as B-bit levels, the same at every timestep.

    Brightness in [0, 1] is rounded to an integer from 0 to 2**B - 1. The chip
    carries each level on B axons, one per bit, so unlike the analog "direct"
    coding this deploys. B = 1 thresholds brightness at one half.
    """

    name = "levels"
    analog = False

    def __init__(self, T, size=None, input_bits=1):
        super().__init__(T, size)
        self.input_bits = max(1, int(input_bits))

    def encode(self, x):
        top = 2 ** self.input_bits - 1
        x = torch.round(self._resize(x).clamp(0.0, 1.0) * top)
        if x.dim() == 5:
            return x
        return x.unsqueeze(1).expand(-1, self.T, *x.shape[1:]).contiguous()

    def describe(self):
        d = super().describe()
        d["input_bits"] = self.input_bits
        return d


CODINGS = {
    "direct": DirectEncoder,
    "poisson": PoissonEncoder,
    "temporal": TemporalEncoder,
    "passthrough": PassthroughEncoder,
    "levels": LevelsEncoder,
}


def build_encoder(coding, T, resize_to=None, input_bits=1, **kwargs):
    """Look up a coding by name, with a message that lists the alternatives."""
    key = (coding or "direct").lower()
    if key not in CODINGS:
        raise ValueError(
            f"unknown coding {coding!r}. Available: {', '.join(sorted(CODINGS))}.\n"
            "  direct      analog value repeated each step (default, static images)\n"
            "  poisson     stochastic spikes proportional to intensity\n"
            "  temporal    one spike per unit, earlier means brighter\n"
            "  passthrough input is already events (DVS)\n"
            "  levels      still image as B-bit levels, deployable on chip")
    size = (resize_to, resize_to) if isinstance(resize_to, int) and resize_to else resize_to
    if key in ("passthrough", "levels"):
        kwargs["input_bits"] = input_bits
    elif int(input_bits) > 1:
        raise ValueError(f"input_bits={input_bits} needs coding 'passthrough' or "
                         f"'levels'; {key!r} produces 0/1 spikes or analog values")
    return CODINGS[key](T=T, size=size, **kwargs)
