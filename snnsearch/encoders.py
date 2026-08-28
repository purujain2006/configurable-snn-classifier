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
              The input layer is not itself spiking, which purists dislike; on
              HiAER-Spike it means the first layer's input arrives over axons
              carrying a value rather than an event.

  POISSON     Emit a spike at each timestep with probability proportional to
              intensity. Genuinely spiking end to end and the most biologically
              motivated, but it injects sampling noise that costs accuracy at
              small T, and two runs of the same image differ.

  TEMPORAL    Brighter pixel fires earlier, once. Extremely sparse -- one spike
              per pixel per sample -- so it is the cheapest by SynOps, but it
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

    Input is expected in [0, 1]; values outside are clamped, since a
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
    """The input is already events. Optionally resize, then binarize.

    Averaging a sparse binary frame during resize produces small fractional
    values a spiking input cannot represent, and isolated events vanish. So
    anything non-zero after interpolation becomes a full spike: no event is
    lost, and event density per pixel rises as resolution falls.
    """

    name = "passthrough"
    analog = False

    def __init__(self, T, size=None, binarize=True):
        super().__init__(T, size)
        self.binarize = binarize

    def encode(self, x):
        x = self._resize(x)
        return (x > 0).to(x.dtype) if self.binarize else x

    def describe(self):
        d = super().describe()
        d["binarize"] = self.binarize
        return d


CODINGS = {
    "direct": DirectEncoder,
    "poisson": PoissonEncoder,
    "temporal": TemporalEncoder,
    "passthrough": PassthroughEncoder,
}


def build_encoder(coding, T, resize_to=None, **kwargs):
    """Look up a coding by name, with a message that lists the alternatives."""
    key = (coding or "direct").lower()
    if key not in CODINGS:
        raise ValueError(
            f"unknown coding {coding!r}. Available: {', '.join(sorted(CODINGS))}.\n"
            "  direct      analog value repeated each step (default, static images)\n"
            "  poisson     stochastic spikes proportional to intensity\n"
            "  temporal    one spike per unit, earlier means brighter\n"
            "  passthrough input is already events (DVS)")
    size = (resize_to, resize_to) if isinstance(resize_to, int) and resize_to else resize_to
    return CODINGS[key](T=T, size=size, **kwargs)
