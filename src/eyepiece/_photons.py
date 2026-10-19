"""Seeded photon arrivals on an exposure clock, accumulated as count images.

A detector frame is a count image. Given the expected counts per unit
exposure in every pixel and an exposure clock, `arrivals` draws the photons
that land by each clock time as one seeded Poisson process per pixel, so a
clip can show an image developing from its own photons and a re-render
replays exactly the same photons. The arrays stay in NumPy: this is the I/O
edge of a clip, not a model.
"""

import dataclasses
from collections.abc import Callable
from numbers import Integral

import numpy as np

from eyepiece._result import MotionContract, PlotResult


@dataclasses.dataclass(frozen=True, eq=False)
class Arrivals:
    """Cumulative photon counts at each time of an exposure clock.

    Attributes:
        rate: Expected counts per unit exposure, one value per pixel.
        exposure: The exposure clock, one time per sample, non-decreasing.
        counts: Integer counts shaped ``(n, *rate.shape)``: the photons that
            arrived in each pixel by ``exposure[k]``.
        seed: The seed the photons were drawn with.
    """

    rate: np.ndarray
    exposure: np.ndarray
    counts: np.ndarray
    seed: int

    @property
    def n(self):
        """The number of clock samples."""
        return len(self.exposure)

    def new(self, k):
        """The photons that arrived in sample `k`, since sample ``k - 1``.

        Args:
            k: A sample index; negative indices count from the end.

        Returns:
            An integer array shaped like `rate`; sample 0 returns its counts.
        """
        k = range(self.n)[k]
        return self.counts[k] - self.counts[k - 1] if k else self.counts[0].copy()

    def expected(self, k):
        """The expected counts by sample `k`: rate times exposure."""
        return self.rate * self.exposure[k]


def arrivals(rate, exposure, *, seed) -> Arrivals:
    """Draw seeded Poisson photon arrivals on an exposure clock.

    Each pixel is an independent Poisson process at its `rate`. The counts at
    every clock time are one realization of those processes, so they never
    decrease, the counts that arrive between two clock times are Poisson with
    mean rate times the elapsed exposure and independent of the earlier ones,
    and the expected counts at time ``t`` are ``rate * t``. The draw is fixed
    by `seed`: the same rate, clock and seed give the same photons on every
    call, which makes the seed part of a clip's identity.

    The counts at the last clock time are drawn first, from the seed alone,
    and the earlier counts are split from them (a binomial thinning, exact
    for a Poisson process). So the final image depends only on the rate, the
    final exposure and the seed, not on how many samples the clock has: the
    last frame of a clip equals the still drawn at its final exposure, at any
    frame rate.

    Args:
        rate: Expected counts per unit exposure, a non-negative finite array
            of any shape (an image for a detector frame).
        exposure: The exposure clock, a non-empty one-dimensional array of
            non-negative finite times, non-decreasing (a log clock is
            ``np.geomspace``; a repeated time holds the counts).
        seed: A non-negative integer seed. There is no default, so a clip
            always records the photons it shows.

    Returns:
        An `Arrivals` with integer ``counts`` shaped ``(len(exposure),
        *rate.shape)``. Memory is one 8-byte count per pixel per sample.

    Raises:
        TypeError: If `seed` is not an integer.
        ValueError: If `rate` is negative or not finite, `exposure` is empty,
            not one-dimensional, negative, not finite or decreasing, or
            `seed` is negative.

    Example::

        clock = np.geomspace(1.0, 1e5, 30)
        photons = ep.arrivals(star + 1e-8 * planet, clock, seed=7)
        photons.counts[-1]  # the image at 1e5 s
    """
    rate = np.asarray(rate, dtype=float)
    if not np.all(np.isfinite(rate)) or np.any(rate < 0.0):
        raise ValueError("arrivals rate must be finite and non-negative")
    clock = np.asarray(exposure, dtype=float)
    if clock.ndim != 1 or clock.size == 0:
        raise ValueError(
            f"arrivals exposure must be a non-empty 1D array, got shape {clock.shape}"
        )
    if not np.all(np.isfinite(clock)) or np.any(clock < 0.0):
        raise ValueError("arrivals exposure must be finite and non-negative")
    if np.any(np.diff(clock) < 0.0):
        raise ValueError("arrivals exposure must be non-decreasing")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"arrivals seed must be an integer, got {seed!r}")
    if seed < 0:
        raise ValueError(f"arrivals seed must be non-negative, got {seed}")
    rng = np.random.default_rng(int(seed))
    counts = np.empty((clock.size, *rate.shape), dtype=np.int64)
    counts[-1] = rng.poisson(rate * clock[-1])
    for k in range(clock.size - 2, -1, -1):
        later = clock[k + 1]
        keep = clock[k] / later if later > 0.0 else 0.0
        counts[k] = rng.binomial(counts[k + 1], keep)
    return Arrivals(rate=rate, exposure=clock, counts=counts, seed=int(seed))


# Exposure clock units: the largest that keeps the value at least 1, with
# hours kept up to two days so a night reads in hours.
_CLOCK_UNITS = (("s", 1.0, 60.0), ("min", 60.0, 3600.0), ("h", 3600.0, 172800.0))
_DAY_S = 86400.0
_CLOCK_SLOTS = 4


def exposure_text(seconds, *, pad=False):
    """Format an exposure time for a log clock: ``2.8 h``, ``28 h``, ``11.6 d``.

    The unit is the largest of seconds, minutes, hours and days that keeps the
    value at least 1, with hours kept up to two days. Below 10 the value has
    one decimal and from 10 up none, so a clock that runs over decades keeps
    two or three significant figures; days keep one decimal up to 100. Below
    one second the value has two significant figures.

    Args:
        seconds: A non-negative finite exposure time in seconds.
        pad: Right-align the number in four character slots, so a clock in a
            monospaced font keeps its unit in place while the digits run.

    Returns:
        The text, number and unit separated by a space.

    Raises:
        ValueError: If `seconds` is negative or not finite.
    """
    t = float(seconds)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError(f"exposure_text needs a non-negative time, got {seconds!r}")
    unit, scale, decimals_below = "d", _DAY_S, 100.0
    for name, size, upper in _CLOCK_UNITS:
        if t < upper:
            unit, scale, decimals_below = name, size, 10.0
            break
    value = t / scale
    if value < 1.0 and unit == "s":
        number = f"{value:.2g}"
    elif round(value, 1) < decimals_below:
        number = f"{value:.1f}"
    else:
        number = f"{value:.0f}"
    if pad:
        number = number.rjust(_CLOCK_SLOTS)
    return f"{number} {unit}"


# A head pixel is fully bright when its arrivals raise its displayed value by
# a quarter of the color scale; smaller rises glow in proportion.
_HEAD_GAIN = 4.0

# The default log scale starts at half a count, so one photon shows above
# an empty pixel.
_LOG_FLOOR = 0.5

# Tolerance on u * n, so a knob landing on a window edge in floating point
# shows the new sample on the frame the edge falls on.
_WINDOW_TOL = 1e-9


@dataclasses.dataclass(frozen=True, eq=False)
class _PhotonsResult(PlotResult):
    """A `PlotResult` that also reports the sample shown and its draw times."""

    sample: Callable | None = None
    steps: Callable | None = None
    arrivals: Arrivals | None = None


def _displayed(norm, counts):
    """Counts mapped through the norm into [0, 1], zero counts at 0."""
    shown = np.ma.filled(norm(np.ma.masked_less_equal(counts, 0)), 0.0)
    return np.clip(np.asarray(shown, dtype=float), 0.0, 1.0)


def photons(
    arrivals,
    *,
    ax=None,
    extent=None,
    cmap=None,
    norm="log",
    origin="lower",
    head_color=None,
    settle=1.0,
    zorder=None,
) -> PlotResult:
    """Draw photon arrivals as a count image whose newest photons flash.

    The mark of a detector integrating: the image is the counts by the
    current clock sample, drawn pixel for pixel (nearest-neighbor, never
    smoothed) on one fixed color scale, so a pixel brightens only because
    photons landed in it. The photons that arrived in the current sample are
    a head: each pixel they land in glows in the head color, as bright as the
    rise they gave its displayed value (fully bright for a quarter of the
    scale, so a steady bright pixel barely glows), and a pixel's first photon
    always flashes fully. The glow fades as the sample is shown, settling
    into the image before the next sample's photons arrive.

    Motion contract (exchangeable draws): only pixel values change; the
    image's extent, scale and colormap never move. ``update(u)`` gives each
    clock sample after the first an equal window of `u`, so the image
    changes in a cut at the start of each window, the draw, and is otherwise
    still except for the fading head.
    The counts shown never decrease. The cuts are by design: declare the
    frames ``steps`` returns to `eyepiece.testing.check_clip` (or their
    times to ``motion_faults``).
    At ``u = 0`` the first sample shows settled, with no glow, so a hold
    before the run is still (start the clock at zero exposure for an empty
    detector); at ``u = 1``, the rest state, the last sample shows settled,
    which is the plain still of the final counts on the same scale. Build
    the arrivals on the clip's own clock: one sample per draw, at a cadence
    the eye can follow (three to six draws per second for a run of
    exchangeable draws).

    Args:
        arrivals: An `Arrivals` with a two-dimensional rate.
        ax: Axes to draw on, or None for a new figure.
        extent: The image extent ``(left, right, bottom, top)`` in data
            coordinates, or None for pixel coordinates.
        cmap: A colormap or its name; None uses the image default. Zero
            counts take the colormap's lowest color.
        norm: ``"log"`` (the default) for counts from half a count, so one
            photon shows above an empty pixel, to the largest final count,
            ``"linear"`` for 0 to it, or a `Normalize`. The scale is fixed
            for the whole clip.
        origin: The image origin, ``"lower"`` (row 0 at the bottom) or
            ``"upper"``.
        head_color: Color of the arriving photons. None uses the text color,
            read now.
        settle: Fraction of a sample's window over which the head fades, in
            (0, 1].
        zorder: Drawing order of the image; the head draws just above it.

    Returns:
        A `PlotResult` with ``artists["image"]`` (the count image, which a
        colorbar can read), ``artists["head"]`` (the head layer), the two
        declared in ``contract.moves``, and ``update(u, level=1.0)``, where
        ``level`` dims both. Its ``sample(u=None)`` returns the index of the
        sample shown (at the last `u` set when None), for a readout read off
        the mark; ``steps(knob)``, given the knob's value `u` at every frame
        of a clip, returns the indices of the frames that show a new sample
        (the draws, as `check_clip` takes them; for `motion_faults` use the
        times of those frames); and
        ``arrivals`` is the channel drawn.

    Raises:
        TypeError: If `arrivals` is not an `Arrivals`.
        ValueError: If the rate is not two-dimensional, `norm` or `origin`
            is unknown, or `settle` is outside (0, 1].

    Example::

        clock = np.geomspace(1.0, 1e5, 25)
        mark = ep.photons(ep.arrivals(rate, clock, seed=7), ax=ax, cmap="magma")
        for u in np.linspace(0.0, 1.0, 300):
            mark.update(u)
            rec.frame()
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm, Normalize, to_rgb

    from eyepiece import _style

    if not isinstance(arrivals, Arrivals):
        raise TypeError(f"photons needs Arrivals, got {type(arrivals).__name__}")
    if arrivals.rate.ndim != 2:
        raise ValueError(
            f"photons needs a 2D rate (an image), got shape {arrivals.rate.shape}"
        )
    if origin not in ("lower", "upper"):
        raise ValueError(f"photons origin must be 'lower' or 'upper', got {origin!r}")
    settle = float(settle)
    if not 0.0 < settle <= 1.0:
        raise ValueError(f"photons settle must be in (0, 1], got {settle}")
    top = float(max(arrivals.counts[-1].max(), 1))
    if isinstance(norm, Normalize):
        scale = norm
    elif norm == "log":
        scale = LogNorm(vmin=_LOG_FLOOR, vmax=top)
    elif norm == "linear":
        scale = Normalize(vmin=0.0, vmax=top)
    else:
        raise ValueError(
            f"photons norm must be 'log', 'linear' or a Normalize, got {norm!r}"
        )
    if ax is None:
        _, ax = plt.subplots()
    base = (
        colormaps[cmap or plt.rcParams["image.cmap"]]
        if not hasattr(cmap, "N")
        else cmap
    )
    low = base(0.0)
    colors = base.with_extremes(bad=low, under=low)
    counts = arrivals.counts
    n = arrivals.n
    kw = {
        "extent": extent,
        "origin": origin,
        "interpolation": "nearest",
        "resample": False,
    }
    image = ax.imshow(counts[-1], cmap=colors, norm=scale, **kw)
    if zorder is not None:
        image.set_zorder(zorder)
    ink = np.asarray(to_rgb(_style.neutral(1.0) if head_color is None else head_color))
    glow = np.zeros((*arrivals.rate.shape, 4))
    glow[..., :3] = ink
    head = ax.imshow(glow, **kw)
    head.set_zorder(image.get_zorder() + 0.01)
    head.set_visible(False)
    ax.set_aspect(image.axes.get_aspect())
    alpha0 = image.get_alpha()
    state = {"u": 1.0}

    def where(u):
        # Sample 0 is the state before the first draw; sample k >= 1 owns the
        # window ((k - 1) / (n - 1), k / (n - 1)] and glows from its start.
        x = u * (n - 1)
        if n == 1 or x <= _WINDOW_TOL:
            return 0, 1.0
        k = min(int(np.floor(x - _WINDOW_TOL)) + 1, n - 1)
        return k, min(x - (k - 1), 1.0)

    def sample(u=None):
        u = state["u"] if u is None else min(max(float(u), 0.0), 1.0)
        return where(u)[0]

    def update(u, level=1.0):
        u = min(max(float(u), 0.0), 1.0)
        level = min(max(float(level), 0.0), 1.0)
        state["u"] = u
        k, phase = where(u)
        image.set_data(counts[k])
        base_alpha = 1.0 if alpha0 is None else alpha0
        image.set_alpha(alpha0 if level == 1.0 else level * base_alpha)
        fade = max(0.0, 1.0 - phase / settle)
        if fade <= 0.0:
            head.set_visible(False)
            return
        before = counts[k - 1] if k else np.zeros_like(counts[k])
        rise = _displayed(scale, counts[k]) - _displayed(scale, before)
        flash = np.clip(_HEAD_GAIN * rise, 0.0, 1.0)
        flash[(before == 0) & (counts[k] > 0)] = 1.0
        glow[..., 3] = flash * fade * level
        head.set_data(glow)
        head.set_visible(True)

    def steps(knob):
        shown = [sample(u) for u in knob]
        return [i for i in range(1, len(shown)) if shown[i] != shown[i - 1]]

    return _PhotonsResult(
        ax=ax,
        artists={"image": image, "head": head},
        update=update,
        contract=MotionContract(moves=(image, head)),
        sample=sample,
        steps=steps,
        arrivals=arrivals,
    )
