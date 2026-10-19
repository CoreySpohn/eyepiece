"""Seeded photon arrivals on an exposure clock, accumulated as count images.

A detector frame is a count image. Given the expected counts per unit
exposure in every pixel and an exposure clock, `arrivals` draws the photons
that land by each clock time as one seeded Poisson process per pixel, so a
clip can show an image developing from its own photons and a re-render
replays exactly the same photons. The arrays stay in NumPy: this is the I/O
edge of a clip, not a model.
"""

import dataclasses
from numbers import Integral

import numpy as np


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
