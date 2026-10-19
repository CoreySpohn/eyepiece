"""The aperture readout is read off the photons mark and its S/N is calibrated."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _scene():
    yy, xx = np.mgrid[-8:8:32j, -8:8:32j]
    leak = 50.0 * np.exp(-(xx**2 + yy**2) / 8.0) + 0.2
    planet = 3.0 * np.exp(-((xx - 3.5) ** 2 + (yy - 1.5) ** 2) / 0.5)
    ring = np.hypot(xx - 3.5, yy - 1.5) <= 1.0
    return leak, planet, ring


def test_counts_are_the_drawn_pixels_of_the_sample_shown():
    leak, planet, ring = _scene()
    channel = ep.arrivals(
        leak + planet, np.r_[0.0, np.geomspace(1.0, 100.0, 9)], seed=3
    )
    _, ax = plt.subplots()
    mark = ep.photons(channel, ax=ax)
    for u in (0.0, 0.3, 0.62, 1.0):
        mark.update(u)
        read = ep.aperture_counts(mark, ring, leak=leak)
        shown = np.asarray(mark.artists["image"].get_array())
        assert read.sample == mark.sample()
        assert read.counts == int(shown[ring].sum())
        assert read.exposure == channel.exposure[read.sample]
        assert read.expected == pytest.approx(
            (leak + planet)[ring].sum() * read.exposure
        )
        assert read.leak == pytest.approx(leak[ring].sum() * read.exposure)
        assert read.signal == pytest.approx(read.counts - read.leak)
        assert read.pixels == int(ring.sum())
    assert ep.aperture_counts(mark, ring, leak=leak, sample=2).sample == 2


def test_snr_follows_the_stated_estimator():
    leak, planet, ring = _scene()
    channel = ep.arrivals(leak + planet, [0.0, 10.0, 1e3], seed=5)
    _, ax = plt.subplots()
    mark = ep.photons(channel, ax=ax)
    mark.update(1.0)
    read = ep.aperture_counts(mark, ring, leak=leak)
    noise = np.sqrt(read.expected)
    assert read.snr == pytest.approx((read.counts - read.leak) / noise)
    assert read.snr_expected == pytest.approx(planet[ring].sum() * 1e3 / noise)
    start = ep.aperture_counts(mark, ring, leak=leak, sample=0)
    assert (start.counts, start.snr, start.snr_expected) == (0, 0.0, 0.0)


def test_draw_snr_scatters_about_the_expected_with_unit_spread():
    leak, planet, ring = _scene()
    _, ax = plt.subplots()
    z = []
    for seed in range(400):
        mark = ep.photons(ep.arrivals(leak + planet, [5.0], seed=seed), ax=ax)
        read = ep.aperture_counts(mark, ring, leak=leak)
        z.append(read.snr - read.snr_expected)
        mark.artists["image"].remove()
        mark.artists["head"].remove()
    z = np.asarray(z)
    assert abs(z.mean()) < 5.0 / np.sqrt(len(z))
    assert abs(z.std() - 1.0) < 0.15


def test_rejects_bad_inputs():
    leak, planet, ring = _scene()
    _, ax = plt.subplots()
    mark = ep.photons(ep.arrivals(leak + planet, [1.0, 2.0], seed=0), ax=ax)
    with pytest.raises(TypeError):
        ep.aperture_counts(ep.arrivals(leak, [1.0], seed=0), ring, leak=leak)
    with pytest.raises(ValueError, match="boolean"):
        ep.aperture_counts(mark, ring.astype(int), leak=leak)
    with pytest.raises(ValueError, match="boolean"):
        ep.aperture_counts(mark, ring[:4], leak=leak)
    with pytest.raises(ValueError, match="empty"):
        ep.aperture_counts(mark, np.zeros_like(ring), leak=leak)
    with pytest.raises(ValueError, match="non-negative"):
        ep.aperture_counts(mark, ring, leak=-leak)
    with pytest.raises(ValueError, match="exceeds"):
        ep.aperture_counts(mark, ring, leak=leak + 2.0 * planet + 1.0)
