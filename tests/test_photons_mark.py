"""The photons mark: counts on one scale; the newest arrivals flash, then settle."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import LogNorm

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _board():
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100, facecolor="black")
    ax.set_axis_off()
    return fig, ax


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)


def _rate():
    yy, xx = np.mgrid[-1:1:24j, -1:1:24j]
    return 400.0 * np.exp(-(xx**2 + yy**2) / 0.1) + 0.05


def _channel(n=8):
    return ep.arrivals(_rate(), np.geomspace(0.01, 10.0, n), seed=4)


def test_rest_is_the_plain_still_of_the_final_counts():
    photons = _channel()
    fig, ax = _board()
    mark = ep.photons(photons, ax=ax, cmap="magma")
    for u in (0.0, 0.3, 0.71):
        mark.update(u)
        _pixels(fig)
    mark.update(1.0)
    shown = _pixels(fig)
    fig2, ax2 = _board()
    low = colormaps["magma"](0.0)
    ax2.imshow(
        photons.counts[-1],
        cmap=colormaps["magma"].with_extremes(bad=low, under=low),
        norm=LogNorm(0.5, photons.counts[-1].max()),
        origin="lower",
        interpolation="nearest",
    )
    np.testing.assert_array_equal(shown, _pixels(fig2))
    assert not mark.artists["head"].get_visible()


def test_each_sample_gets_an_equal_window_and_counts_never_decrease():
    photons = _channel(n=5)
    _, ax = _board()
    mark = ep.photons(photons, ax=ax)
    assert [mark.sample(u) for u in (0.0, 0.01, 0.25, 0.26, 0.74, 0.76, 1.0)] == [
        0, 1, 1, 2, 3, 4, 4,
    ]  # fmt: skip
    last = np.zeros_like(photons.counts[0])
    for u in np.linspace(0.0, 1.0, 41):
        mark.update(u)
        assert mark.sample() == mark.sample(u)
        data = np.asarray(mark.artists["image"].get_array())
        np.testing.assert_array_equal(data, photons.counts[mark.sample(u)])
        assert np.all(data >= last)
        last = data
    assert mark.steps(np.linspace(0.0, 1.0, 11)) == [1, 3, 6, 8]


def test_head_glows_where_arrivals_raised_the_display_and_fades_in_its_window():
    rate = np.zeros((4, 4))
    rate[0, 0] = 1e6  # steady and bright: barely rises on a log scale
    rate[3, 3] = 0.5  # sparse: a first photon is a big rise
    for seed in range(50):  # a seed whose sparse pixel gets its first photon late
        photons = ep.arrivals(rate, [1.0, 2.0, 3.0, 4.0], seed=seed)
        firsts = [
            i
            for i in range(1, photons.n)
            if photons.counts[i - 1, 3, 3] == 0 and photons.counts[i, 3, 3] > 0
        ]
        if firsts:
            k = firsts[0]
            break
    _, ax = _board()
    mark = ep.photons(photons, ax=ax)
    head = mark.artists["head"]
    window = 1.0 / (photons.n - 1)
    mark.update((k - 1) * window + 1e-6)
    alpha = np.asarray(head.get_array())[..., 3]
    assert head.get_visible()
    assert alpha[3, 3] == pytest.approx(1.0, abs=1e-4)
    assert alpha[0, 0] < 0.25  # its counts doubled: a dim glow, not a flash
    assert np.all(alpha[photons.new(k) == 0] == 0.0)
    mark.update((k - 0.5) * window)
    assert np.asarray(head.get_array())[3, 3, 3] == pytest.approx(0.5)
    mark.update((k - 0.001) * window)
    assert np.asarray(head.get_array())[3, 3, 3] < 0.01
    mark.update(k * window)
    assert not head.get_visible()


def test_start_is_settled_so_a_hold_before_the_run_is_still():
    photons = ep.arrivals(_rate(), np.geomspace(1.0, 10.0, 6), seed=4)
    _, ax = _board()
    mark = ep.photons(photons, ax=ax)
    mark.update(0.0)
    assert not mark.artists["head"].get_visible()
    np.testing.assert_array_equal(
        np.asarray(mark.artists["image"].get_array()), photons.counts[0]
    )


def test_level_dims_image_and_head():
    _, ax = _board()
    mark = ep.photons(_channel(), ax=ax)
    mark.update(0.5, level=0.25)
    assert mark.artists["image"].get_alpha() == pytest.approx(0.25)
    assert np.asarray(mark.artists["head"].get_array())[..., 3].max() <= 0.25
    mark.update(1.0)
    assert mark.artists["image"].get_alpha() is None


def test_draws_are_the_only_jumps():
    photons = _channel(n=6)
    fig, ax = _board()
    mark = ep.photons(photons, ax=ax, cmap="magma")
    times = np.arange(46) / 30.0  # 1.5 s at the video rate

    def at(t):
        mark.update(t / 1.5)

    draws = [times[i] for i in mark.steps(times / 1.5)]
    assert len(draws) == photons.n - 1
    clean = ept.motion_faults(fig, at, times, moves=[mark], steps=draws)
    assert clean == []
    undeclared = ept.motion_faults(fig, at, times, moves=[mark])
    assert any(f.kind == "jump" for f in undeclared)


def test_rejects_bad_inputs():
    _, ax = _board()
    with pytest.raises(TypeError):
        ep.photons(np.ones((3, 3)), ax=ax)
    with pytest.raises(ValueError, match="2D"):
        ep.photons(ep.arrivals(np.ones(5), [1.0], seed=0), ax=ax)
    with pytest.raises(ValueError, match="norm"):
        ep.photons(_channel(), ax=ax, norm="sqrt")
    with pytest.raises(ValueError, match="settle"):
        ep.photons(_channel(), ax=ax, settle=0.0)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0.25, "0.25 s"),
        (5.0, "5.0 s"),
        (45.0, "45 s"),
        (90.0, "1.5 min"),
        (2.8 * 3600, "2.8 h"),
        (28 * 3600, "28 h"),
        (11.6 * 86400, "11.6 d"),
        (150 * 86400, "150 d"),
    ],
)
def test_exposure_text_reads_a_log_clock(seconds, text):
    assert ep.exposure_text(seconds) == text


def test_exposure_text_pads_to_fixed_slots():
    assert ep.exposure_text(28 * 3600, pad=True) == "  28 h"
    assert ep.exposure_text(11.6 * 86400, pad=True) == "11.6 d"
    with pytest.raises(ValueError):
        ep.exposure_text(-1.0)
