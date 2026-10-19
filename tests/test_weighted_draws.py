"""The weighted-draw mark: weights drawn at their true values, never equalized."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _set(n=40, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.standard_normal((n, 2))
    w = np.exp(-0.5 * ((xy[:, 0] - 0.8) ** 2 + xy[:, 1] ** 2) / 0.3)
    return xy, w


def _by_input(mark, xy, values):
    """Scatter values put back in the input order of the points."""
    offsets = mark.artists["scatter"].get_offsets()
    out = np.empty(len(values))
    for row, (x, y) in enumerate(offsets):
        i = int(np.flatnonzero((xy[:, 0] == x) & (xy[:, 1] == y))[0])
        out[i] = values[row]
    return out


def _mark(xy, w, **kw):
    _, ax = plt.subplots(figsize=(3, 3), dpi=100)
    return ep.weighted_draws(xy, w, ax=ax, **kw)


def test_area_is_proportional_to_weight():
    xy, w = _set()
    mark = _mark(xy, w, max_size=200.0)
    sizes = _by_input(mark, xy, mark.artists["scatter"].get_sizes())
    np.testing.assert_allclose(sizes / sizes.max(), w / w.max())
    assert sizes.max() == pytest.approx(200.0)
    np.testing.assert_allclose(mark.weights(), w / w.sum())


def test_alpha_is_proportional_to_weight_at_one_size():
    xy, w = _set()
    mark = _mark(xy, w, encode="alpha", max_size=50.0)
    assert np.all(mark.artists["scatter"].get_sizes() == 50.0)
    alpha = _by_input(mark, xy, mark.artists["scatter"].get_facecolor()[:, 3])
    np.testing.assert_allclose(alpha / alpha.max(), w / w.max())


def test_tiny_weights_are_not_floored():
    xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    mark = _mark(xy, [1.0, 1e-6], max_size=100.0)
    sizes = _by_input(mark, xy, mark.artists["scatter"].get_sizes())
    np.testing.assert_allclose(sizes, [100.0, 1e-4])


def test_resampling_is_systematic_and_seeded():
    xy, w = _set(n=200)
    a = _mark(xy, w, resample=5)
    b = _mark(xy, w, resample=5)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert a.counts.sum() == 200
    # Systematic resampling: each count is the floor or the ceiling of n * w.
    expected = 200 * w / w.sum()
    assert np.all(a.counts >= np.floor(expected))
    assert np.all(a.counts <= np.ceil(expected))


def test_resampled_set_keeps_the_true_weights_on_one_scale():
    xy, w = _set(n=60)
    mark = _mark(xy, w, resample=1, max_size=120.0)
    sizes = _by_input(mark, xy, mark.artists["scatter"].get_sizes())  # rest: resampled
    end = mark.counts / 60
    np.testing.assert_allclose(mark.weights(), end)
    top = max((w / w.sum()).max(), end.max())
    np.testing.assert_allclose(sizes, 120.0 * end / top)
    assert np.all(sizes[mark.counts == 0] == 0.0)
    assert len(set(np.round(sizes[mark.counts > 0], 9))) > 1  # not equalized
    mark.update(0.0)
    sizes0 = _by_input(mark, xy, mark.artists["scatter"].get_sizes())
    np.testing.assert_allclose(sizes0, 120.0 * (w / w.sum()) / top)
    mark.update(0.25)
    np.testing.assert_allclose(
        _by_input(mark, xy, mark.artists["scatter"].get_sizes()),
        0.75 * sizes0 + 0.25 * sizes,
    )
    assert mark.ess == pytest.approx(1.0 / np.sum((w / w.sum()) ** 2))


def test_heavy_draws_go_down_first():
    xy, w = _set()
    mark = _mark(xy, w)
    sizes = mark.artists["scatter"].get_sizes()
    assert np.all(np.diff(sizes) <= 0.0)


def test_without_resample_update_holds_the_weights():
    xy, w = _set()
    mark = _mark(xy, w)
    before = mark.artists["scatter"].get_sizes().copy()
    mark.update(0.4)
    np.testing.assert_array_equal(mark.artists["scatter"].get_sizes(), before)
    assert mark.counts is None


def test_beat_is_continuous_with_no_jump():
    xy, w = _set(n=30)
    _, ax = plt.subplots(figsize=(3, 3), dpi=100)
    mark = ep.weighted_draws(xy, w, ax=ax, resample=2)
    times = np.arange(31) / 30.0

    def at(t):
        mark.update(ep.ease(t))

    faults = ept.motion_faults(ax.figure, at, times, moves=[mark])
    assert [f for f in faults if f.kind == "jump"] == []


def test_level_dims():
    xy, w = _set()
    mark = _mark(xy, w)
    mark.update(1.0, level=0.5)
    assert np.all(mark.artists["scatter"].get_facecolor()[:, 3] == pytest.approx(0.5))


def test_rejects_bad_inputs():
    _, ax = plt.subplots()
    xy, w = _set()
    with pytest.raises(ValueError, match="shaped"):
        ep.weighted_draws(np.zeros((4, 3)), np.ones(4), ax=ax)
    with pytest.raises(ValueError, match="weights for"):
        ep.weighted_draws(xy, w[:5], ax=ax)
    with pytest.raises(ValueError, match="non-negative"):
        ep.weighted_draws(xy, -w, ax=ax)
    with pytest.raises(ValueError, match="not all zero"):
        ep.weighted_draws(xy, np.zeros_like(w), ax=ax)
    with pytest.raises(ValueError, match="encode"):
        ep.weighted_draws(xy, w, ax=ax, encode="size")
    with pytest.raises(ValueError, match="max_size"):
        ep.weighted_draws(xy, w, ax=ax, max_size=0.0)


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)


@pytest.mark.parametrize(
    "kw", [{}, {"resample": 3}, {"encode": "alpha"}, {"encode": "alpha", "resample": 3}]
)
def test_rest_is_the_still(kw):
    xy, w = _set()
    mark = _mark(xy, w, **kw)
    still = _pixels(mark.ax.figure)
    for u in (0.0, 0.4):
        mark.update(u)
        _pixels(mark.ax.figure)
    mark.update(1.0)
    np.testing.assert_array_equal(_pixels(mark.ax.figure), still)
    fresh = _mark(xy, w, **kw)
    np.testing.assert_array_equal(_pixels(fresh.ax.figure), still)


def test_tiny_weights_are_not_floored_in_alpha():
    xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    mark = _mark(xy, [1.0, 1e-6], encode="alpha")
    alpha = _by_input(mark, xy, mark.artists["scatter"].get_facecolor()[:, 3])
    np.testing.assert_allclose(alpha, [1.0, 1e-6])


def test_a_parent_with_three_offspring_is_three_times_one_with_one():
    xy, w = _set(n=60)
    mark = _mark(xy, w, resample=1, max_size=120.0)
    sizes = _by_input(mark, xy, mark.artists["scatter"].get_sizes())
    ones, threes = sizes[mark.counts == 1], sizes[mark.counts == 3]
    assert len(ones) and len(threes)
    np.testing.assert_allclose(threes, 3.0 * ones[0])
    np.testing.assert_allclose(ones, ones[0])
