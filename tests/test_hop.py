"""The hop mover: exchangeable draws in turn, ghosts that dim, a converging tally."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _draws(n=12, m=40, seed=3):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, m)
    a = 1.0 + 0.1 * rng.standard_normal(n)
    b = 0.6 + 0.1 * rng.standard_normal(n)
    return np.stack(
        [
            np.column_stack([a[i] * np.cos(theta), b[i] * np.sin(theta)])
            for i in range(n)
        ]
    )


def _board(tally=True):
    if tally:
        fig, (ax, panel) = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
        return fig, ax, panel
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    return fig, ax, None


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)


def _hop(n=12, **kw):
    draws = _draws(n)
    hits = draws[:, 0, 0] > 1.0
    fig, ax, panel = _board()
    mark = ep.hop(draws, ax=ax, tally=hits, tally_ax=panel, ref=0.5, **kw)
    return fig, mark, draws, hits


def test_rest_is_the_still_of_the_run():
    fig, mark, _, _ = _hop(tally_fmt=lambda e, lo, hi, k: f"{e:.0%} of {k}")
    still = _pixels(fig)
    for u in (0.0, 0.31, 0.77):
        mark.update(u)
        _pixels(fig)
    mark.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), still)
    fig2, _, _, _ = _hop(tally_fmt=lambda e, lo, hi, k: f"{e:.0%} of {k}")
    np.testing.assert_array_equal(_pixels(fig2), still)


def test_each_draw_gets_an_equal_window_at_the_declared_rate():
    _, mark, draws, _ = _hop(n=4, rate=5.0)
    assert mark.n == 4
    assert mark.duration == pytest.approx(0.8)
    us = (0.0, 0.1, 0.25, 0.26, 0.5, 0.74, 0.76, 1.0)
    assert [mark.draw(u) for u in us] == [1, 1, 1, 2, 2, 3, 4, 4]
    assert mark.steps(np.linspace(0.0, 1.0, 9)) == [3, 5, 7]
    mark.update(0.6)
    np.testing.assert_allclose(
        np.column_stack(mark.artists["line"].get_data()), draws[2]
    )


@pytest.mark.parametrize("rate", [0.0, -1.0, 8.5, 12.0])
def test_rate_outside_the_band_raises(rate):
    _, ax, _ = _board(tally=False)
    with pytest.raises(ValueError, match="rate"):
        ep.hop(_draws(), ax=ax, rate=rate)


@pytest.mark.parametrize("rate", [1.0, 2.5, 6.5, 7.5])
def test_rate_outside_the_readable_band_warns(rate):
    _, ax, _ = _board(tally=False)
    with pytest.warns(UserWarning, match="separate draws"):
        ep.hop(_draws(), ax=ax, rate=rate)


@pytest.mark.parametrize("rate", [3.0, 4.0, 6.0])
def test_rate_inside_the_readable_band_is_quiet(rate, recwarn):
    _, ax, _ = _board(tally=False)
    ep.hop(_draws(), ax=ax, rate=rate)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_ghosts_are_bounded_and_dim_with_age():
    _, ax, _ = _board(tally=False)
    draws = _draws(n=10)
    mark = ep.hop(draws, ax=ax, ghosts=3, ghost_alpha=0.6)
    pool = mark.artists["lines"]
    assert len(pool) == 3
    mark.update(0.15)  # draw 2: one ghost, draw 1
    assert [g.get_visible() for g in pool] == [True, False, False]
    np.testing.assert_allclose(np.column_stack(pool[0].get_data()), draws[0])
    mark.update(0.75)  # draw 8: ghosts are draws 7, 6, 5
    assert all(g.get_visible() for g in pool)
    for age, ghost in enumerate(pool, start=1):
        np.testing.assert_allclose(np.column_stack(ghost.get_data()), draws[7 - age])
    alphas = [g.get_alpha() for g in pool]
    # The newest ghost is at the bound, and each older one strictly dimmer
    # but still visible; the exact ladder is free.
    assert alphas[0] == pytest.approx(0.6)
    assert all(a > b for a, b in itertools.pairwise(alphas))
    assert alphas[-1] > 0.0
    assert mark.artists["line"].get_zorder() > max(g.get_zorder() for g in pool)


def test_counter_shows_the_draw_index():
    _, mark, _, _ = _hop(n=12)

    def shown():
        return "".join(t.get_text() for t in mark.artists["text"])

    mark.update(0.0)
    assert shown().split() == ["draw", "1", "of", "12"]
    mark.update(0.5)
    assert shown().split() == ["draw", "6", "of", "12"]
    mark.update(1.0)
    assert shown().split() == ["draw", "12", "of", "12"]


def test_fraction_tally_is_a_wilson_interval_that_converges():
    n = 400
    rng = np.random.default_rng(1)
    hits = rng.random(n) < 0.3
    draws = rng.standard_normal((n, 2))
    _, ax, panel = _board()
    mark = ep.hop(draws, ax=ax, tally=hits, tally_ax=panel, level=0.95)
    k = 50
    p = hits[:k].mean()
    z = 1.959963984540054
    denom = 1.0 + z * z / k
    center = (p + z * z / (2 * k)) / denom
    half = z * np.sqrt(p * (1 - p) / k + z * z / (4 * k * k)) / denom
    est, lo, hi = mark.tally(k)
    assert est == pytest.approx(p)
    assert (lo, hi) == pytest.approx((center - half, center + half))
    widths = [mark.tally(j)[2] - mark.tally(j)[1] for j in (10, 50, 400)]
    assert widths == sorted(widths, reverse=True)
    est, lo, hi = mark.tally()
    assert lo < 0.3 < hi
    mark.update(k / n)
    bar = mark.tally_artists["lines"][0]
    np.testing.assert_allclose(bar.get_data(), [[k, k], [center - half, center + half]])
    np.testing.assert_allclose(mark.tally_artists["scatter"].get_offsets(), [[k, p]])
    x, y = mark.tally_artists["line"].get_data()
    assert len(x) == k
    assert y[-1] == pytest.approx(p)


def test_mean_tally_has_no_interval_at_one_draw():
    values = np.array([1.0, 3.0, 2.0, 2.0])
    _, ax, panel = _board()
    mark = ep.hop(np.zeros((4, 2)) + np.arange(4)[:, None], ax=ax, tally=values,
                  tally_ax=panel, kind="mean")  # fmt: skip
    mark.update(0.0)
    assert not mark.tally_artists["lines"][0].get_visible()
    est, lo, hi = mark.tally(1)
    assert est == 1.0 and np.isnan(lo) and np.isnan(hi)
    mark.update(1.0)
    assert mark.tally_artists["lines"][0].get_visible()
    est, lo, hi = mark.tally()
    assert est == pytest.approx(2.0)
    assert hi - est == pytest.approx(1.959963984540054 * np.std(values, ddof=1) / 2.0)


def test_tally_text_reads_the_current_count():
    _, mark, _, hits = _hop(n=12, tally_fmt=lambda e, lo, hi, k: f"{k}: {e:.2f}")
    mark.update(0.5)
    assert mark.tally_artists["text"].get_text() == f"6: {hits[:6].mean():.2f}"


def test_level_dims_the_run():
    _, mark, _, _ = _hop()
    mark.update(0.5, level=0.25)
    assert mark.artists["line"].get_alpha() == pytest.approx(0.25)
    assert mark.artists["lines"][0].get_alpha() == pytest.approx(0.25 * 0.45)
    mark.update(1.0)
    assert mark.artists["line"].get_alpha() is None


def test_draws_are_the_only_jumps():
    fig, mark, _, _ = _hop(n=6, rate=4.0)
    times = np.arange(round(mark.duration * 30) + 1) / 30.0
    # A marker gliding steadily across the draws' axes, as a clip around the
    # run would have, so the frame-to-frame steps have a baseline.
    glide = fig.add_artist(
        plt.Line2D([0.1], [0.02], marker="s", color="0.5", ms=4, ls="none")
    )

    def at(t):
        mark.update(t / mark.duration)
        glide.set_data([0.1 + 0.8 * t / mark.duration], [0.02])

    draws = [times[i] for i in mark.steps(times / mark.duration)]
    assert len(draws) == mark.n - 1
    glides = {"artists": [glide]}
    assert ept.motion_faults(fig, at, times, moves=[mark, glides], steps=draws) == []
    undeclared = ept.motion_faults(fig, at, times, moves=[mark, glides])
    assert any(f.kind == "jump" for f in undeclared)


def test_points_draw_as_markers_and_limits_hold_every_draw():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((20, 2))
    _, ax, _ = _board(tally=False)
    mark = ep.hop(pts, ax=ax)
    assert mark.artists["line"].get_marker() == "o"
    assert mark.artists["line"].get_linestyle() == "None"
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert x0 <= pts[:, 0].min() and pts[:, 0].max() <= x1
    assert y0 <= pts[:, 1].min() and pts[:, 1].max() <= y1
    for u in (0.0, 0.5, 1.0):
        mark.update(u)
        assert ax.get_xlim() == (x0, x1)


def test_rejects_bad_inputs():
    _, ax, panel = _board()
    with pytest.raises(ValueError, match="shaped"):
        ep.hop(np.zeros((3, 4, 3)), ax=ax)
    with pytest.raises(ValueError, match="finite"):
        ep.hop(np.full((3, 2), np.nan), ax=ax)
    with pytest.raises(ValueError, match="ghosts"):
        ep.hop(_draws(), ax=ax, ghosts=-1)
    with pytest.raises(ValueError, match="ghost_alpha"):
        ep.hop(_draws(), ax=ax, ghost_alpha=0.0)
    with pytest.raises(ValueError, match="kind"):
        ep.hop(_draws(), ax=ax, kind="median")
    with pytest.raises(ValueError, match="tally_ax"):
        ep.hop(_draws(), ax=ax, tally=np.ones(12))
    with pytest.raises(ValueError, match="12 draws"):
        ep.hop(_draws(), ax=ax, tally=np.ones(5), tally_ax=panel)
    with pytest.raises(ValueError, match="True or False"):
        ep.hop(_draws(), ax=ax, tally=np.full(12, 0.5), tally_ax=panel)
    with pytest.raises(ValueError, match="level"):
        ep.hop(_draws(), ax=ax, level=1.0)


def test_tally_panel_starts_at_zero_and_takes_extra_kwargs():
    draws = _draws(n=12)
    _, ax, panel = _board()
    mark = ep.hop(
        draws,
        ax=ax,
        tally=draws[:, 0, 0] > 1.0,
        tally_ax=panel,
        ref=0.5,
        ref_label="reference",
        tally_kw={"label_side": "below"},
        tally_fmt=lambda e, lo, hi, k: f"{e:.0%}",
        text_kw={"fontsize": 7},
    )
    assert panel.get_xlim()[0] <= 0.0
    assert panel.get_xlim()[1] >= 13.0
    label = next(t for t in panel.texts if t.get_text() == "reference")
    assert label.get_va() == "top"
    assert mark.tally_artists["text"].get_fontsize() == 7


@pytest.mark.parametrize("width", [0.15, 0.4])
def test_tally_points_clear_the_panel_edges_at_any_width(width):
    n = 40
    rng = np.random.default_rng(4)
    hits = np.r_[False, False, rng.random(n - 2) < 0.5]
    fig = plt.figure(figsize=(6.0, 3.0), dpi=200)
    ax = fig.add_axes([0.05, 0.1, 0.4, 0.8])
    panel = fig.add_axes([0.55, 0.3, width, 0.3])
    mark = ep.hop(rng.standard_normal((n, 2)), ax=ax, tally=hits, tally_ax=panel)
    fig.canvas.draw()
    box = panel.bbox
    for u in (0.0, 1.0):
        mark.update(u)
        scatter = mark.tally_artists["scatter"]
        x, y = panel.transData.transform(scatter.get_offsets()[0])
        radius = 0.5 * np.sqrt(scatter.get_sizes()[0]) * fig.dpi / 72.0
        assert box.x0 + radius <= x <= box.x1 - radius
        assert box.y0 + radius <= y <= box.y1 - radius
