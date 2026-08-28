"""Radial-profile line plot, contrast curve, and the hwoutils convenience."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.layout_engine import ConstrainedLayoutEngine

from eyepiece import _style
from eyepiece.profiles import plot_contrast_curve, plot_radial, radial_profile_plot


def _profile(n=20):
    r = np.linspace(0.1, 10.0, n)
    values = 1e-8 * np.exp(-r)
    return r, values


def test_plot_radial_keys_and_handed_ax():
    r, values = _profile()
    fig, ax = plt.subplots()
    res = plot_radial(r, values, ax=ax)
    assert set(res.artists) == {"line"}
    assert res.ax is ax
    np.testing.assert_allclose(res.artists["line"].get_ydata(), values)
    plt.close(fig)


def test_plot_radial_creates_own_figure():
    r, values = _profile()
    res = plot_radial(r, values)
    assert len(res.fig.axes) == 1
    assert isinstance(res.fig.get_layout_engine(), ConstrainedLayoutEngine)
    plt.close(res.fig)


def test_plot_radial_log_sets_yscale():
    r, values = _profile()
    res = plot_radial(r, values, log=True)
    assert res.ax.get_yscale() == "log"
    plt.close(res.fig)


def test_plot_radial_defaults_to_linear_yscale():
    r, values = _profile()
    res = plot_radial(r, values)
    assert res.ax.get_yscale() == "linear"
    plt.close(res.fig)


def test_plot_radial_color_override():
    r, values = _profile()
    res = plot_radial(r, values, color="red")
    assert res.artists["line"].get_color() == "red"
    plt.close(res.fig)


def test_plot_radial_label_reaches_legend():
    r, values = _profile()
    res = plot_radial(r, values, label="model")
    assert res.artists["line"].get_label() == "model"
    plt.close(res.fig)


def test_plot_radial_line_kw_routes_to_plot():
    r, values = _profile()
    res = plot_radial(r, values, line_kw={"linestyle": ":"})
    assert res.artists["line"].get_linestyle() == ":"
    plt.close(res.fig)


def test_plot_radial_sets_no_label_unless_asked():
    r, values = _profile()
    res = plot_radial(r, values)
    assert res.ax.get_xlabel() == ""
    assert res.ax.get_ylabel() == ""
    plt.close(res.fig)


def test_plot_radial_xlabel_ylabel_applied_when_given():
    r, values = _profile()
    res = plot_radial(r, values, xlabel="separation", ylabel="value")
    assert res.ax.get_xlabel() == "separation"
    assert res.ax.get_ylabel() == "value"
    plt.close(res.fig)


def test_plot_contrast_curve_draws_curve():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast)
    assert "line" in res.artists
    np.testing.assert_allclose(res.artists["line"].get_ydata(), contrast)
    plt.close(res.fig)


def test_plot_contrast_curve_iwa_owa_markers():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast, iwa=1.0, owa=8.0)
    assert len(res.artists["fill"]) == 2
    assert len(res.artists["text"]) == 2
    labels = {t.get_text() for t in res.artists["text"]}
    assert labels == {"IWA", "OWA"}
    plt.close(res.fig)


def test_plot_contrast_curve_markers_clear_the_title():
    # the markers used to sit at y=1.02 in axes coords, which is exactly where
    # set_title draws -- so every titled contrast curve overlapped its own
    # annotations. The title here is deliberately wide enough to span the
    # axes, which is what makes it reach a marker near the left edge.
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast, iwa=1.0, owa=8.0)
    title = res.ax.set_title(
        "On-axis stellar leakage: ~1e-11 (true) vs ~1e-6 (single-MFT)"
    )
    res.fig.canvas.draw()
    renderer = res.fig.canvas.get_renderer()
    title_box = title.get_window_extent(renderer)
    for text in res.artists["text"]:
        box = text.get_window_extent(renderer)
        assert not title_box.overlaps(box), f"{text.get_text()} collides with the title"
    plt.close(res.fig)


def test_plot_contrast_curve_markers_stay_inside_the_axes():
    # the invariant behind the fix, independent of any particular title:
    # the markers belong inside the axes, not in the band above it.
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast, iwa=1.0, owa=8.0)
    res.fig.canvas.draw()
    renderer = res.fig.canvas.get_renderer()
    axes_top = res.ax.get_window_extent(renderer).y1
    for text in res.artists["text"]:
        assert text.get_window_extent(renderer).y1 <= axes_top, (
            f"{text.get_text()} is drawn above the axes, where the title lives"
        )
    plt.close(res.fig)


def test_plot_contrast_curve_iwa_only():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast, iwa=1.0)
    assert len(res.artists["fill"]) == 1
    assert res.artists["text"][0].get_text() == "IWA"
    plt.close(res.fig)


def test_plot_contrast_curve_floors_labeled():
    r, contrast = _profile()
    floors = [
        (r, contrast * 0.5, "photon floor"),
        (r, contrast * 0.25, "speckle floor"),
    ]
    res = plot_contrast_curve(r, contrast, floors=floors)
    assert len(res.artists["lines"]) == 2
    labels = [line.get_label() for line in res.artists["lines"]]
    assert labels == ["photon floor", "speckle floor"]
    plt.close(res.fig)


def test_plot_contrast_curve_no_markers_no_annotation_keys():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast)
    assert "fill" not in res.artists
    assert "text" not in res.artists
    assert "lines" not in res.artists
    plt.close(res.fig)


def test_plot_contrast_curve_annotate_once_per_axes():
    r, contrast = _profile()
    fig, ax = plt.subplots()
    res1 = plot_contrast_curve(r, contrast, ax=ax, iwa=1.0, owa=8.0, label="a")
    res2 = plot_contrast_curve(r, contrast * 0.5, ax=ax, iwa=1.0, owa=8.0, label="b")

    # exactly one set of IWA/OWA annotation artists across both calls
    assert len(res1.artists["fill"]) == 2
    assert len(res1.artists["text"]) == 2
    assert "fill" not in res2.artists
    assert "text" not in res2.artists
    assert len(ax.patches) == 2
    assert len(ax.texts) == 2

    # but two curves, one per call
    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == ["a", "b"]
    plt.close(fig)


def test_plot_contrast_curve_annotate_once_does_not_redraw_floors():
    r, contrast = _profile()
    floors = [(r, contrast * 0.5, "floor")]
    fig, ax = plt.subplots()
    res1 = plot_contrast_curve(r, contrast, ax=ax, floors=floors)
    res2 = plot_contrast_curve(r, contrast * 0.5, ax=ax, floors=floors)
    assert len(res1.artists["lines"]) == 1
    assert "lines" not in res2.artists
    # 2 main curves + 1 floor curve, not 2 floor curves
    assert len(ax.lines) == 3
    plt.close(fig)


def test_plot_contrast_curve_annotations_independent_per_axes():
    r, contrast = _profile()
    fig, axes = plt.subplots(1, 2)
    res_a = plot_contrast_curve(r, contrast, ax=axes[0], iwa=1.0, owa=8.0)
    res_b = plot_contrast_curve(r, contrast, ax=axes[1], iwa=1.0, owa=8.0)
    assert "fill" in res_a.artists
    assert "fill" in res_b.artists
    plt.close(fig)


def test_plot_contrast_curve_first_call_without_markers_does_not_poison_later_calls():
    # Regression: a first call with NO markers must not permanently block a
    # later call on the same ax that DOES ask for markers -- the guard
    # tracks what was drawn, not merely whether the function ran before.
    r, contrast = _profile()
    fig, ax = plt.subplots()
    plot_contrast_curve(r, contrast, ax=ax)
    res2 = plot_contrast_curve(r, contrast * 0.5, ax=ax, iwa=1.0, owa=8.0)
    assert "fill" in res2.artists
    assert "text" in res2.artists
    assert len(res2.artists["fill"]) == 2
    assert len(res2.artists["text"]) == 2
    assert len(ax.patches) == 2
    assert len(ax.texts) == 2
    plt.close(fig)


def test_plot_contrast_curve_redraws_annotations_after_ax_clear():
    # Regression: ax.clear() detaches the previously drawn annotation
    # artists but does not touch this module's cached state attribute (a
    # plain Python attribute survives clear()). A call after clear() must
    # redraw, not silently no-op.
    r, contrast = _profile()
    fig, ax = plt.subplots()
    plot_contrast_curve(r, contrast, ax=ax, iwa=1.0, owa=8.0)
    ax.clear()
    res = plot_contrast_curve(r, contrast, ax=ax, iwa=1.0, owa=8.0)
    assert "fill" in res.artists
    assert "text" in res.artists
    assert len(ax.patches) == 2
    assert len(ax.texts) == 2
    plt.close(fig)


def test_plot_contrast_curve_marker_kinds_annotate_independently():
    # IWA and OWA are tracked as independent kinds: a first call giving
    # only iwa, then a second giving only owa, must end with BOTH drawn --
    # not just the one from whichever call happened to run the annotation
    # block first.
    r, contrast = _profile()
    fig, ax = plt.subplots()
    res1 = plot_contrast_curve(r, contrast, ax=ax, iwa=1.0)
    res2 = plot_contrast_curve(r, contrast * 0.5, ax=ax, owa=8.0)
    assert res1.artists["text"][0].get_text() == "IWA"
    assert res2.artists["text"][0].get_text() == "OWA"
    assert len(ax.patches) == 2
    assert len(ax.texts) == 2
    plt.close(fig)


def test_plot_contrast_curve_shading_reaches_axes_edge_after_a_wider_curve():
    # The shaded regions mark separations the instrument cannot see, so
    # they have to reach the edge of the axes no matter what the x limits
    # end up being. Shading between the limits READ AT CALL TIME leaves the
    # far side unshaded once a second, wider curve moves them -- a reader
    # would take the unshaded gap for working range.
    fig, ax = plt.subplots(layout="constrained")
    r = np.linspace(0.5, 20.0, 50)
    res = plot_contrast_curve(r, 1e-8 * np.exp(-r), ax=ax, iwa=1.0, owa=15.0)
    iwa_fill, owa_fill = res.artists["fill"]

    wide = np.linspace(0.5, 60.0, 50)
    plot_contrast_curve(wide, 1e-8 * np.exp(-wide / 5.0), ax=ax)
    fig.canvas.draw()

    left, right = ax.get_xlim()
    assert right > 20.0  # the second curve really did widen the limits
    assert iwa_fill.get_x() <= left
    assert owa_fill.get_x() + owa_fill.get_width() >= right
    ax_box = ax.get_window_extent()
    assert iwa_fill.get_window_extent().x0 <= ax_box.x0
    assert owa_fill.get_window_extent().x1 >= ax_box.x1
    plt.close(fig)


def test_plot_contrast_curve_shading_does_not_drive_autoscale():
    # The annotation must not set the data range: a span reaching the axes
    # edge cannot be allowed to push the edge further out in turn.
    r, contrast = _profile()
    fig_bare, ax_bare = plt.subplots(layout="constrained")
    plot_contrast_curve(r, contrast, ax=ax_bare)
    fig_bare.canvas.draw()
    expected = ax_bare.get_xlim()

    fig, ax = plt.subplots(layout="constrained")
    plot_contrast_curve(r, contrast, ax=ax, iwa=1.0, owa=8.0)
    fig.canvas.draw()
    assert ax.get_xlim() == pytest.approx(expected)
    plt.close(fig_bare)
    plt.close(fig)


def test_plot_contrast_curve_successive_calls_cycle_the_palette():
    r, contrast = _profile()
    fig, ax = plt.subplots()
    res1 = plot_contrast_curve(r, contrast, ax=ax, label="a")
    res2 = plot_contrast_curve(r, contrast * 0.5, ax=ax, label="b")
    c1 = to_rgba(res1.artists["line"].get_color())
    c2 = to_rgba(res2.artists["line"].get_color())
    assert c1 != c2
    assert c1 == to_rgba(_style.color(0))
    assert c2 == to_rgba(_style.color(1))
    plt.close(fig)


def test_plot_contrast_curve_explicit_color_wins_without_advancing_the_cycle():
    r, contrast = _profile()
    fig, ax = plt.subplots()
    res1 = plot_contrast_curve(r, contrast, ax=ax, color="red")
    res2 = plot_contrast_curve(r, contrast * 0.5, ax=ax)
    assert res1.artists["line"].get_color() == "red"
    assert to_rgba(res2.artists["line"].get_color()) == to_rgba(_style.color(0))
    plt.close(fig)


def test_plot_contrast_curve_floors_continue_the_same_cycle():
    r, contrast = _profile()
    floors = [(r, contrast * 0.5, "floor")]
    fig, ax = plt.subplots()
    res1 = plot_contrast_curve(r, contrast, ax=ax, floors=floors)
    res2 = plot_contrast_curve(r, contrast * 0.5, ax=ax)
    assert to_rgba(res1.artists["lines"][0].get_color()) == to_rgba(_style.color(1))
    # the second curve takes the next free color, not the floor's
    assert to_rgba(res2.artists["line"].get_color()) == to_rgba(_style.color(2))
    plt.close(fig)


def test_plot_radial_successive_calls_cycle_the_palette():
    r, values = _profile()
    fig, ax = plt.subplots()
    res1 = plot_radial(r, values, ax=ax)
    res2 = plot_radial(r, values * 0.5, ax=ax)
    c1 = to_rgba(res1.artists["line"].get_color())
    c2 = to_rgba(res2.artists["line"].get_color())
    assert c1 != c2
    assert c1 == to_rgba(_style.color(0))
    assert c2 == to_rgba(_style.color(1))
    plt.close(fig)


def test_plot_radial_explicit_color_wins_without_advancing_the_cycle():
    r, values = _profile()
    fig, ax = plt.subplots()
    res1 = plot_radial(r, values, ax=ax, color="red")
    res2 = plot_radial(r, values * 0.5, ax=ax)
    assert res1.artists["line"].get_color() == "red"
    assert to_rgba(res2.artists["line"].get_color()) == to_rgba(_style.color(0))
    plt.close(fig)


def test_plot_contrast_curve_line_kw_routes_to_plot():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast, line_kw={"linestyle": ":"})
    assert res.artists["line"].get_linestyle() == ":"
    plt.close(res.fig)


def test_plot_contrast_curve_span_kw_routes_to_axvspan():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast, iwa=1.0, span_kw={"alpha": 0.9})
    assert res.artists["fill"][0].get_alpha() == pytest.approx(0.9)
    plt.close(res.fig)


def test_plot_contrast_curve_floor_kw_routes_to_plot():
    r, contrast = _profile()
    floors = [(r, contrast * 0.5, "floor")]
    # "ls" (not "linestyle") -- the default floor kwargs already set "ls",
    # and matplotlib rejects being handed both spellings of the same alias
    # at once, so overriding via the alias already in use is what a caller
    # actually has to do here.
    res = plot_contrast_curve(r, contrast, floors=floors, floor_kw={"ls": "-."})
    assert res.artists["lines"][0].get_linestyle() == "-."
    plt.close(res.fig)


def test_plot_contrast_curve_sets_no_label_unless_asked():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast)
    assert res.ax.get_xlabel() == ""
    assert res.ax.get_ylabel() == ""
    plt.close(res.fig)


def test_plot_contrast_curve_xlabel_ylabel_applied_when_given():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast, xlabel="separation", ylabel="contrast")
    assert res.ax.get_xlabel() == "separation"
    assert res.ax.get_ylabel() == "contrast"
    plt.close(res.fig)


def test_plot_contrast_curve_creates_own_figure():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast)
    assert len(res.fig.axes) == 1
    assert isinstance(res.fig.get_layout_engine(), ConstrainedLayoutEngine)
    plt.close(res.fig)


def test_plot_contrast_curve_default_log_yscale():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast)
    assert res.ax.get_yscale() == "log"
    plt.close(res.fig)


def test_plot_contrast_curve_handed_ax_only_draws_into_its_own_axes():
    # plot_contrast_curve never sets aspect and adds no colorbar/inset, so
    # the get_position(original=True) idiom used elsewhere in this library
    # (which exists to catch aspect="equal"-driven constrained-layout
    # reflow from image primitives) cannot discriminate here: this module
    # sets no aspect and adds no colorbar, so original and active
    # positions are identical before and after regardless of correctness.
    # A bare "sibling position unchanged" assertion does not hold either:
    # under a real constrained-layout figure, the "IWA"/"OWA" text this
    # module draws sits just outside axes[0]'s box, and a CORRECT
    # implementation legitimately makes the constrained-layout engine
    # reflow margins (and so the sibling's slot) to make room for it. What
    # this module actually guarantees on the handed-ax path is structural,
    # not positional: it draws only into the given ax, creating no new
    # Axes and attaching no artist to a different one -- exactly the kind
    # of bug a leaked/misdirected annotation-state reference could cause.
    r, contrast = _profile()
    fig, axes = plt.subplots(1, 2, layout="constrained")
    n_axes_before = len(fig.axes)
    floors = [(r, contrast * 0.5, "floor")]
    plot_contrast_curve(r, contrast, ax=axes[0], iwa=1.0, owa=8.0, floors=floors)
    fig.canvas.draw()
    assert len(fig.axes) == n_axes_before
    assert len(axes[1].patches) == 0
    assert len(axes[1].texts) == 0
    assert len(axes[1].lines) == 0
    plt.close(fig)


def _image(n=16):
    rng = np.random.default_rng(0)
    return rng.uniform(1e-10, 1e-6, (n, n))


def test_radial_profile_plot_computes_and_plots():
    try:
        import hwoutils  # noqa: F401
    except ImportError:
        img = _image()
        with pytest.raises(ImportError, match=r"eyepiece\[hwo\]"):
            radial_profile_plot(img, 1.0)
    else:
        from hwoutils import radial

        img = _image()
        expected_sep, expected_profile = radial.radial_profile(img, 1.0)
        res = radial_profile_plot(img, 1.0)
        np.testing.assert_allclose(
            res.artists["line"].get_xdata(), np.asarray(expected_sep)
        )
        np.testing.assert_allclose(
            res.artists["line"].get_ydata(), np.asarray(expected_profile)
        )
        plt.close(res.fig)


def test_radial_profile_plot_forwards_plot_kwargs():
    try:
        import hwoutils  # noqa: F401
    except ImportError:
        pytest.skip("hwoutils not installed")
    img = _image()
    res = radial_profile_plot(img, 1.0, color="red", log=True)
    assert res.artists["line"].get_color() == "red"
    assert res.ax.get_yscale() == "log"
    plt.close(res.fig)


def test_radial_profile_plot_default_xlabel_is_lod_mathtext():
    try:
        import hwoutils  # noqa: F401
    except ImportError:
        pytest.skip("hwoutils not installed")
    img = _image()
    res = radial_profile_plot(img, 1.0)
    assert res.ax.get_xlabel() == r"$r$ [$\lambda/D$]"
    plt.close(res.fig)


def test_radial_profile_plot_xlabel_override_wins():
    try:
        import hwoutils  # noqa: F401
    except ImportError:
        pytest.skip("hwoutils not installed")
    img = _image()
    res = radial_profile_plot(img, 1.0, xlabel="custom")
    assert res.ax.get_xlabel() == "custom"
    plt.close(res.fig)


def test_radial_profile_plot_forwards_center_and_nbins():
    try:
        import hwoutils  # noqa: F401
    except ImportError:
        pytest.skip("hwoutils not installed")
    from hwoutils import radial

    img = _image()
    expected_sep, expected_profile = radial.radial_profile(
        img, 1.0, center=(3.0, 3.0), nbins=5
    )
    res = radial_profile_plot(img, 1.0, center=(3.0, 3.0), nbins=5)
    np.testing.assert_allclose(
        res.artists["line"].get_xdata(), np.asarray(expected_sep)
    )
    np.testing.assert_allclose(
        res.artists["line"].get_ydata(), np.asarray(expected_profile)
    )
    plt.close(res.fig)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e10])
def test_plot_contrast_curve_shading_is_scale_free(scale):
    # The shading was built by stepping a FIXED 1e9 data units off the
    # working angle, which silently breaks at both ends of that constant.
    # In radians (x ~ 1e-8) the step swamps the anchor, `inner + width`
    # rounds to -1e9 and the region's inner edge lands on 0 instead of on
    # the IWA -- the excluded zone reads as working range. At x ~ 1e10 the
    # step is far too short and both regions stop mid-axes, leaving an
    # unshaded gap outside the working angles. Neither raises. The reach
    # has to come from the axes limits, which are always in the data's own
    # units, rather than from a constant that assumes arcseconds.
    fig, ax = plt.subplots(layout="constrained")
    r = np.linspace(1.0, 10.0, 50) * scale
    iwa, owa = 2.0 * scale, 8.0 * scale
    res = plot_contrast_curve(r, np.logspace(-11, -8, 50), ax=ax, iwa=iwa, owa=owa)
    fig.canvas.draw()

    iwa_fill, owa_fill = res.artists["fill"]
    left, right = ax.get_xlim()

    # The edge facing the visible region sits exactly on the working angle.
    assert iwa_fill.get_x() + iwa_fill.get_width() == pytest.approx(iwa, rel=1e-6)
    assert owa_fill.get_x() == pytest.approx(owa, rel=1e-6)
    # And the far edge still reaches past the axes, at every scale.
    assert iwa_fill.get_x() <= left
    assert owa_fill.get_x() + owa_fill.get_width() >= right
    plt.close(fig)
