"""Radial-profile line plot, contrast curve, and the hwoutils convenience."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
    assert res.ax.figure is not None
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


def test_plot_contrast_curve_creates_own_figure():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast)
    assert res.ax.figure is not None
    plt.close(res.fig)


def test_plot_contrast_curve_default_log_yscale():
    r, contrast = _profile()
    res = plot_contrast_curve(r, contrast)
    assert res.ax.get_yscale() == "log"
    plt.close(res.fig)


def test_plot_contrast_curve_handed_ax_does_not_steal_sibling_space():
    r, contrast = _profile()
    fig, axes = plt.subplots(1, 2)
    floors = [(r, contrast * 0.5, "floor")]
    plot_contrast_curve(r, contrast, ax=axes[0], iwa=1.0, owa=8.0, floors=floors)
    fig.canvas.draw()
    # original=True reports the gridspec slot BEFORE anything else adjusts
    # it -- the geometry rule is that a primitive handed an ax never alters
    # a sibling axes' slot.
    w0 = axes[0].get_position(original=True).width
    w1 = axes[1].get_position(original=True).width
    assert w0 == pytest.approx(w1, rel=0.01)
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
