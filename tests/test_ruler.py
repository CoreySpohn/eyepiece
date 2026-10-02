"""ruler: a double-headed dimension arrow with its label on a backing box."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import eyepiece as ep
from eyepiece._phasor import HEAD_LENGTH, HEAD_PER_MARKER


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_keys_tones_and_placement_in_either_mode(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots()
    ax.imshow(np.random.default_rng(0).random((8, 8)), extent=(-4, 4, -4, 4))
    res = ep.ruler(ax, (-2.0, -3.0), (2.0, -3.0), "4 units")
    assert set(res.artists) == {"arrow", "lines", "text"}
    assert set(res.artists) <= ep.ARTIST_KEYS
    arrow = res.artists["arrow"]
    (under,) = res.artists["lines"]
    text = res.artists["text"]
    assert isinstance(arrow, FancyArrowPatch) and isinstance(under, Line2D)
    assert arrow._ends == ((-2.0, -3.0), (2.0, -3.0))
    np.testing.assert_allclose(under.get_xydata(), [(-2.0, -3.0), (2.0, -3.0)])
    text_color = to_rgb(plt.rcParams["text.color"])
    face = to_rgb(plt.rcParams["axes.facecolor"])
    assert to_rgb(arrow.get_edgecolor()) == pytest.approx(text_color)
    assert to_rgb(under.get_color()) == pytest.approx(face)
    assert under.get_linewidth() > arrow.get_linewidth()
    # The backing line is drawn first, on the arrow's layer; the label above.
    order = ax.get_children()
    assert order.index(under) < order.index(arrow)
    assert under.get_zorder() == arrow.get_zorder() < text.get_zorder()
    assert text.get_text() == "4 units"
    np.testing.assert_allclose(text.xy, (0.0, -3.0))
    assert (text.get_ha(), text.get_va()) == ("center", "bottom")
    assert text.get_position()[1] > 0.0
    assert to_rgb(text.get_bbox_patch().get_facecolor()) == pytest.approx(face)
    fig.canvas.draw()
    plt.close(fig)


def test_the_ruler_leaves_the_limits_alone():
    fig, ax = plt.subplots()
    ax.imshow(np.ones((4, 4)), extent=(-1, 1, -1, 1))
    limits = ax.get_xlim(), ax.get_ylim()
    ep.ruler(ax, (-5.0, 0.0), (5.0, 3.0), "far")
    fig.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    plt.close(fig)


@pytest.mark.parametrize(
    ("side", "ha", "va", "sign"),
    [
        ("above", "center", "bottom", (0, 1)),
        ("below", "center", "top", (0, -1)),
        ("left", "right", "center", (-1, 0)),
        ("right", "left", "center", (1, 0)),
    ],
)
def test_side_places_the_label(side, ha, va, sign):
    fig, ax = plt.subplots()
    res = ep.ruler(ax, (0, 0), (0, 1), "h", side=side, offset_pt=5.0)
    text = res.artists["text"]
    assert (text.get_ha(), text.get_va()) == (ha, va)
    np.testing.assert_allclose(text.get_position(), 5.0 * np.array(sign), atol=1e-12)
    plt.close(fig)


def test_unknown_side_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="side"):
        ep.ruler(ax, (0, 0), (1, 0), "x", side="top")
    plt.close(fig)


def test_no_backing_and_no_text_on_a_plain_plot():
    fig, ax = plt.subplots()
    res = ep.ruler(ax, (0, 0), (1, 0), backing=False, color="0.6")
    assert set(res.artists) == {"arrow"}
    assert to_rgb(res.artists["arrow"].get_edgecolor()) == to_rgb("0.6")
    labeled = ep.ruler(ax, (0, 0), (1, 0), "bare", backing=False)
    assert labeled.artists["text"].get_bbox_patch() is None
    plt.close(fig)


def test_heads_scale_with_the_marker_size_and_fit_a_short_ruler():
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    with plt.rc_context({"lines.markersize": 8.0}):
        res = ep.ruler(ax, (10, 50), (90, 50))
    arrow = res.artists["arrow"]
    assert arrow.get_mutation_scale() == pytest.approx(HEAD_PER_MARKER * 8.0)
    # A ruler shorter than its two heads caps each head at half its length.
    res.update(p1=(10.5, 50))
    length_pt = arrow._length_pt()
    assert arrow.get_mutation_scale() == pytest.approx(length_pt / (2 * HEAD_LENGTH))
    plt.close(fig)


def test_update_moves_the_same_artists_and_keeps_styles():
    fig, ax = plt.subplots()
    res = ep.ruler(ax, (0, 0), (1, 0), "1", arrow_kw={"color": "red", "lw": 2.0})
    n_children = len(ax.get_children())
    arrow, (under,), text = (res.artists[k] for k in ("arrow", "lines", "text"))
    res.update(p1=(3.0, 0.0), text="3")
    assert len(ax.get_children()) == n_children
    assert arrow._ends == ((0.0, 0.0), (3.0, 0.0))
    np.testing.assert_allclose(under.get_xdata(), [0.0, 3.0])
    np.testing.assert_allclose(text.xy, (1.5, 0.0))
    assert text.get_text() == "3"
    assert to_rgb(arrow.get_edgecolor()) == to_rgb("red")
    assert arrow.get_linewidth() == pytest.approx(2.0)
    res.update(p0=(1.0, 1.0))
    assert arrow._ends == ((1.0, 1.0), (3.0, 0.0))
    assert text.get_text() == "3"
    plt.close(fig)


def test_a_zero_length_ruler_draws_only_its_label_without_warning(recwarn):
    fig, ax = plt.subplots()
    res = ep.ruler(ax, (1, 1), (1, 1), "0")
    fig.canvas.draw()
    res.update(p1=(2, 1))
    fig.canvas.draw()
    res.update(p1=(1, 1))
    fig.canvas.draw()
    assert res.artists["arrow"]._vanishing()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    plt.close(fig)


def test_update_cannot_add_a_label():
    fig, ax = plt.subplots()
    res = ep.ruler(ax, (0, 0), (1, 0))
    with pytest.raises(ValueError, match="without a label"):
        res.update(text="late")
    plt.close(fig)


def test_routed_kwargs_reach_their_artists():
    fig, ax = plt.subplots()
    res = ep.ruler(
        ax,
        (0, 0),
        (1, 0),
        "x",
        arrow_kw={"gid": "r"},
        line_kw={"lw": 7.0},
        text_kw={"fontsize": 20},
        head_scale=0.0,
    )
    assert res.artists["arrow"].get_gid() == "r"
    assert res.artists["lines"][0].get_linewidth() == pytest.approx(7.0)
    assert res.artists["text"].get_fontsize() == pytest.approx(20)
    fig.canvas.draw()
    plt.close(fig)
