"""Cameras: the overview box, the page camera, and the spotlight veil."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.layout_engine import ConstrainedLayoutEngine, PlaceHolderLayoutEngine
from matplotlib.patches import PathPatch, Rectangle

import eyepiece as ep

# ---------------------------------------------------------------------------
# overview_box


def test_overview_box_draws_the_view_and_moves_with_update():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((8, 8)), extent=(-4, 4, -4, 4))
    res = ep.overview_box((1.0, -1.0, 2.0), ax=ax, aspect=0.5)
    box = res.artists["ellipse"]
    assert isinstance(box, Rectangle)
    assert box.get_xy() == pytest.approx((0.0, -1.5))
    assert (box.get_width(), box.get_height()) == pytest.approx((2.0, 1.0))
    res.update((0.0, 0.0, 4.0))
    assert res.artists["ellipse"] is box
    assert box.get_xy() == pytest.approx((-2.0, -1.0))
    assert (box.get_width(), box.get_height()) == pytest.approx((4.0, 2.0))
    plt.close(fig)


def test_overview_box_creates_its_own_figure_and_routes_box_kw():
    res = ep.overview_box((0.0, 0.0, 1.0), box_kw={"lw": 3.0, "ls": "--"})
    assert res.fig.get_layout_engine() is not None
    assert res.artists["ellipse"].get_linewidth() == pytest.approx(3.0)
    assert res.artists["ellipse"].get_fill() is False
    plt.close(res.fig)


def test_overview_box_defaults_to_the_text_color():
    with matplotlib.rc_context({"text.color": "#123456"}):
        res = ep.overview_box((0.0, 0.0, 1.0))
    assert res.artists["ellipse"].get_edgecolor() == pytest.approx(to_rgba("#123456"))
    plt.close(res.fig)


# ---------------------------------------------------------------------------
# PageCamera


def _composite(facecolor="white"):
    fig, axes = plt.subplots(1, 3, figsize=(8, 4.5), layout="constrained")
    fig.set_facecolor(facecolor)
    for k, a in enumerate(axes):
        a.imshow(np.arange(64.0).reshape(8, 8) * (k + 1), interpolation="nearest")
        a.set_title(f"panel {k}")
    return fig, axes


@pytest.mark.parametrize("overview", [False, True])
def test_page_camera_renders_exactly_the_output_size(overview):
    fig, axes = _composite()
    cam = ep.PageCamera(fig, (320, 180), overview=overview)
    for view in (cam.page, cam.fit_axes(axes[1]), (7.5, 0.5, 2.0), (-1.0, 2.0, 3.0)):
        frame = cam.render(view)
        assert frame.shape == (180, 320, 4)
        assert frame.dtype == np.uint8
    cam.release()
    plt.close(fig)


def test_page_camera_freezes_the_layout_and_release_restores_it():
    fig, _ = _composite()
    cam = ep.PageCamera(fig, (320, 180))
    assert isinstance(fig.get_layout_engine(), PlaceHolderLayoutEngine)
    cam.release()
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
    plt.close(fig)


def test_page_camera_background_is_opaque_even_when_savefig_is_transparent():
    fig, _ = _composite("white")
    with matplotlib.rc_context({"savefig.transparent": True}):
        frame = ep.PageCamera(fig, (320, 180)).render((0.2, 0.2, 0.4))
    assert frame[0, 0, 3] == 255
    assert tuple(frame[0, 0, :3]) == (255, 255, 255)
    plt.close(fig)


def test_overview_margin_is_the_facecolor_and_holds_the_page():
    fig, axes = _composite("#202020")
    cam = ep.PageCamera(fig, (480, 270), overview=True)
    frame = cam.render(cam.fit_axes(axes[0]))
    margin = 480 // 6
    bottom_right = frame[-5, -5, :3]
    assert tuple(bottom_right) == (0x20, 0x20, 0x20)
    strip = frame[: 270 // 2, -margin:, :3].astype(int)
    assert np.ptp(strip) > 50  # the thumbnail and its box are drawn there
    plt.close(fig)


def test_fit_keeps_views_inside_the_page_and_matches_the_view_aspect():
    fig, _ = _composite()
    cam = ep.PageCamera(fig, (320, 180))
    cx, cy, w = cam.fit((0.0, 0.0, 1.0, 1.0), pad=0.0)
    h = w * 180 / 320
    assert cx - w / 2 >= -1e-9 and cy - h / 2 >= -1e-9
    assert cam.aspect == pytest.approx(180 / 320)
    over = ep.PageCamera(fig, (320, 180), overview=True)
    assert over.aspect == pytest.approx(180 / (320 - 320 // 6))
    plt.close(fig)


def test_page_camera_magnifies_crisply():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow([[0.0, 1.0], [1.0, 0.0]], interpolation="nearest", cmap="gray")
    ax.set_axis_off()
    cam = ep.PageCamera(fig, (200, 200))
    frame = cam.render(cam.fit_axes(ax, pad=0.0)).astype(int)
    # A nearest-pixel checkerboard magnified stays two tones, no blur ramp.
    values = np.unique(frame[20:-20, 20:-20, 0])
    assert len(values) <= 4
    plt.close(fig)


def test_page_camera_works_on_a_figure_pyplot_has_closed():
    fig, axes = _composite()
    plt.close(fig)  # what a notebook's inline backend does after each cell
    cam = ep.PageCamera(fig, (160, 90))
    assert cam.render(cam.fit_axes(axes[2])).shape == (90, 160, 4)


def test_page_camera_rejects_bad_sizes():
    fig, _ = _composite()
    with pytest.raises(ValueError, match="size_px"):
        ep.PageCamera(fig, (0, 100))
    plt.close(fig)


# ---------------------------------------------------------------------------
# spotlight


def test_spotlight_cuts_one_hole_per_rect_in_figure_inches():
    fig, axes = _composite()
    res = ep.spotlight(fig, [(0.5, 0.5, 2.0, 2.0), (3.0, 0.5, 4.0, 2.0)])
    veil = res.artists["fill"]
    assert isinstance(veil, PathPatch)
    assert len(veil.get_path().vertices) == 5 * 3
    assert veil.get_alpha() == pytest.approx(0.75)
    assert veil.get_in_layout() is False
    assert list(res.axes) == list(axes)
    assert len(fig.axes) == 3  # no axes added
    plt.close(fig)


def test_spotlight_needs_axes():
    fig = plt.figure()
    with pytest.raises(ValueError, match="axes"):
        ep.spotlight(fig, [(0.0, 0.0, 1.0, 1.0)])
    plt.close(fig)


def test_spotlight_update_moves_holes_and_scales_strength():
    fig, _ = _composite()
    res = ep.spotlight(fig, [(0.5, 0.5, 2.0, 2.0)], strength=0.0)
    veil = res.artists["fill"]
    assert veil.get_alpha() == pytest.approx(0.0)
    res.update(rects=[(1.0, 1.0, 3.0, 3.0)], strength=0.5)
    assert res.artists["fill"] is veil
    assert veil.get_alpha() == pytest.approx(0.375)
    assert veil.get_path().vertices[5:].min(axis=0) == pytest.approx((1.0, 1.0))
    res.update(strength=1.0)
    assert len(veil.get_path().vertices) == 10
    plt.close(fig)


def test_spotlight_rejects_overlapping_or_inverted_rects():
    fig, _ = _composite()
    with pytest.raises(ValueError, match="overlap"):
        ep.spotlight(fig, [(0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0)])
    with pytest.raises(ValueError, match="x0 < x1"):
        ep.spotlight(fig, [(2.0, 0.0, 1.0, 1.0)])
    res = ep.spotlight(fig, [(0.0, 0.0, 1.0, 1.0)])
    with pytest.raises(ValueError, match="overlap"):
        res.update(rects=[(0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0)])
    assert len(res.artists["fill"].get_path().vertices) == 10
    plt.close(fig)


def test_spotlight_dims_outside_the_hole_in_a_render():
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.set_facecolor("black")
    ax.set_position([0, 0, 1, 1])
    ax.imshow(np.ones((2, 2)), cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax.set_axis_off()
    ep.spotlight(fig, [(0.0, 0.0, 2.0, 2.0)])
    frame = ep.PageCamera(fig, (200, 100)).render((2.0, 1.0, 4.0))
    lit, dim = frame[50, 50, 0], frame[50, 150, 0]
    assert lit > 200 > dim
    plt.close(fig)


# ---------------------------------------------------------------------------
# Review fixes: small frames, release, transparency, refresh, side effects


def test_overview_thumbnail_of_a_wide_figure_on_a_small_frame_renders():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot([0, 1], [0, 1])
    ax.set_title("a title that FreeType must draw")
    with ep.PageCamera(fig, (320, 180), overview=True) as cam:
        assert cam.render(cam.fit_axes(ax)).shape == (180, 320, 4)
    plt.close(fig)


def test_a_view_too_wide_for_text_still_renders_by_downsampling():
    fig, _ = _composite()
    with ep.PageCamera(fig, (100, 56)) as cam:
        assert cam.render((4.0, 2.0, 200.0)).shape == (56, 100, 4)
    big, ax = plt.subplots(figsize=(20, 5))
    ax.set_title("a title")
    with ep.PageCamera(big, (160, 90)) as cam:
        assert cam.render(cam.page).shape == (90, 160, 4)
    plt.close(fig)
    plt.close(big)


def test_positions_hold_across_renders_and_release_ends_the_camera():
    fig, axes = _composite()
    with ep.PageCamera(fig, (320, 180)) as cam:
        before = [a.get_position().bounds for a in axes]
        cam.render(cam.page)
        cam.render(cam.fit_axes(axes[0]))  # a different dpi
        assert [a.get_position().bounds for a in axes] == before
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
    with pytest.raises(RuntimeError, match="released"):
        cam.render(cam.page)
    plt.close(fig)


def test_a_transparent_figure_still_gives_opaque_frames():
    fig, _ = _composite("none")
    frame = ep.PageCamera(fig, (160, 90)).render((0.2, 0.2, 0.4))
    assert frame[0, 0, 3] == 255
    plt.close(fig)


def test_refresh_redraws_the_cached_overview():
    fig, axes = _composite("#202020")
    cam = ep.PageCamera(fig, (480, 270), overview=True)
    margin = slice(480 - 480 // 6, None)
    first = cam.render(cam.page)[:120, margin].copy()
    for a in axes:
        a.images[0].set_data(np.zeros((8, 8)))
    assert np.array_equal(cam.render(cam.page)[:120, margin], first)
    cam.refresh()
    assert not np.array_equal(cam.render(cam.page)[:120, margin], first)
    plt.close(fig)


def test_spotlight_keeps_pyplot_current_axes_and_tight_saves():
    import io

    fig, axes = _composite()
    plt.sca(axes[1])

    def tight_size():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=50)
        buf.seek(0)
        from PIL import Image

        return Image.open(buf).size

    before = tight_size()
    ep.spotlight(fig, [(0.5, 0.5, 2.0, 2.0)])
    assert plt.gca() is axes[1]
    assert tight_size() == before
    plt.close(fig)


def test_overview_box_stays_out_of_the_data_limits():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((8, 8)), extent=(-4, 4, -4, 4))
    ep.overview_box((0.0, 0.0, 20.0), ax=ax)
    ax.autoscale_view()
    assert ax.get_xlim() == pytest.approx((-4.0, 4.0))
    plt.close(fig)


def test_spotlight_holes_stay_put_in_offset_views():
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.set_facecolor("black")
    ax.set_position([0, 0, 1, 1])
    ax.imshow(np.ones((2, 2)), cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax.set_axis_off()
    ep.spotlight(fig, [(0.0, 0.0, 2.0, 2.0)])
    with ep.PageCamera(fig, (100, 100)) as cam:
        right = cam.render((3.0, 1.0, 2.0))  # the veiled half, alone
        left = cam.render((1.0, 1.0, 2.0))  # the hole, alone
        straddle = cam.render((2.0, 1.0, 2.0))
    assert right[50, 50, 0] < 100
    assert left[50, 50, 0] > 200
    assert straddle[50, 20, 0] > 200 > 100 > straddle[50, 80, 0]
    plt.close(fig)
