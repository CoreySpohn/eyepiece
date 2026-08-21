"""Image primitives: floor, geometry rule, update path, shared-norm row."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eyepiece import _style
from eyepiece.images import (
    compare_row,
    imshow_diverging,
    imshow_log,
    show_field,
    triptych,
)


def _img(lo=1e-12, hi=1e-6):
    rng = np.random.default_rng(0)
    return rng.uniform(lo, hi, (8, 8))


def test_imshow_log_floor_and_keys():
    img = _img()
    img[0, 0] = 0.0  # would break LogNorm without the floor
    res = imshow_log(img, floor=1e-14)
    assert {"image", "cbar"} <= set(res.artists)
    assert res.artists["image"].get_array().min() >= 1e-14
    assert res.artists["image"].get_interpolation() == "nearest"
    plt.close(res.fig)


def test_imshow_log_extent_applied():
    res = imshow_log(_img(), extent=(-2, 2, -2, 2))
    assert tuple(res.artists["image"].get_extent()) == (-2, 2, -2, 2)
    plt.close(res.fig)


def test_colorbar_does_not_steal_sibling_space():
    fig, axes = plt.subplots(1, 2)
    imshow_log(_img(), ax=axes[0])
    fig.canvas.draw()
    # original=True reports the gridspec slot BEFORE aspect="equal" shrinks
    # the box to fit the image data -- the post-adjustment get_position()
    # legitimately differs between panels even when neither stole space.
    w0 = axes[0].get_position(original=True).width
    w1 = axes[1].get_position(original=True).width
    assert w0 == pytest.approx(w1, rel=0.01)
    plt.close(fig)


def test_update_no_artist_leak():
    res = imshow_log(_img())
    n0 = len(res.ax.images)
    for _ in range(3):
        res.update(_img())
    assert len(res.ax.images) == n0
    plt.close(res.fig)


def test_compare_row_shared_norm():
    a, b = _img(), _img(1e-10, 1e-5)
    res = compare_row([a, b], titles=["A", "B"])
    ims = [res.artists["image"][0], res.artists["image"][1]]
    assert ims[0].norm is ims[1].norm
    plt.close(res.fig)


def test_diverging_symmetric():
    res = imshow_diverging(np.array([[-3.0, 1.0], [2.0, -0.5]]))
    im = res.artists["image"]
    assert im.norm.vmin == -im.norm.vmax
    plt.close(res.fig)


def test_compare_row_single_image():
    res = compare_row([_img()])
    assert res.axes.shape == (1,)
    assert len(res.artists["image"]) == 1
    plt.close(res.fig)


def test_compare_row_handed_axes_does_not_steal_sibling_space():
    fig, axes = plt.subplots(1, 3)
    compare_row([_img(), _img()], axes=axes[:2])
    fig.canvas.draw()
    # original=True reports the gridspec slot BEFORE aspect="equal" shrinks
    # the box to fit the image data -- the post-adjustment get_position()
    # legitimately differs between panels even when neither stole space.
    w2 = axes[2].get_position(original=True).width
    w0 = axes[0].get_position(original=True).width
    assert w2 == pytest.approx(w0, rel=0.01)
    plt.close(fig)


def test_compare_row_empty_raises():
    with pytest.raises(ValueError, match="at least one image"):
        compare_row([])


def test_compare_row_new_params_in_signature():
    sig = inspect.signature(compare_row)
    for p in ("vmin", "vmax", "imshow_kw", "cbar_kw"):
        assert p in sig.parameters


def test_compare_row_defaults_unchanged_created_figure():
    # Structural regression lock, replacing a one-time comparison of
    # rendered PNG hashes. Hash equality is brittle across platforms and
    # matplotlib versions, so this asserts on the norm, counts, titles, and
    # cmap a regression to compare_row's defaults would actually break.
    a, b = _img(), _img(1e-10, 1e-5)
    res = compare_row(
        [a, b], titles=["A", "B"], vmin=None, vmax=None, imshow_kw=None, cbar_kw=None
    )
    assert res.axes.shape == (2,)
    assert len(res.artists["image"]) == 2
    assert res.artists["image"][0].norm is res.artists["image"][1].norm
    floor = 1e-20
    clipped = [np.clip(a, floor, None), np.clip(b, floor, None)]
    expected_vmin = max(
        min(float(np.nanmin(clipped[0])), float(np.nanmin(clipped[1]))), floor
    )
    expected_vmax = max(float(np.nanmax(clipped[0])), float(np.nanmax(clipped[1])))
    norm = res.artists["image"][0].norm
    assert norm.vmin == pytest.approx(expected_vmin)
    assert norm.vmax == pytest.approx(expected_vmax)
    assert res.axes[0].get_title() == "A"
    assert res.axes[1].get_title() == "B"
    assert res.artists["image"][0].get_interpolation() == "nearest"
    assert res.artists["image"][0].get_cmap().name == _style.cmap("intensity").name
    assert len(res.fig.axes) == 3  # 2 image panels + 1 shared colorbar axes
    plt.close(res.fig)


def test_compare_row_defaults_unchanged_handed_axes():
    a, b = _img(), _img(1e-10, 1e-5)
    fig, axes = plt.subplots(1, 3)
    n_axes_before = len(fig.axes)
    res = compare_row(
        [a, b], axes=axes[:2], vmin=None, vmax=None, imshow_kw=None, cbar_kw=None
    )
    assert len(axes[0].images) == 1
    assert len(axes[1].images) == 1
    assert axes[0].images[0].norm is axes[1].images[0].norm
    floor = 1e-20
    clipped = [np.clip(a, floor, None), np.clip(b, floor, None)]
    expected_vmin = max(
        min(float(np.nanmin(clipped[0])), float(np.nanmin(clipped[1]))), floor
    )
    expected_vmax = max(float(np.nanmax(clipped[0])), float(np.nanmax(clipped[1])))
    norm = axes[0].images[0].norm
    assert norm.vmin == pytest.approx(expected_vmin)
    assert norm.vmax == pytest.approx(expected_vmax)
    assert len(axes[2].images) == 0  # untouched sibling
    assert axes[2].get_title() == ""
    assert res.artists["cbar"].mappable is axes[1].images[0]
    # the colorbar's axes is an inset child of the last handed axes, not a
    # new top-level sibling in fig.axes
    assert res.artists["cbar"].ax in axes[1].child_axes
    assert len(fig.axes) == n_axes_before
    plt.close(fig)


def test_compare_row_defaults_unchanged_diverging_norm():
    c = np.array([[-3.0, 1.0], [2.0, -0.5]])
    d = np.array([[1.0, -1.0], [0.5, 0.2]])
    res = compare_row(
        [c, d], norm="diverging", vmin=None, vmax=None, imshow_kw=None, cbar_kw=None
    )
    norm = res.artists["image"][0].norm
    assert res.artists["image"][0].norm is res.artists["image"][1].norm
    assert norm.vmin == pytest.approx(-3.0)
    assert norm.vmax == pytest.approx(3.0)
    assert res.artists["image"][0].get_cmap().name == _style.cmap("residual").name
    assert res.axes[0].get_title() == ""
    assert res.axes[1].get_title() == ""
    plt.close(res.fig)


def test_compare_row_vmin_pins_log_lower_bound():
    a, b = _img(), _img(1e-10, 1e-5)
    floor = 1e-20
    clipped = [np.clip(a, floor, None), np.clip(b, floor, None)]
    expected_vmax = max(float(np.nanmax(clipped[0])), float(np.nanmax(clipped[1])))
    res = compare_row([a, b], vmin=1e-8)
    norm = res.artists["image"][0].norm
    assert norm.vmin == 1e-8
    assert norm.vmax == pytest.approx(expected_vmax)
    plt.close(res.fig)


def test_compare_row_vmax_pins_log_upper_bound():
    a, b = _img(), _img(1e-10, 1e-5)
    floor = 1e-20
    clipped = [np.clip(a, floor, None), np.clip(b, floor, None)]
    expected_vmin = max(
        min(float(np.nanmin(clipped[0])), float(np.nanmin(clipped[1]))), floor
    )
    res = compare_row([a, b], vmax=5e-6)
    norm = res.artists["image"][0].norm
    assert norm.vmax == 5e-6
    assert norm.vmin == pytest.approx(expected_vmin)
    plt.close(res.fig)


def test_compare_row_vmin_vmax_pin_linear_norm():
    a, b = _img(), _img(1e-10, 1e-5)
    res = compare_row([a, b], norm="linear", vmin=-1.0, vmax=2.0)
    norm = res.artists["image"][0].norm
    assert norm.vmin == -1.0
    assert norm.vmax == 2.0
    plt.close(res.fig)


def test_compare_row_vmin_vmax_pin_diverging_norm():
    a, b = _img(), _img(1e-10, 1e-5)
    res = compare_row([a, b], norm="diverging", vmin=-3.0, vmax=5.0)
    norm = res.artists["image"][0].norm
    # vlim = max(abs(vmin), abs(vmax)) -- the diverging norm stays
    # symmetric even when vmin/vmax are pinned asymmetrically.
    assert norm.vmax == 5.0
    assert norm.vmin == -5.0
    plt.close(res.fig)


def test_compare_row_imshow_kw_overrides_interpolation():
    res = compare_row([_img(), _img()], imshow_kw={"interpolation": "bilinear"})
    assert res.artists["image"][0].get_interpolation() == "bilinear"
    plt.close(res.fig)


def test_compare_row_cbar_kw_reaches_colorbar():
    res = compare_row([_img(), _img()], cbar_kw={"extend": "both"})
    assert res.artists["cbar"].extend == "both"
    plt.close(res.fig)


def test_triptych_ratio_three_panels_and_shared_ab_norm():
    a, b = _img(), _img(1e-10, 1e-5)
    res = triptych(a, b, mode="ratio")
    assert len(res.artists["image"]) == 3
    assert res.artists["image"][0].norm is res.artists["image"][1].norm
    plt.close(res.fig)


def test_triptych_ratio_norm_centered_on_one():
    a, b = _img(), _img(1e-10, 1e-5)
    res = triptych(a, b, mode="ratio")
    norm = res.artists["image"][2].norm
    assert (norm.vmin + norm.vmax) / 2.0 == pytest.approx(1.0)
    plt.close(res.fig)


def test_triptych_ratio_default_clip_is_the_99th_percentile():
    # Pins the magnitude of the default clip, not just that the norm is
    # centered on 1: the percentile constant sets how much of every default
    # ratio panel is saturated, so a silent change to it must land red.
    rng = np.random.default_rng(5)
    a = rng.uniform(1.0, 2.0, (16, 16))
    b = a * rng.uniform(0.5, 1.5, (16, 16))
    res = triptych(a, b, mode="ratio")
    deviation = np.abs(b / a - 1.0)
    expected = float(np.percentile(deviation, 99.0))
    norm = res.artists["image"][2].norm
    assert norm.vmax == pytest.approx(1.0 + expected)
    assert norm.vmin == pytest.approx(1.0 - expected)
    # another percentile would give a materially different clip on this
    # data, so the assertion above really does pin the 99th
    assert float(np.percentile(deviation, 50.0)) < 0.8 * expected
    plt.close(res.fig)


def test_triptych_ratio_tight_clip_is_overridable():
    a, b = _img(), _img(1e-10, 1e-5)
    res = triptych(a, b, mode="ratio", ratio_clip=0.5)
    norm = res.artists["image"][2].norm
    assert norm.vmin == pytest.approx(0.5)
    assert norm.vmax == pytest.approx(1.5)
    plt.close(res.fig)


def test_triptych_ratio_division_by_zero_is_defined():
    a = np.array([[0.0, 1.0], [2.0, 4.0]])
    b = np.array([[3.0, 1.0], [2.0, 4.0]])
    res = triptych(a, b, mode="ratio", ratio_clip=0.5)
    ratio_panel = res.artists["image"][2].get_array()
    # a[0, 0] == 0 and b[0, 0] == 3.0 > 0 -> +inf ratio, clipped to the
    # panel's upper bound (1 + clip) rather than left as inf.
    assert np.isfinite(ratio_panel).all()
    assert ratio_panel[0, 0] == pytest.approx(1.5)
    plt.close(res.fig)


def test_triptych_ratio_zero_over_zero_is_defined():
    a = np.array([[0.0, 1.0], [2.0, 4.0]])
    b = np.array([[0.0, 1.0], [2.0, 4.0]])
    res = triptych(a, b, mode="ratio", ratio_clip=0.5)
    ratio_panel = res.artists["image"][2].get_array()
    # 0 / 0 is undefined; treated as "no change" (ratio == 1) rather than
    # left as nan.
    assert np.isfinite(ratio_panel).all()
    assert ratio_panel[0, 0] == pytest.approx(1.0)
    plt.close(res.fig)


def test_triptych_residual_reuses_imshow_diverging_norm():
    a, b = _img(), _img(1e-10, 1e-5)
    res = triptych(a, b, mode="residual")
    norm = res.artists["image"][2].norm
    assert norm.vmin == -norm.vmax
    direct = imshow_diverging(b - a)
    assert norm.vmax == pytest.approx(direct.artists["image"].norm.vmax)
    plt.close(res.fig)
    plt.close(direct.fig)


def test_triptych_default_and_custom_titles():
    a, b = _img(), _img(1e-10, 1e-5)
    res_default = triptych(a, b, mode="ratio")
    assert res_default.axes[0].get_title() == "A"
    assert res_default.axes[1].get_title() == "B"
    assert res_default.axes[2].get_title() == "B / A"
    plt.close(res_default.fig)

    res_custom = triptych(a, b, mode="residual", titles=["X", "Y", "Z"])
    assert res_custom.axes[0].get_title() == "X"
    assert res_custom.axes[1].get_title() == "Y"
    assert res_custom.axes[2].get_title() == "Z"
    plt.close(res_custom.fig)


def test_triptych_axes_draws_into_caller_axes():
    fig, axes = plt.subplots(1, 3)
    a, b = _img(), _img(1e-10, 1e-5)
    res = triptych(a, b, axes=axes)
    assert res.axes[0] is axes[0]
    assert res.axes[1] is axes[1]
    assert res.axes[2] is axes[2]
    plt.close(fig)


def test_triptych_extent_reaches_all_three_panels():
    a, b = _img(), _img(1e-10, 1e-5)
    for mode in ("ratio", "residual"):
        res = triptych(a, b, mode=mode, extent=(-2, 2, -2, 2))
        for im in res.artists["image"]:
            assert tuple(im.get_extent()) == (-2, 2, -2, 2)
        plt.close(res.fig)


def test_triptych_imshow_kw_overrides_interpolation_on_all_panels():
    a, b = _img(), _img(1e-10, 1e-5)
    for mode in ("ratio", "residual"):
        res = triptych(a, b, mode=mode, imshow_kw={"interpolation": "bilinear"})
        for im in res.artists["image"]:
            assert im.get_interpolation() == "bilinear"
        plt.close(res.fig)


def test_triptych_cbar_kw_reaches_both_colorbars():
    a, b = _img(), _img(1e-10, 1e-5)
    res = triptych(a, b, mode="ratio", cbar_kw={"extend": "both"})
    for cb in res.artists["cbar"]:
        assert cb.extend == "both"
    plt.close(res.fig)


def test_triptych_wrong_axes_count_raises_before_drawing():
    fig, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="needs 3 axes"):
        triptych(_img(), _img(), axes=axes[:2])
    assert all(len(ax.images) == 0 for ax in axes)  # nothing was drawn
    plt.close(fig)


def test_triptych_wrong_titles_count_raises_before_drawing():
    fig, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="needs 3 titles"):
        triptych(_img(), _img(), axes=axes, titles=["X", "Y"])
    assert all(len(ax.images) == 0 for ax in axes)
    plt.close(fig)


def test_triptych_unknown_a_b_norm_raises():
    fig, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="unknown a_b_norm"):
        triptych(_img(), _img(), axes=axes, a_b_norm="diverging")
    assert all(len(ax.images) == 0 for ax in axes)
    plt.close(fig)


def test_triptych_handed_axes_does_not_steal_sibling_space():
    fig, axes = plt.subplots(1, 4)
    a, b = _img(), _img(1e-10, 1e-5)
    triptych(a, b, axes=axes[:3])
    fig.canvas.draw()
    # original=True reports the gridspec slot BEFORE aspect="equal" shrinks
    # the box to fit the image data -- the post-adjustment get_position()
    # legitimately differs between panels even when neither stole space.
    w3 = axes[3].get_position(original=True).width
    w0 = axes[0].get_position(original=True).width
    assert w3 == pytest.approx(w0, rel=0.01)
    plt.close(fig)


def _axes_of(result):
    return [result.ax] if hasattr(result, "ax") else list(np.ravel(result.axes))


def test_every_image_primitive_hides_index_ticks_without_extent():
    # array indices are not a measurement, and show_field has dropped them
    # since it was ported -- the rest of the module has to answer "no extent"
    # the same way, or the same figure gains ticks by choice of primitive
    img = np.abs(np.random.default_rng(0).normal(size=(8, 8))) + 1e-6
    results = [
        imshow_log(img),
        imshow_diverging(img - img.mean()),
        compare_row([img, img]),
        triptych(img, img * 2),
        show_field(img + 1j * img),
    ]
    for result in results:
        for ax in _axes_of(result):
            assert ax.get_xticks().size == 0
            assert ax.get_yticks().size == 0
        plt.close(result.fig)


def test_every_image_primitive_keeps_ticks_with_an_extent():
    # the flip side: an extent means the axes carry real coordinates, and
    # those ticks are the whole point of passing it
    img = np.abs(np.random.default_rng(1).normal(size=(8, 8))) + 1e-6
    extent = (-4.0, 4.0, -4.0, 4.0)
    results = [
        imshow_log(img, extent=extent),
        imshow_diverging(img - img.mean(), extent=extent),
        compare_row([img, img], extent=extent),
        triptych(img, img * 2, extent=extent),
        show_field(img + 1j * img, extent=extent),
    ]
    for result in results:
        for ax in _axes_of(result):
            assert ax.get_xticks().size > 0, result
        plt.close(result.fig)


def test_imshow_diverging_updates_in_place_on_a_fixed_scale():
    # animating a residual or OPD map is a normal case, so this primitive
    # needs the same update hook imshow_log has -- and the symmetric norm
    # must NOT refit per frame, or the colors change meaning mid-animation
    first = np.linspace(-1.0, 1.0, 16).reshape(4, 4)
    res = imshow_diverging(first)
    assert res.update is not None
    norm = res.artists["image"].norm
    vmin, vmax = norm.vmin, norm.vmax
    res.update(first * 0.1)
    assert np.allclose(res.artists["image"].get_array(), first * 0.1)
    assert (norm.vmin, norm.vmax) == (vmin, vmax)
    plt.close(res.fig)


def test_figure_colorbar_is_a_real_figure_axes():
    # The structural difference: "figure" produces a figure-level axes, not
    # a child of the image axes. Consumers pin exactly this (an image axes
    # with no children) to prove they are not getting an inset.
    img = _img()
    fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    res = imshow_log(img, ax=ax, colorbar="figure", cbar_label="contrast")
    fig.canvas.draw()
    assert len(ax.child_axes) == 0
    assert res.artists["cbar"].ax in fig.axes
    plt.close(fig)


def test_figure_colorbar_makes_layout_reserve_room():
    # Constrained layout reserves room for BOTH kinds (an inset is a child
    # axes and get_tightbbox unions those), so this pins the smaller,
    # genuine difference: a figure-level colorbar takes slightly more room
    # than an inset. The regime where the two really diverge is a figure
    # with no layout engine -- see the clipping test below.
    img = _img()
    fig_i, ax_i = plt.subplots(figsize=(4, 4), layout="constrained")
    imshow_log(img, ax=ax_i, colorbar=True, cbar_label="contrast")
    fig_i.canvas.draw()
    fig_f, ax_f = plt.subplots(figsize=(4, 4), layout="constrained")
    imshow_log(img, ax=ax_f, colorbar="figure", cbar_label="contrast")
    fig_f.canvas.draw()
    assert ax_f.get_position().x1 < ax_i.get_position().x1
    plt.close(fig_i)
    plt.close(fig_f)


def test_default_colorbar_stays_an_inset():
    # Backwards compatibility: True keeps the inset, so existing callers
    # (and the sibling-space tests above) are unaffected.
    img = _img()
    fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    imshow_log(img, ax=ax, colorbar=True)
    assert len(ax.child_axes) == 1
    plt.close(fig)


def test_diverging_offers_the_same_figure_colorbar():
    img = np.linspace(-1.0, 1.0, 256).reshape(16, 16)
    fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    res = imshow_diverging(img, ax=ax, colorbar="figure", cbar_label="resid")
    fig.canvas.draw()
    assert len(ax.child_axes) == 0
    assert res.artists["cbar"].ax in fig.axes
    plt.close(fig)


def test_unknown_colorbar_mode_raises_naming_the_options():
    img = _img()
    with pytest.raises(ValueError, match=r"figure"):
        imshow_log(img, colorbar="inset-please")


def test_compare_row_scales_its_owned_figure_with_panel_count():
    # A row of k SQUARE panels at matplotlib's default 6.4x4.8 renders as k
    # small squares stranded in a tall figure beside a full-height colorbar.
    # An owned figure has to grow with the panel count.
    images = [_img() for _ in range(4)]
    res = compare_row(images, cbar_label="contrast")
    w, h = res.fig.get_size_inches()
    assert w > 2.0 * h
    plt.close(res.fig)


def test_compare_row_panel_size_is_tunable():
    images = [_img() for _ in range(3)]
    small = compare_row(images, panel_size=2.0)
    big = compare_row(images, panel_size=4.0)
    assert big.fig.get_size_inches()[0] > small.fig.get_size_inches()[0]
    plt.close(small.fig)
    plt.close(big.fig)


def test_compare_row_handed_axes_figure_is_left_alone():
    # The caller owns the figure; the primitive must not resize it.
    images = [_img() for _ in range(4)]
    fig, axes = plt.subplots(1, 4, figsize=(5.0, 5.0))
    before = tuple(fig.get_size_inches())
    compare_row(images, axes=axes)
    assert tuple(fig.get_size_inches()) == before
    plt.close(fig)


def test_inset_colorbar_clips_only_without_a_layout_engine():
    # The regime that actually bites. Constrained layout DOES see an inset
    # (it is a child axes, and get_tightbbox unions child_axes), so it
    # reserves room either way. On a figure with NO layout engine -- plain
    # plt.subplots, matplotlib's default -- nothing reserves anything and
    # the inset's label runs off the canvas. colorbar="figure" steals the
    # room from the axes instead, so it stays on-canvas.
    img = _img()

    def overflow(mode):
        fig, ax = plt.subplots(figsize=(4, 4))  # deliberately no layout engine
        res = imshow_log(img, ax=ax, colorbar=mode, cbar_label="contrast [ph/s/pix]")
        fig.canvas.draw()
        bb = res.artists["cbar"].ax.get_tightbbox(fig.canvas.get_renderer())
        over = bb.x1 - fig.get_window_extent().x1
        plt.close(fig)
        return over

    assert overflow(True) > 0.0
    assert overflow("figure") <= 0.0


def test_compare_row_clamps_a_degenerate_panel_aspect():
    # A row of wide strips must not collapse to a sliver, and a row of tall
    # panels must not blow the figure up to a hundred inches; before the
    # clamp these were 0.1in and 102.4in tall respectively.
    wide = compare_row([np.full((32, 1024), 1e-9) for _ in range(3)])
    tall = compare_row([np.full((1024, 32), 1e-9) for _ in range(3)])
    assert wide.fig.get_size_inches()[1] > 1.0
    assert tall.fig.get_size_inches()[1] < 12.0
    plt.close(wide.fig)
    plt.close(tall.fig)


def test_compare_row_takes_its_aspect_from_extent_when_given():
    # imshow derives the drawn aspect from extent, not from array shape, so
    # sizing the figure off the shape puts an extent-carrying call straight
    # back into "k thin panels stranded in a tall figure".
    images = [_img() for _ in range(3)]
    square = compare_row(images, extent=(0.0, 1.0, 0.0, 1.0))
    wide = compare_row(images, extent=(0.0, 10.0, 0.0, 1.0))
    assert wide.fig.get_size_inches()[1] < square.fig.get_size_inches()[1]
    plt.close(square.fig)
    plt.close(wide.fig)
