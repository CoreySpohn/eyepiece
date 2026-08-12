"""Image primitives: floor, geometry rule, update path, shared-norm row."""

import hashlib
import inspect
import io

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eyepiece.images import compare_row, imshow_diverging, imshow_log, triptych


def _img(lo=1e-12, hi=1e-6):
    rng = np.random.default_rng(0)
    return rng.uniform(lo, hi, (8, 8))


def _hash_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# Captured from compare_row on commit 8c47f45, the last commit before
# vmin/vmax/imshow_kw/cbar_kw existed, via _hash_fig() on these exact
# scenarios. Regenerate only if compare_row's rendered output is
# intentionally changing.
_GOLDEN_CREATED_LOG = "e2835bddbe0b1367fd08a656b14ff4aaaf260d8179a21eabf496222907d80b8c"
_GOLDEN_HANDED_AXES = "04de2d635fd78707c01bf3cc0b7262c6a298fbda76cf2e1a8734e37d4809ad64"
_GOLDEN_DIVERGING = "a5087ba800cf021e7dcf029c0ce1f1b322fa5ae98970025488c5d14883e2cb74"


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
    a, b = _img(), _img(1e-10, 1e-5)
    res = compare_row(
        [a, b], titles=["A", "B"], vmin=None, vmax=None, imshow_kw=None, cbar_kw=None
    )
    assert _hash_fig(res.fig) == _GOLDEN_CREATED_LOG
    plt.close(res.fig)


def test_compare_row_defaults_unchanged_handed_axes():
    a, b = _img(), _img(1e-10, 1e-5)
    fig, axes = plt.subplots(1, 3)
    compare_row(
        [a, b], axes=axes[:2], vmin=None, vmax=None, imshow_kw=None, cbar_kw=None
    )
    assert _hash_fig(fig) == _GOLDEN_HANDED_AXES
    plt.close(fig)


def test_compare_row_defaults_unchanged_diverging_norm():
    c = np.array([[-3.0, 1.0], [2.0, -0.5]])
    d = np.array([[1.0, -1.0], [0.5, 0.2]])
    res = compare_row(
        [c, d], norm="diverging", vmin=None, vmax=None, imshow_kw=None, cbar_kw=None
    )
    assert _hash_fig(res.fig) == _GOLDEN_DIVERGING
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
