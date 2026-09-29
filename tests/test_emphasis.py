"""fade blends toward the real background and restores; capture collects a block."""

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

import eyepiece as ep


def _blend(color, level, bg):
    c, b = np.array(to_rgba(color)), np.array(to_rgba(bg))
    out = b + level * (c - b)
    out[3] = c[3]
    return out


def test_line_keeps_level_of_its_contrast_and_its_opacity():
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    (line,) = ax.plot([0, 1], [0, 1], color=(0.0, 0.2, 0.8, 0.6))
    ep.fade(line, 0.25)
    np.testing.assert_allclose(to_rgba(line.get_color()), [0.75, 0.8, 0.95, 0.6])
    plt.close(fig)


def test_dark_mode_fades_toward_the_dark_face():
    hwostyle.use("dark")
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], color="white")
    ep.fade(line, 0.0)
    np.testing.assert_allclose(to_rgba(line.get_color()), to_rgba(ax.get_facecolor()))
    plt.close(fig)


def test_axes_fade_reaches_every_kind_but_its_own_face():
    fig, ax = plt.subplots()
    face = to_rgba(ax.get_facecolor())
    (line,) = ax.plot([0, 1], [0, 1], "o-", color="red")
    patch = ax.add_patch(Rectangle((0, 0), 0.5, 0.5, facecolor="blue", alpha=0.8))
    scatter = ax.scatter([0.2], [0.3], color="green")
    text = ax.annotate(
        "x", (0.5, 0.5), xytext=(0.8, 0.8), color="black", arrowprops={"color": "k"}
    )
    text.set_bbox({"facecolor": "yellow", "edgecolor": "black"})
    image = ax.imshow(np.ones((2, 2)), extent=(0, 1, 0, 1))
    handle = ep.fade(ax, 0.5)

    bg = face
    np.testing.assert_allclose(to_rgba(line.get_color()), _blend("red", 0.5, bg))
    np.testing.assert_allclose(
        to_rgba(line.get_markerfacecolor()), _blend("red", 0.5, bg)
    )
    blended = _blend("blue", 0.5, bg)
    blended[3] = 0.8
    np.testing.assert_allclose(patch.get_facecolor(), blended)
    np.testing.assert_allclose(scatter.get_facecolor()[0], _blend("green", 0.5, bg))
    np.testing.assert_allclose(to_rgba(text.get_color()), _blend("black", 0.5, bg))
    np.testing.assert_allclose(
        text.get_bbox_patch().get_facecolor(), _blend("yellow", 0.5, bg)
    )
    np.testing.assert_allclose(
        text.arrow_patch.get_edgecolor(), _blend("black", 0.5, bg)
    )
    assert image.get_alpha() == pytest.approx(0.5)
    spine = ax.spines["left"]
    assert to_rgba(spine.get_edgecolor()) != to_rgba("black")
    assert to_rgba(ax.patch.get_facecolor()) == face
    assert line in handle.artists
    plt.close(fig)


def test_restore_puts_every_original_back():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], "s--", color="C3", mfc="none")
    patch = ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="purple"))
    scatter = ax.scatter([0, 1], [0, 1], c=[0.0, 1.0])
    image = ax.imshow(np.ones((2, 2)), alpha=0.7)
    before = (
        to_rgba(line.get_color()),
        line.get_markerfacecolor(),
        tuple(patch.get_edgecolor()),
        scatter.get_array().copy(),
    )
    handle = ep.fade(ax, 0.3)
    assert scatter.get_array() is None
    handle.restore()
    handle.restore()  # a second restore is a no-op
    assert to_rgba(line.get_color()) == before[0]
    assert line.get_markerfacecolor() == before[1]
    assert tuple(patch.get_edgecolor()) == before[2]
    np.testing.assert_array_equal(scatter.get_array(), before[3])
    assert image.get_alpha() == pytest.approx(0.7)
    plt.close(fig)


def test_colormapped_collection_draws_blended():
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    scatter = ax.scatter([0, 1], [0, 1], c=[0.0, 1.0], cmap="viridis")
    fig.canvas.draw()
    mapped = scatter.get_facecolor().copy()
    ep.fade(scatter, 0.4)
    fig.canvas.draw()
    expected = [_blend(c, 0.4, "white") for c in mapped]
    np.testing.assert_allclose(scatter.get_facecolor(), expected)
    plt.close(fig)


def test_fading_twice_compounds():
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    (line,) = ax.plot([0, 1], [0, 1], color="black")
    ep.fade(line, 0.5)
    ep.fade(line, 0.5)
    np.testing.assert_allclose(to_rgba(line.get_color()), _blend("black", 0.25, "w"))
    plt.close(fig)


def test_keep_leaves_matching_artists_untouched():
    fig, ax = plt.subplots()
    (kept,) = ax.plot([0, 1], [0, 1], color="red", gid="disk/ring")
    (faded,) = ax.plot([0, 1], [1, 0], color="red")
    ep.fade(ax, 0.2, keep=lambda a: (a.get_gid() or "").startswith("disk/"))
    assert to_rgba(kept.get_color()) == to_rgba("red")
    assert to_rgba(faded.get_color()) != to_rgba("red")
    plt.close(fig)


def test_unframed_inset_fades_toward_the_face_it_sits_on():
    fig, ax = plt.subplots()
    fig.set_facecolor("white")
    ax.set_facecolor("#202020")
    inset = ax.inset_axes([0.1, 0.1, 0.4, 0.4])
    inset.axis("off")
    (line,) = inset.plot([0, 1], [0, 1], color="white")
    ep.fade(ax, 0.0)
    np.testing.assert_allclose(to_rgba(line.get_color()), to_rgba("#202020"))
    plt.close(fig)


def test_explicit_background_wins():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], color="white")
    ep.fade([line], 0.0, background="red")
    assert to_rgba(line.get_color()) == to_rgba("red")
    plt.close(fig)


def test_hatch_color_fades():
    fig, ax = plt.subplots()
    ax.set_facecolor("white")
    patch = ax.add_patch(
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="k", hatch="//")
    )
    if not hasattr(patch, "set_hatchcolor"):
        pytest.skip("hatch colors are settable from matplotlib 3.10")
    patch.set_hatchcolor("black")
    ep.fade(patch, 0.5)
    np.testing.assert_allclose(
        to_rgba(patch.get_hatchcolor()), _blend("black", 0.5, "white")
    )
    plt.close(fig)


@pytest.mark.parametrize("level", [-0.1, 1.5])
def test_level_outside_unit_interval_raises(level):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="level"):
        ep.fade(ax, level)
    plt.close(fig)


def test_blend_keeps_level_of_the_contrast_and_the_color_alpha():
    out = ep.blend((0.0, 0.2, 0.8, 0.6), 0.25, background=(1.0, 1.0, 1.0, 0.1))
    np.testing.assert_allclose(out, [0.75, 0.8, 0.95, 0.6])
    assert ep.blend("red", 1.0, background="blue") == to_rgba("red")
    assert ep.blend("red", 0.0, background="blue") == to_rgba("blue")


def test_blend_is_the_color_fade_draws():
    fig, ax = plt.subplots()
    ax.set_facecolor("#203040")
    (line,) = ax.plot([0, 1], [0, 1], color="#e0a020")
    ep.fade(line, 0.45)
    np.testing.assert_allclose(
        to_rgba(line.get_color()),
        ep.blend("#e0a020", 0.45, background=ax.get_facecolor()),
    )
    plt.close(fig)


def test_blend_without_background_reads_the_face_color_at_call_time():
    with matplotlib.rc_context({"axes.facecolor": "black"}):
        dark = ep.blend("white", 0.25)
    with matplotlib.rc_context({"axes.facecolor": "white"}):
        light = ep.blend("black", 0.25)
    np.testing.assert_allclose(dark, [0.25, 0.25, 0.25, 1.0])
    np.testing.assert_allclose(light, [0.75, 0.75, 0.75, 1.0])


@pytest.mark.parametrize("level", [-0.1, 1.5])
def test_blend_level_outside_unit_interval_raises(level):
    with pytest.raises(ValueError, match="level"):
        ep.blend("red", level, background="white")


def test_capture_collects_only_what_the_block_added():
    fig, ax = plt.subplots()
    (earlier,) = ax.plot([0, 1], [0, 1])
    with ep.capture(ax) as added:
        (line,) = ax.plot([0, 1], [1, 0])
        text = ax.text(0.5, 0.5, "star")
        inset = ax.inset_axes([0.6, 0.6, 0.3, 0.3])
        inset.plot([0, 1], [0, 1])
        assert added == []
    assert set(map(id, added)) == {id(line), id(text), id(inset)}
    assert earlier not in added
    plt.close(fig)


def test_capture_then_fade_leaves_the_new_element_at_full_strength():
    fig, ax = plt.subplots()
    with ep.capture(ax) as seen:
        (old,) = ax.plot([0, 1], [0, 1], color="red")
    (new,) = ax.plot([0, 1], [1, 0], color="red")
    ep.fade(seen, 0.3)
    assert to_rgba(new.get_color()) == to_rgba("red")
    assert to_rgba(old.get_color()) != to_rgba("red")
    plt.close(fig)
