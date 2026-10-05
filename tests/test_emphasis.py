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


def test_fade_by_alpha_scales_each_drawn_alpha_and_restores():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], alpha=0.6)
    patch = ax.add_patch(Rectangle((0, 0), 1, 1, alpha=None))
    text = ax.text(0.5, 0.5, "label", bbox={"facecolor": "w", "alpha": 0.8})
    face_before = to_rgba(patch.get_facecolor())
    faded = ep.fade([line, patch, text], 0.5, by="alpha")
    assert line.get_alpha() == pytest.approx(0.3)
    assert patch.get_alpha() == pytest.approx(0.5)
    assert text.get_alpha() == pytest.approx(0.5)
    assert text.get_bbox_patch().get_alpha() == pytest.approx(0.4)
    # Opacity, not color: the line keeps its hue.
    assert to_rgba(line.get_color())[:3] == to_rgba(f"C{0}")[:3]
    assert all(a.get_visible() for a in (line, patch, text))
    faded.restore()
    assert line.get_alpha() == 0.6
    assert patch.get_alpha() is None
    assert to_rgba(patch.get_facecolor()) == face_before
    assert text.get_bbox_patch().get_alpha() == pytest.approx(0.8)
    plt.close(fig)


def test_fade_by_alpha_at_zero_hides_and_restore_shows_again():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], alpha=0.7)
    (hidden,) = ax.plot([0, 1], [1, 0])
    hidden.set_visible(False)
    faded = ep.fade([line, hidden], 0.0, by="alpha")
    assert not line.get_visible() and not hidden.get_visible()
    faded.restore()
    assert line.get_visible() and line.get_alpha() == 0.7
    # A level above zero never unhides what the caller had hidden.
    ep.fade(hidden, 1.0, by="alpha")
    assert not hidden.get_visible()
    plt.close(fig)


def test_fade_by_alpha_steps_relative_to_the_drawn_alpha_through_restore():
    fig, ax = plt.subplots()
    group = [ax.plot([0, 1], [0, 1], alpha=a)[0] for a in (0.2, 1.0)]
    faded = ep.fade(group, 0.0, by="alpha")
    for u in (0.25, 0.5, 1.0):
        faded.restore()
        faded = ep.fade(group, u, by="alpha")
        assert [ln.get_alpha() for ln in group] == pytest.approx([0.2 * u, u])
    plt.close(fig)


def test_fade_by_alpha_reaches_an_axes_and_its_images():
    fig, ax = plt.subplots()
    image = ax.imshow(np.ones((3, 3)), alpha=0.8)
    (line,) = ax.plot([0, 1], [0, 1])
    ep.fade(ax, 0.5, by="alpha")
    assert image.get_alpha() == pytest.approx(0.4)
    assert line.get_alpha() == pytest.approx(0.5)
    plt.close(fig)


def test_fade_rejects_an_unknown_by():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError, match="by"):
        ep.fade(line, 0.5, by="size")
    plt.close(fig)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_step_list_current_bright_finished_dim_later_hidden(mode):
    hwostyle.use(mode)
    rows = [("Steer", "hold the star"), "Correct", ("Block", "stop the light")]
    res = ep.step_list(rows, text_kw={"fontsize": 9})
    assert set(res.artists) == {"text"}
    texts = res.artists["text"]
    assert [t.get_text() for t in texts] == [
        "1. Steer",
        "hold the star",
        "2. Correct",
        "3. Block",
        "stop the light",
    ]
    assert all(t.get_visible() for t in texts)
    assert texts[0].get_fontsize() == 9
    ink = to_rgba(matplotlib.rcParams["text.color"])
    dim = to_rgba(ep._style.neutral(0.55))
    res.update(1)
    head0, body0, head1, head2, body2 = texts
    assert to_rgba(head0.get_color()) == dim and to_rgba(body0.get_color()) == dim
    assert head0.get_fontweight() == "normal"
    assert to_rgba(head1.get_color()) == ink and head1.get_fontweight() == "bold"
    assert not head2.get_visible() and not body2.get_visible()
    res.update(-1)
    assert not any(t.get_visible() for t in texts)
    res.update(0, done=True)
    assert all(t.get_visible() for t in texts)
    assert all(to_rgba(t.get_color()) == dim for t in texts)
    assert all(t.get_fontweight() == "normal" for t in texts)
    res.fig.canvas.draw()
    plt.close(res.fig)


def test_step_list_layout_and_initial_current():
    _, ax = plt.subplots()
    res = ep.step_list(
        [("a", "b"), ("c", "d")],
        ax=ax,
        current=-1,
        numbered=False,
        gap=0.3,
        indent=0.1,
        detail_offset=0.05,
    )
    head0, body0, head1, _ = res.artists["text"]
    assert head0.get_text() == "a"
    assert head0.get_position() == pytest.approx((0.0, 0.95))
    assert body0.get_position() == pytest.approx((0.1, 0.9))
    assert head1.get_position() == pytest.approx((0.0, 0.65))
    assert not any(t.get_visible() for t in res.artists["text"])
    assert not ax.axison
    with pytest.raises(ValueError, match="at least one"):
        ep.step_list([])
    plt.close("all")
