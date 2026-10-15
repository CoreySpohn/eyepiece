"""Updatable arrows, angle arcs and compasses.

An arrow grows and moves on one artist; an arc keeps its head tangent as it
sweeps; a compass puts east on the side the caller names, never a default
one. The rest state, the one a still shows, is the figure as drawn.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep


def _axes():
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.set(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), xticks=[], yticks=[])
    ax.set_aspect("equal")
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, ax


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _head(arc_result, ax):
    """Tip and direction (degrees, display) of the arc's last head."""
    patch = arc_result.artists["arrow"]
    paths, _ = patch._get_path_in_displaycoord()
    vertices = paths[-1].vertices[:3]
    return vertices


def _tangent_error(result, ax, center, radius, angle, clockwise):
    """Degrees between the head's axis and the arc's tangent at `angle`."""
    vertices = _head(result, ax)
    rad = math.radians(angle)
    tip = ax.transData.transform(
        np.asarray(center) + radius * np.array([math.cos(rad), math.sin(rad)])
    )
    k = int(np.argmin(np.hypot(*(vertices - tip).T)))
    axis = vertices[k] - np.delete(vertices, k, axis=0).mean(axis=0)
    tangent = np.array([-math.sin(rad), math.cos(rad)]) * (-1.0 if clockwise else 1.0)
    cos = float(axis @ tangent) / float(np.hypot(*axis))
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))


# Angle arc


@pytest.mark.parametrize("end", [30.0, 95.0, 170.0, 250.0, 340.0, -45.0, -200.0])
def test_arc_head_stays_tangent_through_a_sweep(end):
    fig, ax = _axes()
    arc = ep.angle_arc((0.2, -0.1), 0.0, 20.0, 1.2, ax=ax)
    clockwise = end < 0.0
    for u in (1.0, 0.8, 0.5, 0.3):
        arc.update(u, end_angle=end)
        fig.canvas.draw()
        stop = u * end
        assert _tangent_error(arc, ax, (0.2, -0.1), 1.2, stop, clockwise) < 0.5


def test_arc_head_is_tangent_at_the_start_end_too():
    fig, ax = _axes()
    arc = ep.angle_arc((0.0, 0.0), 20.0, 150.0, 1.0, ax=ax, head="start")
    fig.canvas.draw()
    patch = arc.artists["arrow"]
    paths, _ = patch._get_path_in_displaycoord()
    vertices = paths[-1].vertices[:3]
    rad = math.radians(20.0)
    tip = ax.transData.transform([math.cos(rad), math.sin(rad)])
    k = int(np.argmin(np.hypot(*(vertices - tip).T)))
    axis = vertices[k] - np.delete(vertices, k, axis=0).mean(axis=0)
    tangent = -np.array([-math.sin(rad), math.cos(rad)])
    assert float(axis @ tangent) / float(np.hypot(*axis)) > math.cos(math.radians(0.5))


def test_arc_has_two_heads_when_asked():
    fig, ax = _axes()
    arc = ep.angle_arc((0.0, 0.0), 0.0, 120.0, 1.0, ax=ax, head="both")
    fig.canvas.draw()
    _, fillable = arc.artists["arrow"]._get_path_in_displaycoord()
    assert sum(1 for f in fillable if f) == 2
    plain = ep.angle_arc((0.0, 0.0), 0.0, 120.0, 1.0, ax=ax, head="none")
    fig.canvas.draw()
    _, fillable = plain.artists["arrow"]._get_path_in_displaycoord()
    assert not np.any(fillable)


def test_arc_sweeps_on_the_same_artists_and_keeps_the_limits():
    fig, ax = _axes()
    arc = ep.angle_arc((0.0, 0.0), 0.0, 90.0, 1.0, ax=ax, label=r"$\theta$")
    patch, text = arc.artists["arrow"], arc.artists["text"]
    children = list(ax.get_children())
    limits = (ax.get_xlim(), ax.get_ylim())
    for u in np.linspace(0.0, 1.0, 9):
        arc.update(u, end_angle=90.0 + 40.0 * u)
        fig.canvas.draw()
    assert arc.artists["arrow"] is patch and arc.artists["text"] is text
    assert list(ax.get_children()) == children
    assert (ax.get_xlim(), ax.get_ylim()) == limits


def test_arc_rest_state_is_pixel_exact():
    fig, ax = _axes()
    arc = ep.angle_arc((0.0, 0.0), 0.0, 75.0, 1.0, ax=ax, label="a")
    rest = _pixels(fig)
    for u in (0.0, 0.4, 0.9):
        arc.update(u, end_angle=75.0 + 50.0 * u, level=0.5)
        assert not np.array_equal(_pixels(fig), rest)
    arc.update(1.0, end_angle=75.0, level=1.0)
    assert np.array_equal(_pixels(fig), rest)


def test_arc_is_absent_at_u_zero():
    fig, ax = _axes()
    bare = _pixels(fig)
    arc = ep.angle_arc((0.0, 0.0), 0.0, 75.0, 1.0, ax=ax, label="a")
    arc.update(0.0)
    assert np.array_equal(_pixels(fig), bare)
    assert not arc.artists["text"].get_visible()


def test_arc_label_sits_at_the_midpoint_on_a_backing_box():
    _, ax = _axes()
    arc = ep.angle_arc((0.5, 0.0), 10.0, 70.0, 1.0, ax=ax, label="x")
    text = arc.artists["text"]
    mid = math.radians(40.0)
    expected = (0.5 + math.cos(mid), math.sin(mid))
    assert text.xy == pytest.approx(expected)
    assert text.get_bbox_patch() is not None
    arc.update(0.5)
    half = math.radians(25.0)
    assert text.xy == pytest.approx((0.5 + math.cos(half), math.sin(half)))
    out = ep.angle_arc((0.0, 0.0), 0.0, 90.0, 1.0, ax=ax, label="y", label_scale=1.5)
    assert out.artists["text"].xy == pytest.approx(
        (1.5 * math.cos(math.radians(45.0)), 1.5 * math.sin(math.radians(45.0)))
    )


def test_arc_clockwise_goes_the_short_way():
    _, ax = _axes()
    short = ep.angle_arc((0.0, 0.0), 0.0, -10.0, 1.0, ax=ax)
    patch = short.artists["arrow"]
    verts = patch._path_original.vertices
    assert verts[:, 1].max() <= 1e-9
    assert verts[-1] == pytest.approx(
        [math.cos(math.radians(-10.0)), math.sin(math.radians(-10.0))]
    )


def test_arc_rejects_bad_input():
    _, ax = _axes()
    with pytest.raises(ValueError, match="head"):
        ep.angle_arc((0, 0), 0, 10, 1.0, ax=ax, head="middle")
    with pytest.raises(ValueError, match="positive"):
        ep.angle_arc((0, 0), 0, 10, 0.0, ax=ax)
    with pytest.raises(ValueError, match="360"):
        ep.angle_arc((0, 0), 0, 400, 1.0, ax=ax)
    with pytest.raises(ValueError, match="finite"):
        ep.angle_arc((0, 0), 0, float("nan"), 1.0, ax=ax)


def test_tiny_arc_draws_nothing_instead_of_raising():
    fig, ax = _axes()
    bare = _pixels(fig)
    arc = ep.angle_arc((0.0, 0.0), 0.0, 1e-9, 1.0, ax=ax)
    assert np.array_equal(_pixels(fig), bare)
    arc.update(1.0, end_angle=0.0)
    assert np.array_equal(_pixels(fig), bare)


# Arrow


@pytest.mark.parametrize("kind", ["vector", "double", "dimension"])
def test_arrow_updates_reuse_the_artists(kind):
    fig, ax = _axes()
    result = ep.arrow((-1.0, 0.0), (1.0, 0.5), ax=ax, kind=kind, label="d")
    patch, text = result.artists["arrow"], result.artists["text"]
    children = list(ax.get_children())
    limits = (ax.get_xlim(), ax.get_ylim())
    for u in np.linspace(0.0, 1.0, 7):
        result.update(u, tail=(-1.0, -0.5 * u), head=(1.0 + u, 0.5))
        fig.canvas.draw()
    assert result.artists["arrow"] is patch and result.artists["text"] is text
    assert list(ax.get_children()) == children
    assert (ax.get_xlim(), ax.get_ylim()) == limits


@pytest.mark.parametrize("kind", ["vector", "double", "dimension"])
def test_arrow_rest_state_is_pixel_exact(kind):
    fig, ax = _axes()
    result = ep.arrow((-1.0, 0.0), (1.0, 0.5), ax=ax, kind=kind, label="d")
    rest = _pixels(fig)
    for u in (0.0, 0.3, 0.8):
        result.update(u, tail=(-1.0, 0.2), head=(0.5, -0.5), level=0.4)
        assert not np.array_equal(_pixels(fig), rest)
    result.update(1.0, tail=(-1.0, 0.0), head=(1.0, 0.5), level=1.0)
    assert np.array_equal(_pixels(fig), rest)


@pytest.mark.parametrize("kind", ["vector", "double", "dimension"])
def test_arrow_is_absent_at_u_zero(kind):
    fig, ax = _axes()
    bare = _pixels(fig)
    result = ep.arrow((-1.0, 0.0), (1.0, 0.5), ax=ax, kind=kind, label="d")
    result.update(0.0)
    assert np.array_equal(_pixels(fig), bare)


def test_vector_grows_from_the_tail_and_two_headed_kinds_from_the_middle():
    _, ax = _axes()
    vector = ep.arrow((-1.0, 0.0), (1.0, 0.0), ax=ax)
    vector.update(0.5)
    a, b = vector.artists["arrow"]._ends
    assert a == pytest.approx((-1.0, 0.0)) and b == pytest.approx((0.0, 0.0))
    gap = ep.arrow((-1.0, 0.0), (1.0, 0.0), ax=ax, kind="dimension")
    gap.update(0.5)
    a, b = gap.artists["arrow"]._ends
    assert a == pytest.approx((-0.5, 0.0)) and b == pytest.approx((0.5, 0.0))


def test_double_and_dimension_have_two_heads_and_vector_one():
    fig, ax = _axes()
    counts = {}
    for kind in ("vector", "double", "dimension"):
        result = ep.arrow((-1.0, 0.0), (1.0, 0.0), ax=ax, kind=kind)
        fig.canvas.draw()
        _, fillable = result.artists["arrow"]._get_path_in_displaycoord()
        counts[kind] = int(np.sum(fillable)) if kind != "dimension" else None
    assert counts["vector"] == 1 and counts["double"] == 2


def test_dimension_label_sits_on_the_line_in_a_backing_box():
    _, ax = _axes()
    gap = ep.arrow((-1.0, 0.0), (1.0, 0.0), ax=ax, kind="dimension", label="d")
    text = gap.artists["text"]
    assert text.get_bbox_patch() is not None
    assert text.xy == pytest.approx((0.0, 0.0))
    assert text.xyann == pytest.approx((0.0, 0.0))
    side = ep.arrow((-1.0, 0.0), (1.0, 0.0), ax=ax, label="v")
    assert side.artists["text"].get_bbox_patch() is None
    assert side.artists["text"].xyann[1] > 0.0


def test_a_moved_arrow_moves_its_label():
    _, ax = _axes()
    result = ep.arrow((-1.0, 0.0), (1.0, 0.0), ax=ax, label="v")
    result.update(tail=(0.0, 1.0), head=(2.0, 1.0))
    assert result.artists["text"].xy == pytest.approx((1.0, 1.0))
    result.update()
    assert result.artists["arrow"]._ends == ((0.0, 1.0), (2.0, 1.0))


def test_level_dims_and_restores_the_alpha():
    _, ax = _axes()
    result = ep.arrow((-1.0, 0.0), (1.0, 0.0), ax=ax, label="v")
    patch = result.artists["arrow"]
    result.update(level=0.5)
    assert patch.get_alpha() == pytest.approx(0.5)
    result.update(level=0.0)
    assert not result.artists["text"].get_visible()
    result.update(level=1.0)
    assert patch.get_alpha() is None and result.artists["text"].get_visible()


def test_arrow_rejects_bad_input():
    _, ax = _axes()
    with pytest.raises(ValueError, match="kind"):
        ep.arrow((0, 0), (1, 1), ax=ax, kind="curly")
    with pytest.raises(ValueError, match="finite"):
        ep.arrow((0, 0), (float("inf"), 1), ax=ax)
    with pytest.raises(ValueError, match="pair"):
        ep.arrow((0, 0, 0), (1, 1), ax=ax)


def test_a_stage_can_hold_the_arrow_and_the_arc():
    fig, ax = _axes()
    arrow = ep.arrow((-1.0, 0.0), (1.0, 0.0), ax=ax, label="v")
    arc = ep.angle_arc((0.0, 0.0), 0.0, 60.0, 1.0, ax=ax, label="a")
    rest = _pixels(fig)
    stage = ep.Stage({"a": [*arrow.artists.values()], "b": [*arc.artists.values()]})
    stage.set(shown=["a"], dimmed=["b"])
    assert not np.array_equal(_pixels(fig), rest)
    stage.rest()
    assert np.array_equal(_pixels(fig), rest)


# Compass


def _tips(result):
    north, east = result.artists["text"]
    return np.array(north.xy), np.array(east.xy)


@pytest.mark.parametrize("angle", [0.0, 30.0, 135.0, -70.0, 200.0])
def test_compass_puts_east_left_of_north_for_the_sky(angle):
    _, ax = _axes()
    center = np.array([0.1, -0.2])
    result = ep.compass(center, 0.8, ax=ax, east="left", angle=angle)
    north, east = _tips(result)
    origin = ax.transData.transform(center)
    shown_n, shown_e = (ax.transData.transform(p) - origin for p in (north, east))
    # East is a quarter turn counterclockwise from north on the screen.
    assert shown_n[0] * shown_e[1] - shown_n[1] * shown_e[0] > 0.0
    assert shown_n @ shown_e == pytest.approx(0.0, abs=1e-9)
    assert np.hypot(*(north - center)) == pytest.approx(0.8)
    assert np.hypot(*(east - center)) == pytest.approx(0.8)


@pytest.mark.parametrize("angle", [0.0, 30.0, -70.0])
def test_compass_east_right_is_the_mirror(angle):
    _, ax = _axes()
    center = np.array([0.0, 0.0])
    sky = ep.compass(center, 0.8, ax=ax, east="left", angle=angle)
    chart = ep.compass(center, 0.8, ax=ax, east="right", angle=angle)
    (n1, e1), (n2, e2) = _tips(sky), _tips(chart)
    assert n1 == pytest.approx(n2)
    assert e1 == pytest.approx(-e2)
    origin = ax.transData.transform(center)
    shown = [ax.transData.transform(p) - origin for p in (n2, e2)]
    assert shown[0][0] * shown[1][1] - shown[0][1] * shown[1][0] < 0.0


def test_compass_north_up_east_left_at_zero_angle():
    _, ax = _axes()
    result = ep.compass((0.0, 0.0), 1.0, ax=ax, east="left")
    north, east = _tips(result)
    assert north == pytest.approx((0.0, 1.0)) and east == pytest.approx((-1.0, 0.0))


def test_compass_has_no_default_for_east():
    _, ax = _axes()
    with pytest.raises(TypeError, match="east"):
        ep.compass((0.0, 0.0), 1.0, ax=ax)
    with pytest.raises(ValueError, match="east"):
        ep.compass((0.0, 0.0), 1.0, ax=ax, east="up")
    with pytest.raises(ValueError, match="length"):
        ep.compass((0.0, 0.0), 0.0, ax=ax, east="left")


def test_compass_turns_on_the_same_artists_and_rests_pixel_exact():
    fig, ax = _axes()
    result = ep.compass((0.0, 0.0), 0.8, ax=ax, east="left")
    arrows, texts = list(result.artists["arrow"]), list(result.artists["text"])
    children = list(ax.get_children())
    rest = _pixels(fig)
    for u, angle in ((0.0, 10.0), (0.5, 80.0), (1.0, 200.0)):
        result.update(u, angle=angle)
        assert not np.array_equal(_pixels(fig), rest)
    assert list(result.artists["arrow"]) == arrows
    assert list(result.artists["text"]) == texts
    assert list(ax.get_children()) == children
    result.update(1.0, angle=0.0)
    assert np.array_equal(_pixels(fig), rest)


def test_compass_is_absent_at_u_zero_and_can_drop_its_letters():
    fig, ax = _axes()
    bare = _pixels(fig)
    result = ep.compass((0.0, 0.0), 0.8, ax=ax, east="right", labels=None)
    assert "text" not in result.artists
    result.update(0.0)
    assert np.array_equal(_pixels(fig), bare)
    named = ep.compass((0.0, 0.0), 0.8, ax=ax, east="right", labels=("N", "E"))
    named.update(0.0)
    assert np.array_equal(_pixels(fig), bare)
