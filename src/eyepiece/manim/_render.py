"""Native Manim (Cairo) objects for prepared views, and in-place updates.

`render` turns a prepared view tree into a Manim `Group` and returns a
`ManimResult` whose `.parts` exposes every mobject by the element ID it was
drawn from. Only image samples (and the colormap strip of a colorbar) are
raster `ImageMobject`s; axes, paths, points, error bars, regions, reference
lines, and labels are native vector mobjects.

Every mark is placed through its panel's frame: the rectangle whose corners
are the axis limits. Updates read that frame's current corners, so a panel
the caller has moved or uniformly scaled keeps the same data-coordinate
mapping, and marks, markers, and label text follow it. The x axis
reversal (positive RA on the left) flips the display mapping once; the data
and the image storage are never reversed, and image row 0 is the bottom
pixel row, as in the Matplotlib renderer.

Opacity belongs to the caller. An update rewrites geometry, image pixels,
and label outlines, but never sets a part's opacity: a nonfinite coordinate
is a gap whose marker, error bar, or path segment has no points (the
handle stays, and a later finite update gives it points again), and an
image keeps the opacity the caller last set on it. A new path weight
rescales the path's current opacity rather than replacing it.
"""

import math
from types import MappingProxyType

import manim
import numpy as np

from eyepiece._prepared_render import (
    VIEW_TYPES,
    array_key,
    check_topology,
    default_cast,
    gap_nan,
    leaf_views,
    path_alphas,
    region_fill_opacity,
    resolve_styles,
    topology,
    visible_xy,
)
from eyepiece.prepared import (
    ImageView,
    PanelGroup,
    Path,
    Points,
    ReferenceLine,
    Region,
    map_rgba,
)
from eyepiece.style import snapshot_profile

# Scene-unit layout of a freshly rendered panel. A 16:9 Manim frame is 8
# units tall, so one panel fits beside a second one or a title. Callers
# move and scale the returned mobject freely afterward.
_PANEL_BOX = (6.0, 4.0)
_PANEL_BUFF = 0.8
_COLORBAR_WIDTH = 0.25
_COLORBAR_BUFF = 0.2
_TICK_LENGTH = 0.1
_TICK_TARGET = 5
_TICK_FONT_SCALE = 0.75
_MARKER_RADIUS = 0.07
_DASH_LENGTH = 0.15

# Mark draw order, matching the Matplotlib renderer's z-orders: regions
# below lines and error bars, points and labels on top.
_REGION, _LINE, _TOP = 0, 1, 2

# Leaf-view parts added beside the element IDs: the frame (the live data
# mapping anchor) and the axis decoration (ticks and axis labels).
_LEAF_PARTS = ("frame", "axes")


# --- Geometry -----------------------------------------------------------------


def _panel_size(view):
    """Frame (width, height) in scene units for `view`'s axis spec."""
    spec = view.axes
    x_low, x_high = (float(v) for v in spec.x_limits)
    y_low, y_high = (float(v) for v in spec.y_limits)
    box_width, box_height = _PANEL_BOX
    if spec.aspect == "auto":
        return box_width, box_height
    try:
        ratio = 1.0 if spec.aspect == "equal" else float(spec.aspect)
    except (TypeError, ValueError):
        raise ValueError(
            f"{view.id}: aspect {spec.aspect!r} is not 'equal', 'auto', or a number"
        ) from None
    if not (math.isfinite(ratio) and ratio > 0):
        raise ValueError(f"{view.id}: aspect must be positive, got {spec.aspect!r}")
    height_per_width = ratio * (y_high - y_low) / (x_high - x_low)
    if box_width * height_per_width <= box_height:
        return box_width, box_width * height_per_width
    return box_height / height_per_width, box_height


def _basis(frame):
    """(origin, x_vec, y_vec) of a frame: its lower-left corner and edges."""
    _, upper_left, lower_left, lower_right = frame.get_vertices()
    return lower_left, lower_right - lower_left, upper_left - lower_left


def _to_scene(xy, spec, basis):
    """Map (P, 2) data coordinates to (P, 3) scene points; NaN rows stay NaN."""
    origin, x_vec, y_vec = basis
    xy = gap_nan(np.asarray(xy, dtype=float).reshape(-1, 2))
    x_low, x_high = (float(v) for v in spec.x_limits)
    y_low, y_high = (float(v) for v in spec.y_limits)
    u = (xy[:, 0] - x_low) / (x_high - x_low)
    if spec.x_reverse:
        u = 1.0 - u
    v = (xy[:, 1] - y_low) / (y_high - y_low)
    return origin + u[:, None] * x_vec + v[:, None] * y_vec


def _segments(starts, ends):
    """Straight cubic Bezier curves from `starts` to `ends`, as Manim points."""
    delta = ends - starts
    curves = np.stack(
        [starts, starts + delta / 3, starts + 2 * delta / 3, ends], axis=1
    )
    return curves.reshape(-1, 3)


def _polyline(points):
    """Manim points for a polyline that breaks at every NaN vertex."""
    finite = np.all(np.isfinite(points), axis=1)
    keep = finite[:-1] & finite[1:]
    if not np.any(keep):
        return np.zeros((0, 3))
    return _segments(points[:-1][keep], points[1:][keep])


def _unit_shapes():
    """Unit Bezier outlines (radius about 1) for each SourceCast marker code."""
    plus = [
        (1, 1 / 3),
        (1 / 3, 1 / 3),
        (1 / 3, 1),
        (-1 / 3, 1),
        (-1 / 3, 1 / 3),
        (-1, 1 / 3),
        (-1, -1 / 3),
        (-1 / 3, -1 / 3),
        (-1 / 3, -1),
        (1 / 3, -1),
        (1 / 3, -1 / 3),
        (1, -1 / 3),
    ]
    shapes = {
        "o": manim.Circle(radius=1.0),
        "s": manim.Square(side_length=1.6),
        "D": manim.Square(side_length=1.6).rotate(manim.PI / 4),
        "^": manim.Triangle().scale_to_fit_height(2.0),
        "v": manim.Triangle().scale_to_fit_height(2.0).rotate(manim.PI),
        "P": manim.Polygon(*[(x, y, 0.0) for x, y in plus]),
    }
    return {code: shape.center().points.copy() for code, shape in shapes.items()}


_UNIT_CIRCLE = manim.Circle(radius=1.0).points[:, :2].copy()
_MARKERS = _unit_shapes()


def _region_points(region, spec, basis):
    """Manim points of a circle, or of an annulus with its hole reversed."""
    center = np.asarray(region.center, dtype=float)
    outer = _to_scene(center + region.outer_radius * _UNIT_CIRCLE, spec, basis)
    if region.inner_radius == 0.0:
        return outer
    inner = _to_scene(center + region.inner_radius * _UNIT_CIRCLE, spec, basis)
    return np.concatenate([outer, inner[::-1]])


def _reference_points(line, dashes, spec, basis):
    """Manim points of `dashes` equal dashes spanning the panel at `line.value`."""
    starts = np.arange(dashes) / dashes
    ends = starts + 0.5 / dashes
    if line.axis == "x":
        low, high = (float(v) for v in spec.y_limits)
        fixed, index = line.value, 1
    else:
        low, high = (float(v) for v in spec.x_limits)
        fixed, index = line.value, 0
    data = np.full((2 * dashes, 2), fixed)
    data[:, index] = low + (high - low) * np.concatenate([starts, ends])
    points = _to_scene(data, spec, basis)
    return _segments(points[:dashes], points[dashes:])


def _marker_points(template, center, radius):
    """A marker outline around `center`, or no points for a gap sample."""
    if not np.all(np.isfinite(center)):
        return np.zeros((0, 3))
    return template * radius + center


def _error_points(points_mark, name, spec, basis):
    """One straight segment per sample; an empty array for each gap row."""
    xy = np.asarray(points_mark.xy, dtype=float)
    err = np.asarray(getattr(points_mark, name), dtype=float)
    offset = np.zeros_like(xy)
    offset[:, 0 if name == "xerr" else 1] = err
    low = _to_scene(xy - offset, spec, basis)
    high = _to_scene(xy + offset, spec, basis)
    segments = []
    for start, end in zip(low, high, strict=True):
        if np.all(np.isfinite(start)) and np.all(np.isfinite(end)):
            segments.append(_segments(start[None], end[None]))
        else:
            segments.append(np.zeros((0, 3)))
    return segments


# --- Ticks and text ---------------------------------------------------------


def _nice_ticks(low, high):
    """About `_TICK_TARGET` round tick values in [low, high], and their text."""
    raw = (high - low) / _TICK_TARGET
    exponent = math.floor(math.log10(raw))
    for mantissa in (1.0, 2.0, 2.5, 5.0, 10.0):
        if mantissa * 10.0**exponent >= raw:
            break
    step = mantissa * 10.0**exponent
    decimals = max(0, -exponent + (mantissa == 2.5) - (mantissa == 10.0))
    first = math.ceil(low / step - 1e-9)
    last = math.floor(high / step + 1e-9)
    values = [k * step for k in range(first, last + 1)]
    return values, [f"{v:.{decimals}f}" for v in values]


def _text(text, profile, font_size):
    return manim.Text(
        text,
        font=profile.font_family,
        font_size=font_size,
        color=profile.text_color,
    )


def _label_outline(label, profile, scale):
    """(Bezier points of the label text, baseline drop per unit scale).

    Every glyph outline of a laid-out `Text` goes into one points array,
    so a label is drawn by one `VMobject` whose points change with the
    text. A scene flattens the moving mobjects' families when an animation
    begins, so swapping glyph submobjects mid-play would leave the old
    glyphs drawn; rewriting the points of one fixed mobject does not.

    A data-space label sits on its baseline like Matplotlib's
    `va="baseline"`, but a Manim `Text` only knows its glyph box. Laying
    out the same text with a trailing "x" (which sits on the baseline)
    gives how far the glyph box bottom lies above the baseline.
    """
    glyphs = _text(label.text, profile, profile.text_size_pt)
    drop = 0.0
    if label.space == "data" and len(glyphs.submobjects):
        probe = _text(label.text + "x", profile, profile.text_size_pt)
        body_bottom = min(g.get_bottom()[1] for g in probe.submobjects[:-1])
        drop = body_bottom - probe.submobjects[-1].get_bottom()[1]
    glyphs.scale(scale)
    outlines = [g.points for g in glyphs.family_members_with_points()]
    points = np.concatenate(outlines) if outlines else np.zeros((0, 3))
    return points, drop


def _label_offset(points, label, spec, basis, drop):
    """The shift that puts outline `points` at `label`'s anchor on the frame."""
    if not len(points):
        return np.zeros(3)
    origin, x_vec, y_vec = basis
    left, bottom = points[:, 0].min(), points[:, 1].min()
    if label.space == "panel":
        fx, fy = label.xy
        corner = np.array([left, points[:, 1].max(), 0.0])
        return origin + fx * x_vec + fy * y_vec - corner
    target = _to_scene(np.array([label.xy]), spec, basis)[0]
    return target - np.array([left, bottom - drop, 0.0])


# --- Panel state (kept on the result, never on a mobject) --------------------


class _Panel:
    """Renderer bookkeeping for one drawn leaf view."""

    def __init__(self, view, frame, width):
        self.spec = view.axes
        self.frame = frame
        self.width = width
        self.dashes = {}
        self.label_drops = {}
        self.label_texts = {}
        self.markers = {}
        self.image_key = None
        self.image_arrays = None
        self.weights, self.alphas = path_alphas(view)

    def set_weights(self, weights, alphas):
        self.weights, self.alphas = weights, alphas

    def basis(self):
        return _basis(self.frame)

    def scale(self):
        """Current frame width over its width at render time."""
        return float(np.linalg.norm(self.basis()[1])) / self.width


def _display_pixels(rgba, spec):
    """RGBA rows ordered for display: data row 0 at the bottom, RA reversed."""
    pixels = rgba[::-1]
    if spec.x_reverse:
        pixels = pixels[:, ::-1]
    return np.ascontiguousarray(pixels)


def _raster(pixels, width, height):
    image = manim.ImageMobject(pixels)
    image.set_resampling_algorithm(manim.RESAMPLING_ALGORITHMS["nearest"])
    image.stretch_to_fit_width(width)
    image.stretch_to_fit_height(height)
    return image


def _axis_decoration(view, frame, profile):
    """Tick marks, tick labels, and axis labels around `frame`."""
    spec = view.axes
    basis = _basis(frame)
    stroke = profile.stroke_width_pt
    tick_size = profile.text_size_pt * _TICK_FONT_SCALE
    # Data coordinates of the displayed bottom and left edges.
    bottom = float(spec.y_limits[0])
    left = float(spec.x_limits[1] if spec.x_reverse else spec.x_limits[0])
    ticks = manim.VGroup()
    x_labels = manim.VGroup()
    y_labels = manim.VGroup()
    for axis, labels in (("x", x_labels), ("y", y_labels)):
        limits = spec.x_limits if axis == "x" else spec.y_limits
        low, high = (float(v) for v in limits)
        values, texts = _nice_ticks(low, high)
        for value, text in zip(values, texts, strict=True):
            data = (value, bottom) if axis == "x" else (left, value)
            base = _to_scene(np.array([data]), spec, basis)[0]
            inward = manim.UP if axis == "x" else manim.RIGHT
            ticks.add(
                manim.Line(
                    base,
                    base + _TICK_LENGTH * inward,
                    stroke_width=stroke,
                    color=profile.text_color,
                )
            )
            if spec.show_ticks:
                glyphs = _text(text, profile, tick_size)
                glyphs.next_to(base, -inward, buff=0.1)
                labels.add(glyphs)
    decoration = manim.VGroup(ticks, x_labels, y_labels)
    if spec.x_label:
        x_title = _text(spec.x_label, profile, profile.text_size_pt)
        x_title.next_to(manim.VGroup(frame, x_labels), manim.DOWN, buff=0.15)
        decoration.add(x_title)
    if spec.y_label:
        y_title = _text(spec.y_label, profile, profile.text_size_pt)
        y_title.rotate(manim.PI / 2)
        y_title.next_to(manim.VGroup(frame, y_labels), manim.LEFT, buff=0.15)
        decoration.add(y_title)
    return decoration


def _colorbar(view, frame, profile):
    """The colormap strip of `view`'s scale, with its bounds and quantity."""
    scale = view.scale
    lut = profile.colormaps[scale.cmap_role]
    strip = _raster(
        np.ascontiguousarray(lut[::-1, None, :]), _COLORBAR_WIDTH, frame.height
    )
    strip.next_to(frame, manim.RIGHT, buff=_COLORBAR_BUFF)
    outline = manim.SurroundingRectangle(
        strip,
        buff=0,
        color=profile.text_color,
        stroke_width=profile.stroke_width_pt,
    )
    tick_size = profile.text_size_pt * _TICK_FONT_SCALE
    top = _text(f"{scale.vmax:g}", profile, tick_size)
    top.next_to(strip, manim.RIGHT, buff=0.1).align_to(strip, manim.UP)
    bottom = _text(f"{scale.vmin:g}", profile, tick_size)
    bottom.next_to(strip, manim.RIGHT, buff=0.1).align_to(strip, manim.DOWN)
    parts = [strip, outline, top, bottom]
    if view.quantity:
        quantity = _text(view.quantity, profile, profile.text_size_pt)
        quantity.rotate(-manim.PI / 2)
        quantity.next_to(manim.Group(top, bottom, strip), manim.RIGHT, buff=0.15)
        parts.append(quantity)
    return manim.Group(*parts)


def _draw_panel(view, styles, rgba, profile, parts):
    """Build one leaf view's Group, recording every part; return (group, state)."""
    width, height = _panel_size(view)
    stroke = profile.stroke_width_pt
    frame = manim.Rectangle(
        width=width,
        height=height,
        color=profile.text_color,
        stroke_width=stroke,
    )
    state = _Panel(view, frame, width)
    spec = view.axes
    basis = state.basis()
    parts[f"{view.id}/frame"] = frame
    decoration = _axis_decoration(view, frame, profile)
    parts[f"{view.id}/axes"] = decoration
    layers = [frame, decoration]
    if isinstance(view, ImageView):
        image = _raster(_display_pixels(rgba, spec), width, height)
        image.move_to(frame)
        parts[view.id] = image
        state.image_key = (array_key(view.data), array_key(view.valid))
        state.image_arrays = (view.data, view.valid)
        colorbar = _colorbar(view, frame, profile)
        parts[f"{view.id}/colorbar"] = colorbar
        layers = [image, *layers, colorbar]

    ordered = {_REGION: [], _LINE: [], _TOP: []}
    _, alphas = path_alphas(view)
    for mark in view.marks:
        style = styles.get(mark.id, {})
        if isinstance(mark, Path):
            mob = manim.VMobject(
                stroke_color=style["color"],
                stroke_width=stroke,
                stroke_opacity=alphas[mark.id],
                fill_opacity=0.0,
            )
            mob.set_points(_polyline(_to_scene(visible_xy(mark), spec, basis)))
            parts[mark.id] = mob
            ordered[_LINE].append(mob)
        elif isinstance(mark, Points):
            template = _MARKERS[style["marker"]]
            state.markers[mark.id] = template
            centers = _to_scene(mark.xy, spec, basis)
            markers = manim.VGroup(
                *(
                    manim.VMobject(
                        fill_color=style["color"], fill_opacity=1.0, stroke_width=0
                    )
                    for _ in centers
                )
            )
            for marker, center in zip(markers, centers, strict=True):
                marker.set_points(_marker_points(template, center, _MARKER_RADIUS))
            parts[mark.id] = markers
            ordered[_TOP].append(markers)
            for name in ("xerr", "yerr"):
                if getattr(mark, name) is None:
                    continue
                bars = manim.VGroup(
                    *(
                        manim.VMobject(
                            stroke_color=style["color"],
                            stroke_width=stroke,
                            fill_opacity=0.0,
                        )
                        for _ in centers
                    )
                )
                segments = _error_points(mark, name, spec, basis)
                for bar, points in zip(bars, segments, strict=True):
                    bar.set_points(points)
                parts[f"{mark.id}/{name}"] = bars
                ordered[_LINE].append(bars)
        elif isinstance(mark, Region):
            mob = manim.VMobject(
                fill_color=style["color"],
                fill_opacity=region_fill_opacity(view),
                stroke_color=style["color"],
                stroke_width=stroke,
                stroke_opacity=1.0,
            )
            mob.set_points(_region_points(mark, spec, basis))
            parts[mark.id] = mob
            ordered[_REGION].append(mob)
        elif isinstance(mark, ReferenceLine):
            length = height if mark.axis == "x" else width
            dashes = max(1, round(length / (2 * _DASH_LENGTH)))
            state.dashes[mark.id] = dashes
            mob = manim.VMobject(
                stroke_color=style["color"], stroke_width=stroke, fill_opacity=0.0
            )
            mob.set_points(_reference_points(mark, dashes, spec, basis))
            parts[mark.id] = mob
            ordered[_LINE].append(mob)
        else:
            points, drop = _label_outline(mark, profile, 1.0)
            glyphs = manim.VMobject(
                fill_color=profile.text_color, fill_opacity=1.0, stroke_width=0
            )
            glyphs.set_points(points + _label_offset(points, mark, spec, basis, drop))
            state.label_drops[mark.id] = drop
            state.label_texts[mark.id] = mark.text
            wrapper = manim.VGroup(glyphs)
            parts[mark.id] = wrapper
            ordered[_TOP].append(wrapper)

    group = manim.Group(*layers, *ordered[_REGION], *ordered[_LINE], *ordered[_TOP])
    if not isinstance(view, ImageView):
        parts[view.id] = group
    return group, state


def _build(view, styles, rgbas, profile, parts, panels, states):
    """Recursively build `view`, arranging panel groups by direction."""
    if isinstance(view, PanelGroup):
        children = [
            _build(child, styles, rgbas, profile, parts, panels, states)
            for child in view.views
        ]
        direction = manim.RIGHT if view.direction == "row" else manim.DOWN
        group = manim.Group(*children).arrange(direction, buff=_PANEL_BUFF)
        parts[view.id] = group
        return group
    group, state = _draw_panel(view, styles, rgbas.get(view.id), profile, parts)
    panels[view.id] = group
    states[view.id] = state
    return group


# --- Result --------------------------------------------------------------------


class ManimResult:
    """A rendered prepared view: its root mobject and parts by element ID.

    Attributes:
        mobject: The root `manim.Group` (raster and vector parts together).
            Add it to a scene, move it, and scale it like any mobject.
        parts: Read-only mapping from element ID to its mobject. An
            `ImageView` ID maps to its `ImageMobject`, a `CurveView` or
            `TrackView` ID to its panel `Group`, a `PanelGroup` ID to the
            arranged `Group`, a `Path` or `ReferenceLine` to a `VMobject`,
            a `Region` to a filled `VMobject` (an annulus has a hole), a
            `Points` to a `VGroup` with one marker per sample, and a `Label`
            to a stable `VGroup` wrapper whose single child is one
            `VMobject` holding every glyph outline of the current text
            (its points are rewritten when the text changes; the mobject
            itself is never replaced, so a playing scene redraws it).
            Derived parts use
            suffixed keys: `"<view id>/frame"` (the axes box, whose
            corners are the axis limits), `"<view id>/axes"` (ticks and
            axis labels), `"<image id>/colorbar"`, and
            `"<points id>/xerr"`/`"<points id>/yerr"` (one segment per
            sample).
        panels: Read-only mapping from each leaf view's ID to its panel
            `Group`, in depth-first order.
        profile: The `RenderProfile` this result was rendered with. Updates
            reuse it; they never take a new snapshot.
        cast: The `SourceCast` this result was rendered with.
        view: The view tree currently displayed.
    """

    def __init__(self, mobject, parts, panels, profile, cast, view, states):
        """Bind a rendered tree; built by `render`, not called directly."""
        self.mobject = mobject
        self.parts = MappingProxyType(parts)
        self.panels = MappingProxyType(panels)
        self.profile = profile
        self.cast = cast
        self.view = view
        self._topology = topology(view)
        self._states = states

    def update(self, view):
        """Show a new state of the same tree, reusing every mobject.

        The whole of `view` is validated against the rendered topology, and
        every new image buffer, geometry array, and label outline is
        computed, before any mobject changes, so an invalid update leaves
        every part exactly as it was. An image whose data and validity
        arrays are the same borrowed storage as the one displayed is not
        re-mapped. No scientific calculation happens here.

        Args:
            view: A prepared view with the same topology as the rendered
                one (see `eyepiece.mpl.MplResult.update`): values, path
                visible intervals, point and region positions, reference
                values, label text, and path weights may change.

        Raises:
            TypeError: If `view` is not a prepared view.
            ValueError: If `view` differs in topology, naming the element.
        """
        if not isinstance(view, VIEW_TYPES):
            raise TypeError(f"update takes a prepared view, got {type(view).__name__}")
        check_topology(self._topology, topology(view))
        pending = []
        for leaf in leaf_views(view):
            pending.extend(self._leaf_changes(leaf, self._states[leaf.id]))
        for apply in pending:
            apply()
        self.view = view

    def _leaf_changes(self, leaf, state):
        """Zero-argument callables that bring one panel's parts to `leaf`."""
        changes = []
        spec = state.spec
        basis = state.basis()
        scale = state.scale()
        if isinstance(leaf, ImageView):
            key = (array_key(leaf.data), array_key(leaf.valid))
            if key != state.image_key:
                rgba = map_rgba(
                    leaf.data, valid=leaf.valid, scale=leaf.scale, profile=self.profile
                )
                pixels = _display_pixels(rgba, spec)
                arrays = (leaf.data, leaf.valid)
                changes.append(
                    lambda: _set_pixels(self.parts[leaf.id], pixels, state, key, arrays)
                )

        weights, alphas = path_alphas(leaf)
        if weights != state.weights:
            for mark_id, alpha in alphas.items():
                factor = alpha / state.alphas[mark_id]
                changes.append(_rescale_stroke(self.parts[mark_id], factor))
            changes.append(lambda: state.set_weights(weights, alphas))

        for mark in leaf.marks:
            part = self.parts[mark.id]
            if isinstance(mark, Path):
                points = _polyline(_to_scene(visible_xy(mark), spec, basis))
                changes.append(_setter(part, points))
            elif isinstance(mark, Points):
                template = state.markers[mark.id]
                centers = _to_scene(mark.xy, spec, basis)
                radius = _MARKER_RADIUS * scale
                for marker, center in zip(part, centers, strict=True):
                    changes.append(
                        _setter(marker, _marker_points(template, center, radius))
                    )
                for name in ("xerr", "yerr"):
                    if getattr(mark, name) is None:
                        continue
                    segments = _error_points(mark, name, spec, basis)
                    bars = self.parts[f"{mark.id}/{name}"]
                    for bar, points in zip(bars, segments, strict=True):
                        changes.append(_setter(bar, points))
            elif isinstance(mark, Region):
                changes.append(_setter(part, _region_points(mark, spec, basis)))
            elif isinstance(mark, ReferenceLine):
                dashes = state.dashes[mark.id]
                points = _reference_points(mark, dashes, spec, basis)
                changes.append(_setter(part, points))
            else:
                changes.append(self._label_change(mark, part, state, basis, scale))
        return changes

    def _label_change(self, mark, wrapper, state, basis, scale):
        """New outline points (laid out only when the text changed), placed."""
        glyphs = wrapper.submobjects[0]
        if mark.text != state.label_texts[mark.id]:
            points, drop = _label_outline(mark, self.profile, scale)
        else:
            points, drop = glyphs.points, state.label_drops[mark.id]
        placed = points + _label_offset(points, mark, state.spec, basis, drop * scale)

        def apply():
            glyphs.set_points(placed)
            state.label_texts[mark.id] = mark.text
            state.label_drops[mark.id] = drop

        return apply


def _setter(mob, points):
    def apply():
        mob.set_points(points)

    return apply


def _rescale_stroke(mob, factor):
    def apply():
        mob.set_stroke(opacity=mob.get_stroke_opacity() * factor)

    return apply


def _set_pixels(image, pixels, state, key, arrays):
    """Rewrite an image's pixel buffer in place, keeping the caller's opacity.

    `ImageMobject.set_opacity(a)` records `a` as its stroke opacity and
    scales the source alpha into the buffer, so a new sample is written the
    same way: its own alpha times the opacity the caller last set.
    """
    image.orig_alpha_pixel_array = pixels[..., 3].copy()
    image.pixel_array[..., :3] = pixels[..., :3]
    image.pixel_array[..., 3] = image.orig_alpha_pixel_array * image.stroke_opacity
    state.image_key = key
    state.image_arrays = arrays


# --- Public entry point ----------------------------------------------------------


def render(view, *, cast=None, profile=None):
    """Render a prepared view tree as native Manim mobjects.

    Every element is validated, every appearance resolved, and every image
    mapped before any mobject is built, so a rejected view builds nothing.
    A `PanelGroup` arranges its children in a row or column; the result is
    centered at the scene origin and may be moved and scaled freely.

    Args:
        view: An `ImageView`, `CurveView`, `TrackView`, or `PanelGroup`.
        cast: `SourceCast` assigning colors and markers to source IDs.
            None builds one from the tree's source IDs in first-encounter
            (depth-first) order.
        profile: `RenderProfile` supplying colors, colormap tables, and
            text settings. None takes `eyepiece.style.snapshot_profile()`
            once, now, with its talk-sized defaults (24 pt text, 1.5 pt
            strokes; the Matplotlib rc sizes are not used); the result
            keeps it for every later update. Its `text_size_pt` is Manim's
            `font_size` and its `stroke_width_pt` Manim's `stroke_width`.
            A `Region` on an `ImageView` is an outline only; on a curve or
            track panel its interior is shaded.

    Returns:
        A `ManimResult`.

    Raises:
        TypeError: If `view` is not a prepared view.
        ValueError: If a source ID is not in `cast`, a region or reference
            line has an unsupported role, a colormap role is not in
            `profile`, an axis aspect is not usable, or a derived part key
            collides with an element ID. Each error names the element.
    """
    if not isinstance(view, VIEW_TYPES):
        raise TypeError(f"render takes a prepared view, got {type(view).__name__}")
    profile = snapshot_profile() if profile is None else profile
    cast = default_cast(view) if cast is None else cast
    styles = resolve_styles(
        view, cast, profile, renderer="Manim", leaf_parts=_LEAF_PARTS
    )
    leaves = leaf_views(view)
    for leaf in leaves:
        _panel_size(leaf)
    rgbas = {
        leaf.id: map_rgba(
            leaf.data, valid=leaf.valid, scale=leaf.scale, profile=profile
        )
        for leaf in leaves
        if isinstance(leaf, ImageView)
    }
    parts, panels, states = {}, {}, {}
    root = _build(view, styles, rgbas, profile, parts, panels, states)
    root.move_to(manim.ORIGIN)
    return ManimResult(root, parts, panels, profile, cast, view, states)
