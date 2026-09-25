"""Matplotlib renderer for prepared views and sequences.

`render` turns a prepared view tree (`eyepiece.prepared`) into Matplotlib
artists and returns an `MplResult` whose `.parts` exposes every artist by
the element ID it was drawn from; `animate` plays a prepared `Sequence`
through the existing multi-sink `eyepiece.Animation`, one output frame per
entry of the sequence's shared output schedule.

The renderer computes no science. Image colors come from
`eyepiece.prepared.map_rgba` under the view's own `Scale` and the render
profile's colormap table, and the colorbar is a scalar mappable over that
same scale and table, so the image and its colorbar cannot disagree. Every
mark on a panel is drawn in that panel's data coordinates, so paths, points,
their error bars, and regions stay aligned when the x axis is reversed (the
increasing-RA-left convention reverses the axis display once; the data and
the image extent are never flipped).

A result is bound to the topology it was rendered from. `MplResult.update`
accepts a new state of the same tree -- same element IDs and kinds, image
shapes, scales, axes, point counts, and source identities -- and validates
the whole new tree before touching any artist, so an invalid update raises
without partially changing a panel group. Updates never add, remove, or
restyle artists, and never change an artist's visibility, so a part the
caller hid stays hidden.
"""

from types import MappingProxyType

import numpy as np
from matplotlib.patches import Annulus

from eyepiece._mpl_parts import (
    LEAF_TYPES,
    array_key,
    default_cast,
    draw_leaf,
    error_segments,
    iter_elements,
    leaf_views,
    owned_axes,
    path_alphas,
    resolve_axes,
    resolve_styles,
    root_figure,
    visible_xy,
)
from eyepiece.anim import animate as legacy_animate
from eyepiece.prepared import (
    CurveView,
    ImageView,
    PanelGroup,
    Path,
    Points,
    ReferenceLine,
    Region,
    TrackView,
    map_rgba,
)
from eyepiece.style import snapshot_profile

_VIEW_TYPES = (*LEAF_TYPES, PanelGroup)


# --- Topology -----------------------------------------------------------------


def _axes_key(spec):
    return (
        spec.x_label,
        spec.y_label,
        tuple(float(v) for v in spec.x_limits),
        tuple(float(v) for v in spec.y_limits),
        spec.aspect,
        bool(spec.x_reverse),
        bool(spec.show_ticks),
    )


def _topology_entry(element):
    """What an update may not change about one element, as a comparable tuple.

    Values, path visible intervals, mark positions, region geometry, label
    text, and path weights are state; everything returned here is topology.
    """
    if isinstance(element, ImageView):
        scale = element.scale
        detail = (
            element.data.shape,
            (
                scale.kind,
                float(scale.vmin),
                float(scale.vmax),
                scale.cmap_role,
                None if scale.floor is None else float(scale.floor),
            ),
            _axes_key(element.axes),
            element.quantity,
        )
    elif isinstance(element, (CurveView, TrackView)):
        detail = _axes_key(element.axes)
    elif isinstance(element, PanelGroup):
        detail = (element.direction, len(element.views))
    elif isinstance(element, Path):
        detail = (element.xy.shape[0], element.source_id)
    elif isinstance(element, Points):
        detail = (
            element.xy.shape[0],
            element.source_id,
            element.xerr is None,
            element.yerr is None,
        )
    elif isinstance(element, Region):
        detail = (element.inner_radius == 0.0, element.role)
    elif isinstance(element, ReferenceLine):
        detail = (element.axis, element.role)
    else:
        detail = (element.space,)
    return (element.id, type(element).__name__, detail)


def _topology(view):
    return tuple(_topology_entry(e) for e in iter_elements(view))


def _check_topology(expected, actual):
    """Raise a ValueError naming the first element whose topology changed."""
    if expected == actual:
        return
    for old, new in zip(expected, actual, strict=False):
        if old != new:
            if old[0] != new[0] or old[1] != new[1]:
                raise ValueError(
                    f"{new[0]}: update replaces {old[1]} {old[0]!r} with "
                    f"{new[1]} {new[0]!r}; a changed tree needs a new render"
                )
            raise ValueError(
                f"{new[0]}: update changes this {new[1]}'s topology "
                f"({old[2]!r} -> {new[2]!r}); a changed shape, scale, axes, "
                "point count, or source needs a new render"
            )
    # Every shared position matched, so one tree is a prefix of the other.
    if len(actual) > len(expected):
        element_id, verb = actual[len(expected)][0], "adds"
    else:
        element_id, verb = expected[len(actual)][0], "removes"
    raise ValueError(
        f"{element_id}: update {verb} this element; a changed tree needs a new render"
    )


# --- Result --------------------------------------------------------------------


class MplResult:
    """A rendered prepared view: its figure, axes, and parts by element ID.

    Attributes:
        fig: The Figure drawn on (the caller's, when `ax`/`axes` was given).
        axes: Read-only mapping from each drawn view's ID to its Axes, in
            depth-first panel order.
        parts: Read-only mapping from element ID to its artist. An
            `ImageView` ID maps to its `AxesImage`, a `CurveView`/`TrackView`
            ID to its Axes, a `Path` or `ReferenceLine` to a `Line2D`, a
            `Points` to a `PathCollection`, a `Region` to a `Circle` or
            `Annulus`, and a `Label` to a `Text`. Derived parts use suffixed
            keys: `"<image id>/colorbar"` (the `Colorbar`), and
            `"<points id>/xerr"`/`"<points id>/yerr"` (a `LineCollection`
            of error-bar segments).
        profile: The `RenderProfile` this result was rendered with. Updates
            reuse it; they never take a new snapshot.
        cast: The `SourceCast` this result was rendered with.
        view: The view tree currently displayed.
    """

    def __init__(self, fig, axes, parts, profile, cast, view, image_keys):
        """Bind a rendered tree; built by `render`, not called directly."""
        self.fig = fig
        self.axes = MappingProxyType(axes)
        self.parts = MappingProxyType(parts)
        self.profile = profile
        self.cast = cast
        self.view = view
        self._topology = _topology(view)
        # image id -> (storage key, (data, valid)). Holding the displayed
        # arrays keeps their addresses from being reused by new storage, so
        # an equal key really is the same borrowed sample.
        self._image_keys = image_keys
        self._weights = {leaf.id: path_alphas(leaf)[0] for leaf in leaf_views(view)}

    @property
    def ax(self):
        """The single Axes of a one-panel result.

        Raises:
            ValueError: If the result has more than one panel.
        """
        if len(self.axes) != 1:
            raise ValueError(
                f"{self.view.id}: this result has {len(self.axes)} panels; "
                "use .axes[<view id>]"
            )
        return next(iter(self.axes.values()))

    def update(self, view):
        """Show a new state of the same tree, reusing every artist.

        The whole of `view` is validated against the rendered topology, and
        every new image color table is computed, before any artist changes,
        so an invalid update leaves the figure exactly as it was. An image
        whose data and validity arrays are the same borrowed storage as
        the one displayed is not re-mapped.

        Args:
            view: A prepared view with the same topology as the rendered
                one: element IDs and kinds, image shapes, scales, axes,
                point counts, error-bar presence, source IDs, region kinds,
                reference-line axes, and label spaces. Values, path visible
                intervals, point and region positions, reference values,
                label text, and path weights may change.

        Raises:
            ValueError: If `view` differs in topology, naming the element.
        """
        _check_topology(self._topology, _topology(view))
        pending = []
        image_keys = dict(self._image_keys)
        for leaf in leaf_views(view):
            if isinstance(leaf, ImageView):
                key = (array_key(leaf.data), array_key(leaf.valid))
                if key != self._image_keys[leaf.id][0]:
                    rgba = map_rgba(
                        leaf.data,
                        valid=leaf.valid,
                        scale=leaf.scale,
                        profile=self.profile,
                    )
                    pending.append((self.parts[leaf.id].set_data, rgba))
                    image_keys[leaf.id] = (key, (leaf.data, leaf.valid))
            weights, alphas = path_alphas(leaf)
            if weights != self._weights[leaf.id]:
                pending.extend(
                    (self.parts[mark_id].set_alpha, alpha)
                    for mark_id, alpha in alphas.items()
                )
            for mark in leaf.marks:
                pending.extend(self._mark_changes(mark))

        for setter, value in pending:
            setter(value)
        self._image_keys = image_keys
        self._weights = {leaf.id: path_alphas(leaf)[0] for leaf in leaf_views(view)}
        self.view = view

    def _mark_changes(self, mark):
        """The `(setter, value)` pairs that bring one mark's artists to `mark`."""
        part = self.parts[mark.id]
        if isinstance(mark, Path):
            return [(part.set_data, visible_xy(mark).T)]
        if isinstance(mark, Points):
            changes = [(part.set_offsets, np.asarray(mark.xy, dtype=float))]
            for name in ("xerr", "yerr"):
                if getattr(mark, name) is not None:
                    changes.append(
                        (
                            self.parts[f"{mark.id}/{name}"].set_segments,
                            error_segments(mark, name),
                        )
                    )
            return changes
        if isinstance(mark, Region):
            center = tuple(float(v) for v in mark.center)
            if isinstance(part, Annulus):
                return [
                    (part.set_center, center),
                    (part.set_radii, mark.outer_radius),
                    (part.set_width, mark.outer_radius - mark.inner_radius),
                ]
            return [(part.set_center, center), (part.set_radius, mark.outer_radius)]
        if isinstance(mark, ReferenceLine):
            setter = part.set_xdata if mark.axis == "x" else part.set_ydata
            return [(setter, [mark.value, mark.value])]
        return [(part.set_text, mark.text), (part.set_position, mark.xy)]


# --- Public entry points --------------------------------------------------------


def render(view, *, ax=None, axes=None, cast=None, profile=None):
    """Render a prepared view tree with Matplotlib.

    Every element is validated and every appearance resolved before
    anything is drawn, so a rejected view leaves the caller's axes empty.
    Caller axes keep their place in their figure: the colorbar hangs in an
    inset beside its image rather than taking room from the axes, and no
    layout engine is installed on a caller's figure. Without `ax`/`axes`,
    a new constrained-layout figure is created, with panel groups laid out
    as nested rows and columns.

    Args:
        view: An `ImageView`, `CurveView`, `TrackView`, or `PanelGroup`.
        ax: One Axes to draw a single-panel view into.
        axes: A sequence (or array) of Axes, one per drawn panel in
            depth-first order, all on one figure. Mutually exclusive with
            `ax`.
        cast: `SourceCast` assigning colors and markers to source IDs.
            None builds one from the tree's source IDs in first-encounter
            (depth-first) order.
        profile: `RenderProfile` supplying colors, colormap tables, and
            text settings. None takes `eyepiece.style.snapshot_profile()`
            once, now; the result keeps it for every later update.

    Returns:
        An `MplResult`.

    Raises:
        TypeError: If `view` is not a prepared view.
        ValueError: If the axes do not match the tree, a source ID is not
            in `cast`, a region or reference line has an unsupported role,
            a colormap role is not in `profile`, or a derived part key
            collides with an element ID. Each error names the element.
    """
    if not isinstance(view, _VIEW_TYPES):
        raise TypeError(f"render takes a prepared view, got {type(view).__name__}")
    leaves = leaf_views(view)
    caller_axes = resolve_axes(view, leaves, ax, axes)
    profile = snapshot_profile() if profile is None else profile
    cast = default_cast(view) if cast is None else cast
    styles = resolve_styles(view, cast, profile)
    rgbas = {
        leaf.id: map_rgba(
            leaf.data, valid=leaf.valid, scale=leaf.scale, profile=profile
        )
        for leaf in leaves
        if isinstance(leaf, ImageView)
    }

    if caller_axes is None:
        fig, drawn_axes = owned_axes(view)
    else:
        fig, drawn_axes = root_figure(caller_axes[0]), caller_axes

    parts = {}
    axes_by_id = {}
    for leaf, leaf_ax in zip(leaves, drawn_axes, strict=True):
        axes_by_id[leaf.id] = leaf_ax
        draw_leaf(leaf_ax, leaf, styles, rgbas.get(leaf.id), profile, parts)

    image_keys = {
        leaf.id: (
            (array_key(leaf.data), array_key(leaf.valid)),
            (leaf.data, leaf.valid),
        )
        for leaf in leaves
        if isinstance(leaf, ImageView)
    }
    return MplResult(fig, axes_by_id, parts, profile, cast, view, image_keys)


def animate(sequence, *, run_time, fps=30, ax=None, axes=None, cast=None, profile=None):
    """Play a prepared `Sequence` through the multi-sink `eyepiece.Animation`.

    Output frames follow `sequence.schedule(run_time=run_time, fps=fps)`:
    `ceil(run_time * fps)` physical times spanning the first through the
    last sample, so the encoded duration is that count over `fps` and the
    final sample is always encoded. Each frame shows the left
    sample-and-hold state at its physical time, through
    `MplResult.update`, which re-maps an image only when the held sample
    changes.

    Args:
        sequence: A prepared `Sequence`.
        run_time: Presentation duration in seconds (before quantization to
            whole frames).
        fps: Output frame rate.
        ax: As for `render`.
        axes: As for `render`.
        cast: As for `render`.
        profile: As for `render`; snapshotted once when None.

    Returns:
        An `eyepiece.Animation`; nothing is encoded until `.save`,
        `.jshtml`, or `.video` is called.
    """
    output_times = sequence.schedule(run_time=run_time, fps=fps)
    result = render(sequence.frame(0), ax=ax, axes=axes, cast=cast, profile=profile)

    def draw(_fig, index):
        result.update(sequence.at(float(output_times[index])).view)

    return legacy_animate(result.fig, draw, len(output_times), fps=fps)


__all__ = ["MplResult", "animate", "render"]
