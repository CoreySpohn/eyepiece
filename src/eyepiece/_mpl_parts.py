"""Drawing helpers for `eyepiece.mpl`: appearance, geometry, and artists.

Everything here turns one already-validated prepared element into
Matplotlib state -- a resolved color, a segment array, an artist -- and
none of it decides whether an update is allowed; that topology check lives
beside `MplResult` in `eyepiece.mpl`. Private: not part of the public API.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm, Normalize, to_rgba
from matplotlib.patches import Annulus, Circle

from eyepiece.images import _attach_colorbar
from eyepiece.prepared import (
    CurveView,
    ImageView,
    PanelGroup,
    Path,
    Points,
    ReferenceLine,
    Region,
    TrackView,
    weight_opacity,
)
from eyepiece.style import SourceCast

LEAF_TYPES = (ImageView, CurveView, TrackView)

# Fill opacity of a Region's interior; its outline is drawn fully opaque.
_REGION_FILL_ALPHA = 0.2

_SUPPORTED_ROLES = ("reference",)


# --- Tree walking ---------------------------------------------------------------


def iter_elements(element):
    """Yield every view and mark in `element`'s tree, depth first."""
    yield element
    if isinstance(element, PanelGroup):
        for child in element.views:
            yield from iter_elements(child)
    elif isinstance(element, LEAF_TYPES):
        yield from element.marks


def leaf_views(view):
    """The drawable (non-group) views of a tree, depth first."""
    return [e for e in iter_elements(view) if isinstance(e, LEAF_TYPES)]


# --- Appearance resolution (all checks before anything is drawn) ------------


def default_cast(view):
    """A SourceCast over the tree's source IDs in first-encounter order."""
    names = []
    for element in iter_elements(view):
        source_id = getattr(element, "source_id", None)
        if source_id is not None and source_id not in names:
            names.append(source_id)
    return SourceCast(tuple(names))


def _source_color(mark, cast, profile):
    if mark.source_id is None:
        return profile.text_color
    try:
        slot = cast.slot(mark.source_id)
    except KeyError:
        raise ValueError(
            f"{mark.id}: source_id {mark.source_id!r} is not in the cast {cast.names!r}"
        ) from None
    return profile.colors[slot % len(profile.colors)]


def _source_marker(mark, cast):
    return "o" if mark.source_id is None else cast.marker(mark.source_id)


def _role_color(mark, profile):
    if mark.role not in _SUPPORTED_ROLES:
        raise ValueError(
            f"{mark.id}: role {mark.role!r} is not supported by the Matplotlib "
            f"renderer; use one of {_SUPPORTED_ROLES}"
        )
    return profile.reference_color


def resolve_styles(view, cast, profile):
    """Resolve every element's appearance, raising before anything is drawn.

    Returns:
        A dict mapping element ID to its style dict.
    """
    ids = {element.id for element in iter_elements(view)}
    styles = {}
    for element in iter_elements(view):
        derived = ()
        if isinstance(element, ImageView):
            role = element.scale.cmap_role
            if role not in profile.colormaps:
                raise ValueError(
                    f"{element.id}: colormap role {role!r} is not in the render profile"
                )
            derived = (f"{element.id}/colorbar",)
        elif isinstance(element, (Path, Points)):
            styles[element.id] = {
                "color": _source_color(element, cast, profile),
                "marker": _source_marker(element, cast),
            }
            if isinstance(element, Points):
                derived = tuple(
                    f"{element.id}/{name}"
                    for name in ("xerr", "yerr")
                    if getattr(element, name) is not None
                )
        elif isinstance(element, (Region, ReferenceLine)):
            styles[element.id] = {"color": _role_color(element, profile)}
        elif isinstance(element, PanelGroup) and not element.views:
            raise ValueError(f"{element.id}: panel group has no views to draw")
        for key in derived:
            if key in ids:
                raise ValueError(
                    f"{key}: element ID collides with the rendered part "
                    f"{key!r} of {element.id!r}"
                )
    return styles


def path_alphas(leaf):
    """Opacity per Path on one panel, through the shared weight mapping."""
    paths = [m for m in leaf.marks if isinstance(m, Path)]
    weights = tuple(float(p.weight) for p in paths)
    alphas = weight_opacity(np.asarray(weights, dtype=float))
    return weights, {p.id: float(a) for p, a in zip(paths, alphas, strict=True)}


# --- Geometry helpers ----------------------------------------------------------


def visible_xy(path):
    """The revealed `(start, stop)` slice of a Path's vertices (a view)."""
    start, stop = path.visible
    return path.xy[start:stop]


def error_segments(points, name):
    """(P, 2, 2) error-bar segments for `points`' `xerr` or `yerr`."""
    xy = np.asarray(points.xy, dtype=float)
    err = np.asarray(getattr(points, name), dtype=float)
    offset = np.zeros_like(xy)
    offset[:, 0 if name == "xerr" else 1] = err
    return np.stack([xy - offset, xy + offset], axis=1)


def array_key(arr):
    """Identity of borrowed storage: address, shape, strides, dtype."""
    if arr is None:
        return None
    interface = arr.__array_interface__
    return (interface["data"][0], arr.shape, arr.strides, arr.dtype.str)


def _norm(scale):
    if scale.kind == "log":
        return LogNorm(vmin=scale.vmin, vmax=scale.vmax)
    return Normalize(vmin=scale.vmin, vmax=scale.vmax)


def _lut_colormap(scale, profile):
    lut = profile.colormaps[scale.cmap_role]
    colormap = ListedColormap(lut / 255.0, name=f"eyepiece-{scale.cmap_role}")
    return colormap.with_extremes(bad=np.asarray(profile.bad_rgba) / 255.0)


def _apply_axis_spec(ax, spec):
    x_low, x_high = (float(v) for v in spec.x_limits)
    ax.set_xlim((x_high, x_low) if spec.x_reverse else (x_low, x_high))
    ax.set_ylim(tuple(float(v) for v in spec.y_limits))
    ax.set_aspect(spec.aspect)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
    if not spec.show_ticks:
        ax.tick_params(labelbottom=False, labelleft=False)


# --- Axes placement -----------------------------------------------------------


def root_figure(ax):
    """The top-level Figure that owns `ax`, even inside a SubFigure."""
    try:
        return ax.get_figure(root=True)
    except TypeError:  # matplotlib < 3.10 has no `root`; `.figure` is the root
        return ax.figure


def owned_axes(view):
    """A new constrained-layout figure with one axes per leaf, nested by group."""
    fig = plt.figure(layout="constrained")
    axes = []

    def place(element, spec):
        if isinstance(element, PanelGroup):
            count = len(element.views)
            shape = (1, count) if element.direction == "row" else (count, 1)
            grid = (
                fig.add_gridspec(*shape) if spec is None else spec.subgridspec(*shape)
            )
            for index, child in enumerate(element.views):
                place(child, grid[index])
        else:
            axes.append(fig.add_subplot(111 if spec is None else spec))

    place(view, None)
    return fig, axes


def resolve_axes(view, leaves, ax, axes):
    """Check the caller's axes against the tree, without drawing anything."""
    if ax is not None and axes is not None:
        raise ValueError(f"{view.id}: pass ax or axes, not both")
    if ax is not None:
        if len(leaves) != 1:
            raise ValueError(
                f"{view.id}: ax= takes a single panel, this tree has "
                f"{len(leaves)}; pass axes= with one Axes per panel"
            )
        return [ax]
    if axes is not None:
        axes = list(np.asarray(axes, dtype=object).ravel())
        if len(axes) != len(leaves):
            raise ValueError(
                f"{view.id}: axes= has {len(axes)} Axes for {len(leaves)} panels"
            )
        figures = {id(root_figure(a)) for a in axes}
        if len(figures) != 1:
            raise ValueError(f"{view.id}: every Axes must belong to one figure")
        return axes
    return None


# --- Drawing -------------------------------------------------------------------


def draw_leaf(ax, leaf, styles, rgba, profile, parts):
    """Draw one leaf view and its marks onto `ax`, recording every part."""
    stroke = profile.stroke_width_pt
    if isinstance(leaf, ImageView):
        x_low, x_high = (float(v) for v in leaf.axes.x_limits)
        y_low, y_high = (float(v) for v in leaf.axes.y_limits)
        parts[leaf.id] = ax.imshow(
            rgba,
            extent=(x_low, x_high, y_low, y_high),
            origin="lower",
            interpolation="nearest",
        )
        mappable = ScalarMappable(
            norm=_norm(leaf.scale), cmap=_lut_colormap(leaf.scale, profile)
        )
        parts[f"{leaf.id}/colorbar"] = _attach_colorbar(
            ax, mappable, True, leaf.quantity, None
        )
    else:
        parts[leaf.id] = ax

    _, alphas = path_alphas(leaf)
    for mark in leaf.marks:
        style = styles.get(mark.id, {})
        if isinstance(mark, Path):
            xy = visible_xy(mark)
            (parts[mark.id],) = ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=style["color"],
                alpha=alphas[mark.id],
                linewidth=stroke,
                label=mark.label,
            )
        elif isinstance(mark, Points):
            parts[mark.id] = ax.scatter(
                mark.xy[:, 0],
                mark.xy[:, 1],
                color=style["color"],
                marker=style["marker"],
                label=mark.label,
                zorder=3,
            )
            for name in ("xerr", "yerr"):
                if getattr(mark, name) is None:
                    continue
                collection = LineCollection(
                    error_segments(mark, name),
                    colors=style["color"],
                    linewidths=stroke,
                )
                parts[f"{mark.id}/{name}"] = ax.add_collection(
                    collection, autolim=False
                )
        elif isinstance(mark, Region):
            color = style["color"]
            common = {
                "facecolor": to_rgba(color, _REGION_FILL_ALPHA),
                "edgecolor": to_rgba(color, 1.0),
                "linewidth": stroke,
                "label": mark.label,
            }
            center = tuple(float(v) for v in mark.center)
            if mark.inner_radius == 0.0:
                patch = Circle(center, mark.outer_radius, **common)
            else:
                patch = Annulus(
                    center,
                    mark.outer_radius,
                    mark.outer_radius - mark.inner_radius,
                    **common,
                )
            parts[mark.id] = ax.add_patch(patch)
        elif isinstance(mark, ReferenceLine):
            line = ax.axvline if mark.axis == "x" else ax.axhline
            parts[mark.id] = line(
                mark.value,
                color=style["color"],
                linewidth=stroke,
                linestyle="--",
                label=mark.label,
            )
        else:
            panel = mark.space == "panel"
            parts[mark.id] = ax.text(
                *mark.xy,
                mark.text,
                transform=ax.transAxes if panel else ax.transData,
                color=profile.text_color,
                fontsize=profile.text_size_pt,
                fontfamily=profile.font_family,
                ha="left",
                va="top" if panel else "baseline",
            )
    _apply_axis_spec(ax, leaf.axes)
