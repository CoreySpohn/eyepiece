"""Drawing helpers for `eyepiece.mpl`: appearance, geometry, and artists.

Everything here turns one already-validated prepared element into
Matplotlib state -- a resolved color, a segment array, an artist -- and
none of it decides whether an update is allowed; that topology check, the
tree walk, and style resolution are renderer-neutral and live in
`eyepiece._prepared_render`. Private: not part of the public API.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm, Normalize, to_rgba
from matplotlib.patches import Annulus, Circle

from eyepiece._prepared_render import (
    LABEL_HALO_EM,
    gap_nan,
    label_halo,
    mark_alphas,
    region_fill_opacity,
    visible_xy,
)
from eyepiece.images import _attach_colorbar
from eyepiece.prepared import (
    ImageView,
    PanelGroup,
    Path,
    Points,
    ReferenceLine,
    Region,
)

# --- Geometry helpers ----------------------------------------------------------


def point_offsets(points):
    """(P, 2) scatter offsets for `points`, with gap rows masked (hidden)."""
    return np.ma.masked_invalid(gap_nan(points.xy))


def error_segments(points, name):
    """(P, 2, 2) error-bar segments for `points`' `xerr` or `yerr`.

    A gap row's segment is all NaN, which Matplotlib does not draw.
    """
    xy = gap_nan(points.xy)
    err = np.asarray(getattr(points, name), dtype=float)
    offset = np.zeros_like(xy)
    offset[:, 0 if name == "xerr" else 1] = err
    return np.stack([xy - offset, xy + offset], axis=1)


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

    _, alphas = mark_alphas(leaf)
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
            offsets = point_offsets(mark)
            parts[mark.id] = ax.scatter(
                offsets[:, 0],
                offsets[:, 1],
                color=style["color"],
                marker=style["marker"],
                alpha=alphas[mark.id],
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
                    alpha=alphas[mark.id],
                )
                parts[f"{mark.id}/{name}"] = ax.add_collection(
                    collection, autolim=False
                )
        elif isinstance(mark, Region):
            color = style["color"]
            common = {
                "facecolor": to_rgba(color, region_fill_opacity(leaf)),
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
            halo = []
            if label_halo(leaf):
                halo = [
                    patheffects.withStroke(
                        linewidth=LABEL_HALO_EM * profile.text_size_pt,
                        foreground=profile.background_color,
                    )
                ]
            parts[mark.id] = ax.text(
                *mark.xy,
                mark.text,
                transform=ax.transAxes if panel else ax.transData,
                color=profile.text_color,
                fontsize=profile.text_size_pt,
                fontfamily=profile.font_family,
                ha="left",
                va="top" if panel else "baseline",
                path_effects=halo,
            )
    _apply_axis_spec(ax, leaf.axes)
