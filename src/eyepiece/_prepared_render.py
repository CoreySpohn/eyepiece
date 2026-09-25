"""Renderer-neutral helpers shared by `eyepiece.mpl` and `eyepiece.manim`.

Both prepared-view renderers walk the same tree, resolve the same
appearance from a `SourceCast` and a `RenderProfile`, treat nonfinite
coordinates as the same gaps, and bind a result to the same topology
signature, so a view accepted by one renderer's `update` is accepted by the
other's. Everything here is plain NumPy and standard library: no artist, no
mobject. Private: not part of the public API.
"""

import numpy as np

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
VIEW_TYPES = (*LEAF_TYPES, PanelGroup)

# Fill opacity of a Region's interior on a coordinate panel (a shaded IWA);
# its outline is drawn fully opaque. See `region_fill_opacity`.
REGION_FILL_OPACITY = 0.2

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


def _role_color(mark, profile, renderer):
    if mark.role not in _SUPPORTED_ROLES:
        raise ValueError(
            f"{mark.id}: role {mark.role!r} is not supported by the {renderer} "
            f"renderer; use one of {_SUPPORTED_ROLES}"
        )
    return profile.reference_color


def resolve_styles(view, cast, profile, *, renderer, leaf_parts=()):
    """Resolve every element's appearance, raising before anything is drawn.

    Args:
        view: The prepared view tree.
        cast: `SourceCast` for source colors and markers.
        profile: `RenderProfile` for colors and colormap tables.
        renderer: Renderer name used in error messages.
        leaf_parts: Extra derived-part suffixes the renderer adds to every
            leaf view (e.g. `("axes",)` gives `"<view id>/axes"`), checked
            for collisions with element IDs like the built-in ones.

    Returns:
        A dict mapping element ID to its style dict (`color`, and `marker`
        for paths and points).

    Raises:
        ValueError: If a source ID is not in `cast`, a role is unsupported,
            a colormap role is not in `profile`, a panel group is empty, or
            a derived part key collides with an element ID.
    """
    ids = {element.id for element in iter_elements(view)}
    styles = {}
    for element in iter_elements(view):
        derived = ()
        if isinstance(element, LEAF_TYPES):
            derived = tuple(f"{element.id}/{suffix}" for suffix in leaf_parts)
        if isinstance(element, ImageView):
            role = element.scale.cmap_role
            if role not in profile.colormaps:
                raise ValueError(
                    f"{element.id}: colormap role {role!r} is not in the render profile"
                )
            derived = (*derived, f"{element.id}/colorbar")
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
            styles[element.id] = {"color": _role_color(element, profile, renderer)}
        elif isinstance(element, PanelGroup) and not element.views:
            raise ValueError(f"{element.id}: panel group has no views to draw")
        for key in derived:
            if key in ids:
                raise ValueError(
                    f"{key}: element ID collides with the rendered part "
                    f"{key!r} of {element.id!r}"
                )
    return styles


def region_fill_opacity(leaf):
    """Fill opacity for a Region drawn on `leaf`.

    On an `ImageView` a Region is an outline only: a translucent fill
    would change the colors of the measured pixels under it, so they would
    no longer read against the colorbar. On a `CurveView` or `TrackView`
    the interior is shaded at `REGION_FILL_OPACITY`.
    """
    return 0.0 if isinstance(leaf, ImageView) else REGION_FILL_OPACITY


def path_alphas(leaf):
    """Opacity per Path on one panel, through the shared weight mapping."""
    paths = [m for m in leaf.marks if isinstance(m, Path)]
    weights = tuple(float(p.weight) for p in paths)
    alphas = weight_opacity(np.asarray(weights, dtype=float))
    return weights, {p.id: float(a) for p, a in zip(paths, alphas, strict=True)}


# --- Geometry and storage identity -------------------------------------------


def gap_nan(xy):
    """Float copy of (P, 2) `xy` with every gap row (any nonfinite) all NaN.

    An infinite coordinate is never drawn as a far-away point, so every
    nonfinite row becomes NaN in both components. The row count never
    changes, so a handle keeps one vertex/offset per sample and a later
    finite update shows the sample again.
    """
    xy = np.array(xy, dtype=float)
    xy[~np.all(np.isfinite(xy), axis=1)] = np.nan
    return xy


def visible_xy(path):
    """The revealed `(start, stop)` slice of a Path's vertices, gaps as NaN."""
    start, stop = path.visible
    return gap_nan(path.xy[start:stop])


def array_key(arr):
    """Identity of borrowed storage: address, shape, strides, dtype."""
    if arr is None:
        return None
    interface = arr.__array_interface__
    return (interface["data"][0], arr.shape, arr.strides, arr.dtype.str)


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


def topology(view):
    """The topology signature of a whole tree, depth first."""
    return tuple(_topology_entry(e) for e in iter_elements(view))


def check_topology(expected, actual):
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
