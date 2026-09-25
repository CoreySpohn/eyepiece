"""Pure numerical prepared-view records: scenes as plain data.

A prepared view is a frozen dataclass tree -- `ImageView`, `CurveView`,
`TrackView`, and `PanelGroup`, decorated with typed `Mark` overlays (`Path`,
`Points`, `Region`, `ReferenceLine`, `Label`) -- carrying only borrowed
numeric arrays and plain metadata. Nothing here is a matplotlib artist or a
Manim mobject; renderers (``eyepiece.mpl``, ``eyepiece.manim``) turn a tree
into one. Building a tree validates it once, naming the offending element's
`id` in every error, so a renderer can trust the shapes and bounds it is
handed instead of re-checking them.

Every element -- a view or a mark -- carries a globally unique `id` within
its own tree. `find_element` and `replace_elements` look up and edit trees
by that id; a replacement must be the same kind (class) as the element it
replaces, and only the path from the root down to a changed element is
rebuilt, so every array an edit did not touch is the same object afterward,
not a copy.

This module imports only the standard library and NumPy.
"""

import dataclasses
import math
from numbers import Integral

import numpy as np

from eyepiece.prepared._scale import check_scale

_AXES = ("x", "y")
_SPACES = ("panel", "data")
_DIRECTIONS = ("row", "column")


def _mask_to_valid(values):
    """Split a possibly-masked array into plain data and a validity mask.

    Capturing the mask before coercing to a plain array is what lets a
    borrowed `numpy.ma.MaskedArray` keep sharing memory with its caller:
    `numpy.ma.getdata` returns the underlying array itself, never a copy.

    Args:
        values: An array, or a `numpy.ma.MaskedArray`.

    Returns:
        A `(data, valid)` pair. `data` is `values` with any mask stripped
        (the same object when `values` is not a masked array). `valid` is
        `None` when `values` carries no mask, otherwise a boolean array
        shaped like `data` that is True wherever `values` is not masked.
    """
    mask = np.ma.getmask(values)
    data = np.asarray(np.ma.getdata(values))
    valid = None if mask is np.ma.nomask else np.logical_not(mask)
    return data, valid


def _combine_valid(mask_valid, explicit_valid):
    """Combine a captured mask's validity with an explicit `valid=` array.

    Args:
        mask_valid: Validity derived from an input mask, or None.
        explicit_valid: A caller-supplied validity array, or None.

    Returns:
        `None` when both inputs are `None`; otherwise the elementwise AND
        of both when both are given, or whichever one is not `None`.
    """
    if mask_valid is None:
        return explicit_valid
    if explicit_valid is None:
        return mask_valid
    return np.logical_and(mask_valid, explicit_valid)


def _check_numeric(arr, owner_id, field_name):
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(
            f"{owner_id}: {field_name} must be numeric, got dtype {arr.dtype}"
        )


def _check_xy(value, owner_id, field_name):
    """Validate a (P, 2) coordinate array, returning it as an ndarray.

    A row with any nonfinite component (NaN or inf) is accepted as a
    missing-data gap, never as a coordinate: renderers hide it, and
    anything that derives bounds from coordinates skips it.
    """
    arr = np.asarray(value)
    _check_numeric(arr, owner_id, field_name)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"{owner_id}: {field_name} must be shaped (P, 2), got {arr.shape}"
        )
    return arr


def _gap_rows(xy):
    """Boolean (P,) array, True where a coordinate row is a gap."""
    return ~np.all(np.isfinite(xy), axis=1)


def _check_center(value, owner_id):
    """Validate a (2,) center coordinate, returning it as an ndarray."""
    arr = np.asarray(value)
    _check_numeric(arr, owner_id, "center")
    if arr.shape != (2,):
        raise ValueError(f"{owner_id}: center must be shaped (2,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{owner_id}: center must be finite")
    return arr


def _check_error_vector(value, gaps, owner_id, field_name):
    """Validate an optional nonnegative error vector matching the rows.

    Args:
        value: The error vector, or None.
        gaps: Boolean (P,) array from `_gap_rows`, True on gap rows.
        owner_id: Element id named in errors.
        field_name: "xerr" or "yerr", named in errors.

    An error may be nonfinite only on a gap row (a missing point has no
    meaningful error); on every other row it must be finite. No error may
    be negative.
    """
    if value is None:
        return None
    p_count = gaps.shape[0]
    arr = np.asarray(value)
    _check_numeric(arr, owner_id, field_name)
    if arr.ndim != 1 or arr.shape[0] != p_count:
        raise ValueError(
            f"{owner_id}: {field_name} must be shaped ({p_count},), got {arr.shape}"
        )
    if np.any(arr < 0):
        raise ValueError(f"{owner_id}: {field_name} must be nonnegative")
    if not np.all(np.isfinite(arr[~gaps])):
        raise ValueError(
            f"{owner_id}: {field_name} must be finite wherever xy is finite"
        )
    return arr


def _check_visible(visible, p_count, owner_id):
    if not isinstance(visible, tuple) or len(visible) != 2:
        raise ValueError(
            f"{owner_id}: visible must be a (start, stop) tuple, got {visible!r}"
        )
    start, stop = visible
    if not isinstance(start, Integral) or start < 0 or start > p_count:
        raise ValueError(
            f"{owner_id}: visible start must be an integer in [0, {p_count}], "
            f"got {start!r}"
        )
    if stop is not None and (
        not isinstance(stop, Integral) or stop < start or stop > p_count
    ):
        raise ValueError(
            f"{owner_id}: visible stop must be None or an integer in "
            f"[{start}, {p_count}], got {stop!r}"
        )


def _check_axis_spec(axes, owner_id):
    if not isinstance(axes, AxisSpec):
        raise ValueError(
            f"{owner_id}: axes must be an AxisSpec, got {type(axes).__name__}"
        )
    _check_limits(axes.x_limits, owner_id, "x_limits")
    _check_limits(axes.y_limits, owner_id, "y_limits")


def _check_limits(limits, owner_id, field_name):
    if not isinstance(limits, tuple) or len(limits) != 2:
        raise ValueError(
            f"{owner_id}: {field_name} must be a (low, high) tuple, got {limits!r}"
        )
    low, high = (float(v) for v in limits)
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"{owner_id}: {field_name} must be finite, got {limits!r}")
    if not low < high:
        raise ValueError(
            f"{owner_id}: {field_name} must be strictly increasing, got {limits!r}"
        )


def _check_scale(scale, owner_id):
    """Check that `scale` is a `Scale` and is internally coherent.

    The "is this a `Scale`" check is specific to a view's field (a
    dataclass constructor argument can be anything); coherence itself --
    known kind, finite ordered bounds, and the kind-specific constraints --
    is `eyepiece.prepared._scale.check_scale`, shared with the display
    mapping path (`normalize_values`/`map_rgba`) so a malformed or
    incoherent `Scale` is rejected identically wherever it is first used,
    rather than surfacing later as a bare `TypeError`/`ValueError` out of
    the arithmetic.
    """
    if not isinstance(scale, Scale):
        raise ValueError(
            f"{owner_id}: scale must be a Scale, got {type(scale).__name__}"
        )
    check_scale(scale, owner_id)


def _check_marks(marks, owner_id):
    for mark in marks:
        if not isinstance(mark, _MARK_TYPES):
            names = [t.__name__ for t in _MARK_TYPES]
            raise ValueError(
                f"{owner_id}: marks must be one of {names}, got {type(mark).__name__}"
            )


def _iter_ids(element):
    yield element.id
    if isinstance(element, (ImageView, CurveView, TrackView)):
        for mark in element.marks:
            yield from _iter_ids(mark)
    elif isinstance(element, PanelGroup):
        for child in element.views:
            yield from _iter_ids(child)


def _check_unique_ids(root):
    seen = set()
    for element_id in _iter_ids(root):
        if element_id in seen:
            raise ValueError(f"{element_id}: duplicate element id in tree")
        seen.add(element_id)


def _finalize_axis_view(self):
    """Shared __post_init__ body for a plain axes-plus-marks view.

    CurveView and TrackView differ only in the domain intent their
    docstrings describe; both validate their `axes`, coerce and validate
    `marks`, and check id uniqueness identically, so both call this instead
    of repeating the same four lines.
    """
    _check_axis_spec(self.axes, self.id)
    marks = tuple(self.marks)
    _check_marks(marks, self.id)
    object.__setattr__(self, "marks", marks)
    _check_unique_ids(self)


@dataclasses.dataclass(frozen=True, eq=False)
class AxisSpec:
    """Axis labels, pixel-edge extent, and display direction for one view.

    Attributes:
        x_label: Label for the x axis.
        y_label: Label for the y axis.
        x_limits: `(low, high)` pixel-edge extent along x, finite and
            strictly increasing.
        y_limits: `(low, high)` pixel-edge extent along y, finite and
            strictly increasing.
        aspect: Axes aspect setting, passed through to the renderer.
        x_reverse: Whether x increases right-to-left on display (the RA
            convention), without changing `x_limits` itself.
        show_ticks: Whether to draw tick labels.
    """

    x_label: str
    y_label: str
    x_limits: tuple
    y_limits: tuple
    aspect: str = "equal"
    x_reverse: bool = False
    show_ticks: bool = True


@dataclasses.dataclass(frozen=True, eq=False)
class Scale:
    """A value-to-display mapping: clipping, normalization, and colormap role.

    Attributes:
        kind: One of "linear", "symmetric" (linear, diverging about zero),
            or "log".
        vmin: Lower display bound.
        vmax: Upper display bound, strictly greater than `vmin`.
        cmap_role: Colormap role name resolved through the active style.
        floor: Positive display floor, required when `kind` is "log".
    """

    kind: str
    vmin: float
    vmax: float
    cmap_role: str = "intensity"
    floor: float | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class Path:
    """A polyline mark, optionally revealed over a visible sub-range.

    Attributes:
        id: Element id, unique within the tree.
        xy: Coordinates shaped (P, 2). A row with a nonfinite component
            is a gap: the polyline breaks there instead of passing
            through a coordinate.
        source_id: Optional identity of the scientific source this path
            belongs to, distinct from `id` (several paths can share one
            `source_id` while remaining independently addressable).
        weight: Nonnegative candidate weight, mapped to display opacity.
        label: Legend/description text.
        visible: `(start, stop)` index range into `xy` that is currently
            revealed; `stop=None` means "through the end". `start == stop`
            is valid hidden geometry, not an error.
    """

    id: str
    xy: np.ndarray
    source_id: str | None = None
    weight: float = 1.0
    visible: tuple = (0, None)
    label: str = ""

    def __post_init__(self):
        xy = _check_xy(self.xy, self.id, "xy")
        object.__setattr__(self, "xy", xy)
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError(
                f"{self.id}: weight must be a nonnegative finite number, "
                f"got {self.weight!r}"
            )
        _check_visible(self.visible, xy.shape[0], self.id)


@dataclasses.dataclass(frozen=True, eq=False)
class Points:
    """A discrete point-set mark, with optional per-point error bars.

    Attributes:
        id: Element id, unique within the tree.
        xy: Coordinates shaped (P, 2). A row with a nonfinite component
            is a missing point (a gap), hidden by renderers.
        source_id: Optional identity of the scientific source this point
            set belongs to, distinct from `id`.
        xerr: Optional nonnegative x error shaped (P,); nonfinite only on
            gap rows.
        yerr: Optional nonnegative y error shaped (P,); nonfinite only on
            gap rows.
        label: Legend/description text.
    """

    id: str
    xy: np.ndarray
    source_id: str | None = None
    xerr: np.ndarray | None = None
    yerr: np.ndarray | None = None
    label: str = ""

    def __post_init__(self):
        xy = _check_xy(self.xy, self.id, "xy")
        object.__setattr__(self, "xy", xy)
        gaps = _gap_rows(xy)
        object.__setattr__(
            self, "xerr", _check_error_vector(self.xerr, gaps, self.id, "xerr")
        )
        object.__setattr__(
            self, "yerr", _check_error_vector(self.yerr, gaps, self.id, "yerr")
        )


@dataclasses.dataclass(frozen=True, eq=False)
class Region:
    """A circle (`inner_radius=0`) or annulus reference mark.

    Attributes:
        id: Element id, unique within the tree.
        center: Coordinates shaped (2,).
        outer_radius: Positive outer radius.
        inner_radius: Nonnegative inner radius, strictly less than
            `outer_radius`; zero (the default) draws a circle.
        role: Semantic role resolved through the active style (e.g. an
            IWA/OWA reference).
        label: Legend/description text.
    """

    id: str
    center: np.ndarray
    outer_radius: float
    inner_radius: float = 0.0
    role: str = "reference"
    label: str = ""

    def __post_init__(self):
        center = _check_center(self.center, self.id)
        object.__setattr__(self, "center", center)
        outer = float(self.outer_radius)
        inner = float(self.inner_radius)
        if not math.isfinite(outer) or outer <= 0:
            raise ValueError(
                f"{self.id}: outer_radius must be a positive finite number, "
                f"got {outer!r}"
            )
        if not math.isfinite(inner) or inner < 0:
            raise ValueError(
                f"{self.id}: inner_radius must be a nonnegative finite number, "
                f"got {inner!r}"
            )
        if inner >= outer:
            raise ValueError(
                f"{self.id}: inner_radius ({inner}) must be less than "
                f"outer_radius ({outer})"
            )
        object.__setattr__(self, "outer_radius", outer)
        object.__setattr__(self, "inner_radius", inner)


@dataclasses.dataclass(frozen=True, eq=False)
class ReferenceLine:
    """A full-axis reference line at a fixed data value.

    Attributes:
        id: Element id, unique within the tree.
        axis: Which axis the line is drawn against, "x" or "y".
        value: Finite data-coordinate value the line is drawn at.
        role: Semantic role resolved through the active style.
        label: Legend/description text.
    """

    id: str
    axis: str
    value: float
    role: str = "reference"
    label: str = ""

    def __post_init__(self):
        if self.axis not in _AXES:
            raise ValueError(
                f"{self.id}: axis must be one of {_AXES}, got {self.axis!r}"
            )
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError(f"{self.id}: value must be finite, got {self.value!r}")
        object.__setattr__(self, "value", value)


@dataclasses.dataclass(frozen=True, eq=False)
class Label:
    """A text annotation placed in panel or data space.

    Attributes:
        id: Element id, unique within the tree.
        text: The label text.
        xy: `(x, y)` placement, finite.
        space: "panel" (figure-relative) or "data" (data-coordinate).
    """

    id: str
    text: str
    xy: tuple
    space: str = "panel"

    def __post_init__(self):
        if self.space not in _SPACES:
            raise ValueError(
                f"{self.id}: space must be one of {_SPACES}, got {self.space!r}"
            )
        xy = tuple(float(v) for v in self.xy)
        if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise ValueError(
                f"{self.id}: xy must be a finite (x, y) pair, got {self.xy!r}"
            )
        object.__setattr__(self, "xy", xy)


_MARK_TYPES = (Path, Points, Region, ReferenceLine, Label)


@dataclasses.dataclass(frozen=True, eq=False)
class ImageView:
    """A 2D scalar field displayed as an image.

    ImageView has no separate extent field: its pixel-edge extent is
    `(axes.x_limits, axes.y_limits)`.

    Validity combines an input mask with an explicit `valid=` array. Both
    given: the elementwise AND. Only one given: that one. Neither: `None`.
    A captured mask never allocates a whole-sequence mask on its own; it is
    a per-frame concern, derived once here from whatever `data` and `valid`
    this particular frame was built with.

    Attributes:
        id: Element id, unique within the tree.
        data: 2D array shaped (Y, X). When `data` is a
            `numpy.ma.MaskedArray`, its mask is captured into `valid` and
            its underlying data array is stored here unchanged; a plain
            array is stored as given. Never dtype-promoted or copied to
            convert it.
        axes: AxisSpec giving labels, pixel-edge extent, and display
            direction; see the class docstring for the extent convention.
        scale: Scale mapping data values to display.
        quantity: Name of the displayed physical quantity (e.g.
            "intensity").
        valid: Boolean array shaped like `data`, True where a sample is
            valid, or `None` when no mask and no explicit `valid` were
            given.
        marks: Overlays drawn on the image (Path, Points, Region,
            ReferenceLine, Label).
    """

    id: str
    data: np.ndarray
    axes: AxisSpec
    scale: Scale
    quantity: str
    valid: np.ndarray | None = None
    marks: tuple = ()

    def __post_init__(self):
        data, mask_valid = _mask_to_valid(self.data)
        _check_numeric(data, self.id, "data")
        if data.ndim != 2:
            raise ValueError(f"{self.id}: data must be shaped (Y, X), got {data.shape}")
        object.__setattr__(self, "data", data)

        explicit_valid = (
            None if self.valid is None else np.asarray(self.valid, dtype=bool)
        )
        if explicit_valid is not None and explicit_valid.shape != data.shape:
            raise ValueError(
                f"{self.id}: valid shape {explicit_valid.shape} does not match "
                f"data shape {data.shape}"
            )
        object.__setattr__(self, "valid", _combine_valid(mask_valid, explicit_valid))

        _check_axis_spec(self.axes, self.id)
        _check_scale(self.scale, self.id)

        marks = tuple(self.marks)
        _check_marks(marks, self.id)
        object.__setattr__(self, "marks", marks)

        _check_unique_ids(self)


@dataclasses.dataclass(frozen=True, eq=False)
class CurveView:
    """A 1D curve panel: axes plus typed marks (traces, points, references).

    Attributes:
        id: Element id, unique within the tree.
        axes: AxisSpec giving labels, pixel-edge extent, and display
            direction.
        marks: Overlays drawn on the curve (Path, Points, Region,
            ReferenceLine, Label).
    """

    id: str
    axes: AxisSpec
    marks: tuple = ()

    def __post_init__(self):
        _finalize_axis_view(self)


@dataclasses.dataclass(frozen=True, eq=False)
class TrackView:
    """A 2D coordinate-frame panel: axes plus typed marks (tracks, observations).

    Composes the same typed marks as CurveView, with 2D spatial rather than
    curve domain intent (e.g. a sky-plane track sharing one coordinate
    mapping across truth paths, observations, a star, and an IWA region).

    Attributes:
        id: Element id, unique within the tree.
        axes: AxisSpec giving labels, pixel-edge extent, and display
            direction.
        marks: Overlays drawn on the track (Path, Points, Region,
            ReferenceLine, Label).
    """

    id: str
    axes: AxisSpec
    marks: tuple = ()

    def __post_init__(self):
        _finalize_axis_view(self)


@dataclasses.dataclass(frozen=True, eq=False)
class PanelGroup:
    """Named views laid out in a row or column.

    Attributes:
        id: Element id, unique within the tree.
        views: Child views (ImageView, CurveView, TrackView, or nested
            PanelGroup).
        direction: "row" or "column".
    """

    id: str
    views: tuple
    direction: str = "row"

    def __post_init__(self):
        views = tuple(self.views)
        for view in views:
            if not isinstance(view, _VIEW_TYPES):
                names = [t.__name__ for t in _VIEW_TYPES]
                raise ValueError(
                    f"{self.id}: views must be one of {names}, "
                    f"got {type(view).__name__}"
                )
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"{self.id}: direction must be one of {_DIRECTIONS}, "
                f"got {self.direction!r}"
            )
        object.__setattr__(self, "views", views)
        _check_unique_ids(self)


_VIEW_TYPES = (ImageView, CurveView, TrackView, PanelGroup)

View = ImageView | CurveView | TrackView | PanelGroup
Mark = Path | Points | Region | ReferenceLine | Label


def find_element(view, id):
    """Find the view or mark with `id` inside `view`'s tree.

    Args:
        view: The root of a prepared view tree.
        id: The element id to find.

    Returns:
        The View or Mark whose `.id` equals `id`.

    Raises:
        KeyError: If no element in the tree has that id.
    """
    found = _find(view, id)
    if found is None:
        raise KeyError(id)
    return found


def _find(element, id):
    if element.id == id:
        return element
    if isinstance(element, (ImageView, CurveView, TrackView)):
        for mark in element.marks:
            if mark.id == id:
                return mark
    elif isinstance(element, PanelGroup):
        for child in element.views:
            found = _find(child, id)
            if found is not None:
                return found
    return None


def replace_elements(view, changes):
    """Return a new tree with the elements named in `changes` replaced.

    Every replacement is validated -- a known id, and the same kind (class)
    as the element currently at that id -- before any part of the tree is
    rebuilt, so an invalid `changes` mapping never partially applies.
    Elements not named in `changes`, and the arrays they hold, are the same
    objects as in `view`; only the path from the root down to each changed
    element is rebuilt.

    Args:
        view: The root of a prepared view tree.
        changes: Mapping from element id to its replacement.

    Returns:
        A new root (or `view` itself, unchanged, if nothing on its path
        changed).

    Raises:
        KeyError: If a key in `changes` does not name an element in the
            tree.
        TypeError: If a replacement is not the same kind as the element
            currently at that id.
    """
    for element_id, replacement in changes.items():
        current = find_element(view, element_id)
        if type(replacement) is not type(current):
            raise TypeError(
                f"{element_id}: replacement is {type(replacement).__name__}, "
                f"expected {type(current).__name__}"
            )
    return _rebuild(view, changes)


def _rebuild(element, changes):
    if element.id in changes:
        return changes[element.id]
    if isinstance(element, (ImageView, CurveView, TrackView)):
        new_marks = tuple(_rebuild(mark, changes) for mark in element.marks)
        if all(new is old for new, old in zip(new_marks, element.marks, strict=True)):
            return element
        return dataclasses.replace(element, marks=new_marks)
    if isinstance(element, PanelGroup):
        new_views = tuple(_rebuild(child, changes) for child in element.views)
        if all(new is old for new, old in zip(new_views, element.views, strict=True)):
            return element
        return dataclasses.replace(element, views=new_views)
    return element
