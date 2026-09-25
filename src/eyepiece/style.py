"""Explicit appearance snapshots for prepared-view renderers.

`eyepiece.prepared` and its renderers (`eyepiece.mpl`, `eyepiece.manim`)
never pick a color or a colormap themselves; they take a `RenderProfile`
snapshot and, for named scientific sources, a `SourceCast`. Building
either of those is this module's only job, and it is the only module in
`eyepiece` allowed to import `hwostyle` or Matplotlib -- everything a
prepared-view renderer needs from the active style is captured here, once,
as plain read-only data.

`snapshot_profile` reads `hwostyle.palette`, `hwostyle.cmaps`, and
`hwostyle.roles` -- the CURRENT state of those, whatever mode is active,
including hwostyle's own un-activated default -- along with the current
Matplotlib text rcParams, at the moment it is called. It never calls
`hwostyle.use()` or otherwise changes global state; a caller who wants a
particular mode calls `hwostyle.use(...)` before snapshotting.
"""

import dataclasses
from types import MappingProxyType

import hwostyle
import matplotlib
import numpy as np

# Roles documented on `hwostyle.colormaps.Colormaps`: sequential intensity
# families (idealized/model images, detector readouts, high-dynamic-range),
# diverging families (a signed residual/OPD), a cyclic phase map, a
# probability map, and the brand/mask maps. Listed explicitly (rather than
# introspecting hwostyle's private per-mode dict) so a role hwostyle drops
# fails loudly here instead of silently vanishing from every profile.
_CMAP_ROLES = (
    "intensity",
    "readouts",
    "high_dynamic_range",
    "residual",
    "opd",
    "phase",
    "probability",
    "mask",
    "brand_intensity",
    "brand_diverging",
)

# A fixed "impossible color" for masked/invalid samples: saturated magenta
# does not occur in any of hwostyle's sequential, diverging, or cyclic
# colormaps, so a bad pixel never reads as a clipped valid extreme by
# coincidence.
_BAD_RGBA = (255, 0, 255, 255)

# Marker cycle for `SourceCast.marker`, matching the one `SourceStyles`
# (`eyepiece.layout`) already assigns colors and markers from, so a cast
# and a legacy `SourceStyles` built from the same names agree on marker.
_MARKERS = ("o", "s", "D", "^", "v", "P")


def _lookup_colormap(role):
    """Resolve one hwostyle colormap role to a matplotlib Colormap.

    `hwostyle.cmaps.<role>` returns a registered colormap name (a string)
    for most roles but an already-built `Colormap` for others (e.g. the
    list-defined "mask" and brand maps); both are normalized here to a
    `Colormap` instance so the caller has one type to sample.
    """
    value = getattr(hwostyle.cmaps, role)
    if isinstance(value, str):
        return matplotlib.colormaps[value]
    return value


def _sample_lut(colormap):
    """Sample `colormap` into a fixed 256-entry uint8 RGBA LUT."""
    samples = colormap(np.linspace(0.0, 1.0, 256))
    lut = np.clip(np.round(np.asarray(samples) * 255.0), 0, 255).astype(np.uint8)
    lut.setflags(write=False)
    return lut


def _resolve_font_family(font_family):
    """Resolve `font_family`, or the current Matplotlib font stack's first entry."""
    if font_family is not None:
        return font_family
    generic = matplotlib.rcParams["font.family"]
    generic = generic[0] if generic else "sans-serif"
    concrete = matplotlib.rcParams.get(f"font.{generic}")
    if concrete:
        return concrete[0]
    return generic


@dataclasses.dataclass(frozen=True, eq=False)
class RenderProfile:
    """A read-only snapshot of the active hwostyle appearance.

    Every field is a copy taken at snapshot time: mutating hwostyle's
    global state afterward (a later `hwostyle.use(...)`, or taking another
    snapshot) never changes an existing `RenderProfile`.

    Attributes:
        colors: The active palette, in cycle order, as hex strings. A
            renderer resolves a named source's color as
            ``colors[cast.slot(name) % len(colors)]``.
        colormaps: Role name -> a `(256, 4)` uint8 RGBA lookup table,
            sampled from hwostyle's colormap for that role. A read-only
            mapping of read-only arrays.
        bad_rgba: The uint8 RGBA color for a masked (invalid or
            non-finite) sample, distinct from any LUT entry.
        text_color: Matplotlib's active `text.color`.
        background_color: Matplotlib's active `axes.facecolor`.
        reference_color: hwostyle's "reference" brand role color (IWA/OWA
            rings, floor curves, and similar scenery).
        font_family: Resolved font family name.
        text_size_pt: Text size, in points.
        stroke_width_pt: Line/stroke width, in points.
    """

    colors: tuple
    colormaps: dict
    bad_rgba: np.ndarray
    text_color: str
    background_color: str
    reference_color: str
    font_family: str
    text_size_pt: float
    stroke_width_pt: float

    def __post_init__(self):
        """Freeze `colors`, `colormaps`, and `bad_rgba` into read-only copies."""
        object.__setattr__(self, "colors", tuple(self.colors))

        frozen_luts = {}
        for role, lut in self.colormaps.items():
            arr = np.array(lut, dtype=np.uint8, copy=True)
            arr.setflags(write=False)
            frozen_luts[role] = arr
        object.__setattr__(self, "colormaps", MappingProxyType(frozen_luts))

        bad_rgba = np.array(self.bad_rgba, dtype=np.uint8, copy=True)
        bad_rgba.setflags(write=False)
        object.__setattr__(self, "bad_rgba", bad_rgba)

        object.__setattr__(self, "text_size_pt", float(self.text_size_pt))
        object.__setattr__(self, "stroke_width_pt", float(self.stroke_width_pt))


def snapshot_profile(*, font_family=None, text_size_pt=24, stroke_width_pt=1.5):
    """Snapshot the currently active hwostyle appearance into a `RenderProfile`.

    Reads `hwostyle.palette`, `hwostyle.cmaps`, and `hwostyle.roles` as
    they stand right now (including hwostyle's own un-activated default,
    if `hwostyle.use()` has never been called); never calls
    `hwostyle.use()` or otherwise mutates global state. Call
    `hwostyle.use(...)` first to snapshot a particular mode.

    Args:
        font_family: Font family name. Defaults to the first concrete font
            in Matplotlib's currently active font stack.
        text_size_pt: Text size, in points.
        stroke_width_pt: Line/stroke width, in points.

    Returns:
        A `RenderProfile`.
    """
    colors = tuple(hwostyle.palette.as_list)
    colormaps = {role: _sample_lut(_lookup_colormap(role)) for role in _CMAP_ROLES}
    return RenderProfile(
        colors=colors,
        colormaps=colormaps,
        bad_rgba=_BAD_RGBA,
        text_color=str(matplotlib.rcParams["text.color"]),
        background_color=str(matplotlib.rcParams["axes.facecolor"]),
        reference_color=hwostyle.roles.reference,
        font_family=_resolve_font_family(font_family),
        text_size_pt=text_size_pt,
        stroke_width_pt=stroke_width_pt,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class SourceCast:
    """Deterministic slot/marker assignment for a fixed set of named sources.

    A cast fixes only WHICH slot (a cycle index) and marker glyph each
    name gets, purely from its position in `names`; the color itself comes
    from a `RenderProfile` at render time
    (``profile.colors[cast.slot(name) % len(profile.colors)]``). Keeping
    the two separate is what lets one cast be reused against any profile
    (a dark-mode snapshot, a light-mode one, a future palette) without
    rebuilding it.

    Args:
        names: Source names, in slot-assignment order. Every name must be
            unique.

    Raises:
        ValueError: If `names` contains a duplicate.
    """

    names: tuple

    def __post_init__(self):
        """Coerce `names` to a tuple and build the name -> slot index."""
        names = tuple(self.names)
        object.__setattr__(self, "names", names)
        index = {}
        for i, name in enumerate(names):
            if name in index:
                raise ValueError(f"{name}: duplicate name in SourceCast")
            index[name] = i
        object.__setattr__(self, "_index", index)

    def slot(self, name):
        """The cycle-index slot assigned to `name`.

        Args:
            name: A name passed to the constructor.

        Returns:
            The 0-based index of `name` in the constructor's `names`.

        Raises:
            KeyError: If `name` was not passed to the constructor.
        """
        return self._index[name]

    def marker(self, name):
        """The marker glyph assigned to `name`.

        Args:
            name: A name passed to the constructor.

        Returns:
            A matplotlib marker string, cycling through a fixed sequence
            by `name`'s slot.

        Raises:
            KeyError: If `name` was not passed to the constructor.
        """
        return _MARKERS[self.slot(name) % len(_MARKERS)]


__all__ = ["RenderProfile", "SourceCast", "snapshot_profile"]
