"""The single Scale-coherence checker, shared by construction and display.

`Scale` is built in `_views.py` (a view's field) and consumed in
`_display.py` (`normalize_values`, and `map_rgba` through it). Both call
sites need the same answer to "is this Scale internally coherent" --
a known `kind`, finite and ordered bounds, and the kind-specific
constraints (`symmetric` centered on zero, `log` with a usable positive
floor and a positive `vmin`) -- so a malformed or incoherent `Scale`
fails the same way, named the same way, whichever path reaches it first.
`check_scale` here is that one answer; neither `_views.py` nor
`_display.py` re-implements it.

This module only inspects `scale.kind`/`vmin`/`vmax`/`floor` by attribute
(no `Scale` import), so it has no circular dependency on `_views.py`,
which is the module that DOES import `Scale` and additionally checks that
a field is one before handing it here.
"""

import math

SCALE_KINDS = ("linear", "symmetric", "log")


def check_scale(scale, label):
    """Validate that `scale` is internally coherent, raising a named error.

    Args:
        scale: An object with `kind`, `vmin`, `vmax`, and `floor`
            attributes (a `Scale`).
        label: Prefixed onto every error message -- a view's element id at
            construction time, or the scale's own `kind` when there is no
            owning view (e.g. a `Scale` handed directly to
            `normalize_values`/`map_rgba`).

    Raises:
        ValueError: If `kind` is not one of `SCALE_KINDS`; if `vmin`/`vmax`
            are not both finite and strictly increasing; if `kind` is
            "symmetric" and the bounds do not satisfy
            `vmin == -vmax > 0`; if `kind` is "log" and `floor` is `None`,
            non-finite, or not positive; or if `kind` is "log" and `vmin`
            is not positive (a non-positive lower bound has no logarithm).
    """
    if scale.kind not in SCALE_KINDS:
        raise ValueError(
            f"{label}: scale kind must be one of {SCALE_KINDS}, got {scale.kind!r}"
        )
    vmin, vmax = float(scale.vmin), float(scale.vmax)
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        raise ValueError(
            f"{label}: scale bounds must be finite, got "
            f"({scale.vmin!r}, {scale.vmax!r})"
        )
    if not vmin < vmax:
        raise ValueError(
            f"{label}: scale bounds must be nondegenerate, got vmin={vmin}, vmax={vmax}"
        )
    if scale.kind == "symmetric" and not (vmin == -vmax and vmax > 0):
        raise ValueError(
            f"{label}: symmetric scale requires vmin == -vmax > 0, "
            f"got vmin={vmin}, vmax={vmax}"
        )
    if scale.kind == "log":
        if scale.floor is None:
            raise ValueError(f"{label}: log scale requires a positive floor")
        floor = float(scale.floor)
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError(f"{label}: log scale floor must be positive, got {floor}")
        if vmin <= 0:
            raise ValueError(f"{label}: log scale requires vmin > 0, got vmin={vmin}")
