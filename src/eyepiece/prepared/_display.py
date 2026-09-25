"""Shared numeric display mapping: validity, normalization, bounds, opacity.

Every prepared renderer (``eyepiece.mpl``, ``eyepiece.manim``) maps a
scientific array to a display value the same way, through the functions
here, so an image and its colorbar -- or an image rendered twice by two
different renderers -- never disagree about where a value sits on its
scale. This module imports only the standard library and NumPy; it never
picks a colormap or a color itself (that is ``eyepiece.style``'s job,
snapshotted into a ``RenderProfile`` and handed in).

Validity throughout combines two independent sources: an explicit
``valid`` array (``False`` marks a scientifically invalid sample, e.g. a
saturated pixel) and non-finite values in the data itself (``NaN``/``inf``,
regardless of what ``valid`` says). Either one masks a sample.

Display CLIPPING (an out-of-range or sub-floor but otherwise valid, finite
sample is pinned to the nearest end of the display range) is deliberately
distinct from MASKING (an invalid or non-finite sample is hidden entirely,
drawn in a renderer's dedicated "bad" color): a valid zero on a log scale
clips to the floor and stays visible; a NaN next to it is masked and reads
as unmistakably different data.
"""

import math

import numpy as np

from eyepiece.prepared._scale import check_scale

_LINEAR_KINDS = ("linear", "symmetric")


def _invalid_mask(raw, valid):
    """Combine non-finite values with an explicit validity array.

    Args:
        raw: Plain (unmasked) ndarray of values.
        valid: Boolean array shaped like `raw` (True = valid), or None.

    Returns:
        Boolean array shaped like `raw`, True where a sample is invalid.
    """
    invalid = ~np.isfinite(raw)
    if valid is not None:
        invalid = invalid | ~np.asarray(valid, dtype=bool)
    return invalid


def normalize_values(data, *, valid, scale):
    """Map `data` to masked display values in [0, 1] under `scale`.

    Linear and symmetric scales both use ``(x - vmin) / (vmax - vmin)``
    (a symmetric scale is a linear one whose bounds happen to satisfy
    ``vmin == -vmax``; the distinction is in `Scale` validation, not in
    this mapping). A log scale maps in log space, after first clipping
    every displayed (non-masked) sample up to `scale.floor` -- "positive
    display clipping": a valid sample at or below the floor still renders,
    pinned to the bottom of the scale, rather than disappearing as masked.
    Only a non-finite or explicitly invalid sample is masked.

    The mapped value is clamped to [0, 1] after the fact, so an
    in-range-valid sample outside `[vmin, vmax]` clips to the nearest
    display edge instead of extrapolating past it. `data` is never
    mutated.

    Args:
        data: Array of raw values.
        valid: Boolean array shaped like `data` (True = valid), or None
            (every finite sample is valid).
        scale: A `Scale` describing the mapping.

    Returns:
        A `numpy.ma.MaskedArray` shaped like `data`, values in [0, 1],
        masked wherever a sample is non-finite or explicitly invalid.

    Raises:
        ValueError: If `scale` is not internally coherent (see
            `eyepiece.prepared._scale.check_scale`); this is checked
            before any mapping runs, and before `map_rgba` (which calls
            this) indexes anything.
    """
    check_scale(scale, scale.kind)
    raw = np.asarray(data)
    invalid = _invalid_mask(raw, valid)

    if scale.kind == "log":
        # Non-finite entries are replaced before the floor clip and the
        # log so the log of a masked NaN/inf never runs (and never warns);
        # the replacement value is discarded either way once masked.
        safe = np.where(np.isfinite(raw), raw, scale.floor)
        clipped = np.maximum(safe, scale.floor)
        log_vmin = math.log(scale.vmin)
        log_vmax = math.log(scale.vmax)
        mapped = (np.log(clipped) - log_vmin) / (log_vmax - log_vmin)
    else:
        mapped = (raw - scale.vmin) / (scale.vmax - scale.vmin)

    clamped = np.clip(mapped, 0.0, 1.0)
    return np.ma.masked_array(clamped, mask=invalid)


def _validate_bounds(vmin, vmax, kind):
    """Validate a `(vmin, vmax)` pair for `kind`, naming `kind` in errors."""
    vmin = float(vmin)
    vmax = float(vmax)
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        raise ValueError(f"{kind}: bounds must be finite, got ({vmin}, {vmax})")
    if kind == "symmetric":
        if not (vmin == -vmax and vmax > 0):
            raise ValueError(
                f"{kind}: bounds must satisfy vmin == -vmax > 0, "
                f"got vmin={vmin}, vmax={vmax}"
            )
    elif kind == "log":
        if not (0 < vmin < vmax):
            raise ValueError(
                f"{kind}: bounds must satisfy 0 < vmin < vmax, "
                f"got vmin={vmin}, vmax={vmax}"
            )
    elif kind == "linear":
        if not vmin < vmax:
            raise ValueError(
                f"{kind}: bounds must be nondegenerate, got vmin={vmin}, vmax={vmax}"
            )
    else:
        raise ValueError(f"unknown scale kind {kind!r}")
    return vmin, vmax


def resolve_bounds(frames, *, kind, bounds=None):
    """Resolve `(vmin, vmax)` display bounds for `kind`, over `frames`.

    Args:
        frames: A replayable iterable of `(data, valid)` pairs (one pass
            is made over `frames`; each pair is reduced immediately, so a
            whole sequence is never concatenated into one array).
        kind: One of "linear", "symmetric", "log". Named in every error.
        bounds: Explicit `(vmin, vmax)`. When given, it is validated and
            returned without scanning `frames` at all.

    Returns:
        A validated `(vmin, vmax)` tuple of floats.

    Raises:
        ValueError: If `kind` is unknown; if `bounds` is given but invalid
            for `kind`; if every frame is entirely invalid (or, for "log",
            has no positive finite valid sample); or if the bounds
            computed from `frames` are degenerate (constant input),
            which needs an explicit `bounds` instead.
    """
    if kind not in (*_LINEAR_KINDS, "log"):
        raise ValueError(f"unknown scale kind {kind!r}")

    if bounds is not None:
        return _validate_bounds(bounds[0], bounds[1], kind)

    running_min = math.inf
    running_max = -math.inf
    any_valid = False
    for data, valid in frames:
        raw = np.asarray(data)
        invalid = _invalid_mask(raw, valid)
        finite_valid = raw[~invalid]
        if kind == "log":
            finite_valid = finite_valid[finite_valid > 0]
        if finite_valid.size == 0:
            continue
        any_valid = True
        local_min = float(finite_valid.min())
        local_max = float(finite_valid.max())
        running_min = min(running_min, local_min)
        running_max = max(running_max, local_max)

    if not any_valid:
        detail = (
            "no positive finite valid sample in any frame"
            if kind == "log"
            else "every frame is invalid"
        )
        raise ValueError(f"{kind}: cannot resolve bounds, {detail}")

    if kind == "symmetric":
        vmax = max(abs(running_min), abs(running_max))
        vmin = -vmax
    else:
        vmin, vmax = running_min, running_max

    try:
        return _validate_bounds(vmin, vmax, kind)
    except ValueError as exc:
        raise ValueError(f"{exc} (constant input; pass explicit bounds)") from None


def weight_opacity(weights):
    """Map nonnegative candidate weights to relative visual emphasis.

    This is a relative visual-emphasis mapping, not a probability scale:
    the formula ``0.15 + 0.65 * weight / max_weight`` puts the least
    heavily weighted candidate AMONG THOSE GIVEN at opacity 0.15, not 0,
    so it stays visible, and the most heavily weighted one at 0.80. When
    every weight is zero, every weight divides by a zero maximum equally;
    that case is defined to give every candidate the same (floor) opacity
    rather than raising or comparing 0/0.

    Args:
        weights: Array of candidate weights.

    Returns:
        An ndarray of opacities in [0.15, 0.80], shaped like `weights`.

    Raises:
        ValueError: If any weight is negative or non-finite.
    """
    arr = np.asarray(weights, dtype=float)
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("weight_opacity: weights must be finite")
    if arr.size and np.any(arr < 0):
        raise ValueError("weight_opacity: weights must be nonnegative")
    max_weight = float(arr.max()) if arr.size else 0.0
    if max_weight <= 0.0:
        return np.full(arr.shape, 0.15, dtype=float)
    return 0.15 + 0.65 * (arr / max_weight)


def map_rgba(data, *, valid, scale, profile):
    """Map `data` to uint8 RGBA under `scale`, using `profile`'s LUT.

    Uses `normalize_values` (the same mapper a scalar colorbar lookup
    uses) to get a display value in [0, 1] per sample, then indexes the
    256-entry LUT `profile.colormaps[scale.cmap_role]`. A masked
    (invalid) sample is painted `profile.bad_rgba` instead -- distinct
    from a clipped-but-valid extreme, which reads as the LUT's own first
    or last entry.

    Args:
        data: Array of raw values.
        valid: Boolean array shaped like `data` (True = valid), or None.
        scale: A `Scale` describing the mapping and colormap role.
        profile: A `RenderProfile` (see `eyepiece.style`) supplying the
            LUT and bad-pixel color.

    Returns:
        A uint8 ndarray shaped `data.shape + (4,)`.

    Raises:
        ValueError: If `scale` is not internally coherent; checked by the
            `normalize_values` call below, before anything is drawn.
    """
    normalized = normalize_values(data, valid=valid, scale=scale)
    lut = profile.colormaps[scale.cmap_role]
    filled = np.ma.filled(normalized, 0.0)
    index = np.clip(np.round(filled * 255.0), 0, 255).astype(np.intp)
    rgba = lut[index]
    bad = np.ma.getmaskarray(normalized)
    bad_rgba = np.asarray(profile.bad_rgba, dtype=np.uint8)
    rgba = np.where(bad[..., None], bad_rgba, rgba)
    return rgba.astype(np.uint8)
