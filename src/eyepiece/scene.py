"""Trajectory and sky-plane primitives: paths, weighted fans, fading tracks.

`trail` draws a 2D or 3D trajectory with markers whose size and layering cue
depth on the 3D path; the mechanism is deliberately generic, taking any
`(N, 2|3)` array of positions rather than baking in orbit- or
simulation-specific vocabulary -- a type-aware renderer belongs in a
consumer library, not here. `sky_fan` overlays several candidate sky-plane
tracks scaled by weight, with an optional inner-working-angle disk and
observed-epoch errorbars. `fading_track` draws a single path whose opacity
ramps from tail to head, built as an explicit per-segment RGBA array rather
than a colormap deferred to draw time, so the ramp is visible to
introspection on the returned artist.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from eyepiece import _style
from eyepiece._result import PlotResult


def _viewer_vectors(positions, azim_deg, elev_deg):
    """Viewer position and the per-point object/object-viewer vectors.

    The viewer is placed far enough from the trajectory (a large multiple
    of its own extent) that only the viewing direction, not the exact
    distance, sets the depth cues below.

    Args:
        positions: `(N, 3)` array of object positions.
        azim_deg: Camera azimuth, in degrees.
        elev_deg: Camera elevation, in degrees.

    Returns:
        `(r_v, r_o, r_ov)`: the viewer position `(3,)`, the object
        positions `(N, 3)` (identical to `positions`), and the
        object-minus-viewer vector `(N, 3)`.
    """
    radius = float(np.max(np.linalg.norm(positions, axis=-1)))
    dist = max(radius, 1.0) * 1.0e3

    elev_rad = np.deg2rad(elev_deg)
    azim_rad = np.deg2rad(azim_deg)
    r_v = dist * np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ]
    )

    r_o = positions
    r_ov = r_o - r_v
    return r_v, r_o, r_ov


def _viewer_angle(r_o, r_ov):
    """Per-point angle between the object and object-viewer vectors.

    Args:
        r_o: `(N, 3)` object positions.
        r_ov: `(N, 3)` object-minus-viewer vectors.

    Returns:
        `(N,)` array of angles in radians, in `[0, pi]`.
    """
    dot = -np.einsum("ij,ij->i", r_ov, r_o)
    denom = np.linalg.norm(r_ov, axis=-1) * np.linalg.norm(r_o, axis=-1)
    denom = np.where(denom == 0.0, np.finfo(float).eps, denom)
    return np.arccos(np.clip(dot / denom, -1.0, 1.0))


def _viewer_orth_dist(r_v, r_o):
    """Per-point distance from the viewer to each object, along the view axis.

    Args:
        r_v: `(3,)` viewer position.
        r_o: `(N, 3)` object positions.

    Returns:
        `(N,)` array of signed distances; smaller is nearer the viewer.
    """
    r_v_hat = r_v / np.linalg.norm(r_v)
    scalar_proj = r_o @ r_v_hat
    return np.linalg.norm(r_v) - scalar_proj


def trail(xyz, *, ax=None, style=None, marker_scale=25.0, trail_kw=None):
    """Draw a trajectory as a connected path with per-point markers.

    A `(N, 3)` path gets depth cues on its markers: size shrinks on the far
    side of the trajectory (`marker_scale * (1 + cos(viewer_angle)) / 2`,
    so a point facing away from the camera nearly vanishes) and the marker
    collection's overall layering is set from the trajectory's mean
    distance from the camera along the view axis, so multiple `trail`
    calls sharing one 3D axes stack correctly. A `(N, 2)` path has no depth
    to cue and skips this entirely.

    Args:
        xyz: `(N, 2)` or `(N, 3)` array-like of positions.
        ax: Axes to draw into. None creates a new figure; a 3D `xyz`
            creates a `projection="3d"` axes, a 2D one a plain axes.
        style: Color override; None uses `_style.color(0)`.
        marker_scale: Marker size at full illumination for the 3D depth
            cue, and the fixed marker size on a 2D path.
        trail_kw: Extra kwargs passed to the connecting-line `ax.plot`
            call, applied last.

    Returns:
        A `PlotResult` with artists `"line"` (the connecting `Line2D`) and
        `"scatter"` (the per-point `PathCollection`).
    """
    positions = np.asarray(xyz, dtype=float)
    is_3d = positions.shape[-1] == 3

    created = ax is None
    if created:
        subplot_kw = {"projection": "3d"} if is_3d else None
        _, ax = plt.subplots(subplot_kw=subplot_kw, layout="constrained")
    elif is_3d and getattr(ax, "name", None) != "3d":
        raise ValueError(
            'trail got (N, 3) xyz but a non-3D ax; pass a projection="3d" '
            "axes (or ax=None to let trail create one) for 3D positions."
        )

    color = _style.color(0, style)
    lkw = {"color": color, "lw": 1.5, **(trail_kw or {})}

    if is_3d:
        (line,) = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], **lkw)

        azim = getattr(ax, "azim", -60.0)
        elev = getattr(ax, "elev", 30.0)
        r_v, r_o, r_ov = _viewer_vectors(positions, azim, elev)
        angle = _viewer_angle(r_o, r_ov)
        orth_dist = _viewer_orth_dist(r_v, r_o)

        sizes = marker_scale * (1.0 + np.cos(angle)) / 2.0
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=sizes,
            color=color,
            zorder=-float(np.mean(orth_dist)),
        )
    else:
        (line,) = ax.plot(positions[:, 0], positions[:, 1], **lkw)
        scatter = ax.scatter(
            positions[:, 0], positions[:, 1], s=marker_scale, color=color
        )

    return PlotResult(ax=ax, artists={"line": line, "scatter": scatter})


def sky_fan(
    tracks, *, ax=None, colors=None, weights=None, iwa=None, data=None, fan_kw=None
):
    """Draw a fan of sky-plane tracks, faded by weight.

    Each track is a candidate path in the sky plane (RA/Dec offsets, or
    any two comparable coordinates); `weights` -- typically the posterior
    mass on each track -- fades a less-probable track toward invisible
    without hiding it outright. An optional inner-working-angle disk marks
    the region a coronagraph cannot see into, and optional observed-epoch
    errorbars overlay the fan.

    Args:
        tracks: List of `(x, y)` array-like coordinate pairs, one per
            track.
        ax: Axes to draw into. None creates a new figure and axes.
        colors: Optional list of per-track color overrides, same length as
            `tracks`. None defaults every track to `_style.color(k)`.
        weights: Optional list of per-track weights, same length as
            `tracks`, scaling each track's alpha. None weights every track
            equally.
        iwa: Optional inner-working-angle radius, in the same units as
            `tracks`; drawn as a shaded disk with an edge.
        data: Optional `(x, y, err)` tuple of array-likes for observed
            epochs, drawn as symmetric errorbars.
        fan_kw: Extra kwargs passed to each track's `ax.plot` call, folded
            in and applied last -- a caller-supplied `"color"` or
            `"zorder"` wins over the per-track default, and `"alpha"`
            overrides the base alpha the per-track weight scales (default
            0.25), not the final per-track alpha directly.

    Returns:
        A `PlotResult` with artist `"lines"` (list of per-track `Line2D`,
        one per track). Also has `"ellipse"` (the IWA disk `Polygon`) when
        `iwa` is given, and `"collection"` (the epoch errorbar's y-error
        `LineCollection`) when `data` is given.

    Note:
        Always sets the axes' aspect to `"equal"`, including on a
        caller-supplied `ax` -- a sky chart should not distort RA/Dec (or
        whatever two coordinates `tracks` carries).

        The scenery that is not a track -- the IWA disk, the central-star
        marker, the epoch errorbars -- is drawn in neutral tones read from
        the active rcParams at call time, so it stays legible against a
        light or a dark background instead of fixing one gray for both.
    """
    n = len(tracks)
    resolved_weights = list(weights) if weights is not None else [1.0] * n

    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    kw = {"lw": 0.7, "zorder": 1, "alpha": 0.25, **(fan_kw or {})}
    base_alpha = kw.pop("alpha")

    lines = []
    for k, (x, y) in enumerate(tracks):
        c = _style.color(k, colors[k] if colors is not None else None)
        line_alpha = min(1.0, base_alpha * (0.35 + resolved_weights[k]))
        line_kw = {"color": c, "alpha": line_alpha, **kw}
        (line,) = ax.plot(np.asarray(x), np.asarray(y), **line_kw)
        lines.append(line)
    artists = {"lines": lines}

    if iwa is not None:
        theta = np.linspace(0, 2 * np.pi, 200)
        xc, yc = iwa * np.cos(theta), iwa * np.sin(theta)
        (ellipse,) = ax.fill(xc, yc, color=_style.neutral(0.15), zorder=2, lw=0)
        ax.plot(xc, yc, color=_style.neutral(0.4), lw=0.8, zorder=3)
        artists["ellipse"] = ellipse

    ax.plot(0, 0, marker="*", color=_style.neutral(0.8), markersize=10, zorder=4)

    if data is not None:
        x_obs, y_obs, err = data
        errbar = ax.errorbar(
            np.asarray(x_obs),
            np.asarray(y_obs),
            xerr=np.asarray(err),
            yerr=np.asarray(err),
            fmt="o",
            color=_style.neutral(0.8),
            markersize=4,
            lw=1.0,
            zorder=5,
        )
        # errbar.lines[2] is (xerr_collection, yerr_collection) since both
        # xerr and yerr are always passed above; take the y-error one as
        # the single representative Collection artist.
        artists["collection"] = errbar.lines[2][1]

    ax.set_aspect("equal")
    return PlotResult(ax=ax, artists=artists)


def fading_track(
    xy, *, ax=None, color=None, alpha_range=(0.1, 1.0), collection_kw=None
):
    """Draw a 2D path whose opacity ramps from tail to head.

    The per-segment RGBA array is built explicitly rather than driving the
    fade through a `LineCollection` colormap: mapping a colormap at draw
    time means `LineCollection.get_colors()` does not return the alpha a
    caller expects, so the ramp is baked into the colors array up front
    instead.

    Args:
        xy: `(N, 2)` array-like path.
        ax: Axes to draw into. None creates a new figure and axes.
        color: Line color; None uses `_style.color(0)`.
        alpha_range: `(tail_alpha, head_alpha)` the per-segment alpha
            ramps across, from the first segment to the last.
        collection_kw: Extra kwargs passed to `LineCollection`, applied
            last -- a caller-supplied `"colors"` overrides the ramp built
            here.

    Returns:
        A `PlotResult` with artist `"collection"` (the `LineCollection`).
    """
    positions = np.asarray(xy, dtype=float)

    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    points = positions.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    n_segments = len(segments)

    rgb = to_rgb(_style.color(0, color))
    rgba = np.tile(np.array([*rgb, 1.0]), (n_segments, 1))
    rgba[:, 3] = np.linspace(alpha_range[0], alpha_range[1], n_segments)

    ckw = {"colors": rgba, "lw": 1.5, **(collection_kw or {})}
    collection = LineCollection(segments, **ckw)
    ax.add_collection(collection)
    ax.autoscale_view()

    return PlotResult(ax=ax, artists={"collection": collection})
