"""Presentation time and camera geometry for animations, in NumPy alone.

An explanatory animation is a sequence of beats: holds, in which the viewer
reads while nothing moves, and ramps, in which something moves while no new
text arrives. `Timeline` declares those beats in seconds and the presentation
knobs they set or move (a title key, an angle, a zoom width); `at_fps`
resolves them for one frame rate into a `Plan`, whose frames a caller's draw
function reads. One timeline serves several venues, since frame boundaries
are computed from cumulative time for each rate.

A knob is a presentation quantity, never a physical sample index: a ramp
interpolates how a figure is shown, not simulated time, so nothing here
touches a prepared `Sequence`, whose time is sampled and never interpolated.

The other pieces are the geometry a moving figure needs: easing curves,
per-item stagger, the van Wijk and Nuij optimal pan-and-zoom path between two
views, the axis limits of a view, and the quadrilaterals of a pixel grid, so
an image can be drawn as one movable shape per pixel.

This module imports only the standard library and NumPy, so the timing
helpers never load matplotlib.
"""

import math
from dataclasses import dataclass, field

import numpy as np

_SNAP = 1e-9
_EASINGS = ("linear", "smoothstep", "smootherstep", "cosine")
_RESERVED = frozenset({"beat", "u", "index", "t"})
_SPACES = ("linear", "log")
_SQRT2 = math.sqrt(2.0)


def frame_count(seconds, fps):
    """Number of frames that `seconds` lasts at `fps`.

    The product is rounded half up, after snapping a value within 1e-9 of a
    whole or half number onto it, so float noise such as ``0.29 * 50 =
    14.499999999999998`` rounds the way the written numbers say (to 15).

    Args:
        seconds: Duration in seconds, finite and not negative.
        fps: Frame rate in frames per second, positive.

    Returns:
        The frame count, an int.

    Raises:
        ValueError: If `seconds` is negative or not finite, or `fps` is not
            positive.
    """
    seconds = float(seconds)
    fps = float(fps)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"seconds must be finite and not negative, got {seconds}")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    x = seconds * fps
    doubled = round(2.0 * x)
    if abs(2.0 * x - doubled) < _SNAP * max(1.0, abs(x)):
        x = doubled / 2.0
    return math.floor(x + 0.5)


def ease(u, kind="smoothstep"):
    """Eased progress for progress `u`, clipped to [0, 1].

    Every curve fixes both ends (0 at 0, 1 at 1) and passes through 0.5 at
    the midpoint. ``smoothstep`` and ``cosine`` start and stop with zero
    slope; ``smootherstep`` also with zero curvature. Bound with
    `functools.partial` to a kind, it is a valid Manim ``rate_func``.

    Args:
        u: Progress, a float or an array; values outside [0, 1] are clipped.
        kind: One of ``"linear"``, ``"smoothstep"``, ``"smootherstep"``, or
            ``"cosine"``.

    Returns:
        A float for a scalar `u`, otherwise an array of the same shape.

    Raises:
        ValueError: If `kind` is not one of the curves above.
    """
    a = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    if kind == "linear":
        out = a
    elif kind == "smoothstep":
        out = a * a * (3.0 - 2.0 * a)
    elif kind == "smootherstep":
        out = a**3 * (a * (6.0 * a - 15.0) + 10.0)
    elif kind == "cosine":
        out = 0.5 - 0.5 * np.cos(np.pi * a)
    else:
        raise ValueError(f"unknown easing {kind!r}; use one of {_EASINGS}")
    return float(out) if np.ndim(u) == 0 else out


def stagger(u, rank, *, span=0.35):
    """Per-item progress for items that start one after another.

    Item `i` starts at ``rank[i] * (1 - span)`` of the beat and finishes
    `span` later, so rank 0 starts at the beginning and rank 1 finishes at
    the end. Pass the beat's uneased progress and ease each item's progress
    afterwards; easing both would ease twice.

    Args:
        u: The beat's progress in [0, 1].
        rank: Array of item ranks in [0, 1], for example a normalized radius.
            With a `Plan`, pass ``plan.progress(k, beat, eased=False)`` as `u`,
            not a frame's ``u`` field, which is already eased.
        span: Fraction of the beat each item takes, in (0, 1].

    Returns:
        An array of per-item progress in [0, 1], shaped like `rank`.

    Raises:
        ValueError: If `span` is outside (0, 1].
    """
    span = float(span)
    if not 0.0 < span <= 1.0:
        raise ValueError(f"span must be in (0, 1], got {span}")
    rank = np.asarray(rank, dtype=float)
    start = rank * (1.0 - span)
    return np.clip((np.asarray(u, dtype=float) - start) / span, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Timeline


@dataclass(frozen=True)
class Beat:
    """One resolved beat of a `Plan`.

    Attributes:
        name: The beat's name, unique within its timeline.
        kind: ``"hold"`` or ``"ramp"``.
        start: Index of the beat's first frame.
        stop: Index one past its last frame.
        seconds: Its declared duration.
        ease: The easing of a ramp (a kind name or a callable); ``"linear"``
            for a hold.
    """

    name: str
    kind: str
    start: int
    stop: int
    seconds: float
    ease: object = field(default="linear", compare=False)

    @property
    def n(self):
        """Number of frames in the beat."""
        return self.stop - self.start


def _apply_ease(r, how):
    return float(how(r)) if callable(how) else ease(r, how)


def _interp(a, b, e, space):
    if space == "log":
        return a * (b / a) ** e
    return a + (b - a) * e


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


class Timeline:
    """Beats of an explanatory animation, declared in seconds.

    Holds and ramps are appended in order. A hold may change knobs when it
    starts (a new title, a new cue); a ramp moves numeric knobs to targets
    and changes nothing else, so text never arrives during motion. A
    timeline cannot open with a ramp, whose start state would never be
    shown. Resolve it for a frame rate with `at_fps`.

    Example::

        tl = Timeline(knobs={"text": "intro", "theta": 0.0})
        tl.hold(2.0, name="intro")
        tl.hold(1.0, name="cue", set={"text": "turn"})
        tl.ramp(3.0, name="turn", to={"theta": 1.0})
        tl.hold(2.0, name="end")
        plan = tl.at_fps(30)
        for frame in plan.frames:
            draw(frame["text"], frame["theta"])

    Args:
        knobs: The knob names and their initial values. Every name a later
            `set` or `to` uses must be declared here; ``beat``, ``u``,
            ``index`` and ``t`` are reserved for the frame fields.

    Raises:
        ValueError: If a knob name is reserved.
    """

    def __init__(self, knobs=None):
        self._initial = dict(knobs or {})
        for name in self._initial:
            if name in _RESERVED:
                raise ValueError(
                    f"knob name {name!r} is reserved for frame fields "
                    f"{sorted(_RESERVED)}"
                )
        self._state = dict(self._initial)
        self._steps = []

    def _check_known(self, names):
        for name in names:
            if name not in self._initial:
                raise ValueError(
                    f"knob {name!r} was not declared; declared knobs are "
                    f"{sorted(self._initial)}"
                )

    def _add(self, kind, seconds, name, payload, how="linear"):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(f"beat seconds must be finite and positive, got {seconds}")
        if name is None:
            name = f"{kind}{len(self._steps)}"
        if any(step[0] == name for step in self._steps):
            raise ValueError(f"a beat named {name!r} already exists")
        self._steps.append((name, kind, seconds, payload, how))
        return self

    def hold(self, seconds, *, name=None, set=None):
        """Append a beat in which nothing moves.

        Args:
            seconds: Duration of the hold.
            name: The beat's name; None generates one.
            set: Knob values that change when the hold starts.

        Returns:
            This timeline, for chaining.
        """
        changes = dict(set or {})
        self._check_known(changes)
        self._add("hold", seconds, name, changes)
        self._state.update(changes)
        return self

    def ramp(self, seconds, *, name=None, to=None, ease="smoothstep", space="linear"):
        """Append a beat that moves numeric knobs to targets.

        Frame ``i`` of an ``n``-frame ramp reaches progress ``(i + 1) / n``,
        so its first frame already differs from the hold before it and its
        last frame shows the targets exactly.

        Args:
            seconds: Duration of the ramp.
            name: The beat's name; None generates one.
            to: Target values for numeric knobs.
            ease: An easing kind for `eyepiece.ease`, or a callable mapping
                progress in [0, 1] to eased progress.
            space: ``"linear"`` interpolates values; ``"log"`` interpolates
                their logarithms, for widths and other scales, and needs
                both ends positive.

        Returns:
            This timeline, for chaining.

        Raises:
            ValueError: If the timeline has no beat yet, a knob is
                undeclared or not numeric, a log-space end is not positive,
                or `ease` or `space` is unknown.
        """
        if not self._steps:
            raise ValueError("a timeline must open with a hold, not a ramp")
        targets = dict(to or {})
        self._check_known(targets)
        if space not in _SPACES:
            raise ValueError(f"unknown space {space!r}; use one of {_SPACES}")
        if not callable(ease) and ease not in _EASINGS:
            raise ValueError(f"unknown easing {ease!r}; use one of {_EASINGS}")
        for key, target in targets.items():
            start = self._state[key]
            if not (_is_number(start) and _is_number(target)):
                raise ValueError(f"knob {key!r} must be numeric to ramp")
            if space == "log" and (start <= 0 or target <= 0):
                raise ValueError(
                    f"a log-space ramp of {key!r} needs positive ends, "
                    f"got {start} and {target}"
                )
        self._add("ramp", seconds, name, (targets, space), how=ease)
        self._state.update(targets)
        return self

    def read(self, text, *, name=None, s_per_word=0.3, min_s=2.5, set=None):
        """Append a hold long enough to read `text`.

        Args:
            text: The words on screen.
            name: The beat's name; None generates one.
            s_per_word: Reading time per word, in seconds.
            min_s: Shortest hold, in seconds.
            set: Knob values that change when the hold starts.

        Returns:
            This timeline, for chaining.
        """
        seconds = max(float(min_s), float(s_per_word) * len(str(text).split()))
        return self.hold(seconds, name=name, set=set)

    @property
    def seconds(self):
        """Total declared duration."""
        return math.fsum(step[2] for step in self._steps)

    def at_fps(self, fps):
        """Resolve the beats for one frame rate.

        Beat boundaries are ``frame_count`` of the cumulative time at each
        boundary, so the total frame count is ``frame_count(seconds, fps)``
        and does not drift with the number of beats.

        Args:
            fps: Frame rate in frames per second.

        Returns:
            A `Plan`.

        Raises:
            ValueError: If the timeline is empty, or a beat rounds to no
                frames at this rate.
        """
        if not self._steps:
            raise ValueError("the timeline has no beats")
        # Each boundary from the exactly summed prefix of durations, so the
        # total is frame_count(self.seconds, fps) whatever the beat count.
        bounds = [0]
        for i in range(len(self._steps)):
            elapsed = math.fsum(step[2] for step in self._steps[: i + 1])
            bounds.append(frame_count(elapsed, fps))
        beats = []
        frames = []
        state = dict(self._initial)
        for (name, kind, seconds, payload, how), start, stop in zip(
            self._steps, bounds[:-1], bounds[1:], strict=True
        ):
            n = stop - start
            if n <= 0:
                raise ValueError(
                    f"beat {name!r} ({seconds} s) rounds to no frames at {fps} fps"
                )
            beats.append(Beat(name, kind, start, stop, seconds, how))
            if kind == "hold":
                state.update(payload)
                for j in range(n):
                    frames.append(
                        {**state, "beat": name, "u": (j + 1) / n,
                         "index": start + j, "t": (start + j) / fps}
                    )  # fmt: skip
                continue
            targets, space = payload
            begin = {key: state[key] for key in targets}
            for j in range(n):
                e = _apply_ease((j + 1) / n, how)
                values = {
                    key: (
                        targets[key]
                        if j == n - 1
                        else _interp(begin[key], targets[key], e, space)
                    )
                    for key in targets
                }
                frames.append(
                    {**state, **values, "beat": name, "u": e,
                     "index": start + j, "t": (start + j) / fps}
                )  # fmt: skip
            state.update(targets)
        return Plan(float(fps), tuple(beats), tuple(frames))


class Plan:
    """A timeline resolved for one frame rate.

    Attributes:
        fps: The frame rate.
        beats: The `Beat` records, in order.
        frames: One dict per frame, each a fresh copy: the knob values plus
            ``beat`` (the beat's name), ``u`` (the beat's eased progress),
            ``index`` (the frame index) and ``t`` (seconds from the start).
    """

    def __init__(self, fps, beats, frames):
        self.fps = fps
        self.beats = beats
        self.frames = frames
        self._by_name = {b.name: b for b in beats}

    @property
    def n_frames(self):
        """Total frame count."""
        return len(self.frames)

    @property
    def seconds(self):
        """Running time at this frame rate."""
        return self.n_frames / self.fps

    def _beat(self, name):
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"no beat named {name!r}") from None

    def progress(self, k, beat, *, eased=True):
        """Progress of `beat` at frame `k`: 0 before it, 1 after it.

        Args:
            k: Frame index.
            beat: The beat's name.
            eased: Apply the beat's easing (a hold's progress is linear).

        Returns:
            A float in [0, 1].
        """
        b = self._beat(beat)
        if k < b.start:
            return 0.0
        if k >= b.stop:
            return 1.0
        r = (k - b.start + 1) / b.n
        return _apply_ease(r, b.ease) if eased else r

    def beat_at(self, k):
        """The `Beat` that frame `k` belongs to.

        Raises:
            IndexError: If `k` is outside the plan.
        """
        for b in self.beats:
            if b.start <= k < b.stop:
                return b
        raise IndexError(f"frame {k} is outside the plan's {self.n_frames} frames")

    def start(self, beat):
        """Index of the beat's first frame."""
        return self._beat(beat).start

    def last(self, beat):
        """Index of the beat's last frame, its settled state."""
        return self._beat(beat).stop - 1

    def mid(self, beat):
        """Index of the frame at the beat's halfway progress (rounded down)."""
        b = self._beat(beat)
        return b.start + (b.n - 1) // 2

    def find(self, where, *, last=False):
        """Index of the first (or last) frame whose fields match `where`.

        Args:
            where: A dict of field names and values, all of which must match.
            last: Return the last matching frame instead of the first.

        Raises:
            ValueError: If no frame matches.
        """
        indices = range(self.n_frames - 1, -1, -1) if last else range(self.n_frames)
        for k in indices:
            frame = self.frames[k]
            if all(frame.get(key) == value for key, value in where.items()):
                return k
        raise ValueError(f"no frame matches {where!r}")

    def nearest(self, key, value):
        """Index of the frame whose numeric field `key` is closest to `value`."""
        values = np.array([float(f[key]) for f in self.frames])
        return int(np.argmin(np.abs(values - float(value))))


# ---------------------------------------------------------------------------
# Views


def _view(view):
    cx, cy, w = (float(v) for v in view)
    if not (math.isfinite(w) and w > 0.0):
        raise ValueError(f"a view's width must be finite and positive, got {w}")
    return cx, cy, w


def zoom_path(view0, view1, *, rho=_SQRT2):
    """The optimal pan-and-zoom path between two views (van Wijk and Nuij 2003).

    A view is ``(cx, cy, w)``: a center and a width in the same data units.
    The path zooms out while it pans and back in as it arrives, so a target
    farther than the view is wide enters the view before the camera does,
    and its parameter is proportional to the perceived distance travelled.
    This is the algorithm behind d3's ``interpolateZoom``.

    Args:
        view0: Start view ``(cx, cy, w)``.
        view1: End view ``(cx, cy, w)``.
        rho: Trade-off between zooming and panning; ``sqrt(2)`` is the value
            van Wijk and Nuij recommend.

    Returns:
        ``(path, length)``: ``path(s)`` maps ``s`` in [0, 1] (a float or an
        array) to ``(cx, cy, w)``; ``length`` is the path length in their
        metric, a natural scale for the move's duration.

    Raises:
        ValueError: If a width is not positive.
    """
    x0, y0, w0 = _view(view0)
    x1, y1, w1 = _view(view1)
    dx, dy = x1 - x0, y1 - y0
    d2 = dx * dx + dy * dy
    rho2 = rho * rho
    rho4 = rho2 * rho2
    if math.sqrt(d2) <= 1e-12 * max(w0, w1):
        length = abs(math.log(w1 / w0)) / rho
        k = math.copysign(1.0, w1 - w0)

        def flat(s):
            s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
            w = w0 * np.exp(k * rho * s * length)
            return _out(x0 + 0.0 * s, y0 + 0.0 * s, w)

        return flat, float(length)
    d1 = math.sqrt(d2)
    b0 = (w1 * w1 - w0 * w0 + rho4 * d2) / (2.0 * w0 * rho2 * d1)
    b1 = (w1 * w1 - w0 * w0 - rho4 * d2) / (2.0 * w1 * rho2 * d1)
    # log(sqrt(b^2 + 1) - b), written as -asinh(b) because the log form
    # cancels to log(0) when b is large: a zoom out with a tiny pan.
    r0 = -math.asinh(b0)
    r1 = -math.asinh(b1)
    length = (r1 - r0) / rho
    coshr0 = math.cosh(r0)

    def path(s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        t = s * length
        # cosh(r0) tanh(rho t + r0) - sinh(r0) = sinh(rho t) / cosh(rho t + r0),
        # the form without cancellation between two large terms.
        u = w0 / (rho2 * d1) * np.sinh(rho * t) / np.cosh(rho * t + r0)
        w = w0 * coshr0 / np.cosh(rho * t + r0)
        cx, cy = x0 + u * dx, y0 + u * dy
        # The two ends exactly, whatever rounding the formulas carry.
        cx = np.where(s >= 1.0, x1, np.where(s <= 0.0, x0, cx))
        cy = np.where(s >= 1.0, y1, np.where(s <= 0.0, y0, cy))
        w = np.where(s >= 1.0, w1, np.where(s <= 0.0, w0, w))
        return _out(cx, cy, w)

    return path, float(length)


def _out(cx, cy, w):
    if np.ndim(w) == 0:
        return float(cx), float(cy), float(w)
    return cx, cy, w


def view_limits(view, *, aspect=1.0):
    """Axis limits of a view.

    Args:
        view: ``(cx, cy, w)``, a center and a width in data units.
        aspect: Height over width of the view, in data units.

    Returns:
        ``((x0, x1), (y0, y1))``, for ``ax.set_xlim`` and ``ax.set_ylim``.

    Raises:
        ValueError: If the width or `aspect` is not positive.
    """
    cx, cy, w = _view(view)
    aspect = float(aspect)
    if not (math.isfinite(aspect) and aspect > 0.0):
        raise ValueError(f"aspect must be finite and positive, got {aspect}")
    h = w * aspect
    return (cx - w / 2.0, cx + w / 2.0), (cy - h / 2.0, cy + h / 2.0)


def pixel_quads(corners, keep=None):
    """The quadrilateral of every pixel, from its grid of corners.

    Args:
        corners: ``(ny + 1, nx + 1, 2)`` corner positions, ``[..., 0]`` the x
            and ``[..., 1]`` the y coordinate. Corners may be moved anywhere;
            each pixel stays the quad of its four corners.
        keep: Optional ``(ny, nx)`` boolean mask of the pixels to return.

    Returns:
        ``(n, 4, 2)`` vertices, counter-clockwise from the lower left for an
        unmoved grid whose rows go up the page.

    Raises:
        ValueError: If `corners` is not ``(ny + 1, nx + 1, 2)`` or `keep` is
            not ``(ny, nx)``.
    """
    c = np.asarray(corners, dtype=float)
    if c.ndim != 3 or c.shape[2] != 2 or c.shape[0] < 2 or c.shape[1] < 2:
        raise ValueError(f"corners must be (ny + 1, nx + 1, 2), got {c.shape}")
    quads = np.stack([c[:-1, :-1], c[:-1, 1:], c[1:, 1:], c[1:, :-1]], axis=-2)
    if keep is None:
        return quads.reshape(-1, 4, 2)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != quads.shape[:2]:
        raise ValueError(f"keep must be {quads.shape[:2]}, got {keep.shape}")
    return quads[keep]
