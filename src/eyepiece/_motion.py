"""Presentation time and camera geometry for animations, in NumPy alone.

The clock itself (`Timeline`, `Plan`) lives in `eyepiece._clock`; this module
holds what it is built from: the frame rule (`frame_count`), the easing
curves, and `Tempo`, the one place a staged clip's pace is set. A knob is a
presentation quantity, never a physical sample index: a move interpolates how
a figure is shown, not simulated time, so nothing here touches a prepared
`Sequence`, whose time is sampled and never interpolated.

The other pieces are the geometry a moving figure needs: easing curves,
per-item stagger, the van Wijk and Nuij optimal pan-and-zoom path between two
views, the axis limits of a view, and the quadrilaterals of a pixel grid, so
an image can be drawn as one movable shape per pixel. `Tempo` sets the pace
of a staged clip in one place: one factor for every move, a settle after
each new subject, and a reading hold that grows with a caption's words.

This module imports only the standard library and NumPy, so the timing
helpers never load matplotlib.
"""

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

_SNAP = 1e-9
_EASINGS = (
    "linear",
    "smooth",
    "smoothstep",
    "smootherstep",
    "cosine",
    "back_out",
    "running_start",
)
_STRENGTH_KINDS = ("back_out", "running_start")
_SPACES = ("linear", "log")
_SQRT2 = math.sqrt(2.0)
_SMOOTH_INFLECTION = 10.0
# The curve a move, an entrance or an exit takes when none is named.
DEFAULT_EASE = "smooth"


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


_BACK_DEFAULT = 0.1
_START_DEFAULT = 0.1
_STRENGTH_MAX = 1000.0


def _bisect(excess, target, hi):
    """The x in [0, hi] where the increasing function `excess` reaches `target`."""
    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if excess(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@lru_cache(maxsize=32)
def _back_constant(strength):
    """Penner's overshoot constant whose peak is `strength` above 1."""
    return _bisect(lambda s: 4.0 * s**3 / (27.0 * (s + 1.0) ** 2), strength, 1e4)


@lru_cache(maxsize=32)
def _start_pull(strength):
    """Control value of the pull-back curve whose lowest point is -`strength`."""
    t = np.linspace(0.0, 0.5, 4001)

    def depth(k):
        return -np.min(_pull_curve(t, -k))

    return -_bisect(depth, strength, 1e4)


def _pull_curve(t, pull):
    """Degree-6 Bezier through control values 0, 0, pull, pull, 1, 1, 1."""
    q = 1.0 - t
    return (
        pull * (15.0 * t**2 * q**4 + 20.0 * t**3 * q**3)
        + 15.0 * t**4 * q**2
        + 6.0 * t**5 * q
        + t**6
    )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ease(u, kind=DEFAULT_EASE, strength=None):
    """Eased progress for progress `u`; the input is clipped to [0, 1].

    Every curve fixes both ends exactly (0 at 0, 1 at 1). ``smoothstep`` and
    ``cosine`` start and stop with zero slope and pass through 0.5 at the
    midpoint; ``smootherstep`` also has zero curvature at the ends.
    ``smooth`` is Manim's default rate function, a logistic sigmoid with
    inflection 10 rescaled to fix both ends: it passes through 0.5 at the
    midpoint, dwells longer near both ends than ``smoothstep`` (slope about
    0.067 there) and is steeper in the middle (slope about 2.5 against 1.5).
    Bound with `functools.partial` to a kind, it is a valid Manim
    ``rate_func``.

    ``"back_out"`` overshoots 1 and then settles back onto it, and
    ``"running_start"`` first pulls back below 0 and then goes,
    after Manim's curve with a shallower default pull.
    These two leave [0, 1] in between, and neither passes through 0.5 at the
    midpoint. Only the input `u` is clipped, never the result, so a move that
    eases a position can show the overshoot. A move that uses the eased value
    as an opacity, a fraction of a path, or a color blend must clip it at the
    use site; every move in this package does.

    Args:
        u: Progress, a float or an array; values outside [0, 1] are clipped.
        kind: One of ``"linear"``, ``"smooth"``, ``"smoothstep"``,
            ``"smootherstep"``, ``"cosine"``, ``"back_out"``, or
            ``"running_start"``; default `DEFAULT_EASE`.
        strength: For ``"back_out"``, the peak overshoot beyond 1 as a
            fraction of the move (default 0.1, the classic back curve). For
            ``"running_start"``, the depth of the pull-back below 0 as a
            fraction of the move (default 0.1). Positive and at most 1000; the
            other kinds take none.

    Returns:
        A float for a scalar `u`, otherwise an array of the same shape.

    Raises:
        ValueError: If `kind` is not one of the curves above, or `strength` is
            given for a curve that takes none or is not positive and finite.
    """
    if kind not in _EASINGS:
        raise ValueError(f"unknown easing {kind!r}; use one of {_EASINGS}")
    if strength is not None:
        if kind not in _STRENGTH_KINDS:
            raise ValueError(f"easing {kind!r} takes no strength")
        if not math.isfinite(strength) or not 0.0 < strength <= _STRENGTH_MAX:
            raise ValueError(
                f"strength must be in (0, {_STRENGTH_MAX:g}], got {strength}"
            )
    a = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    if kind == "linear":
        out = a
    elif kind == "smooth":
        k = _SMOOTH_INFLECTION
        floor = _sigmoid(-0.5 * k)
        out = (_sigmoid(k * (a - 0.5)) - floor) / (1.0 - 2.0 * floor)
        out = np.where(a >= 1.0, 1.0, out)
    elif kind == "smoothstep":
        out = a * a * (3.0 - 2.0 * a)
    elif kind == "smootherstep":
        out = a**3 * (a * (6.0 * a - 15.0) + 10.0)
    elif kind == "cosine":
        out = 0.5 - 0.5 * np.cos(np.pi * a)
    elif kind == "back_out":
        s = _back_constant(_BACK_DEFAULT if strength is None else float(strength))
        out = (s + 1.0) * a**3 - (2.0 * s + 3.0) * a**2 + (s + 3.0) * a
        out = np.where(a >= 1.0, 1.0, out)
    else:
        pull = _start_pull(_START_DEFAULT if strength is None else float(strength))
        out = np.where(a >= 1.0, 1.0, _pull_curve(a, pull))
    out = np.where(a <= 0.0, 0.0, out)
    return float(out) if np.ndim(u) == 0 else out


def stagger(u, rank, *, span=0.35):
    """Per-item progress for items that start one after another.

    This staggers the items of one move within its progress; `Timeline.stagger`
    staggers whole spans on the clock, one verb call per target.

    Item `i` starts at ``rank[i] * (1 - span)`` of the beat and finishes
    `span` later, so rank 0 starts at the beginning and rank 1 finishes at
    the end. Pass the beat's uneased progress and ease each item's progress
    afterwards; easing both would ease twice.

    Args:
        u: The beat's progress in [0, 1].
        rank: Array of item ranks in [0, 1], for example a normalized radius.
            With a `Plan`, pass ``plan.progress(k, span, eased=False)`` as `u`.
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
# Tempo

_MATH_SPAN = re.compile(r"\$[^$]*\$")
_TEX_COMMAND = re.compile(r"\\([A-Za-z]+)")
_SYMBOL = re.compile(r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?")
# TeX commands that are operators, relations, spacing or formatting: read
# as part of the symbols around them, not as symbols of their own.
_TEX_NOT_SYMBOLS = frozenset(
    "mid pm mp cdot cdots ldots dots times div le leq ge geq ne neq approx sim"
    " simeq propto equiv to rightarrow leftarrow gets frac dfrac tfrac sqrt"
    " left right big Big rm mathrm mathit mathbf mathcal bf it text operatorname"
    " hat bar tilde vec dot prime quad qquad".split()
)


def reading_words(text):
    r"""Words a viewer reads in `text`, a symbol in math counting as two.

    Plain words are split on whitespace. Inside ``$...$`` each symbol counts
    as two words: a letter, a number, a function name, or a TeX command that
    names a symbol (``\lambda``, ``\Phi``). Operators, relations, spacing
    and formatting commands (``\mid``, ``\pm``, ``\frac``, ``\rm``) count
    as nothing.

    Args:
        text: The caption, with math in ``$...$``: a string, or the pieces
            a `callout` takes, each a string or a tuple whose first item is
            the string.

    Returns:
        The word count, an int.
    """
    if not isinstance(text, str):
        text = "".join(p if isinstance(p, str) else str(p[0]) for p in text)
    words = 0
    for span in _MATH_SPAN.findall(text):
        commands = _TEX_COMMAND.findall(span)
        words += 2 * sum(name not in _TEX_NOT_SYMBOLS for name in commands)
        words += 2 * len(_SYMBOL.findall(_TEX_COMMAND.sub(" ", span)))
    return words + len(_MATH_SPAN.sub(" ", text).split())


@dataclass(frozen=True)
class Tempo:
    """The pace of a staged clip, set in one place.

    A clip written against a tempo states its moves and holds in nominal
    seconds and lets the tempo turn them into seconds on screen, so one
    factor slows or quickens the whole clip at once. Three rules set the
    pace:

    - a move or hold of nominal ``s`` seconds lasts ``factor * s``
      (`seconds`);
    - a caption holds, after it has finished arriving, for
      ``factor * (base_s + word_s * words)``, with a symbol in math counting
      as two words (`read_s`, `reading_words`);
    - after an axis is built or a new subject arrives, the frame rests for
      ``factor * settle_s`` before anything else moves (`rest_s`).

    It is the one reading rule of the package: `Timeline.read` holds for
    `read_s` too, and a caption leaves once read for `read_s`.

    At the defaults a caption holds about 1 s plus 0.4 s per word and a
    settle lasts about 0.8 s; ``factor=1`` is a brisker pace, 0.75 s plus
    0.3 s per word.

    Attributes:
        factor: Global tempo; every duration is multiplied by it, so 2 plays
            the clip at half speed.
        base_s: Nominal reading hold of an empty caption, in seconds.
        word_s: Nominal reading time per word, in seconds.
        settle_s: Nominal rest after an axis build or a new subject.
    """

    factor: float = 1.35
    base_s: float = 0.75
    word_s: float = 0.3
    settle_s: float = 0.6

    def __post_init__(self):
        """Check that every duration is finite, the factor positive."""
        if not math.isfinite(self.factor) or self.factor <= 0.0:
            raise ValueError(f"factor must be finite and positive, got {self.factor}")
        for name in ("base_s", "word_s", "settle_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and not negative, got {value}")

    def seconds(self, nominal):
        """On-screen seconds of a move or hold of `nominal` seconds."""
        nominal = float(nominal)
        if not math.isfinite(nominal) or nominal < 0.0:
            raise ValueError(f"seconds must be finite and not negative, got {nominal}")
        return self.factor * nominal

    def read_s(self, text):
        """Seconds to hold `text` on screen after it has finished arriving.

        Args:
            text: A string, or caption pieces (see `reading_words`).

        Returns:
            The hold in seconds.
        """
        return self.factor * (self.base_s + self.word_s * reading_words(text))

    @property
    def rest_s(self):
        """Seconds the frame rests after an axis build or a new subject."""
        return self.factor * self.settle_s


# ---------------------------------------------------------------------------
# Timeline


def _apply_ease(r, how=DEFAULT_EASE):
    """Eased progress for a kind name or a callable; the input is clipped first."""
    if callable(how):
        return float(how(min(1.0, max(0.0, float(r)))))
    return ease(r, how)


class _Unsnap:
    """Pixel snapping off while artists move, back to as drawn at rest.

    Agg snaps axis-aligned lines and patch edges to whole pixels, which crisps
    a still and turns a slow move into one-pixel jumps. A move calls
    `moving(True)` on the frames between its ends and `moving(False)` at the
    ends, so outside a recording the rest state is the snapped figure exactly.
    Inside `eyepiece.record`, which draws every frame unsnapped by default,
    the rest frame matches a snapped still only to within half a pixel. Each
    artist's own snap setting is read once, here, and put back as it was.

    Args:
        artists: The artists the move reshapes or displaces.
    """

    def __init__(self, artists):
        self._saved = [(a, a.get_snap()) for a in artists]

    def moving(self, flag):
        """Unsnap every artist when `flag` is true, else restore its setting."""
        for artist, snap in self._saved:
            artist.set_snap(False if flag else snap)


def unsnapped(artists):
    """Switch pixel snapping off on artists, so slow moves glide.

    Agg snaps markers, axis-aligned lines and rectilinear patch edges to whole
    pixels: a mark moved a tenth of a pixel per frame stalls, then jumps. An
    unsnapped artist lands at its sub-pixel position on every frame. Use it on
    artists drawn outside `eyepiece.record`, which unsnaps a whole recording.

    Args:
        artists: One artist or an iterable of artists.

    Returns:
        The same argument, for chaining.
    """
    for artist in [artists] if hasattr(artists, "set_snap") else artists:
        artist.set_snap(False)
    return artists


def _unsnapped(artist):
    """Switch snapping off for good on a ghost, which is drawn only mid-move."""
    artist.set_snap(False)
    return artist


def _interp(a, b, e, space):
    if space == "log":
        return a * (b / a) ** e
    return a + (b - a) * e


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


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


def pulse_shape(u, *, rise=0.35, hold=0.3, fall=0.35, ease=DEFAULT_EASE):
    """Rise to one, hold a plateau, and fall to zero over progress [0, 1].

    Args:
        u: Finite scalar progress. Outside [0, 1] the shape is zero.
        rise: Positive relative duration of the rise.
        hold: Nonnegative relative duration of the plateau.
        fall: Positive relative duration of the fall. Phase durations are
            normalized by their sum, so (1, 1, 2) gives quarters, then half.
        ease: Easing kind or callable used on each edge, clipped to [0, 1].

    Returns:
        A float in [0, 1], exactly zero at the ends and one on the plateau.

    Example::

        strength = ep.pulse_shape(0.5)
        held = ep.pulse_shape(0.4, rise=1, hold=2, fall=1)
    """
    u, rise, hold, fall = map(float, (u, rise, hold, fall))
    total = rise + hold + fall
    if not all(math.isfinite(x) for x in (u, rise, hold, fall, total)):
        raise ValueError("pulse progress and durations must be finite")
    if rise <= 0 or hold < 0 or fall <= 0:
        raise ValueError("pulse rise and fall must be positive and hold nonnegative")
    if u <= 0 or u >= 1:
        return 0.0
    rise, fall = rise / total, fall / total
    if rise <= u <= 1 - fall:
        return 1.0
    edge = u / rise if u < rise else (1 - u) / fall
    return min(1.0, max(0.0, _apply_ease(edge, ease)))
