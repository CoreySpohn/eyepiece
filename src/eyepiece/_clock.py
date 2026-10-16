"""One clock for every clip: named spans, the knobs they write, and the plan.

Everything timed in an explanatory animation is a span: a hold, a subject
entering or leaving, a move, a caption arriving, a pulse, a ring, a line of
narration, a section. A `Timeline` declares spans in seconds; `at_fps` turns
them into a `Plan`, one value per knob per frame, plus the events (starts,
middles, ends, cues) a reviewer or a linter needs. A draw function never sees
time: it reads `Plan.value(k, knob)`.

Frame rules, applied to every conversion of seconds to frames:

- A span from ``s`` to ``e`` seconds covers frames ``k0 = frame_count(s)``
  up to ``k1 = max(k0 + 1, frame_count(e))``, ``n = k1 - k0`` of them. A span
  rounded up to one frame is a fault.
- Its progress is 0 before ``k0``, ``ease((k - k0 + 1) / n)`` on frame ``k``
  of the span, and exactly 1 from ``k1`` on, so the last frame shows the
  target exactly.
- Times are exact sums, never accumulated floats.

A clip with a `eyepiece.narration.Narration` places each span on the word that
names it, and the same function serves the silent clip with ``narration=None``.

This module imports only the standard library, NumPy and `eyepiece._motion`.
"""

import difflib
import itertools
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction

from eyepiece._motion import (
    _EASINGS,
    _SPACES,
    Tempo,
    _apply_ease,
    _interp,
    _is_number,
    frame_count,
)

SHOW_LEAD_S = 0.1  # a subject starts arriving this long before its word
PULSE_S = 0.9  # seconds on screen; the swell peaks half way through
RING_S = 1.3  # seconds on screen; the outline draws on, then fades
CAPTION_S = 0.4  # a caption takes this long to arrive
LEAVE_S = 0.3  # nominal seconds a caption or subject takes to leave
ENTER_S = 0.6  # nominal seconds a subject takes to arrive
LINE_TAIL_S = 0.6  # a line outlasts its last extending span by this
SECTION_TAIL_S = 0.0  # a plain section ends with its last extending span

_RESERVED = frozenset({"index", "t", "section"})
_EPS = 1e-9

# kind -> (mark kind a cue carries, seconds the span starts before its word)
_MARKS = {
    "enter": ("show", SHOW_LEAD_S),
    "text": ("show", SHOW_LEAD_S),
    "caption": ("show", SHOW_LEAD_S),
    "pulse": ("indicate", 0.5 * PULSE_S),
    "ring": ("circumscribe", 0.0),
    "move": ("move", 0.0),
}
_ATTENTION = ("pulse", "ring")
_MOTION = ("enter", "move", "run")
_ARRIVAL = ("text", "caption")
_KINDS = (
    "hold",
    "enter",
    "leave",
    "text",
    "caption",
    "move",
    "run",
    "pulse",
    "ring",
    "line",
)
_UNPACED = ("text", "caption", "pulse", "ring", "run")
_LINEAR = ("hold", "pulse", "ring", "run")


def _exact(value):
    """The exact `Fraction` of a number or `Time`."""
    if isinstance(value, Time):
        return value.exact
    if isinstance(value, Fraction):
        return value
    return Fraction(float(value))


class Time(float):
    """A time in seconds that adds exactly: ``Time + seconds`` is a `Time`.

    A handle's ``start`` and ``end`` are `Time`, so ``at=turn.end + 0.2``
    places a span 0.2 s after another ends without float drift.
    """

    exact: Fraction

    def __new__(cls, value=0.0):
        """Build from seconds (a number, `Fraction` or `Time`)."""
        exact = _exact(value)
        self = super().__new__(cls, float(exact))
        self.exact = exact
        return self

    def __add__(self, other):
        """Add seconds exactly."""
        return Time(self.exact + _exact(other))

    __radd__ = __add__

    def __sub__(self, other):
        """Subtract seconds exactly."""
        return Time(self.exact - _exact(other))

    def __rsub__(self, other):
        """Seconds minus this time, exactly."""
        return Time(_exact(other) - self.exact)


@dataclass(frozen=True)
class Span:
    """One named stretch of the clip, as declared and as resolved.

    The handle a verb returns has no frames yet (``k0`` is None); the records
    of `Plan.spans` carry them.

    Attributes:
        name: Unique among the timeline's span and knob names (lines and
            sections have a namespace of their own).
        kind: One of ``hold``, ``enter``, ``leave``, ``text``, ``caption``,
            ``move``, ``run``, ``pulse``, ``ring``, ``line``.
        section: Name of the section or line it was declared in ("" if none).
        start_s: Start in seconds, exact before frame rounding.
        end_s: End in seconds.
        k0: First frame of the span at the plan's frame rate (None before a
            plan).
        k1: One past its last frame.
        writes: The knobs it writes.
        ease: Easing kind (a name or a callable).
        text: Caption text, else None.
        read_s: Seconds the text needs to be read, else None.
        with_: Names of spans it may overlap (for the arrival checks).
        extends: Whether it holds its section open.
        collapsed: Rounded up to one frame at this frame rate.
        clamped: Its cue fell before its line, so it starts at the line.
    """

    name: str
    kind: str
    section: str
    start_s: float
    end_s: float
    k0: int | None = None
    k1: int | None = None
    writes: tuple = ()
    ease: object = field(default="linear", compare=False)
    text: str | None = None
    read_s: float | None = None
    with_: tuple = ()
    extends: bool = True
    collapsed: bool = False
    clamped: bool = False
    _t0: Fraction = field(default=Fraction(0), repr=False, compare=False)
    _t1: Fraction = field(default=Fraction(0), repr=False, compare=False)

    @property
    def start(self):
        """Start as a `Time`."""
        return Time(self._t0)

    @property
    def end(self):
        """End as a `Time`."""
        return Time(self._t1)

    @property
    def n(self):
        """Number of frames (needs a plan)."""
        return self.k1 - self.k0


@dataclass(frozen=True)
class Event:
    """A moment of the clip, as a time and the frame that shows it.

    Attributes:
        name: ``"water.start"``, ``"water.mid"``, ``"water.end"``,
            ``"cue:water:show:water"``, ``"section:water.end"``.
        what: ``"cue"``, ``"start"``, ``"mid"``, ``"end"``,
            ``"section-start"``, ``"section-end"`` or ``"line-start"``.
        kind: The span kind; "" for cues and sections.
        t_s: Time in seconds.
        frame: The frame that shows it.
        before: ``frame - 1``, clamped at 0, to compare against.
        section: The section or line it is in.
    """

    name: str
    what: str
    kind: str
    t_s: float
    frame: int
    before: int
    section: str


@dataclass(frozen=True)
class Fault:
    """A finding of the clock's checks.

    Attributes:
        code: Short name of the check, for example ``"text-over-motion"``.
        severity: ``"error"`` (raises in a strict timeline) or ``"info"``.
        message: What was found, naming the spans.
        spans: Names of the spans involved.
    """

    code: str
    severity: str
    message: str
    spans: tuple = ()


@dataclass
class _Write:
    """One knob a span writes."""

    knob: str
    to: object
    space: str = "linear"
    start: object = None  # value the ramp starts from; None means the knob's own


class _Section:
    """An open or closed stretch of the clip; a plain call or a ``with`` block."""

    def __init__(self, clock, name, start, beat=None):
        self._clock = clock
        self.name = name
        self._start = start
        self._end = None
        self.beat = beat
        self.key = beat.key if beat is not None else None
        self.max_ext = None
        self.pending = []

    @property
    def start(self):
        """Start as a `Time`."""
        return Time(self._start)

    @property
    def end(self):
        """End as a `Time`; raises while the section is still open."""
        if self._end is None:
            raise ValueError(
                f"{self.name!r} is still open, so its end is not known; read it "
                "after the next section or line begins"
            )
        return Time(self._end)

    def __enter__(self):
        """Enter the block; the section closes on exit."""
        return self

    def __exit__(self, *exc):
        """Close the section unless an error is propagating."""
        if exc[0] is None and self._end is None and self._clock._open is self:
            self._clock._close_open()
        return False


class _Together:
    """Children start together; the cursor moves to the latest end on exit."""

    def __init__(self, clock):
        self._clock = clock

    def __enter__(self):
        """Open the group at the current time."""
        c = self._clock
        c._together.append([c._now, c._now])
        return self

    def __exit__(self, *exc):
        """Advance the cursor to the latest end of the group."""
        _base, latest = self._clock._together.pop()
        self._clock._now = max(self._clock._now, latest)
        return False


def _no_colon(name):
    if not isinstance(name, str) or not name:
        raise ValueError(f"a name must be a non-empty string, got {name!r}")
    if ":" in name:
        raise ValueError(
            f"the name {name!r} holds ':', which event names and cues use as a "
            "separator; pick a name without it"
        )


def _same(a, b):
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return a is b


def _numeric(value):
    return _is_number(value)


class Timeline:
    """Spans of an explanatory animation, declared in seconds.

    Every verb returns a `Span`. Default placement starts a span at ``now``
    and advances it; ``at=`` places a span anywhere (a number of seconds or a
    `Time` from a handle) and never advances ``now``; with a narration, a
    verb with no ``at=`` starts on the word of its cue mark and, unless it is
    a pulse or a ring, advances ``now`` to its end. ``with_`` only declares an
    overlap the checks may allow.

    The span name is the knob it writes: ``enter("x")`` ramps knob ``x`` from
    0 to 1, ``leave("x")`` (span ``"x.leave"``) back to 0, ``pulse("t")``
    writes ``pulse.t``, ``ring("t")`` writes ``ring.t`` and
    ``caption("c", text)`` writes ``caption.c``. A dict ``to={"R": 3.0}``
    writes the named knobs instead. Subject knobs default to 0; a knob a
    ``to`` names must be declared in `knobs`.

    Example::

        tl = Timeline({"R": 1.0}, tempo=Tempo())
        tl.section("intro")
        tl.enter("source")
        slide = tl.move("slide", 2.0, to={"R": 4.0}, space="log")
        tl.pulse("sag", at=slide.end)
        plan = tl.at_fps(30)
        for k in range(plan.n_frames):
            draw(plan.value(k, "source"), plan.value(k, "R"))

    Args:
        knobs: The knob names and their initial values. ``index``, ``t`` and
            ``section`` are reserved for the frame fields.
        tempo: A `Tempo`: paced spans (holds, entries, moves, leaves) last
            ``tempo.seconds(nominal)``, and captions are read for
            ``tempo.read_s``. None paces nothing and reads at the default.
        narration: A `eyepiece.narration.Narration`; each ``line(key)`` then
            places the line's speech, and cue marks place the spans.
        strict: If True a failed check raises; otherwise it is recorded in
            `Plan.faults`.
        max_late_s: The most seconds a cue-placed span may wait for the
            cursor past its cue before ``late`` is an error rather than
            info (a strict timeline raises, otherwise it is a fault in
            `Plan.faults`).

    Raises:
        ValueError: If a knob name is reserved or holds ``:``, or
            `max_late_s` is negative.
    """

    def __init__(
        self, knobs=None, *, tempo=None, narration=None, strict=True, max_late_s=0.5
    ):
        self._initial = dict(knobs or {})
        for name in self._initial:
            if name in _RESERVED:
                raise ValueError(
                    f"knob name {name!r} is reserved for frame fields "
                    f"{sorted(_RESERVED)}"
                )
            _no_colon(name)
        if not float(max_late_s) >= 0.0:
            raise ValueError(f"max_late_s must be at least 0, got {max_late_s!r}")
        self._max_late = Fraction(float(max_late_s))
        self._tempo = tempo
        self._read_tempo = tempo if tempo is not None else Tempo()
        self._source = narration
        self._strict = bool(strict)
        self._spans = {}
        self._writes = {}
        self._owner = {}
        self._names = set()
        self._labels = set()
        self._sections = []
        self._open = None
        self._now = Fraction(0)
        self._faults = []
        self._known = dict.fromkeys(self._initial)
        self._together = []
        self._used = set()
        self._lines = []
        self._pending = []
        self._count = 0
        self._tails = []

    # -- small helpers ---------------------------------------------------

    @property
    def now(self):
        """The cursor, as a `Time`."""
        return Time(self._now)

    @property
    def narration(self):
        """The narration with the solved line starts and ends, or None."""
        if self._source is None:
            return None
        return self._rebuilt()

    @property
    def seconds(self):
        """Declared length so far: the latest end of any extending span."""
        ends = [s._t1 for s in self._spans.values() if s.extends]
        ends += [s._end for s in self._sections if s._end is not None]
        ends += self._tails
        return float(max(ends, default=Fraction(0)))

    def _fault(self, code, message, spans=(), severity="error"):
        fault = Fault(code, severity, message, tuple(spans))
        if severity == "error" and self._strict:
            raise ValueError(message)
        self._faults.append(fault)

    def _fresh(self, base):
        name = base
        while name in self._names:
            self._count += 1
            name = f"{base}{self._count}"
        return name

    def _claim(self, name):
        _no_colon(name)
        if name in self._names:
            raise ValueError(
                f"the span name {name!r} is already used; span and knob names "
                "are unique (lines and sections have their own namespace)"
            )
        self._names.add(name)

    def _claim_label(self, name):
        """Claim a section or line name; only labels clash with labels."""
        _no_colon(name)
        if name in self._labels:
            raise ValueError(
                f"the section or line name {name!r} is already used; sections "
                "and lines share a namespace of their own, apart from spans"
            )
        self._labels.add(name)

    def _secs(self, seconds, paced):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(f"span seconds must be finite and positive, got {seconds}")
        if paced and self._tempo is not None:
            seconds = self._tempo.seconds(seconds)
        return Fraction(seconds)

    def _knob(self, name):
        _no_colon(name)
        if name in _RESERVED:
            raise ValueError(f"{name!r} is reserved for the frame fields")
        self._known.setdefault(name, None)

    def _need_declared(self, names):
        for name in names:
            if name not in self._known:
                near = difflib.get_close_matches(name, list(self._known), n=3)
                hint = f"; did you mean {near}" if near else ""
                raise ValueError(
                    f"knob {name!r} was not declared{hint}; declared knobs are "
                    f"{sorted(self._known)}"
                )

    def _ensure_open(self):
        if self._open is None:
            self._open = _Section(self, "", self._now)
            self._sections.append(self._open)
        return self._open

    # -- sections and lines ----------------------------------------------

    def section(self, name):
        """Open a section, closing the one before it.

        Works as a plain call or a ``with`` block; a section also closes when
        the next section or line opens or the plan is built. It ends at its
        last extending span (a caption read in silent mode included).

        Args:
            name: The section's name, unique among sections and lines.

        Returns:
            The section, whose ``start`` and (once closed) ``end`` are `Time`.
        """
        self._close_open()
        self._claim_label(name)
        self._open = _Section(self, name, self._now)
        self._sections.append(self._open)
        return self._open

    def line(self, key):
        """Open the narration line `key`, closing the one before it.

        Lines open in script order, one per line of the script. The line
        ends at the later of its speech plus padding and its last extending
        span plus a 0.6 s tail.

        Args:
            key: The key of the line in the narration's script.

        Returns:
            The section for the line.

        Raises:
            ValueError: If the timeline has no narration, the line is not the
                next one of the script, or the key holds ``:``.
        """
        if self._source is None:
            raise ValueError("line() needs a Timeline built with narration=")
        order = [ln.key for ln in self._source.script.lines]
        if key not in order:
            raise KeyError(f"no line named {key!r}; lines are {order}")
        want = order[len(self._lines)] if len(self._lines) < len(order) else None
        if key != want:
            raise ValueError(
                f"lines open in script order: the next is {want!r}, not {key!r}"
            )
        self._close_open()
        self._claim_label(key)
        beat = self._source.place(key, float(self._now))
        sec = _Section(self, key, self._now, beat)
        self._open = sec
        self._sections.append(sec)
        self._lines.append(sec)
        return sec

    def _close_open(self):
        sec = self._open
        if sec is None:
            return
        reads = [c for c in sec.pending if c["policy"] == "read"]
        for cap in reads:
            read = cap["arrive_end"] + Fraction(cap["read_s"])
            if sec.max_ext is None or read > sec.max_ext:
                sec.max_ext = read
        for cap in reads:
            self._leave_caption(cap, sec.max_ext, False)
        end = sec._start
        if sec.beat is not None:
            need = None
            if sec.max_ext is not None:
                need = sec.max_ext + Fraction(LINE_TAIL_S)
            placed = self._source.place(
                sec.key, float(sec._start), end_s=None if need is None else float(need)
            )
            sec.beat = placed
            end = Fraction(placed.end_s)
        elif sec.max_ext is not None:
            end = max(end, sec.max_ext + Fraction(SECTION_TAIL_S))
        sec._end = end
        self._now = max(self._now, end)
        self._open = None

    # -- placement -------------------------------------------------------

    def _mark_in(self, sec, kind, target):
        for cue in sec.beat.cues:
            if (cue.kind, cue.target) == (kind, target):
                return cue
        return None

    def _find_cue(self, spec, *, inferred):
        """The `CueTime` for ``kind:target`` from the current line or earlier."""
        kind, sep, target = str(spec).partition(":")
        if not sep or not kind or not target:
            raise ValueError(f"a cue is 'kind:target', got {spec!r}")
        cur = self._open if self._open is not None and self._open.beat else None
        if cur is None:
            raise ValueError(f"cue {spec!r} used outside a narration line")
        for sec in reversed(self._lines[: self._lines.index(cur) + 1]):
            if inferred and sec is not cur:
                break
            cue = self._mark_in(sec, kind, target)
            if cue is not None:
                self._used.add((cue.line, kind, target))
                return cue
        for sec in self._source.script.lines:
            if any((c.kind, c.target) == (kind, target) for c in sec.cues):
                if sec.key not in [ln.key for ln in self._lines]:
                    raise ValueError(
                        f"cue {spec!r} is in line {sec.key!r}, which comes after "
                        f"the current line {cur.key!r}: a span cannot wait on a "
                        "later line"
                    )
        marks = [f"{c.kind}:{c.target}" for c in cur.beat.cues]
        raise ValueError(
            f"line {cur.key!r} has no mark {spec!r}; its marks are {marks}"
        )

    def _start_for(self, kind, at, cue, target):
        """Where a span starts: ``(start, advance, cue_s, clamped, skipped)``."""
        sec = self._ensure_open()
        base = self._together[-1][0] if self._together else self._now
        attention = kind in _ATTENTION
        if at is not None:
            start = _exact(at)
            if start < sec._start:
                raise ValueError(
                    f"at={float(start)} is before the start of section "
                    f"{sec.name!r} ({float(sec._start)})"
                )
            if cue is not None and self._source is not None and sec.beat is not None:
                self._find_cue(cue, inferred=False)
            return start, False, None, False, False
        narrated = self._source is not None and sec.beat is not None
        spec = cue
        if narrated and spec is None and kind in _MARKS:
            spec = f"{_MARKS[kind][0]}:{target}"
            try:
                found = self._find_cue(spec, inferred=True)
            except ValueError as err:
                self._fault("missing-mark", str(err))
                found = None
        elif narrated and spec is not None:
            found = self._find_cue(spec, inferred=False)
        else:
            found = None
        if found is None:
            if attention and not narrated:
                return base, False, None, False, True
            return base, not attention, None, False, False
        lead = Fraction(_MARKS[kind][1]) if kind in _MARKS else Fraction(0)
        cue_s = Fraction(found.at_s)
        raw = cue_s - lead
        clamped = raw < sec._start
        start = max(raw, sec._start)
        if not attention:
            start = max(start, base)
        late = start - raw
        if late > Fraction(1, 10**9) and not clamped:
            over = late > self._max_late
            self._fault(
                "late",
                f"{kind} {target!r} starts {float(late):.3f} s after its cue "
                f"word {found.word.text!r}"
                + (
                    f", over the {float(self._max_late):g} s allowed (max_late_s); "
                    "finish the earlier span sooner, place it with at=, or raise "
                    "max_late_s"
                    if over
                    else ""
                ),
                (target,),
                severity="error" if over else "info",
            )
        return start, not attention, float(cue_s), clamped, False

    def _add(
        self,
        name,
        kind,
        start,
        seconds,
        writes,
        *,
        ease="smoothstep",
        extends=True,
        advance=False,
        text=None,
        read_s=None,
        with_=(),
        clamped=False,
        owner=None,
    ):
        for other in with_:
            if other not in self._spans:
                raise ValueError(
                    f"with_ names {other!r}, which is not an earlier span; spans "
                    f"are {sorted(self._spans)}"
                )
        self._claim(name)
        sec = self._ensure_open()
        end = start + seconds
        span = Span(
            name,
            kind,
            sec.name,
            float(start),
            float(end),
            writes=tuple(w.knob for w in writes),
            ease=ease,
            text=text,
            read_s=read_s,
            with_=tuple(with_),
            extends=extends,
            clamped=clamped,
            _t0=start,
            _t1=end,
        )
        self._spans[name] = span
        self._writes[name] = tuple(writes)
        self._owner[name] = owner
        if extends:
            sec.max_ext = end if sec.max_ext is None else max(sec.max_ext, end)
        if advance:
            if self._together:
                self._together[-1][1] = max(self._together[-1][1], end)
            else:
                self._now = max(self._now, end)
        return span

    def _skipped(self, name, kind):
        t = self._now
        return Span(name, kind, self._ensure_open().name, float(t), float(t),
                    extends=False, _t0=t, _t1=t)  # fmt: skip

    # -- verbs -----------------------------------------------------------

    def hold(self, seconds, *, name=None, to=None):
        """A span in which nothing moves; ``to`` sets knobs when it starts.

        Args:
            seconds: Nominal seconds (paced by the tempo).
            name: The span's name; None generates one.
            to: Knob values that step when the hold starts.

        Returns:
            The `Span`.
        """
        return self._hold(seconds, name, to, paced=True)

    def _hold(self, seconds, name, to, *, paced):
        writes = self._writes_of(to, None, step_only=True)
        name = name if name is not None else self._fresh("hold")
        start, _, _, _, _ = self._start_for("hold", None, None, name)
        return self._add(
            name,
            "hold",
            start,
            self._secs(seconds, paced),
            writes,
            ease="linear",
            advance=True,
        )

    def read(self, text, *, name=None, tempo=None, to=None):
        """A hold long enough to read `text` (`Tempo.read_s`).

        Args:
            text: The words on screen, a string or caption pieces.
            name: The span's name; None generates one.
            tempo: The `Tempo` that sets the hold; None uses the timeline's.
            to: Knob values that step when the hold starts.

        Returns:
            The `Span`.
        """
        tempo = tempo if tempo is not None else self._read_tempo
        return self._hold(tempo.read_s(text), name, to, paced=False)

    def settle(self):
        """Rest for ``Tempo.rest_s`` after a new subject or an axis build."""
        return self._hold(
            self._read_tempo.rest_s, self._fresh("settle"), None, paced=False
        )

    def _writes_of(self, to, name, *, step_only=False, space="linear", start=None):
        if to is None:
            return []
        if not isinstance(to, dict):
            if name is None:
                raise ValueError("to= must be a dict of knob values here")
            to = {name: to}
        self._need_declared(to)
        out = []
        for knob, value in to.items():
            if isinstance(value, str) and value[:2] in ("+=", "-="):
                raise ValueError(f"knob {knob!r}: relative targets are not allowed")
            out.append(_Write(knob, value, space, start))
        if step_only:
            for w in out:
                w.space = "step"
        return out

    def _subject(self, names, value, start=None):
        for n in names:
            self._knob(n)
        return [_Write(n, value, "linear", start) for n in names]

    def _with_caption(self, name, caption, kind, at, cue, paced_seconds):
        """Place a subject span with its caption; returns start and advance."""
        start, advance, cue_s, clamped, _ = self._start_for(kind, at, cue, name)
        narrated = cue_s is not None
        cap = None
        if caption is not None:
            if narrated or at is not None:
                cap = self._caption_span(
                    f"cap_{name}", caption, start, None, owner=name
                )
            else:
                cap = self._caption_span(
                    f"cap_{name}", caption, start, None, owner=name, advance=True
                )
                start = max(start, cap._t1)
        return start, advance, clamped, cap

    def enter(
        self, name, seconds=ENTER_S, *, also=(), caption=None, at=None, cue=None,
        with_=(),
    ):  # fmt: skip
        """A subject arrives: knob `name` (and `also`) ramps from 0 to 1.

        Args:
            name: The subject and the knob; the cue target by default.
            seconds: Nominal seconds (paced by the tempo).
            also: More knobs that arrive with it.
            caption: Caption text. Silent, the caption arrives first and the
                subject follows it; narrated, both start on the cue.
            at: Start time in seconds, or a `Time`; None places by cue or at
                ``now``.
            cue: A full ``"kind:target"`` mark that overrides ``show:name``.
            with_: Spans it may overlap.

        Returns:
            The `Span`.
        """
        writes = self._subject((name, *also), 1.0)
        start, advance, clamped, cap = self._with_caption(
            name, caption, "enter", at, cue, seconds
        )
        pair = (*with_, cap.name) if cap is not None else tuple(with_)
        span = self._add(
            name, "enter", start, self._secs(seconds, True), writes,
            advance=advance, with_=pair, clamped=clamped,
        )  # fmt: skip
        if cap is not None:
            self._pair(cap, span)
        return span

    def _pair(self, cap, span):
        self._spans[cap.name] = replace(
            self._spans[cap.name], with_=(*self._spans[cap.name].with_, span.name)
        )

    def leave(self, name, seconds=LEAVE_S, *, also=(), at=None, cue=None, with_=()):
        """A subject goes: knob `name` (and `also`) ramps back to 0.

        The span is named ``"<name>.leave"``.

        Args:
            name: The subject and the knob.
            seconds: Nominal seconds (paced by the tempo).
            also: More knobs that leave with it.
            at: Start time in seconds or a `Time`; None places at ``now``.
            cue: A full ``"kind:target"`` mark; leaving has none by default.
            with_: Spans it may overlap.

        Returns:
            The `Span`.
        """
        writes = self._subject((name, *also), 0.0)
        start, advance, _, clamped, _ = self._start_for("leave", at, cue, name)
        label = f"{name}.leave"
        n = 1
        while label in self._names:
            n += 1
            label = f"{name}.leave.{n}"
        return self._add(
            label, "leave", start, self._secs(seconds, True), writes,
            advance=advance, with_=with_, clamped=clamped, owner=name,
        )  # fmt: skip

    def text(self, name, seconds=None, *, also=(), at=None, cue=None):
        """Text arrives: knob `name` (and `also`) ramps from 0 to 1.

        Args:
            name: The text and the knob; the cue target by default.
            seconds: Nominal seconds (paced); None is the 0.4 s of a caption.
            also: More knobs that arrive with it.
            at: Start time in seconds or a `Time`.
            cue: A full ``"kind:target"`` mark that overrides ``show:name``.

        Returns:
            The `Span`.
        """
        writes = self._subject((name, *also), 1.0)
        start, advance, _, clamped, _ = self._start_for("text", at, cue, name)
        secs = self._secs(seconds, True) if seconds is not None else Fraction(CAPTION_S)
        return self._add(
            name, "text", start, secs, writes, advance=advance, clamped=clamped
        )

    def _caption_span(self, name, text, start, policy, *, owner=None, advance=False,
                      clamped=False):  # fmt: skip
        if not isinstance(text, str):
            text = "".join(p if isinstance(p, str) else str(p[0]) for p in text)
        knob = f"caption.{name}"
        self._knob(knob)
        self._flush_next(start)
        read = self._read_tempo.read_s(text)
        span = self._add(
            name, "caption", start, Fraction(CAPTION_S), [_Write(knob, 1.0)],
            advance=advance, text=text, read_s=read, clamped=clamped, owner=owner,
        )  # fmt: skip
        if policy is None:
            policy = "next" if self._source is not None else "read"
        sec = self._ensure_open()
        cap = {
            "name": name, "policy": policy, "arrive_end": span._t1,
            "arrive": span._t0, "read_s": read, "knob": knob,
        }  # fmt: skip
        if policy != "never":
            sec.pending.append(cap)
            self._pending.append(cap)
        return span

    def _flush_next(self, at):
        for cap in list(self._pending):
            if cap["policy"] == "next":
                self._leave_caption(cap, at, False)

    def _leave_caption(self, cap, at, extends):
        leave_name = f"{cap['name']}.leave"
        start = max(at, cap["arrive"])
        span = self._add(
            leave_name, "leave", start, self._secs(LEAVE_S, True),
            [_Write(cap["knob"], 0.0)], extends=extends, owner=cap["name"],
        )  # fmt: skip
        if not extends:
            self._tails.append(span._t1)
        need = cap["arrive_end"] + Fraction(cap["read_s"])
        # Speech reads a caption in a narrated clip, so the next caption (or a
        # clear) may take it down; the read time is a silent-clip rule.
        spoken = self._source is not None and cap["policy"] == "next"
        if start < need - Fraction(1, 10**6) and not spoken:
            self._fault(
                "short-read",
                f"caption {cap['name']!r} leaves {float(need - start):.2f} s before "
                "it has been read",
                (cap["name"],),
            )
        for lst in (self._pending, *(s.pending for s in self._sections)):
            if cap in lst:
                lst.remove(cap)
        return span

    def caption(self, name, text, *, leaves=None, at=None, cue=None):
        """A caption arrives over 0.4 s: knob ``caption.<name>`` ramps 0 to 1.

        It leaves once read (arrival plus `Tempo.read_s`), when the next
        caption arrives, or never. The default is to leave once read in a
        silent clip and when the next caption arrives in a narrated one.

        Args:
            name: The span's name; the cue target by default.
            text: The caption, a string or pieces (see `reading_words`).
            leaves: ``"read"``, ``"next"`` or ``"never"``; None picks by mode.
            at: Start time in seconds or a `Time`.
            cue: A full ``"kind:target"`` mark that overrides ``show:name``.

        Returns:
            The `Span`.
        """
        if leaves not in (None, "read", "next", "never"):
            raise ValueError(f"leaves is read, next or never, got {leaves!r}")
        start, advance, _, clamped, _ = self._start_for("caption", at, cue, name)
        return self._caption_span(
            name, text, start, leaves, advance=advance, clamped=clamped
        )

    def clear_caption(self, *, at=None):
        """Take down every caption on screen, now or at `at`.

        Args:
            at: Start time in seconds or a `Time`; None places at ``now``.

        Returns:
            The `Span` of the last caption's leave.

        Raises:
            ValueError: If no caption is on screen.
        """
        if not self._pending:
            raise ValueError("no caption is on screen to clear")
        start, _, _, _, _ = self._start_for("leave", at, None, "caption")
        span = None
        for cap in list(self._pending):
            span = self._leave_caption(cap, start, True)
        if at is None and not self._together:
            self._now = max(self._now, span._t1)
        return span

    def move(
        self, name, seconds, *, to=None, space="linear", ease="smoothstep",
        caption=None, at=None, cue=None, with_=(),
    ):  # fmt: skip
        """Knobs move to targets; without `to` it is a progress-only span.

        Args:
            name: The span's name; the knob when `to` is a number, and the
                cue target by default.
            seconds: Nominal seconds (paced by the tempo).
            to: A number for knob `name`, or a dict of knob targets. Knobs
                must be declared; a numeric knob ramps from its value on the
                frame before, a non-numeric one steps at the start.
            space: ``"linear"``, or ``"log"`` (positive ends).
            ease: An easing kind or a callable on [0, 1].
            caption: Caption text; see `enter`.
            at: Start time in seconds or a `Time`.
            cue: A full ``"kind:target"`` mark that overrides ``move:name``.
            with_: Spans it may overlap.

        Returns:
            The `Span`.
        """
        return self._move("move", name, seconds, to, space, ease, True, caption,
                          at, cue, with_, True)  # fmt: skip

    def _move(self, kind, name, seconds, to, space, ease, paced, caption, at, cue,
              with_, extends):  # fmt: skip
        if space not in _SPACES:
            raise ValueError(f"unknown space {space!r}; use one of {_SPACES}")
        if not callable(ease) and ease not in _EASINGS:
            raise ValueError(f"unknown easing {ease!r}; use one of {_EASINGS}")
        writes = self._writes_of(to, name, space=space)
        start, advance, clamped, cap = self._with_caption(
            name, caption, kind, at, cue, seconds
        )
        pair = (*with_, cap.name) if cap is not None else tuple(with_)
        span = self._add(
            name, kind, start, self._secs(seconds, paced), writes, ease=ease,
            extends=extends, advance=advance, with_=pair, clamped=clamped,
        )  # fmt: skip
        if cap is not None:
            self._pair(cap, span)
        return span

    def run(self, name, seconds, *, to=None, at=None):
        """A linear run in seconds on screen, a rate: not paced, not eased.

        Args:
            name: The span's name; the knob when `to` is a number.
            seconds: Seconds on screen.
            to: A number for knob `name`, or a dict of knob targets.
            at: Start time in seconds or a `Time`.

        Returns:
            The `Span`.
        """
        return self._move("run", name, seconds, to, "linear", "linear", False, None,
                          at, None, (), True)  # fmt: skip

    def _attention(self, kind, target, seconds, at, cue):
        base = f"{kind}.{target}"
        name = base
        n = 1
        while name in self._names:
            n += 1
            name = f"{base}.{n}"
        start, _, _, clamped, skipped = self._start_for(kind, at, cue, target)
        knob = base
        self._knob(knob)
        if skipped:
            self._fault(
                "skipped",
                f"{kind} {target!r} is placed only by a cue and the clip is silent",
                (name,),
                severity="info",
            )
            return self._skipped(name, kind)
        return self._add(
            name, kind, start, Fraction(seconds), [_Write(knob, 1.0, "linear", 0.0)],
            ease="linear", extends=False, clamped=clamped,
        )  # fmt: skip

    def pulse(self, target, *, at=None, cue=None):
        """A pulse on `target`: knob ``pulse.<target>`` runs 0 to 1 in 0.9 s.

        It is centered on its word, never holds a section open, and in a
        silent clip is skipped unless `at` places it.

        Args:
            target: What it acts on; the cue target (``indicate:<target>``).
            at: Start time in seconds or a `Time`.
            cue: A full ``"kind:target"`` mark that overrides the default.

        Returns:
            The `Span`.
        """
        return self._attention("pulse", target, PULSE_S, at, cue)

    def ring(self, target, *, at=None, cue=None):
        """An outline on `target`: knob ``ring.<target>`` runs 0 to 1 in 1.3 s.

        It starts on its word, never holds a section open, and in a silent
        clip is skipped unless `at` places it.

        Args:
            target: What it acts on; the cue target (``circumscribe:<target>``).
            at: Start time in seconds or a `Time`.
            cue: A full ``"kind:target"`` mark that overrides the default.

        Returns:
            The `Span`.
        """
        return self._attention("ring", target, RING_S, at, cue)

    def span(
        self, name, seconds, *, kind, to=None, paced=None, ease=None, extends=None,
        at=None, cue=None, with_=(),
    ):  # fmt: skip
        """A span with every default stated, for the cases the verbs do not cover.

        Args:
            name: The span's name.
            seconds: Seconds (paced by the tempo when `paced`).
            kind: One of ``hold``, ``enter``, ``leave``, ``text``,
                ``caption``, ``move``, ``run``, ``pulse``, ``ring``.
            to: A number for knob `name`, or a dict of knob targets; None is a
                progress-only span. A ``hold`` steps its targets at the start.
            paced: Pace by the tempo; None follows the kind.
            ease: An easing kind or a callable; None follows the kind.
            extends: Whether it holds its section open; None follows the kind.
            at: Start time in seconds or a `Time`.
            cue: A full ``"kind:target"`` mark; kinds with a mark infer one.
            with_: Spans it may overlap.

        Returns:
            The `Span`.
        """
        if kind not in _KINDS or kind == "line":
            raise ValueError(f"unknown span kind {kind!r}; use one of {_KINDS[:-1]}")
        paced = (kind not in _UNPACED) if paced is None else paced
        ease = ("linear" if kind in _LINEAR else "smoothstep") if ease is None else ease
        extends = (kind not in _ATTENTION) if extends is None else extends
        writes = self._writes_of(to, name, step_only=(kind == "hold"))
        start, advance, _, clamped, _ = self._start_for(kind, at, cue, name)
        return self._add(
            name, kind, start, self._secs(seconds, paced), writes, ease=ease,
            extends=extends, advance=advance, with_=with_, clamped=clamped,
        )  # fmt: skip

    def together(self):
        """A ``with`` block whose spans all start at ``now``.

        On exit ``now`` is the latest end of its spans.
        """
        return _Together(self)

    def stagger(self, verb, targets, *, each_s, **kwargs):
        """Call `verb` on each target, each one `each_s` seconds after the last.

        Every call is placed with ``at=``, so none of them moves ``now``
        by itself. When the verb extends its section (``enter``, ``leave``,
        ``text``, ``move`` and the like), ``now`` then advances to the latest
        end, so what follows is placed after the whole group, the way
        sequential calls would be. A pulse or a ring does not extend, so
        ``now`` stays put.

        Staggered ``enter`` spans must not overlap: one subject enters at a
        time. If `each_s` is shorter than an entry, the call raises; use
        sequential calls, or an `each_s` at least as long as the entry.

        Args:
            verb: ``"pulse"``, ``"ring"``, ``"enter"``, ``"leave"`` or another
                verb whose first argument names its subject.
            targets: The names, in order.
            each_s: Seconds between starts.
            **kwargs: Passed to the verb.

        Returns:
            The list of `Span`.

        Raises:
            ValueError: If staggered ``enter`` spans overlap in time.
        """
        base = self._now
        fn = getattr(self, verb)
        out = [
            fn(t, at=base + i * float(each_s), **kwargs) for i, t in enumerate(targets)
        ]
        if verb == "enter":
            for a, b in itertools.pairwise(out):
                if b._t0 < a._t1 and b.name not in a.with_ and a.name not in b.with_:
                    raise ValueError(
                        f"staggered enters overlap: {a.name!r} lasts until "
                        f"{float(a._t1):.3f} s but {b.name!r} starts at "
                        f"{float(b._t0):.3f} s, and one subject enters at a "
                        f"time; use each_s of at least {float(a._t1 - a._t0):.3f} "
                        "s or place the enters one after another"
                    )
        ends = [s._t1 for s in out if s.extends]
        if ends:
            latest = max(ends)
            if self._together:
                self._together[-1][1] = max(self._together[-1][1], latest)
            else:
                self._now = max(self._now, latest)
        return out

    def cue(self, spec):
        """When the mark ``kind:target`` is said: the word start, no lead.

        The mark is in the current line or an earlier one. A silent clip
        returns ``now``.

        Args:
            spec: A full ``"kind:target"`` string.

        Returns:
            The word start as a `Time`.

        Raises:
            ValueError: If the mark is not in the current or an earlier line.
        """
        if self._source is None:
            return self.now
        return Time(Fraction(self._find_cue(spec, inferred=False).at_s))

    # -- narration -------------------------------------------------------

    def _rebuilt(self):
        src = self._source
        starts, ends = {}, {}
        for sec in self._lines:
            starts[sec.key] = float(sec._start)
            end = sec._end
            if end is None:
                need = None
                if sec.max_ext is not None:
                    need = float(sec.max_ext + Fraction(LINE_TAIL_S))
                end = Fraction(src.place(sec.key, starts[sec.key], end_s=need).end_s)
            ends[sec.key] = float(end)
        return type(src)(src.script, src.spoken, starts=starts, ends=ends)

    # -- resolving -------------------------------------------------------

    def at_fps(self, fps):
        """Resolve the spans for one frame rate.

        Closes the open section. Every boundary is ``frame_count`` of the
        exact time, so the total is ``frame_count`` of the clip's length.

        Args:
            fps: Frame rate in frames per second.

        Returns:
            A `Plan`.

        Raises:
            ValueError: If the timeline is empty, a knob has two writers on
                one frame, a log move has a nonpositive end, a script line was
                never opened, or (strict) a check fails.
        """
        self._close_open()
        if not self._spans:
            raise ValueError("the timeline has no spans")
        fps = float(fps)
        found = []
        info = [f for f in self._faults]
        narration = None
        if self._source is not None:
            narration = self._rebuilt()
            if len(self._lines) < len(self._source.script.lines):
                left = [ln.key for ln in self._source.script.lines[len(self._lines) :]]
                raise ValueError(f"script lines never opened with line(): {left}")
            for b in narration.beats:
                for c in b.cues:
                    if (b.key, c.kind, c.target) not in self._used:
                        found.append(
                            Fault(
                                "unused-mark",
                                "error",
                                f"line {b.key!r}: {c.kind}:{c.target} is fired by "
                                "no verb",
                                (b.key,),
                            )
                        )
        clip = max(
            [s._t1 for s in self._spans.values() if s.extends]
            + [sec._end for sec in self._sections if sec._end is not None]
            + self._tails
        )
        if narration is not None:
            clip = max(clip, Fraction(narration.total_s))
        n_frames = frame_count(float(clip), fps)
        spans = {}
        for name, s in self._spans.items():
            k0 = frame_count(float(s._t0), fps)
            raw = frame_count(float(s._t1), fps)
            k1 = max(k0 + 1, raw)
            if s.extends:
                n_frames = max(n_frames, k1)
            spans[name] = replace(s, k0=k0, k1=k1, collapsed=raw <= k0)
        for name, s in list(spans.items()):
            if s.k1 > n_frames:
                # Truncate at the clip end and keep the speed: the span keeps
                # its length in frames, so its last frames are never drawn.
                k0 = min(s.k0, n_frames - 1)
                cut = k0 + s.n - n_frames
                spans[name] = replace(s, k0=k0, k1=k0 + s.n)
                info.append(
                    Fault(
                        "clamped-to-clip",
                        "info",
                        f"{s.kind} {name!r} ends after the clip and was "
                        f"truncated by {cut} frames at the clip end",
                        (name,),
                    )
                )
        for name, s in spans.items():
            if s.collapsed:
                found.append(
                    Fault(
                        "collapsed",
                        "error",
                        f"{s.kind} {name!r} ({s.end_s - s.start_s:.4g} s) rounds to "
                        f"no frames at {fps:g} fps",
                        (name,),
                    )
                )
        knobs = self._build_knobs(spans, n_frames)
        found += self._kind_checks(spans)
        if self._strict and found:
            raise ValueError("; ".join(dict.fromkeys(f.message for f in found)))
        sections = []
        for sec in self._sections:
            if sec._end is None or (not sec.name and sec.beat is None):
                continue
            a = frame_count(float(sec._start), fps)
            b = max(a + 1, frame_count(float(sec._end), fps))
            sections.append(
                (sec.name, sec.key, float(sec._start), float(sec._end), a, b)
            )
        return Plan(fps, n_frames, spans, knobs, tuple(sections),
                    tuple(info + found), narration)  # fmt: skip

    def _build_knobs(self, spans, n):
        by_knob = {}
        for name, s in spans.items():
            for w in self._writes[name]:
                by_knob.setdefault(w.knob, []).append((s, w))
        values = {}
        for knob in self._known:
            initial = self._initial.get(knob, 0.0)
            vals = [initial] * n
            writers = sorted(by_knob.get(knob, []), key=lambda p: p[0].k0)
            prev = None
            for s, w in writers:
                step = w.space == "step" or not _numeric(w.to)
                end = s.k0 + 1 if step else s.k1
                if prev is not None and s.k0 < prev[1]:
                    raise ValueError(
                        f"knob {knob!r} is written by {prev[0]!r} and {s.name!r} on "
                        f"the same frames ({s.k0} to {min(prev[1], end) - 1})"
                    )
                prev = (s.name, end)
                if s.k0 >= n:
                    continue
                if step:
                    vals[s.k0 :] = [w.to] * (n - s.k0)
                    continue
                frm = w.start
                if frm is None:
                    frm = vals[s.k0 - 1] if s.k0 > 0 else initial
                if not _numeric(frm):
                    raise ValueError(f"knob {knob!r} must be numeric to move")
                if w.space == "log" and (frm <= 0 or w.to <= 0):
                    raise ValueError(
                        f"a log-space move of {knob!r} needs positive ends, got "
                        f"{frm} and {w.to}"
                    )
                m = s.k1 - s.k0
                for j in range(min(m, n - s.k0)):
                    e = _apply_ease((j + 1) / m, s.ease)
                    vals[s.k0 + j] = (
                        w.to if j == m - 1 else _interp(frm, w.to, e, w.space)
                    )
                vals[s.k1 :] = [w.to] * max(0, n - s.k1)
            values[knob] = vals
        return values

    def _kind_checks(self, spans):
        out = []
        items = list(spans.values())

        def hit(a, b):
            return a.k0 < b.k1 and b.k0 < a.k1

        def paired(a, b):
            return b.name in a.with_ or a.name in b.with_

        for a in items:
            if a.kind in _ARRIVAL:
                for b in items:
                    if b.kind in _MOTION and hit(a, b) and not paired(a, b):
                        out.append(
                            Fault(
                                "text-over-motion",
                                "error",
                                f"{a.kind} {a.name!r} arrives while {b.kind} "
                                f"{b.name!r} moves; text arrives before motion "
                                "(declare the pair with with_= to allow it)",
                                (a.name, b.name),
                            )
                        )
                    if b.kind == "leave" and hit(a, b):
                        mine = self._owner.get(b.name)
                        theirs = self._owner.get(a.name)
                        if mine is not None and mine in (a.name, theirs):
                            out.append(
                                Fault(
                                    "leave-over-arrival",
                                    "error",
                                    f"{b.name!r} leaves while its own {a.kind} "
                                    f"{a.name!r} arrives",
                                    (a.name, b.name),
                                )
                            )
        enters = [s for s in items if s.kind == "enter"]
        for i, a in enumerate(enters):
            for b in enters[i + 1 :]:
                if hit(a, b) and not paired(a, b):
                    out.append(
                        Fault(
                            "enter-overlap",
                            "error",
                            f"enter {a.name!r} and enter {b.name!r} arrive on the "
                            "same frames; one subject enters at a time",
                            (a.name, b.name),
                        )
                    )
        return out


class _FrameDict(dict):
    """A frame's dict that records which knobs a draw function reads."""

    def __init__(self, data, read):
        super().__init__(data)
        self._read = read

    def __getitem__(self, key):
        self._read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._read.add(key)
        return super().get(key, default)

    def __iter__(self):
        self._read.update(super().keys())
        return super().__iter__()

    def items(self):
        self._read.update(super().keys())
        return super().items()

    def values(self):
        self._read.update(super().keys())
        return super().values()


class _Frames:
    """The frames of a plan as dicts built on demand, each a fresh copy."""

    def __init__(self, plan):
        self._plan = plan

    def __len__(self):
        return self._plan.n_frames

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self[i] for i in range(*k.indices(len(self)))]
        n = len(self)
        if k < 0:
            k += n
        if not 0 <= k < n:
            raise IndexError(f"frame {k} is outside the plan's {n} frames")
        return self._plan._frame(k)

    def __iter__(self):
        return (self[k] for k in range(len(self)))


class Plan:
    """A timeline resolved for one frame rate.

    Attributes:
        fps: The frame rate.
        n_frames: Total frame count.
        spans: Name to `Span`, in declaration order, with ``k0`` and ``k1``.
        faults: The `Fault` of every non-strict check, and the info findings.
        narration: The narration with the solved line ends, or None.
        frames: One dict per frame, each a fresh copy: the knob values plus
            ``index``, ``t`` (seconds at the start of the frame) and
            ``section``.
    """

    def __init__(self, fps, n_frames, spans, knobs, sections, faults, narration):
        self.fps = fps
        self.n_frames = n_frames
        self.spans = spans
        self.faults = faults
        self.narration = narration
        self._knobs = knobs
        self._sections = sections
        self._events = None
        self._read = set()
        self.frames = _Frames(self)
        labels = [""] * n_frames
        for name, _key, _a, _b, k0, k1 in sections:
            for k in range(k0, min(k1, n_frames)):
                labels[k] = name
        self._labels = labels

    @property
    def seconds(self):
        """Running time at this frame rate."""
        return self.n_frames / self.fps

    @property
    def knobs(self):
        """Names of the knobs a draw function may read."""
        return tuple(self._knobs)

    def _frame(self, k):
        out = {name: vals[k] for name, vals in self._knobs.items()}
        out.update(index=k, t=k / self.fps, section=self._labels[k])
        return _FrameDict(out, self._read)

    def unused_knobs(self):
        """Knobs a draw function has not read, and knobs that never change.

        Reads are those made through `value`, `frames` and `progress` since
        the plan was built (or since `record` began). The ``pulse.*`` and
        ``ring.*`` knobs are left out of both lists: every pulse or ring
        target has one, and in a silent clip they stay at 0 by design.

        Returns:
            A pair of sorted tuples, ``(unread, constant)``.
        """
        keep = [k for k in self._knobs if not k.startswith(("pulse.", "ring."))]
        unread = tuple(sorted(k for k in keep if k not in self._read))
        constant = []
        for k in keep:
            first = self._knobs[k][0]
            if all(_same(v, first) for v in self._knobs[k]):
                constant.append(k)
        return unread, tuple(sorted(constant))

    def value(self, k, knob):
        """The value of `knob` on frame `k`.

        The frame fields ``"t"`` (seconds), ``"index"`` and ``"section"``
        are also accepted; reading one does not count as knob use.

        Raises:
            KeyError: If the knob is unknown, naming near matches.
            IndexError: If `k` is outside the plan.
        """
        if knob in _RESERVED:
            if not 0 <= k < self.n_frames:
                raise IndexError(
                    f"frame {k} is outside the plan's {self.n_frames} frames"
                )
            return {"index": k, "t": k / self.fps, "section": self._labels[k]}[knob]
        self._read.add(knob)
        try:
            vals = self._knobs[knob]
        except KeyError:
            near = difflib.get_close_matches(knob, list(self._knobs), n=3)
            hint = f"; did you mean {near}" if near else ""
            raise KeyError(
                f"no knob named {knob!r}{hint}; knobs are {sorted(self._knobs)}"
            ) from None
        if not 0 <= k < self.n_frames:
            raise IndexError(f"frame {k} is outside the plan's {self.n_frames} frames")
        return vals[k]

    def _span(self, name):
        try:
            return self.spans[name]
        except KeyError:
            raise KeyError(f"no span named {name!r}") from None

    def progress(self, k, span, *, eased=True):
        """Progress of `span` at frame `k`: 0 before it, 1 from its end on.

        Args:
            k: Frame index.
            span: The span's name.
            eased: Apply the span's easing.

        Returns:
            A float in [0, 1].
        """
        s = self._span(span)
        self._read.update(s.writes)
        if k < s.k0:
            return 0.0
        if k >= s.k1:
            return 1.0
        r = (k - s.k0 + 1) / s.n
        if not eased:
            return r
        return min(1.0, max(0.0, _apply_ease(r, s.ease)))

    def frame_of(self, span, where="last"):
        """Frame of a span's ``"start"``, ``"mid"`` or ``"last"`` (settled) state."""
        s = self._span(span)
        if where == "start":
            return s.k0
        if where == "mid":
            return s.k0 + (s.n - 1) // 2
        if where == "last":
            return s.k1 - 1
        raise ValueError(f"where is start, mid or last, got {where!r}")

    def _make_events(self):
        out = []
        fps, last = self.fps, self.n_frames - 1

        def event(name, what, kind, t, frame, section):
            frame = min(max(frame, 0), last)
            out.append(Event(name, what, kind, t, frame, max(frame - 1, 0), section))

        for name, s in self.spans.items():
            mid = s.k0 + (s.n - 1) // 2
            event(f"{name}.start", "start", s.kind, s.start_s, s.k0, s.section)
            event(f"{name}.mid", "mid", s.kind, 0.5 * (s.start_s + s.end_s), mid,
                  s.section)  # fmt: skip
            event(f"{name}.end", "end", s.kind, s.end_s, s.k1 - 1, s.section)
        for name, key, a, b, k0, k1 in self._sections:
            if key is not None:
                event(f"line:{name}.start", "line-start", "", a, k0, name)
            event(f"section:{name}.start", "section-start", "", a, k0, name)
            event(f"section:{name}.end", "section-end", "", b, k1 - 1, name)
        if self.narration is not None:
            for c in self.narration.cues():
                event(
                    f"cue:{c.line}:{c.kind}:{c.target}", "cue", "", c.at_s,
                    frame_count(c.at_s, fps), c.line,
                )  # fmt: skip
        out.sort(key=lambda e: (e.t_s, e.what))
        return out

    def events(self, *, what=None, kind=None, section=None):
        """The events of the clip, sorted by time.

        Args:
            what: Keep only this ``what`` (``"cue"``, ``"start"``, ...).
            kind: Keep only events of spans of this kind.
            section: Keep only events in this section or line.

        Returns:
            A list of `Event`.
        """
        if self._events is None:
            self._events = self._make_events()
        return [
            e
            for e in self._events
            if (what is None or e.what == what)
            and (kind is None or e.kind == kind)
            and (section is None or e.section == section)
        ]

    def review_frames(self):
        """The frames a contact sheet draws, sorted and unique.

        Every cue frame and the frame before it, every span middle and end,
        and every section end.
        """
        frames = set()
        for e in self.events():
            if e.what == "cue":
                frames.update((e.frame, e.before))
            elif e.what in ("mid", "end", "section-end"):
                frames.add(e.frame)
        return tuple(sorted(frames))

    def spans_json(self):
        """Each span's ``[start_s, end_s]`` rounded to milliseconds, for a manifest."""
        return {
            name: [round(s.start_s, 3), round(s.end_s, 3)]
            for name, s in self.spans.items()
        }
