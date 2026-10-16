"""One color, one meaning: check a figure or a clip for colors that change meaning.

A viewer remembers a color across a clip. A cyan outline that marks a library
badge in one clip and an Earth twin in the next has taught two meanings to one
hue, and no per-frame check sees it. `meaning_faults` poses the figure at
sample times, reads which meaning key each drawn artist carries, and reports
four kinds of fault:

- ``"same-frame"``: two different keys, not in a sharing set, drawn on one
  frame in colors closer than ``fault_de`` (a warning up to ``warn_de``),
  measured in CAM02-UCS under normal vision and the three dichromacies at full
  severity. A pair that differs in marker, line style or fill has a second
  channel and is reported one level lower.
- ``"rebound"``: over the whole clip, one hue cluster that serves more than one
  key. Reported with each key's first and last visible time and the gap
  between them. Only a declared ``rebind`` time releases a hue.
- ``"cue-form"``: a cue key drawn as a line, marker or text rather than a ring,
  band or underlay, or a cue hue that another key of the clip also uses.
- ``"unkeyed"``: a visible colored artist that carries no key.

A key is attached to an artist by its ``gid``: the ``keys`` mapping gives a
``gid`` prefix and the key it stands for, and the longest matching prefix wins.
An artist with no ``gid`` takes the one of its nearest ancestor that has one.

The result is a list of `MeaningFault` records. `meaning_report` prints them,
waived ones included, and says "no tier 1 or 2 findings" when there are none,
because whether the keys are the right meanings is a judgment the check cannot
make. `assert_meaning` raises `MeaningError` for the faults of a test.

Example::

    import eyepiece.testing as ept

    faults = ept.meaning_faults(
        fig, at=clip.pose, times=[0.0, 1.5, 3.0],
        keys={"orbit": "planet", "badge": "library"},
        sharing=[("disk", "prior")], cue_keys=["cue"],
    )
    print(ept.meaning_report(faults))
"""

import itertools
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
from matplotlib.axis import Axis
from matplotlib.collections import Collection, LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.image import AxesImage, BboxImage, FigureImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.spines import Spine
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from eyepiece.attention import _renderer

__all__ = [
    "MEANING_KINDS",
    "MeaningError",
    "MeaningFault",
    "assert_meaning",
    "meaning_faults",
    "meaning_faults_chain",
    "meaning_report",
]

MEANING_KINDS = ("same-frame", "rebound", "cue-form", "unkeyed")

_SHOWN = 0.05  # opacity below which an artist counts as not drawn (as in testing)
_INK_FLOOR = 4.0  # smallest drawn extent, in square pixels, that counts as ink
_INVISIBLE = (
    0.02  # largest per-channel RGB difference from the background that is no ink
)
_NEUTRAL = 0.06  # largest max-minus-min RGB spread of a neutral gray
_CVD = ("deuteranomaly", "protanomaly", "tritanomaly")
_LEVELS = ("fault", "warning")
_IMAGES = (AxesImage, BboxImage, FigureImage)


class _Part(NamedTuple):
    """One drawn color of one artist on one frame."""

    key: object  # meaning key, or None when the artist carries none
    gid: object
    form: str  # "line", "marker", "text", "patch" or "collection"
    eff: tuple  # RGB over the background it sits on, after opacity and dimming
    base: tuple  # RGB before opacity
    alpha: float
    filled: bool
    zorder: float
    sig: tuple  # (marker, line style, hatch): the non-color channels
    scope: object  # id of the axes, None at figure level
    artist: object


@dataclass(frozen=True, eq=False)
class MeaningFault:
    """One finding of the meaning check.

    Attributes:
        kind: One of `MEANING_KINDS`.
        severity: ``"fault"``, ``"warning"`` or ``"advisory"`` (an unkeyed
            artist when the check is not strict).
        keys: The meaning keys involved: a pair for ``"same-frame"`` and
            ``"rebound"``, one key for ``"cue-form"``, and the artist's ``gid``
            or type name for ``"unkeyed"``.
        times: The sample times at which the finding holds.
        detail: A one-line description.
        de: The smallest CAM02-UCS distance behind the finding, None when the
            finding is not about a color distance.
        artists: The artists at fault on the first affected frame.
        waived: The reason text when a waiver covers the finding, else None.
    """

    kind: str
    severity: str
    keys: tuple
    times: tuple
    detail: str
    de: object = None
    artists: tuple = ()
    waived: object = None

    @property
    def ident(self):
        """The waiver identifier: the kind, a colon, the key or ``a+b`` pair."""
        return f"{self.kind}:{'+'.join(sorted(self.keys))}"

    def __str__(self):
        """One line: waiver status, severity, kind, detail and times."""
        times = ", ".join("as drawn" if t is None else f"{t:g}" for t in self.times)
        shown = (
            times if len(self.times) <= 4 else f"{times[:40]}... ({len(self.times)})"
        )
        tag = f"waived ({self.waived}) " if self.waived is not None else ""
        return f"{tag}[{self.severity}] {self.kind}: {self.detail} (t = {shown})"


class MeaningError(AssertionError):
    """A figure or clip has meaning faults.

    Attributes:
        faults: The list of `MeaningFault`, waived ones included.
    """

    def __init__(self, faults):
        """Build the report from the faults."""
        self.faults = list(faults)
        super().__init__(meaning_report(self.faults))


def _vision():
    """The `colorspacious` module, imported on first use."""
    try:
        import colorspacious  # lazy: keeps `import eyepiece` light
    except ImportError as err:
        raise ImportError(
            "the meaning check measures CAM02-UCS distances with colorspacious; "
            "install it with `pip install colorspacious` or `pip install "
            "'eyepiece[test]'`"
        ) from err
    return colorspacious


_LAB = {}


def _lab(rgb):
    """CAM02-UCS coordinates of one sRGB color under the four vision types, (4, 3)."""
    key = tuple(round(255 * float(c)) for c in rgb)
    if key not in _LAB:
        cs = _vision()
        srgb = np.array(key, dtype=float) / 255.0
        rows = [cs.cspace_convert(srgb, "sRGB1", "CAM02-UCS")]
        for cvd in _CVD:
            space = {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}
            rows.append(cs.cspace_convert(srgb, space, "CAM02-UCS"))
        _LAB[key] = np.array(rows)
    return _LAB[key]


def _de(a, b, vision=None):
    """Smallest CAM02-UCS distance of two colors over the four vision types.

    `vision` is an index into (normal, deuteranomaly, protanomaly, tritanomaly)
    to measure that one type.
    """
    d = np.linalg.norm(_lab(a) - _lab(b), axis=1)
    return float(d.min()) if vision is None else float(d[vision])


def _neutral(rgb):
    return max(rgb) - min(rgb) <= _NEUTRAL


def _shown(rgba):
    return rgba[3] >= _SHOWN


def _over(rgba, background):
    a = rgba[3]
    return tuple(a * rgba[i] + (1.0 - a) * background[i] for i in range(3))


def _background(artist, fig, cache):
    ax = artist.axes
    if ax in cache:
        return cache[ax]
    color = to_rgba(fig.get_facecolor())[:3]
    if ax is not None and ax.patch.get_visible():
        face = to_rgba(ax.patch.get_facecolor())
        if face[3] > 0.0:
            color = _over(face, color)
    cache[ax] = color
    return color


def _walk(artist, gid, fig):
    """(artist, gid) below `artist`: the gid is the nearest ancestor's when unset."""
    own = artist.get_gid()
    gid = own if own else gid
    yield artist, gid
    for child in artist.get_children():
        if not child.get_visible():
            continue
        if isinstance(child, (Axis, Spine, Legend)) or child is fig.patch:
            continue
        if getattr(artist, "patch", None) is child:
            continue
        yield from _walk(child, gid, fig)


def _extent(artist, renderer, dpi, pad_pt):
    """Drawn area in square pixels, clipped to the artist's clip box."""
    try:
        box = artist.get_window_extent(renderer)
    except (TypeError, ValueError, AttributeError):
        return 0.0
    if not np.all(np.isfinite([box.x0, box.y0, box.x1, box.y1])):
        return 0.0
    pad = 0.5 * pad_pt * dpi / 72.0
    box = Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
    if artist.get_clip_on() and artist.get_clip_box() is not None:
        box = Bbox.intersection(box, artist.get_clip_box())
        if box is None:
            return 0.0
    return float(box.width * box.height)


def _unique(colors, limit=6):
    seen, out = set(), []
    for rgba in colors:
        k = tuple(round(float(c), 3) for c in rgba)
        if k not in seen:
            seen.add(k)
            out.append(tuple(float(c) for c in rgba))
        if len(out) >= limit:
            break
    return out


def _parts_of(artist, gid, key, fig, renderer, backgrounds, owner=None):
    """The drawn `_Part` records of one artist; empty when it is not drawn.

    A text with a colored box also yields the box's parts, so an outline drawn
    around a name is read; a gray backing box is not a mark and is skipped.
    `owner` is the text whose box this is.
    """
    dpi = fig.dpi
    host = artist if owner is None else owner
    bg = _background(host, fig, backgrounds)
    scope = id(host.axes) if host.axes is not None else None
    z = float(artist.get_zorder())
    raw = []  # (rgba, form, filled, sig, pad_pt)
    extra = []
    if isinstance(artist, Line2D):
        alpha = artist.get_alpha()
        alpha = 1.0 if alpha is None else float(alpha)
        marker = artist.get_marker()
        has_marker = marker not in (None, "None", "none", "", " ")
        style = artist.get_linestyle()
        has_line = style not in ("None", "none", "", " ") and artist.get_linewidth() > 0
        if not has_line and not has_marker:
            return []
        rgba = to_rgba(artist.get_color())
        rgba = (*rgba[:3], rgba[3] * alpha)
        pad = max(artist.get_linewidth() if has_line else 0.0, artist.get_markersize())
        form = "line" if has_line else "marker"
        sig = (marker if has_marker else None, style if has_line else None, None)
        raw.append((rgba, form, False, sig, pad))
    elif isinstance(artist, Patch):
        hatch = artist.get_hatch()
        face = to_rgba(artist.get_facecolor())
        edge = to_rgba(artist.get_edgecolor())
        filled = artist.get_fill() and face[3] > 0.0
        lw = artist.get_linewidth()
        if filled:
            raw.append((face, "patch", True, (None, "fill", hatch), lw))
        if lw > 0.0 and edge[3] > 0.0 and (not filled or edge[:3] != face[:3]):
            raw.append(
                (edge, "patch", False, (None, artist.get_linestyle(), hatch), lw)
            )
    elif isinstance(artist, Collection):
        if artist.get_array() is not None:
            return []  # colormapped data (a mesh, a colorbar) is an image, not a mark
        if isinstance(artist, LineCollection):
            colors, form, filled = artist.get_edgecolor(), "line", False
        else:
            faces = artist.get_facecolor()
            filled = len(faces) > 0 and faces[0][3] > 0.0
            colors = faces if filled else artist.get_edgecolor()
            form = "marker" if isinstance(artist, PathCollection) else "collection"
        hatch = artist.get_hatch()
        style = None if form == "marker" else "fill" if filled else "-"
        for rgba in _unique(colors):
            raw.append((rgba, form, filled, (None, style, hatch), 1.0))
    elif isinstance(artist, Text):
        if not artist.get_text().strip():
            return []
        alpha = artist.get_alpha()
        alpha = 1.0 if alpha is None else float(alpha)
        rgba = to_rgba(artist.get_color())
        raw.append(((*rgba[:3], rgba[3] * alpha), "text", False, (None, None, None), 0))
        box = artist.get_bbox_patch()
        if box is not None and box.get_visible():
            sub = _parts_of(box, gid, key, fig, renderer, backgrounds, owner=artist)
            extra = [p for p in sub if not _neutral(p.base)]  # gray boxes are backing
    else:
        return []
    out = list(extra)
    area = None
    for rgba, form, filled, sig, pad in raw:
        if not _shown(rgba):
            continue
        if (
            max(abs(a - b) for a, b in zip(_over(rgba, bg), bg, strict=True))
            < _INVISIBLE
        ):
            continue  # drawn in the color it sits on: no ink
        if area is None:
            area = _extent(artist, renderer, dpi, pad)
            if area < _INK_FLOOR:
                return out
        out.append(
            _Part(
                key,
                gid,
                form,
                _over(rgba, bg),
                tuple(rgba[:3]),
                float(rgba[3]),
                filled,
                z,
                sig,
                scope,
                artist,
            )
        )
    return out


def _resolver(keys):
    ordered = sorted(keys.items(), key=lambda item: -len(item[0]))

    def resolve(gid):
        if not gid:
            return None
        for prefix, key in ordered:
            if gid.startswith(prefix):
                return key
        return None

    return resolve


def _collect(fig, at, times, keys):
    """(time, [_Part]) per sample time."""
    resolve = _resolver(keys)
    samples = [None] if at is None else list(times)
    frames = []
    for t in samples:
        if at is not None:
            at(t)
        fig.canvas.draw()
        renderer = _renderer(fig)
        backgrounds = {}
        parts = []
        for artist, gid in _walk(fig, None, fig):
            if isinstance(artist, _IMAGES) or artist is fig:
                continue
            parts += _parts_of(artist, gid, resolve(gid), fig, renderer, backgrounds)
        frames.append((t, parts))
    return frames


def _shared(sharing):
    groups = [frozenset(g) for g in sharing]

    def together(a, b):
        return any(a in g and b in g for g in groups)

    return together


def _level(de, fault_de, warn_de):
    if de < fault_de:
        return 0
    if de < warn_de:
        return 1
    return 2


def _same_frame(frames, together, fault_de, warn_de):
    found = {}
    for t, parts in frames:
        by_key = {}
        for p in parts:
            if p.key is None or (p.form == "text" and _neutral(p.base)):
                continue
            by_key.setdefault(p.key, {}).setdefault((p.eff, p.sig), p)
        for k1, k2 in itertools.combinations(sorted(by_key), 2):
            if together(k1, k2):
                continue
            worst = None
            for a in by_key[k1].values():
                for b in by_key[k2].values():
                    de = _de(a.eff, b.eff)
                    level = _level(de, fault_de, warn_de)
                    if a.sig != b.sig:
                        level += 1
                    if level < 2 and (worst is None or (level, de) < worst[:2]):
                        worst = (level, de, a, b)
            if worst is not None:
                rec = found.setdefault((k1, k2), {"times": [], "worst": worst})
                rec["times"].append(t)
                rec["worst"] = min(rec["worst"], worst, key=lambda w: (w[0], w[1]))
    out = []
    for (k1, k2), rec in found.items():
        level, de, a, b = rec["worst"]
        shape = "" if a.sig == b.sig else f", but they differ in {_channel(a, b)}"
        detail = (
            f"{k1!r} and {k2!r} are drawn together in colors "
            f"{de:.1f} CAM02-UCS dE apart at the closest vision type{shape}"
        )
        out.append(
            MeaningFault(
                "same-frame",
                _LEVELS[level],
                (k1, k2),
                tuple(rec["times"]),
                detail,
                de,
                (a.artist, b.artist),
            )
        )
    return out


def _channel(a, b):
    names = ("marker", "line style", "hatch")
    diff = [n for n, x, y in zip(names, a.sig, b.sig, strict=True) if x != y]
    return " and ".join(diff) if diff else "form"


def _epochs_by_time(frames, rebind):
    """Hue-memory epoch of each frame: the number of rebind times at or before it."""
    cuts = sorted(rebind)
    return [sum(1 for c in cuts if _tnum(t) >= c) for t, _ in frames]


def _rebound(frames, together, epochs, fault_de, strict):
    # hue clusters, found greedily over every non-neutral color of the clip
    leaders = []
    seen = {}
    for _, parts in frames:
        for p in parts:
            if p.key is None or _neutral(p.base):
                continue
            if p.eff in seen:
                continue
            for i, lead in enumerate(leaders):
                if _de(p.eff, lead, vision=0) < fault_de:
                    seen[p.eff] = i
                    break
            else:
                leaders.append(p.eff)
                seen[p.eff] = len(leaders) - 1
    uses = {}  # (epoch, cluster) -> key -> list of (frame index, time)
    for i, (t, parts) in enumerate(frames):
        for p in parts:
            if p.key is None or p.eff not in seen:
                continue
            slot = uses.setdefault((epochs[i], seen[p.eff]), {})
            slot.setdefault(p.key, []).append((i, t))
    out = {}
    for (_, _cluster), by_key in uses.items():
        for k1, k2 in itertools.combinations(sorted(by_key), 2):
            f1, f2 = {i for i, _ in by_key[k1]}, {i for i, _ in by_key[k2]}
            t1, t2 = (
                [_tnum(t) for _, t in by_key[k1]],
                [_tnum(t) for _, t in by_key[k2]],
            )
            shared = together(k1, k2)
            if not shared and f1 & f2:
                continue  # on one frame together: same-frame reports it
            rec = out.setdefault((k1, k2), {"times": [], "detail": None})
            rec["times"] += sorted(set(t1) | set(t2))
            if rec["detail"] is None:
                rec["detail"] = _rebound_detail(k1, k2, t1, t2, shared)
    severity = "fault" if strict else "warning"
    return [
        MeaningFault(
            "rebound", severity, pair, tuple(sorted(set(rec["times"]))), rec["detail"]
        )
        for pair, rec in out.items()
    ]


def _tnum(t):
    return 0.0 if t is None else float(t)


def _span(times):
    return min(times), max(times)


def _rebound_detail(k1, k2, t1, t2, shared):
    a0, a1 = _span(t1)
    b0, b1 = _span(t2)
    gap = max(b0 - a1, a0 - b1, 0.0)
    why = (
        "declared never to share a clip, yet both appear"
        if shared
        else ("one hue serves two meanings")
    )
    return (
        f"{why}: {k1!r} visible {a0:g} to {a1:g}, {k2!r} visible {b0:g} to {b1:g}, "
        f"gap {gap:g}"
    )


def _cue_form(frames, cue_keys, fault_de):
    cues = set(cue_keys)
    if not cues:
        return []
    bad = {}  # (key, reason) -> times, artist
    for t, parts in frames:
        for p in parts:
            if p.key not in cues:
                continue
            reason = None
            if p.form in ("line", "marker", "text"):
                reason = f"drawn as a {p.form}"
            elif p.form in ("patch", "collection") and p.filled and p.alpha >= 1.0:
                others = [
                    q.zorder
                    for q in parts
                    if q.key is not None and q.key not in cues and q.scope == p.scope
                ]
                if others and p.zorder >= min(others):
                    reason = "an opaque filled patch above the subject"
            if reason:
                rec = bad.setdefault((p.key, reason), ([], p.artist))
                rec[0].append(t)
    out = [
        MeaningFault(
            "cue-form",
            "fault",
            (key,),
            tuple(times),
            f"cue {key!r} is {reason}; a cue is a ring, band or underlay",
            None,
            (artist,),
        )
        for (key, reason), (times, artist) in bad.items()
    ]
    cue_colors, other_colors = {}, {}
    for t, parts in frames:
        for p in parts:
            if p.key is None or _neutral(p.base):
                continue
            target = cue_colors if p.key in cues else other_colors
            target.setdefault(p.key, {}).setdefault(p.base, []).append(t)
    for ck, cc in cue_colors.items():
        for ok, oc in other_colors.items():
            hit = [
                (_de(a, b, vision=0), a, b)
                for a in cc
                for b in oc
                if _de(a, b, vision=0) < fault_de
            ]
            if hit:
                de = min(hit)[0]
                times = sorted({t for color in cc for t in cc[color]})
                out.append(
                    MeaningFault(
                        "cue-form",
                        "fault",
                        (ck, ok),
                        tuple(times),
                        f"cue {ck!r} takes the hue of {ok!r} "
                        f"({de:.1f} dE); the cue must not use a hue a key uses",
                        de,
                    )
                )
    return out


def _unkeyed(frames, strict):
    found = {}
    for t, parts in frames:
        for p in parts:
            if p.key is not None or _neutral(p.base):
                continue
            name = p.gid or f"<{type(p.artist).__name__}>"
            found.setdefault(name, ([], p.artist))[0].append(t)
    severity = "fault" if strict else "advisory"
    return [
        MeaningFault(
            "unkeyed",
            severity,
            (name,),
            tuple(times),
            f"colored artist {name!r} carries no meaning key",
            None,
            (artist,),
        )
        for name, (times, artist) in found.items()
    ]


def _apply_waivers(faults, waive):
    waive = dict(waive or {})
    out = []
    for f in faults:
        reason = None
        names = [f.ident]
        if len(f.keys) == 2:
            names.append(f"{f.kind}:{'+'.join(reversed(sorted(f.keys)))}")
            names += [f"{f.kind}:{'+'.join(f.keys)}"]
        for name in names:
            if name in waive:
                reason = waive[name]
                break
        out.append(replace(f, waived=reason) if reason is not None else f)
    return out


def _order(faults):
    rank = {kind: i for i, kind in enumerate(MEANING_KINDS)}
    return sorted(
        faults,
        key=lambda f: (
            rank[f.kind],
            f.severity != "fault",
            min((t for t in f.times if t is not None), default=-1.0),
            f.keys,
        ),
    )


def _analyze(
    frames, kinds, *, sharing, cue_keys, waive, epochs, fault_de, warn_de, strict
):
    together = _shared(sharing)
    found = []
    if "same-frame" in kinds:
        found += _same_frame(frames, together, fault_de, warn_de)
    if "rebound" in kinds:
        found += _rebound(frames, together, epochs, fault_de, strict)
    if "cue-form" in kinds:
        found += _cue_form(frames, cue_keys, fault_de)
    if "unkeyed" in kinds:
        found += _unkeyed(frames, strict)
    found = [replace(f, times=tuple(dict.fromkeys(f.times))) for f in found]
    return _order(_apply_waivers(found, waive))


def meaning_faults(
    fig,
    at=None,
    times=None,
    *,
    keys,
    sharing=(),
    cue_keys=(),
    waive=None,
    rebind=(),
    fault_de=10.0,
    warn_de=20.0,
    strict=False,
):
    """Check a figure, or a clip sampled at chosen times, for colors with two meanings.

    The figure is drawn at each sample time and the checks run on the colors
    that are drawn, after artist opacity and any `Stage` dimming. An artist
    counts as drawn when its opacity is at least 0.05 and its extent covers a
    few pixels, the rule `readability_faults` uses.

    Args:
        fig: The figure to check.
        at: Callable ``at(t)`` that poses the figure at time `t`; None checks
            the figure as drawn.
        times: Sample times passed to `at`; give them with `at`. Include a
            time just after each cue, since the uniform frames alone miss it.
        keys: Mapping from an artist ``gid`` prefix to the meaning key it
            carries. The longest matching prefix wins.
        sharing: Iterable of key groups that may share a hue because they are
            never in the same clip. A group is exempt from ``"same-frame"``; if
            two of its keys do appear in the clip, ``"rebound"`` reports it.
        cue_keys: Keys that are cues. A cue must be a ring, band or underlay.
        waive: Mapping ``"<kind>:<key>"`` or ``"<kind>:<key>+<key>"`` to a
            reason (who, when). A waived finding stays in the result with its
            reason in `MeaningFault.waived`.
        rebind: Times at which hue memory is released, such as a scene cut. A
            hue may mean a new thing after one.
        fault_de: CAM02-UCS distance under which two colors look alike.
        warn_de: Distance under which two colors are close enough to warn.
        strict: True makes ``"unkeyed"`` artists and ``"rebound"`` hues faults;
            otherwise they are advisory and warning.

    Returns:
        A list of `MeaningFault`, ordered by kind, then severity, then time.

    Raises:
        ValueError: If `at` and `times` are not given together.
        ImportError: If `colorspacious` is not installed.

    Example::

        faults = ept.meaning_faults(
            fig, at=pose, times=[0.0, 2.0], keys={"orbit": "planet"}
        )
    """
    if (at is None) != (times is None):
        raise ValueError("give `at` and `times` together, or neither")
    _vision()
    frames = _collect(fig, at, times, keys)
    return _analyze(
        frames,
        MEANING_KINDS,
        sharing=sharing,
        cue_keys=cue_keys,
        waive=waive,
        epochs=_epochs_by_time(frames, rebind),
        fault_de=fault_de,
        warn_de=warn_de,
        strict=strict,
    )


def meaning_faults_chain(
    clips,
    *,
    sharing=(),
    waive=None,
    rebind=(),
    fault_de=10.0,
    warn_de=20.0,
    strict=False,
):
    """Check hue memory across clips played in order.

    Each clip is sampled as for `meaning_faults`, then shifted so it starts where
    the previous one ended, and the ``"rebound"`` check runs over all of them
    together. The per-frame kinds belong to each clip's own call.

    Args:
        clips: Sequence of ``(fig, at, times, keys)``, in playing order. A clip
            with no `at` is one still of zero length.
        sharing: As for `meaning_faults`.
        waive: As for `meaning_faults`.
        rebind: Indices of clips that open a new hue memory: a hue may mean a
            new thing from the start of each of these clips (a scene cut).
        fault_de: As for `meaning_faults`.
        warn_de: As for `meaning_faults`.
        strict: As for `meaning_faults`.

    Returns:
        A list of ``"rebound"`` `MeaningFault`, with times on the chain's clock.
    """
    _vision()
    merged, epochs = [], []
    start = 0.0
    cuts = sorted(set(rebind))
    for index, (fig, at, times, keys) in enumerate(clips):
        frames = _collect(fig, at, times, keys)
        stamps = [t for t, _ in frames if t is not None]
        t0 = min(stamps) if stamps else 0.0
        length = (max(stamps) - t0) if stamps else 0.0
        merged += [
            (start + ((t - t0) if t is not None else 0.0), parts) for t, parts in frames
        ]
        epochs += [sum(1 for c in cuts if index >= c)] * len(frames)
        start += length
    return _analyze(
        merged,
        ("rebound",),
        sharing=sharing,
        cue_keys=(),
        waive=waive,
        epochs=epochs,
        fault_de=fault_de,
        warn_de=warn_de,
        strict=strict,
    )


def meaning_report(faults):
    """The findings as text, waived ones marked, never dropped.

    Args:
        faults: A list of `MeaningFault`.

    Returns:
        A multi-line report. With no unwaived fault or warning it says "no
        tier 1 or 2 findings" and that the choice of meanings stays with the
        reviewer; it never says "clean".
    """
    faults = list(faults)
    active = [f for f in faults if f.waived is None and f.severity != "advisory"]
    lines = []
    if active:
        n = sum(f.severity == "fault" for f in active)
        lines.append(f"{len(active)} meaning finding(s), {n} fault(s):")
    else:
        lines.append(
            "no tier 1 or 2 findings (whether the keys are the right meanings "
            "is a tier 3 judgment for the reviewer)"
        )
    lines += [f"  {f}" for f in faults]
    return "\n".join(lines)


def assert_meaning(fig, at=None, times=None, **kwargs):
    """Raise `MeaningError` if the figure or clip has an unwaived fault.

    Takes the arguments of `meaning_faults`, with ``strict`` on unless given:
    the check is advisory when a clip is saved and strict in its test.

    Args:
        fig: The figure to check.
        at: Callable that poses the figure, as for `meaning_faults`.
        times: Sample times, as for `meaning_faults`.
        **kwargs: Passed to `meaning_faults`; ``keys`` is required.

    Returns:
        The list of `MeaningFault` when none is an unwaived fault: warnings and
        waived findings are returned, not raised.

    Raises:
        MeaningError: If any finding is an unwaived fault; its message is the
            full report.
    """
    kwargs.setdefault("strict", True)
    found = meaning_faults(fig, at, times, **kwargs)
    if any(f.severity == "fault" and f.waived is None for f in found):
        raise MeaningError(found)
    return found
