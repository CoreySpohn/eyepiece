"""Equations as glyph outlines: `equation` lays one out, `transform` moves it.

`term_row` sets one text per term, which is the right tool when the text of
a readout changes but its structure does not. `equation` is for fixed math
that has structure: fractions, scripts, a stretching bar. It reads the
layout matplotlib's mathtext computes, so every glyph is an outline in its
own place, and groups the glyphs by name with a marker in the string,
``{{name:...}}``. Each group is an artist, so the moves that take an artist
(`Anchor.from_artist`, `callout`, `indicate`, `circumscribe`, `Stage`)
take a group unchanged.

`transform` carries one equation into another. Glyphs that are the same
character in both move from one place to the other; glyphs that match
nothing fade out where they stand or fade in where they will stand; a
fraction bar changes its length through the same arc-length sampling as
`morph`.
"""

import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import LoadFlags
from matplotlib.mathtext import MathTextParser
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.textpath import text_to_path
from matplotlib.transforms import Bbox, IdentityTransform

from eyepiece._motion import _apply_ease as _ease
from eyepiece._motion import _Unsnap
from eyepiece._motion import stagger as _stagger
from eyepiece._result import PlotResult
from eyepiece.attention import (
    Anchor,
    _as_anchor,
    _check_u,
    _morph_pairs,
    _path_parts,
    _pixels,
)

# Mathtext is parsed at this size and the outlines scaled down, as TextPath
# does, so the rounding of rule positions does not depend on the font size.
_PARSE_PT = 100.0

# A piece that the phantom parse moves by more than this (in parse-size
# units) counts as a layout change.
_PHANTOM_TOL = 0.05

_RULE_POINTS = 64

_OPEN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*):")
_DOLLAR = re.compile(r"(?<!\\)\$")


# Tags


def _inner_close(s, i):
    """Index of the brace that closes a marker's inner group, searching from `i`."""
    depth = 2
    while i < len(s):
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        if depth == 1:
            return i
        i += 1
    raise ValueError("unclosed {{name:...}} marker")


def _math_body(tex):
    """The text between the dollars of a one-span math string."""
    s = tex.strip()
    marks = _DOLLAR.findall(s)
    if not marks:
        body = s
    elif len(marks) == 2 and s.startswith("$") and s.endswith("$") and len(s) > 2:
        body = s[1:-1]
    else:
        raise ValueError(
            f"equation takes one math span ($...$) and no text outside it; got {tex!r}"
        )
    if re.search(r"\\[{}$]", body):
        raise ValueError(
            "escaped braces and dollars (\\{, \\}, \\$) are not supported in "
            f"an equation: {tex!r}"
        )
    return body


def _strip_tags(body):
    r"""Remove ``{{name:...}}`` markers; return the plain TeX and the tag spans.

    A marker is one TeX group, so ``\frac{{num:A}}{{den:B}}`` becomes
    ``\frac{A}{B}``. A span is ``(name, start, end)`` of the content in the
    stripped string, innermost first.
    """
    out, spans = [], []

    def walk(s):
        i = 0
        while i < len(s):
            m = _OPEN.match(s, i)
            if not m:
                out.append(s[i])
                i += 1
                continue
            j = _inner_close(s, m.end())
            if s[j + 1 : j + 2] != "}":
                raise ValueError(
                    f"marker {m.group(1)!r} must be one balanced {{{{name:...}}}} group"
                )
            out.append("{")
            start = len(out)
            walk(s[m.end() : j])
            spans.append((m.group(1), start, len(out)))
            out.append("}")
            i = j + 2

    walk(body)
    names = [name for name, _, _ in spans]
    dup = sorted(n for n, k in Counter(names).items() if k > 1)
    if dup:
        raise ValueError(f"tag names must be unique; repeated: {dup}")
    return "".join(out), spans


# Glyphs


class _Piece:
    """One glyph outline or rule, in points from the baseline at the left."""

    def __init__(self, key, local, codes):
        self.key = key
        self.local = local
        self.codes = codes
        self.tag = None
        self.box = np.array([*local.min(axis=0), *local.max(axis=0)])
        self.center = 0.5 * (self.box[:2] + self.box[2:])

    @property
    def is_rule(self):
        return self.key[0] == "rule"

    def signature(self):
        return self.key, self.center


def _parse(plain, fontsize):
    """Pieces of one mathtext string at `fontsize` points, untagged, and its width."""
    prop = FontProperties(size=_PARSE_PT)
    vp = MathTextParser("path").parse(f"${plain}$", 72, prop)
    shrink = fontsize / _PARSE_PT
    pieces = []
    for font, fs, _code, index, ox, oy in vp.glyphs:
        font.clear()
        font.set_size(text_to_path.FONT_SCALE, text_to_path.DPI)
        font.load_glyph(index, flags=LoadFlags.NO_HINTING)
        verts, codes = font.get_path()
        if len(verts) == 0:
            continue  # a space has no outline
        local = (np.asarray(verts) * (fs / text_to_path.FONT_SCALE) + [ox, oy]) * shrink
        pieces.append(_Piece(("g", font.postscript_name, index), local, codes))
    for x, y, w, h in vp.rects:
        box = Path.unit_rectangle()
        local = (box.vertices * [w, h] + [x, y]) * shrink
        pieces.append(_Piece(("rule",), local, box.codes))
    return pieces, float(vp.width) * shrink


def _tagged_pieces(body, fontsize):
    r"""The pieces of an equation, each tagged with the group it belongs to.

    A tag is found by difference: the string is parsed again with the
    group's content wrapped in ``\phantom``, which keeps the layout and draws
    nothing, and the pieces that disappear belong to the group. A piece that
    stays but moves means the phantom changed the layout, which raises.
    """
    plain, spans = _strip_tags(body)
    pieces, width = _parse(plain, fontsize)
    tol = _PHANTOM_TOL * fontsize / _PARSE_PT
    for name, a, b in spans:
        wrapped = plain[:a] + r"\phantom{" + plain[a:b] + "}" + plain[b:]
        rest, _ = _parse(wrapped, fontsize)
        left = list(rest)
        for piece in pieces:
            near = [
                (float(np.hypot(*(q.center - piece.center))), k)
                for k, q in enumerate(left)
                if q.key == piece.key
            ]
            near = [t for t in near if t[0] <= tol]
            if near:
                left.pop(min(near)[1])
            elif piece.tag is None:
                piece.tag = name
        if left:
            raise ValueError(
                f"tag {name!r} changed the layout of {len(left)} pieces; "
                "wrap a whole term, fraction or group instead"
            )
    if not spans:
        for piece in pieces:
            piece.tag = "all"
        spans = [("all", 0, 0)]
    order = [name for name, _, _ in spans]
    present = {p.tag for p in pieces}
    empty = [n for n in order if n not in present]
    if empty:
        raise ValueError(f"tag {empty[0]!r} holds no glyph")
    if not pieces:
        raise ValueError("equation has no glyphs")
    return pieces, width, order


# Artists


class _Eq:
    """The shared layout of one equation: pieces, placement, colors."""

    def __init__(self, anchor, ha, fontsize, pieces, width, tags, colors, zorder):
        self.anchor = anchor
        self.ha = ha
        self.fontsize = fontsize
        self.pieces = pieces
        self.width = width
        self.tags = tags
        self.colors = colors
        self.zorder = zorder
        self.glyphs = []
        self.terms = {}

    def scale(self, renderer):
        return _pixels(renderer, self.anchor.figure, 1.0)

    def origin(self, renderer):
        x, y = self.anchor.display(renderer)
        frac = {"left": 0.0, "center": 0.5, "right": 1.0}[self.ha]
        return np.array([x - frac * self.width * self.scale(renderer), y])

    def place(self, local, renderer):
        return local * self.scale(renderer) + self.origin(renderer)


class _Glyph(PathPatch):
    """One piece of an equation, drawn where the equation (or a transform) says."""

    def __init__(self, eq, piece, rgb):
        super().__init__(
            Path(np.zeros((1, 2)), [Path.MOVETO]),
            transform=IdentityTransform(),
            facecolor=rgb,
            edgecolor="none",
            linewidth=0.0,
            zorder=eq.zorder,
        )
        self.eq = eq
        self.piece = piece
        self.rgb = rgb
        self.pose = None
        self.mixed = False
        self.set_clip_on(False)
        self.set_in_layout(False)

    def vertices(self, renderer):
        """The piece in display pixels, in its place or part way to its partner."""
        own = self.eq.place(self.piece.local, renderer)
        pose = self.pose
        if pose is None:
            return own
        partner = pose.partner
        v = (1.0 - pose.e) * own + pose.e * partner.eq.place(
            partner.piece.local, renderer
        )
        v[:, 1] += (
            pose.lift_em
            * self.eq.fontsize
            * self.eq.scale(renderer)
            * math.sin(math.pi * pose.e)
        )
        return v

    def rule_path(self, renderer):
        pose = self.pose
        eq, other = self.eq, pose.partner.eq
        vertices, codes = [], []
        for pa, pb, closed in pose.rule:
            v = (1.0 - pose.e) * eq.place(pa, renderer) + pose.e * other.place(
                pb, renderer
            )
            v[:, 1] += (
                pose.lift_em
                * eq.fontsize
                * eq.scale(renderer)
                * math.sin(math.pi * pose.e)
            )
            vertices += [v, v[:1]] if closed else [v]
            codes += [Path.MOVETO, *[Path.LINETO] * (len(v) - 1)]
            if closed:
                codes.append(Path.CLOSEPOLY)
        return Path(np.vstack(vertices), codes)

    def draw(self, renderer):
        """Put the piece where its equation (or a transform) says, then draw."""
        if not self.get_visible():
            return
        if self.pose is not None and self.pose.rule is not None:
            self.set_path(self.rule_path(renderer))
        else:
            self.set_path(Path(self.vertices(renderer), self.piece.codes))
        super().draw(renderer)


@dataclass
class _Pose:
    """A glyph part way to its partner in the other equation."""

    partner: object
    e: float
    lift_em: float
    rule: object = None


class _Term(PathPatch):
    """One tag of an equation: draws nothing, stands for its glyphs.

    The glyphs are the term's `_pieces`, so a pulse or a `Stage` acts on them,
    and the term's box is the union of their boxes where the equation is laid
    out, which is what an anchor, a callout and an outline read.
    """

    def __init__(self, eq, name, glyphs):
        super().__init__(
            Path(np.zeros((1, 2)), [Path.MOVETO]),
            transform=IdentityTransform(),
            fill=False,
            linewidth=0.0,
            zorder=eq.zorder,
        )
        self.eq = eq
        self.name = name
        self._pieces = glyphs
        self.set_clip_on(False)
        self.set_in_layout(False)

    def get_window_extent(self, renderer=None, dpi=None):
        """The box of the term's glyphs where the equation is laid out."""
        boxes = np.array([g.piece.box for g in self._pieces])
        local = np.array(
            [
                [boxes[:, 0].min(), boxes[:, 1].min()],
                [boxes[:, 2].max(), boxes[:, 3].max()],
            ]
        )
        (x0, y0), (x1, y1) = self.eq.place(local, renderer)
        return Bbox([[x0, y0], [x1, y1]])

    def draw(self, renderer):
        """Draw nothing; the glyphs draw themselves."""

    def __repr__(self):
        return f"<Term {self.name!r} of {len(self._pieces)} glyphs>"


@dataclass(frozen=True, eq=False)
class _EquationResult(PlotResult):
    """A `PlotResult` of `equation`, which also holds ``term(name)``."""

    term: object = None
    layout: object = None


def _set_alpha(glyph, factor):
    glyph.set_alpha(None if factor >= 1.0 else max(0.0, factor))
    glyph.set_visible(factor > 0.0)


def equation(
    at,
    tex,
    *,
    fontsize,
    colors=None,
    ha="center",
    zorder=10,
):
    r"""Lay math out as glyph outlines grouped into named terms.

    The string is one mathtext span (``$...$``) with optional markers
    ``{{name:...}}`` around the pieces that should be addressable. A marker is
    read as one TeX group, so ``\frac{{num:A}}{{den:B}}`` is the fraction
    ``A / B`` with ``A`` called ``num``. Markers may nest (the inner one wins
    its glyphs) and need unique names. A string with no markers is one term,
    ``"all"``. A term is found by difference, not by counting: the string is
    parsed again with the term wrapped in ``\phantom`` and the glyphs that
    vanish are the term's, so fractions, scripts and rules are handled, and a
    layout that the phantom would change raises rather than drawing wrong.

    Each term is an artist (``result.artists["terms"][name]``, or
    ``result.term(name)``) with the box of its glyphs, so it is a target for
    `Anchor.from_artist`, `callout`, `indicate`, `circumscribe` and `Stage`;
    an equation never has pulse, outline or caption moves of its own. The
    glyphs are figure-level patches in display pixels, like the terms of
    `term_row`, so they never join an axes' autoscaling. A term's box is its
    laid-out place and does not follow a glyph that a `transform` is moving.

    Which tool lays out math: `equation` is for fixed math with structure and
    for derivations that `transform` rearranges; `term_row` is for a running
    readout whose text changes while its structure does not.

    ``update(u, level=1.0)`` writes the equation on, left to right: each term
    (and each glyph that belongs to no term) arrives in turn, and `level` is
    an opacity over all of it, never geometry. At ``u = 1``, the rest state,
    the equation is drawn whole.

    Not supported: ``usetex`` (mathtext only), several math spans or text
    outside the span in one string, escaped braces (``\{`` and ``\}``, which
    raise), and the size animation of ``\left ... \right``.

    Args:
        at: Where the equation sits: an `Anchor`, an artist, or an Axes (its
            center). The anchor's point is the baseline's left end, center,
            or right end by `ha`.
        tex: The math string.
        fontsize: Font size in points.
        colors: Mapping of term name to color. A glyph outside every named
            term, or a term not listed, takes ``rcParams["text.color"]``. A
            name that is no term of this equation is ignored, so one mapping
            can serve several equations.
        ha: Which end of the equation `at` marks: ``"left"``, ``"center"``
            or ``"right"``.
        zorder: Drawing order among the figure's children.

    Returns:
        A `PlotResult` on the anchor's axes (None for an anchor drawn on the
        figure), with ``artists["terms"]`` (name to term artist, in source
        order), ``artists["glyphs"]`` (every glyph and rule in layout order),
        ``term(name)`` and ``update(u, level=1.0)``.

    Raises:
        TypeError: If `at` is not an Anchor, an artist or an Axes.
        ValueError: If the string is not one math span, holds an escaped
            brace, a marker is malformed or repeated, a tag holds no glyph,
            a phantom changes the layout, `ha` is
            unknown, or `fontsize` is not positive.

    Example::

        eq = ep.equation(ax, r"${{lhs:y=}}{{rhs:m x + b}}$", fontsize=40,
                         colors={"lhs": "C0", "rhs": "C1"})
        ep.indicate(eq.term("rhs"))  # a term is an artist
        eq.term("nope")  # KeyError naming the known terms
    """
    if ha not in ("left", "center", "right"):
        raise ValueError(f"unknown ha: {ha!r}; known: ['left', 'center', 'right']")
    fontsize = float(fontsize)
    if not fontsize > 0.0:
        raise ValueError(f"fontsize must be positive, got {fontsize}")
    anchor = Anchor.from_artist(at) if isinstance(at, Axes) else _as_anchor(at, "at")
    pieces, width, order = _tagged_pieces(_math_body(tex), fontsize)
    colors = dict(colors or {})
    default = to_rgb(matplotlib.rcParams["text.color"])
    eq = _Eq(anchor, ha, fontsize, pieces, width, order, colors, zorder)
    figure = anchor.figure
    for piece in sorted(pieces, key=lambda p: (p.center[0], p.center[1])):
        rgb = to_rgb(colors[piece.tag]) if piece.tag in colors else default
        glyph = _Glyph(eq, piece, rgb)
        figure.add_artist(glyph)
        eq.glyphs.append(glyph)
    for name in order:
        term = _Term(eq, name, [g for g in eq.glyphs if g.piece.tag == name])
        figure.add_artist(term)
        eq.terms[name] = term

    # Write-on order: each term, and each glyph in no term, by its left edge.
    items = [
        (t._pieces, min(g.piece.box[0] for g in t._pieces)) for t in eq.terms.values()
    ]
    items += [([g], g.piece.box[0]) for g in eq.glyphs if g.piece.tag is None]
    items.sort(key=lambda it: it[1])

    def update(u, level=1.0):
        u = _check_u(u)
        level = min(1.0, max(0.0, float(level)))
        n = len(items)
        for i, (members, _) in enumerate(items):
            r = min(1.0, max(0.0, n * u - i))
            shaped = 1.0 if r >= 1.0 else min(1.0, max(0.0, _ease(r, "smoothstep")))
            for glyph in members:
                _set_alpha(glyph, shaped * level)

    def term(name):
        try:
            return eq.terms[name]
        except KeyError:
            raise KeyError(f"no term {name!r}; terms are {list(eq.terms)}") from None

    return _EquationResult(
        ax=anchor.axes,
        artists={"terms": eq.terms, "glyphs": eq.glyphs},
        update=update,
        term=term,
        layout=eq,
    )


# Transform


def _warn_unpaired_tags(eq, left, other_left, other_tags, mapped):
    """Warn about a tag with no partner whose left-over glyphs `other` also has.

    `left` and `other_left` are the unpaired glyphs of `eq` and the other
    equation. A tag that merely adds new content (its glyphs match nothing
    left over) stays silent.
    """
    wanted = {g.piece.key for g in other_left}
    for tag in eq.tags:
        if tag == "all" or tag in mapped or tag in other_tags:
            continue
        keys = {g.piece.key for g in left if g.piece.tag == tag}
        if keys & wanted:
            warnings.warn(
                f"tag {tag!r} has no partner in the other equation, so its "
                "glyphs fade instead of moving; name the partner in key_map "
                "or give both equations the same tag",
                UserWarning,
                stacklevel=4,
            )


def _match(eq_a, eq_b, key_map):
    """Pair glyphs of two equations; return pairs and the unmatched of each.

    Pairs are indices into each equation's ``glyphs``. Inside a tag the two
    equations share (or a pair named in `key_map`), equal glyphs pair in
    left-to-right order; glyphs in no tag then pair across the equations. A
    tagged glyph never pairs outside its tag. Equations that share no tag name
    and have no `key_map` pair every glyph by order, ignoring tags, so a
    tagged equation can move into an untagged one.
    """
    used_a, used_b, pairs = set(), set(), []

    def ordered(eq, tag, used, anytag=False):
        idx = [
            i
            for i, g in enumerate(eq.glyphs)
            if (anytag or g.piece.tag == tag) and i not in used
        ]
        return sorted(idx, key=lambda i: tuple(eq.glyphs[i].piece.center))

    def pair_up(ia, ib):
        for i in ia:
            for j in ib:
                if (
                    j not in used_b
                    and eq_a.glyphs[i].piece.key == eq_b.glyphs[j].piece.key
                ):
                    pairs.append((i, j))
                    used_a.add(i)
                    used_b.add(j)
                    break

    mapped = dict(key_map)
    if not mapped and not set(eq_a.tags) & set(eq_b.tags):
        pair_up(ordered(eq_a, None, used_a, True), ordered(eq_b, None, used_b, True))
        gone = [i for i in range(len(eq_a.glyphs)) if i not in used_a]
        new = [j for j in range(len(eq_b.glyphs)) if j not in used_b]
        return pairs, gone, new
    for ta, tb in [*mapped.items(), *((t, t) for t in eq_a.tags if t not in mapped)]:
        if tb in eq_b.tags and ta in eq_a.tags:
            pair_up(ordered(eq_a, ta, used_a), ordered(eq_b, tb, used_b))
    pair_up(ordered(eq_a, None, used_a), ordered(eq_b, None, used_b))
    gone = [i for i in range(len(eq_a.glyphs)) if i not in used_a]
    new = [j for j in range(len(eq_b.glyphs)) if j not in used_b]
    gone_g = [eq_a.glyphs[i] for i in gone]
    new_g = [eq_b.glyphs[j] for j in new]
    _warn_unpaired_tags(eq_a, gone_g, new_g, eq_b.tags, mapped)
    _warn_unpaired_tags(
        eq_b, new_g, gone_g, eq_a.tags, {v: k for k, v in mapped.items()}
    )
    return pairs, gone, new


@dataclass(frozen=True, eq=False)
class _TransformResult(PlotResult):
    """A `PlotResult` of `transform`, which also reports how the glyphs paired."""

    pairs: tuple = ()
    gone: tuple = ()
    new: tuple = ()


def transform(
    eq_a,
    eq_b,
    *,
    stagger=0.55,
    arc=0.0,
    ease="smoothstep",
    key_map=None,
    fade=0.4,
):
    """Carry one equation into another, glyph by glyph.

    Glyphs that are the same character in both equations (the same font and
    glyph, so a bar matches a bar and an ``E`` an ``E``) move from their place
    in `eq_a` to their place in `eq_b`. They pair inside the tags the two
    equations share, in left-to-right order, and then across the glyphs that
    belong to no tag; a tagged glyph never pairs outside its tag, so a writer
    who tagged ``num`` meant it. `key_map` pairs two differently named tags,
    `eq_a`'s name to `eq_b`'s. What matches nothing fades out where it stands
    (from `eq_a`) or fades in where it will stand (from `eq_b`). A fraction
    bar changes length by the same arc-length sampling as `morph`. Each
    matched glyph runs its own eased progress, staggered from left to right
    by its place in `eq_a`, so the left end moves first. The fades take the
    first and last `fade` of the beat and overlap the start and end of the
    move; the matched glyphs carry the eye.

    The move owns the visibility of both equations: at ``u = 0`` `eq_a` is
    drawn and `eq_b` is hidden, at ``u = 1`` `eq_b` is drawn and `eq_a` is
    hidden, and in between the pieces of both are drawn. The rest state, the
    state the move is made in, is ``u = 1``: the figure as drawn shows `eq_b`.
    A still that wants `eq_a` calls ``update(0)``. Do not call the equations'
    own ``update`` while the transform runs. While `u` is inside (0, 1) the
    move sets the facecolor and alpha of the glyphs it moves, so do not dim a
    term with a `Stage` then; the stage would be overwritten or would fight
    the move. Use `level`. `level` is an opacity over
    everything the move draws and never geometry, so a `Stage` can dim it.
    Pass the uneased beat progress, as for `stagger`; the easing lives in
    the move.

    A narration can start the move with a ``{transform:name}`` cue.

    Args:
        eq_a: The `equation` result to start from.
        eq_b: The `equation` result to end on.
        stagger: Fraction of the beat each matched glyph takes, in (0, 1];
            1 moves every glyph together.
        arc: How far a matched glyph rises on the way, in font sizes (a sine
            bump, up on the screen); 0 moves along a straight line.
        ease: An easing kind for `eyepiece.ease`, or a callable from progress
            to eased progress, applied to each glyph's progress.
        key_map: Mapping of a tag name of `eq_a` to a tag name of `eq_b` whose
            glyphs pair.
        fade: Fraction of the beat spent fading out unmatched glyphs (first)
            and fading in unmatched glyphs (last), in (0, 1].

    Returns:
        A `PlotResult` on `eq_a`'s axes with ``artists["glyphs"]`` (the glyphs
        of both equations), ``update(u, level=1.0)``, and ``pairs``, ``gone``
        and ``new``: the matched index pairs ``(i, j)`` into the two
        equations' ``artists["glyphs"]``, the unmatched indices of `eq_a`, and
        the unmatched indices of `eq_b`. Every glyph of each equation is in
        exactly one of these.

    Raises:
        TypeError: If an end is not the result of `equation`.
        ValueError: If `stagger` or `fade` is outside (0, 1], the equations
            are in different figures, or `arc` is not finite.
        KeyError: If a `key_map` name is not a tag of its equation.

    Example::

        a = ep.equation(ax, r"${{l:a^2+b^2}}=c^2$", fontsize=48)
        b = ep.equation(ax, r"${{l:a^2}}=c^2-b^2$", fontsize=48)
        move = ep.transform(a, b)
        move.update(0.5)
    """
    for end in (eq_a, eq_b):
        if not isinstance(end, _EquationResult):
            raise TypeError(
                f"transform takes results of equation, got {type(end).__name__}"
            )
    ea, eb = eq_a.layout, eq_b.layout
    if ea.anchor.figure is not eb.anchor.figure:
        raise ValueError("transform needs both equations in one figure")
    stagger = float(stagger)
    fade = float(fade)
    arc = float(arc)
    if not 0.0 < stagger <= 1.0:
        raise ValueError(f"stagger must be in (0, 1], got {stagger}")
    if not 0.0 < fade <= 1.0:
        raise ValueError(f"fade must be in (0, 1], got {fade}")
    if not math.isfinite(arc):
        raise ValueError(f"arc must be finite, got {arc}")
    key_map = dict(key_map or {})
    for ta, tb in key_map.items():
        if ta not in ea.tags:
            raise KeyError(f"no term {ta!r} in the first equation; terms are {ea.tags}")
        if tb not in eb.tags:
            raise KeyError(
                f"no term {tb!r} in the second equation; terms are {eb.tags}"
            )

    pairs, gone, new = _match(ea, eb, key_map)
    xs = np.array([ea.glyphs[i].piece.center[0] for i, _ in pairs])
    span = float(np.ptp(xs)) if len(xs) else 0.0
    ranks = (xs - xs.min()) / span if span > 0.0 else np.zeros(len(xs))
    rules = {}
    for i, j in pairs:
        pa, pb = ea.glyphs[i].piece, eb.glyphs[j].piece
        if pa.is_rule:
            rules[i] = _morph_pairs(
                _path_parts(Path(pa.local, pa.codes)),
                _path_parts(Path(pb.local, pb.codes)),
                _RULE_POINTS,
                True,
            )
    unsnap = _Unsnap([*ea.glyphs, *eb.glyphs])

    def update(u, level=1.0):
        u = _check_u(u)
        level = min(1.0, max(0.0, float(level)))
        unsnap.moving(0.0 < u < 1.0)
        for g in ea.glyphs:
            g.pose = None
            if g.mixed:
                g.set_facecolor(g.rgb)
                g.mixed = False
        for g in eb.glyphs:
            g.pose = None
        if u <= 0.0 or u >= 1.0:
            shown, hidden = (ea, eb) if u <= 0.0 else (eb, ea)
            for g in shown.glyphs:
                _set_alpha(g, level)
            for g in hidden.glyphs:
                g.set_visible(False)
            return
        for g in eb.glyphs:
            g.set_visible(False)
        for rank, (i, j) in zip(ranks, pairs, strict=True):
            a, b = ea.glyphs[i], eb.glyphs[j]
            e = float(_ease(float(_stagger(u, rank, span=stagger)), ease))
            a.pose = _Pose(b, e, arc, rules.get(i))
            mix = min(1.0, max(0.0, e))
            a.set_facecolor(
                tuple((1.0 - mix) * np.array(a.rgb) + mix * np.array(b.rgb))
            )
            a.mixed = True
            _set_alpha(a, level)
        out = 1.0 - min(1.0, max(0.0, _ease(min(1.0, u / fade), "smoothstep")))
        for i in gone:
            _set_alpha(ea.glyphs[i], out * level)
        arrive = min(
            1.0,
            max(
                0.0, _ease(min(1.0, max(0.0, (u - (1.0 - fade)) / fade)), "smoothstep")
            ),
        )
        for j in new:
            _set_alpha(eb.glyphs[j], arrive * level)

    update(1.0)
    return _TransformResult(
        ax=eq_a.ax,
        artists={"glyphs": [*ea.glyphs, *eb.glyphs]},
        update=update,
        pairs=tuple(pairs),
        gone=tuple(gone),
        new=tuple(new),
    )
