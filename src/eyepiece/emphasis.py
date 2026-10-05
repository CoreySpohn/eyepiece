"""Fade drawn artists, collect what a block drew, and narrate steps.

A figure that builds toward a whole in steps carries the elements a reader
has already seen into the next step, and shows them as "already seen": still
present, still in place, but quieter than what is new. `fade` does that by
blending every color an artist draws with toward the background it sits on,
rather than lowering its opacity. A blended element never lets what is
under it show through, and because the target is the actual background a
faded element reads correctly on a light page and a dark one alike.

`capture` is the other half: a context manager that collects the artists a
block of drawing code added to an axes, so a caller can fade exactly that
group later without tagging each artist by hand. `blend` is the color blend
itself, for a single color rather than a drawn artist.

`step_list` is the narration beside such a build-up: a numbered list whose
current step is bright, whose finished steps are dim, and whose later steps
are not yet shown.
"""

import contextlib

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.figure import FigureBase
from matplotlib.image import AxesImage, BboxImage, FigureImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from eyepiece import _style
from eyepiece._result import PlotResult

_IMAGES = (AxesImage, BboxImage, FigureImage)


class Faded:
    """Handle returned by `fade`: the artists it changed and how to undo it.

    Attributes:
        artists: The leaf artists whose colors (or, for images and under
            `by="alpha"`, opacity) `fade` changed, in the order it visited
            them.
    """

    def __init__(self):
        """Start with nothing changed."""
        self.artists = []
        self._undo = []
        self._restored = False

    def _record(self, artist, setter, value):
        self._undo.append((setter, value))
        if not self.artists or self.artists[-1] is not artist:
            self.artists.append(artist)

    def restore(self):
        """Put every changed color back as it was before the fade.

        Restoring twice is a no-op. When the same artists were faded more
        than once, restore the handles in the reverse order of the fades.
        """
        if self._restored:
            return
        for setter, value in reversed(self._undo):
            setter(value)
        self._restored = True


def _check_level(level):
    level = float(level)
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"level must be in [0, 1], got {level}")
    return level


def _blend(color, level, background):
    """`color` moved toward `background`, keeping `level` of its contrast."""
    r, g, b, a = to_rgba(color)
    br, bg, bb, _ = to_rgba(background)
    return (br + level * (r - br), bg + level * (g - bg), bb + level * (b - bb), a)


def blend(color, level, background=None):
    """A color moved toward a background, keeping `level` of its contrast.

    This is the blend `fade` applies to every color an artist draws with,
    exposed for a color that has no artist yet, or for a property `fade`
    does not reach on its own (for example a hatch color set after the
    fact). Each RGB channel moves linearly toward the background's; the
    color's own alpha is kept, and the background's alpha is ignored.

    Args:
        color: Any matplotlib color.
        level: The fraction of contrast kept, from 0.0 (the background
            color itself) to 1.0 (`color` unchanged).
        background: The color to blend toward. None uses
            `rcParams["axes.facecolor"]`, read at call time, which is the
            last fallback `fade` itself uses when an artist sits on no
            opaque face.

    Returns:
        The blended color as an `(r, g, b, a)` tuple of floats.

    Raises:
        ValueError: If `level` is outside [0, 1].

    Example::

        hatch = ep.blend(accent, 0.45, background=ax.get_facecolor())
    """
    level = _check_level(level)
    if background is None:
        background = matplotlib.rcParams["axes.facecolor"]
    return _blend(color, level, background)


def _opaque(color):
    return to_rgba(color)[3] > 0


def _axes_background(ax):
    """The axes face color if the axes paints an opaque face, else None."""
    patch = ax.patch
    if ax.axison and ax.get_frame_on() and patch.get_visible():
        face = patch.get_facecolor()
        if _opaque(face):
            return face
    return None


def _figure_background(fig):
    """The nearest opaque face among a (sub)figure and its parents, or None."""
    while fig is not None:
        face = fig.get_facecolor()
        if _opaque(face):
            return face
        parent = getattr(fig, "figure", None)
        fig = None if parent is fig else parent
    return None


def _resolve_background(artist):
    """The color an artist sits on: its axes face, else a figure face, else rc."""
    ax = artist if isinstance(artist, Axes) else getattr(artist, "axes", None)
    if ax is not None:
        face = _axes_background(ax)
        if face is not None:
            return face
    fig = artist if isinstance(artist, FigureBase) else getattr(artist, "figure", None)
    face = _figure_background(fig)
    if face is not None:
        return face
    return matplotlib.rcParams["axes.facecolor"]


def _fade_text(text, level, bg, handle):
    handle._record(text, text.set_color, text.get_color())
    text.set_color(_blend(text.get_color(), level, bg))


def _fade_line(line, level, bg, handle):
    # Read every color before writing any: a marker color left on "auto"
    # reads back as the line color, which would otherwise blend twice.
    pairs = [
        (line.get_color(), line.set_color),
        (line.get_markerfacecolor(), line.set_markerfacecolor),
        (line.get_markeredgecolor(), line.set_markeredgecolor),
        (line.get_markerfacecoloralt(), line.set_markerfacecoloralt),
    ]
    for value, setter in pairs:
        if isinstance(value, str) and value.lower() == "none":
            continue
        handle._record(line, setter, value)
        setter(_blend(value, level, bg))


def _fade_patch(patch, level, bg, handle):
    face = patch.get_facecolor()
    if _opaque(face):
        handle._record(patch, patch.set_facecolor, face)
        patch.set_facecolor(_blend(face, level, bg))
    edge = patch.get_edgecolor()
    if _opaque(edge):
        handle._record(patch, patch.set_edgecolor, edge)
        patch.set_edgecolor(_blend(edge, level, bg))
    if patch.get_hatch():
        hatch = patch.get_hatchcolor()
        handle._record(patch, patch.set_hatchcolor, hatch)
        patch.set_hatchcolor(_blend(hatch, level, bg))


def _fade_collection(coll, level, bg, handle):
    # A colormapped collection recomputes its faces from the array at draw
    # time, so bake the mapped colors first and detach the array until the
    # handle restores it.
    if coll.get_array() is not None:
        coll.update_scalarmappable()
        handle._record(coll, coll.set_array, coll.get_array())
        faces = coll.get_facecolor().copy()
        edges = coll.get_edgecolor().copy()
        coll.set_array(None)
        coll.set_facecolor(faces)
        coll.set_edgecolor(edges)
    # Read both before writing either: an edge color of "face" reads back
    # as the face colors, which would otherwise be blended twice.
    faces = coll.get_facecolor().copy()
    edges = coll.get_edgecolor().copy()
    if len(faces):
        handle._record(coll, coll.set_facecolor, faces.copy())
        coll.set_facecolor([_blend(c, level, bg) for c in faces])
    if len(edges):
        handle._record(coll, coll.set_edgecolor, edges.copy())
        coll.set_edgecolor([_blend(c, level, bg) for c in edges])


def _fade_image(image, level, handle):
    alpha = image.get_alpha()
    handle._record(image, image.set_alpha, alpha)
    image.set_alpha(level * (1.0 if alpha is None else alpha))


def _fade_alpha(artist, level, handle):
    """Scale an artist's own alpha by `level`; hide it outright at 0."""
    alpha = artist.get_alpha()
    handle._record(artist, artist.set_alpha, alpha)
    artist.set_alpha(level * (1.0 if alpha is None else alpha))
    if level == 0.0:
        handle._record(artist, artist.set_visible, artist.get_visible())
        artist.set_visible(False)


def _own_background(artist):
    """The face a container paints itself, or None when it paints none."""
    if isinstance(artist, Axes):
        return _axes_background(artist)
    face = artist.get_facecolor()
    return face if _opaque(face) else None


def _visit(artist, level, explicit, inherited, keep, handle, seen, by="color"):
    """Fade one artist, or recurse into a container's children.

    `explicit` is the caller's background, which wins everywhere.
    `inherited` is the background resolved for the container this artist
    was reached from; a container that paints its own face overrides it
    for its children. `by="alpha"` scales each leaf's own alpha instead of
    blending its colors, and needs no background.
    """
    if id(artist) in seen:
        return
    seen.add(id(artist))
    if keep is not None and keep(artist):
        return
    container = isinstance(artist, (Axes, FigureBase))
    if by == "alpha":
        bg = None
    elif explicit is not None:
        bg = explicit
    elif container and _own_background(artist) is not None:
        bg = _own_background(artist)
    elif inherited is not None:
        bg = inherited
    else:
        bg = _resolve_background(artist)

    def recurse(child):
        _visit(child, level, explicit, bg, keep, handle, seen, by)

    if container:
        for child in artist.get_children():
            if child is not artist.patch:
                recurse(child)
        return
    leaf = isinstance(artist, (Text, Line2D, Patch, Collection, *_IMAGES))
    if by == "alpha" and leaf:
        _fade_alpha(artist, level, handle)
        if isinstance(artist, Text):
            for part in (artist.get_bbox_patch(), getattr(artist, "arrow_patch", None)):
                if part is not None:
                    recurse(part)
        return
    if isinstance(artist, Text):
        _fade_text(artist, level, bg, handle)
        box = artist.get_bbox_patch()
        if box is not None:
            recurse(box)
        arrow = getattr(artist, "arrow_patch", None)
        if arrow is not None:
            recurse(arrow)
        return
    if isinstance(artist, Line2D):
        _fade_line(artist, level, bg, handle)
    elif isinstance(artist, Patch):
        _fade_patch(artist, level, bg, handle)
    elif isinstance(artist, Collection):
        _fade_collection(artist, level, bg, handle)
    elif isinstance(artist, _IMAGES):
        _fade_image(artist, level, handle)
    else:
        for child in artist.get_children():
            recurse(child)


# How `fade` quiets an artist.
_FADE_BY = ("color", "alpha")


def fade(target, level, *, background=None, keep=None, by="color"):
    """Blend artists toward the background they sit on, or lower their opacity.

    Every color an artist draws with moves toward the background, keeping
    `level` of its contrast: a line's color and marker colors, a patch's
    face, edge, and hatch, a collection's faces and edges, and a text's
    color, backing box, and annotation arrow. Opacity is left alone, so a
    faded element still hides what is beneath it, exactly as the unfaded
    one did. Because the target is the real background, the same call fades
    correctly on a light figure and a dark one.

    An image fades by opacity instead, at `level` times its current alpha,
    which equals the color blend exactly when the image sits on a uniform
    background (the usual case: an image fills its own axes).

    Fading twice compounds: two fades at 0.5 leave a quarter of the
    contrast. Colors are read at call time, so fade after the artists take
    their final colors.

    `by="alpha"` fades by opacity instead, for an element that should come
    and go rather than recede, such as a part of a figure that arrives
    during a build-up and must not paint the background color over what is
    already drawn: every leaf artist's own alpha becomes `level` times the
    alpha it had (None counting as 1), and at `level` 0 the artist is also
    hidden, so it draws nothing at all. An artist that carries its
    transparency only in its colors, with no alpha of its own, is drawn at
    `level` times full opacity. To animate an arrival from the drawn
    opacities, restore the previous handle before each new fade, which
    keeps every step relative to the alpha the artists were drawn with::

        faded = ep.fade(group, 0.0, by="alpha")
        for u in np.linspace(0.0, 1.0, 12):
            faded.restore()
            faded = ep.fade(group, u, by="alpha")

    Args:
        target: What to fade. An Artist; an Axes, which fades everything it
            holds (inset axes, spines, ticks, titles, and legends included)
            except its own background patch; a Figure or SubFigure, which
            fades every axes and figure-level artist in it except its
            background patch; or an iterable of any of these.
        level: The fraction of contrast kept, from 0.0 (the element
            vanishes into the background) to 1.0 (unchanged).
        background: The color to blend toward. None resolves it per
            artist: the face color of the axes it is drawn in when that
            axes paints an opaque face, else the nearest opaque figure face,
            else `rcParams["axes.facecolor"]`. An inset axes with no face
            of its own blends toward the face of the axes it was reached
            from.
        keep: Optional predicate called on every artist visited, containers
            included. Where it returns True the artist, and everything
            inside it, is left untouched.
        by: `"color"` (the default) blends colors toward the background;
            `"alpha"` scales each artist's own opacity and hides it at 0.
            `background` is ignored under `"alpha"`.

    Returns:
        A `Faded` handle. Its `artists` lists the leaf artists changed, and
        its `restore()` puts every original color (or alpha and visibility)
        back, so an animation can toggle a region without redrawing it.

    Raises:
        ValueError: If `level` is outside [0, 1], or `by` is not `"color"`
            or `"alpha"`.

    Example::

        with ep.capture(ax) as earlier:
            ax.plot(x, y)
        ax.plot(x, y2)  # the new element, at full strength
        ep.fade(earlier, 0.3)
    """
    level = _check_level(level)
    if by not in _FADE_BY:
        raise ValueError(f"unknown fade by: {by!r}; known: {list(_FADE_BY)}")
    targets = [target] if isinstance(target, Artist) else list(target)
    handle = Faded()
    seen = set()
    for item in targets:
        _visit(item, level, background, None, keep, handle, seen, by)
    return handle


@contextlib.contextmanager
def capture(container):
    """Collect the artists added to an axes (or figure) inside a block.

    The yielded list is empty inside the block and holds, once the block
    exits, every direct child that `container` gained during it: lines,
    patches, collections, images, texts, and inset axes, in the order
    `container.get_children()` reports them. An inset axes is collected as
    one entry, so `fade` on the list reaches everything drawn in it. Only
    new artists are collected; a property changed on an existing artist,
    such as a new title string, is not.

    Args:
        container: An Axes, Figure, or SubFigure.

    Yields:
        The list that is filled when the block exits.

    Example::

        with ep.capture(ax) as scene:
            ax.plot(x, y)
            ax.text(0.5, 0.5, "star")
        ep.fade(scene, 0.3)
    """
    before = {id(child) for child in container.get_children()}
    added = []
    try:
        yield added
    finally:
        added.extend(
            child for child in container.get_children() if id(child) not in before
        )


def _step_rows(rows):
    """`(heading, detail or None)` for each row: a string or a pair."""
    out = []
    for row in rows:
        if isinstance(row, str):
            out.append((row, None))
        else:
            heading, detail = row
            out.append((heading, detail))
    return out


def step_list(
    rows,
    *,
    ax=None,
    current=None,
    numbered=True,
    gap=0.2,
    indent=0.05,
    detail_offset=0.075,
    dim=None,
    text_kw=None,
):
    """Write a numbered list of steps whose current step stands out.

    A narrated build-up, or a tour of a figure one part at a time, reads
    better beside a list of its steps that keeps the reader's place: the
    current step bright and bold, the steps already taken dim, and the steps
    still to come not yet shown. `update` moves the place without drawing
    anything new, so an animation can step it per frame.

    The list fills its own axes, top down, in axes coordinates, with the
    axis turned off.

    Args:
        rows: The steps, in order. Each is a heading string, or a
            `(heading, detail)` pair whose detail is written under the
            heading, indented.
        ax: Axes to write into. None creates a new figure and axes.
        current: The step to show as current on the first draw, 0-based;
            -1 shows none yet. None shows every step plain, as a static
            list.
        numbered: Prefix each heading with its step number, `"1. "`.
        gap: Vertical distance between consecutive steps, as a fraction of
            the axes height. The first heading's top sits at 0.95.
        indent: Horizontal indent of a detail, as a fraction of the axes
            width.
        detail_offset: Distance from a heading's top down to its detail's
            top, as a fraction of the axes height.
        dim: Color of a finished step. None uses a neutral tone halfway
            between the background and the text color.
        text_kw: Extra kwargs for every `Text` (for example `fontsize`),
            applied last. Color and weight carry the step's state and are
            set by `update`.

    Returns:
        A `PlotResult` whose `artists["text"]` lists the `Text` artists in
        row order, each heading followed by its detail when the row has
        one, and an `update(current, done=False)` that shows steps 0 to
        `current` (-1 for none), the current one in the text color and
        bold and the earlier ones dim, and hides the rest; `done=True` shows
        every step dim, as a finished list.

    Raises:
        ValueError: If `rows` is empty.

    Example::

        res = ep.step_list([("Steer", "hold the star on the mask"),
                            ("Correct", "dig the dark hole")], current=-1)
        for step in range(2):
            res.update(step)
    """
    entries = _step_rows(rows)
    if not entries:
        raise ValueError("step_list needs at least one row")
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.axis("off")
    ink = matplotlib.rcParams["text.color"]
    faint = _style.neutral(0.55) if dim is None else dim
    kw = {"ha": "left", "va": "top", "color": ink, **(text_kw or {})}
    texts = []
    groups = []
    for i, (heading, detail) in enumerate(entries):
        y = 0.95 - i * gap
        head_text = f"{i + 1}. {heading}" if numbered else heading
        group = [ax.text(0.0, y, head_text, **kw)]
        if detail is not None:
            group.append(ax.text(indent, y - detail_offset, detail, **kw))
        texts += group
        groups.append(group)

    def update(current, done=False):
        current = int(current)
        for i, group in enumerate(groups):
            shown = done or i <= current
            now = i == current and not done
            for text in group:
                text.set_visible(shown)
                text.set_color(ink if now else faint)
            group[0].set_fontweight("bold" if now else "normal")

    if current is not None:
        update(current)
    return PlotResult(ax=ax, artists={"text": texts}, update=update)
