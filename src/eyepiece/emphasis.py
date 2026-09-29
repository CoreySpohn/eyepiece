"""Fade drawn artists toward the background, and collect what a block drew.

A figure that builds toward a whole in steps carries the elements a reader
has already seen into the next step, and shows them as "already seen": still
present, still in place, but quieter than what is new. `fade` does that by
blending every color an artist draws with toward the background it sits on,
rather than lowering its opacity. A blended element never lets what is
under it show through, and because the target is the actual background a
faded element reads correctly on a light page and a dark one alike.

`capture` is the other half: a context manager that collects the artists a
block of drawing code added to an axes, so a caller can fade exactly that
group later without tagging each artist by hand.
"""

import contextlib

import matplotlib
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.figure import FigureBase
from matplotlib.image import AxesImage, BboxImage, FigureImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

_IMAGES = (AxesImage, BboxImage, FigureImage)


class Faded:
    """Handle returned by `fade`: the artists it changed and how to undo it.

    Attributes:
        artists: The leaf artists whose colors (or, for images, opacity)
            `fade` changed, in the order it visited them.
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


def _blend(color, level, background):
    """`color` moved toward `background`, keeping `level` of its contrast."""
    r, g, b, a = to_rgba(color)
    br, bg, bb, _ = to_rgba(background)
    return (br + level * (r - br), bg + level * (g - bg), bb + level * (b - bb), a)


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
    # Hatch colors are settable from matplotlib 3.10; before that a hatch
    # keeps the color it was built with.
    if patch.get_hatch() and hasattr(patch, "set_hatchcolor"):
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


def _own_background(artist):
    """The face a container paints itself, or None when it paints none."""
    if isinstance(artist, Axes):
        return _axes_background(artist)
    face = artist.get_facecolor()
    return face if _opaque(face) else None


def _visit(artist, level, explicit, inherited, keep, handle, seen):
    """Fade one artist, or recurse into a container's children.

    `explicit` is the caller's background, which wins everywhere.
    `inherited` is the background resolved for the container this artist
    was reached from; a container that paints its own face overrides it
    for its children.
    """
    if id(artist) in seen:
        return
    seen.add(id(artist))
    if keep is not None and keep(artist):
        return
    container = isinstance(artist, (Axes, FigureBase))
    if explicit is not None:
        bg = explicit
    elif container and _own_background(artist) is not None:
        bg = _own_background(artist)
    elif inherited is not None:
        bg = inherited
    else:
        bg = _resolve_background(artist)

    def recurse(child):
        _visit(child, level, explicit, bg, keep, handle, seen)

    if container:
        for child in artist.get_children():
            if child is not artist.patch:
                recurse(child)
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


def fade(target, level, *, background=None, keep=None):
    """Blend artists toward the background they sit on.

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

    Returns:
        A `Faded` handle. Its `artists` lists the leaf artists changed, and
        its `restore()` puts every original color back, so an animation can
        toggle a region without redrawing it.

    Raises:
        ValueError: If `level` is outside [0, 1].

    Example::

        with ep.capture(ax) as earlier:
            ax.plot(x, y)
        ax.plot(x, y2)  # the new element, at full strength
        ep.fade(earlier, 0.3)
    """
    level = float(level)
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"fade level must be in [0, 1], got {level}")
    targets = [target] if isinstance(target, Artist) else list(target)
    handle = Faded()
    seen = set()
    for item in targets:
        _visit(item, level, background, None, keep, handle, seen)
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
