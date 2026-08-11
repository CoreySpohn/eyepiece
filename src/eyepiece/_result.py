"""Return contract for plotting primitives.

Every primitive returns a `PlotResult` (single axes) or `MosaicResult`
(multi-panel), never bare matplotlib objects. Both are frozen dataclasses
so a caller cannot accidentally mutate `.ax`/`.axes` out from under the
figure; redraw by calling the primitive again or, where offered, `.update`.

`ARTIST_KEYS` is the fixed, documented vocabulary a primitive's `.artists`
dict draws its keys from. It is a convention, not an enforced schema: a
primitive is free to populate only the keys it actually draws. Each key
holds either a single matplotlib artist or, for a multi-panel primitive
that draws the same kind of artist once per panel, a list of them (one
entry per panel, in panel order):

    image: the `AxesImage` from `imshow`/`pcolormesh` (or a list of one
        per panel for a mosaic primitive).
    cbar: the `Colorbar` attached to an image or scalar mappable (or a
        list of one per panel).
    line: a single `Line2D` (or a list of one per panel).
    lines: a list of `Line2D` artists drawn together on one axes (e.g. a
        multi-curve plot), as distinct from `line`'s one-per-panel list.
    fill: the `PolyCollection` from `fill_between`/`fill_betweenx` (or a
        list of one per panel).
    hist: the `BarContainer` (or patch list) from `hist` (or a list of
        one per panel).
    scatter: the `PathCollection` from `scatter` (or a list of one per
        panel).
    ellipse: an `Ellipse` (or other `Patch`) artist marking a region
        (or a list of one per panel).
    collection: a generic `Collection` artist not covered by a more
        specific key above (or a list of one per panel).
    text: a `Text` artist placed as an annotation, not the title (or a
        list of one per panel).
    title: the `Text` artist returned by `set_title` (or a list of one
        per panel).
"""

from dataclasses import dataclass, field

ARTIST_KEYS = frozenset(
    {
        "image",
        "cbar",
        "line",
        "lines",
        "fill",
        "hist",
        "scatter",
        "ellipse",
        "collection",
        "text",
        "title",
    }
)


@dataclass(frozen=True)
class PlotResult:
    """Result of a single-axes plotting primitive.

    Attributes:
        ax: The matplotlib Axes drawn on.
        artists: Artists keyed by name; see the module docstring for the
            key vocabulary.
        update: Optional callable that redraws with new data, reusing the
            transform (log floor, phase NaN-mask, decade titles, ...) the
            primitive applied on first draw. None when the primitive has
            no such transform to reuse.
    """

    ax: object
    artists: dict
    update: object = field(default=None)

    @property
    def fig(self):
        """The Figure that owns `ax`."""
        return self.ax.figure


@dataclass(frozen=True)
class MosaicResult:
    """Result of a multi-panel plotting primitive.

    Attributes:
        axes: The array of matplotlib Axes drawn on (as returned by
            `plt.subplots`, e.g. a numpy object array).
        artists: Artists keyed by name; see the module docstring for the
            key vocabulary. A multi-panel primitive that draws the same
            kind of artist once per panel stores a list under that key.
        update: Optional callable that redraws with new data, reusing the
            transform the primitive applied on first draw. None when the
            primitive has no such transform to reuse.
    """

    axes: object
    artists: dict
    update: object = field(default=None)

    @property
    def fig(self):
        """The Figure that owns `axes`."""
        return self.axes.flat[0].figure
