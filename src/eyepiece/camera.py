"""Cameras for explanatory animations: a view box, a page camera, a spotlight.

Two kinds of camera move a figure. A data camera changes an axes' limits,
so the data magnify while text and ticks keep their size; `view_limits` and
`zoom_path` drive it, `overview_box` marks its view on an overview panel,
and `record(free_limits=...)` lets its limits move without tripping the
scale-drift guard. A page camera moves over a finished figure as a whole,
so everything magnifies together; `PageCamera` renders any view of a figure
at a fixed output size by cropping `savefig` at the zoomed dpi, so text,
lines and images are re-rendered rather than enlarged, and its frames go to
a `RawSink`.

`spotlight` is the fixed-camera alternative to a page camera: a veil in the
figure background color dims everything outside one or more holes, which
move from stop to stop while the figure stays still.
"""

import io
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D, Bbox

from eyepiece import _style
from eyepiece._motion import _view, view_limits
from eyepiece._result import MosaicResult, PlotResult

# Below this dpi FreeType cannot set a font of ordinary size (its pixel size
# rounds to zero); a view that would render there is rendered at this dpi
# and downsampled instead.
_MIN_DPI = 10.0
# The overview thumbnail renders at least this sharp, then is downsampled.
_THUMB_DPI = 30.0


def _opaque_face(fig):
    """The figure's facecolor as RGBA, white where the figure is transparent."""
    rgba = to_rgba(fig.get_facecolor())
    return rgba if rgba[3] > 0.0 else to_rgba("white")


def overview_box(view, *, ax=None, aspect=1.0, color=None, box_kw=None):
    """Draw a camera's view as a box on an overview panel.

    A zoom that passes about ten times loses the viewer's place; an overview
    of the whole field beside the zoomed panel, with the current view boxed
    on it, keeps it. Draw the overview with any image primitive, then call
    this on its axes and `update` the box every frame.

    Args:
        view: ``(cx, cy, w)``, the camera's center and width in the overview's
            data units.
        ax: Axes to draw on. None creates and owns a new figure.
        aspect: Height over width of the view, in data units.
        color: Box edge color. None uses the text color at call time.
        box_kw: Extra keyword arguments for the `Rectangle`, merged last.

    Returns:
        A `PlotResult` whose ``artists["ellipse"]`` is the `Rectangle`, with
        ``update(view)`` moving it.

    Raises:
        ValueError: If the width or `aspect` is not positive.
    """
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    (x0, x1), (y0, y1) = view_limits(view, aspect=aspect)
    edge = matplotlib.rcParams["text.color"] if color is None else color
    kw = {"fill": False, "edgecolor": edge, "linewidth": 1.2, "zorder": 5}
    kw.update(box_kw or {})
    # add_artist, not add_patch: the box marks a view, it is not data, so it
    # must not widen the overview's limits on a later autoscale.
    box = Rectangle((x0, y0), x1 - x0, y1 - y0, **kw)
    ax.add_artist(box)

    def update(new_view):
        (a0, a1), (b0, b1) = view_limits(new_view, aspect=aspect)
        box.set_bounds(a0, b0, a1 - a0, b1 - b0)

    return PlotResult(ax=ax, artists={"ellipse": box}, update=update)


def _restore_engine(fig, engine):
    """Put back a saved layout engine, keeping None meaning no engine."""
    if engine is not None:
        fig.set_layout_engine(engine)
        return
    with matplotlib.rc_context(
        {"figure.autolayout": False, "figure.constrained_layout.use": False}
    ):
        fig.set_layout_engine(None)


class PageCamera:
    """Render any view of a finished figure at a fixed output size.

    A view is ``(cx, cy, w)`` in figure inches, with the output's aspect.
    Each render crops ``savefig`` to the view at ``dpi = width_px / w``, so a
    magnified view re-renders text, lines and images at the larger size
    instead of enlarging a raster: a page camera is as sharp at five times
    as at one. Construction solves and freezes the figure's layout, so a
    render at another dpi cannot move the panels between frames; `release`,
    or leaving a ``with`` block, restores the figure's layout engine and ends
    the camera.

    With ``overview=True`` the view fills the left five sixths of each frame
    and a margin in the figure's facecolor holds a thumbnail of the whole
    page with the view boxed on it, so a zoomed viewer keeps their place and
    the overview never covers the figure. The thumbnail is drawn once, at
    the first render; call `refresh` after changing what the figure shows.
    A transparent figure is filmed on white.

    Example::

        with PageCamera(fig, (1920, 1080), overview=True) as cam:
            path, _ = zoom_path(cam.page, cam.fit_axes(axes[2]))
            with RawSink("tour.mp4", cam.size_px, fps=30) as sink:
                for s in np.linspace(0.0, 1.0, 60):
                    sink.write(cam.render(path(ease(s))))

    Args:
        fig: The figure to film. Draw everything first.
        size_px: ``(width, height)`` of each frame in pixels.
        overview: Reserve a margin for the page thumbnail.

    Raises:
        ValueError: If a dimension is not a positive integer.
    """

    def __init__(self, fig, size_px=(1920, 1080), *, overview=False):
        """Solve and freeze the figure's layout and fix the frame size."""
        width, height = (int(v) for v in size_px)
        if width <= 0 or height <= 0:
            raise ValueError(f"size_px must be positive, got {size_px}")
        self.fig = fig
        self.size_px = (width, height)
        self.overview = bool(overview)
        self._margin = width // 6 if self.overview else 0
        self._view_px = (width - self._margin, height)
        self._engine = fig.get_layout_engine()
        fig.draw_without_rendering()
        fig.set_layout_engine("none")
        self._thumb = None
        self._released = False

    def __enter__(self):
        """Return the camera."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Release the camera, whether or not the block raised."""
        self.release()

    @property
    def aspect(self):
        """Height over width of the view region of a frame."""
        return self._view_px[1] / self._view_px[0]

    def release(self):
        """Restore the figure's layout engine and end the camera.

        Renders after this would see a layout free to move between dpis, so
        `render` refuses them. Releasing twice is harmless.
        """
        if not self._released:
            _restore_engine(self.fig, self._engine)
            self._released = True

    def refresh(self):
        """Redraw the overview thumbnail at the next render."""
        self._thumb = None

    @property
    def page(self):
        """The whole figure as a view, letterboxed to the view aspect."""
        w_in, h_in = self.fig.get_size_inches()
        return self.fit((0.0, 0.0, w_in, h_in), pad=0.0)

    def fit(self, rect_in, *, pad=0.04):
        """The smallest view holding ``rect_in = (x0, y0, x1, y1)`` inches.

        The view is widened by `pad` on each side, then slid back inside the
        page where it fits, so a stop near an edge shows no empty canvas.
        """
        x0, y0, x1, y1 = (float(v) for v in rect_in)
        w = max(x1 - x0, (y1 - y0) / self.aspect) * (1.0 + 2.0 * float(pad))
        h = w * self.aspect
        page_w, page_h = self.fig.get_size_inches()
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if w <= page_w:
            cx = min(max(cx, w / 2.0), page_w - w / 2.0)
        if h <= page_h:
            cy = min(max(cy, h / 2.0), page_h - h / 2.0)
        return cx, cy, w

    def fit_axes(self, axes, *, pad=0.04):
        """The view holding the tight bounding boxes of one or more axes."""
        axes = list(axes) if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
        # No renderer argument: the figure resolves its own, which also works
        # after pyplot has closed the figure (as a notebook does per cell).
        boxes = [a.get_tightbbox() for a in axes]
        box = Bbox.union(boxes).transformed(self.fig.dpi_scale_trans.inverted())
        return self.fit((box.x0, box.y0, box.x1, box.y1), pad=pad)

    def _crop(self, view, size):
        """RGBA pixels of `view` at exactly `size` = (width, height)."""
        from PIL import Image

        cx, cy, w = _view(view)
        width, height = size
        h = w * height / width
        dpi = width / w
        if dpi < _MIN_DPI:
            # Too few pixels per inch to draw text: render at the floor and
            # downsample, which only matters for tiny previews of wide views.
            scale = _MIN_DPI / dpi
            big = self._crop(
                view, (math.ceil(width * scale), math.ceil(height * scale))
            )
            return np.asarray(
                Image.fromarray(big).resize((width, height), Image.Resampling.LANCZOS)
            )
        kw = {
            # One part in a million above the exact dpi, so savefig's
            # truncation of a canvas size such as 179.99999 px lands on the
            # requested pixel; far too small to add a pixel.
            "dpi": dpi * (1.0 + 1e-6),
            "bbox_inches": Bbox.from_bounds(cx - w / 2.0, cy - h / 2.0, w, h),
            "facecolor": _opaque_face(self.fig),
            # A style that saves transparent figures would otherwise give
            # every frame a transparent, and in a movie black, background.
            "transparent": False,
        }
        buf = io.BytesIO()
        self.fig.savefig(buf, format="raw", **kw)
        data = np.frombuffer(buf.getvalue(), np.uint8)
        if data.size == width * height * 4:
            return data.reshape(height, width, 4)
        # savefig rounded the canvas by a pixel: read its true size from a
        # PNG of the same crop, then resample to the requested size.
        png = io.BytesIO()
        self.fig.savefig(png, format="png", **kw)
        png.seek(0)
        img = Image.open(png).convert("RGBA")
        return np.asarray(img.resize((width, height), Image.Resampling.LANCZOS))

    def render(self, view):
        """RGBA uint8 pixels of `view`, exactly `size_px`, top row first.

        Args:
            view: ``(cx, cy, w)`` in figure inches. A view reaching past the
                page shows the figure's facecolor there.

        Returns:
            A ``(height, width, 4)`` uint8 array.

        Raises:
            RuntimeError: If the camera has been released.
        """
        if self._released:
            raise RuntimeError("this PageCamera was released; make a new one")
        frame = self._crop(view, self._view_px)
        if not self.overview:
            return frame
        return self._compose(frame, view)

    def _compose(self, frame, view):
        from PIL import Image, ImageDraw

        width, height = self.size_px
        page_w, page_h = self.fig.get_size_inches()
        if self._thumb is None:
            tw = max(self._margin - 16, 8)
            th = max(round(tw * page_h / page_w), 1)
            # A small margin on a wide figure means a few dpi, too few to
            # draw text; render at least _THUMB_DPI and downsample, which is
            # fine for a thumbnail, a picture of the page.
            scale = max(1.0, _THUMB_DPI * page_w / tw)
            big = (round(tw * scale), max(round(th * scale), 1))
            page = self._crop((page_w / 2, page_h / 2, page_w), big)
            self._thumb = Image.fromarray(page).resize(
                (tw, th), Image.Resampling.LANCZOS
            )
            self._box_rgba = tuple(
                round(255 * c) for c in to_rgba(matplotlib.rcParams["text.color"])
            )
            self._frame_rgba = (*(round(255 * c) for c in _style.neutral(0.4)), 255)
        thumb = self._thumb.copy()
        tw, th = thumb.size
        cx, cy, w = _view(view)
        h = w * self.aspect
        box = [
            (cx - w / 2) / page_w * tw,
            (1.0 - (cy + h / 2) / page_h) * th,
            (cx + w / 2) / page_w * tw,
            (1.0 - (cy - h / 2) / page_h) * th,
        ]
        draw = ImageDraw.Draw(thumb)
        draw.rectangle(box, outline=self._box_rgba, width=2)
        draw.rectangle([0, 0, tw - 1, th - 1], outline=self._frame_rgba, width=1)
        face = tuple(round(255 * c) for c in _opaque_face(self.fig))
        out = Image.new("RGBA", (width, height), (*face[:3], 255))
        out.paste(Image.fromarray(frame), (0, 0))
        out.paste(
            thumb.convert("RGBA"), (width - self._margin + (self._margin - tw) // 2, 16)
        )
        return np.asarray(out)


def _check_rects(rects):
    rects = [tuple(float(v) for v in r) for r in rects]
    for x0, y0, x1, y1 in rects:
        if not (x0 < x1 and y0 < y1):
            raise ValueError(
                "each rect must be (x0, y0, x1, y1) with x0 < x1 and y0 < y1, "
                f"got {(x0, y0, x1, y1)}"
            )
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                raise ValueError(
                    "spotlight rects must not overlap: overlapping holes would "
                    "be filled back in"
                )
    return rects


def _inch_transform(fig):
    """Figure inches to display, through the figure fraction.

    Not `fig.dpi_scale_trans`: a cropped save (`bbox_inches=`, which every
    `PageCamera` view and every tight save is) moves `transFigure` to the
    crop but leaves `dpi_scale_trans` where it was, which would slide the
    veil off its holes in any view not anchored at the page's corner.
    """
    w, h = fig.get_size_inches()
    return Affine2D().scale(1.0 / w, 1.0 / h) + fig.transFigure


def _veil_path(fig, rects):
    w, h = fig.get_size_inches()
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * 4
    verts = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h), (0.0, 0.0)]
    all_codes = list(codes)
    for x0, y0, x1, y1 in rects:
        # Wound against the outer square, so each rectangle is a hole.
        verts += [(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]
        all_codes += codes
    return MplPath(verts, all_codes)


def spotlight(fig, rects, *, strength=1.0, color=None, alpha=0.75):
    """Dim a whole figure outside one or more rectangular holes.

    The veil is a figure-level patch in the figure's background color, drawn
    above every axes, in figure inches, the same units as `PageCamera`
    views. It adds no axes and stays out of the layout and of tight bounding
    boxes, so the figure's panels, pyplot's current axes and a later
    ``bbox_inches="tight"`` save are as they were. Several holes light parts
    that belong together (a panel and the colorbar it reads against); they
    must not overlap.

    Args:
        fig: The figure to dim; it must already hold its axes.
        rects: ``(x0, y0, x1, y1)`` holes in figure inches.
        strength: Fraction of `alpha` the veil shows, so a move into or out
            of the spotlight can fade it.
        color: Veil color. None uses the figure's facecolor (white for a
            transparent figure).
        alpha: Veil opacity at full strength.

    Returns:
        A `MosaicResult` over the figure's axes, the panels the veil covers,
        whose ``artists["fill"]`` is the veil `PathPatch`, with
        ``update(rects=None, strength=None)`` moving the holes or fading the
        veil without a new artist.

    Raises:
        ValueError: If the figure has no axes, a rect is inverted, or two
            rects overlap.
    """
    if not fig.axes:
        raise ValueError("spotlight needs a figure that already holds its axes")
    rects = _check_rects(rects)
    face = _opaque_face(fig) if color is None else color
    state = {"strength": float(strength)}
    veil = PathPatch(
        _veil_path(fig, rects),
        transform=_inch_transform(fig),
        facecolor=face,
        edgecolor="none",
        alpha=alpha * state["strength"],
        zorder=1000,
    )
    # Out of the layout and out of tight bounding boxes: the veil spans the
    # whole figure and would otherwise make every tight save the full page.
    veil.set_in_layout(False)
    fig.add_artist(veil)

    def update(rects=None, strength=None):
        if rects is not None:
            veil.set_path(_veil_path(fig, _check_rects(rects)))
            veil.set_transform(_inch_transform(fig))
        if strength is not None:
            state["strength"] = float(strength)
            veil.set_alpha(alpha * state["strength"])

    return MosaicResult(
        axes=np.array(fig.axes, dtype=object), artists={"fill": veil}, update=update
    )
