"""Exact reuse of rasterized glyphs across Agg redraws.

Matplotlib's Agg text path lays out every string and renders every glyph
through FreeType on each draw. A movie redraws the same strings at the same
sizes thousands of times, and text dominates its raster time, so this module
caches both steps while a `glyph_cache` block is open:

- layouts, keyed by the resolved font, size, dpi, string, hinting and the
  text's font features and language;
- glyph bitmaps, keyed by font, size, dpi, glyph index, rotation and the
  position of the glyph origin within its pixel on FreeType's 1/64 px grid.

Translating an outline by whole pixels does not change its coverage, so a
cached bitmap placed with a whole-pixel shift is the bitmap FreeType would
have rendered, and the result matches the stock renderer byte for byte.

The cached path calls private `FT2Font` methods of matplotlib 3.11. Before the
first use in a process it draws a reference figure both ways; if the pixels
differ, or the methods are missing, the stock `draw_text` stays in place.
"""

import math
import threading
from contextlib import contextmanager

import numpy as np
from matplotlib.backends import backend_agg

_STOCK = backend_agg.RendererAgg.draw_text
_MAX_LAYOUTS = 10_000
_MAX_GLYPH_BYTES = 64 * 1024**2
_layouts = {}
_glyphs = {}
_glyph_bytes = 0
_lock = threading.Lock()
_depth = 0
_installed = False
_previous = None
_usable = None


def _hashable(value):
    if value is None or isinstance(value, str):
        return value
    return tuple(_hashable(v) for v in value)


def _cached_draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
    """`RendererAgg.draw_text` with layouts and glyph bitmaps reused."""
    if ismath:
        return _STOCK(self, gc, x, y, s, prop, angle, ismath, mtext)
    flags = backend_agg.get_hinting_flag()
    features = mtext.get_fontfeatures() if mtext is not None else None
    language = mtext.get_language() if mtext is not None else None
    font = self._prepare_font(prop)
    size = prop.get_size_in_points()
    try:
        layout_key = (font, size, self.dpi, s, flags, _hashable(features),
                      _hashable(language))  # fmt: skip
        items = _layouts.get(layout_key)
    except TypeError:  # an unhashable feature or language spec
        return _STOCK(self, gc, x, y, s, prop, angle, ismath, mtext)
    if items is None:
        if len(_layouts) >= _MAX_LAYOUTS:
            _layouts.clear()
        items = _layouts[layout_key] = tuple(
            (item.ft_object, item.glyph_index, item.x, item.y)
            for item in font._layout(s, flags=flags, features=features,
                                     language=language)
        )  # fmt: skip
    # The arithmetic below is RendererAgg._draw_text_glyphs_and_boxes with
    # each glyph origin split into whole pixels and a 1/64 px remainder.
    cos = math.cos(math.radians(angle))
    sin = math.sin(math.radians(angle))
    matrix = (0x10000 * np.array([[cos, -sin], [sin, cos]])).round().astype(int)
    matrix_key = tuple(matrix.ravel().tolist())
    antialiased = gc.get_antialiased()
    mode = backend_agg.RenderMode.NORMAL if antialiased else backend_agg.RenderMode.MONO
    height = self.height
    global _glyph_bytes
    for ft_object, glyph_index, dx, dy in items:
        origin_x = round(0x40 * (x + dx * cos - dy * sin))
        origin_y = round(0x40 * (height - y + dx * sin + dy * cos))
        glyph_key = (ft_object, size, self.dpi, glyph_index, matrix_key,
                     origin_x & 63, origin_y & 63, antialiased, flags)  # fmt: skip
        glyph = _glyphs.get(glyph_key)
        if glyph is None:
            ft_object.set_size(size, self.dpi)
            ft_object._set_transform(matrix, [origin_x & 63, origin_y & 63])
            bitmap = ft_object._render_glyph(glyph_index, flags, mode)
            buffer = np.array(bitmap.buffer)
            if not antialiased:
                buffer *= 0xFF
            if _glyph_bytes + buffer.nbytes > _MAX_GLYPH_BYTES:
                _glyphs.clear()
                _glyph_bytes = 0
            glyph = _glyphs[glyph_key] = (buffer, bitmap.left, bitmap.top)
            _glyph_bytes += buffer.nbytes
        buffer, left, top = glyph
        self._renderer.draw_text_image(
            buffer,
            left + (origin_x >> 6),
            int(height) - (top + (origin_y >> 6)) + buffer.shape[0],
            0,
            gc,
        )


def _reference_pixels(draw_text):
    """Pixels of a figure exercising rotation, sizes and both antialias modes.

    `draw_text` is bound to this figure's renderer only, so the probe never
    touches the class other figures draw through.
    """
    import types

    from matplotlib import rc_context
    from matplotlib.figure import Figure

    with rc_context({"text.usetex": False}):
        fig = Figure(figsize=(2.4, 1.2), dpi=97)
        canvas = backend_agg.FigureCanvasAgg(fig)
        fig.text(0.031, 0.71, "Wavefront 10^-11", size=9.3)
        fig.text(0.403, 0.33, "rotated AVW", size=13, rotation=33)
        fig.text(0.6, 0.85, "mono ink", size=7, antialiased=False)
        fig.text(0.517, 0.12, "fi fl -> x", size=11.7, rotation=-90, weight="bold")
        renderer = canvas.get_renderer()
        renderer.draw_text = types.MethodType(draw_text, renderer)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()


def _self_test():
    """True when the cached path reproduces stock pixels on this matplotlib."""
    try:
        stock = _reference_pixels(_STOCK)
        cached = _reference_pixels(_cached_draw_text)
    except Exception:  # any failure of the private text path means stock drawing
        return False
    return bool(np.array_equal(stock, cached))


@contextmanager
def glyph_cache():
    """Reuse text layouts and glyph bitmaps in every Agg draw inside the block.

    Re-entrant. Pixels are identical to stock drawing. Nothing changes on a
    matplotlib whose text path this module does not reproduce exactly, or when
    something else has already replaced `RendererAgg.draw_text`; leaving the
    outermost block restores the method found on entry. The caches outlive
    the block (a render worker or a run of checks keeps them warm) and are
    bounded: at most `_MAX_LAYOUTS` layouts and `_MAX_GLYPH_BYTES` of bitmaps.
    """
    global _depth, _installed, _previous, _usable
    if _usable is None:
        usable = _self_test()  # outside the lock: it draws a figure
        with _lock:
            if _usable is None:
                _usable = usable
    with _lock:
        if _depth == 0:
            _previous = backend_agg.RendererAgg.draw_text
            _installed = _usable and _previous is _STOCK
            if _installed:
                backend_agg.RendererAgg.draw_text = _cached_draw_text
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0 and _installed:
                if backend_agg.RendererAgg.draw_text is _cached_draw_text:
                    backend_agg.RendererAgg.draw_text = _previous
                _installed = False
