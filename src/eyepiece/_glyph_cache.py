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
_MAX_GLYPHS = 200_000
_layouts = {}
_glyphs = {}
_lock = threading.Lock()
_depth = 0
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
        items = _layouts[layout_key] = tuple(
            (item.ft_object, item.glyph_index, item.x, item.y)
            for item in font._layout(s, flags=flags, features=features,
                                     language=language)
        )  # fmt: skip
    if len(_glyphs) > _MAX_GLYPHS:
        _glyphs.clear()
        _layouts.clear()
    # The arithmetic below is RendererAgg._draw_text_glyphs_and_boxes with
    # each glyph origin split into whole pixels and a 1/64 px remainder.
    cos = math.cos(math.radians(angle))
    sin = math.sin(math.radians(angle))
    matrix = (0x10000 * np.array([[cos, -sin], [sin, cos]])).round().astype(int)
    matrix_key = tuple(matrix.ravel().tolist())
    antialiased = gc.get_antialiased()
    mode = backend_agg.RenderMode.NORMAL if antialiased else backend_agg.RenderMode.MONO
    height = self.height
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
            glyph = _glyphs[glyph_key] = (buffer, bitmap.left, bitmap.top)
        buffer, left, top = glyph
        self._renderer.draw_text_image(
            buffer,
            left + (origin_x >> 6),
            int(height) - (top + (origin_y >> 6)) + buffer.shape[0],
            0,
            gc,
        )


def _reference_pixels(figure_draw_text):
    """Pixels of a figure exercising rotation, sizes and both antialias modes."""
    from matplotlib.figure import Figure

    backend_agg.RendererAgg.draw_text = figure_draw_text
    try:
        fig = Figure(figsize=(2.4, 1.2), dpi=97)
        backend_agg.FigureCanvasAgg(fig)
        fig.text(0.031, 0.71, "Wavefront 10^-11", size=9.3)
        fig.text(0.403, 0.33, "rotated AVW", size=13, rotation=33)
        fig.text(0.6, 0.85, "mono ink", size=7, antialiased=False)
        fig.text(0.517, 0.12, "fi fl -> x", size=11.7, rotation=-90, weight="bold")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba()).copy()
    finally:
        backend_agg.RendererAgg.draw_text = _STOCK


def _self_test():
    """True when the cached path reproduces stock pixels on this matplotlib."""
    try:
        stock = _reference_pixels(_STOCK)
        cached = _reference_pixels(_cached_draw_text)
    except (AttributeError, TypeError, ValueError):
        return False
    finally:
        _glyphs.clear()
        _layouts.clear()
    return bool(np.array_equal(stock, cached))


@contextmanager
def glyph_cache():
    """Reuse text layouts and glyph bitmaps in every Agg draw inside the block.

    Re-entrant. Pixels are identical to stock drawing; on a matplotlib whose
    text path this module does not reproduce exactly, nothing changes.
    """
    global _depth, _usable
    with _lock:
        if _usable is None:
            _usable = _self_test()
        enter = _usable and _depth == 0
        _depth += 1
        if enter:
            backend_agg.RendererAgg.draw_text = _cached_draw_text
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0 and _usable:
                backend_agg.RendererAgg.draw_text = _STOCK
