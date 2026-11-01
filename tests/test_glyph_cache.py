"""The glyph cache reproduces stock Agg text pixels exactly."""

import numpy as np
import pytest
from matplotlib.backends import backend_agg
from matplotlib.figure import Figure

from eyepiece import _glyph_cache
from eyepiece._glyph_cache import glyph_cache


def text_figure(dpi=100):
    fig = Figure(figsize=(3.2, 2.0), dpi=dpi)
    backend_agg.FigureCanvasAgg(fig)
    fig.text(0.02, 0.9, "Plain text, 10^-11 contrast", size=9)
    fig.text(0.05, 0.6, "two\nlines", size=12.3, style="italic")
    fig.text(0.5, 0.5, "rotated", size=14, rotation=17.5, weight="bold")
    fig.text(0.8, 0.2, "upright", size=10, rotation=90)
    fig.text(0.6, 0.85, "upside", size=8, rotation=180)
    fig.text(0.3, 0.1, "mono", size=7, antialiased=False)
    fig.text(0.1, 0.3, r"$\lambda/D = 3.2$", size=11)
    fig.text(0.4, 0.35, "", size=11)
    featured = fig.text(0.45, 0.72, "office fit", size=10)
    featured.set_fontfeatures(["-liga"])
    spoken = fig.text(0.7, 0.6, "sprache", size=9)
    spoken.set_language("de")
    ax = fig.add_axes((0.55, 0.15, 0.2, 0.3))
    ax.plot([0, 1], [0, 1])
    ax.set_title("axes", size=8)
    return fig


def pixels(fig, **savefig):
    if savefig:
        import io

        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="rgba", **savefig)
            return buffer.getvalue()
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).tobytes()


def test_cached_text_is_byte_identical_to_stock():
    fig = text_figure()
    stock = pixels(fig)
    with glyph_cache():
        first = pixels(fig)
        warm = pixels(fig)  # every glyph now comes from the cache
    assert first == stock
    assert warm == stock


def test_subpixel_moves_and_other_dpis_stay_identical():
    fig = text_figure()
    moving = fig.text(0.2, 0.45, "drifting label", size=10.5)
    for step in range(9):
        moving.set_position((0.2 + step * 0.0011, 0.45 - step * 0.0007))
        stock = pixels(fig)
        with glyph_cache():
            assert pixels(fig) == stock, step
    for dpi in (72, 150):
        stock = pixels(fig, dpi=dpi)
        with glyph_cache():
            assert pixels(fig, dpi=dpi) == stock, dpi


def test_block_restores_stock_drawing_and_is_reentrant():
    stock = backend_agg.RendererAgg.draw_text
    with glyph_cache():
        with glyph_cache():
            assert backend_agg.RendererAgg.draw_text is not stock
        assert backend_agg.RendererAgg.draw_text is not stock
    assert backend_agg.RendererAgg.draw_text is stock
    with pytest.raises(RuntimeError), glyph_cache():
        raise RuntimeError
    assert backend_agg.RendererAgg.draw_text is stock


def test_a_failed_self_test_keeps_stock_drawing(monkeypatch):
    monkeypatch.setattr(_glyph_cache, "_usable", None)
    monkeypatch.setattr(_glyph_cache, "_self_test", lambda: False)
    stock = backend_agg.RendererAgg.draw_text
    with glyph_cache():
        assert backend_agg.RendererAgg.draw_text is stock


def test_self_test_ignores_a_tex_setting(monkeypatch):
    import matplotlib

    monkeypatch.setitem(matplotlib.rcParams, "text.usetex", True)
    assert _glyph_cache._self_test()


def test_a_foreign_draw_text_hook_is_left_in_place(monkeypatch):
    def hook(self, *args, **kwargs):
        return _glyph_cache._STOCK(self, *args, **kwargs)

    monkeypatch.setattr(backend_agg.RendererAgg, "draw_text", hook)
    with glyph_cache():
        assert backend_agg.RendererAgg.draw_text is hook
    assert backend_agg.RendererAgg.draw_text is hook


def test_layouts_and_glyph_bytes_stay_bounded(monkeypatch):
    monkeypatch.setattr(_glyph_cache, "_MAX_LAYOUTS", 50)
    monkeypatch.setattr(_glyph_cache, "_MAX_GLYPH_BYTES", 20_000)
    fig = Figure(figsize=(2, 1), dpi=100)
    backend_agg.FigureCanvasAgg(fig)
    label = fig.text(0.1, 0.5, "", size=11)
    with glyph_cache():
        for n in range(400):
            label.set_text(f"{n:06d} at size {9 + n % 7}")
            label.set_size(9 + n % 7)
            fig.canvas.draw()
            assert len(_glyph_cache._layouts) <= 50
            assert _glyph_cache._glyph_bytes <= 20_000
