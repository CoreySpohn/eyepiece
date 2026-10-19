"""Write lays ink down glyph by glyph: a thin pen traces, then each glyph fills."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept

STRING = "Write the light down"


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _text_fig(string=STRING, size=30):
    fig = plt.figure(figsize=(6.0, 1.6), dpi=100, facecolor="white")
    text = fig.text(0.04, 0.35, string, fontsize=size, color="black")
    fig.canvas.draw()
    return fig, text


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)


def test_progress_is_linear_inside_each_glyph_window():
    _, text = _text_fig()
    ghost = ep.write(text).artists["fill"]
    n = len(ghost.glyphs())
    us = np.linspace(0.0, 0.9, 901)
    traced = np.array([ghost.times(u)[0] for u in us])
    fill = np.array([ghost.times(u)[1] for u in us])
    for i in (0, n // 2, n - 1):
        for series in (traced[:, i], fill[:, i]):
            moving = (series > 0.0) & (series < 1.0)
            slope = np.diff(series)[moving[1:] & moving[:-1]]
            assert slope.size > 10
            np.testing.assert_allclose(slope, slope[0], rtol=1e-6)


def test_a_glyph_traces_first_then_fills():
    _, text = _text_fig()
    ghost = ep.write(text).artists["fill"]
    for u in np.linspace(0.0, 1.0, 101):
        traced, fill = ghost.times(u)
        for t, f in zip(traced, fill, strict=True):
            assert f == 0.0 or t == 1.0
    assert ghost.times(1.0)[1] == [1.0] * len(ghost.glyphs())


def test_a_short_front_of_glyphs_is_in_flux():
    _, text = _text_fig()
    ghost = ep.write(text).artists["fill"]
    n = len(ghost.glyphs())
    assert n >= 15
    busy = []
    for u in np.linspace(0.0, 0.9, 181):
        traced, fill = ghost.times(u)
        busy.append(sum(t > 0.0 and f < 1.0 for t, f in zip(traced, fill, strict=True)))
    assert 4 <= max(busy) <= 7


def test_glyphs_finish_one_by_one_in_reading_order():
    fig, text = _text_fig()
    final = _pixels(fig)
    move = ep.write(text)
    ghost = move.artists["fill"]
    glyphs = ghost.glyphs()
    move.update(0.45)
    mid = _pixels(fig)
    height = mid.shape[0]

    def ink(pixels, glyph):
        v = np.vstack([p for p, _ in glyph])
        x0, x1 = int(v[:, 0].min()), int(np.ceil(v[:, 0].max()))
        y0, y1 = height - int(np.ceil(v[:, 1].max())), height - int(v[:, 1].min())
        return np.count_nonzero(pixels[y0:y1, x0:x1].min(axis=-1) < 128)

    first, last = glyphs[0], glyphs[-1]
    assert ink(mid, first) >= 0.9 * ink(final, first)
    assert ink(mid, last) == 0


def test_the_pen_is_thin_by_default():
    fig, text = _text_fig()
    move = ep.write(text)
    move.update(0.2)
    fig.canvas.draw()
    assert move.artists["fill"]._stroke.get_linewidth() <= 0.75


def test_handoff_never_lightens_the_finished_ink():
    fig, text = _text_fig()
    final = _pixels(fig)
    ink = final.min(axis=-1) < 40
    move = ep.write(text)
    move.update(0.9)
    start = _pixels(fig)
    # The filled glyphs cross-fade into the text: no hand-off frame is lighter
    # than both its ends, so the finished ink never flickers.
    bound = np.maximum(start, final).min(axis=-1)[ink] + 24
    for u in np.linspace(0.9, 1.0, 11):
        move.update(u)
        frame = _pixels(fig)
        assert np.all(frame.min(axis=-1)[ink] <= bound)
    np.testing.assert_array_equal(_pixels(fig), final)


def test_write_clip_has_no_motion_fault():
    fig, text = _text_fig()
    move = ep.write(text)
    assert (
        ept.motion_faults(fig, move.update, np.linspace(0, 1, 61), moves=[move]) == []
    )


def _equation_fig():
    fig, ax = plt.subplots(figsize=(6.0, 2.0), dpi=100, facecolor="white")
    ax.set_axis_off()
    eq = ep.equation(ax, r"${{lhs:C}} = \frac{{num:F_p}}{{den:F_s}}$", fontsize=34)
    fig.canvas.draw()
    return fig, eq


def test_equation_write_rests_on_the_equation_as_made():
    fig, eq = _equation_fig()
    rest = _pixels(fig)
    for u in (0.0, 0.3, 0.7):
        eq.update(u, reveal="write")
        _pixels(fig)
    eq.update(0.0, reveal="write")
    assert np.all(_pixels(fig) == 255)
    eq.update(1.0, reveal="write")
    np.testing.assert_array_equal(_pixels(fig), rest)
    assert all(g.get_alpha() is None for g in eq.artists["glyphs"])
    eq.update(0.5, reveal="write")
    eq.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), rest)


def test_equation_write_traces_with_a_pen_before_filling():
    fig, eq = _equation_fig()
    eq.update(0.0, reveal="write")
    blank = _pixels(fig)
    eq.update(0.08, reveal="write")
    traced = _pixels(fig)
    first = min(eq.artists["glyphs"], key=lambda g: g.piece.box[0])
    assert not first.get_visible()  # no fill yet
    assert np.any(traced != blank)  # but the pen is drawing


def test_equation_write_clip_has_no_motion_fault_and_bad_reveal_raises():
    fig, eq = _equation_fig()

    def at(u):
        eq.update(u, reveal="write")

    assert ept.motion_faults(fig, at, np.linspace(0, 1, 61), moves=[eq]) == []
    with pytest.raises(ValueError, match="reveal"):
        eq.update(0.5, reveal="type")
