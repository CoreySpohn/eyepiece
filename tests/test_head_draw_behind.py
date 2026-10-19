"""Lines drawn behind a head-drawn path follow its body and never lead the head."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _board():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100, facecolor="white")
    ax.set(xlim=(0, 1), ylim=(0, 1), facecolor="white")
    ax.set_axis_off()
    return fig, ax


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)


def _halo_right(fig):
    """The rightmost display column holding red halo ink, or -inf for none."""
    px = _pixels(fig)
    red = (px[..., 0] > 200) & (px[..., 1] < 90) & (px[..., 2] < 90)
    cols = np.nonzero(red.any(axis=0))[0]
    return float(cols.max()) if len(cols) else -np.inf


def _tip_x(fig, move):
    """The head's tip in display x, read where the head dot was drawn."""
    fig.canvas.draw()
    return float(move.contract.moves[0].get_xydata()[-1, 0])


def _bar(ax):
    """A blue line over a wide red halo, left to right across the axes."""
    (halo,) = ax.plot([0.1, 0.9], [0.5, 0.5], color="red", lw=14)
    (line,) = ax.plot([0.1, 0.9], [0.5, 0.5], color="C0", lw=2)
    return line, halo


def test_halo_ends_with_the_body_while_the_head_shows():
    fig, ax = _board()
    line, halo = _bar(ax)
    move = ep.head_draw(line, behind={line: halo}, ease="linear")
    for u in (0.2, 0.5, 0.8):
        move.update(u)
        assert _halo_right(fig) <= _tip_x(fig, move) + 0.5


def test_a_plain_draw_on_halo_leads_the_head():
    # The defect `behind` fixes: a halo revealed beside the path, not behind it,
    # runs to the tip and its cap shows ahead of the head.
    fig, ax = _board()
    line, halo = _bar(ax)
    move = ep.head_draw(line, ease="linear")
    under = ep.draw_on(halo, ease="linear")
    move.update(0.5)
    under.update(0.5)
    assert _halo_right(fig) > _tip_x(fig, move) + 3.0


def test_halo_never_shows_more_past_the_tip_than_in_the_still():
    fig, ax = _board()
    line, halo = _bar(ax)
    end = ax.transData.transform((0.9, 0.5))[0]
    still = _halo_right(fig) - end
    move = ep.head_draw(line, behind=halo)
    for u in np.linspace(0.02, 0.99, 40):
        move.update(u)
        assert _halo_right(fig) - _tip_x(fig, move) <= still + 1.0


def test_rest_is_the_plain_still_and_start_has_no_ink():
    fig, ax = _board()
    line, halo = _bar(ax)
    without = _pixels(fig)
    move = ep.head_draw(line, behind={line: [halo]})
    move.update(0.0)
    assert np.all(_pixels(fig) == 255)
    for u in (0.3, 0.97):
        move.update(u, level=0.5)
        _pixels(fig)
    move.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), without)
    assert halo.get_alpha() is None


def test_ring_with_a_halo_has_no_motion_fault():
    fig, ax = _board()
    th = np.linspace(0, 2 * np.pi, 181)
    x, y = 0.5 + 0.3 * np.cos(th), 0.5 + 0.3 * np.sin(th)
    (halo,) = ax.plot(x, y, color="white", lw=7, alpha=0.9, zorder=1.9)
    (ring,) = ax.plot(x, y, color="C0", lw=2, zorder=2)
    (curve,) = ax.plot([0.1, 0.5, 0.9], [0.1, 0.2, 0.15], color="C1", lw=2)
    move = ep.head_draw([ring, curve], lag=0.4, behind={ring: halo})
    assert set(move.artists["behind"]) >= {halo}
    frames = np.linspace(0.0, 1.0, 61)
    assert ept.motion_faults(fig, move.update, frames, moves=[move]) == []


def test_behind_must_name_drawn_lines():
    _, ax = _board()
    line, halo = _bar(ax)
    (other,) = ax.plot([0, 1], [0, 0])
    with pytest.raises(ValueError, match="not drawn"):
        ep.head_draw(line, behind={other: halo})
    with pytest.raises(ValueError, match="map"):
        ep.head_draw([line, other], behind=halo)
    with pytest.raises(TypeError):
        ep.head_draw(line, behind={line: ax.text(0, 0, "x")})
