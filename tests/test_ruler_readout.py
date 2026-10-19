"""One knob drives the curve head, the ruler and the counter, read off the mark."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept

N_TOTAL = 1_000_000


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _scene(*, final=False):
    fig = plt.figure(figsize=(6.4, 3.2), dpi=100, facecolor="white")
    ax_p = fig.add_axes([0.04, 0.1, 0.4, 0.8])
    ax_c = fig.add_axes([0.55, 0.15, 0.4, 0.7])
    ax_p.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    ax_c.set(xlim=(-0.05, 1.1), ylim=(-0.05 * N_TOTAL, 1.1 * N_TOTAL))
    for a in (ax_p, ax_c):
        a.set_axis_off()
    r = np.linspace(0.0, 1.0, 181)
    (curve,) = ax_c.plot(r, N_TOTAL * (1 - np.exp(-3 * r**2)), color="C0", lw=2)
    end = curve.get_xydata()[-1]
    at = ep.Anchor.from_data(ax_c, (0.0, 1.0 * N_TOTAL))
    start = round(end[1]) if final else 0
    count = ep.readout(at, start, fmt=",d", digits=len(f"{N_TOTAL:,d}") - 2)
    rule = ep.ruler(
        ax_p,
        (0.0, 0.0),
        (-end[0], 0.0) if final else (0.0, 0.0),
        f"r = {end[0]:.2f}" if final else "r = 0.00",
        head_scale=0,
    )
    return fig, ax_c, curve, count, rule


def _bind(curve, count, rule, **kw):
    head = ep.head_draw(curve, **kw)
    knob = ep.ruler_readout(
        head,
        readout=count,
        value=lambda r, n: n,
        ruler=rule,
        ruler_to=lambda r, n: (-r, 0.0),
        ruler_text=lambda r, n: f"r = {r:.2f}",
    )
    return head, knob


def _shown(count):
    return "".join(t.get_text() for t in count.artists["text"])


def test_counter_and_ruler_read_the_drawn_tip_on_every_frame():
    fig, ax_c, curve, count, rule = _scene()
    head, knob = _bind(curve, count, rule)
    copy = head.contract.moves[0]
    for u in np.linspace(0.05, 0.95, 19):
        knob.update(u)
        fig.canvas.draw()
        tip = ax_c.transData.inverted().transform(copy.get_xydata()[-1])
        r, n = head.head()
        np.testing.assert_allclose((r, n), tip, rtol=1e-9, atol=1e-6 * N_TOTAL)
        assert _shown(count) == f"{round(n):,d}"
        xs, ys = rule.artists["lines"][0].get_data()
        assert (xs[1], ys[1]) == pytest.approx((-r, 0.0))
        assert rule.artists["text"].get_text() == f"r = {r:.2f}"


def test_count_never_goes_back_and_its_slots_never_move():
    fig, _, curve, count, rule = _scene()
    _, knob = _bind(curve, count, rule)
    values, slots = [], set()
    for u in np.linspace(0.0, 1.0, 61):
        knob.update(u)
        fig.canvas.draw()
        values.append(int(_shown(count).replace(",", "")))
        slots.add(tuple(t.get_window_extent().x0 for t in count.artists["text"]))
    assert np.all(np.diff(values) >= 0)
    assert len(slots) == 1


def test_rest_equals_the_still_of_the_final_quantities():
    fig, _, curve, count, rule = _scene()
    _, knob = _bind(curve, count, rule)
    for u in (0.3, 0.8):
        knob.update(u)
        fig.canvas.draw()
    knob.update(1.0)
    fig.canvas.draw()
    moved = np.asarray(fig.canvas.buffer_rgba()).copy()
    still_fig = _scene(final=True)[0]
    still_fig.canvas.draw()
    np.testing.assert_array_equal(moved, np.asarray(still_fig.canvas.buffer_rgba()))


def test_level_dims_ruler_and_counter_but_not_the_head():
    _, _, curve, count, rule = _scene()
    head, knob = _bind(curve, count, rule)
    knob.update(0.5, level=0.0)
    assert not rule.artists["arrow"].get_visible()
    assert not any(t.get_visible() for t in count.artists["text"])
    assert head.contract.moves[0].get_visible()
    knob.update(0.5)
    assert rule.artists["arrow"].get_visible()
    assert all(t.get_visible() for t in count.artists["text"])


def test_clip_has_no_motion_fault():
    fig, _, curve, count, rule = _scene()
    _, knob = _bind(curve, count, rule)
    frames = np.linspace(0.0, 1.0, 61)
    assert ept.motion_faults(fig, knob.update, frames, moves=[knob]) == []


def test_needs_a_target_and_a_ruler_end():
    _, _, curve, _, rule = _scene()
    head = ep.head_draw(curve)
    with pytest.raises(ValueError, match="readout, a ruler"):
        ep.ruler_readout(head)
    with pytest.raises(ValueError, match="ruler_to"):
        ep.ruler_readout(head, ruler=rule)
