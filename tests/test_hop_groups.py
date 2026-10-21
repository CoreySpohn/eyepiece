"""A declared hop group: its frame-to-frame cuts are by design, at a stated rate."""

import matplotlib.pyplot as plt
import pytest

import eyepiece.testing as ept

FPS = 30.0
TIMES = [k / FPS for k in range(61)]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _scene():
    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax = fig.add_axes([0.05, 0.1, 0.4, 0.8])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    side = fig.add_axes([0.55, 0.1, 0.4, 0.8])
    side.set_axis_off()
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    (dot,) = ax.plot([0.5], [0.5], "o", color="C0", ms=14)
    (glider,) = side.plot([0.1], [0.15], "s", color="0.5", ms=6)
    label = side.text(0.1, 0.6, "label", fontsize=12)
    return fig, dot, glider, label


def _poser(dot, glider, label, every, label_jump=None):
    """Hop the dot to a new place every `every` frames."""
    places = [(0.2, 0.3), (0.8, 0.7), (0.3, 0.8), (0.7, 0.2)]

    def at(t):
        k = round(t * FPS)
        dot.set_data([places[(k // every) % 4][0]], [places[(k // every) % 4][1]])
        glider.set_data([0.1 + 0.8 * t / TIMES[-1]], [0.15])
        if label_jump is not None:
            label.set_x(0.6 if k >= label_jump else 0.1)

    return at


def test_hops_at_a_declared_rate_are_clean():
    fig, dot, glider, label = _scene()
    at = _poser(dot, glider, label, every=10)  # 3 draws per second
    hops = [{"artists": [dot], "rate": 3.0}]
    faults = ept.motion_faults(fig, at, TIMES, hops=hops, moves=[{"artists": [glider]}])
    assert faults == []
    # Undeclared, the same cuts are reported.
    blind = ept.motion_faults(fig, at, TIMES, moves=[{"artists": [glider]}])
    assert any(f.kind == "jump" for f in blind)


def test_a_rate_above_eight_per_second_is_a_fault():
    fig, dot, glider, label = _scene()
    at = _poser(dot, glider, label, every=3)  # 10 draws per second
    hops = [{"artists": [dot], "rate": 10.0}]
    faults = ept.motion_faults(fig, at, TIMES, hops=hops, moves=[{"artists": [glider]}])
    rates = [f for f in faults if f.kind == "hop-rate"]
    assert len(rates) > 1
    assert rates[0].artist is dot
    assert rates[0].value == pytest.approx(10.0)


def test_a_declared_rate_that_the_clip_exceeds_is_a_fault():
    fig, dot, glider, label = _scene()
    at = _poser(dot, glider, label, every=3)  # drawn at 10 per second
    hops = [{"artists": [dot], "rate": 4.0}]
    faults = ept.motion_faults(fig, at, TIMES, hops=hops, moves=[{"artists": [glider]}])
    assert len([f for f in faults if f.kind == "hop-rate"]) > 1


def test_another_artist_cutting_with_a_hop_is_still_reported():
    fig, dot, glider, label = _scene()
    at = _poser(dot, glider, label, every=10, label_jump=20)
    hops = [{"artists": [dot], "rate": 3.0}]
    faults = ept.motion_faults(fig, at, TIMES, hops=hops, moves=[{"artists": [glider]}])
    jumps = [f for f in faults if f.kind == "jump"]
    assert len(jumps) == 1
    assert jumps[0].time == pytest.approx(20 / FPS)
    assert jumps[0].box[0] > 300  # the side axes, clear of the hopping dot


def test_a_move_contract_can_name_the_hops():
    fig, dot, glider, label = _scene()
    at = _poser(dot, glider, label, every=10)
    moves = [
        {"artists": [glider]},
        {"hops": [{"artists": [dot], "rate": 3.0}]},
    ]
    assert ept.motion_faults(fig, at, TIMES, moves=moves) == []


def test_a_hop_without_a_rate_is_refused():
    fig, dot, glider, label = _scene()
    at = _poser(dot, glider, label, every=10)
    with pytest.raises(ValueError, match="rate"):
        ept.motion_faults(fig, at, TIMES, hops=[{"artists": [dot]}])


def test_every_excessive_hop_window_is_reported():
    fig, dot, _, _ = _scene()
    times = [k / 10 for k in range(41)]

    def at(t):
        count = sum(t >= cut - 1e-9 for cut in (0.1, 0.2, 0.3, 2.1, 2.2, 2.3))
        dot.set_xdata([0.2 if count % 2 else 0.8])

    faults = ept.motion_faults(
        fig, at, times, hops=[{"artists": [dot], "rate": 1}], kinds=["hop-rate"]
    )
    assert [f.time for f in faults] == pytest.approx([0.1, 2.1])
    assert [f.value for f in faults] == [3, 3]
