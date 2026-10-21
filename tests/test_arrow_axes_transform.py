"""A scaled arrow growth, a staccato axis build, and unmatched glyphs that drift."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece.attention import _GhostWrite


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _ink(pixels, face):
    back = np.round(np.array(face[:3]) * 255)
    return int(np.count_nonzero(np.any(pixels[..., :3] != back, axis=-1)))


# Arrow


def _board():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100, facecolor="white")
    ax.set(xlim=(0, 4), ylim=(0, 3))
    ax.set_axis_off()
    return fig, ax


def _display(ax, patch):
    """The drawn vertices in pixels, without a closing placeholder."""
    path = patch.get_path()
    keep = path.codes != Path.CLOSEPOLY
    return ax.transData.transform(path.vertices[keep])


@pytest.mark.parametrize("kind", ["vector", "double", "dimension"])
def test_scaled_growth_is_the_drawn_arrow_scaled_about_its_origin(kind):
    fig, ax = _board()
    tail, head = (0.5, 0.5), (3.5, 2.0)
    grown = ep.arrow(tail, head, ax=ax, kind=kind, lw=3.0, grow="scaled")
    patch = grown.artists["arrow"]
    fig.canvas.draw()
    rest = _display(ax, patch)
    a, b = ax.transData.transform([tail, head])
    origin = a if kind == "vector" else 0.5 * (a + b)
    for u in (0.2, 0.5, 0.8):
        grown.update(u)
        fig.canvas.draw()
        assert patch.get_linewidth() == pytest.approx(3.0 * u)
        np.testing.assert_allclose(
            _display(ax, patch) - origin, u * (rest - origin), atol=1e-6
        )


def test_scaled_growth_rests_on_the_arrow_as_made():
    fig, ax = _board()
    still = _pixels(fig)
    plain = ep.arrow((0.5, 0.5), (3.5, 2.0), ax=ax, label="v")
    made = _pixels(fig)
    plain.artists["arrow"].remove()
    plain.artists["text"].remove()
    np.testing.assert_array_equal(_pixels(fig), still)
    grown = ep.arrow((0.5, 0.5), (3.5, 2.0), ax=ax, label="v", grow="scaled")
    np.testing.assert_array_equal(_pixels(fig), made)
    for u in np.linspace(0.0, 1.0, 13):
        grown.update(u)
        _pixels(fig)
    np.testing.assert_array_equal(_pixels(fig), made)
    grown.update(0.0)
    np.testing.assert_array_equal(_pixels(fig), still)


def test_length_growth_is_unchanged_by_default():
    _, ax = _board()
    plain = ep.arrow((0.5, 0.5), (3.5, 2.0), ax=ax, lw=3.0)
    patch = plain.artists["arrow"]
    nominal = FancyArrowPatch.get_mutation_scale(patch)
    plain.update(0.4)
    assert patch.get_linewidth() == 3.0
    assert FancyArrowPatch.get_mutation_scale(patch) == nominal
    with pytest.raises(ValueError, match="grow"):
        ep.arrow((0, 0), (1, 1), ax=ax, grow="tips")


def test_scaled_arrow_clip_has_no_motion_fault():
    fig, ax = _board()
    grown = ep.arrow((0.5, 0.5), (3.5, 2.0), ax=ax, label="v", grow="scaled")
    frames = np.linspace(0.0, 1.0, 37)
    assert ept.motion_faults(fig, grown.update, frames, moves=[grown]) == []


# Axis build


def _axes(log=False):
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100, layout="constrained")
    ax.set(xlim=(0.0, 1.0), ylim=(0.5, 2.0), xlabel="albedo", ylabel="radius")
    if log:
        ax.set_yscale("log")
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, ax


@pytest.mark.parametrize("axis", ["x", "y", "both"])
@pytest.mark.parametrize("log", [False, True])
def test_staccato_build_starts_bare_and_rests_on_the_axes_as_made(axis, log):
    fig, ax = _axes(log)
    plain = _pixels(fig)
    build = ep.build_axes(ax, axis=axis, style="staccato")
    np.testing.assert_array_equal(_pixels(fig), plain)
    for u in np.linspace(0.0, 1.0, 41):
        build.update(u)
        _pixels(fig)
    np.testing.assert_array_equal(_pixels(fig), plain)
    build.update(0.5)
    build.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), plain)


def _levels(parts):
    return [
        1.0 if p.artist.get_alpha() is None else p.artist.get_alpha() for p in parts
    ]


def test_staccato_is_one_beat_at_a_time_in_order_along_the_axis():
    _, ax = _axes()
    build = ep.build_axes(ax, axis="x", style="staccato", spine_end=0.7)
    beats = ax.xaxis  # the build reads the axis' own ticks
    ticks = [t for t in beats.get_major_ticks() if 0.0 <= t.get_loc() <= 1.0]
    first = {}
    for u in np.linspace(0.0, 0.7, 701):
        build.update(u)
        partial = [
            a
            for t in ticks
            for a in (t.tick1line, t.label1)
            if a.get_alpha() is not None and 0.0 < a.get_alpha() < 1.0
        ]
        assert len(partial) <= 1
        for t in ticks:
            if t.tick1line.get_alpha() != 0.0 and t not in first:
                first[t] = u
    starts = [first[t] for t in sorted(ticks, key=lambda t: t.get_loc())]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_staccato_draws_the_whole_spine_before_the_first_tick():
    fig, ax = _axes()
    plain = _pixels(fig)
    face = fig.get_facecolor()
    (x0, y0), (x1, _) = ax.transAxes.transform([[0, 0], [1, 1]])
    row = round(fig.bbox.height - y0)
    cols = slice(int(x0) + 2, int(x1) - 2)
    build = ep.build_axes(ax, axis="x", style="staccato")
    tick = ax.xaxis.get_major_ticks()[0].tick1line
    for u in np.linspace(0.0, 0.7, 141):
        build.update(u)
        if tick.get_alpha() != 0.0:
            break
    assert 0.0 < u < 0.7
    # Every pixel the finished spine inks is inked; the ticks crossing it
    # are still fading, so they may shade it.
    spine = _pixels(fig)[row - 1 : row + 2, cols]
    assert _ink(spine, face) == _ink(plain[row - 1 : row + 2, cols], face) > 0


@pytest.mark.parametrize("which", ["x", "y"])
def test_staccato_writes_the_axis_label_after_the_numbers(which):
    fig, ax = _axes()
    build = ep.build_axes(ax, axis=which, style="staccato", spine_end=0.6)
    label = getattr(ax, f"{which}axis").label
    face = fig.get_facecolor()
    box = label.get_window_extent()
    rows = slice(int(fig.bbox.height - box.y1), int(fig.bbox.height - box.y0) + 1)
    cols = slice(int(box.x0), int(box.x1) + 1)

    def label_ink(u):
        build.update(u)
        return _ink(_pixels(fig)[rows, cols], face)

    assert label_ink(0.6) == 0
    assert 0 < label_ink(0.75)
    ghosts = [a for a in fig.get_children() if isinstance(a, _GhostWrite)]
    assert len(ghosts) == 1 and ghosts[0].get_visible()
    assert not label.get_visible()
    build.update(1.0)
    assert label.get_visible() and not ghosts[0].get_visible()


def test_staccato_without_a_label_writes_nothing_and_style_is_checked():
    fig, ax = _axes()
    ax.set_xlabel("")
    plain = _pixels(fig)
    build = ep.build_axes(ax, axis="x", style="staccato")
    build.update(0.9)
    build.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), plain)
    with pytest.raises(ValueError, match="style"):
        ep.build_axes(ax, style="legato")


@pytest.mark.parametrize("seconds", [1.0, 3.0])
def test_staccato_clip_has_no_motion_fault(seconds):
    # An equal beat for the spine swept it across in a frame or two (a jump).
    fig, ax = _axes()
    build = ep.build_axes(ax, style="staccato")
    faults = ept.motion_faults(
        fig, lambda t: build.update(t / seconds), ept.Window(0, seconds), moves=[build]
    )
    assert faults == []


# Transform


def _pair():
    fig, ax = plt.subplots(figsize=(5, 2), dpi=100, facecolor="white")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    at = ep.Anchor.from_data(ax, (0.5, 0.5))
    a = ep.equation(at, r"${{l:x}}+{{r:y}}-c$", fontsize=28)
    b = ep.equation(at, r"${{l:x}}+{{r:y}}=d^2$", fontsize=28)
    fig.canvas.draw()
    return fig, a, b


def _centers(glyphs):
    return np.array([g.get_path().vertices.mean(axis=0) for g in glyphs])


def test_unmatched_glyphs_drift_toward_their_counterpart_while_fading():
    fig, a, b = _pair()
    move = ep.transform(a, b, fade_toward=True)
    assert move.gone and move.new and not move.replaced
    leaving = [a.artists["glyphs"][i] for i in move.gone]
    arriving = [b.artists["glyphs"][j] for j in move.new]
    move.update(1.0)
    fig.canvas.draw()
    end = _centers(arriving)
    # In motion a rule carries its rest snap as a fixed offset, so shifts are
    # measured from the first moving frame.
    move.update(0.05)
    fig.canvas.draw()
    start = _centers(leaving)
    toward = end.mean(axis=0) - start.mean(axis=0)
    shifts, alphas = [], []
    for u in np.linspace(0.1, 0.95, 18):
        move.update(u)
        fig.canvas.draw()
        here = _centers(leaving)
        shift = here - start
        np.testing.assert_allclose(shift, shift[:1].repeat(len(shift), 0), atol=1e-6)
        shifts.append(float(shift[0] @ toward))
        alphas.append((leaving[0].get_alpha(), arriving[0].get_alpha()))
    assert np.all(np.diff(shifts) > 0.0)
    out, into = np.array(alphas).T
    assert np.all(np.diff(out) < 0.0) and np.all(np.diff(into) > 0.0)


def test_drift_runs_with_the_motion_not_after_it():
    _, a, b = _pair()
    plain = ep.transform(a, b)
    plain.update(0.3)
    new = b.artists["glyphs"][plain.new[0]]
    assert not new.get_visible()
    _, a2, b2 = _pair()
    drift = ep.transform(a2, b2, fade_toward=True)
    drift.update(0.3)
    assert b2.artists["glyphs"][drift.new[0]].get_visible()


def test_drift_rests_where_the_plain_transform_rests():
    fig, a, b = _pair()
    plain = ep.transform(a, b)
    plain_rest = _pixels(fig)
    plain.update(0.0)
    eq_a = _pixels(fig)
    fig2, a2, b2 = _pair()
    drift = ep.transform(a2, b2, fade_toward=True)
    np.testing.assert_array_equal(_pixels(fig2), plain_rest)
    drift.update(0.0)
    np.testing.assert_array_equal(_pixels(fig2), eq_a)
    drift.update(0.5)
    drift.update(1.0)
    np.testing.assert_array_equal(_pixels(fig2), plain_rest)


def test_drift_clip_has_no_motion_fault_and_declares_the_drifters():
    fig, a, b = _pair()
    move = ep.transform(a, b, fade_toward=True)
    leaving = {a.artists["glyphs"][i] for i in move.gone}
    assert leaving <= set(move.contract.moves)
    assert not leaving & set(move.contract.rest)
    frames = np.linspace(0.0, 1.0, 61)
    assert ept.motion_faults(fig, move.update, frames, moves=[move]) == []


@pytest.mark.parametrize("fade_toward", [False, True])
def test_stagger_one_moves_every_group_on_one_curve(fade_toward):
    fig, a, b = _pair()
    move = ep.transform(a, b, stagger=1.0, fade_toward=fade_toward)
    sources = a.artists["glyphs"]
    for u in np.linspace(0.05, 0.95, 10):
        move.update(u)
        es = {round(sources[i].pose.e, 12) for i, _ in move.pairs}
        assert len(es) == 1
    frames = np.linspace(0.0, 1.0, 61)
    assert ept.motion_faults(fig, move.update, frames, moves=[move]) == []
