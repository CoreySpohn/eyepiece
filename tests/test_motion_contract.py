"""Built-in moves declare the geometry that their real artists draw."""

import dataclasses
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import rcParams
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Affine2D

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _board():
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_axis_off()
    return fig, ax


def _transform(ax):
    at = ep.Anchor.from_data(ax, (0.15, 0.5))
    a = ep.equation(
        at, r"${{lhs:ab}}=\frac{{num:xy}}{{den:z}}$", fontsize=23, ha="left"
    )
    b = ep.equation(
        ep.Anchor.from_data(ax, (0.18, 0.52)),
        r"${{lhs:ab}}=\frac{{num:xy}}{{den:zz}}+1$",
        fontsize=23,
        ha="left",
    )
    return ep.transform(a, b, align=None), a, b


def _clip(name):
    fig, ax = _board()

    def anchor(xy):
        return ep.Anchor.from_data(ax, xy)

    if name == "write":
        result = ep.write(ax.text(0.3, 0.5, "motion", fontsize=22))
    elif name == "draw_on":
        (line,) = ax.plot([0.2, 0.5, 0.8], [0.3, 0.5, 0.6])
        result = ep.draw_on(line)
    elif name == "head_draw":
        (line,) = ax.plot([0.2, 0.5, 0.8], [0.3, 0.5, 0.6])
        result = ep.head_draw(line)
    elif name == "ruler_readout":
        (line,) = ax.plot([0.2, 0.5, 0.8], [0.3, 0.5, 0.6])
        result = ep.ruler_readout(
            ep.head_draw(line),
            readout=ep.readout(anchor((0.1, 0.9)), 0.0, fmt=".2f"),
            ruler=ep.ruler(ax, (0.1, 0.1), (0.1, 0.1), "x", head_scale=0),
            ruler_to=lambda x, y: (x, 0.1),
        )
    elif name == "click":
        (line,) = ax.plot([0.2, 0.5, 0.8], [0.3, 0.5, 0.6])
        result = ep.click(
            line,
            dot=anchor((0.8, 0.6)),
            readout=ep.readout(anchor((0.1, 0.9)), 0.42, fmt=".2f"),
            ruler=ep.ruler(ax, (0.1, 0.1), (0.6, 0.1), "x", head_scale=0),
        )
    elif name == "morph":
        (a,) = ax.plot([0.2, 0.4, 0.5], [0.3, 0.6, 0.3])
        (b,) = ax.plot([0.3, 0.5, 0.7], [0.3, 0.7, 0.3])
        result = ep.morph(a, b)
    elif name == "flight":
        (dot,) = ax.plot([0.2], [0.4], "o")
        result = ep.flight(dot, anchor((0.7, 0.6)), fig=fig)
    elif name == "collapse":
        segments = np.array([[[0.2, 0.4], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.6]]])
        result = ep.collapse(
            ax,
            segments,
            [1, 2],
            anchor((0.3, 0.5)),
            anchor((0.7, 0.5)),
            fig=fig,
            colors=["C0", "C1"],
        )
        # Like the gallery, the caller draws the cells and the landed bar.
        kw = {
            "colors": ["C0", "C1"],
            "linewidths": 2 * rcParams["lines.linewidth"],
            "antialiaseds": False,
            "capstyle": "butt",
        }
        cells = ax.add_collection(LineCollection(segments, **kw))
        landed = ax.add_collection(
            LineCollection(
                [
                    [(0.3, 0.5), (0.3 + 0.4 / 3, 0.5)],
                    [(0.3 + 0.4 / 3, 0.5), (0.7, 0.5)],
                ],
                **kw,
            )
        )

        def at(t):
            u = t / 2
            # The caller's cells stay drawn under the copy, which fades in and out
            # over them; swapping in the landed bar is the caller's own beat.
            cells.set_visible(True)
            landed.set_visible(False)
            result.update(u)

        return fig, result, at
    elif name == "warp":
        (line,) = ax.plot([0.2, 0.4], [0.3, 0.6])
        result = ep.warp(line, [[1, 0.2], [0, 1]])
    elif name == "arrow":
        result = ep.arrow((0.2, 0.3), (0.7, 0.6), ax=ax, label="v")
    elif name == "angle_arc":
        result = ep.angle_arc(
            (0.4, 0.4), 0, 90, 0.25, ax=ax, label="a", label_scale=1.3
        )
    elif name == "compass":
        result = ep.compass((0.5, 0.3), 0.3, ax=ax, east="left")
    elif name == "follow_zoom":
        ax.plot([0.2, 0.4, 0.6], [0.2, 0.4, 0.3])
        result = ep.follow_zoom(
            ax,
            lambda t: (0.25 + 0.06 * t, 0.25 + 0.03 * t),
            size=0.3,
            strict=False,
        )
        return fig, result, lambda t: result.update(t / 2, time=t)
    elif name == "words":
        result = ep.words(anchor((0.15, 0.5)), "one small move", fontsize=16)
    elif name == "readout":
        result = ep.readout(anchor((0.2, 0.5)), 0.0, prefix="t = ", unit="s")
        result.set(3.2)
    elif name == "Stage":
        (line,) = ax.plot([0.2, 0.7], [0.3, 0.6])
        result = ep.Stage({"line": line})
        return fig, result, lambda t: result.set(shown=[], u=t / 2)
    else:
        result, _, _ = _transform(ax)
    return fig, result, lambda t: result.update(t / 2)


def _members(contract):
    yield from contract.moves
    yield from contract.rest
    yield from contract.arrives_after_motion
    yield from contract.rectilinear
    for group in contract.groups:
        yield from group["artists"] if isinstance(group, dict) else group
    for pair in contract.may_overlap:
        yield from pair


@pytest.mark.parametrize(
    "name",
    [
        "write",
        "draw_on",
        "head_draw",
        "ruler_readout",
        "click",
        "morph",
        "flight",
        "collapse",
        "warp",
        "arrow",
        "angle_arc",
        "compass",
        "follow_zoom",
        "words",
        "readout",
        "Stage",
        "transform",
    ],
)
def test_moves_declare_real_artists_and_lint_cleanly(name):
    fig, result, at = _clip(name)
    assert isinstance(result.contract, ep.MotionContract)
    members = set(_members(result.contract))
    # A write declares nothing: its real text is hidden until the end.
    assert members or name == "write"
    seen = set()
    for t in np.linspace(0, 2, 61):
        at(t)
        fig.canvas.draw()
        for artist in members:
            assert isinstance(artist, Artist)
            assert artist.get_figure(root=True) is fig
            assert artist in fig.findobj()
            if artist.get_visible() and artist.get_alpha() != 0:
                if not isinstance(artist, Text) or artist.get_text():
                    seen.add(artist)
    assert seen == members
    assert ept.motion_faults(fig, at, ept.Span(0, 2), moves=[result]) == []


def _centers(group):
    return np.array([g.get_window_extent().get_points().mean(axis=0) for g in group])


def test_transform_rules_stay_rectilinear_and_matched_words_stay_rigid():
    fig, ax = _board()
    move, a, b = _transform(ax)
    assert set(move.contract.rectilinear) == {
        g for eq in (a, b) for g in eq.artists["glyphs"] if g.piece.is_rule
    }
    word = tuple(a.term("lhs")._pieces)
    assert word in move.contract.groups
    move.update(0)
    fig.canvas.draw()
    sources = a.artists["glyphs"]
    partners = {sources[i]: b.artists["glyphs"][j] for i, j in move.pairs}
    spacings = []
    for group in move.contract.groups:
        centers = _centers(group)
        spacings.append(centers[:, None, :] - centers[None, :, :])
    for u in np.linspace(0, 1, 61):
        move.update(u)
        fig.canvas.draw()
        for rule in move.contract.rectilinear:
            if not rule.get_visible():
                continue
            path = rule.get_path()
            assert set(path.codes) <= {Path.MOVETO, Path.LINETO, Path.CLOSEPOLY}
            steps = np.diff(path.vertices, axis=0)
            assert np.all(np.isclose(steps[:, 0], 0) | np.isclose(steps[:, 1], 0))
        for group, spacing in zip(move.contract.groups, spacings, strict=True):
            shown = group if u < 1 else tuple(partners[g] for g in group)
            centers = _centers(shown)
            np.testing.assert_allclose(
                centers[:, None, :] - centers[None, :, :],
                spacing,
                atol=0.5,
            )
    assert move.contract.arrives_after_motion == ()


def test_motion_contract_is_frozen_defaults_and_moves_alias():
    contract = ep.MotionContract()
    assert contract.settles is True
    assert list(_members(contract)) == []
    with pytest.raises(FrozenInstanceError):
        contract.moves = ()
    with pytest.raises(TypeError):
        ep.MotionContract(settles=False)
    _, ax = _board()
    (line,) = ax.plot([0.2, 0.4], [0.3, 0.5])
    result = ep.PlotResult(
        ax, {"line": line}, contract=ep.MotionContract(moves=(line,))
    )
    assert ept._declared(result, "artists") == (line,)
    assert ept._declared({"moves": (line,)}, "artists") == (line,)
    assert ept._declared(SimpleNamespace(contract={"moves": (line,)}), "artists") == (
        line,
    )
    assert ept._declared({"artists": (), "moves": (line,)}, "artists") == ()
    assert ept._declared(ep.PlotResult(ax, {}), "artists", ()) == ()
    assert ep.MosaicResult(np.array([ax]), {}).contract is None


def test_transform_does_not_promise_rigidity_when_matched_spacing_changes():
    fig, ax = _board()
    at = ep.Anchor.from_data(ax, (0.3, 0.5))
    a = ep.equation(at, r"${{word:ab}}$", fontsize=24)
    b = ep.equation(at, r"${{word:a+b}}$", fontsize=24)
    move = ep.transform(a, b)
    word = set(a.term("word")._pieces)
    for group in move.contract.groups:
        members = group["artists"] if isinstance(group, dict) else group
        assert not word <= set(members)
    assert (
        ept.motion_faults(
            fig, lambda t: move.update(t / 2), ept.Span(0, 2), moves=[move]
        )
        == []
    )


@pytest.mark.parametrize("image", [False, True])
def test_stage_of_an_axes_declares_drawn_members(image):
    fig, ax = _board()
    (line,) = ax.plot([0.2, 0.7], [0.3, 0.6])
    label = ax.text(0.2, 0.7, "steady")
    raster = (
        ax.imshow(np.arange(4).reshape(2, 2), extent=(0.2, 0.7, 0.2, 0.7))
        if image
        else None
    )
    stage = ep.Stage({"panel": ax})
    if raster is not None:
        assert raster in stage.contract.rest
    assert line in stage.contract.rest
    assert label in stage.contract.rest
    assert all(not isinstance(a, Text) or a.get_text() for a in stage.contract.rest)
    assert (
        ept.motion_faults(
            fig, lambda t: stage.set(shown=[], u=t / 2), ept.Span(0, 2), moves=[stage]
        )
        == []
    )


def test_stage_accepts_per_point_opacity_without_changing_it():
    _, ax = _board()
    points = ax.scatter([0.2, 0.4], [0.3, 0.6], alpha=np.array([0.4, 0.9]))
    stage = ep.Stage({"points": points})
    assert points in stage.contract.rest
    stage.rest()
    np.testing.assert_array_equal(points.get_alpha(), [0.4, 0.9])


def test_words_contract_covers_both_public_reveal_modes():
    fig, ax = _board()
    result = ep.words(ep.Anchor.from_data(ax, (0.2, 0.5)), "many letters")
    assert (
        ept.motion_faults(
            fig,
            lambda t: result.reveal(t / 2, by="letter"),
            ept.Span(0, 2),
            moves=[result],
        )
        == []
    )


def test_result_subclasses_keep_positional_arguments():
    from eyepiece._equation import _EquationResult, _TransformResult
    from eyepiece.attention import _ReadoutResult, _WordsResult

    _, ax = _board()

    def fn():
        pass

    assert _WordsResult(ax, {}, None, (), fn).reveal is fn
    assert _ReadoutResult(ax, {}, None, (), fn).set is fn
    assert _EquationResult(ax, {}, None, (), fn, "layout").term is fn
    assert _TransformResult(ax, {}, None, (), ((0, 1),)).pairs == ((0, 1),)


def test_flight_contract_resolves_drawn_leaves_and_excludes_prototypes():
    from matplotlib.lines import Line2D

    fig, ax = _board()
    start = ep.Anchor.from_data(ax, (0.2, 0.3))
    eq = ep.equation(ep.Anchor.from_data(ax, (0.6, 0.5)), r"${{x:ab}}$", fontsize=24)
    prototype = Line2D([0], [0], marker="o")
    move = ep.flight(start, eq.term("x"), fig=fig, mark=prototype)
    assert prototype not in set(_members(move.contract))
    assert eq.term("x") not in set(_members(move.contract))
    assert {pair[1] for pair in move.contract.may_overlap} == set(eq.term("x")._pieces)
    assert all(a.get_figure(root=True) is fig for a in _members(move.contract))


# Negative controls: a promise that is broken must be reported.


def _shifted(at, artists, px_per_s):
    """A clip that also slides `artists` sideways at a steady speed."""
    originals = {a: a.get_transform() for a in artists}

    def pose(t):
        at(t)
        for artist, base in originals.items():
            artist.set_transform(base + Affine2D().translate(px_per_s * t, 0.0))

    return pose


@pytest.mark.parametrize("name", ["follow_zoom", "readout", "Stage", "transform"])
def test_a_rest_artist_that_moves_is_reported_as_drift(name):
    fig, result, at = _clip(name)
    assert result.contract.rest
    pose = _shifted(at, [result.contract.rest[0]], 30.0)
    found = ept.motion_faults(fig, pose, ept.Span(0, 2), moves=[result])
    assert "drift" in {f.kind for f in found}


@pytest.mark.parametrize("name", ["angle_arc", "compass", "follow_zoom"])
def test_dropping_the_declared_overlap_is_reported(name):
    fig, result, at = _clip(name)
    assert result.contract.may_overlap
    bare = dataclasses.replace(result.contract, may_overlap=())
    found = ept.motion_faults(fig, at, ept.Span(0, 2), moves=[bare])
    assert "overlap" in {f.kind for f in found}


def test_a_glyph_pulled_out_of_a_rigid_group_is_reported_as_a_tear():
    fig, ax = _board()
    move, _a, _b = _transform(ax)
    group = move.contract.groups[0]
    pose = _shifted(lambda t: move.update(t / 2), [group[0]], 20.0)
    found = ept.motion_faults(fig, pose, ept.Span(0, 2), moves=[move])
    assert "tear" in {f.kind for f in found}


def test_a_rule_inside_a_replaced_term_is_not_declared_rectilinear():
    _fig, ax = _board()
    at = ep.Anchor.from_data(ax, (0.15, 0.5))
    a = ep.equation(at, r"${{t:xy}}=\frac{{n:p}}{{d:q}}$", fontsize=23, ha="left")
    b = ep.equation(
        at, r"${{t:\frac{r}{s}}}=\frac{{n:p}}{{d:q}}$", fontsize=23, ha="left"
    )
    move = ep.transform(a, b, align=None)
    rules_b = [g for g in b.artists["glyphs"] if g.piece.is_rule]
    term = set(b.term("t")._pieces)
    inside = [g for g in rules_b if g in term]
    assert inside, "the replaced term holds a rule"
    assert not set(inside) & set(move.contract.rectilinear)
    kept = [g for g in a.artists["glyphs"] if g.piece.is_rule]
    assert kept and set(kept) <= set(move.contract.rectilinear)
