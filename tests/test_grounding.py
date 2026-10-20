"""Equation terms share their marks' meaning, color, and cues."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

import eyepiece as ep
from eyepiece.motifs import Motif, MotifSet


@pytest.fixture
def board():
    fig = plt.figure(figsize=(5, 2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    yield fig, ax
    plt.close(fig)


def _equation(ax):
    return ep.equation(ax, r"${{sigma:\sigma_x}}$", fontsize=30)


def _assert_rgb(glyphs, color):
    assert glyphs
    for glyph in glyphs:
        np.testing.assert_allclose(glyph.get_facecolor()[:3], to_rgb(color), atol=1e-6)


@pytest.mark.parametrize("as_key", [False, True], ids=["motif", "key"])
def test_bind_uses_motif_label_ink_and_gid(board, as_key):
    _, ax = board
    # restore_style_state puts the global style back after this test.
    hwostyle.use("dark", palette_family="cyberpunk")
    eq = _equation(ax)
    motif = (
        MotifSet.default()["local_zodi"]
        if as_key
        else Motif(key="spread", noun="spread", role="zodi", label_color="zodi")
    )
    assert to_rgb(motif.resolve_label()) != to_rgb(motif.resolve())
    # A motif can give meaning to any Artist, even one without a color.
    mark = Artist()
    term = eq.bind("sigma", mark, motif=motif.key if as_key else motif)

    assert term is eq.term("sigma")
    _assert_rgb(eq.artists["glyphs"], motif.resolve_label())
    assert all(g.get_gid() == motif.key for g in eq.artists["glyphs"])
    binding = eq.bindings["sigma"]
    assert binding.term is term
    assert binding.mark is mark
    assert binding.motif == motif
    if not as_key:
        assert binding.motif is motif


def test_bind_keeps_current_alpha_and_existing_gids(board):
    _, ax = board
    eq = _equation(ax)
    glyphs = eq.artists["glyphs"]
    glyphs[0].set_gid("existing")
    glyphs[0].set_facecolor((0.1, 0.2, 0.3, 0.4))
    glyphs[1].set_alpha(0.25)
    before = [g.get_facecolor()[3] for g in glyphs]

    eq.bind("sigma", Artist(), motif="planet")

    assert glyphs[0].get_gid() == "existing"
    assert glyphs[1].get_gid() == "planet"
    assert [g.get_facecolor()[3] for g in glyphs] == before


@pytest.mark.parametrize(
    "kind",
    [
        "line",
        "text",
        "patch_face",
        "patch_edge",
        "patch_zero_width",
        "collection_face",
        "collection_edge",
    ],
)
def test_bind_takes_the_marks_color_without_a_motif(board, kind):
    _, ax = board
    eq = _equation(ax)
    if kind == "line":
        (mark,) = ax.plot([0.1, 0.9], [0.2, 0.3], color="C2")
        expected = "C2"
    elif kind == "text":
        mark = ax.text(0.2, 0.2, "spread", color="C3")
        expected = "C3"
    elif kind.startswith("patch"):
        mark = Rectangle(
            (0.1, 0.1),
            0.2,
            0.2,
            facecolor="C4",
            edgecolor="none" if kind == "patch_face" else "C5",
            linewidth=0 if kind == "patch_zero_width" else 2,
        )
        ax.add_patch(mark)
        expected = "C5" if kind == "patch_edge" else "C4"
    elif kind == "collection_face":
        mark = ax.scatter([0.2, 0.4], [0.2, 0.4], c=["C6", "C7"], edgecolors="C8")
        expected = "C6"
    else:
        mark = LineCollection([[(0.1, 0.2), (0.8, 0.2)]], colors=["C8", "C9"])
        ax.add_collection(mark)
        expected = "C8"

    assert eq.bind("sigma", mark) is eq.term("sigma")
    _assert_rgb(eq.artists["glyphs"], expected)
    assert eq.bindings["sigma"].motif is None


def test_bind_unknown_term_names_the_known_terms(board):
    _, ax = board
    eq = _equation(ax)
    with pytest.raises(KeyError) as lookup:
        eq.term("missing")
    with pytest.raises(KeyError) as binding:
        eq.bind("missing", Artist(), motif="planet")
    assert binding.value.args == lookup.value.args
    assert "sigma" in str(binding.value)


def test_bind_refuses_to_bind_a_term_twice(board):
    _, ax = board
    eq = _equation(ax)
    mark = Artist()
    eq.bind("sigma", mark, motif="planet")
    before = [g.get_facecolor() for g in eq.artists["glyphs"]]
    with pytest.raises(ValueError, match="sigma"):
        eq.bind("sigma", Artist(), motif="star")
    assert eq.bindings["sigma"].mark is mark
    assert [g.get_facecolor() for g in eq.artists["glyphs"]] == before


def test_bind_requires_an_artist(board):
    _, ax = board
    eq = _equation(ax)
    with pytest.raises(TypeError):
        eq.bind("sigma", "not an artist", motif="planet")
    assert eq.bindings == {}


def test_bind_requires_a_motif_for_an_artist_without_color(board):
    _, ax = board
    eq = _equation(ax)
    with pytest.raises(ValueError, match="motif"):
        eq.bind("sigma", Artist())
    assert eq.bindings == {}


def test_bind_unknown_motif_uses_the_motifset_error(board):
    _, ax = board
    eq = _equation(ax)
    with pytest.raises(KeyError) as lookup:
        MotifSet.default()["missing"]
    with pytest.raises(KeyError) as binding:
        eq.bind("sigma", Artist(), motif="missing")
    assert binding.value.args == lookup.value.args
    assert eq.bindings == {}


def test_bind_leaves_unbound_terms_and_untagged_glyphs_unchanged(board):
    _, ax = board
    eq = ep.equation(
        ax,
        r"${{sigma:\sigma_x}}+{{other:y}}=0$",
        fontsize=30,
        colors={"sigma": "C0", "other": "C1"},
    )
    sigma_rgb = to_rgb("C0")
    untouched = [
        (g, g.get_facecolor(), g.get_gid())
        for g in eq.artists["glyphs"]
        if g.get_facecolor()[:3] != sigma_rgb
    ]
    assert len(untouched) == 4  # other, plus, equals, zero

    eq.bind("sigma", Artist(), motif="planet")

    for glyph, rgba, gid in untouched:
        assert glyph.get_facecolor() == rgba
        assert glyph.get_gid() == gid


def test_linked_accepts_names_terms_and_marks_and_returns_fresh_lists(board):
    _, ax = board
    eq = ep.equation(ax, r"${{sigma:\sigma}}+{{other:y}}$", fontsize=30)
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color="C2")
    term = eq.bind("sigma", mark)
    for target in ("sigma", term, mark):
        linked = eq.linked(target)
        assert isinstance(linked, list)
        assert len(linked) == 2
        assert linked[0] is term
        assert linked[1] is mark
        linked.clear()
        assert eq.linked(target) == [term, mark]
    for target in ("other", eq.term("other"), Artist()):
        with pytest.raises(KeyError):
            eq.linked(target)


def test_linked_matches_marks_by_identity(board):
    class EqualArtist(Artist):
        def __eq__(self, other):
            return isinstance(other, EqualArtist)

        __hash__ = object.__hash__

    _, ax = board
    eq = _equation(ax)
    mark, other = EqualArtist(), EqualArtist()
    assert mark is not other
    assert mark == other
    term = eq.bind("sigma", mark, motif="planet")

    linked = eq.linked(mark)
    assert linked[0] is term
    assert linked[1] is mark
    with pytest.raises(KeyError):
        eq.linked(other)


def test_indicate_cues_the_bound_term_and_mark_and_restores_them_exactly(board):
    _, ax = board
    eq = _equation(ax)
    (mark,) = ax.plot([0.1, 0.9], [0.2, 0.3], color="C2", linewidth=2)
    eq.bind("sigma", mark)
    rest_width = mark.get_linewidth()
    rest_colors = [g.get_facecolor() for g in eq.artists["glyphs"]]
    pulse = ep.indicate(eq.linked("sigma"))

    pulse.update(0.5)
    assert mark.get_linewidth() > rest_width
    assert all(
        g.get_facecolor() != rgba
        for g, rgba in zip(eq.artists["glyphs"], rest_colors, strict=True)
    )
    pulse.update(1.0)
    assert mark.get_linewidth() == rest_width
    assert [g.get_facecolor() for g in eq.artists["glyphs"]] == rest_colors


def test_bindings_hold_named_records_in_bind_order(board):
    _, ax = board
    eq = ep.equation(ax, r"${{sigma:\sigma}}+{{other:y}}$", fontsize=30)
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color="C2")
    motif = MotifSet.default()["planet"]
    other = eq.bind("other", Artist(), motif=motif)
    sigma = eq.bind("sigma", mark)

    assert isinstance(eq.bindings, dict)
    assert list(eq.bindings) == ["other", "sigma"]
    binding = eq.bindings["other"]
    assert isinstance(binding, tuple)
    assert type(binding).__name__ == "Binding"
    assert binding.term is other
    assert binding.motif is motif
    assert eq.bindings["sigma"].term is sigma
    assert eq.bindings["sigma"].mark is mark
    assert eq.bindings["sigma"].motif is None


@pytest.mark.parametrize("with_motif", [True, False], ids=["motif", "mark"])
def test_write_reveal_keeps_the_bound_color(board, with_motif):
    fig, ax = board
    eq = _equation(ax)
    if with_motif:
        motif = MotifSet.default()["planet"]
        eq.bind("sigma", Artist(), motif=motif)
        expected = motif.resolve_label()
    else:
        (mark,) = ax.plot([0.1, 0.9], [0.2, 0.3], color="C2")
        eq.bind("sigma", mark)
        expected = "C2"

    eq.update(0.5, reveal="write")
    fig.canvas.draw()
    _assert_rgb(eq.artists["glyphs"], expected)
    for glyph in eq.artists["glyphs"]:
        np.testing.assert_allclose(glyph.rgb, to_rgb(expected), atol=1e-6)
    eq.update(1.0, reveal="write")
    fig.canvas.draw()
    _assert_rgb(eq.artists["glyphs"], expected)
    assert all(g.get_facecolor()[3] == 1 for g in eq.artists["glyphs"])
