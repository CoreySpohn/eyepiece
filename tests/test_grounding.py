"""Equation terms share their marks' meaning, color, and cues."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import rc_context
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Rectangle

import eyepiece as ep
import eyepiece.testing as ept
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


def test_grounding_reports_shown_unbound_colored_terms(board):
    _, ax = board
    eq = ep.equation(ax, r"${{sigma:\sigma_x}}$", fontsize=30, colors={"sigma": "C1"})
    eq.update(1)

    faults = ept.grounding_faults(eq)

    assert len(faults) == 1
    fault = faults[0]
    assert isinstance(fault, ept.GroundingFault)
    assert (fault.kind, fault.term, fault.time_s) == ("no-referent", "sigma", None)
    assert "sigma" in fault.detail
    assert "unbound" in fault.detail


@pytest.mark.parametrize("color", [None, "neutral", "text"])
def test_grounding_allows_unbound_neutral_and_text_ink(board, color):
    _, ax = board
    with rc_context({"text.color": "#a020c0"}):
        colors = (
            {}
            if color is None
            else {"sigma": ep.neutral(0.8) if color == "neutral" else "#a020c0"}
        )
        eq = ep.equation(ax, r"${{sigma:\sigma_x}}$", fontsize=30, colors=colors)
        eq.update(1)
        assert ept.grounding_faults(eq) == []


def test_grounding_allows_rounded_neutral_with_tinted_text(board):
    _, ax = board
    # restore_style_state puts the global style back after this test.
    hwostyle.use("dark", palette_family="cyberpunk")
    # Some hwostyle versions use white text, so make the tint explicit.
    plt.rcParams["text.color"] = "#b0d0f0"
    neutral = to_hex(ep.neutral(0.6))
    assert max(to_rgb(neutral)) - min(to_rgb(neutral)) > 0.08
    eq = ep.equation(
        ax, r"${{sigma:\sigma_x}}$", fontsize=30, colors={"sigma": neutral}
    )
    eq.update(1)
    assert ept.grounding_faults(eq) == []

    for glyph in eq.term("sigma")._pieces:
        glyph.set_facecolor("C1")
    assert [(f.kind, f.term) for f in ept.grounding_faults(eq)] == [
        ("no-referent", "sigma")
    ]


@pytest.mark.parametrize("hidden", ["visible", "alpha", "axes", "removed"])
def test_grounding_requires_a_shown_bound_mark(board, hidden):
    _, ax = board
    eq = _equation(ax)
    (mark,) = ax.plot([0.1, 0.9], [0.2, 0.3], color="C1")
    eq.bind("sigma", mark)
    eq.update(1)
    assert ept.grounding_faults(eq) == []

    if hidden == "visible":
        mark.set_visible(False)
    elif hidden == "alpha":
        mark.set_alpha(0)
    elif hidden == "axes":
        ax.set_visible(False)
    else:
        mark.remove()

    faults = ept.grounding_faults(eq)
    assert [(f.kind, f.term) for f in faults] == [("no-referent", "sigma")]


@pytest.mark.parametrize("hidden", ["visible", "alpha", "face", "combined"])
def test_grounding_checks_only_shown_glyphs(board, hidden):
    _, ax = board
    eq = ep.equation(ax, r"${{sigma:\sigma_x}}$", fontsize=30, colors={"sigma": "C1"})
    eq.update(1)
    for glyph in eq.artists["glyphs"]:
        if hidden == "visible":
            glyph.set_visible(False)
        elif hidden == "alpha":
            glyph.set_alpha(0)
        elif hidden == "face":
            glyph.set_facecolor((*to_rgb("C1"), 0.01))
        else:
            glyph.set_alpha(0.2)
    assert ept.grounding_faults(eq) == []

    glyph = eq.artists["glyphs"][-1]
    glyph.set_visible(True)
    glyph.set_alpha(1)
    assert [f.kind for f in ept.grounding_faults(eq)] == ["no-referent"]


def test_grounding_reports_changed_term_color_with_hex_colors(board):
    _, ax = board
    eq = _equation(ax)
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color="#cc2200")
    eq.bind("sigma", mark)
    eq.update(1)
    for glyph in eq.artists["glyphs"]:
        glyph.set_facecolor("#0022cc")

    faults = ept.grounding_faults(eq, at=lambda t: None, times=[2, 3, 4])

    assert [(f.kind, f.term, f.time_s) for f in faults] == [
        ("color-mismatch", "sigma", 2)
    ]
    assert "#cc2200" in faults[0].detail
    assert "#0022cc" in faults[0].detail
    assert "2" in faults[0].detail


def test_grounding_compares_motif_mark_and_label_colors_separately(board):
    _, ax = board
    hwostyle.use("dark", palette_family="cyberpunk")
    motif = Motif(key="spread", noun="spread", role="zodi", label_color="zodi")
    assert to_rgb(motif.resolve()) != to_rgb(motif.resolve_label())
    eq = _equation(ax)
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color=motif.resolve(), label="spread mark")
    eq.bind("sigma", mark, motif=motif)
    eq.update(1)
    assert ept.grounding_faults(eq) == []

    mark.set_color("#ff00ff")
    faults = ept.grounding_faults(eq)
    assert [(f.kind, f.term) for f in faults] == [("color-mismatch", "sigma")]
    assert "mark" in faults[0].detail
    assert "spread mark" in faults[0].detail
    assert "#ff00ff" in faults[0].detail


def test_grounding_reports_changed_term_color_with_motif(board):
    _, ax = board
    hwostyle.use("dark", palette_family="cyberpunk")
    motif = Motif(key="spread", noun="spread", role="zodi", label_color="zodi")
    eq = _equation(ax)
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color=motif.resolve())
    eq.bind("sigma", mark, motif=motif)
    eq.update(1)
    assert ept.grounding_faults(eq) == []

    for glyph in eq.term("sigma")._pieces:
        glyph.set_facecolor("#ff00ff")

    assert to_rgb(mark.get_color()) == to_rgb(motif.resolve())
    assert [(f.kind, f.term) for f in ept.grounding_faults(eq)] == [
        ("color-mismatch", "sigma")
    ]


@pytest.mark.parametrize("first_matches", [True, False])
def test_grounding_uses_first_glyph_color_for_mixed_color_term(board, first_matches):
    _, ax = board
    hwostyle.use("dark", palette_family="cyberpunk")
    motif = Motif(key="spread", noun="spread", role="zodi", label_color="zodi")
    eq = _equation(ax)
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color=motif.resolve())
    eq.bind("sigma", mark, motif=motif)
    eq.update(1)
    glyphs = eq.term("sigma")._pieces
    assert len(glyphs) >= 2
    glyphs[0 if first_matches else 1].set_facecolor(motif.resolve_label())
    glyphs[1 if first_matches else 0].set_facecolor("#ff00ff")

    faults = ept.grounding_faults(eq)

    assert [(f.kind, f.term) for f in faults] == (
        [] if first_matches else [("color-mismatch", "sigma")]
    )


def test_grounding_resolves_motif_colors_at_check_time(board):
    _, ax = board
    hwostyle.use("dark", palette_family="cyberpunk")
    motif = Motif(key="spread", noun="spread", role="zodi", label_color="zodi")
    eq = _equation(ax)
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color=motif.resolve())
    eq.bind("sigma", mark, motif=motif)
    eq.update(1)
    hwostyle.use("light", palette_family="cyberpunk")
    mark.set_color(motif.resolve())
    for glyph in eq.artists["glyphs"]:
        glyph.set_facecolor(motif.resolve_label())
    assert ept.grounding_faults(eq) == []


@pytest.mark.parametrize("term_first", [True, False])
@pytest.mark.parametrize("times", [[2, 0, 1], [0, 1]])
def test_grounding_requires_concrete_before_abstract(board, term_first, times):
    _, ax = board
    eq = _equation(ax)
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color="C1")
    eq.bind("sigma", mark)
    sampled = []

    def at(t):
        sampled.append(t)
        eq.update(1 if term_first or t >= 1 else 0)
        mark.set_alpha(1 if not term_first or t >= 1 else 0)

    faults = ept.grounding_faults(eq, at=at, times=times)
    assert sampled == sorted(times)
    if term_first:
        assert {(f.kind, f.time_s) for f in faults} == {
            ("no-referent", 0),
            ("symbol-first", 0),
        }
    else:
        assert faults == []


def test_grounding_symbol_first_includes_a_mark_never_shown(board):
    _, ax = board
    eq = _equation(ax)
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color="C1")
    eq.bind("sigma", mark)
    eq.update(1)
    mark.set_visible(False)
    assert [f.kind for f in ept.grounding_faults(eq)] == ["no-referent"]
    faults = ept.grounding_faults(eq, at=lambda t: None, times=[0, 1, 2])
    assert [(f.kind, f.time_s) for f in faults] == [
        ("no-referent", 0),
        ("symbol-first", 0),
    ]


@pytest.mark.parametrize("kwargs", [{"at": lambda t: None}, {"times": [0, 1]}])
def test_grounding_requires_at_and_times_together(board, kwargs):
    _, ax = board
    with pytest.raises(ValueError):
        ept.grounding_faults(_equation(ax), **kwargs)


def test_assert_grounded_returns_clean_list_or_reports_every_fault(board):
    _, ax = board
    eq = ep.equation(
        ax,
        r"${{sigma:\sigma}}+{{other:y}}$",
        fontsize=30,
        colors={"sigma": "C1", "other": "C2"},
    )
    eq.update(1)
    with pytest.raises(ept.GroundingError) as err:
        ept.assert_grounded(eq)
    assert isinstance(err.value, AssertionError)
    assert err.value.faults == ept.grounding_faults(eq)
    lines = str(err.value).splitlines()
    for name in ("sigma", "other"):
        assert any(name in line and "no-referent" in line for line in lines)
    eq.update(0)
    assert ept.assert_grounded(eq) == []


def test_grounding_deduplicates_per_equation_term_and_kind(board):
    _, ax = board
    eq = ep.equation(
        ax,
        r"${{sigma:\sigma}}+{{other:y}}$",
        fontsize=30,
        colors={"sigma": "C1", "other": "C2"},
    )
    eq.update(1)
    second = ep.equation(
        ax, r"${{sigma:\sigma_x}}$", fontsize=30, colors={"sigma": "C1"}
    )
    second.update(1)
    faults = ept.grounding_faults([eq, second], at=lambda t: None, times=[2, 0, 1])

    assert [(f.term, f.kind, f.time_s) for f in faults] == [
        ("other", "no-referent", 0),
        ("sigma", "no-referent", 0),
        ("sigma", "no-referent", 0),
    ]
    assert all(f.kind in ept.GROUNDING_KINDS for f in faults)
    assert "equation 0" in faults[1].detail
    assert "equation 1" in faults[2].detail
    assert ept.grounding_faults(iter([eq, second])) == ept.grounding_faults(
        [eq, second]
    )


def test_grounding_sorts_findings_by_time_then_term(board):
    _, ax = board
    eq = ep.equation(
        ax,
        r"${{alpha:a}}+{{zeta:z}}$",
        fontsize=30,
        colors={"alpha": "C1", "zeta": "C2"},
    )

    def at(t):
        for name in ("alpha", "zeta"):
            for glyph in eq.term(name)._pieces:
                glyph.set_alpha(1 if name == "zeta" or t >= 1 else 0)

    faults = ept.grounding_faults(eq, at=at, times=[2, 1, 0])
    assert [(f.term, f.time_s) for f in faults] == [("zeta", 0), ("alpha", 1)]


def test_grounding_tolerance_applies_to_gray_and_color_comparisons(board):
    _, ax = board
    eq = ep.equation(
        ax, r"${{sigma:\sigma_x}}$", fontsize=30, colors={"sigma": (0.4, 0.45, 0.4)}
    )
    eq.update(1)
    assert ept.grounding_faults(eq) == []
    assert [f.kind for f in ept.grounding_faults(eq, tol=0.02)] == ["no-referent"]
    (mark,) = ax.plot([0, 1], [0.2, 0.3], color=(0.4, 0.45, 0.4))
    eq.bind("sigma", mark)
    for glyph in eq.artists["glyphs"]:
        glyph.set_facecolor((0.4, 0.5, 0.4))
    assert ept.grounding_faults(eq) == []
    assert [f.kind for f in ept.grounding_faults(eq, tol=0.02)] == ["color-mismatch"]


def test_grounding_public_names_are_exported():
    assert {
        "GROUNDING_KINDS",
        "GroundingFault",
        "GroundingError",
        "grounding_faults",
        "assert_grounded",
    } <= set(ept.__all__)
