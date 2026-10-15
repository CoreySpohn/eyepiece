"""Equations as glyph outlines: tags, matching, and the move between two."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept

BAYES_A = (
    r"${{lhs:\mathrm{Pr}(H|E)=}}\frac{{num:\mathrm{Pr}(E|H)\,\mathrm{Pr}(H)}}"
    r"{{den:\mathrm{Pr}(E)}}$"
)
BAYES_B = r"${{lhs:\mathrm{Pr}(H|E)=}}\frac{{num:0.24}}{{den:0.24+0.07}}{{res:=0.77}}$"
PYTH_A = r"$a^2+b^2=c^2$"
PYTH_B = r"$a^2=c^2-b^2$"


def _stage(size=(8.0, 3.0)):
    fig = plt.figure(figsize=size, dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax, ep.Anchor.from_data(ax, (0.5, 0.45))


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _pair(tex_a, tex_b, **kw):
    fig, _ax, at = _stage()
    a = ep.equation(at, tex_a, fontsize=40, **kw)
    b = ep.equation(at, tex_b, fontsize=40, **kw)
    return fig, a, b


def _alone(tex):
    fig, _ax, at = _stage()
    ep.equation(at, tex, fontsize=40)
    return _pixels(fig)


def _geometry(result, renderer):
    """Every glyph's drawn vertices in display pixels."""
    return [g.vertices(renderer) for g in result.artists["glyphs"]]


def test_exports_and_no_shadowing_module():
    import eyepiece._equation as mod

    assert ep.equation is mod.equation
    assert ep.transform is mod.transform


def test_glyph_tuple_format_is_pinned():
    """A known equation has a known glyph and rule count (matplotlib 3.11 layout)."""
    _fig, _ax, at = _stage()
    eq = ep.equation(at, r"$\frac{a}{b}=x^2$", fontsize=30)
    glyphs = eq.artists["glyphs"]
    rules = [g for g in glyphs if g.piece.is_rule]
    assert len(glyphs) == 6  # a, b, =, x, 2 and the bar
    assert len(rules) == 1
    assert list(eq.artists["terms"]) == ["all"]


def test_no_markers_is_one_term_all():
    _fig, _ax, at = _stage()
    eq = ep.equation(at, r"$a+b$", fontsize=30)
    assert list(eq.artists["terms"]) == ["all"]
    assert len(eq.term("all")._pieces) == 3


def test_phantom_tagging_isolates_a_group_and_leaves_others_in_place():
    _fig, _ax, at = _stage()
    tagged = ep.equation(at, BAYES_A, fontsize=40)
    plain = ep.equation(
        at,
        BAYES_A.replace("{{lhs:", "{{").replace("{{num:", "{{").replace("{{den:", "{{"),
        fontsize=40,
    )
    assert [len(t._pieces) for t in tagged.artists["terms"].values()] == [8, 12, 5]
    # The tags change nothing about where glyphs sit.
    ours = sorted(
        tuple(np.round(g.piece.local.mean(0), 3)) for g in tagged.artists["glyphs"]
    )
    theirs = sorted(
        tuple(np.round(g.piece.local.mean(0), 3)) for g in plain.artists["glyphs"]
    )
    assert ours == theirs
    assert sum(g.piece.tag is None for g in tagged.artists["glyphs"]) == 1  # the bar


def test_a_layout_shift_raises(monkeypatch):
    """A phantom parse that moves a surviving glyph is an error, not a wrong tag."""
    import eyepiece._equation as mod

    real = mod._parse

    def shifted(plain, fontsize):
        pieces, width = real(plain, fontsize)
        if "phantom" in plain:
            for piece in pieces[:1]:
                piece.center = piece.center + 5.0
        return pieces, width

    monkeypatch.setattr(mod, "_parse", shifted)
    _fig, _ax, at = _stage()
    with pytest.raises(ValueError, match="changed the layout"):
        ep.equation(at, r"$a+{{t:b}}+c$", fontsize=30)


def test_escaped_braces_and_bad_markers_raise():
    _fig, _ax, at = _stage()
    with pytest.raises(ValueError, match="escaped braces"):
        ep.equation(at, r"$\{{{t:a}}\}$", fontsize=30)
    with pytest.raises(ValueError, match="unclosed"):
        ep.equation(at, r"${{t:a$", fontsize=30)
    with pytest.raises(ValueError, match="unique"):
        ep.equation(at, r"${{t:a}}+{{t:b}}$", fontsize=30)
    with pytest.raises(ValueError, match="one math span"):
        ep.equation(at, r"$a$ and $b$", fontsize=30)


def test_unknown_term_raises_key_error_listing_the_tags():
    _fig, _ax, at = _stage()
    eq = ep.equation(at, BAYES_A, fontsize=30)
    with pytest.raises(KeyError, match=r"lhs.*num.*den"):
        eq.term("nope")
    assert eq.term("num") is eq.artists["terms"]["num"]


def test_term_is_an_anchor_target_with_a_box_inside_the_equation():
    fig, _ax, at = _stage()
    eq = ep.equation(at, BAYES_A, fontsize=40)
    fig.canvas.draw()
    whole = eq.term("lhs").get_window_extent(), eq.term("den").get_window_extent()
    assert whole[0].x1 < whole[1].x1
    point = ep.Anchor.from_artist(eq.term("num")).display()
    assert whole[0].x1 < point[0]
    ep.circumscribe(eq.term("num")).update(0.4)
    ep.indicate(eq.term("num")).update(0.5)
    fig.canvas.draw()


def test_stage_dims_a_term_through_its_glyphs():
    fig, _ax, at = _stage()
    fig.set_facecolor("white")
    eq = ep.equation(at, BAYES_A, fontsize=40, colors={"num": "C1"})
    stage = ep.Stage({"num": eq.term("num")}, dim=0.2)
    before = _pixels(fig)
    stage.set(shown=[], dimmed=["num"])
    assert not np.array_equal(before, _pixels(fig))
    stage.rest()
    assert np.array_equal(before, _pixels(fig))


def test_equation_writes_on_and_rests_whole():
    fig, _ax, at = _stage()
    eq = ep.equation(at, BAYES_A, fontsize=40)
    whole = _pixels(fig)
    eq.update(0.0)
    assert not any(g.get_visible() for g in eq.artists["glyphs"])
    eq.update(0.5)
    shown = [g.get_visible() for g in eq.artists["glyphs"]]
    assert any(shown) and not all(shown)
    eq.update(1.0)
    assert np.array_equal(whole, _pixels(fig))


def test_equation_level_is_opacity_only():
    fig, _ax, at = _stage()
    eq = ep.equation(at, BAYES_A, fontsize=40)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    full = _geometry(eq, renderer)
    eq.update(1.0, level=0.3)
    for a, b in zip(full, _geometry(eq, renderer), strict=True):
        assert np.array_equal(a, b)
    assert all(g.get_alpha() == pytest.approx(0.3) for g in eq.artists["glyphs"])


def test_transform_rest_is_b_and_ends_equal_each_equation_alone():
    fig, a, b = _pair(BAYES_A, BAYES_B)
    move = ep.transform(a, b)
    rest = _pixels(fig)
    move.update(0.0)
    assert np.array_equal(_pixels(fig), _alone(BAYES_A))
    move.update(1.0)
    assert np.array_equal(_pixels(fig), _alone(BAYES_B))
    assert np.array_equal(rest, _pixels(fig))
    move.update(0.5)
    assert not np.array_equal(rest, _pixels(fig))


def test_every_glyph_is_accounted_for_exactly_once():
    _fig, a, b = _pair(BAYES_A, BAYES_B)
    move = ep.transform(a, b)
    ia = [i for i, _ in move.pairs] + list(move.gone)
    ib = [j for _, j in move.pairs] + list(move.new)
    assert sorted(ia) == list(range(len(a.artists["glyphs"])))
    assert sorted(ib) == list(range(len(b.artists["glyphs"])))
    assert (len(move.pairs), len(move.gone), len(move.new)) == (9, 17, 18)


def test_matched_glyphs_move_monotonically_and_unmatched_never_move():
    fig, a, b = _pair(PYTH_A, PYTH_B)
    move = ep.transform(a, b)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    assert len(move.pairs) >= 6  # most glyphs of the rearrangement match
    rest = {i: g.vertices(renderer).copy() for i, g in enumerate(ga)}
    for i, j in move.pairs:
        start, end = rest[i].mean(0), gb[j].vertices(renderer).mean(0)
        axis = end - start
        last = -1.0
        for u in np.linspace(0.02, 0.98, 25):
            move.update(u)
            here = (ga[i].vertices(renderer).mean(0) - start) @ axis
            assert here >= last - 1e-9
            last = here
    for u in (0.2, 0.5, 0.8):
        move.update(u)
        for i in move.gone:
            assert np.array_equal(ga[i].vertices(renderer), rest[i])
        for j in move.new:
            assert gb[j].pose is None


def test_a_tagged_glyph_never_pairs_outside_its_tag():
    _fig, a, b = _pair(
        r"${{x:a}}+a$",
        r"$a+{{x:a}}$",
    )
    move = ep.transform(a, b)
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    for i, j in move.pairs:
        assert ga[i].piece.tag == gb[j].piece.tag
    assert len(move.pairs) == 3  # tagged a, plus and untagged a


def test_key_map_pairs_two_named_tags():
    _fig, a, b = _pair(r"${{p:x}}+{{q:y}}$", r"${{r:y}}+{{s:x}}$")
    none = ep.transform(a, b)
    assert len(none.pairs) == 3  # tags that share nothing pair by order
    _fig, a, b = _pair(r"${{p:x}}+{{q:y}}$", r"${{r:y}}+{{s:x}}$")
    mapped = ep.transform(a, b, key_map={"p": "s", "q": "r"})
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    named = {(ga[i].piece.tag, gb[j].piece.tag) for i, j in mapped.pairs}
    assert {("p", "s"), ("q", "r")} <= named
    with pytest.raises(KeyError, match="terms are"):
        ep.transform(a, b, key_map={"zz": "s"})


def test_transform_level_changes_alpha_only():
    fig, a, b = _pair(BAYES_A, BAYES_B, colors={"num": "C1"})
    move = ep.transform(a, b, arc=0.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    glyphs = move.artists["glyphs"]
    move.update(0.45)
    full = [g.vertices(renderer).copy() for g in glyphs]
    alphas = [g.get_alpha() for g in glyphs]
    move.update(0.45, level=0.3)
    for g, v in zip(glyphs, full, strict=True):
        assert np.array_equal(g.vertices(renderer), v)
    dimmed = [g.get_alpha() for g in glyphs]
    assert any(d != a for d, a in zip(dimmed, alphas, strict=True))
    for d, a in zip(dimmed, alphas, strict=True):
        assert d is None or d <= (1.0 if a is None else a) * 0.3 + 1e-9


def test_a_fraction_bar_changes_length_through_the_morph_sampling():
    fig, a, b = _pair(r"$\frac{a}{b}$", r"$\frac{a+c}{b+d}$")
    move = ep.transform(a, b)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bar = next(g for g in a.artists["glyphs"] if g.piece.is_rule)
    move.update(0.5)
    path = bar.rule_path(renderer)
    width_a = np.ptp(a.artists["glyphs"][0].eq.place(bar.piece.local, renderer)[:, 0])
    assert np.ptp(path.vertices[:, 0]) > width_a


def test_transform_checks_its_arguments():
    _fig, a, b = _pair(PYTH_A, PYTH_B)
    with pytest.raises(TypeError):
        ep.transform(a, "b")
    with pytest.raises(ValueError, match="stagger"):
        ep.transform(a, b, stagger=0.0)
    with pytest.raises(ValueError, match="fade"):
        ep.transform(a, b, fade=1.5)


def test_clip_frames_are_readable():
    fig, a, b = _pair(PYTH_A, PYTH_B)
    move = ep.transform(a, b)
    move.update(0.0)
    ept.assert_readable(fig)
    move.update(1.0)
    ept.assert_readable(fig)


def test_a_tag_keeps_a_group_together_where_order_would_not():
    _fig, a, b = _pair(r"$a^2+{{b:b^2}}=c^2$", r"$a^2=c^2-{{b:b^2}}$")
    move = ep.transform(a, b)
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    for i, j in move.pairs:
        assert ga[i].piece.key == gb[j].piece.key
        assert ga[i].piece.tag == gb[j].piece.tag
    assert len(move.pairs) == 7  # a, 2, =, c, 2 and the tagged b, 2
    assert len(move.gone) == 1 and len(move.new) == 1  # the plus and the minus


def test_tagged_into_untagged_pairs_by_glyph_order():
    _fig, a, b = _pair(r"${{t:a^2}}+b^2=c^2$", PYTH_B)
    move = ep.transform(a, b)
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    assert len(move.pairs) >= 6
    for i, j in move.pairs:
        assert ga[i].piece.key == gb[j].piece.key


def test_a_tag_with_a_lost_partner_warns():
    _fig, a, b = _pair(r"${{t:a}}+{{u:b}}$", r"${{t:b}}+{{v:a}}$")
    with pytest.warns(UserWarning, match="'u'"):
        ep.transform(a, b)


def test_a_new_tag_that_adds_content_does_not_warn(recwarn):
    _fig, a, b = _pair(BAYES_A, BAYES_B)
    ep.transform(a, b)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
