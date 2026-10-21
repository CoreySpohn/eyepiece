"""Equations as glyph outlines: tags, matching, and the move between two."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.path import Path

import eyepiece as ep
import eyepiece.testing as ept

BAYES_A = (
    r"${{lhs:\mathrm{Pr}(H|E)=}}\frac{{num:\mathrm{Pr}(E|H)\,\mathrm{Pr}(H)}}"
    r"{{den:\mathrm{Pr}(E)}}$"
)
BAYES_B = r"${{lhs:\mathrm{Pr}(H|E)=}}\frac{{num:0.24}}{{den:0.24+0.07}}{{res:=0.77}}$"
PYTH_A = r"$a^2+b^2=c^2$"
PYTH_B = r"$a^2=c^2-b^2$"


def _stage(size=(8.0, 3.0), dpi=100):
    fig = plt.figure(figsize=size, dpi=dpi)
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


def _center(glyph, renderer):
    return glyph.vertices(renderer).mean(0)


def _term_of(result, glyph, renderer):
    """The name of the term whose laid-out box holds the glyph's center."""
    x, y = _center(glyph, renderer)
    for name, term in result.artists["terms"].items():
        box = term.get_window_extent(renderer)
        if box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1:
            return name
    return None


def _geometry(result, renderer):
    """Every glyph's drawn vertices in display pixels."""
    return [g.vertices(renderer) for g in result.artists["glyphs"]]


def test_exports_and_no_shadowing_module():
    import eyepiece._equation as mod

    assert ep.equation is mod.equation
    assert ep.transform is mod.transform


def test_glyph_tuple_format_is_pinned():
    """Layout canary: a known equation has a known glyph count under matplotlib.

    This pins how matplotlib's mathtext lays out the string (the fraction bar
    counts as one glyph), so a matplotlib upgrade that changes the layout
    fails here first, with a clear cause.
    """
    _fig, _ax, at = _stage()
    eq = ep.equation(at, r"$\frac{a}{b}=x^2$", fontsize=30)
    glyphs = eq.artists["glyphs"]
    assert len(glyphs) == 6  # a, b, =, x, 2 and the bar
    assert list(eq.artists["terms"]) == ["all"]


def test_no_markers_is_one_term_all():
    _fig, _ax, at = _stage()
    eq = ep.equation(at, r"$a+b$", fontsize=30)
    assert list(eq.artists["terms"]) == ["all"]
    assert len(eq.artists["glyphs"]) == 3


def test_tags_change_nothing_about_where_glyphs_sit():
    plain = (
        BAYES_A.replace("{{lhs:", "{{").replace("{{num:", "{{").replace("{{den:", "{{")
    )
    assert np.array_equal(_alone(BAYES_A), _alone(plain))
    fig, _ax, at = _stage()
    eq = ep.equation(at, BAYES_A, fontsize=40)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lhs, num, den = (
        eq.term(n).get_window_extent(renderer) for n in ("lhs", "num", "den")
    )
    assert lhs.x1 <= num.x0 + 1e-6 and lhs.x1 <= den.x0 + 1e-6  # the left side is left


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
    move = ep.transform(a, b, align=None)
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


def test_bayes_pair_count_canary():
    """Canary: the Bayes example pairs 9 glyphs, drops 17 and adds 18.

    The numbers depend on matplotlib's mathtext layout and on the matching
    rules; a change in either shows up here first, so a failure means look at
    both before adjusting the numbers.
    """
    _fig, a, b = _pair(BAYES_A, BAYES_B)
    move = ep.transform(a, b)
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
    new = {j: gb[j].vertices(renderer).copy() for j in move.new}
    for u in (0.2, 0.5, 0.8):
        move.update(u)
        for j in move.new:
            assert np.array_equal(gb[j].vertices(renderer), new[j])


def test_a_tagged_glyph_never_pairs_outside_its_tag():
    fig, a, b = _pair(
        r"${{x:a}}+a$",
        r"$a+{{x:a}}$",
    )
    move = ep.transform(a, b)
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, j in move.pairs:
        assert _term_of(a, ga[i], renderer) == _term_of(b, gb[j], renderer)
    assert len(move.pairs) == 3  # tagged a, plus and untagged a


def test_key_map_pairs_two_named_tags():
    fig, a, b = _pair(r"${{p:x}}+{{q:y}}$", r"${{r:y}}+{{s:x}}$")
    none = ep.transform(a, b)
    assert len(none.pairs) == 3  # tags that share nothing pair by order
    _fig, a, b = _pair(r"${{p:x}}+{{q:y}}$", r"${{r:y}}+{{s:x}}$")
    mapped = ep.transform(a, b, key_map={"p": "s", "q": "r"})
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    named = {
        (_term_of(a, ga[i], renderer), _term_of(b, gb[j], renderer))
        for i, j in mapped.pairs
    }
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

    def bar_width(result):
        fig.canvas.draw()
        return max(
            np.ptp(g.get_path().vertices[:, 0]) for g in result.artists["glyphs"]
        )

    move.update(0.0)
    start = bar_width(a)
    move.update(0.25)
    mid = bar_width(a)
    move.update(1.0)
    end = bar_width(b)
    assert start < mid < end


def test_a_fraction_bar_stays_a_rectangle_while_it_stretches():
    """The ends of the bar stay square at every step, never slanted."""
    fig, a, b = _pair(r"$\frac{a}{b}$", r"$\frac{a+c}{b+d}$")
    move = ep.transform(a, b)
    for u in (0.1, 0.25, 0.5, 0.75, 0.9):
        move.update(u)
        fig.canvas.draw()
        bars = [g for g in a.artists["glyphs"] if g.piece.is_rule and g.get_visible()]
        assert bars, "the bar is drawn by the first equation's rule"
        for bar in bars:
            v = bar.get_path().vertices
            xs = np.unique(np.round(v[:, 0], 6))
            ys = np.unique(np.round(v[:, 1], 6))
            assert len(xs) == 2 and len(ys) == 2, (u, v)


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
    fig, a, b = _pair(r"$a^2+{{b:b^2}}=c^2$", r"$a^2=c^2-{{b:b^2}}$")
    move = ep.transform(a, b)
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, j in move.pairs:
        assert (_term_of(a, ga[i], renderer) == "b") == (
            _term_of(b, gb[j], renderer) == "b"
        )
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


def test_update_order_and_interleaved_levels_do_not_change_pixels():
    us = [0.0, 0.1, 0.35, 0.5, 0.72, 0.9, 1.0]
    fig, a, b = _pair(BAYES_A, BAYES_B)
    move = ep.transform(a, b)
    inorder = {}
    for u in us:
        move.update(u)
        inorder[u] = _pixels(fig)
    rng = np.random.default_rng(3)
    for u in rng.permutation(us):
        move.update(float(rng.choice(us)), level=0.3)
        move.update(float(u), level=float(rng.uniform(0.1, 0.9)))
        move.update(float(u))
        assert np.array_equal(_pixels(fig), inorder[float(u)]), u


def test_set_dpi_round_trip_and_native_200_dpi_agree():
    def build(dpi):
        fig, _ax, at = _stage(dpi=dpi)
        a = ep.equation(at, BAYES_A, fontsize=40)
        b = ep.equation(at, BAYES_B, fontsize=40)
        return fig, ep.transform(a, b)

    fig, move = build(100)
    move.update(0.5)
    before = _pixels(fig)
    fig.set_dpi(200)
    big = _pixels(fig)
    native, native_move = build(200)
    native_move.update(0.5)
    assert np.array_equal(big, _pixels(native))
    fig.set_dpi(100)
    assert np.array_equal(before, _pixels(fig))


def test_savefig_at_200_dpi_matches_a_native_200_dpi_build():
    import io

    def build(dpi):
        fig, _ax, at = _stage(dpi=dpi)
        a = ep.equation(at, BAYES_A, fontsize=40)
        b = ep.equation(at, BAYES_B, fontsize=40)
        move = ep.transform(a, b)
        move.update(0.5)
        return fig

    def raw(fig, dpi):
        buf = io.BytesIO()
        fig.savefig(buf, format="rgba", dpi=dpi)
        return buf.getvalue()

    fig = build(100)
    saved = raw(fig, 200)
    assert saved == raw(build(200), 200)
    assert np.array_equal(_pixels(fig), _pixels(build(100)))  # savefig left dpi alone


def _aligned_pair(**kwargs):
    fig, a, b = _pair(BAYES_A, BAYES_B)
    return fig, a, b, ep.transform(a, b, **kwargs)


def test_aligned_end_states_are_each_equation_as_placed():
    fig, a, b, move = _aligned_pair()
    move.update(0.0)
    assert np.array_equal(_pixels(fig), _alone(BAYES_A))
    move.update(1.0)
    rest = _pixels(fig)
    for g in a.artists["glyphs"]:
        g.set_visible(False)
    for g in b.artists["glyphs"]:
        g.set_visible(True)
    assert np.array_equal(rest, _pixels(fig))
    move.update(0.4)
    assert not np.array_equal(rest, _pixels(fig))


def test_the_aligned_glyphs_never_move():
    fig, a, _b, move = _aligned_pair()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ga = a.artists["glyphs"]
    lhs = [i for i, _ in move.pairs if _term_of(a, ga[i], renderer) == "lhs"]
    assert lhs
    rest = {i: ga[i].vertices(renderer).copy() for i in lhs}
    first = {}
    for u in np.linspace(0.01, 0.99, 12):
        move.update(float(u))
        for i in lhs:
            v = ga[i].vertices(renderer)
            # Only the pixel snapping Agg would add at rest (under half a pixel).
            assert np.allclose(v, rest[i], atol=0.5)
            assert np.allclose(v, first.setdefault(i, v), atol=1e-6)


def test_align_names_a_tag_or_none_and_checks_it():
    fig, a, b = _pair(r"${{p:x}}+{{q:y}}$", r"${{q:y}}+{{p:x}}$")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    move = ep.transform(a, b, align="q")
    (i, j) = next(p for p in move.pairs if _term_of(a, ga[p[0]], renderer) == "q")
    assert np.allclose(_center(ga[i], renderer), _center(gb[j], renderer), atol=1e-6)
    with pytest.raises(KeyError, match="align"):
        ep.transform(a, b, align="zz")
    none = ep.transform(a, b, align=None)
    assert none.pairs


def test_a_word_moves_rigidly():
    fig, a, _b, move = _aligned_pair()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ga = a.artists["glyphs"]
    word = [i for i, _ in move.pairs if _term_of(a, ga[i], renderer) == "lhs"]
    rest = np.array([_center(ga[i], renderer) for i in word])
    gaps = np.linalg.norm(rest[:, None] - rest[None], axis=-1)
    for u in np.linspace(0.0, 0.99, 12):
        move.update(float(u))
        here = np.array([_center(ga[i], renderer) for i in word])
        now = np.linalg.norm(here[:, None] - here[None], axis=-1)
        assert np.allclose(now, gaps, atol=0.5)


def test_a_group_that_does_not_travel_does_not_rise():
    fig, a, b = _pair(r"$a^2+{{b:b^2}}=c^2$", r"$a^2=c^2-{{b:b^2}}$")
    move = ep.transform(a, b, arc=0.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ga = a.artists["glyphs"]
    still = [
        i
        for i, j in move.pairs
        if np.allclose(
            _center(ga[i], renderer),
            _center(b.artists["glyphs"][j], renderer),
            atol=0.5,
        )
    ]
    assert still
    rest = {i: _center(ga[i], renderer)[1] for i in still}
    for u in np.linspace(0.0, 0.99, 12):
        move.update(float(u))
        for i in still:
            assert abs(_center(ga[i], renderer)[1] - rest[i]) < 1e-6
    # The group that crosses the equals sign does rise on the arc.
    travelling = [i for i, _ in move.pairs if i not in still]
    base = {i: _center(ga[i], renderer) for i in travelling}
    move.update(0.0)
    start = {i: _center(ga[i], renderer)[1] for i in travelling}
    rises = []
    for u in np.linspace(0.05, 0.95, 19):
        move.update(float(u))
        rises += [_center(ga[i], renderer)[1] - start[i] for i in travelling]
    assert max(rises) > 1.0 and base


def test_replaced_terms_cross_fade_inside_a_moving_box():
    fig, a, b, move = _aligned_pair(replace="fade")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    names = [{_term_of(a, ga[i], renderer) for i in ia} for ia, _ in move.replaced]
    assert {"num"} in names and {"den"} in names
    ia, ib = next(
        r for r in move.replaced if _term_of(a, ga[r[0][0]], renderer) == "num"
    )
    move.update(1.0)
    final = [gb[j].vertices(renderer).copy() for j in ib]
    move.update(0.3)
    src = [ga[i].get_alpha() for i in ia]
    dst = [gb[j].get_alpha() for j in ib]
    assert all(s is not None and 0.0 < s < 1.0 for s in src)
    assert all(d is not None and 0.0 <= d < 1.0 for d in dst)
    move.update(0.55)
    # The target sits inside the box the source is leaving by, never in place.
    assert not all(
        np.allclose(gb[j].vertices(renderer), v) for j, v in zip(ib, final, strict=True)
    )
    move.update(1.0)
    for j, v in zip(ib, final, strict=True):
        assert np.array_equal(gb[j].vertices(renderer), v)


def test_unpaired_text_arrives_only_after_the_motion_settles():
    fig, a, b, move = _aligned_pair()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gb = b.artists["glyphs"]
    res = [j for j in move.new if _term_of(b, gb[j], renderer) == "res"]
    assert res
    for u in (0.1, 0.3, 0.5, 0.59):
        move.update(u)
        assert all(not gb[j].get_visible() for j in res)
    move.update(0.8)
    assert all(gb[j].get_visible() for j in res)
    ga = a.artists["glyphs"]
    move.update(0.6)
    settled = [ga[i].vertices(renderer).copy() for i, _ in move.pairs]
    move.update(0.9)
    for (i, _), v in zip(move.pairs, settled, strict=True):
        assert np.allclose(ga[i].vertices(renderer), v, atol=1e-6)


def _group_box(glyphs, renderer):
    v = np.vstack([g.vertices(renderer) for g in glyphs])
    return v.min(0), v.max(0)


def test_arc_direction_follows_travel_direction():
    fig, a, b = _pair(r"${{p:x}}+{{q:y}}$", r"${{q:y}}+{{p:x}}$")
    move = ep.transform(a, b, stagger=1.0, arc=0.5, align=None)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    right, left = {}, {}
    for i, j in move.pairs:
        dx = _center(gb[j], renderer)[0] - _center(ga[i], renderer)[0]
        if abs(dx) > 5.0:
            (right if dx > 0 else left)[i] = _center(ga[i], renderer)[1]
    assert right and left
    move.update(0.3, level=1.0)
    for i, y in right.items():
        assert _center(ga[i], renderer)[1] > y + 1.0
    for i, y in left.items():
        assert _center(ga[i], renderer)[1] < y - 1.0


def test_swapping_glyphs_never_overlap():
    fig, a, b = _pair(r"$a^2+{{b:b^2}}=c^2$", r"$a^2=c^2-{{b:b^2}}$")
    move = ep.transform(a, b, stagger=0.7, arc=0.7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    movers = [
        (i, j)
        for i, j in move.pairs
        if abs(_center(gb[j], renderer)[0] - _center(ga[i], renderer)[0]) > 5.0
    ]
    bsq = [i for i, _ in movers if _term_of(a, ga[i], renderer) == "b"]
    csq = [i for i, _ in movers if _term_of(a, ga[i], renderer) != "b"]
    assert bsq and csq
    for u in np.linspace(0.0, 1.0, 21)[1:-1]:
        move.update(float(u))
        for i in bsq:
            for j in csq:
                b0, b1 = _group_box([ga[i]], renderer)
                c0, c1 = _group_box([ga[j]], renderer)
                assert ((b1 < c0) | (c1 < b0)).any(), (u, i, j)


def test_auto_align_keeps_the_shared_shift_still_without_a_tag():
    fig, a, b = _pair(r"$a^2+{{b:b^2}}=c^2$", r"$a^2=c^2-{{b:b^2}}$")
    move = ep.transform(a, b)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    first = min(range(len(ga)), key=lambda i: _center(ga[i], renderer)[0])
    j = dict(move.pairs)[first]
    assert np.allclose(
        _center(ga[first], renderer), _center(gb[j], renderer), atol=1e-6
    )
    start = ga[first].vertices(renderer).copy()
    for u in (0.2, 0.5, 0.8):
        move.update(u)
        assert np.allclose(ga[first].vertices(renderer), start, atol=1e-6)


def _left_anchored_bayes(replace="fade"):
    """The gallery's Bayes clip; ``replace=None`` takes transform's default."""
    fig = plt.figure(figsize=(7.6, 2.4), dpi=80)
    board = fig.add_axes([0, 0, 1, 1])
    board.set(xlim=(0, 1), ylim=(0, 1))
    board.axis("off")
    left = ep.Anchor.from_data(board, (0.06, 0.5))
    colors = {"lhs": "C0", "num": "C1", "den": "C3", "res": "C2"}
    a = ep.equation(left, BAYES_A, fontsize=34, colors=colors, ha="left")
    b = ep.equation(left, BAYES_B, fontsize=34, colors=colors, ha="left")
    options = {} if replace is None else {"replace": replace}
    return fig, ep.transform(a, b, stagger=0.55, **options)


def test_left_anchored_bayes_moves_without_a_jump_into_or_out_of_rest():
    """Pixel snapping on the fraction bar and the signs is not a step at the ends."""
    fig, move = _left_anchored_bayes()
    duration = 2.0

    def at(t):
        move.update(min(1.0, max(0.0, (t - 0.4) / 1.2)))

    ept.assert_motion(fig, at, ept.Window(0.0, duration), fps=30)


def test_the_bar_stays_an_axis_aligned_rectangle_through_the_move():
    """The check that catches a fraction bar resampled into a slant."""
    fig, move = _left_anchored_bayes()
    bars = [g for g in move.artists["glyphs"] if g.piece.is_rule]
    assert bars

    def at(t):
        move.update(min(1.0, max(0.0, (t - 0.4) / 1.2)))

    ept.assert_motion(
        fig, at, ept.Window(0.0, 2.0), fps=30, rectilinear=bars, kinds=["shape-class"]
    )


def test_a_transform_result_carries_its_rectilinear_bar_into_the_check():
    """The result goes in as the move; its contract names the bar."""
    fig, move = _left_anchored_bayes()
    assert any(g.piece.is_rule for g in move.contract.rectilinear)

    def at(t):
        move.update(min(1.0, max(0.0, (t - 0.4) / 1.2)))

    found = ept.motion_faults(
        fig, at, ept.Window(0.0, 2.0), fps=30, moves=[move], kinds=["shape-class"]
    )
    assert found == []
    # The same read finds bars that lean: tip a corner after each draw.
    for bar in move.contract.rectilinear:
        bar.draw = _leaning(bar)
    found = ept.motion_faults(
        fig, at, ept.Window(0.0, 2.0), fps=30, moves=[move], kinds=["shape-class"]
    )
    assert found
    assert {f.kind for f in found} == {"shape-class"}


def _leaning(bar):
    draw = bar.draw

    def tipped(renderer):
        draw(renderer)
        bar.get_path().vertices[1, 1] += 5.0

    return tipped


def _all_vertices(move, renderer):
    return [g.vertices(renderer) for g in move.artists["glyphs"] if g.get_visible()]


def test_replace_must_be_fade_or_morph():
    _fig, a, b = _pair(BAYES_A, BAYES_B)
    with pytest.raises(ValueError, match="replace"):
        ep.transform(a, b, replace="dissolve")


def test_morph_and_fade_pair_the_same_glyphs():
    _fig, a, b = _pair(BAYES_A, BAYES_B)
    fade = ep.transform(a, b, replace="fade")
    morph = ep.transform(a, b, replace="morph")
    assert (fade.pairs, fade.gone, fade.new, fade.replaced) == (
        morph.pairs,
        morph.gone,
        morph.new,
        morph.replaced,
    )


def test_morph_rest_states_are_each_equation_as_drawn():
    fig, a, b = _pair(BAYES_A, BAYES_B)
    move = ep.transform(a, b, replace="morph")
    move.update(0.0)
    assert np.array_equal(_pixels(fig), _alone(BAYES_A))
    move.update(1.0)
    rest = _pixels(fig)
    for g in a.artists["glyphs"]:
        g.set_visible(False)
    for g in b.artists["glyphs"]:
        g.set_visible(True)
    assert np.array_equal(rest, _pixels(fig))
    move.update(0.4)
    assert not np.array_equal(rest, _pixels(fig))
    for u in (0.5, 0.9, 0.3, 0.0, 1.0, 0.7):
        move.update(u)
    move.update(1.0)
    assert np.array_equal(_pixels(fig), rest)


def test_left_anchored_morph_moves_without_a_jump_into_or_out_of_rest():
    fig, move = _left_anchored_bayes("morph")

    def at(t):
        move.update(min(1.0, max(0.0, (t - 0.4) / 1.2)))

    ept.assert_motion(fig, at, ept.Window(0.0, 2.0), fps=30)


def test_morph_changes_shape_where_fade_only_moves_a_box():
    """A replaced term's outlines are not scaled copies of the old ones mid-move."""
    fig, a, b = _pair(BAYES_A, BAYES_B)
    morph = ep.transform(a, b, replace="morph")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ia, ib = next(
        r
        for r in morph.replaced
        if _term_of(a, a.artists["glyphs"][r[0][0]], renderer) == "num"
    )
    ga, gb = a.artists["glyphs"], b.artists["glyphs"]
    morph.update(0.45)
    # The 13 source glyphs outnumber the 4 target glyphs: the sources draw the
    # morphs and the targets stay hidden until the move ends.
    assert len(ia) > len(ib)
    assert all(ga[i].get_visible() for i in ia)
    assert all(not gb[j].get_visible() for j in ib)
    alphas = [ga[i].get_alpha() for i in ia]
    assert all(al is None or 0.0 <= al <= 1.0 for al in alphas)
    assert any(al is not None and al < 1.0 for al in alphas)  # copies fade out


def test_morph_never_makes_a_nan_or_an_infinity():
    fig, a, b = _pair(BAYES_A, BAYES_B)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    move = ep.transform(a, b, replace="morph")
    for u in np.linspace(0.0, 1.0, 41):
        move.update(float(u))
        for v in _all_vertices(move, renderer):
            assert np.isfinite(v).all(), u
    for u in (1e-9, 0.5, 1 - 1e-9):
        move.update(u, level=0.0)
        move.update(u, level=0.5)
        for v in _all_vertices(move, renderer):
            assert np.isfinite(v).all(), u


@pytest.mark.parametrize(
    "tex_a, tex_b",
    [
        (r"${{t:.}}$", r"${{t:=}}$"),  # a dot becomes a rectilinear bar pair
        (r"${{t:-}}+c$", r"${{t:x^2_i}}+c$"),
        (r"${{t:\frac{a}{b}}}$", r"${{t:0}}$"),  # a rule into a glyph
        (r"${{t:abc}}$", r"${{t:|}}$"),
        (r"${{t:8}}$", r"${{t:Pr(E|H)Pr(H)}}$"),
    ],
)
def test_morph_degenerate_pairs_stay_finite_and_rest_exactly(tex_a, tex_b):
    fig, a, b = _pair(tex_a, tex_b)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    move = ep.transform(a, b, replace="morph", align=None)
    assert move.replaced
    for u in np.linspace(0.0, 1.0, 21):
        move.update(float(u))
        for v in _all_vertices(move, renderer):
            assert np.isfinite(v).all(), (u, tex_a, tex_b)
    move.update(0.0)
    assert np.array_equal(_pixels(fig), _alone(tex_a))
    move.update(1.0)
    rest = _pixels(fig)
    for g in a.artists["glyphs"]:
        g.set_visible(False)
    for g in b.artists["glyphs"]:
        g.set_visible(True)
    assert np.array_equal(rest, _pixels(fig))


def test_morph_carrier_reaches_the_target_outline_as_the_move_ends():
    """One short of the end, each drawn outline already lies on the target's."""
    fig, a, b = _pair(r"${{t:0}}$", r"${{t:1}}$")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    move = ep.transform(a, b, replace="morph", align=None, stagger=1.0)
    (glyph,) = [g for g in a.artists["glyphs"]]
    target = b.artists["glyphs"][0]
    move.update(1.0)
    # Matplotlib puts a dummy vertex at the origin on each closing code.
    real = np.asarray(target.piece.draw_codes) != Path.CLOSEPOLY
    final = target.vertices(renderer)[real]
    move.update(1.0 - 1e-6)
    drawn = glyph.vertices(renderer)
    assert drawn.min(0) == pytest.approx(final.min(0), abs=0.05)
    assert drawn.max(0) == pytest.approx(final.max(0), abs=0.05)


def test_morph_is_a_move_of_the_motion_part_of_the_beat():
    """New text still arrives after the motion, as with the box."""
    fig, a, b = _pair(BAYES_A, BAYES_B)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    move = ep.transform(a, b, replace="morph")
    gb = b.artists["glyphs"]
    res = [j for j in move.new if _term_of(b, gb[j], renderer) == "res"]
    for u in (0.1, 0.3, 0.5, 0.59):
        move.update(u)
        assert all(not gb[j].get_visible() for j in res)
    ga = a.artists["glyphs"]
    move.update(0.6)
    settled = [v.copy() for v in _all_vertices(move, renderer)]
    move.update(0.9)
    again = _all_vertices(move, renderer)
    # The morph has finished by 0.6, so only the arriving text changes.
    assert len(again) >= len(settled) and ga
    for v, w in zip(settled, again, strict=False):
        assert np.allclose(v, w, atol=1e-6)


@pytest.mark.parametrize("replace", ["fade", "morph"])
def test_curved_glyphs_do_not_pop_when_a_move_starts_or_ends(replace):
    """A curved glyph rasterizes alike at rest and in motion: no antialiasing step."""
    fig, a, b = _pair(r"${{t:a}}$", r"${{t:g}}$")
    move = ep.transform(a, b, replace=replace, align=None, stagger=1.0)
    move.update(0.0)
    start = _pixels(fig).astype(int)
    move.update(1.0)
    end = _pixels(fig).astype(int)
    for u, rest in ((1e-3, start), (1.0 - 1e-3, end)):
        move.update(u)
        assert np.abs(_pixels(fig).astype(int) - rest).max() < 25, (replace, u)
    for tex in ("a", r"\mathrm{Pr}(E|H)", "0.24"):
        fig, a, b = _pair(f"${{{{t:{tex}}}}}$", r"${{t:x^2}}$")
        move = ep.transform(a, b, replace=replace, align=None, stagger=1.0)
        move.update(0.0)
        rest = _pixels(fig).astype(int)
        move.update(1e-9)
        assert np.abs(_pixels(fig).astype(int) - rest).max() < 25, (replace, tex)


def test_default_morph_bayes_lints_clean_with_its_declared_contract():
    fig, move = _left_anchored_bayes(None)
    duration = 2.0

    def at(t):
        move.update(min(1.0, max(0.0, (t - 0.4) / 1.2)))

    assert ept.motion_faults(fig, at, ept.Window(0.0, duration), moves=[move]) == []
