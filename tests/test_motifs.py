"""Motifs: keyed objects whose colors resolve at call time, and the set checks."""

import dataclasses

import hwostyle
import matplotlib.pyplot as plt
import pytest

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece import motifs as motif_module

MODES = ("dark", "light", "paper")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def table_set():
    """The default set built from the hwostyle table; skipped when it is absent."""
    pytest.importorskip("hwostyle.motifs")
    return ep.MotifSet.default()


@pytest.fixture(params=["table", "fallback"])
def default_set(request, monkeypatch):
    """The default set from the hwostyle table, and from the stand-in table."""
    if request.param == "table":
        pytest.importorskip("hwostyle.motifs")
    else:
        monkeypatch.setattr(motif_module, "_hwostyle_table", lambda: None)
    return ep.MotifSet.default()


def _m(key, role, **kw):
    return ep.Motif(key, key, role, **kw)


def test_a_motif_stores_a_role_and_resolves_the_color_when_asked():
    star = _m("star", "star", marker="*", linestyle="none")
    assert star.role == "star"
    assert not star.role.startswith("#")
    colors = []
    for mode in MODES:
        hwostyle.use(mode)
        colors.append(star.color)
        assert star.color == ep.role("star")
        assert star.resolve(mode) == star.color
    assert len(set(colors)) == 3


def test_a_mode_switch_is_seen_without_rebuilding_the_motif():
    planet = _m("planet", "planet", marker="o")
    hwostyle.use("dark")
    dark = planet.color
    hwostyle.use("light")
    assert planet.color != dark


def test_a_motif_is_frozen_and_derive_changes_a_copy():
    star = _m("star", "star", marker="*")
    with pytest.raises(dataclasses.FrozenInstanceError):
        star.marker = "o"
    issued = star.derive(key="issued", marker="s")
    assert (star.marker, issued.marker, issued.role) == ("*", "s", "star")


def test_kwargs_follow_how_the_motif_is_drawn():
    hwostyle.use("light")
    disk = _m("disk", "disk", hatch="///", fill_alpha=0.12, linewidth=0.8)
    patch = disk.patch_kw()
    assert patch["hatch"] == "///"
    assert patch["edgecolor"] == disk.color
    assert patch["facecolor"][3] == pytest.approx(0.12)
    assert disk.line_kw() == {
        "color": disk.color,
        "ls": "-",
        "lw": 0.8,
        "gid": "disk",
    }
    point = _m("star", "star", marker="*", linestyle="none")
    assert point.marker_kw(size=9) == {
        "color": point.color,
        "marker": "*",
        "ls": "none",
        "ms": 9,
        "gid": "star",
    }


@pytest.mark.parametrize("helper", ["line_kw", "marker_kw", "patch_kw", "label_kw"])
def test_motif_kwargs_carry_the_key_as_gid(helper):
    motif = _m("planet", "planet", marker="o")
    kwargs = getattr(motif, helper)
    assert kwargs()["gid"] == motif.key
    assert kwargs(gid=None)["gid"] == motif.key


@pytest.mark.parametrize("helper", ["line_kw", "marker_kw", "patch_kw", "label_kw"])
@pytest.mark.parametrize("gid", ["planet:orbit", ""])
def test_motif_kwargs_accept_an_explicit_gid(helper, gid):
    motif = _m("planet", "planet", marker="o")
    assert getattr(motif, helper)(gid=gid)["gid"] == gid


def test_meaning_keys_maps_each_motif_key_to_itself():
    motifs = ep.MotifSet([_m("a", "star"), _m("b", "planet")])
    assert motifs.meaning_keys() == {"a": "a", "b": "b"}
    assert motifs.only("b").meaning_keys() == {"b": "b"}
    assert ep.MotifSet([]).meaning_keys() == {}


def test_meaning_keys_finds_a_same_hue_pair_drawn_with_motif_kwargs():
    hwostyle.use("dark")
    motifs = ep.MotifSet([_m("a", "planet"), _m("b", "planet")])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    for y, motif in zip((3, 6), motifs.values(), strict=True):
        ax.plot([1, 9], [y, y], **motif.line_kw())
    found = ept.meaning_faults(fig, keys=motifs.meaning_keys())
    (same,) = [f for f in found if f.kind == "same-frame"]
    assert same.severity == "fault"
    assert set(same.keys) == {"a", "b"}
    assert same.de == pytest.approx(0.0)
    assert "unkeyed" not in [f.kind for f in found]


def test_meaning_keys_passes_distinct_hues_drawn_with_motif_kwargs():
    hwostyle.use("dark")
    motifs = ep.MotifSet([_m("a", "planet"), _m("b", "star")])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    for y, motif in zip((3, 6), motifs.values(), strict=True):
        ax.plot([1, 9], [y, y], **motif.line_kw())
    assert ept.meaning_faults(fig, keys=motifs.meaning_keys()) == []


def test_the_fill_is_stronger_on_a_dark_page():
    disk = _m("disk", "disk", fill_alpha=0.12, fill_alpha_dark=0.22)
    assert disk.fill("dark") == 0.22
    assert disk.fill("light") == 0.12


def test_a_set_is_keyed_by_name_and_reports_the_options():
    motifs = ep.MotifSet([_m("a", "star"), _m("b", "planet")])
    assert list(motifs) == ["a", "b"]
    assert motifs["b"].role == "planet"
    with pytest.raises(KeyError, match="Available"):
        motifs["c"]
    with pytest.raises(ValueError, match="duplicate"):
        ep.MotifSet([_m("a", "star"), _m("a", "planet")])


def test_only_derive_and_add_return_new_sets():
    motifs = ep.MotifSet([_m("a", "star"), _m("b", "planet")])
    assert list(motifs.only("b")) == ["b"]
    assert list(motifs) == ["a", "b"]
    assert motifs.derive("a", marker="*")["a"].marker == "*"
    assert list(motifs.add(_m("c", "model"))) == ["a", "b", "c"]
    with pytest.raises(KeyError):
        motifs.only("missing")


def test_the_default_set_has_the_core_objects_in_every_mode(default_set):
    for key in ("star", "planet", "disk", "model", "truth", "prior", "cue"):
        assert key in default_set
    for mode in (*MODES, "barbie"):
        for motif in default_set.values():
            assert motif.resolve(mode).startswith("#")


def test_default_set_shares_its_never_together_pairs(default_set):
    assert default_set.never_together("disk", "prior")
    assert default_set.never_together("prior", "disk")
    assert default_set.never_together("pupil", "probability")
    assert not default_set.never_together("star", "planet")


def test_the_cue_resolves_to_the_text_color(default_set):
    for mode in MODES:
        hwostyle.use(mode)
        text = plt.rcParams["text.color"]
        assert default_set["cue"].resolve(None).lower() == _hex(text)


def _hex(color):
    from matplotlib.colors import to_hex

    return to_hex(color).lower()


def test_the_dim_level_follows_the_mode():
    assert ep._style.dim_level("dark") == 0.35
    assert ep._style.dim_level("light") == 0.5
    assert ep._style.dim_level("paper") == 0.5


def test_stage_dim_resolves_by_mode_and_an_explicit_value_wins():
    for mode, expected in (("dark", 0.35), ("light", 0.5), ("paper", 0.5)):
        hwostyle.use(mode)
        fig, ax = plt.subplots()
        (line,) = ax.plot([0, 1], [0, 1])
        assert ep.Stage({"a": [line]}).dim == expected
        assert ep.Stage({"a": [line]}, dim=0.2).dim == 0.2
        plt.close(fig)


# checks


def test_two_motifs_with_one_color_and_one_signature_are_a_fault():
    hwostyle.use("light")
    pair = ep.MotifSet([_m("a", "model"), _m("b", "model")])
    (fault,) = [f for f in ept.motif_faults(pair) if f.kind == "signature"]
    assert fault.keys == ("a", "b")
    with pytest.raises(ept.MotifError, match="signature"):
        ept.assert_motifs(pair)
    apart = pair.derive("b", linestyle="--")
    assert "signature" not in [f.kind for f in ept.motif_faults(apart)]


def test_a_sharing_pair_in_one_set_is_a_fault(default_set):
    hwostyle.use("light")
    both = default_set.only("disk", "prior")
    assert "sharing" in [f.kind for f in ept.motif_faults(both)]
    assert "sharing" not in [f.kind for f in ept.motif_faults(default_set.only("disk"))]


def test_the_hue_budget_needs_a_second_channel_past_four():
    hwostyle.use("dark")
    base = [
        _m("a", "star", marker="*"),
        _m("b", "planet", marker="o"),
        _m("c", "model", marker="x"),
        _m("d", "disk", hatch="//"),
    ]
    five = ep.MotifSet([*base, _m("e", "alert")])
    kinds = [f.kind for f in ept.motif_faults(five)]
    assert "budget" in kinds
    with_channel = ep.MotifSet([*base, _m("e", "alert", marker="X")])
    assert "budget" not in [f.kind for f in ept.motif_faults(with_channel)]
    assert "budget" not in [f.kind for f in ept.motif_faults(ep.MotifSet(base))]


def test_a_seventh_hue_is_always_a_fault():
    hwostyle.use("dark")
    roles = ["star", "planet", "model", "disk", "alert", "posterior", "positive"]
    seven = ep.MotifSet([_m(r, r, marker="o", hatch="/") for r in roles])
    msgs = [f.detail for f in ept.motif_faults(seven) if f.kind == "budget"]
    assert any("never use more than six" in m for m in msgs)


def test_confusable_hues_are_a_fault_unless_a_second_channel_separates_them():
    plain = ep.MotifSet([_m("a", "alert"), _m("r", "reference")])
    (fault,) = [f for f in ept.motif_faults(plain, mode="dark") if f.kind == "cvd"]
    assert fault.severity == "fault"
    assert 5.0 < fault.de < 10.0
    marked = ep.MotifSet([_m("a", "alert"), _m("r", "reference", marker="o")])
    (soft,) = [f for f in ept.motif_faults(marked, mode="dark") if f.kind == "cvd"]
    assert soft.severity == "warning"
    far = ep.MotifSet([_m("s", "star"), _m("p", "planet")])
    assert "cvd" not in [f.kind for f in ept.motif_faults(far, mode="dark")]


def test_dark_star_and_zodi_stay_apart_under_every_vision_type(table_set):
    margins = ept.motif_margins(table_set.only("star", "local_zodi"), mode="dark")
    assert margins["min_de"] > 20.0


def test_margins_report_the_dimmed_minimum_at_the_mode_level():
    motifs = ep.MotifSet([_m("a", "star"), _m("b", "planet"), _m("c", "model")])
    dark = ept.motif_margins(motifs, mode="dark")
    light = ept.motif_margins(motifs, mode="light")
    assert dark["dim"] == 0.35
    assert light["dim"] == 0.5
    assert dark["hues"] == 3
    assert dark["dim_min_de"] < dark["min_de"]
    assert ept.motif_margins(ep.MotifSet([_m("a", "star")]))["min_de"] is None


def test_the_cue_must_be_a_form_in_the_text_color():
    hwostyle.use("light")
    good = ep.MotifSet([_m("cue", "text", linewidth=2.0, fill_alpha=0.25)])
    assert ept.motif_faults(good, mode="light") == []
    hue = ep.MotifSet([_m("cue", "model", fill_alpha=0.25)])
    assert "cue" in [f.kind for f in ept.motif_faults(hue, mode="light")]
    marker = ep.MotifSet([_m("cue", "text", marker="o")])
    assert any("never a marker" in f.detail for f in ept.motif_faults(marker))


def test_a_motif_in_the_cue_color_is_flagged():
    clash = ep.MotifSet([_m("cue", "text"), _m("label", "text", fontstyle="italic")])
    found = [f for f in ept.motif_faults(clash, mode="light") if f.kind == "cue"]
    assert [f.keys for f in found] == [("cue", "label")]


def test_four_core_hues_pass_in_every_mode(table_set):
    four = table_set.only("star", "planet", "disk", "model")
    for mode in MODES:
        faults = [f for f in ept.motif_faults(four, mode=mode) if f.severity == "fault"]
        assert faults == [], (mode, faults)


# the sharing check reads roles as well as keys


def test_a_sharing_pair_fires_by_role_when_a_custom_motif_takes_the_role():
    """Local zodi with any motif whose role is reference is declared never-together."""
    motifs = ep.MotifSet(
        [_m("local_zodi", "zodi", hatch=".."), _m("curve", "reference")],
        sharing=[("local_zodi", "reference")],
    )
    (fault,) = [
        f for f in ept.motif_faults(motifs, mode="light") if f.kind == "sharing"
    ]
    assert fault.keys == ("local_zodi", "curve")
    alone = ep.MotifSet(
        [_m("local_zodi", "zodi", hatch="..")], sharing=[("local_zodi", "reference")]
    )
    assert "sharing" not in [f.kind for f in ept.motif_faults(alone, mode="light")]


def test_an_image_role_exclusion_fires_for_motifs_keyed_by_the_role_names():
    motifs = ep.MotifSet(
        [_m("pupil", "hardware", marker="o"), _m("probability", "posterior")],
        exclusions=[("pupil", "probability")],
    )
    kinds = [
        f.kind for f in ept.motif_faults(motifs, mode="light") if f.kind == "sharing"
    ]
    assert kinds == ["sharing"]


# the thresholds and the vision types, each with a control that passes


def _cvd(motifs, mode):
    return [f for f in ept.motif_faults(motifs, mode=mode) if f.kind == "cvd"]


def test_below_the_hard_floor_a_second_channel_does_not_soften_the_finding():
    """Spectral star against planet is 4.5 apart: a fault even with a marker."""
    hwostyle.use("dark", palette_family="spectral")
    marked = ep.MotifSet([_m("s", "star", marker="o"), _m("p", "planet")])
    (fault,) = _cvd(marked, None)
    assert fault.de < 5.0
    assert fault.severity == "fault"
    # above the floor the same marker softens it (dark alert against reference)
    above = ep.MotifSet([_m("a", "alert", marker="o"), _m("r", "reference")])
    (soft,) = _cvd(above, "dark")
    assert 5.0 < soft.de < 10.0
    assert soft.severity == "warning"


def test_a_pair_that_is_close_only_under_tritanomaly_is_found():
    """Dark model against the verdict orange: 9.6 under tritanomaly, 40+ otherwise."""
    pair = ep.MotifSet([_m("m", "model"), _m("n", "negative")])
    (fault,) = _cvd(pair, "dark")
    assert fault.severity == "fault"
    assert fault.de == pytest.approx(9.6, abs=0.2)
    # a pair far apart under tritanomaly too is left alone
    far = ep.MotifSet([_m("s", "star"), _m("p", "planet")])
    assert _cvd(far, "dark") == []


def test_a_pair_between_ten_and_twenty_apart_only_under_tritanomaly_warns():
    """Paper disk against alert: 11.8 under tritanomaly, 20 or more otherwise."""
    pair = ep.MotifSet([_m("d", "disk"), _m("a", "alert")])
    (warn,) = _cvd(pair, "paper")
    assert warn.severity == "warning"
    assert 10.0 < warn.de < 20.0


def test_a_declared_never_together_pair_is_skipped_by_the_distance_check():
    motifs = ep.MotifSet([_m("a", "alert"), _m("r", "reference")])
    assert _cvd(motifs, "dark")
    declared = ep.MotifSet(
        [_m("a", "alert"), _m("r", "reference")], sharing=[("a", "r")]
    )
    assert _cvd(declared, "dark") == []
    assert "sharing" in [f.kind for f in ept.motif_faults(declared, mode="dark")]


def test_dimming_blends_toward_the_background_and_never_away():
    """At full contrast the dimmed distance is the plain one; at none it collapses."""
    motifs = ep.MotifSet([_m("a", "star"), _m("b", "planet")])
    for mode in ("dark", "light"):
        full = ept.motif_margins(motifs, mode=mode, dim=1.0)
        assert full["dim_min_de"] == pytest.approx(full["min_de"], abs=0.5)
        gone = ept.motif_margins(motifs, mode=mode, dim=0.0)
        assert gone["dim_min_de"] == pytest.approx(0.0, abs=0.5)
        half = ept.motif_margins(motifs, mode=mode, dim=0.5)
        assert 0.0 < half["dim_min_de"] < full["dim_min_de"]


def test_a_pair_that_collides_once_dimmed_gets_an_advisory_and_a_far_pair_does_not():
    close = ep.MotifSet([_m("d", "disk", hatch="///"), _m("m", "model")])
    (note,) = [f for f in ept.motif_faults(close, mode="dark") if f.kind == "dimmed"]
    assert note.severity == "advisory"
    assert note.de < 10.0
    far = ep.MotifSet([_m("s", "star"), _m("p", "planet")])
    assert "dimmed" not in [f.kind for f in ept.motif_faults(far, mode="dark", dim=1.0)]
    ept.assert_motifs(close, mode="dark")  # advisory findings never raise


def test_truth_is_a_gray_and_does_not_spend_a_hue():
    four = [
        _m("a", "star", marker="*"),
        _m("b", "planet", marker="o"),
        _m("c", "model", marker="x"),
        _m("d", "disk", hatch="//"),
    ]
    with_truth = ep.MotifSet([*four, _m("t", "truth", linestyle="--")])
    assert not with_truth["t"].is_hue
    assert ept.motif_margins(with_truth, mode="dark")["hues"] == 4
    assert "budget" not in [f.kind for f in ept.motif_faults(with_truth, mode="dark")]
    with_hue = ep.MotifSet([*four, _m("t", "posterior")])
    assert with_hue["t"].is_hue
    assert "budget" in [f.kind for f in ept.motif_faults(with_hue, mode="dark")]


def test_a_hex_role_is_refused():
    with pytest.raises(ValueError, match="hex"):
        _m("star", "#F5D300")


def test_a_non_solid_line_style_counts_as_a_second_channel():
    assert _m("a", "model", linestyle="--").has_second_channel
    assert not _m("a", "model").has_second_channel
    assert _m("a", "star", marker="*", linestyle="none").has_second_channel


def test_derive_on_an_unknown_key_raises():
    motifs = ep.MotifSet([_m("a", "star")])
    with pytest.raises(KeyError, match="No motif"):
        motifs.derive("b", marker="o")


def test_the_dim_level_falls_back_when_hwostyle_has_no_motifs(monkeypatch):
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "hwostyle.motifs":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert ep._style.dim_level("dark") == 0.35
    assert ep._style.dim_level("light") == 0.5
    assert ep._style.dim_level("paper") == 0.5


def test_the_stand_in_table_matches_the_hwostyle_table_row_for_row():
    table = pytest.importorskip("hwostyle.motifs")
    standin = [ep.Motif(**r) for r in motif_module._FALLBACK_ROWS]
    real = ep.MotifSet.default()
    assert [m.key for m in standin] == list(real)
    for mine in standin:
        assert mine == real[mine.key], mine.key
    assert motif_module._FALLBACK_RANKS == dict(table.NEUTRAL_RANKS)
    assert motif_module._FALLBACK_SHARING == tuple(table.SHARING_SETS)
    assert motif_module._FALLBACK_EXCLUSIONS == tuple(table.EXCLUSION_PAIRS)
    assert motif_module._FALLBACK_ACCEPTED == tuple(
        motif_module.AcceptedPair(r.a, r.b, r.modes, r.channels, r.de)
        for r in table.ACCEPTED_PAIRS
    )


def test_step_list_finished_steps_use_the_mode_dim_level():
    for mode, level in (("dark", 0.35), ("light", 0.5)):
        hwostyle.use(mode)
        res = ep.step_list(["one", "two"], current=1)
        faint = res.artists["text"][0].get_color()
        assert tuple(faint) == pytest.approx(ep._style.neutral(level), abs=1e-6)
        plt.close("all")


def test_a_label_reads_the_ink_where_hwostyle_has_one_and_the_mark_elsewhere(
    table_set,
):
    zodi = table_set["local_zodi"]
    assert zodi.label_color == "zodi"
    hwostyle.use("dark")
    assert zodi.label_kw()["color"] == "#15903e"
    assert zodi.color == "#117733"
    assert zodi.resolve_label("light") == "#2e7d32"
    assert zodi.resolve_label("paper") == zodi.resolve("paper")
    assert table_set["star"].label_kw()["color"] == table_set["star"].color
    assert table_set["reference_plane"].label_kw()["fontstyle"] == "italic"


def test_a_label_falls_back_to_the_mark_without_a_table(monkeypatch):
    monkeypatch.setattr(motif_module, "_hwostyle_table", lambda: None)
    zodi = ep.MotifSet.default()["local_zodi"]
    hwostyle.use("dark")
    assert zodi.label_color == "zodi"
    assert zodi.label in (zodi.color, "#15903e")


def test_the_default_sets_local_zodi_is_never_with_alert_or_posterior(default_set):
    assert default_set.never_together("local_zodi", "alert")
    assert default_set.never_together("posterior", "local_zodi")
    both = default_set.only("local_zodi", "alert")
    assert "sharing" in [f.kind for f in ept.motif_faults(both, mode="dark")]


def test_an_accepted_pair_passes_only_with_its_second_channels(table_set):
    ref_line = _m("ref", "reference", linestyle="--")
    alert_x = _m("alert", "alert", marker="X", linestyle="none")
    accepted = table_set.only().add(ref_line, alert_x)
    assert accepted.accepts("ref", "alert", "dark")
    assert not accepted.accepts("ref", "alert", "light")
    assert _cvd(accepted, "dark") == []
    plain = table_set.only().add(_m("ref", "reference"), _m("alert", "alert"))
    assert not plain.accepts("ref", "alert", "dark")
    (fault,) = _cvd(plain, "dark")
    assert fault.severity == "fault"
    assert fault.de == pytest.approx(7.3, abs=0.2)
    marker_only = table_set.only().add(
        _m("ref", "reference"), _m("alert", "alert", marker="X")
    )
    (soft,) = _cvd(marker_only, "dark")
    assert soft.severity == "warning"


def test_the_verdict_residual_is_accepted_in_paper_with_a_marker_difference(
    table_set,
):
    hwostyle.use("paper")
    pair = table_set.only("negative").add(_m("ref", "reference", linestyle="--"))
    assert pair.accepts("ref", "negative", "paper")
    assert _cvd(pair, "paper") == []
    same_marker = table_set.only().add(
        _m("ref", "reference", marker="s"), table_set["negative"]
    )
    assert not same_marker.accepts("ref", "negative", "paper")


def test_model_and_negative_are_accepted_in_dark_with_line_and_marker(default_set):
    hwostyle.use("dark")
    pair = default_set.only("model", "negative")
    assert pair["model"].marker != pair["negative"].marker
    assert pair["model"].linestyle != pair["negative"].linestyle
    assert pair.accepts("model", "negative", "dark")
    assert pair.accepts("negative", "model", "dark")
    assert _cvd(pair, "dark") == []
    same = pair.derive("model", marker=pair["negative"].marker)
    assert not same.accepts("model", "negative", "dark")
    (finding,) = _cvd(same, "dark")
    assert set(finding.keys) == {"model", "negative"}
    for mode in ("light", "paper"):
        assert not pair.accepts("model", "negative", mode)


def test_zodi_negative_and_prior_positive_are_accepted_in_paper_with_a_marker(
    default_set,
):
    hwostyle.use("paper")
    for a, b in (("local_zodi", "negative"), ("prior", "positive")):
        pair = default_set.only(a, b)
        assert pair[a].marker != pair[b].marker
        assert pair.accepts(a, b, "paper")
        assert _cvd(pair, "paper") == []
        same = pair.derive(a, marker=pair[b].marker)
        assert not same.accepts(a, b, "paper")
        assert len(_cvd(same, "paper")) == 1


def test_disk_and_positive_are_accepted_in_paper_with_a_marker_difference(default_set):
    hwostyle.use("paper")
    pair = default_set.only("disk", "positive")
    assert pair["disk"].marker != pair["positive"].marker
    assert pair.accepts("disk", "positive", "paper")
    assert pair.accepts("positive", "disk", "paper")
    assert _cvd(pair, "paper") == []
    same_marker = pair.derive("disk", marker=pair["positive"].marker)
    assert not same_marker.accepts("disk", "positive", "paper")
    assert not same_marker.accepts("positive", "disk", "paper")
    (finding,) = _cvd(same_marker, "paper")
    assert set(finding.keys) == {"disk", "positive"}
    for mode in ("dark", "light"):
        assert not pair.accepts("disk", "positive", mode)
        assert not pair.accepts("positive", "disk", mode)
