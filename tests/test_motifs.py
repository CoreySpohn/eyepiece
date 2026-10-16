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
    assert disk.line_kw() == {"color": disk.color, "ls": "-", "lw": 0.8}
    point = _m("star", "star", marker="*", linestyle="none")
    assert point.marker_kw(size=9) == {
        "color": point.color,
        "marker": "*",
        "ls": "none",
        "ms": 9,
    }


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
    plain = ep.MotifSet([_m("m", "model"), _m("r", "reference")])
    (fault,) = [f for f in ept.motif_faults(plain, mode="light") if f.kind == "cvd"]
    assert fault.severity == "fault"
    assert 5.0 < fault.de < 10.0
    marked = ep.MotifSet([_m("m", "model"), _m("r", "reference", marker="o")])
    (soft,) = [f for f in ept.motif_faults(marked, mode="light") if f.kind == "cvd"]
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
