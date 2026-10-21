"""Renamed public names: each old name warns and resolves to its replacement."""

import warnings

import pytest
from matplotlib.colors import to_hex

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece import motifs


def test_testing_span_is_window_with_a_warning():
    with pytest.warns(DeprecationWarning, match=r"eyepiece\.testing\.Window"):
        old = ept.Span
    assert old is ept.Window
    assert "Window" in ept.__all__
    assert "Span" not in ept.__all__


def test_from_import_of_testing_span_warns():
    with pytest.warns(DeprecationWarning, match="Window"):
        from eyepiece.testing import Span
    assert Span(0.0, 1.0) == ept.Window(0.0, 1.0)


def test_spans_is_shade_intervals_with_a_warning():
    with pytest.warns(DeprecationWarning, match=r"eyepiece\.shade_intervals"):
        old = ep.spans
    assert old is ep.shade_intervals
    assert "shade_intervals" in ep.plot.__all__
    assert "spans" not in ep.__all__


def test_deprecated_name_warns_on_every_access():
    for _ in range(2):
        with pytest.warns(DeprecationWarning):
            ep.spans  # noqa: B018
    assert "spans" not in vars(ep)


@pytest.mark.parametrize("name", ["planet", "model", "truth", "hardware", "text"])
@pytest.mark.parametrize("mode", [None, "dark", "light", "paper"])
def test_resolve_role_is_role_with_a_warning(name, mode):
    with pytest.warns(DeprecationWarning, match=r"eyepiece\.role"):
        old = motifs.resolve_role(name, mode)
    assert old == ep.role(name, mode=mode)
    assert "resolve_role" not in motifs.__all__


def test_role_takes_a_neutral_rank_as_its_gray():
    assert ep.role("hardware") == to_hex(ep.neutral(0.7))
    assert ep.role("text") == to_hex(ep.neutral(1.0))
    with pytest.raises(KeyError, match="neutral ranks"):
        ep.role("nonsense")


@pytest.mark.parametrize("old,new", [("pulse", "indicate"), ("ring", "circumscribe")])
def test_old_timeline_verbs_warn_and_place_the_same_span(old, new):
    current = ep.Timeline()
    current.hold(1.0)
    placed = getattr(current, new)("dot", at=0.2)
    legacy = ep.Timeline()
    legacy.hold(1.0)
    with pytest.warns(DeprecationWarning, match=f"Timeline.{new}"):
        kept = getattr(legacy, old)("dot", at=0.2)
    assert (kept.name, kept.kind) == (f"{old}.dot", old)
    assert (placed.name, placed.kind) == (f"{new}.dot", new)
    assert (kept.start_s, kept.end_s, kept.extends) == (
        placed.start_s,
        placed.end_s,
        placed.extends,
    )
    a, b = current.at_fps(30), legacy.at_fps(30)
    assert [a.value(k, f"{new}.dot") for k in range(a.n_frames)] == [
        b.value(k, f"{old}.dot") for k in range(b.n_frames)
    ]
    assert b.unused_knobs() == a.unused_knobs()


@pytest.mark.parametrize("old,new", [("pulse", "indicate"), ("ring", "circumscribe")])
def test_old_span_kinds_warn_and_place_as_the_new_kind(old, new):
    current = ep.Timeline({"k": 0.0})
    placed = current.span("k", 0.5, kind=new, to=1.0)
    legacy = ep.Timeline({"k": 0.0})
    with pytest.warns(DeprecationWarning, match=repr(new)):
        kept = legacy.span("k", 0.5, kind=old, to=1.0)
    assert kept.kind == old
    assert (kept.start_s, kept.end_s, kept.extends, kept.ease) == (
        placed.start_s,
        placed.end_s,
        placed.extends,
        placed.ease,
    )


def test_talk_60_preset_warns_and_is_the_talk_pair():
    with pytest.warns(DeprecationWarning, match=r"PRESETS\['talk'\]"):
        old = ep.PRESETS["talk_60"]
    assert old == ep.PRESETS["talk"]
    with pytest.warns(DeprecationWarning):
        assert ep.PRESETS.get("talk_60") == ep.PRESETS["talk"]
    assert ep.PRESETS.get("nope", 1) == 1


def _warned_here(call):
    """Run `call` and return the files its deprecation warnings point at."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [w.filename for w in caught if w.category is DeprecationWarning]


@pytest.mark.parametrize(
    "call",
    [
        lambda: ept.Span,
        lambda: ep.spans,
        lambda: motifs.resolve_role("planet"),
        lambda: ep.Timeline().pulse("dot", at=0.0),
        lambda: ep.Timeline().ring("dot", at=0.0),
        lambda: ep.Timeline({"k": 0.0}).span("k", 0.5, kind="pulse", to=1.0),
        lambda: ep.Timeline().stagger("pulse", ["a", "b"], each_s=0.3),
        lambda: ep.PRESETS["talk_60"],
        lambda: ep.PRESETS.get("talk_60"),
    ],
    ids=[
        "testing.Span",
        "spans",
        "resolve_role",
        "pulse",
        "ring",
        "span-kind",
        "stagger",
        "preset",
        "preset-get",
    ],
)
def test_each_deprecation_warns_once_at_the_callers_line(call):
    assert _warned_here(call) == [__file__]
