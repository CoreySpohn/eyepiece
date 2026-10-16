"""Plan-driven clip checks, with a clean control for each planted fault."""

from dataclasses import FrozenInstanceError

import matplotlib.pyplot as plt
import pytest
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece import narration as nr


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _marker(fig, x=40, y=80, **kwargs):
    artist = Line2D(
        [x],
        [y],
        marker="o",
        markersize=12,
        linestyle="none",
        transform=IdentityTransform(),
        **kwargs,
    )
    fig.add_artist(artist)
    return artist


def _clip(*, jump=False, unread=False, read_extra=False):
    fig = plt.figure(figsize=(4, 2), dpi=100)
    dot = _marker(fig, color="black")
    knobs = {"x": 0.0, **({"extra": 0.0} if unread else {})}
    tl = ep.Timeline(knobs, tempo=ep.Tempo(factor=1.0))
    tl.hold(0.3)
    tl.run("slide", 0.8, to={"x": 8.0})
    tl.hold(0.3)
    plan = tl.at_fps(10)

    def draw(k):
        x = plan.value(k, "x")
        if read_extra:
            x += plan.frames[k]["extra"]
        dot.set_xdata([40 + x + (80 if jump and k >= 7 else 0)])

    return fig, draw, plan, dot


def _kinds(report):
    return [kind for kind, _ in report.faults]


def test_clean_clip_and_assertion_pass():
    fig, draw, plan, dot = _clip()
    report = ept.check_clip(fig, draw, plan, moves={dot: "slide"})
    assert isinstance(report, ept.ClipReport)
    assert report.ok and report.faults == []
    assert all(value == [] for value in vars(report).values())
    assert "clean" in str(report)
    assert dot.get_xdata() == [48.0]
    with pytest.raises(FrozenInstanceError):
        report.motion = []
    assert ept.assert_clip(fig, draw, plan, moves={dot: "slide"}).ok


def test_jump_is_reported_once_and_assertion_names_it():
    fig, draw, plan, _ = _clip(jump=True)
    report = ept.check_clip(fig, draw, plan)
    assert not report.ok
    assert _kinds(report) == ["jump"]
    assert len(report.motion) == 1
    assert report.motion[0].time == pytest.approx(0.7)
    with pytest.raises(ept.ClipError, match="jump") as exc:
        ept.assert_clip(fig, draw, plan)
    assert isinstance(exc.value, AssertionError)
    assert exc.value.report.faults == exc.value.faults
    fig, draw, plan, _ = _clip()
    assert ept.check_clip(fig, draw, plan).ok


def test_unread_knob_ignores_prior_reads_and_is_reported_once():
    fig, draw, plan, _ = _clip(unread=True)
    plan.value(0, "extra")
    prior = set(plan._read)
    report = ept.check_clip(fig, draw, plan)
    assert report.unread_knobs == ["extra"]
    assert _kinds(report) == ["unread-knob"]
    assert plan._read == prior
    assert ept.check_clip(fig, draw, plan).faults == report.faults
    fig, draw, plan, _ = _clip(unread=True, read_extra=True)
    assert ept.check_clip(fig, draw, plan).ok


def test_knob_read_on_a_non_review_frame_counts():
    fig, draw, plan, _ = _clip(unread=True)
    k_read = next(k for k in range(plan.n_frames) if k not in plan.review_frames())

    def sparse_read(k):
        draw(k)
        if k == k_read:
            plan.value(k, "extra")

    assert ept.check_clip(fig, sparse_read, plan).ok
    assert plan._read == set()


def test_draw_failure_restores_knob_reads():
    fig, _, plan, _ = _clip(unread=True)
    plan.value(0, "extra")
    reads = plan._read

    def broken(k):
        plan.value(k, "x")
        raise RuntimeError("draw failed")

    with pytest.raises(RuntimeError, match="draw failed"):
        ept.check_clip(fig, broken, plan)
    assert plan._read is reads and reads == {"extra"}


def _motif_clip(*, same_hue):
    fig = plt.figure(figsize=(4, 2), dpi=100)
    motifs = ep.MotifSet(
        [
            ep.Motif("first", "first", "model"),
            ep.Motif("second", "second", "model" if same_hue else "text"),
        ]
    )
    for y, motif in zip((60, 130), motifs.values(), strict=True):
        fig.add_artist(
            Line2D([40, 200], [y, y], transform=IdentityTransform(), **motif.line_kw())
        )
    tl = ep.Timeline()
    tl.hold(1.0)
    return fig, lambda k: None, tl.at_fps(10), motifs


def test_same_hue_motifs_are_reported_once_by_kind():
    fig, draw, plan, motifs = _motif_clip(same_hue=True)
    report = ept.check_clip(fig, draw, plan, motifs=motifs)
    assert _kinds(report).count("same-frame") == 1
    assert len([f for f in report.meaning if f.kind == "same-frame"]) == 1
    assert not report.ok
    # Supplying no keys skips meaning, even for the same drawn pair.
    assert ept.check_clip(fig, draw, plan).ok
    # Explicit keys win and can declare that both lines have one meaning.
    assert ept.check_clip(
        fig, draw, plan, motifs=motifs, keys={"first": "one", "second": "one"}
    ).ok
    fig, draw, plan, motifs = _motif_clip(same_hue=False)
    assert ept.check_clip(fig, draw, plan, motifs=motifs).ok


def _narrated_plan(wpm=120, *, cue=False):
    text = "A small {indicate:dot}point moves." if cue else "A small point moves."
    line = nr.Line("a", text, pad_after_s=0.0)
    step = 60 / wpm
    words = tuple(
        nr.Word(word, i * step, (i + 1) * step)
        for i, word in enumerate(("A", "small", "point", "moves."))
    )
    narration = nr.Narration(nr.Script((line,)), {"a": nr.Spoken("a", 4 * step, words)})
    tl = ep.Timeline(narration=narration)
    with tl.line("a"):
        if cue:
            tl.pulse("dot")
        else:
            tl.hold(0.5)
    return tl.at_fps(10)


def test_review_checks_sample_cue_frames_and_motion_samples_every_frame(monkeypatch):
    plan = _narrated_plan(cue=True)
    fig = plt.figure(figsize=(4, 2), dpi=100)
    _marker(fig, color="black")
    samples = {}
    for name in ("readability_faults", "meaning_faults", "motion_faults"):
        original = getattr(ept, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            samples[_name] = [ep.frame_count(t, plan.fps) for t in kwargs["times"]]
            return _original(*args, **kwargs)

        monkeypatch.setattr(ept, name, spy)
    drawn = []
    assert ept.check_clip(fig, drawn.append, plan, keys={}).ok
    review = list(plan.review_frames())
    assert samples["readability_faults"] == samples["meaning_faults"] == review
    assert samples["motion_faults"] == list(range(plan.n_frames))
    for event in plan.events(what="cue"):
        assert event.frame in review and event.before in review
    assert plan.events(what="cue")
    assert drawn == review + review + list(range(plan.n_frames))


def test_speech_rate_is_reported_and_has_a_clean_control():
    fig = plt.figure(figsize=(4, 2), dpi=100)
    _marker(fig, color="black")
    report = ept.check_clip(fig, lambda k: None, _narrated_plan(wpm=240))
    assert _kinds(report) == ["speech-rate"]
    assert report.speech == [("a", pytest.approx(240))]
    assert ept.check_clip(fig, lambda k: None, _narrated_plan()).ok


def test_plan_faults_are_included_and_assertion_names_every_kind():
    fig, draw, plan, _ = _clip(unread=True, jump=True)
    tl = ep.Timeline(strict=False)
    tl.hold(0.01, name="blink")
    tl.hold(1.0)
    faulty = tl.at_fps(10)
    plan.faults = faulty.faults
    report = ept.check_clip(fig, draw, plan)
    assert report.plan_faults == list(faulty.faults)
    assert set(_kinds(report)) == {"jump", "unread-knob", "collapsed"}
    with pytest.raises(ept.ClipError) as exc:
        ept.assert_clip(fig, draw, plan)
    for kind in _kinds(report):
        assert kind in str(exc.value)
    fig, draw, plan, _ = _clip()
    assert ept.check_clip(fig, draw, plan).ok


def test_persistent_readability_fault_is_reported_once():
    fig, draw, plan, _ = _clip()
    label = fig.text(0.5, 0.8, "small", fontsize=5)
    report = ept.check_clip(fig, draw, plan)
    assert _kinds(report) == ["small"]
    assert len(report.readability) == 1
    label.set_fontsize(16)
    assert ept.check_clip(fig, draw, plan).ok


@pytest.mark.parametrize("form", ["name", "span", "declaration"])
def test_move_spans_support_names_spans_and_motion_declarations(form):
    fig, draw, plan, dot = _clip()
    span = plan.spans["slide"]
    value = ept.Span(span.start_s, span.end_s) if form == "span" else "slide"
    moves = (
        [{"artists": [dot], "span": value}] if form == "declaration" else {dot: value}
    )
    assert ept.check_clip(fig, draw, plan, moves=moves).ok

    def stuck(k):
        plan.value(k, "x")
        dot.set_xdata([40])

    report = ept.check_clip(fig, stuck, plan, moves=moves)
    assert _kinds(report) == ["dead"]


def test_public_names():
    for name in ("check_clip", "assert_clip", "ClipReport", "ClipError"):
        assert name in ept.__all__
        assert hasattr(ept, name)


def test_info_plan_findings_are_kept_but_do_not_fail_the_clip():
    fig, draw, plan, _ = _clip()
    plan.faults = [ep.Fault("skipped", "info", "pulse 'x' is placed only by a cue")]
    report = ept.check_clip(fig, draw, plan)
    assert report.plan_faults == plan.faults
    assert report.ok
    plan.faults = [ep.Fault("collapsed", "error", "span 'blink' is too short")]
    assert not ept.check_clip(fig, draw, plan).ok


def _stepped_clip(*, jump=False):
    fig = plt.figure(figsize=(4, 2), dpi=100)
    dot = _marker(fig, color="black")
    mover = _marker(fig, y=140, color="black")
    tl = ep.Timeline({"x": 40.0, "travel": 0.0}, tempo=ep.Tempo(factor=1.0))
    tl.run("traveling", 2.0, to={"travel": 8.0}, at=0.0)
    tl.move("slide", 0.2, at=0.3)
    tl.move("late", 0.2, at=0.8)
    for x in (40.0, 100.0, 160.0, 220.0):
        tl.hold(0.5, to={"x": x})
    plan = tl.at_fps(10)

    def draw(k):
        dot.set_xdata([plan.value(k, "x")])
        mover.set_xdata([40 + plan.value(k, "travel") + (80 if jump and k >= 8 else 0)])

    return fig, draw, plan


def test_discrete_hold_sweep_has_no_jump():
    fig, draw, plan = _stepped_clip()
    assert ept.check_clip(fig, draw, plan).ok
    assert ept.check_clip(fig, draw, plan, steps=[]).ok


def test_discrete_hold_sweep_reports_only_the_unplanned_jump():
    fig, draw, plan = _stepped_clip(jump=True)
    report = ept.check_clip(fig, draw, plan)
    assert _kinds(report) == ["jump"]
    assert len(report.motion) == 1
    assert report.motion[0].time == pytest.approx(0.8)


@pytest.mark.parametrize(("step", "frame"), [("slide", 3), (3, 3), ("late", 8), (8, 8)])
def test_explicit_steps_add_span_starts_or_frame_indices(step, frame):
    fig, draw, plan = _stepped_clip()
    extra = _marker(fig, y=30, color="black")

    def with_jump(k):
        draw(k)
        extra.set_xdata([40 + (80 if k >= frame else 0)])

    assert ept.check_clip(fig, with_jump, plan, steps=[step]).ok
    other = 8 if frame == 3 else 3
    assert not ept.check_clip(fig, with_jump, plan, steps=[other]).ok


def test_unknown_step_span_names_the_missing_span():
    fig, draw, plan, _ = _clip()
    with pytest.raises(KeyError, match="missing-step"):
        ept.check_clip(fig, draw, plan, steps=["missing-step"])


def test_step_frames_ignore_no_op_writes_and_smooth_changes():
    tl = ep.Timeline({"x": 0.0}, tempo=ep.Tempo(factor=1.0))
    tl.hold(0.2, to={"x": 0.0})
    tl.hold(0.2, to={"x": 1.0})
    tl.hold(0.2, to={"x": 1.0})
    tl.run("ramp", 0.2, to={"x": 2.0})
    tl.hold(0.2, to={"x": 2.0})
    tl.hold(0.2, to={"x": 0.0})
    plan = tl.at_fps(10)
    assert plan.step_frames() == (2, 10)
    assert plan.value(1, "x") == 0.0
    assert plan.value(2, "x") == 1.0
    assert plan.value(9, "x") == 2.0
    assert plan.value(10, "x") == 0.0


def test_step_frames_merge_two_knobs_and_include_initial_changes():
    tl = ep.Timeline({"x": 0.0, "label": "first"}, tempo=ep.Tempo(factor=1.0))
    tl.hold(0.2, to={"x": 1.0, "label": "second"})
    tl.hold(0.2, to={"x": 1.0, "label": "second"})
    tl.hold(0.2, to={"label": "third"})
    tl.hold(0.2, to={"x": 2.0, "label": "fourth"})
    tl.move("rename", 0.2, to={"label": "last"})
    assert tl.at_fps(10).step_frames() == (0, 4, 6, 8)


@pytest.mark.parametrize("bad", [-1, 10**6, 2.5, None])
def test_a_step_outside_the_plan_is_refused(bad):
    fig, draw, plan, _ = _clip()
    with pytest.raises(ValueError, match="frame index"):
        ept.check_clip(fig, draw, plan, steps=[bad])


def test_a_bare_span_name_is_one_step():
    fig, draw, plan = _stepped_clip()
    assert ept.check_clip(fig, draw, plan, steps="slide").ok
