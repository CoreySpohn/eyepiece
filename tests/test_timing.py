"""Timing rules checked with hand-made word times."""

import importlib.util
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import eyepiece.narration as nr
import eyepiece.testing as ept


def _line(
    key,
    wpm=120.0,
    n_words=20,
    *,
    pad_before_s=0.0,
    pad_after_s=0.5,
    word_offset_s=0.0,
    tail_s=0.0,
    tokens=None,
):
    """A line and its audio timing, with an exact word span and optional tail."""
    tokens = tokens if tokens is not None else [f"w{i}" for i in range(n_words)]
    count = sum(any(ch.isalnum() for ch in token) for token in tokens)
    span = count * 60.0 / wpm
    step = span / len(tokens)
    words = tuple(
        nr.Word(token, word_offset_s + i * step, word_offset_s + (i + 1) * step)
        for i, token in enumerate(tokens)
    )
    line = nr.Line(
        key,
        " ".join(tokens),
        pad_before_s=pad_before_s,
        pad_after_s=pad_after_s,
    )
    return line, nr.Spoken(key, word_offset_s + span + tail_s, words)


def _narration(*entries):
    """Place hand-timed lines and silent beats on one clock."""
    lines, spoken = [], {}
    for entry in entries:
        if isinstance(entry, nr.Pause):
            lines.append(entry)
        else:
            line, timing = entry
            lines.append(line)
            spoken[line.key] = timing
    return nr.Narration(nr.Script(tuple(lines)), spoken)


def test_a_long_line_above_the_ceiling_is_reported():
    narration = _narration(_line("steady", 159), _line("quick", 161))
    faults = ept.timing_faults(narration)
    assert [(f.kind, f.key) for f in faults] == [("speech-rate", "quick")]
    fault = faults[0]
    assert all(part in fault.message for part in ("quick", "20", "161", "160"))
    beat = narration.beats[1]
    assert (fault.start_s, fault.end_s) == (beat.words[0].start_s, beat.words[-1].end_s)


@pytest.mark.parametrize("wpm, reported", [(165, False), (167, True), (166.2, False)])
def test_a_short_line_may_reach_the_short_ceiling(wpm, reported):
    faults = ept.timing_faults(_narration(_line("brief", wpm, 9)))
    assert [(f.kind, f.key) for f in faults] == (
        [("speech-rate", "brief")] if reported else []
    )
    if reported:
        assert all(part in faults[0].message for part in ("brief", "9", "167", "166"))


@pytest.mark.parametrize("n_words, reported", [(10, False), (11, True)])
def test_the_short_word_cutoff_is_inclusive(n_words, reported):
    faults = ept.timing_faults(_narration(_line("boundary", 165, n_words)))
    assert [(f.kind, f.key) for f in faults] == (
        [("speech-rate", "boundary")] if reported else []
    )


def test_the_short_line_allowance_can_be_removed():
    faults = ept.timing_faults(_narration(_line("brief", 165, 9)), short_ceiling=160)
    assert [(f.kind, f.key) for f in faults] == [("speech-rate", "brief")]
    assert all(part in faults[0].message for part in ("9", "165", "160"))


def test_only_tokens_with_letters_or_digits_count_as_spoken_words():
    tokens = ["w0", "-", *[f"w{i}" for i in range(1, 8)], "-", "w8"]
    assert ept.timing_faults(_narration(_line("brief", 165, tokens=tokens))) == []


@pytest.mark.parametrize("gap, reported", [(0.3, True), (0.5, False)])
def test_the_gap_between_heard_words_separates_lines(gap, reported):
    narration = _narration(_line("first", pad_after_s=gap), _line("second"))
    faults = ept.timing_faults(narration)
    assert [(f.kind, f.key) for f in faults] == (
        [("line-gap", "second")] if reported else []
    )
    if reported:
        fault = faults[0]
        assert fault.start_s == narration.beats[0].words[-1].end_s
        assert fault.end_s == narration.beats[1].words[0].start_s
        assert "0.3" in fault.message and "0.4" in fault.message


def test_trailing_audio_silence_counts_toward_the_heard_gap():
    # Sequential audio with a 0.2 s tail and 0.1 s word offset has a 0.3 s gap.
    narration = _narration(
        _line("first", tail_s=0.2, pad_after_s=0),
        _line("second", word_offset_s=0.1, pad_before_s=0),
    )
    first, second = narration.beats
    assert second.speech_start_s - first.speech_end_s == 0
    faults = ept.timing_faults(narration)
    assert [(f.kind, f.key) for f in faults] == [("line-gap", "second")]
    assert faults[0].end_s - faults[0].start_s == pytest.approx(0.3)


def test_a_half_second_audio_tail_already_separates_the_words():
    narration = _narration(
        _line("first", tail_s=0.5, pad_after_s=0),
        _line("second", pad_before_s=0),
    )
    assert ept.timing_faults(narration) == []


def test_a_pause_adds_no_spoken_line_gap_of_its_own():
    narration = _narration(
        _line("first", pad_after_s=0),
        nr.Pause("hold", 3.0, "A still figure"),
        _line("second"),
    )
    assert ept.timing_faults(narration) == []


def test_spoken_lines_are_compared_across_a_pause():
    narration = _narration(
        _line("first", pad_after_s=0),
        nr.Pause("hold", 0.3, "A still figure"),
        _line("second"),
    )
    assert [(f.kind, f.key) for f in ept.timing_faults(narration)] == [
        ("line-gap", "second")
    ]


def test_an_undeclared_silence_is_returned_with_its_interval():
    narration = _narration(_line("first", pad_after_s=3.0))
    faults = ept.timing_faults(narration)
    assert [(f.kind, f.key) for f in faults] == [("undeclared-silence", None)]
    assert (faults[0].start_s, faults[0].end_s) == (10.0, 13.0)
    assert "3.0 s" in faults[0].message


@pytest.mark.parametrize(
    "cue_word, kind", [(16, "late-first-cue"), (0, "early-last-cue")]
)
def test_silence_conversion_preserves_cue_findings(cue_word, kind):
    tokens = [f"w{i}" for i in range(20)]
    tokens[cue_word] = "{show:a}" + tokens[cue_word]
    line = nr.Line("cued", " ".join(tokens))
    words = tuple(
        nr.Word(word, i * 0.5, (i + 1) * 0.5) for i, word in enumerate(line.words)
    )
    narration = _narration((line, nr.Spoken(line.key, 10.0, words)))
    silence = nr.check_silence(narration)
    assert [(f.code, f.key) for f in silence] == [(kind, "cued")]
    faults = ept.timing_faults(narration)
    assert [(f.kind, f.message, f.key, f.start_s, f.end_s) for f in faults] == [
        (f.code, f.message, f.key, f.start_s, f.end_s) for f in silence
    ]


def test_equal_start_times_preserve_rule_order():
    narration = _narration(
        _line("first", pad_after_s=0),
        _line("quick", 170),
    )
    plan = SimpleNamespace(hero_windows=lambda: [(10.0, 11.0)])
    faults = ept.timing_faults(narration, plan=plan)
    assert [f.start_s for f in faults] == [10.0, 10.0, 10.0]
    assert [(f.kind, f.key) for f in faults] == [
        ("speech-rate", "quick"),
        ("line-gap", "quick"),
        ("speech-in-hero", "quick"),
    ]


def test_findings_use_known_kinds_and_are_sorted_with_untimed_findings_last():
    narration = _narration(
        _line("first", pad_after_s=0.3),
        _line("quick", 170, pad_after_s=3.0),
    )
    plan = SimpleNamespace(hero_windows=lambda: [(0.0, 1.0), (2.0, 3.0)])
    faults = ept.timing_faults(narration, plan=plan)
    assert [f.kind for f in faults] == [
        "speech-in-hero",
        "speech-in-hero",
        "line-gap",
        "speech-rate",
        "undeclared-silence",
        "multiple-hero",
    ]
    assert all(f.kind in ept.TIMING_KINDS for f in faults)
    starts = [f.start_s for f in faults if f.start_s is not None]
    assert starts == sorted(starts)
    silence = nr.check_silence(narration, plan)
    converted = [f for f in faults if f.kind not in ("speech-rate", "line-gap")]
    assert [(f.kind, f.message, f.key, f.start_s, f.end_s) for f in converted] == [
        (f.code, f.message, f.key, f.start_s, f.end_s) for f in silence
    ]


def test_assert_timing_returns_an_empty_list_when_clean():
    assert ept.assert_timing(_narration(_line("steady", 159))) == []


def test_assert_timing_reports_every_fault_and_keeps_the_list():
    narration = _narration(_line("quick", 170, pad_after_s=3.0))
    with pytest.raises(ept.TimingError, match="quick") as err:
        ept.assert_timing(narration)
    assert isinstance(err.value.faults, list)
    assert err.value.faults == ept.timing_faults(narration)
    assert [f.kind for f in err.value.faults] == ["speech-rate", "undeclared-silence"]
    for fault in err.value.faults:
        assert any(fault.message in line for line in str(err.value).splitlines())


def test_unimplemented_readouts_name_the_keyword():
    with pytest.raises(NotImplementedError, match="readouts"):
        ept.timing_faults(_narration(_line("steady")), readouts=[])


def _still_narration():
    """Six words from 0.5 to 3.5 s in a four-second clip."""
    return _narration(_line("steady", n_words=6, word_offset_s=0.5))


def _gray_frame(t):
    """A fixed native-size frame."""
    return np.full((90, 160, 3), 128, dtype=np.uint8)


def test_a_still_screen_under_speech_names_the_line_and_run():
    faults = ept.timing_faults(_still_narration(), source=_gray_frame)
    assert len(faults) == 1
    fault = faults[0]
    assert (fault.kind, fault.key, fault.start_s, fault.end_s) == (
        "still-speech",
        "steady",
        0.0,
        4.0,
    )
    assert fault.message == (
        "still screen 0.00 to 4.00 s (4.0 s) under 3.0 s of speech (steady)"
    )


@pytest.mark.parametrize("still_s, reported", [(2.9, True), (3.0, False), (3.1, False)])
def test_still_speech_must_be_strictly_over_the_limit(still_s, reported):
    faults = ept.timing_faults(_still_narration(), source=_gray_frame, still_s=still_s)
    assert [(f.kind, f.key) for f in faults] == (
        [("still-speech", "steady")] if reported else []
    )


@pytest.mark.parametrize("pixels, reported", [(144, False), (14, True)])
def test_stillness_uses_the_changed_pixel_fraction(pixels, reported):
    def source(t):
        frame = _gray_frame(t)
        if t >= 2.0:
            frame.reshape(-1, 3)[:pixels] = 255
        return frame

    faults = ept.timing_faults(_still_narration(), source=source)
    assert [(f.kind, f.key) for f in faults] == (
        [("still-speech", "steady")] if reported else []
    )


def test_a_change_count_equal_to_the_threshold_ends_the_still_run():
    def source(t):
        frame = _gray_frame(t)
        if t >= 2.0:
            frame[:9, :16] = 255
        return frame

    faults = ept.timing_faults(
        _still_narration(), source=source, still_change=144 / (90 * 160)
    )
    assert faults == []


@pytest.mark.parametrize("ignore_bottom, reported", [(0.0, False), (0.12, True)])
def test_the_caption_band_can_be_ignored(ignore_bottom, reported):
    def source(t):
        frame = _gray_frame(t)
        if t >= 2.0:
            frame[-9:, :16] = 255
        return frame

    faults = ept.timing_faults(
        _still_narration(), source=source, ignore_bottom=ignore_bottom
    )
    assert [(f.kind, f.key) for f in faults] == (
        [("still-speech", "steady")] if reported else []
    )


def test_ignored_rows_still_count_toward_the_changed_pixel_threshold():
    def source(t):
        frame = _gray_frame(t)
        if t >= 2.0:
            # 22 pixels exceed 0.16 percent of kept rows, but not the full frame.
            frame[0, :22] = 255
        return frame

    faults = ept.timing_faults(_still_narration(), source=source, ignore_bottom=0.12)
    assert [(f.kind, f.key, f.start_s, f.end_s) for f in faults] == [
        ("still-speech", "steady", 0.0, 4.0)
    ]


@pytest.mark.parametrize("gap, reported", [(2.5, False), (1.0, True), (2.0, False)])
def test_declared_silence_splits_speech_on_a_still_screen(gap, reported):
    narration = _narration(
        _line("first", wpm=100, n_words=2, pad_after_s=0),
        nr.Pause("hold", gap, "A still figure"),
        _line("second", wpm=100, n_words=2, pad_after_s=2.6 - gap),
    )
    assert narration.total_s == pytest.approx(5.0)
    faults = ept.timing_faults(narration, source=_gray_frame)
    assert [(f.kind, f.key) for f in faults] == (
        [("still-speech", "first")] if reported else []
    )
    if reported:
        assert "3.4 s of speech (first, second)" in faults[0].message


def test_overlapping_words_keep_the_latest_end_when_splitting_speech():
    line = nr.Line("overlap", "one two three", pad_after_s=0.25)
    words = (
        nr.Word("one", 0.5, 3.5),
        nr.Word("two", 1.0, 1.5),
        nr.Word("three", 3.5, 3.75),
    )
    narration = _narration((line, nr.Spoken(line.key, 3.75, words)))
    faults = ept.timing_faults(narration, source=_gray_frame, still_s=3.0)
    assert [(f.kind, f.key, f.start_s, f.end_s) for f in faults] == [
        ("still-speech", "overlap", 0.0, 4.0)
    ]
    assert "3.2 s of speech (overlap)" in faults[0].message


@pytest.mark.parametrize("rgba", [False, True])
@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("changes", [False, True])
def test_float_and_rgba_frames_match_uint8_rgb(rgba, floating, changes):
    def source(t):
        frame = _gray_frame(t)
        if changes and t >= 2.0:
            frame[:9, :16] = 255
        if rgba:
            alpha = np.full((90, 160, 1), 255 if t < 2.0 else 0, dtype=np.uint8)
            frame = np.concatenate((frame, alpha), axis=2)
        return frame.astype(float) / 255 if floating else frame

    faults = ept.timing_faults(_still_narration(), source=source)
    assert [(f.kind, f.key) for f in faults] == (
        [] if changes else [("still-speech", "steady")]
    )


@pytest.mark.parametrize(
    "before, after, reported", [(128.4, 140.6, False), (128.6, 141.4, True)]
)
def test_float_frames_round_before_comparing_channels(before, after, reported):
    def source(t):
        frame = _gray_frame(t).astype(float) / 255
        # Rounding gives deltas of 13 and 12; truncation gives 12 and 13.
        frame[:9, :16, 0] = (before if t < 2.0 else after) / 255
        return frame

    faults = ept.timing_faults(_still_narration(), source=source)
    assert [(f.kind, f.key) for f in faults] == (
        [("still-speech", "steady")] if reported else []
    )


def test_a_source_must_be_a_path_or_callable():
    with pytest.raises(TypeError, match="source"):
        ept.timing_faults(_still_narration(), source=[])


def test_callable_sampling_uses_plan_length_and_requested_fps():
    times = []

    def source(t):
        times.append(t)
        return _gray_frame(t)

    plan = SimpleNamespace(seconds=1.6, hero_windows=lambda: [])
    assert ept.timing_faults(_still_narration(), source=source, plan=plan, fps=5) == []
    assert times == [k / 5 for k in range(8)]


@pytest.mark.parametrize("delta, reported", [(12, True), (13, False), (-12, True)])
def test_a_pixel_changes_only_above_twelve_in_any_channel(delta, reported):
    def source(t):
        frame = _gray_frame(t)
        if t >= 2.0:
            frame[:9, :16, 0] = 128 + delta
        return frame

    faults = ept.timing_faults(_still_narration(), source=source)
    assert [(f.kind, f.key) for f in faults] == (
        [("still-speech", "steady")] if reported else []
    )


def test_a_still_run_compares_with_its_first_frame_as_drift_accumulates():
    def source(t):
        frame = _gray_frame(t)
        frame[:9, :16] = 128 + 5 * int(t)
        return frame

    faults = ept.timing_faults(_still_narration(), source=source)
    assert [(f.kind, f.start_s, f.end_s) for f in faults] == [
        ("still-speech", 0.0, 3.0)
    ]
    assert "2.5 s of speech" in faults[0].message


@pytest.mark.parametrize("changes", [False, True])
def test_video_source_detects_stillness_and_a_color_change(tmp_path, changes):
    if (
        shutil.which("ffmpeg") is None
        and importlib.util.find_spec("imageio_ffmpeg") is None
    ):
        pytest.skip("ffmpeg is unavailable")
    from eyepiece._ffmpeg import resolve_ffmpeg

    path = tmp_path / "color.mp4"
    args = [resolve_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i"]
    if changes:
        args += [
            "color=c=gray:s=160x90:d=2:r=10",
            "-f", "lavfi", "-i", "color=c=white:s=160x90:d=2:r=10",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
        ]  # fmt: skip
    else:
        args += ["color=c=gray:s=160x90:d=4:r=10"]
    subprocess.run(
        [*args, "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True
    )
    faults = ept.timing_faults(
        _still_narration(), source=path if changes else str(path)
    )
    assert [(f.kind, f.key) for f in faults] == (
        [] if changes else [("still-speech", "steady")]
    )


def test_wide_video_frames_are_decoded_at_1280_pixels_with_even_height(tmp_path):
    if (
        shutil.which("ffmpeg") is None
        and importlib.util.find_spec("imageio_ffmpeg") is None
    ):
        pytest.skip("ffmpeg is unavailable")
    from eyepiece._ffmpeg import resolve_ffmpeg

    path = tmp_path / "wide.mp4"
    args = [
        resolve_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=c=gray:s=1920x1082:d=0.2:r=10", "-pix_fmt", "yuv420p", str(path),
    ]  # fmt: skip
    subprocess.run(args, check=True, capture_output=True)
    frames = list(ept._source_frames(path, seconds=0.2, fps=10))
    assert [t for t, _ in frames] == [0.0, 0.1]
    # The proportional height is 721.33 pixels, rounded to the nearest even height.
    assert [frame.shape for _, frame in frames] == [(722, 1280, 3)] * 2
    assert all(frame.dtype == np.uint8 for _, frame in frames)


def test_keywords_for_later_rules_are_accepted():
    assert (
        ept.timing_faults(
            _narration(_line("steady")),
            plan=None,
            source=None,
            at=lambda t: None,
            readouts=None,
            ceiling=160.0,
            short_ceiling=166.0,
            short_words=10,
            min_gap_s=0.4,
            still_s=2.0,
            still_change=0.0016,
            ignore_bottom=0.0,
            fps=10.0,
            settle_tol_s=0.0,
        )
        == []
    )
