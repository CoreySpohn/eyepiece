"""Contact sheets sampled at event times."""

import importlib.util
import shutil
import subprocess
import wave

import numpy as np
import pytest
from matplotlib.image import imread

import eyepiece as ep
import eyepiece.narration as nr
import eyepiece.testing as ept


def _narration(keys=("first", "second"), *, starts=None):
    lines = [
        nr.Line(
            key,
            f"Show {{show:{key}}}this.",
            pad_before_s=0.0,
            pad_after_s=0.5,
        )
        for key in keys
    ]
    spoken = {
        key: nr.Spoken(
            key, 1.5, (nr.Word("Show", 0.0, 0.5), nr.Word("this.", 0.75, 1.5))
        )
        for key in keys
    }
    return nr.Narration(nr.Script(tuple(lines)), spoken, starts=starts)


def _frame(t):
    return np.full((18, 32, 3), 128, dtype=np.uint8)


def _tile_pixels(sheet, column, width):
    # The first band follows the margin, row header and one-line tile label.
    left = 16 + column * (width + 16)
    top = 16 + 24 + 16
    return sheet[top + 1 : top + width * 9 // 16 - 1, left + 1 : left + width - 1, :3]


def test_callable_samples_every_cue_and_beat_end(tmp_path):
    times = []

    def source(t):
        times.append(t)
        return _frame(t)

    path = tmp_path / "nested" / "sheet.png"
    result = ept.review_sheet(path, source, narration=_narration(), width=160)
    assert result == path
    assert "review_sheet" in ept.__all__
    assert len(times) == 4
    assert sorted(set(times)) == pytest.approx(
        [0.75, 1.999, 2.75, 3.999], rel=0, abs=1e-9
    )
    assert imread(path).size > 0


def test_plan_samples_move_mid_and_end_frames(tmp_path):
    timeline = ep.Timeline({"x": 0.0})
    timeline.move("go", 0.75, to={"x": 1.0})
    timeline.hold(1.0, name="wait")
    plan = timeline.at_fps(10)
    times = []

    def source(t):
        times.append(t)
        return _frame(t)

    ept.review_sheet(tmp_path / "plan.png", source, plan=plan, width=160)
    expected = [
        event.frame / plan.fps
        for what in ("mid", "end")
        for event in plan.events(what=what, kind="move")
    ]
    assert times == pytest.approx(expected, rel=0, abs=1e-9)


def test_events_in_the_same_millisecond_share_one_tile(tmp_path):
    times = []

    def source(t):
        times.append(t)
        return _frame(t)

    ept.review_sheet(
        tmp_path / "merged.png",
        source,
        events=iter([(1.0001, "a"), (1.0002, "b")]),
        width=160,
    )
    assert len(times) == 1
    assert times[0] in (1.0001, 1.0002)


def test_video_seeks_to_each_event_time(tmp_path):
    if (
        shutil.which("ffmpeg") is None
        and importlib.util.find_spec("imageio_ffmpeg") is None
    ):
        pytest.skip("ffmpeg is unavailable")
    from eyepiece._ffmpeg import resolve_ffmpeg

    video = tmp_path / "colors.mp4"
    subprocess.run(
        [
            resolve_ffmpeg(), "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=160x90:r=10:d=2",
            "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=10:d=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    sheet = imread(
        ept.review_sheet(
            tmp_path / "video.png",
            video,
            events=[(1.0, "a"), (3.0, "b")],
            columns=2,
        )
    )
    left = _tile_pixels(sheet, 0, 320)
    right = _tile_pixels(sheet, 1, 320)
    assert np.all((left[:, :, 0] > 0.8) & (left[:, :, 1:3].max(axis=2) < 0.2))
    assert np.all((right[:, :, 2] > 0.8) & (right[:, :, :2].max(axis=2) < 0.2))


@pytest.mark.parametrize("event_s", [1.999, 2.5])
def test_video_end_uses_last_decodable_frame(tmp_path, event_s):
    if (
        shutil.which("ffmpeg") is None
        and importlib.util.find_spec("imageio_ffmpeg") is None
    ):
        pytest.skip("ffmpeg is unavailable")
    from eyepiece._ffmpeg import resolve_ffmpeg

    video = tmp_path / "last.mp4"
    subprocess.run(
        [
            resolve_ffmpeg(), "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=160x90:r=10:d=2",
            "-vf", "drawbox=color=blue:t=fill:enable='eq(n,19)'",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    sheet = imread(
        ept.review_sheet(
            tmp_path / "last.png",
            video,
            events=[(event_s, "end")],
            columns=1,
            width=160,
        )
    )
    tile = _tile_pixels(sheet, 0, 160)
    assert np.all((tile[:, :, 2] > 0.8) & (tile[:, :, :2].max(axis=2) < 0.2))


def test_callable_thumbnail_averages_fine_stripes(tmp_path):
    width = 160
    stripes = np.zeros((360, 4 * width, 3), dtype=np.uint8)
    stripes[:, 1::2] = 255
    sheet = imread(
        ept.review_sheet(
            tmp_path / "stripes.png",
            lambda t: stripes,
            events=[(0.0, "stripes")],
            columns=1,
            width=width,
        )
    )
    tile = _tile_pixels(sheet, 0, width)
    np.testing.assert_allclose(tile, 0.5, atol=0.01)


def test_short_audio_window_shows_signal_instead_of_silence(tmp_path):
    rate = 8000
    samples = np.sin(np.arange(80) * 0.2)
    options = {"events": [(0.0, "start"), (0.01, "finish")], "columns": 2, "width": 320}
    signal = imread(
        ept.review_sheet(
            tmp_path / "signal.png", _frame, audio=(samples, rate), **options
        )
    )
    silence = imread(
        ept.review_sheet(
            tmp_path / "silence.png",
            _frame,
            audio=(np.zeros_like(samples), rate),
            **options,
        )
    )
    assert np.any(signal != silence)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("with_narration", [False, True])
def test_array_and_pcm_wav_add_a_waveform_strip(tmp_path, channels, with_narration):
    rate = 8000
    times = np.arange(4 * rate) / rate
    samples = (16000 * np.sin(2 * np.pi * 5 * times)).astype(np.int16)
    if channels == 2:
        second = (16000 * np.sin(2 * np.pi * 13 * times)).astype(np.int16)
        samples = np.column_stack((samples, second))
    wav = tmp_path / "audio.wav"
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.astype("<i2").tobytes())
    options = {"events": [(1.0, "a")], "columns": 2, "width": 160}
    if with_narration:
        options["narration"] = _narration()
    plain = imread(ept.review_sheet(tmp_path / "plain.png", _frame, **options))
    array = imread(
        ept.review_sheet(
            tmp_path / "array.png", _frame, audio=(samples, rate), **options
        )
    )
    pcm = imread(ept.review_sheet(tmp_path / "pcm.png", _frame, audio=wav, **options))
    floating = imread(
        ept.review_sheet(
            tmp_path / "float.png",
            _frame,
            audio=(samples.astype(float) / 32768, rate),
            **options,
        )
    )
    assert array.shape[0] > plain.shape[0]
    assert array.shape == pcm.shape == floating.shape
    np.testing.assert_allclose(array, pcm, atol=1 / 255)
    np.testing.assert_allclose(array, floating, atol=1 / 255)
    silence = imread(
        ept.review_sheet(
            tmp_path / "silence.png",
            _frame,
            audio=(np.zeros_like(samples), rate),
            **options,
        )
    )
    assert np.any(array != silence)
    # Only the event ticks have red ink; frames and text are achromatic.
    red_ticks = (
        (silence[:, :, 0] > 0.5)
        & (silence[:, :, 0] > 1.5 * silence[:, :, 1])
        & (silence[:, :, 2] < 0.3)
    )
    assert red_ticks.sum() > 40
    if channels == 2:
        mean = imread(
            ept.review_sheet(
                tmp_path / "mean.png",
                _frame,
                audio=(samples.mean(axis=1), rate),
                **options,
            )
        )
        first = imread(
            ept.review_sheet(
                tmp_path / "first.png",
                _frame,
                audio=(samples[:, 0], rate),
                **options,
            )
        )
        np.testing.assert_array_equal(array, mean)
        assert np.any(array != first)


@pytest.mark.parametrize("events", [None, []])
def test_missing_events_raise_value_error(tmp_path, events):
    with pytest.raises(ValueError):
        ept.review_sheet(tmp_path / "empty.png", _frame, events=events)


def test_a_beat_without_events_still_adds_a_row(tmp_path):
    two = _narration(("first", "third"), starts={"third": 4.0})
    three = _narration(("first", "second", "third"))
    options = {"events": [(1.0, "a"), (5.0, "b")], "columns": 2, "width": 160}
    shorter = imread(
        ept.review_sheet(tmp_path / "two.png", _frame, narration=two, **options)
    )
    taller = imread(
        ept.review_sheet(tmp_path / "three.png", _frame, narration=three, **options)
    )
    assert taller.shape[0] >= shorter.shape[0] + 14


def test_tiles_are_in_time_order_and_wrap_at_columns(tmp_path):
    times = []

    def source(t):
        times.append(t)
        return _frame(t)

    events = [(3.0, "c"), (1.0, "a"), (2.0, "b")]
    wide = imread(
        ept.review_sheet(
            tmp_path / "wide.png", source, events=events, columns=3, width=160
        )
    )
    assert times == [1.0, 2.0, 3.0]
    narrow = imread(
        ept.review_sheet(
            tmp_path / "narrow.png", _frame, events=events, columns=2, width=160
        )
    )
    assert narrow.shape[0] > wide.shape[0]
