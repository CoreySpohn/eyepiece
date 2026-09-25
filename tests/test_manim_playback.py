"""Prepared sequences as replayable Manim playback, checked on encoded frames."""

import numpy as np
import pytest

from eyepiece.prepared import ArrayChannel, AxisSpec, ImageView, Scale, Sequence
from eyepiece.style import snapshot_profile

manim = pytest.importorskip("manim")


@pytest.fixture(autouse=True)
def media_in_tmp(tmp_path):
    """Keep Manim's text-layout cache out of the working directory."""
    with manim.tempconfig({"media_dir": str(tmp_path / "media")}):
        yield


def _ticks(animation, fps):
    """The frame times Manim's Cairo scene loop feeds this animation."""
    return np.arange(0, animation.run_time, 1 / fps)


def _run_like_a_scene(animation, fps):
    """Drive `animation` through begin, every encoded tick, and finish."""
    animation.begin()
    for t in _ticks(animation, fps):
        animation.interpolate(t / animation.run_time)
    animation.finish()


def _held_means(clip, monkeypatch):
    """Record the image mean every `ManimResult.update` is handed, in order."""
    from eyepiece.manim import _render

    means = []
    original = _render.ManimResult.update

    def spy(self, view):
        means.append(float(np.mean(view.data)))
        return original(self, view)

    monkeypatch.setattr(_render.ManimResult, "update", spy)
    return means


# --- Clip structure ---------------------------------------------------------------


def test_animate_exposes_the_rendered_first_frame(image_sequence):
    import eyepiece.manim as em

    sequence = image_sequence()
    clip = em.animate(sequence)
    assert isinstance(clip, em.ManimClip)
    assert clip.sequence is sequence
    assert clip.parts["image"] is clip.result.parts["image"]
    assert clip.mobject is clip.result.mobject
    assert float(np.mean(clip.result.view.data)) == 0.0


def test_sequence_storage_stays_off_the_mobjects(image_sequence):
    import eyepiece.manim as em

    sequence = image_sequence()
    cube = sequence.channels[0].values
    clip = em.animate(sequence)
    copy = clip.mobject.copy()
    for root in (clip.mobject, copy):
        for mob in root.get_family():
            for value in vars(mob).values():
                assert value is not sequence and value is not clip
                if isinstance(value, np.ndarray):
                    assert not np.shares_memory(value, cube)


def test_playback_quantizes_run_time_to_whole_output_frames(image_sequence):
    import eyepiece.manim as em

    clip = em.animate(image_sequence())
    with manim.tempconfig({"frame_rate": 10}):
        animation = clip.playback(0.25)  # ceil(2.5) = 3 output frames
        assert isinstance(animation, manim.Animation)
        assert animation.run_time == pytest.approx(0.3)
        assert animation.rate_func is manim.linear
        assert len(_ticks(animation, 10)) == 3


@pytest.mark.parametrize(("fps", "count"), [(10, 5), (12, 5), (12, 10), (60, 480)])
def test_every_encoder_tick_lands_on_its_schedule_index(fps, count, monkeypatch):
    """Tick k shows schedule index k, including where float time rounds badly."""
    import eyepiece.manim as em

    times = np.arange(float(count))
    cube = np.arange(float(count))[:, None, None] * np.ones((1, 2, 2))
    view = ImageView(
        "image",
        cube[0],
        AxisSpec("x", "y", (0, 2), (0, 2)),
        Scale("linear", 0, max(count - 1, 1)),
        "signal",
    )
    sequence = Sequence(view, times, "s", (ArrayChannel("image", "data", cube),))
    clip = em.animate(sequence)
    means = _held_means(clip, monkeypatch)
    with manim.tempconfig({"frame_rate": fps}):
        animation = clip.playback(count / fps)
        ticks = _ticks(animation, fps)
        assert len(ticks) == count
        animation.begin()
        means.clear()
        for t in ticks:
            animation.interpolate(t / animation.run_time)
    schedule = sequence.schedule(run_time=count / fps, fps=fps)
    expected = [float(sequence.at(t).index) for t in schedule]
    assert means == expected
    assert means[-1] == count - 1


def test_begin_resets_and_consumer_visibility_survives_seek_and_replay(
    track_sequence,
):
    import eyepiece.manim as em

    clip = em.animate(track_sequence)
    region = clip.parts["iwa"]
    region.set_opacity(0)
    clip.seek(7.0)
    assert clip.result.parts["clock"].submobjects[0].original_text == "7 d"
    with manim.tempconfig({"frame_rate": 10}):
        animation = clip.playback(1.0)
        _run_like_a_scene(animation, 10)
        assert clip.result.parts["clock"].submobjects[0].original_text == "7 d"
        animation.begin()
        assert clip.result.parts["clock"].submobjects[0].original_text == "0 d"
        # Replay is deterministic: the trail restarts from its first vertex.
        assert not clip.parts["orbit"].has_points()
    copy = clip.mobject.copy()
    copied_region = copy.get_family()[clip.mobject.get_family().index(region)]
    for mob in (region, copied_region):
        assert mob.get_fill_opacity() == 0
        assert mob.get_stroke_opacity() == 0
    assert clip.parts["iwa"] is region


def test_an_eight_second_pass_maps_each_held_image_once(monkeypatch):
    """480 ticks over 80 held images: at most 80 mappings, no science calls."""
    import eyepiece.manim as em
    from eyepiece.manim import _render

    science_calls = []

    def simulate(index):
        science_calls.append(index)
        rng = np.random.default_rng(0 + index)
        return rng.random((16, 16))

    cube = np.stack([simulate(i) for i in range(80)])
    science_calls.clear()
    view = ImageView(
        "image",
        cube[0],
        AxisSpec("x", "y", (0, 16), (0, 16)),
        Scale("linear", 0, 1),
        "signal",
    )
    sequence = Sequence(
        view, np.arange(80.0), "s", (ArrayChannel("image", "data", cube),)
    )
    mappings = []
    original = _render.map_rgba

    def spy(*args, **kwargs):
        mappings.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_render, "map_rgba", spy)
    clip = em.animate(sequence)
    with manim.tempconfig({"frame_rate": 60}):
        animation = clip.playback(8)
        assert len(_ticks(animation, 60)) == 480
        _run_like_a_scene(animation, 60)
    assert len(mappings) <= 80
    assert science_calls == []
    image = clip.parts["image"]
    np.testing.assert_array_equal(
        image.pixel_array,
        original(cube[79], valid=None, scale=view.scale, profile=clip.result.profile)[
            ::-1
        ],
    )


# --- Encoded frames ---------------------------------------------------------------


def _encode(clip, run_time, fps, tmp_path):
    """Render `clip.playback(run_time)` at a tiny size; return decoded RGB frames."""
    import av

    class Clip(manim.Scene):
        def construct(self):
            self.add(clip.mobject)
            self.play(clip.playback(run_time))

    settings = {
        "media_dir": str(tmp_path),
        "disable_caching": True,
        "pixel_width": 128,
        "pixel_height": 72,
        "frame_rate": fps,
        "progress_bar": "none",
        "verbosity": "ERROR",
        "output_file": "clip",
    }
    with manim.tempconfig(settings):
        scene = Clip()
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
        frame_width = manim.config.frame_width
        frame_height = manim.config.frame_height
    with av.open(str(movie)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return frames, (frame_width, frame_height)


def _pixel_at(frame, point, frame_size):
    """The decoded RGB at a scene point."""
    frame_width, frame_height = frame_size
    rows, cols = frame.shape[:2]
    col = int((point[0] + frame_width / 2) / frame_width * cols)
    row = int((frame_height / 2 - point[1]) / frame_height * rows)
    return frame[row, col].astype(int)


@pytest.mark.parametrize(("fps", "run_time"), [(10, 0.5), (12, 5 / 12)])
def test_encoded_clip_holds_the_schedule_and_ends_on_the_final_sample(
    image_sequence, tmp_path, fps, run_time
):
    import eyepiece.manim as em

    profile = snapshot_profile()
    sequence = image_sequence()
    clip = em.animate(sequence, profile=profile)
    clip.mobject.scale_to_fit_height(6)
    center = clip.parts["image"].get_center()
    frames, frame_size = _encode(clip, run_time, fps, tmp_path)
    schedule = sequence.schedule(run_time=run_time, fps=fps)
    assert len(frames) == len(schedule) == 5
    lut = profile.colormaps["intensity"]
    # Values 0, 10, 20 on a 0..20 scale pick LUT entries 0, 128, 255.
    entry = {0: 0, 1: 128, 2: 255}
    for frame, t in zip(frames, schedule, strict=True):
        expected = lut[entry[sequence.at(t).index]][:3].astype(int)
        np.testing.assert_allclose(
            _pixel_at(frame, center, frame_size), expected, atol=12
        )
    np.testing.assert_allclose(
        _pixel_at(frames[-1], center, frame_size), lut[255][:3], atol=12
    )
