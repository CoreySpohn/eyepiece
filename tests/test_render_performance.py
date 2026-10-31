"""Pixel identity and draw counts for shared and deduplicated renders."""

import io
import subprocess
from dataclasses import replace
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_render_helpers import CountingRenderer, build_variant_clip, write_render_audio

from eyepiece import _render_pool as pool
from eyepiece import _render_variants as variant_driver
from eyepiece._ffmpeg import resolve_ffmpeg
from eyepiece.render import (
    ClipSpec,
    Manifest,
    Variant,
    Venue,
    render_frames,
    render_variants,
)


def state_key(item):
    return item // 3


def optional_key(item):
    return None if item in (1, 2) else state_key(item)


def draw_steps(fig, item, *, calls):
    with Path(calls).open("a") as stream:
        stream.write(f"{item}\n")
    fig.set_facecolor(str(state_key(item) / 4))


def build_steps(*, calls):
    fig = plt.figure(figsize=(0.96, 0.54))
    fig.patch.draw = partial(paint_background, artist=fig.patch, calls=calls)
    return fig, partial(draw_steps, calls=calls)


def paint_background(renderer, *, artist, calls):
    from matplotlib.patches import Rectangle

    counted = CountingRenderer(renderer)
    Rectangle.draw(artist, counted)
    with Path(str(calls) + ".paint").open("a") as stream:
        stream.write(f"{counted.paths}\n")


def decoded(path):
    return subprocess.run(
        [
            resolve_ffmpeg(), "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout  # fmt: skip


def draw_count(path):
    return len(path.read_text().splitlines())


def test_raw_frames_identical_and_only_consecutive_keys_skip_draw(tmp_path):
    frames = [0, 1, 2, 3, 4, 5, 0, 1, 9]
    baseline = []
    for name, key in (("full", None), ("dedup", state_key)):
        calls = tmp_path / name
        build = partial(build_steps, calls=calls)
        with pool._figure(build, frames, 100, None, None, None, frame_key=key) as state:
            rasters = [pool._raster(state, k).tobytes() for k in range(len(frames))]
        if key is None:
            baseline = rasters
            assert draw_count(calls) == len(frames)
        else:
            assert rasters == baseline
            assert draw_count(calls) == len(frames)
            # Four rasters plus the initial layout freeze.
            assert draw_count(Path(str(calls) + ".paint")) == 5


@pytest.mark.parametrize("workers", [1, 2])
def test_dedup_raw_chunks_encoded_bytes_decoded_frames_holds_and_resume(
    tmp_path, workers
):
    videos = []
    chunks = []
    counts = []
    for name, key in (("full", None), ("dedup", state_key)):
        folder = tmp_path / name
        folder.mkdir()
        calls = folder / "draws"
        kwargs = dict(
            fps=12,
            dpi=100,
            workers=workers,
            chunk_frames=4,
            hold=(2, 3),
            frame_key=key,
            key="steps",
            keep_chunks=True,
            work_dir=folder / "chunks",
        )
        build = partial(build_steps, calls=calls)
        path = folder / "steps.mp4"
        result = render_frames(build, 12, path, **kwargs)
        assert result.frames == 17
        videos.append(path)
        chunks.append(
            [p.read_bytes() for p in sorted((folder / "chunks").glob("*.bin"))]
        )
        counts.append(draw_count(Path(str(calls) + ".paint")))
        resumed = render_frames(build, 12, path, **kwargs)
        assert resumed.resumed == resumed.chunks == 3
    assert counts[1] < counts[0]
    assert chunks[0] == chunks[1]
    assert videos[0].read_bytes() == videos[1].read_bytes()
    assert decoded(videos[0]) == decoded(videos[1])


def test_none_key_disables_reuse_for_that_frame(tmp_path):
    calls = tmp_path / "draws"
    with pool._figure(
        partial(build_steps, calls=calls),
        6,
        100,
        None,
        None,
        None,
        frame_key=optional_key,
    ) as state:
        for index in range(6):
            pool._raster(state, index)
    assert calls.read_text().splitlines() == ["0", "1", "2", "3", "4", "5"]
    assert draw_count(Path(str(calls) + ".paint")) == 5  # includes layout freeze


@pytest.mark.parametrize("key", [42, lambda item: []])
def test_invalid_key_preserves_video_and_cleans_up(tmp_path, key):
    path = tmp_path / "steps.mp4"
    path.write_bytes(b"previous")
    with pytest.raises(TypeError, match=r"frame_key|hashable"):
        render_frames(
            partial(build_steps, calls=tmp_path / "draws"),
            6,
            path,
            fps=12,
            dpi=100,
            workers=1,
            frame_key=key,
        )
    assert path.read_bytes() == b"previous"
    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.glob(".*.partial.mp4"))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_variants_share_one_video_keep_order_posters_and_audio_only_reuse(
    tmp_path, monkeypatch, reverse, workers
):
    monkeypatch.setattr(
        variant_driver, "render_frames", partial(render_frames, chunk_frames=3)
    )
    audio = write_render_audio(tmp_path / "track.wav")
    markers = tmp_path / "builds"
    spec = ClipSpec(
        "shared",
        partial(build_variant_clip, markers=markers),
        6,
        (Venue("doc", fps=12),),
        audio=lambda variant: audio,
        poster=5,
        target_for=lambda variant: Path(variant.name) / "movie.mp4",
    )
    silent, narrated = Variant("doc", "light"), Variant("doc", "light", True)
    selection = [narrated, silent] if reverse else [silent, narrated]
    manifest = Manifest(tmp_path / "manifest.json")
    original = pool._raster
    # Count raster work in the parent; spawned builds are recorded on disk too.
    raster_calls = []

    def raster(state, index):
        raster_calls.append(index)
        return original(state, index)

    monkeypatch.setattr(pool, "_raster", raster)
    results = render_variants(
        spec, selection, tmp_path, manifest=manifest, workers=workers
    )
    assert [r.variant for r in results] == selection
    assert all(r.rendered for r in results)
    assert draw_count(markers / "doc-light") == workers + 1  # workers plus poster
    if workers == 1:
        assert len(raster_calls) == 6  # poster uses its own imported raster helper
    paths = {r.variant: tmp_path / r.outputs[0] for r in results}
    assert decoded(paths[silent]) == decoded(paths[narrated])
    assert len(decoded(paths[silent])) == 6 * 96 * 54 * 4
    assert all(manifest.current(spec.slug, r.variant.name, r.key) for r in results)
    assert len(next(r for r in results if r.variant == silent).outputs) == 2
    before = (paths[silent].read_bytes(), paths[silent].stat().st_mtime_ns)
    count = draw_count(markers / "doc-light")
    # Change audio bytes without touching the frame source.
    write_render_audio(audio)
    with audio.open("r+b") as stream:
        stream.seek(100)
        stream.write(b"\0\0")
    changed = render_variants(
        spec, [silent, narrated], tmp_path, manifest=manifest, workers=workers
    )
    assert [r.rendered for r in changed] == [False, True]
    assert draw_count(markers / "doc-light") == count
    assert (paths[silent].read_bytes(), paths[silent].stat().st_mtime_ns) == before
    assert decoded(paths[silent]) == decoded(paths[narrated])
    assert not list(tmp_path.rglob("*.narration-*"))


@pytest.mark.parametrize("different", ["frames", "mapping", "data"])
def test_sharing_requires_the_same_visual_inputs(tmp_path, different):
    audio = write_render_audio(tmp_path / "track.wav")
    markers = tmp_path / "builds"
    spec = ClipSpec(
        "distinct",
        partial(build_variant_clip, markers=markers),
        6,
        (Venue("doc", fps=12),),
        audio=lambda variant: audio,
    )
    if different == "frames":
        spec = replace(spec, frames_for=lambda variant: 8 if variant.narrated else 6)
    elif different == "mapping":
        spec = replace(
            spec, frame_at_for=lambda variant: state_key if variant.narrated else None
        )
    else:
        spec = replace(spec, data=lambda variant: (variant.narrated,))
    result = render_variants(
        spec,
        [Variant("doc", "light"), Variant("doc", "light", True)],
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )
    assert all(r.rendered for r in result)
    assert draw_count(markers / "doc-light") == 2


@pytest.mark.parametrize(
    "cpus,memory,expected", [(32, 32, 8), (4, 32, 3), (32, 3, 2), (1, 1, 1)]
)
def test_default_worker_budget(monkeypatch, cpus, memory, expected):
    monkeypatch.setattr(pool, "_available_cpus", lambda: cpus)
    monkeypatch.setattr(pool, "_available_memory", lambda: memory * 1024**3)
    assert pool._default_workers() == expected


def test_cpu_and_memory_probe_fallbacks(monkeypatch):
    monkeypatch.delattr(pool.os, "process_cpu_count", raising=False)
    monkeypatch.delattr(pool.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(pool.os, "cpu_count", lambda: None)
    monkeypatch.setattr(pool.os, "sysconf", lambda name: -1)
    assert pool._available_cpus() == 1
    assert pool._available_memory() is None
    assert pool._default_workers() == 1


def test_encoder_receives_rgba_without_copying(tmp_path, monkeypatch):
    received = []

    class Pipe(io.BytesIO):
        def write(self, data):
            received.append(data)
            return super().write(data)

    class Sink:
        frames = 0
        _proc = type("Process", (), {"stdin": Pipe()})()

    pixels = np.zeros((4, 6, 4), dtype=np.uint8)
    sink = Sink()
    pool._write_frame(sink, pixels)
    assert isinstance(received[0], memoryview)
    assert received[0].obj is pixels
    assert sink.frames == 1
    assert sink._proc.stdin.getvalue() == pixels.tobytes()


def test_nonsequential_raster_requests_do_not_reuse_stale_state(tmp_path):
    order = [0, 3, 4, 1, 2, 8, 6, 7, 0]
    outputs = []
    for name, key in (("full", None), ("dedup", state_key)):
        with pool._figure(
            partial(build_steps, calls=tmp_path / name),
            9,
            100,
            None,
            None,
            None,
            frame_key=key,
        ) as state:
            outputs.append([pool._raster(state, index).tobytes() for index in order])
    assert outputs[0] == outputs[1]
    assert draw_count(tmp_path / "dedup") == len(order)
    assert draw_count(tmp_path / "dedup.paint") == 7  # includes layout freeze


def test_key_receives_mapped_items_and_spec_defaults_to_full_draws(tmp_path):
    def build_variant(venue, mode):
        return build_steps(calls=tmp_path / "draws")

    spec = ClipSpec(
        "mapped", build_variant, 12, (Venue("doc", fps=12),), frame_at=state_key
    )
    assert spec.frame_key is None
    selection = [Variant("doc", "light")]
    manifest = Manifest(tmp_path / "manifest.json")
    result = render_variants(spec, selection, tmp_path, manifest=manifest, workers=1)[0]
    full = (tmp_path / result.outputs[0]).read_bytes()
    assert draw_count(tmp_path / "draws") == 12
    (tmp_path / "draws").unlink()
    dedup = render_variants(
        replace(spec, frame_key=state_key),
        selection,
        tmp_path,
        manifest=manifest,
        workers=1,
    )[0]
    assert draw_count(tmp_path / "draws") == 12
    assert dedup.key != result.key
    assert (tmp_path / dedup.outputs[0]).read_bytes() == full


@pytest.mark.parametrize("custom", ["rgb", "matrix", "chroma"])
def test_mux_preserves_base_custom_encoded_picture(tmp_path, custom):
    audio = write_render_audio(tmp_path / "track.wav")
    options = {
        "rgb": {
            "extra_ffmpeg_args": [
                "-c:v",
                "libx264rgb",
                "-pix_fmt",
                "rgb24",
                "-crf",
                "0",
            ]
        },
        "matrix": {"encoder": {"colorspace": None}},
        "chroma": {"encoder": {"pix_fmt": "yuv444p"}},
    }
    spec = ClipSpec(
        "colors",
        partial(build_variant_clip, markers=tmp_path / "builds"),
        6,
        (Venue("doc", fps=12),),
        audio=lambda variant: audio,
        **options[custom],
    )
    results = render_variants(
        spec,
        [Variant("doc", "dark"), Variant("doc", "dark", True)],
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )
    paths = [tmp_path / result.outputs[0] for result in results]
    from eyepiece.narration import mux

    reference = tmp_path / "base-mux.mp4"
    mux(paths[0], audio, reference)
    assert paths[1].read_bytes() == reference.read_bytes()
    assert decoded(reference) == decoded(paths[1])
    assert draw_count(tmp_path / "builds" / "doc-dark") == 1


def test_narrated_only_request_can_reuse_a_current_silent_entry(tmp_path):
    spec = ClipSpec(
        "prior",
        partial(build_variant_clip, markers=tmp_path / "builds"),
        6,
        (Venue("doc", fps=12),),
        target_for=lambda variant: Path(variant.name) / "custom.mp4",
    )
    manifest = Manifest(tmp_path / "manifest.json")
    silent = render_variants(
        spec,
        [Variant("doc", "light")],
        tmp_path,
        manifest=manifest,
        workers=1,
    )[0]
    audio = write_render_audio(tmp_path / "audio.wav")
    result = render_variants(
        replace(spec, audio=lambda variant: audio),
        [Variant("doc", "light", True)],
        tmp_path,
        manifest=manifest,
        workers=1,
    )[0]
    assert result.rendered
    assert draw_count(tmp_path / "builds" / "doc-light") == 1
    assert decoded(tmp_path / silent.outputs[0]) == decoded(
        tmp_path / result.outputs[0]
    )


def test_small_frames_are_buffered_together(tmp_path, monkeypatch):
    writes = []

    class Pipe(io.BytesIO):
        def write(self, data):
            writes.append(bytes(data))
            return super().write(data)

    class Sink:
        frames = 0

        def __init__(self, *args, **kwargs):
            self._proc = type("Process", (), {"stdin": Pipe()})()

        def close(self):
            self._proc.stdin.close()

    monkeypatch.setattr(pool, "RawSink", Sink)
    with pool._encoding_sink(tmp_path / "unused.mp4", (2, 2), 12, None, None) as sink:
        for _ in range(100):
            pool._write_frame(sink, np.zeros((2, 2, 4), dtype=np.uint8))
        assert not writes
    assert sink.frames == 100
    assert writes == [bytes(1600)]
    assert sink._proc.stdin.closed


def test_dedup_keeps_scale_diagnostics_across_chunks_and_resume(tmp_path):
    from test_render_helpers import build_rescale_clip

    from eyepiece._render_chunks import _read_scale_diagnostic
    from eyepiece.anim import ScaleDriftError

    folder = tmp_path / "chunks"
    path = tmp_path / "scales.mp4"
    kwargs = dict(
        fps=12,
        dpi=100,
        workers=1,
        chunk_frames=3,
        frame_key=state_key,
        work_dir=folder,
        key="scales",
        keep_chunks=True,
    )
    build = partial(build_rescale_clip, drift_at=3)
    with pytest.warns(RuntimeWarning, match=r"data scales changed.*frame 3"):
        render_frames(build, 6, path, **kwargs)
    before = path.read_bytes()
    job = (
        folder / "chunk-1.bin",
        pool._identity("scales", 6, (0, 0), 100, (96, 54), 12, 3, 6),
        3,
        6,
    )
    assert "frame 3" in _read_scale_diagnostic(job)
    with pytest.raises(ScaleDriftError, match="frame 3"):
        render_frames(build, 6, path, on_rescale="raise", **kwargs)
    assert path.read_bytes() == before


@pytest.mark.parametrize("reverse", [False, True])
def test_adopted_silent_video_supplies_narration_and_current_poster(tmp_path, reverse):
    from test_render_helpers import build_adopt_variant

    old = partial(build_adopt_variant, markers=tmp_path / "old", color="red")
    new = partial(build_adopt_variant, markers=tmp_path / "new", color="blue")
    spec = ClipSpec("adopted", old, 6, (Venue("doc", fps=12),))
    manifest = Manifest(tmp_path / "manifest.json")
    silent, narrated = Variant("doc", "light"), Variant("doc", "light", True)
    prior = render_variants(spec, [silent], tmp_path, manifest=manifest, workers=1)[0]
    path = tmp_path / prior.outputs[0]
    before = path.read_bytes(), path.stat().st_mtime_ns
    audio = write_render_audio(tmp_path / "audio.wav")
    selection = [narrated, silent] if reverse else [silent, narrated]
    results = render_variants(
        replace(
            spec, build=new, adopt_build=old, poster=5, audio=lambda variant: audio
        ),
        selection,
        tmp_path,
        manifest=manifest,
        workers=1,
        adopt=True,
    )
    assert [result.variant for result in results] == selection
    results.sort(key=lambda result: result.variant.narrated)
    assert results[0].adopted and not results[0].rendered
    assert results[1].rendered and not results[1].adopted
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert decoded(path) == decoded(tmp_path / results[1].outputs[0])
    assert draw_count(tmp_path / "new" / "doc-light") == 1  # current poster only
    assert draw_count(tmp_path / "old" / "doc-light") == 2  # original plus comparison
    assert all(
        manifest.current(spec.slug, result.variant.name, result.key)
        for result in results
    )


def test_equal_keys_still_pose_and_raise_on_changed_limits(tmp_path):
    from eyepiece.anim import ScaleDriftError

    def build():
        fig, ax = plt.subplots(figsize=(0.96, 0.54))
        ax.set_axis_off()

        def draw(fig, item):
            ax.set_xlim(0, 1 + item)

        return fig, draw

    # No ink or decorations: these genuinely have identical pixels.
    with pool._figure(build, 3, 100, None, None, None) as state:
        assert pool._raster(state, 0).tobytes() == pool._raster(state, 1).tobytes()
    with pytest.raises(ScaleDriftError, match="frame 1"):
        render_frames(
            build,
            3,
            tmp_path / "limits.mp4",
            fps=12,
            dpi=100,
            workers=1,
            frame_key=lambda item: 0,
            on_rescale="raise",
        )


def test_visual_canonical_preserves_mapping_sequence_and_path_identity(tmp_path):
    encode = variant_driver._canonical
    assert encode({"first": "red", "last": "blue"}) != encode(
        {"last": "blue", "first": "red"}
    )
    assert encode([1, 2]) != encode([2, 1])
    assert encode([1, 2]) != encode((1, 2))
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert encode(a) != encode(b)


@pytest.mark.parametrize("custom", ["default", "rgb", "matrix", "faststart"])
def test_narrated_output_equals_base_mux(tmp_path, custom):
    # The mux implementation is unchanged from 98aeecd. Import it directly so
    # a mutation of the variant driver's narration.mux call cannot alter this oracle.
    from eyepiece.narration._mux import mux as base_mux

    audio = write_render_audio(tmp_path / "audio.wav")
    options = {
        "default": {},
        "rgb": {
            "extra_ffmpeg_args": [
                "-c:v",
                "libx264rgb",
                "-pix_fmt",
                "rgb24",
                "-crf",
                "0",
            ]
        },
        "matrix": {"encoder": {"colorspace": None}},
        "faststart": {"encoder": {"faststart": False}},
    }
    spec = ClipSpec(
        "base-mux",
        partial(build_variant_clip, markers=tmp_path / "builds"),
        6,
        (Venue("doc", fps=12),),
        audio=lambda variant: audio,
        **options[custom],
    )
    results = render_variants(
        spec,
        [Variant("doc", "dark"), Variant("doc", "dark", True)],
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )
    silent, narrated = [tmp_path / result.outputs[0] for result in results]
    # Frozen call from 98aeecd: no encoder forwarded to the copied stream.
    reference = tmp_path / "base.mp4"
    base_mux(silent, audio, reference)
    assert narrated.read_bytes() == reference.read_bytes()
    assert decoded(narrated) == decoded(reference)


def test_ordered_frame_mappings_do_not_share_pixels(tmp_path):
    def build(venue, mode):
        fig = plt.figure(figsize=(0.96, 0.54))

        def draw(fig, item):
            for color in item.values():
                fig.set_facecolor(color)

        return fig, draw

    audio = write_render_audio(tmp_path / "audio.wav")
    spec = ClipSpec(
        "ordered",
        build,
        None,
        (Venue("doc", fps=12),),
        frames_for=lambda variant: (
            [
                {"first": "red", "last": "blue"}
                if variant.narrated
                else {"last": "blue", "first": "red"}
            ]
            * 3
        ),
        audio=lambda variant: audio,
    )
    results = render_variants(
        spec,
        [Variant("doc", "light"), Variant("doc", "light", True)],
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )
    silent, narrated = [decoded(tmp_path / result.outputs[0]) for result in results]
    silent_pixels = np.frombuffer(silent, np.uint8).reshape(-1, 54, 96, 4)
    narrated_pixels = np.frombuffer(narrated, np.uint8).reshape(-1, 54, 96, 4)
    assert silent_pixels[0, 0, 0, 0] > 200
    assert narrated_pixels[0, 0, 0, 2] > 200
    assert silent != narrated


@pytest.mark.parametrize("audio_seconds", [0.9, 2.0])
def test_narrated_impulse_aligns_with_visual_event_and_holds(tmp_path, audio_seconds):
    import wave

    fps, rate = 12, 48000
    hold, event = 3, 4
    event_time = (hold + event) / fps
    samples = np.zeros(round(audio_seconds * rate), dtype="<i2")
    samples[round(event_time * rate)] = 28000
    if audio_seconds > 1.5:
        # A second impulse beyond the presentation duration must be trimmed.
        samples[round(1.8 * rate)] = 28000
    audio = tmp_path / "impulse.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())

    def build(venue, mode):
        fig = plt.figure(figsize=(0.96, 0.54))

        def draw(fig, item):
            fig.set_facecolor("white" if item < event else "black")

        return fig, draw

    spec = ClipSpec(
        "aligned",
        build,
        12,
        (Venue("doc", fps=fps),),
        hold_s=(hold / fps, hold / fps),
        audio=lambda variant: audio,
        frame_key=lambda item: item >= event,
    )
    result = render_variants(
        spec,
        [Variant("doc", "light", True)],
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )[0]
    video = tmp_path / result.outputs[0]
    frames = np.frombuffer(decoded(video), np.uint8).reshape(-1, 54, 96, 4)
    assert len(frames) == 18
    visual_event = np.flatnonzero(frames[:, 0, 0, 0] < 128)[0]
    assert visual_event == hold + event
    raw = subprocess.run(
        [
            resolve_ffmpeg(),
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    sound = np.frombuffer(raw, dtype="<f4")
    peak = np.argmax(np.abs(sound)) / rate
    assert abs(peak - visual_event / fps) < 0.01
    assert abs(len(sound) / rate - len(frames) / fps) < 1024 / rate
    assert np.max(np.abs(sound[rate:])) < 0.01


def test_shared_variants_paint_only_one_video(tmp_path, monkeypatch):
    from matplotlib.backends.backend_agg import RendererAgg

    paints = []
    original = RendererAgg.draw_path

    def draw_path(self, *args, **kwargs):
        paints.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RendererAgg, "draw_path", draw_path)
    audio = write_render_audio(tmp_path / "audio.wav")
    spec = ClipSpec(
        "paint-once",
        partial(build_variant_clip, markers=tmp_path / "builds"),
        3,
        (Venue("doc", fps=12),),
        audio=lambda variant: audio,
    )
    silent, narrated = Variant("doc", "light"), Variant("doc", "light", True)
    counts = []
    for name, selection in (("one", [silent]), ("pair", [silent, narrated])):
        paints.clear()
        folder = tmp_path / name
        render_variants(
            spec,
            selection,
            folder,
            manifest=Manifest(folder / "manifest.json"),
            workers=1,
        )
        counts.append(len(paints))
    assert counts[0] > 0
    assert counts[0] == counts[1]
