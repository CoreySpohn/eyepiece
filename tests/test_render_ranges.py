"""Excerpt rendering preserves absolute frame items and full-output identities."""

import os
import subprocess
import sys
from dataclasses import replace
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_render_helpers import (
    build_variant_clip,
    decode_variant_video,
    variant_frame_at,
)

import eyepiece as ep


@pytest.fixture
def spec(tmp_path):
    return ep.render.ClipSpec(
        "excerpt",
        partial(build_variant_clip, markers=tmp_path / "builds"),
        6,
        (ep.render.Venue("doc", dpi=100, fps=12),),
    )


def render(spec, tmp_path, time_range=None, **kwargs):
    folder = tmp_path / "out"
    manifest = ep.render.Manifest.load(folder / "manifest.json")
    (result,) = ep.render.render_variants(
        spec,
        [ep.render.Variant("doc", "light")],
        folder,
        manifest=manifest,
        workers=kwargs.pop("workers", 1),
        time_range=time_range,
        **kwargs,
    )
    return result, folder / result.outputs[0]


@pytest.mark.parametrize("mapping", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_variants_excerpt_uses_original_items_and_keys(
    spec, tmp_path, mapping, workers
):
    if mapping:
        spec = replace(spec, frame_at=variant_frame_at)
    full, path = render(spec, tmp_path, workers=workers)
    original = path.read_bytes()
    full_pixels = decode_variant_video(path)
    explicit, _ = render(spec, tmp_path, (0, 0.5), workers=workers)
    assert explicit.key == full.key and not explicit.rendered
    assert path.read_bytes() == original
    part, path = render(spec, tmp_path, (1 / 12, 4 / 12), workers=workers)
    assert part.key != full.key and part.frames == 3
    # Inter-frame encoding can change rounding; the marker's trajectory cannot.
    pixels = decode_variant_video(path)
    assert pixels.shape[0] == 3
    assert np.abs(pixels.astype(float) - full_pixels[1:4]).mean() < 1
    again, _ = render(spec, tmp_path, (1 / 12, 4 / 12), workers=workers)
    assert not again.rendered and again.key == part.key
    neighboring, _ = render(spec, tmp_path, (2 / 12, 5 / 12), workers=workers)
    assert neighboring.key != part.key and neighboring.frames == part.frames
    restored, path = render(spec, tmp_path, workers=workers)
    assert restored.key == full.key
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "time_range",
    [
        (-1, 0.2),
        (0.2, 0.2),
        (0.4, 0.2),
        (0, 1),
        (0, float("nan")),
        (0, 0.01, 0.02),
        (0.001, 0.01),
    ],
)
def test_variant_bad_range_fails_before_build(spec, tmp_path, time_range):
    with pytest.raises(ValueError, match="time_range"):
        render(spec, tmp_path, time_range)
    assert not (tmp_path / "builds").exists()


def test_record_range_filters_source_calls_and_keeps_stills_absolute(tmp_path):
    fig = plt.figure(figsize=(0.96, 0.54), dpi=100)
    text = fig.text(0.2, 0.3, "", fontsize=8)
    path = tmp_path / "part.gif"
    still = tmp_path / "still.png"
    with ep.render.record(
        fig, path, fps=10, time_range=(0.2, 0.5), stills={still: 0.3}
    ) as rec:
        for k in range(7):
            text.set_text(str(k))
            rec.frame()
    from PIL import Image

    with Image.open(path) as image:
        assert image.n_frames == 3
    assert still.exists()
    expected = tmp_path / "expected.gif"
    with ep.render.record(fig, expected, fps=10) as rec:
        for k in range(2, 5):
            text.set_text(str(k))
            rec.frame()
    assert expected.read_bytes() == path.read_bytes()
    plt.close(fig)


def test_record_rejects_range_and_stills_outside_excerpt(tmp_path):
    fig = plt.figure()
    for bounds in [(-1, 2), (2, 1), (0, float("inf"))]:
        with pytest.raises(ValueError, match="time_range"):
            with ep.render.record(fig, tmp_path / "bad.gif", time_range=bounds):
                pass
    with pytest.raises(ValueError, match="still"):
        with ep.render.record(
            fig,
            tmp_path / "bad.gif",
            fps=10,
            time_range=(0.2, 0.5),
            stills={tmp_path / "x.png": 0},
        ):
            pass
    plt.close(fig)


def test_main_builder_error_names_importable_module(tmp_path):
    script = tmp_path / "render_script.py"
    script.write_text("""
import eyepiece as ep
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def build():
    fig = plt.figure(figsize=(1, 1))
    return fig, lambda fig, item: None

ep.render.render_frames(build, 2, Path(sys.argv[1]), fps=10, dpi=50,
                        workers=2, chunk_frames=1)
""")
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"previous")
    done = subprocess.run(
        [sys.executable, str(script), str(path)],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode != 0
    assert "define build" in done.stderr
    assert "importable module" in done.stderr
    assert path.read_bytes() == b"previous"


def test_sequence_excerpt_and_short_budget(spec, tmp_path):
    spec = replace(
        spec,
        frames=(0, 2, 1, 5, 4, 3),
        venues=(spec.venues[0]._replace(frame_budget=2),),
    )
    part, path = render(spec, tmp_path, (2 / 12, 4 / 12))
    assert part.frames == 2
    assert len(decode_variant_video(path)) == 2


@pytest.mark.parametrize("first_hold", [0, 1 / 12])
def test_excerpt_audio_starts_at_the_same_source_time(
    spec, tmp_path, monkeypatch, first_hold
):
    import shutil
    import wave

    rate = 48000
    audio = tmp_path / "speech.wav"
    pcm = np.r_[np.full(rate // 4, 1000), np.full(rate // 4, 3000)].astype("<i2")
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())
    heard = []

    def mux(video, track, path, **kwargs):
        with wave.open(str(track), "rb") as stream:
            samples = np.frombuffer(stream.readframes(stream.getnframes()), "<i2")
            heard.append(samples.copy())
        shutil.copyfile(video, path)

    monkeypatch.setattr(ep.narration, "mux", mux)
    spec = replace(spec, audio=lambda variant: audio, hold_s=(first_hold, 0))
    folder = tmp_path / "out"
    (result,) = ep.render.render_variants(
        spec,
        [ep.render.Variant("doc", "light", True)],
        folder,
        manifest=ep.render.Manifest.load(folder / "manifest.json"),
        workers=1,
        time_range=(0.25, 0.5),
    )
    assert result.frames == 3 + bool(first_hold)
    assert len(heard) == 1
    held_samples = round(first_hold * rate)
    assert len(heard[0]) == rate // 4 + held_samples
    assert np.all(heard[0][:held_samples] == 0)
    assert np.all(heard[0][held_samples:] == 3000)


def test_record_range_requires_matching_plan_fps_and_complete_window(tmp_path):
    fig = plt.figure()
    tl = ep.Timeline()
    tl.hold(1)
    plan = tl.at_fps(10)
    with pytest.raises(ValueError, match=r"plan.*fps"):
        with ep.render.record(
            fig, tmp_path / "bad.gif", fps=5, plan=plan, time_range=(0, 0.5)
        ):
            pass
    with pytest.raises(RuntimeError, match="time_range stop"):
        with ep.render.record(
            fig, tmp_path / "short.gif", fps=10, time_range=(0, 0.5)
        ) as rec:
            rec.hold(3)
    plt.close(fig)


def test_excerpt_is_byte_identical_across_spawned_workers(spec, tmp_path):
    spec = replace(
        spec,
        frames=32,
        venues=(spec.venues[0]._replace(fps=16),),
        frame_at=variant_frame_at,
    )
    serial, path = render(spec, tmp_path, (0.25, 0.875), workers=1)
    reference = path.read_bytes()
    parallel, path = render(spec, tmp_path, (0.25, 0.875), workers=2, force=True)
    assert serial.key == parallel.key
    assert parallel.frames == 10
    assert path.read_bytes() == reference
    pids = (tmp_path / "builds" / "doc-light").read_text().splitlines()
    assert len(pids) == len(set(pids)) == 3


def test_tail_roundoff_does_not_add_a_source_frame(spec, tmp_path):
    full, path = render(spec, tmp_path)
    reference = path.read_bytes()
    rounded, path = render(spec, tmp_path, (0, 0.5 + 5e-10))
    assert rounded.key == full.key
    assert rounded.frames == full.frames
    assert path.read_bytes() == reference


@pytest.mark.parametrize("time_range", [None, (0, None)])
def test_record_full_range_keeps_holds_after_plan(tmp_path, time_range):
    fig = plt.figure(figsize=(1, 1), dpi=50)
    tl = ep.Timeline()
    tl.hold(1)
    plan = tl.at_fps(3)
    path = tmp_path / "held.html"
    kwargs = {} if time_range is None else {"time_range": time_range}
    with ep.render.record(fig, path, plan=plan, fps=3, dpi=50, **kwargs) as rec:
        rec.hold(5)
    assert path.read_text().count("data:image/png") == 5
    plt.close(fig)
