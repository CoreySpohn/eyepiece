"""Each variant resolves its own timeline before any rendering starts."""

import os
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from test_render_helpers import (
    build_variant_clip,
    decode_variant_video,
    variant_frame_at,
    write_render_audio,
)

import eyepiece._render_variants as variant_driver
from eyepiece.render import ClipSpec, Manifest, Variant, Venue, render_variants


@pytest.fixture
def clip(tmp_path):
    return ClipSpec(
        "timelines",
        partial(build_variant_clip, markers=tmp_path / "builds"),
        8,
        (Venue("doc", fps=12),),
    )


@pytest.mark.parametrize("as_sequence", [False, True])
def test_narrated_timeline_counts_and_selective_invalidation(
    clip, tmp_path, as_sequence
):
    audio = write_render_audio(tmp_path / "audio.wav")
    lengths = {False: 8, True: 12}
    sequences = {narrated: list(range(count)) for narrated, count in lengths.items()}
    parent = os.getpid()
    calls = []

    def frames_for(variant):
        assert os.getpid() == parent
        calls.append(variant)
        return sequences[variant.narrated] if as_sequence else lengths[variant.narrated]

    spec = replace(clip, frames_for=frames_for, audio=lambda variant: audio)
    requested = [Variant("doc", "light"), Variant("doc", "dark", True)]
    manifest = Manifest(tmp_path / "manifest.json")
    first = render_variants(
        spec, requested + requested, tmp_path, manifest=manifest, workers=1
    )
    assert calls == requested
    assert [result.frames for result in first] == [8, 12]
    assert [
        len(decode_variant_video(tmp_path / result.outputs[0])) for result in first
    ] == [8, 12]
    silent, narrated = [tmp_path / result.outputs[0] for result in first]
    before_pixels = [decode_variant_video(path).copy() for path in (silent, narrated)]
    counters = [tmp_path / "builds" / name for name in ("doc-light", "doc-dark")]
    assert [len(path.read_text().splitlines()) for path in counters] == [1, 1]
    silent_bytes, silent_mtime = silent.read_bytes(), silent.stat().st_mtime_ns
    if as_sequence:
        sequences[True][3] = 6
    else:
        lengths[True] = 14
    second = render_variants(spec, requested, tmp_path, manifest=manifest, workers=1)
    assert [result.rendered for result in second] == [False, True]
    assert first[0].key == second[0].key
    assert first[1].key != second[1].key
    after_pixels = [
        decode_variant_video(tmp_path / result.outputs[0]) for result in second
    ]
    assert len(after_pixels[1]) == (12 if as_sequence else 14)
    np.testing.assert_array_equal(before_pixels[0], after_pixels[0])
    if as_sequence:
        assert np.any(before_pixels[1][3] != after_pixels[1][3])
    assert [len(path.read_text().splitlines()) for path in counters] == [1, 2]
    assert (silent.read_bytes(), silent.stat().st_mtime_ns) == (
        silent_bytes,
        silent_mtime,
    )


def test_parent_frame_callbacks_and_worker_mapping_match_explicit_frames(
    clip, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        variant_driver,
        "render_frames",
        partial(variant_driver.render_frames, chunk_frames=2),
    )
    parent = os.getpid()
    requested = [Variant("doc", "light"), Variant("doc", "dark")]
    calls = []

    def frames_for(variant):
        assert os.getpid() == parent
        return 4

    def frame_at_for(variant):
        assert os.getpid() == parent
        calls.append(variant)
        return variant_frame_at if variant.mode == "light" else None

    mapped = replace(
        clip,
        frames=None,
        frame_at=variant_frame_at,
        frames_for=frames_for,
        frame_at_for=frame_at_for,
    )
    manifest = Manifest(tmp_path / "manifest.json")
    first = render_variants(mapped, requested, tmp_path, manifest=manifest, workers=2)
    assert calls == requested
    pixels = [decode_variant_video(tmp_path / result.outputs[0]) for result in first]
    explicit = replace(
        clip,
        frames_for=lambda variant: (
            [variant_frame_at(i) for i in range(4)]
            if variant.mode == "light"
            else list(range(4))
        ),
    )
    second = render_variants(
        explicit, requested, tmp_path, manifest=manifest, workers=1
    )
    for baseline, result in zip(pixels, second, strict=True):
        np.testing.assert_array_equal(
            baseline, decode_variant_video(tmp_path / result.outputs[0])
        )


@pytest.mark.parametrize("missing", ["both", "later", "empty", "mapping"])
def test_invalid_variant_frames_preflight_every_build(clip, tmp_path, missing):
    spec = replace(clip, frames=None)
    if missing == "later":
        spec = replace(
            spec, frames_for=lambda variant: 4 if variant.mode == "light" else None
        )
    elif missing == "empty":
        spec = replace(spec, frames_for=lambda variant: [])
    elif missing == "mapping":
        spec = replace(
            spec,
            frames_for=lambda variant: [0, 1],
            frame_at_for=lambda variant: variant_frame_at,
        )
    with pytest.raises(ValueError, match="frames"):
        render_variants(
            spec,
            [Variant("doc", "light"), Variant("doc", "dark")],
            tmp_path,
            manifest=Manifest(tmp_path / "manifest.json"),
            workers=1,
        )
    assert not (tmp_path / "builds").exists()


def test_budget_uses_silent_variant_count(clip, tmp_path):
    spec = replace(
        clip,
        frames=1,
        venues=(Venue("doc", fps=12, frame_budget=4),),
        frames_for=lambda variant: 5 if variant.mode == "dark" else 4,
    )
    with pytest.raises(ValueError, match=r"5.*budget 4"):
        render_variants(
            spec,
            [Variant("doc", "light"), Variant("doc", "dark")],
            tmp_path,
            manifest=Manifest(tmp_path / "manifest.json"),
            workers=1,
        )
    assert not (tmp_path / "builds").exists()


def test_variant_frames_feed_holds_and_poster(clip, tmp_path):
    spec = replace(
        clip,
        frames=1,
        frames_for=lambda variant: 4,
        frame_at_for=lambda variant: variant_frame_at,
        poster=3,
        hold_s=(1 / 12, 2 / 12),
    )
    result = render_variants(
        spec,
        [Variant("doc", "light")],
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )[0]
    pixels = decode_variant_video(tmp_path / result.outputs[0])
    assert result.frames == len(pixels) == 7
    explicit = replace(
        spec,
        frames=4,
        frame_at=variant_frame_at,
        frames_for=None,
        frame_at_for=None,
    )
    expected_dir = tmp_path / "expected"
    expected = render_variants(
        explicit,
        [result.variant],
        expected_dir,
        manifest=Manifest(expected_dir / "manifest.json"),
        workers=1,
    )[0]
    for actual_path, expected_path in zip(
        result.outputs, expected.outputs, strict=True
    ):
        assert (tmp_path / actual_path).read_bytes() == (
            expected_dir / expected_path
        ).read_bytes()


def test_poster_validated_against_each_variant(clip, tmp_path):
    spec = replace(
        clip,
        frames_for=lambda variant: 8 if variant.mode == "light" else 2,
        poster=3,
    )
    with pytest.raises(ValueError, match="poster"):
        render_variants(
            spec,
            [Variant("doc", "light"), Variant("doc", "dark")],
            tmp_path,
            manifest=Manifest(tmp_path / "manifest.json"),
            workers=1,
        )
    assert not (tmp_path / "builds").exists()


def test_poster_index_only_applies_to_silent_timeline(clip, tmp_path):
    audio = write_render_audio(tmp_path / "audio.wav")
    spec = replace(
        clip,
        frames_for=lambda variant: 2 if variant.narrated else 8,
        audio=lambda variant: audio,
        poster=3,
    )
    requested = [Variant("doc", "light"), Variant("doc", "light", True)]
    results = render_variants(
        spec,
        requested,
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )
    assert [result.frames for result in results] == [8, 2]
    assert [result.rendered for result in results] == [True, True]
    assert [len(result.outputs) for result in results] == [2, 1]
    assert (tmp_path / results[0].outputs[1]).is_file()
    assert len(list(tmp_path.glob("*-poster.png"))) == 1
