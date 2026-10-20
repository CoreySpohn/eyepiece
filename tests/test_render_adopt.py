"""Adoption compares old renders without replacing matching committed videos."""

import os
import wave
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from PIL import Image
from test_render_helpers import (
    build_adopt_variant,
    decode_variant_video,
    variant_frame_at,
    write_render_audio,
)

from eyepiece.render import (
    ClipSpec,
    Manifest,
    Variant,
    VariantResult,
    Venue,
    render_variants,
)


@pytest.fixture
def clip(tmp_path):
    return ClipSpec(
        "adoption",
        partial(build_adopt_variant, markers=tmp_path / "old", color="red"),
        4,
        (Venue("doc", fps=12),),
        extra=("old",),
    )


def new_spec(clip, tmp_path):
    return replace(
        clip,
        build=partial(build_adopt_variant, markers=tmp_path / "new", color="blue"),
        adopt_build=clip.build,
        extra=("new",),
    )


def assert_no_scratch(tmp_path):
    assert list(tmp_path.glob(".*.mp4")) == []
    assert list(tmp_path.glob(".*.adopt-*")) == []
    assert list(tmp_path.glob(".adopt-outputs-*")) == []


def preserved_file(path):
    """Capture bytes and an old timestamp that exposes identical rewrites."""
    os.utime(path, (946684800, 946684800))
    return path.read_bytes(), path.stat().st_mtime_ns


@pytest.mark.parametrize("prior_entry", [False, True])
@pytest.mark.parametrize("narrated", [False, True])
def test_matching_old_video_adopts_and_then_skips(
    clip, tmp_path, prior_entry, narrated
):
    if narrated:
        audio = write_render_audio(tmp_path / "track.wav")
        clip = replace(
            clip,
            audio=lambda variant: audio,
            frames=None,
            frames_for=lambda variant: 6,
            frame_at_for=lambda variant: variant_frame_at,
        )
    variant = Variant("doc", "light", narrated)
    old_manifest = Manifest(tmp_path / "old.json")
    old = render_variants(clip, [variant], tmp_path, manifest=old_manifest, workers=1)[
        0
    ]
    target = tmp_path / old.outputs[0]
    before = preserved_file(target)
    manifest = old_manifest if prior_entry else Manifest(tmp_path / "manifest.json")
    spec = new_spec(clip, tmp_path)
    result = render_variants(
        spec, [variant], tmp_path, manifest=manifest, workers=1, adopt=True
    )[0]
    assert result.adopted and not result.rendered
    assert result.key != old.key
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert manifest.current(spec.slug, variant.name, result.key)
    assert not (tmp_path / "new").exists()
    assert_no_scratch(tmp_path)
    builds = (tmp_path / "old" / "doc-light").read_bytes()
    kept = render_variants(spec, [variant], tmp_path, manifest=manifest, workers=1)[0]
    assert not kept.rendered and not kept.adopted
    # Even an explicit adoption request skips all building for a current entry.
    kept = render_variants(
        spec, [variant], tmp_path, manifest=manifest, workers=1, adopt=True
    )[0]
    assert not kept.rendered and not kept.adopted
    assert (tmp_path / "old" / "doc-light").read_bytes() == builds


def test_different_old_candidate_renders_new_build(clip, tmp_path):
    variant = Variant("doc", "light")
    manifest = Manifest(tmp_path / "manifest.json")
    old = render_variants(clip, [variant], tmp_path, manifest=manifest, workers=1)[0]
    target = tmp_path / old.outputs[0]
    before = target.read_bytes()
    spec = replace(
        new_spec(clip, tmp_path),
        adopt_build=partial(
            build_adopt_variant, markers=tmp_path / "candidate", color="green"
        ),
    )
    result = render_variants(
        spec, [variant], tmp_path, manifest=manifest, workers=1, adopt=True
    )[0]
    assert result.rendered and not result.adopted
    assert target.read_bytes() != before
    assert (tmp_path / "candidate" / "doc-light").is_file()
    assert (tmp_path / "new" / "doc-light").is_file()
    assert_no_scratch(tmp_path)


def test_audio_only_mismatch_renders_instead_of_adopting(clip, tmp_path):
    audio = write_render_audio(tmp_path / "track.wav")
    clip = replace(clip, audio=lambda variant: audio)
    variant = Variant("doc", "light", True)
    manifest = Manifest(tmp_path / "manifest.json")
    first = render_variants(clip, [variant], tmp_path, manifest=manifest, workers=1)[0]
    target = tmp_path / first.outputs[0]
    before = target.read_bytes()
    pixels = decode_variant_video(target).copy()
    # Change only the audio samples; both builders draw the original picture.
    with wave.open(str(audio), "rb") as stream:
        params = stream.getparams()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    with wave.open(str(audio), "wb") as stream:
        stream.setparams(params)
        stream.writeframes((-samples).tobytes())
    spec = replace(
        clip,
        build=partial(build_adopt_variant, markers=tmp_path / "new", color="red"),
        adopt_build=clip.build,
    )
    result = render_variants(
        spec, [variant], tmp_path, manifest=manifest, workers=1, adopt=True
    )[0]
    assert result.rendered and not result.adopted
    assert result.key != first.key
    assert target.read_bytes() != before
    np.testing.assert_array_equal(pixels, decode_variant_video(target))
    assert len((tmp_path / "old" / "doc-light").read_text().splitlines()) == 2
    assert len((tmp_path / "new" / "doc-light").read_text().splitlines()) == 1
    assert manifest.current(spec.slug, variant.name, result.key)
    assert_no_scratch(tmp_path)


def test_adopt_requires_builder_in_preflight(clip, tmp_path):
    with pytest.raises(ValueError, match="adopt_build"):
        render_variants(
            clip,
            [Variant("doc", "light")],
            tmp_path,
            manifest=Manifest(tmp_path / "manifest.json"),
            workers=1,
            adopt=True,
        )
    assert not (tmp_path / "old").exists()


def test_force_never_adopts(clip, tmp_path):
    variant = Variant("doc", "light")
    manifest = Manifest(tmp_path / "manifest.json")
    old = render_variants(clip, [variant], tmp_path, manifest=manifest, workers=1)[0]
    target = tmp_path / old.outputs[0]
    before = target.read_bytes()
    old_builds = (tmp_path / "old" / "doc-light").read_bytes()
    result = render_variants(
        new_spec(clip, tmp_path),
        [variant],
        tmp_path,
        manifest=manifest,
        workers=1,
        force=True,
        adopt=True,
    )[0]
    assert result.rendered and not result.adopted
    assert target.read_bytes() != before
    assert (tmp_path / "old" / "doc-light").read_bytes() == old_builds
    assert_no_scratch(tmp_path)


@pytest.mark.parametrize("poster_exists", [False, True])
@pytest.mark.parametrize("after_exists", [False, True])
def test_adoption_regenerates_poster_and_after_outputs(
    clip, tmp_path, poster_exists, after_exists
):
    def after(variant, out_dir):
        path = out_dir / "captions.vtt"
        path.write_text("WEBVTT\n")
        return [path]

    variant = Variant("doc", "light")
    clip = replace(clip, poster=2, after=after)
    first = render_variants(
        clip, [variant], tmp_path, manifest=Manifest(tmp_path / "old.json"), workers=1
    )[0]
    outputs = [tmp_path / path for path in first.outputs]
    for exists, path in zip((poster_exists, after_exists), outputs[1:], strict=True):
        if not exists:
            path.unlink()
    before = preserved_file(outputs[0])
    manifest = Manifest(tmp_path / "manifest.json")
    result = render_variants(
        new_spec(clip, tmp_path),
        [variant],
        tmp_path,
        manifest=manifest,
        workers=1,
        adopt=True,
    )[0]
    assert result.adopted and not result.rendered
    assert result.outputs == first.outputs
    for path in outputs:
        assert path.is_file()
    assert (outputs[0].read_bytes(), outputs[0].stat().st_mtime_ns) == before
    with Image.open(outputs[1]) as poster:
        assert poster.getpixel((2, 2))[:3] == (0, 0, 255)
    assert outputs[2].read_text() == "WEBVTT\n"
    assert manifest.current(clip.slug, variant.name, result.key)
    assert_no_scratch(tmp_path)


@pytest.mark.parametrize("access", ["absolute", "video"])
def test_adoption_after_uses_real_output_directory(clip, tmp_path, access):
    variant = Variant("doc", "light")
    target = tmp_path / f"{clip.slug}-{variant.name}.mp4"
    caption = tmp_path / "captions.vtt"
    calls = []

    def after(variant, out_dir):
        calls.append(out_dir)
        if access == "absolute":
            path = caption
            text = "WEBVTT\n"
        else:
            path = out_dir / "captions.vtt"
            size = len((out_dir / target.name).read_bytes())
            text = f"WEBVTT\n\nNOTE video bytes: {size}\n"
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
        return [path]

    clip = replace(clip, after=after)
    manifest = Manifest(tmp_path / "manifest.json")
    first = render_variants(clip, [variant], tmp_path, manifest=manifest, workers=1)[0]
    before = {path: preserved_file(path) for path in (target, caption)}
    spec = new_spec(clip, tmp_path)
    result = render_variants(
        spec, [variant], tmp_path, manifest=manifest, workers=1, adopt=True
    )[0]
    assert result.adopted and not result.rendered
    assert result.outputs == first.outputs
    assert calls == [tmp_path.resolve(), tmp_path.resolve()]
    for path, expected in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == expected
    assert manifest.current(spec.slug, variant.name, result.key)
    assert_no_scratch(tmp_path)


def test_failing_after_leaves_no_adopted_manifest_entry(clip, tmp_path):
    variant = Variant("doc", "light")
    manifest = Manifest(tmp_path / "manifest.json")
    first = render_variants(clip, [variant], tmp_path, manifest=manifest, workers=1)[0]
    target = tmp_path / first.outputs[0]
    before = preserved_file(target)

    def after(variant, out_dir):
        assert (out_dir / target.name).read_bytes() == before[0]
        raise OSError("deliberate after failure")

    spec = replace(new_spec(clip, tmp_path), after=after)
    with pytest.raises(OSError, match="deliberate after failure"):
        render_variants(
            spec, [variant], tmp_path, manifest=manifest, workers=1, adopt=True
        )
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert manifest.entry(spec.slug, variant.name) is None
    assert Manifest.load(manifest.path).entry(spec.slug, variant.name) is None
    assert_no_scratch(tmp_path)


def test_failed_adoption_cleans_scratch_and_preserves_target(
    clip, tmp_path, monkeypatch
):
    from eyepiece import narration

    audio = write_render_audio(tmp_path / "track.wav")
    clip = replace(clip, audio=lambda variant: audio)
    variant = Variant("doc", "light", True)
    manifest = Manifest(tmp_path / "manifest.json")
    first = render_variants(clip, [variant], tmp_path, manifest=manifest, workers=1)[0]
    target = tmp_path / first.outputs[0]
    before = preserved_file(target)

    def failed_mux(video, audio, path):
        path.write_bytes(b"partial")
        raise OSError("deliberate mux failure")

    monkeypatch.setattr(narration, "mux", failed_mux)
    with pytest.raises(OSError, match="deliberate mux failure"):
        render_variants(
            new_spec(clip, tmp_path),
            [variant],
            tmp_path,
            manifest=manifest,
            workers=1,
            adopt=True,
        )
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert manifest.entry(clip.slug, variant.name) is None
    assert_no_scratch(tmp_path)


def test_variant_result_adopted_default_is_backward_compatible():
    result = VariantResult(Variant("doc", "light"), [], "key", False, 4)
    assert result.adopted is False
