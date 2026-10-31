"""Variant keys avoid speculative builds and preserve deterministic outputs."""

import errno
import os
import subprocess
import tempfile
import wave
from dataclasses import FrozenInstanceError, replace
from functools import partial
from pathlib import Path

import hwostyle
import matplotlib
import numpy as np
import pytest
from test_render_helpers import (
    build_style_variant,
    build_variant_clip,
    setup_video_style,
    variant_frame_at,
)

import eyepiece as ep
import eyepiece._render_variants as variant_driver
import eyepiece.render as render
from eyepiece._ffmpeg import resolve_ffmpeg
from eyepiece.render import (
    ClipSpec,
    Manifest,
    Variant,
    VariantResult,
    Venue,
    render_variants,
    variants,
)


def write_audio(path, frequency=440):
    rate = 48_000
    time = np.arange(rate) / rate
    samples = np.round(6000 * np.sin(2 * np.pi * frequency * time)).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())
    return path


def decode(path):
    result = subprocess.run(
        [
            resolve_ffmpeg(), "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )  # fmt: skip
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 54, 96, 3)


def builds(folder):
    if not folder.exists():
        return {}
    return {path.name: len(path.read_text().splitlines()) for path in folder.iterdir()}


@pytest.fixture
def clip(tmp_path):
    return ClipSpec(
        slug="marker",
        build=partial(build_variant_clip, markers=tmp_path / "builds"),
        frames=6,
        venues=(Venue("doc", {}, 100, 12), Venue("talk", None, 100, 12)),
    )


def test_public_types_defaults_and_fixed_order():
    for name in (
        "Venue",
        "Variant",
        "variants",
        "ClipSpec",
        "VariantResult",
        "render_variants",
    ):
        assert getattr(ep, name) is getattr(render, name)
        assert name in ep.render.__all__
    assert Variant("doc", "light").name == "doc-light"
    assert Variant("doc", "light", True).name == "doc-light-narrated"
    assert Venue("doc", dpi=100, fps=12).rc is None
    expected = [
        Variant("doc", "light"),
        Variant("doc", "dark"),
        Variant("talk", "light"),
        Variant("talk", "dark"),
        Variant("doc", "light", True),
        Variant("doc", "dark", True),
    ]
    assert (
        variants(iter(("doc", "talk")), iter(("light", "dark")), ("doc",)) == expected
    )
    with pytest.raises(ValueError, match="missing"):
        variants(("doc",), narrated=("missing",))


def test_spec_is_frozen(clip):
    with pytest.raises(FrozenInstanceError):
        clip.slug = "changed"


def test_skip_selective_data_force_and_manifest(clip, tmp_path):
    changed = np.array([0.0, 0.0, 1.0])
    fixed = np.array([0.0, 1.0, 0.0])
    data = {"dark": changed, "light": fixed}
    spec = replace(
        clip,
        build=partial(build_variant_clip, markers=tmp_path / "builds", data=data),
        data=lambda variant: (data[variant.mode],),
    )
    out = tmp_path / "out"
    manifest = Manifest(out / "manifest.json")
    requested = variants(("doc",), ("light", "dark"))
    first = render_variants(spec, requested, out, manifest=manifest, workers=1)
    assert all(
        isinstance(result, VariantResult) and result.rendered for result in first
    )
    assert [result.outputs for result in first] == [
        ["marker-doc-light.mp4"],
        ["marker-doc-dark.mp4"],
    ]
    assert all(result.frames == spec.frames for result in first)
    assert builds(tmp_path / "builds") == {"doc-light": 1, "doc-dark": 1}
    loaded = Manifest.load(manifest.path)
    for result in first:
        assert loaded.current(spec.slug, result.variant.name, result.key)
    second = render_variants(spec, requested, out, manifest=loaded, workers=1)
    assert all(not result.rendered for result in second)
    assert [result.key for result in second] == [result.key for result in first]
    assert builds(tmp_path / "builds") == {"doc-light": 1, "doc-dark": 1}
    before_dark = decode(out / first[1].outputs[0]).copy()
    changed[:] = [1.0, 0.0, 0.0]
    third = render_variants(spec, requested, out, manifest=loaded, workers=1)
    assert [result.rendered for result in third] == [False, True]
    assert third[0].key == first[0].key
    assert third[1].key != first[1].key
    np.testing.assert_allclose(before_dark[0, 2, 2], [0, 0, 255], atol=3)
    np.testing.assert_allclose(
        decode(out / third[1].outputs[0])[0, 2, 2], [255, 0, 0], atol=3
    )
    assert builds(tmp_path / "builds") == {"doc-light": 1, "doc-dark": 2}
    forced = render_variants(
        spec, requested + requested, out, manifest=loaded, workers=1, force=True
    )
    assert all(result.rendered for result in forced)
    assert builds(tmp_path / "builds") == {"doc-light": 2, "doc-dark": 3}
    light, dark = [decode(out / result.outputs[0]) for result in first]
    np.testing.assert_allclose(light[0, 2, 2], [0, 255, 0], atol=3)
    np.testing.assert_allclose(dark[0, 2, 2], [255, 0, 0], atol=3)
    assert (out / first[0].outputs[0]).read_bytes() != (
        out / first[1].outputs[0]
    ).read_bytes()


def test_audio_keys_streams_and_mux_determinism(clip, tmp_path):
    audio = write_audio(tmp_path / "track.wav")
    spec = replace(clip, audio=lambda variant: audio)
    out = tmp_path / "out"
    manifest = Manifest(out / "manifest.json")
    requested = variants(("doc",), ("light",), narrated=("doc",))
    first = render_variants(spec, requested, out, manifest=manifest, workers=1)
    assert builds(tmp_path / "builds") == {"doc-light": 1}
    silent, narrated = [out / result.outputs[0] for result in first]
    streams = subprocess.run(
        [resolve_ffmpeg(), "-i", str(narrated)],
        capture_output=True,
        text=True,
        timeout=30,
    ).stderr
    assert "Video:" in streams and "Audio:" in streams
    np.testing.assert_array_equal(decode(silent), decode(narrated))
    before = narrated.read_bytes()
    count = builds(tmp_path / "builds")["doc-light"]
    render_variants(spec, [requested[1]], out, manifest=manifest, workers=1, force=True)
    assert builds(tmp_path / "builds")["doc-light"] == count + 1
    assert narrated.read_bytes() == before
    unchanged = render_variants(spec, requested, out, manifest=manifest, workers=1)
    assert [result.rendered for result in unchanged] == [False, False]
    count = builds(tmp_path / "builds")["doc-light"]
    write_audio(audio, frequency=660)
    changed = render_variants(spec, requested, out, manifest=manifest, workers=1)
    assert [result.rendered for result in changed] == [False, True]
    assert changed[0].key == first[0].key
    assert changed[1].key != first[1].key
    assert builds(tmp_path / "builds")["doc-light"] == count


@pytest.mark.parametrize("problem", ["budget", "unknown", "audio"])
def test_preflight_checks_every_variant_before_build(clip, tmp_path, problem):
    spec = clip
    requested = [Variant("doc", "light"), Variant("talk", "dark")]
    if problem == "budget":
        spec = replace(
            clip, venues=(clip.venues[0], clip.venues[1]._replace(frame_budget=5))
        )
        match = r"marker.*talk.*6.*5"
    elif problem == "unknown":
        requested[1] = Variant("missing", "dark")
        match = "missing"
    else:
        requested[1] = Variant("talk", "dark", True)
        match = "audio"
    with pytest.raises(ValueError, match=match):
        render_variants(
            spec,
            requested,
            tmp_path,
            manifest=Manifest(tmp_path / "manifest.json"),
            workers=1,
        )
    assert builds(tmp_path / "builds") == {}


def test_speech_length_exempts_narrated_variants_from_silent_budget(clip, tmp_path):
    """Speech sets narrated length; the venue budget limits silent clips."""
    audio = write_audio(tmp_path / "track.wav")
    spec = replace(
        clip,
        venues=(clip.venues[0]._replace(frame_budget=1),),
        audio=lambda variant: audio,
    )
    result = render_variants(
        spec,
        [Variant("doc", "light", True)],
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )
    assert result[0].rendered


def test_after_on_kept_variants_missing_extra_and_relative_paths(clip, tmp_path):
    calls = []

    def after(variant, out_dir):
        calls.append(variant)
        relative = Path("captions") / f"{variant.name}.vtt"
        absolute = out_dir / relative
        absolute.parent.mkdir(exist_ok=True)
        absolute.write_text("WEBVTT\n")
        return [relative, absolute]

    spec = replace(clip, after=after)
    out = tmp_path / "out"
    manifest = Manifest(out / "manifest.json")
    requested = [Variant("doc", "light")]
    first = render_variants(spec, requested, out, manifest=manifest, workers=1)[0]
    assert first.outputs == ["marker-doc-light.mp4", "captions/doc-light.vtt"]
    entry = Manifest.load(manifest.path).entry(spec.slug, requested[0].name)
    assert [output["path"] for output in entry["outputs"]] == first.outputs
    second = render_variants(spec, requested, out, manifest=manifest, workers=1)[0]
    assert not second.rendered
    assert len(calls) == 2
    (out / first.outputs[1]).unlink()
    assert render_variants(spec, requested, out, manifest=manifest, workers=1)[
        0
    ].rendered
    assert builds(tmp_path / "builds") == {"doc-light": 2}


def test_prior_entries_saved_on_later_failure(clip, tmp_path, monkeypatch):
    spec = replace(
        clip,
        build=partial(
            build_variant_clip, markers=tmp_path / "builds", fail_mode="dark"
        ),
    )
    manifest = Manifest(tmp_path / "manifest.json")
    saves = []
    save = manifest.save

    def counted_save():
        saves.append(True)
        return save()

    monkeypatch.setattr(manifest, "save", counted_save)
    with pytest.raises(RuntimeError, match="deliberate variant failure"):
        render_variants(
            spec, variants(("doc",)), tmp_path, manifest=manifest, workers=1
        )
    assert len(saves) == 1
    loaded = Manifest.load(manifest.path)
    assert loaded.entry(spec.slug, "doc-light") is not None
    assert loaded.entry(spec.slug, "doc-dark") is None


@pytest.mark.parametrize("separate_work", [False, True])
def test_mux_failure_preserves_target_and_cleans_scratch(
    clip, tmp_path, monkeypatch, separate_work
):
    from eyepiece import narration

    audio = write_audio(tmp_path / "track.wav")
    spec = replace(clip, audio=lambda variant: audio)
    target = tmp_path / "marker-doc-light-narrated.mp4"
    target.write_bytes(b"existing video")
    work = tmp_path / "work" if separate_work else tmp_path
    options = {"work_dir": work} if separate_work else {}

    def fail_mux(video, audio, path, **kwargs):
        assert Path(path).parent.parent == work
        Path(path).write_bytes(b"partial mux")
        raise OSError("deliberate mux failure")

    monkeypatch.setattr(narration, "mux", fail_mux)
    manifest = Manifest(tmp_path / "manifest.json")
    with pytest.raises(OSError, match="deliberate mux failure"):
        render_variants(
            spec,
            [Variant("doc", "light", True)],
            tmp_path,
            manifest=manifest,
            workers=1,
            **options,
        )
    assert target.read_bytes() == b"existing video"
    assert list(tmp_path.rglob("*.mp4")) == [target]
    assert manifest.entry(spec.slug, "doc-light-narrated") is None
    assert list(work.glob("*.narration-*")) == []


def test_all_keys_precede_builds_and_ignore_destination(clip, tmp_path, monkeypatch):
    original = variant_driver.content_key
    keys = []

    def keyed(**kwargs):
        assert builds(tmp_path / "builds") == {}
        keys.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(variant_driver, "content_key", keyed)
    requested = variants(("doc",))
    first = render_variants(
        clip,
        requested,
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )
    assert len(keys) == len(requested)
    monkeypatch.setattr(variant_driver, "content_key", original)
    out = tmp_path / "elsewhere"
    relocated = replace(
        clip, build=partial(build_variant_clip, markers=tmp_path / "other-builds")
    )
    second = render_variants(
        relocated, requested, out, manifest=Manifest(out / "manifest.json"), workers=1
    )
    assert [result.key for result in first] == [result.key for result in second]


def test_spawned_builder_venue_rc_and_frame_source(clip, tmp_path, monkeypatch):
    # Keep the clip tiny while exercising two fresh subprocesses.
    monkeypatch.setattr(
        variant_driver,
        "render_frames",
        partial(variant_driver.render_frames, chunk_frames=3),
    )
    spec = replace(
        clip,
        frames=tuple(range(6)),
        venues=(
            Venue("doc", {"axes.facecolor": "red", "figure.facecolor": "red"}, 100, 12),
        ),
        hold_s=(0.25, 1 / 6),
    )
    requested = [Variant("doc", "light")]
    out = tmp_path / "out"
    manifest = Manifest(out / "manifest.json")
    first = render_variants(spec, requested, out, manifest=manifest, workers=2)[0]
    pids = (tmp_path / "builds" / "doc-light").read_text().splitlines()
    assert len(set(pids)) == 2
    assert str(os.getpid()) not in pids
    assert first.frames == len(spec.frames) + ep.frame_count(
        spec.hold_s[0], 12
    ) + ep.frame_count(spec.hold_s[1], 12)
    pixels = decode(out / first.outputs[0])
    assert pixels.shape[0] == first.frames
    assert pixels[..., 0].mean() > pixels[..., 1].mean() + 100
    assert (out / ".chunks-marker-doc-light" / ".eyepiece-render").exists()
    mapped = replace(spec, frames=6, frame_at=variant_frame_at, clock="half-steps")
    mapped_result = render_variants(
        mapped, requested, out, manifest=manifest, workers=1
    )[0]
    mapped_bytes = (out / mapped_result.outputs[0]).read_bytes()
    explicit = replace(
        mapped, frames=[variant_frame_at(i) for i in range(6)], frame_at=None
    )
    explicit_result = render_variants(
        explicit, requested, out, manifest=manifest, workers=1
    )[0]
    assert (out / explicit_result.outputs[0]).read_bytes() == mapped_bytes


@pytest.mark.parametrize("failure", ["after", "record", "build"])
def test_failed_replacement_discards_old_entry(clip, tmp_path, monkeypatch, failure):
    manifest = Manifest(tmp_path / "manifest.json")
    requested = variants(("doc",))
    render_variants(clip, requested, tmp_path, manifest=manifest, workers=1)
    kept = manifest.entry(clip.slug, "doc-light")
    spec = replace(clip, extra=("changed",))
    if failure == "build":
        spec = replace(
            spec,
            build=partial(
                build_variant_clip, markers=tmp_path / "builds", fail_mode="dark"
            ),
        )
    elif failure == "after":

        def after(variant, out_dir):
            raise RuntimeError("deliberate after failure")

        spec = replace(spec, after=after)
    else:

        def record(*args, **kwargs):
            raise RuntimeError("deliberate record failure")

        monkeypatch.setattr(manifest, "record", record)
    with pytest.raises(RuntimeError, match="deliberate"):
        render_variants(spec, [requested[1]], tmp_path, manifest=manifest, workers=1)
    loaded = Manifest.load(manifest.path)
    assert loaded.entry(clip.slug, "doc-dark") is None
    assert loaded.entry(clip.slug, "doc-light") == kept
    count = builds(tmp_path / "builds")["doc-dark"]
    # Returning to the old key must build even if the previous output survived.
    result = render_variants(
        clip, [requested[1]], tmp_path, manifest=loaded, workers=1
    )[0]
    assert result.rendered
    assert builds(tmp_path / "builds")["doc-dark"] == count + 1


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate",
        "mode",
        "empty-mode",
        "dpi",
        "fps",
        "infinite",
        "rc",
        "rc-value",
        "frames",
    ],
)
def test_invalid_later_settings_prevent_every_build(clip, tmp_path, problem):
    requested = [Variant("doc", "light"), Variant("talk", "dark")]
    venue = clip.venues[1]
    spec = clip
    if problem == "duplicate":
        spec = replace(clip, venues=(*clip.venues, clip.venues[0]))
    elif problem in ("mode", "empty-mode"):
        requested[1] = Variant("talk", "missing" if problem == "mode" else "")
    elif problem == "frames":
        spec = replace(clip, frames=[])
    else:
        values = {
            "dpi": {"dpi": 0},
            "fps": {"fps": -1},
            "infinite": {"dpi": float("inf")},
            "rc": {"rc": [("font.size", 10)]},
            "rc-value": {"rc": {"font.size": "invalid"}},
        }
        spec = replace(clip, venues=(clip.venues[0], venue._replace(**values[problem])))
    with pytest.raises(ValueError):
        render_variants(
            spec,
            requested,
            tmp_path,
            manifest=Manifest(tmp_path / "manifest.json"),
            workers=1,
        )
    assert builds(tmp_path / "builds") == {}


def test_reusing_manifest_with_new_destination_renders_there(clip, tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    requested = [Variant("doc", "light")]
    first = render_variants(clip, requested, tmp_path, manifest=manifest, workers=1)[0]
    out = tmp_path / "new"
    second = render_variants(clip, requested, out, manifest=manifest, workers=1)[0]
    assert second.rendered and second.key == first.key
    assert (out / second.outputs[0]).is_file()
    assert builds(tmp_path / "builds") == {"doc-light": 2}
    assert not render_variants(clip, requested, out, manifest=manifest, workers=1)[
        0
    ].rendered


def test_in_process_variants_restore_style_before_return(clip, tmp_path):
    from eyepiece import _style

    hwostyle.use("dark", palette_family="biosignature")
    ep.use_video(True)
    matplotlib.rcParams["font.size"] = 17
    saved_rc = dict(matplotlib.rcParams)
    names = (
        "_current_mode",
        "_current_family",
        "_activated",
        "palette",
        "cmaps",
        "roles",
    )
    saved_style = {name: getattr(hwostyle.core, name) for name in names}
    render_variants(
        clip,
        [Variant("doc", "light")],
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )
    assert dict(matplotlib.rcParams) == saved_rc
    assert {name: getattr(hwostyle.core, name) for name in names} == saved_style
    assert _style._VIDEO is True


@pytest.mark.parametrize("explicit_setup", [False, True])
def test_worker_and_parent_style_match_despite_parent_toggles(
    clip, tmp_path, monkeypatch, explicit_setup
):
    from eyepiece import _style

    original = variant_driver.render_frames
    monkeypatch.setattr(
        variant_driver, "render_frames", partial(original, chunk_frames=3)
    )
    hwostyle.use("paper")
    ep.use_video(not explicit_setup)
    saved_rc = dict(matplotlib.rcParams)
    before_mode = hwostyle.current_mode()
    spec = replace(clip, build=build_style_variant)
    if explicit_setup:
        build = partial(build_style_variant, spec.venues[0], "dark")
        for workers in (1, 2):
            original(
                build,
                6,
                tmp_path / f"{workers}.mp4",
                fps=12,
                dpi=100,
                setup=setup_video_style,
                workers=workers,
                chunk_frames=3,
            )
        paths = [tmp_path / f"{workers}.mp4" for workers in (1, 2)]
    else:
        paths = []
        for workers in (1, 2):
            out = tmp_path / str(workers)
            result = render_variants(
                spec,
                [Variant("doc", "dark")],
                out,
                manifest=Manifest(out / "manifest.json"),
                workers=workers,
            )[0]
            paths.append(out / result.outputs[0])
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert dict(matplotlib.rcParams) == saved_rc
    assert hwostyle.current_mode() == before_mode
    assert _style._VIDEO is (not explicit_setup)


def test_modes_use_active_backgrounds(clip, tmp_path):
    spec = replace(clip, build=build_style_variant)
    result = render_variants(
        spec,
        variants(("doc",)),
        tmp_path,
        manifest=Manifest(tmp_path / "manifest.json"),
        workers=1,
    )
    colors = []
    for item in result:
        with matplotlib.rc_context():
            hwostyle.use(item.variant.mode)
            colors.append(
                matplotlib.colors.to_rgb(matplotlib.rcParams["figure.facecolor"])
            )
        pixels = decode(tmp_path / item.outputs[0])
        np.testing.assert_allclose(pixels[0, 2, 2], np.array(colors[-1]) * 255, atol=3)
    assert colors[0] != colors[1]


def test_narrated_render_preserves_files_with_former_scratch_names(clip, tmp_path):
    audio = write_audio(tmp_path / "track.wav")
    spec = replace(clip, audio=lambda variant: audio)
    requested = [Variant("doc", "light", True)]
    manifest = Manifest(tmp_path / "manifest.json")
    first = render_variants(spec, requested, tmp_path, manifest=manifest, workers=1)[0]
    target = tmp_path / first.outputs[0]
    before = target.read_bytes()
    for suffix in ("silent", "mux"):
        (tmp_path / f".{target.stem}.{suffix}.mp4").write_bytes(b"stale partial video")
    count = builds(tmp_path / "builds")["doc-light"]
    assert render_variants(
        spec, requested, tmp_path, manifest=manifest, workers=1, force=True
    )[0].rendered
    assert builds(tmp_path / "builds")["doc-light"] == count + 1
    assert target.read_bytes() == before
    for suffix in ("silent", "mux"):
        assert (
            tmp_path / f".{target.stem}.{suffix}.mp4"
        ).read_bytes() == b"stale partial video"


@pytest.mark.parametrize("separate_work", [False, True])
def test_variants_resume_their_own_chunks(clip, tmp_path, monkeypatch, separate_work):
    try:
        resolve_ffmpeg()
    except RuntimeError:
        pytest.skip("needs ffmpeg")
    work = tmp_path / "work" if separate_work else tmp_path
    options = {"work_dir": work} if separate_work else {}
    original = variant_driver.render_frames
    resumed = []

    def recording(*args, **kwargs):
        result = original(*args, **kwargs, chunk_frames=3)
        resumed.append(result.resumed)
        return result

    monkeypatch.setattr(variant_driver, "render_frames", recording)
    spec = replace(
        clip,
        build=partial(
            build_variant_clip,
            markers=tmp_path / "builds",
            failure_root=tmp_path / "failures",
        ),
    )
    manifest = Manifest(tmp_path / "manifest.json")
    requested = variants(("doc",))
    chunks = []
    for variant in requested:
        with pytest.raises(RuntimeError, match="deliberate chunk failure"):
            render_variants(
                spec, [variant], tmp_path, manifest=manifest, workers=1, **options
            )
        chunk = work / f".chunks-marker-{variant.name}" / "chunk-0.bin"
        chunks.append(chunk)
        assert chunk.is_file()
    assert chunks[0].read_bytes() != chunks[1].read_bytes()
    second_before = chunks[1].read_bytes()
    first = render_variants(
        spec, [requested[0]], tmp_path, manifest=manifest, workers=1, **options
    )[0]
    assert chunks[1].read_bytes() == second_before
    second = render_variants(
        spec, [requested[1]], tmp_path, manifest=manifest, workers=1, **options
    )[0]
    assert resumed == [1, 1]
    before = [(tmp_path / item.outputs[0]).read_bytes() for item in (first, second)]
    render_variants(
        spec, requested, tmp_path, manifest=manifest, workers=1, force=True, **options
    )
    assert resumed == [1, 1, 0, 0]
    assert [
        (tmp_path / item.outputs[0]).read_bytes() for item in (first, second)
    ] == before


def test_stamp_changes_require_explicit_key_input(clip, tmp_path, monkeypatch):
    manifest = Manifest(tmp_path / "manifest.json")
    requested = [Variant("doc", "light")]
    first = render_variants(clip, requested, tmp_path, manifest=manifest, workers=1)[0]
    monkeypatch.setattr(ep, "stamp", lambda *args, **kwargs: None)
    second = render_variants(clip, requested, tmp_path, manifest=manifest, workers=1)[0]
    assert not second.rendered and second.key == first.key
    explicit = render_variants(
        replace(clip, extra=("updated stamp text",)),
        requested,
        tmp_path,
        manifest=manifest,
        workers=1,
    )[0]
    assert explicit.rendered and explicit.key != first.key


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("separate_work", [False, True])
def test_target_paths_and_adjacent_outputs(
    clip, tmp_path, custom, separate_work, monkeypatch
):
    try:
        resolve_ffmpeg()
    except RuntimeError as err:
        pytest.skip(str(err))
    out = tmp_path / "finals"
    work = tmp_path / "work"
    options = {"work_dir": str(work)} if separate_work else {}
    mux = ep.narration.mux

    def inspect_mux(video, audio, path, **kwargs):
        if separate_work:
            assert video == out / target(Variant("doc", "light"))
            assert path.parent.parent == (work / "doc" if custom else work)
            assert list(out.rglob("*.narration-*")) == []
        return mux(video, audio, path, **kwargs)

    monkeypatch.setattr(ep.narration, "mux", inspect_mux)
    temporary_directory = tempfile.TemporaryDirectory

    def inspect_scratch(*args, **kwargs):
        scratch = temporary_directory(*args, **kwargs)
        if separate_work:
            assert Path(scratch.name).is_relative_to(work)
        return scratch

    monkeypatch.setattr(tempfile, "TemporaryDirectory", inspect_scratch)

    def target_for(variant):
        return Path(variant.venue) / f"movie-{variant.name}.mp4"

    def target(variant):
        return target_for(variant) if custom else Path(f"marker-{variant.name}.mp4")

    def after(variant, out_dir):
        caption = out_dir / target(variant).with_suffix(".vtt")
        caption.write_text("WEBVTT\n")
        return [caption]

    audio = write_audio(tmp_path / "audio.wav")
    spec = replace(
        clip,
        target_for=target_for if custom else None,
        after=after,
        poster=0,
        audio=lambda variant: audio,
    )
    manifest = Manifest(out / "manifest.json")
    requested = [
        Variant("doc", "light"),
        Variant("talk", "dark"),
        Variant("doc", "light", True),
    ]
    first = render_variants(
        spec, requested, out, manifest=manifest, workers=1, **options
    )
    expected_files = {manifest.path}
    for result in first:
        video = target(result.variant)
        expected = [video.as_posix()]
        if not result.variant.narrated:
            expected.append(video.with_name(f"{video.stem}-poster.png").as_posix())
        expected.append(video.with_suffix(".vtt").as_posix())
        assert result.outputs == expected
        assert all((out / path).is_file() for path in result.outputs)
        expected_files.update(out / path for path in result.outputs)
        if separate_work:
            chunk_marker = (
                work / video.parent / f".chunks-{video.stem}" / ".eyepiece-render"
            )
            assert chunk_marker.is_file() is (not result.variant.narrated)
    if separate_work:
        assert {path for path in out.rglob("*") if path.is_file()} == expected_files
        assert {path for path in out.rglob("*") if path.is_dir()} == (
            {out / "doc", out / "talk"} if custom else set()
        )
        assert list(work.rglob("*.narration-*")) == []
    second = render_variants(
        spec, requested, out, manifest=manifest, workers=1, **options
    )
    assert not any(result.rendered for result in second)


@pytest.mark.parametrize("narrated", [False, True])
def test_adoption_keeps_all_intermediates_in_work_dir(
    clip, tmp_path, monkeypatch, narrated
):
    try:
        resolve_ffmpeg()
    except RuntimeError as err:
        pytest.skip(str(err))
    out, work = tmp_path / "finals", tmp_path / "work"
    audio = write_audio(tmp_path / "track.wav")
    spec = replace(clip, audio=lambda variant: audio)
    variant = Variant("doc", "light", narrated)
    manifest = Manifest(out / "manifest.json")
    first = render_variants(
        spec, [variant], out, manifest=manifest, workers=1, work_dir=work
    )[0]
    target = out / first.outputs[0]
    before = target.read_bytes(), target.stat().st_mtime_ns
    original = variant_driver.render_frames

    def inspect_render(build, frames, path, **kwargs):
        assert path.is_relative_to(work)
        assert kwargs["work_dir"].parent == work
        return original(build, frames, path, **kwargs)

    monkeypatch.setattr(variant_driver, "render_frames", inspect_render)
    current = replace(spec, extra=("new",), adopt_build=spec.build)
    adopted = render_variants(
        current,
        [variant],
        out,
        manifest=manifest,
        workers=1,
        work_dir=work,
        adopt=True,
    )[0]
    assert adopted.adopted and not adopted.rendered
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert set(out.iterdir()) == {target, manifest.path}
    assert {path.name for path in work.iterdir()} == {f".chunks-{target.stem}"}


@pytest.mark.parametrize("copy_fails", [False, True])
def test_narrated_work_dir_can_be_on_another_filesystem(
    clip, tmp_path, monkeypatch, copy_fails
):
    try:
        resolve_ffmpeg()
    except RuntimeError as err:
        pytest.skip(str(err))
    out, work = tmp_path / "finals", tmp_path / "work"
    audio = write_audio(tmp_path / "track.wav")
    spec = replace(clip, audio=lambda variant: audio)
    requested = [Variant("doc", "light", True)]
    manifest = Manifest(out / "manifest.json")
    render_variants(spec, requested, out, manifest=manifest, workers=1, work_dir=work)
    target = out / "marker-doc-light-narrated.mp4"
    before = target.read_bytes()
    original = os.replace

    def cross_device_replace(source, destination):
        if Path(source).is_relative_to(work) and Path(destination) == target:
            raise OSError(errno.EXDEV, "cross-device rename")
        return original(source, destination)

    monkeypatch.setattr(os, "replace", cross_device_replace)
    if copy_fails:

        def fail_copy(source, destination):
            Path(destination).write_bytes(b"partial copy")
            raise OSError("deliberate copy failure")

        monkeypatch.setattr(variant_driver.shutil, "copyfile", fail_copy)
        with pytest.raises(OSError, match="deliberate copy failure"):
            render_variants(
                spec,
                requested,
                out,
                manifest=manifest,
                workers=1,
                work_dir=work,
                force=True,
            )
    else:
        result = render_variants(
            spec,
            requested,
            out,
            manifest=manifest,
            workers=1,
            work_dir=work,
            force=True,
        )
        assert result[0].rendered
    assert target.read_bytes() == before
    assert set(out.iterdir()) == {target, manifest.path}
    assert list(work.rglob("*.narration-*")) == []


def test_same_stem_targets_resume_separately_in_work_dir(clip, tmp_path, monkeypatch):
    try:
        resolve_ffmpeg()
    except RuntimeError as err:
        pytest.skip(str(err))
    out, work = tmp_path / "finals", tmp_path / "work"
    spec = replace(
        clip,
        build=partial(
            build_variant_clip,
            markers=tmp_path / "builds",
            failure_root=tmp_path / "failures",
        ),
        target_for=lambda variant: Path(variant.mode) / "movie.mp4",
    )
    original = variant_driver.render_frames
    resumed = []

    def recording(*args, **kwargs):
        result = original(*args, **kwargs, chunk_frames=3)
        resumed.append(result.resumed)
        return result

    monkeypatch.setattr(variant_driver, "render_frames", recording)
    manifest = Manifest(out / "manifest.json")
    requested = variants(("doc",))
    for variant in requested:
        with pytest.raises(RuntimeError, match="deliberate chunk failure"):
            render_variants(
                spec, [variant], out, manifest=manifest, workers=1, work_dir=work
            )
    assert all(
        (work / variant.mode / ".chunks-movie" / "chunk-0.bin").is_file()
        for variant in requested
    )
    render_variants(spec, requested, out, manifest=manifest, workers=1, work_dir=work)
    assert resumed == [1, 1]


@pytest.mark.parametrize("target", ["../escape.mp4", "/outside.mp4", "wrong.png"])
def test_invalid_target_rejected_before_build(clip, tmp_path, target):
    spec = replace(clip, target_for=lambda variant: target)
    with pytest.raises(ValueError, match="target"):
        render_variants(
            spec,
            [Variant("doc", "light")],
            tmp_path,
            manifest=Manifest(tmp_path / "manifest.json"),
            workers=1,
        )
    assert not (tmp_path / "builds").exists()


@pytest.mark.parametrize("suffix", ["silent", "mux"])
def test_custom_targets_survive_narration_scratch_collision(clip, tmp_path, suffix):
    audio = write_audio(tmp_path / "track.wav")
    silent_target = Path(f".clip.{suffix}.mp4")
    saved = {}

    def after(variant, out_dir):
        if not variant.narrated:
            saved["silent"] = (out_dir / silent_target).read_bytes()
        return []

    spec = replace(
        clip,
        audio=lambda variant: audio,
        target_for=lambda variant: "clip.mp4" if variant.narrated else silent_target,
        after=after,
    )
    requested = [Variant("doc", "light"), Variant("doc", "dark", True)]
    manifest = Manifest(tmp_path / "manifest.json")
    results = render_variants(spec, requested, tmp_path, manifest=manifest, workers=1)
    assert all(result.rendered for result in results)
    assert [result.outputs for result in results] == [
        [silent_target.as_posix()],
        ["clip.mp4"],
    ]
    assert (tmp_path / silent_target).read_bytes() == saved["silent"]
    assert decode(tmp_path / "clip.mp4").shape[0] == spec.frames
    loaded = Manifest.load(manifest.path)
    assert all(
        loaded.current(spec.slug, result.variant.name, result.key) for result in results
    )
    kept = render_variants(spec, requested, tmp_path, manifest=loaded, workers=1)
    assert not any(result.rendered for result in kept)


def test_duplicate_targets_rejected_before_build(clip, tmp_path):
    spec = replace(clip, target_for=lambda variant: "same.mp4")
    with pytest.raises(ValueError, match="target"):
        render_variants(
            spec,
            [Variant("doc", "light"), Variant("talk", "dark")],
            tmp_path,
            manifest=Manifest(tmp_path / "manifest.json"),
            workers=1,
        )
    assert not (tmp_path / "builds").exists()
