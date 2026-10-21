"""Posters share worker pixels; stills and picture comparisons are deterministic."""

import os
import subprocess
import sys
from dataclasses import replace
from functools import partial

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from test_render_helpers import (
    build_poster_clip,
    build_still,
    build_variant_clip,
    poster_frame_at,
    setup_style,
)
from test_render_variants import write_audio

import eyepiece as ep
import eyepiece.render as render
from eyepiece import _render_chunks as chunk_protocol
from eyepiece import _render_pool as pool
from eyepiece._ffmpeg import resolve_ffmpeg


def read_pixels(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGBA"))


def test_public_still_exports():
    for name in ("render_poster", "render_still", "same_picture"):
        assert name in ep.__all__
        assert getattr(ep, name) is getattr(render, name)


def test_poster_matches_raster_and_decoded_video(tmp_path):
    """Match independent and spawned rasters; CRF 0 isolates yuv420p error."""
    frames = [0.0, 0.5, 1.0]
    rc = {"axes.labelpad": 6}
    path = tmp_path / "nested" / "poster.png"
    assert (
        ep.render_poster(
            build_poster_clip, frames, 2, path, dpi=100, setup=setup_style, rc=rc
        )
        == path
    )

    def reference(settle_index):
        with matplotlib.rc_context(matplotlib.rcParamsDefault):
            setup_style()
            with matplotlib.rc_context(rc):
                fig, draw = build_poster_clip()
                destination = tmp_path / f"reference-{settle_index}.png"
                try:
                    with ep.record(
                        fig,
                        tmp_path / "reference.mp4",
                        fps=12,
                        dpi=100,
                        stills={destination: 1 / 12},
                    ) as rec:
                        draw(fig, frames[settle_index])
                        rec.frame()
                        draw(fig, frames[2])
                        rec.frame()
                    return read_pixels(destination)
                finally:
                    plt.close(fig)

    expected = reference(0)
    np.testing.assert_array_equal(read_pixels(path), expected)
    assert expected.shape == (180, 320, 4)
    # Without the frame-zero freeze the wider label moves the axes.
    assert not np.array_equal(expected, reference(2))
    before = path.read_bytes()
    ep.render_poster(
        build_poster_clip, frames, 2, path, dpi=100, setup=setup_style, rc=rc
    )
    assert path.read_bytes() == before
    with Image.open(path) as image:
        assert image.info == {}
    video = tmp_path / "clip.mp4"
    # Lossless encoding isolates yuv420p conversion from lossy text-edge error.
    chunks = tmp_path / "chunks"
    result = ep.render_frames(
        build_poster_clip,
        frames,
        video,
        fps=12,
        dpi=100,
        workers=2,
        chunk_frames=1,
        work_dir=chunks,
        key="poster-reference",
        keep_chunks=True,
        setup=setup_style,
        rc=rc,
        encoder={"crf": 0},
    )
    assert result.workers == 2
    identity = chunk_protocol._identity(
        "poster-reference", 3, (0, 0), 100, (320, 180), 12, 2, 3
    )
    (worker_pixels,) = chunk_protocol._read_chunk(
        (chunks / "chunk-2.bin", identity, 2, 3), (320, 180)
    )
    np.testing.assert_array_equal(read_pixels(path), worker_pixels)
    decoded = subprocess.run(
        [
            resolve_ffmpeg(),
            "-v",
            "error",
            "-i",
            str(video),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(3, 180, 320, 3)
    difference = np.abs(pixels[2].astype(int) - expected[:, :, :3].astype(int))
    # This clip's default CRF 18 profile had a maximum channel error of 10.
    # CRF 0 keeps the <= 3 assertion focused on yuv420p conversion.
    assert difference.max() <= 3


def test_poster_frame_sources_and_fresh_interpreter(tmp_path):
    path = tmp_path / "poster.png"
    rc = {"axes.labelpad": 6}
    ep.render_poster(
        build_poster_clip,
        [0.0, 0.5, 1.0],
        1,
        path,
        dpi=100,
        setup=setup_style,
        rc=rc,
    )
    mapped = tmp_path / "mapped.png"
    ep.render_poster(
        build_poster_clip,
        3,
        1,
        mapped,
        dpi=100,
        frame_at=poster_frame_at,
        setup=setup_style,
        rc=rc,
    )
    assert mapped.read_bytes() == path.read_bytes()
    integer = tmp_path / "integer.png"
    ep.render_poster(build_poster_clip, 2, 1, integer, dpi=100)
    explicit = tmp_path / "explicit.png"
    ep.render_poster(build_poster_clip, [0, 1], 1, explicit, dpi=100)
    assert integer.read_bytes() == explicit.read_bytes()
    fresh = tmp_path / "fresh.png"
    code = """
import sys
import matplotlib
matplotlib.use("Agg")
import eyepiece as ep
from test_render_helpers import build_poster_clip, poster_frame_at, setup_style
ep.render_poster(build_poster_clip, 3, 1, sys.argv[1], dpi=100,
    frame_at=poster_frame_at, setup=setup_style, rc={'axes.labelpad': 6})
"""
    subprocess.run(
        [sys.executable, "-c", code, str(fresh)],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert fresh.read_bytes() == path.read_bytes()


@pytest.mark.parametrize("index", [-1, 3])
def test_poster_rejects_out_of_range_index_before_build(tmp_path, index):
    def unexpected_build():
        pytest.fail("invalid requests must not build a figure")

    with pytest.raises(IndexError):
        ep.render_poster(unexpected_build, 3, index, tmp_path / "poster.png", dpi=100)


def test_poster_rejects_suffix_before_build(tmp_path):
    with pytest.raises(ValueError, match="png"):
        ep.render_poster(None, 3, 1, tmp_path / "poster.pdf", dpi=100)


def test_poster_failed_write_preserves_target_and_removes_partial(
    tmp_path, monkeypatch
):
    path = tmp_path / "poster.png"
    path.write_bytes(b"previous poster")

    def failed_save(image, target, **kwargs):
        target.write_bytes(b"partial PNG")
        raise OSError("write failed")

    monkeypatch.setattr(Image.Image, "save", failed_save)
    with pytest.raises(OSError, match="write failed"):
        ep.render_poster(build_poster_clip, 3, 1, path, dpi=100)
    assert path.read_bytes() == b"previous poster"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kind", ["poster", "still", "frames"])
@pytest.mark.parametrize("failure", [None, "setup", "build", "save"])
def test_figure_and_parent_rc_restored(tmp_path, kind, failure):
    from eyepiece import _style

    hwostyle.use("dark")
    ep.use_video(True)
    parent = plt.figure()
    matplotlib.rcParams.update({"font.size": 19, "axes.facecolor": "red"})
    before = dict(matplotlib.rcParams)
    figures = plt.get_fignums()
    names = (
        "_current_mode",
        "_current_family",
        "_activated",
        "palette",
        "cmaps",
        "roles",
    )
    style = {name: getattr(hwostyle.core, name) for name in names}
    mode = hwostyle.current_mode()
    video = _style._VIDEO
    seen = []

    def setup():
        hwostyle.use("light")
        ep.use_video(False)
        matplotlib.rcParams["font.size"] = 9
        matplotlib.rcParams["axes.facecolor"] = "blue"
        if failure == "setup":
            plt.figure()
            raise RuntimeError("setup failed")

    def build():
        seen.append(
            (matplotlib.rcParams["font.size"], matplotlib.rcParams["axes.facecolor"])
        )
        assert hwostyle.current_mode() == "light"
        assert not _style._VIDEO
        if failure == "build":
            plt.figure()
            plt.figure()
            raise RuntimeError("build failed")
        if kind == "still":
            return build_still()
        fig, draw = build_poster_clip()

        def draw_frame(fig, frame):
            if failure == "save" and frame == 1:
                raise RuntimeError("draw failed")
            draw(fig, frame)

        return fig, draw_frame

    try:
        kwargs = dict(setup=setup, rc={"font.size": 11})
        if kind == "poster":

            def save():
                ep.render_poster(build, 2, 1, tmp_path / "out.png", dpi=100, **kwargs)
        elif kind == "still":
            kwargs["metadata"] = {"unsupported": "value"} if failure == "save" else None

            def save():
                ep.render_still(build, tmp_path / "out.svg", **kwargs)
        else:

            def save():
                ep.render_frames(
                    build,
                    2,
                    tmp_path / "out.mp4",
                    fps=12,
                    dpi=100,
                    workers=1,
                    **kwargs,
                )

        if failure:
            message = (
                "unsupported"
                if kind == "still" and failure == "save"
                else f"{'draw' if failure == 'save' else failure} failed"
            )
            with pytest.raises((RuntimeError, ValueError), match=message):
                save()
        else:
            save()
        assert seen == ([] if failure == "setup" else [(11, "blue")])
        assert dict(matplotlib.rcParams) == before
        assert hwostyle.current_mode() == mode
        for name, value in style.items():
            assert getattr(hwostyle.core, name) is value
        assert _style._VIDEO is video
        assert plt.get_fignums() == figures
        assert plt.gcf() is parent
    finally:
        for number in set(plt.get_fignums()) - set(figures):
            plt.close(number)
        plt.close(parent)


@pytest.mark.parametrize("suffix", ["png", "pdf", "svg"])
def test_still_deterministic_and_save_fig_policy(tmp_path, suffix):
    """Distinct files stay identical despite differing backend clock inputs."""
    path = tmp_path / "nested" / f"still.{suffix}"
    other = path.with_stem("other")
    metadata = {"Title": "A deterministic still"}
    rc = {"savefig.bbox": None, "font.size": 7}
    code = """
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eyepiece as ep
from test_render_helpers import build_still, setup_style
path = Path(sys.argv[1])
assert ep.render_still(build_still, path, dpi=100, setup=setup_style,
    rc={"savefig.bbox": None, "font.size": 7},
    metadata={"Title": "A deterministic still"}) == path
if path.suffix in (".pdf", ".svg"):
    fig = build_still()
    try:
        with matplotlib.rc_context({"svg.hashsalt": "timestamp-control"}):
            fig.savefig(path.with_stem(path.stem + "-timestamped"))
    finally:
        plt.close(fig)
"""
    for destination, epoch in ((path, "946684800"), (other, "978307200")):
        subprocess.run(
            [sys.executable, "-c", code, str(destination)],
            env=dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(sys.path),
                SOURCE_DATE_EPOCH=epoch,
            ),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    before = path.read_bytes()
    assert other.read_bytes() == before
    if suffix == "png":
        with Image.open(path) as image:
            assert image.info["dpi"] == pytest.approx((100, 100), abs=0.01)
            assert image.info["Title"] == metadata["Title"]
    else:
        assert (
            path.with_stem("still-timestamped").read_bytes()
            != other.with_stem("other-timestamped").read_bytes()
        )
        higher_dpi = path.with_stem("higher-dpi")
        ep.render_still(
            build_still,
            higher_dpi,
            dpi=200,
            setup=setup_style,
            rc=rc,
            metadata=metadata,
        )
        assert higher_dpi.read_bytes() == before


def test_same_picture_tolerance_and_bottom_band(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    pixels = np.full((5, 6, 4), 100, dtype=np.uint8)
    Image.fromarray(pixels).save(a)
    changed = pixels.copy()
    changed[1, 1, 0] += 2
    Image.fromarray(changed).save(b)
    assert ep.same_picture(a, b)
    assert ep.same_picture(a, b, tol=2)
    assert not ep.same_picture(a, b, tol=1)
    changed[1, 1, 0] += 1
    Image.fromarray(changed).save(b)
    assert not ep.same_picture(a, b, tol=2)
    changed[1, 1, 0] = 100
    changed[-2:, :, :3] = 255
    Image.fromarray(changed).save(b)
    assert ep.same_picture(a, b, tol=0, ignore_bottom_px=2)
    assert not ep.same_picture(a, b, tol=0, ignore_bottom_px=1)
    Image.fromarray(pixels[:4]).save(b)
    assert not ep.same_picture(a, b)


def test_same_picture_alpha_and_implicit_opaque_channel(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    pixels = np.full((5, 6, 4), 100, dtype=np.uint8)
    pixels[:, :, 3] = 255
    Image.fromarray(pixels).save(a)
    changed = pixels.copy()
    changed[:, :, 3] = 0
    Image.fromarray(changed).save(b)
    assert not ep.same_picture(a, b)
    changed[:, :, 3] = 253
    Image.fromarray(changed).save(b)
    assert ep.same_picture(a, b, tol=2)
    assert not ep.same_picture(a, b, tol=1)
    Image.fromarray(pixels[:, :, :3]).save(a)
    Image.fromarray(pixels).save(b)
    assert ep.same_picture(a, b, tol=0)
    assert ep.same_picture(b, a, tol=0)
    changed[:, :, 3] = 0
    Image.fromarray(changed).save(b)
    assert not ep.same_picture(a, b)
    assert not ep.same_picture(b, a)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": -1},
        {"ignore_bottom_px": -1},
        {"ignore_bottom_px": 5},
        {"ignore_bottom_px": 6},
    ],
)
def test_same_picture_rejects_invalid_comparison(tmp_path, kwargs):
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((5, 6, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError):
        ep.same_picture(path, path, **kwargs)


def test_variant_posters_manifest_skip_and_index_key(tmp_path):
    audio = write_audio(tmp_path / "audio.wav")
    spec = render.ClipSpec(
        slug="marker",
        build=partial(build_variant_clip, markers=tmp_path / "builds"),
        frames=3,
        venues=(render.Venue("doc", {"axes.facecolor": "0.7"}, 100, 12),),
        frame_at=poster_frame_at,
        poster=1,
        audio=lambda variant: audio,
    )
    out = tmp_path / "out"
    manifest = render.Manifest(out / "manifest.json")
    requested = render.variants(("doc",), narrated=("doc",))
    first = ep.render_variants(spec, requested, out, manifest=manifest, workers=1)
    for result in first:
        expected = [f"marker-{result.variant.name}.mp4"]
        if not result.variant.narrated:
            expected.append(f"marker-{result.variant.name}-poster.png")
            venue = spec.venues[0]
            with pool._figure(
                partial(spec.build, venue, result.variant.mode),
                spec.frames,
                venue.dpi,
                spec.frame_at,
                partial(hwostyle.use, result.variant.mode),
                venue.rc,
            ) as state:
                pixels = pool._raster(state, spec.poster)
            np.testing.assert_array_equal(read_pixels(out / expected[1]), pixels)
        assert result.outputs == expected
        assert [
            item["path"]
            for item in manifest.entry("marker", result.variant.name)["outputs"]
        ] == expected
    assert len(list(out.glob("*-poster.png"))) == 2
    second = ep.render_variants(spec, requested, out, manifest=manifest, workers=1)
    assert not any(result.rendered for result in second)
    (out / first[0].outputs[1]).unlink()
    third = ep.render_variants(spec, requested, out, manifest=manifest, workers=1)
    assert [result.rendered for result in third] == [True, False, False, False]
    changed = ep.render_variants(
        replace(spec, poster=2), requested, out, manifest=manifest, workers=1
    )
    assert all(new.key != old.key for new, old in zip(changed, first, strict=True))
    assert all(result.rendered for result in changed)


@pytest.mark.parametrize("poster", [-1, 3])
def test_variant_poster_validation_precedes_any_build(tmp_path, poster):
    spec = render.ClipSpec(
        slug="marker",
        build=partial(build_variant_clip, markers=tmp_path / "builds"),
        frames=3,
        venues=(render.Venue("doc"),),
        poster=poster,
    )
    with pytest.raises(ValueError, match="poster"):
        ep.render_variants(
            spec,
            render.variants(("doc",)),
            tmp_path,
            manifest=render.Manifest(tmp_path / "manifest.json"),
            workers=1,
        )
    assert not (tmp_path / "builds").exists()
