"""Data-scale checks apply equally to fresh and resumed render chunks."""

import warnings
from dataclasses import replace
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Normalize
from test_render_helpers import build_rescale_clip, build_rescale_variant

from eyepiece.anim import ScaleDriftError
from eyepiece.render import (
    ClipSpec,
    Manifest,
    Variant,
    Venue,
    render_frames,
    render_variants,
)

DRIFT_MESSAGE = "data scales changed during recording"


def build_aspect_clip():
    fig = plt.figure(figsize=(0.96, 0.54), dpi=73)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.plot([0, 1], [0, 1])
    ax.set_aspect("equal", adjustable="datalim")
    return fig, draw_aspect_clip


def draw_aspect_clip(fig, frame):
    fig.axes[0].set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])


def build_norm_clip(*, kind, fixed=False, drift_at=1):
    fig = plt.figure(figsize=(0.96, 0.54), dpi=73)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    if kind == "image":
        ax.imshow([[0, 1], [0, 1]], aspect="auto")
    else:
        ax.scatter([0, 1], [0, 1], c=[0, 1])
    ax.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), xticks=[], yticks=[])
    return fig, partial(draw_norm_clip, kind=kind, fixed=fixed, drift_at=drift_at)


def draw_norm_clip(fig, frame, *, kind, fixed, drift_at):
    ax = fig.axes[0]
    # Isolate the drift so parallel failures cannot race across chunk ranges.
    upper = 2 if frame == drift_at else 1
    if kind == "image":
        artist = ax.images[0]
        artist.set_data([[0, upper], [0, upper]])
    else:
        artist = ax.collections[0]
        artist.set_array(np.array([0, upper]))
    artist.set_norm(Normalize(0, 2) if fixed else Normalize())


def build_cached_clip(*, sentinel):
    fig, draw = build_rescale_clip()
    fig._rescale_draws = 0
    return fig, partial(draw_cached_clip, draw=draw, sentinel=sentinel)


def draw_cached_clip(fig, frame, *, draw, sentinel):
    # Workers still initialize frame zero when every chunk is cached.
    if sentinel.exists() and fig._rescale_draws:
        raise AssertionError("cached frames were redrawn")
    fig._rescale_draws += 1
    draw(fig, frame)


def drift_warnings(caught):
    return [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and DRIFT_MESSAGE in str(warning.message)
    ]


def render_clip(tmp_path, *, build=build_rescale_clip, **kwargs):
    return render_frames(
        build, 8, tmp_path / "clip.mp4", fps=12, dpi=100, chunk_frames=2, **kwargs
    )


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("existing", [False, True])
def test_raise_preserves_destination(tmp_path, workers, existing):
    path = tmp_path / "clip.mp4"
    if existing:
        path.write_bytes(b"previous output")
    with pytest.raises(
        ScaleDriftError, match=r"(?s)chunk frames \[\d+, \d+\).*frame \d+"
    ):
        render_clip(tmp_path, workers=workers, on_rescale="raise")
    assert path.read_bytes() == b"previous output" if existing else not path.exists()
    assert not list(tmp_path.rglob("*.partial.mp4"))
    assert not (tmp_path / ".clip.chunks" / "chunk-0.bin").exists()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("mode", ["warn", "ignore"])
def test_warn_once_or_ignore(tmp_path, workers, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = render_clip(tmp_path, workers=workers, on_rescale=mode)
    scale_warnings = drift_warnings(caught)
    assert len(scale_warnings) == (1 if mode == "warn" else 0)
    if scale_warnings:
        message = str(scale_warnings[0].message)
        assert DRIFT_MESSAGE in message
        assert "frame 1" in message
        assert 'pass on_rescale="ignore"' in message
    assert result.path.is_file()
    assert not list(tmp_path.rglob("chunk-*.warn"))


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("mode", ["warn", "raise", "ignore"])
def test_fixed_limits_are_silent(tmp_path, workers, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render_clip(
            tmp_path,
            build=partial(build_rescale_clip, fixed=True),
            workers=workers,
            on_rescale=mode,
        )
    assert not drift_warnings(caught)


@pytest.mark.parametrize("workers", [1, 2])
def test_aspect_adjusted_fixed_limits_are_silent(tmp_path, workers):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = render_clip(
            tmp_path, build=build_aspect_clip, workers=workers, on_rescale="raise"
        )
    assert not drift_warnings(caught)
    assert result.workers == workers
    assert result.chunks > 1


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("kind", ["image", "collection"])
@pytest.mark.parametrize("drift_at", [1, 7])
def test_deferred_norm_drift_is_detected(tmp_path, workers, kind, drift_at):
    build = partial(build_norm_clip, kind=kind, drift_at=drift_at)
    with pytest.raises(
        ScaleDriftError, match=rf"(?s){DRIFT_MESSAGE}.*color norm.*frame {drift_at}"
    ):
        render_clip(tmp_path, build=build, workers=workers, on_rescale="raise")
    assert not (tmp_path / "clip.mp4").exists()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("kind", ["image", "collection"])
def test_fixed_norm_is_silent(tmp_path, workers, kind):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = render_clip(
            tmp_path,
            build=partial(build_norm_clip, kind=kind, fixed=True),
            workers=workers,
            on_rescale="raise",
        )
    assert not drift_warnings(caught)
    assert result.workers == workers
    assert result.chunks > 1


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_late_chunk_uses_frame_zero_baseline(tmp_path, workers):
    with pytest.raises(ScaleDriftError, match=r"(?s)chunk frames \[6, 8\).*frame 6"):
        render_clip(
            tmp_path,
            build=partial(build_rescale_clip, drift_at=6),
            workers=workers,
            on_rescale="raise",
        )
    assert not (tmp_path / "clip.mp4").exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_resume_replays_warning_and_raises_without_rebuilding_chunks(tmp_path, workers):
    sentinel = tmp_path / "cached"
    options = dict(
        workers=workers,
        key="same-pixels",
        keep_chunks=True,
        build=partial(build_cached_clip, sentinel=sentinel),
    )
    with pytest.warns(RuntimeWarning, match=DRIFT_MESSAGE) as caught:
        first = render_clip(tmp_path, on_rescale="warn", **options)
    assert len(drift_warnings(caught)) == 1
    assert first.workers == workers
    assert first.chunks > 1
    sentinel.touch()
    original = first.path.read_bytes()
    with pytest.warns(RuntimeWarning, match=DRIFT_MESSAGE) as caught:
        resumed = render_clip(tmp_path, on_rescale="warn", **options)
    assert len(drift_warnings(caught)) == 1
    assert resumed.resumed == first.chunks
    first.path.unlink()
    with pytest.raises(ScaleDriftError, match=r"chunk frames \[0, 2\).*frame 1"):
        render_clip(tmp_path, on_rescale="raise", **options)
    assert not first.path.exists()
    first.path.write_bytes(original)
    with pytest.raises(ScaleDriftError):
        render_clip(tmp_path, on_rescale="raise", **options)
    assert first.path.read_bytes() == original


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "truncated",
        "invalid_utf8",
        "corrupted",
        "mismatched",
        "mismatched_crc",
    ],
)
def test_invalid_sidecar_rebuilds_chunk(tmp_path, workers, damage):
    options = dict(
        workers=workers,
        key="same-pixels",
        keep_chunks=True,
        build=partial(build_rescale_clip, drift_at=7),
    )
    with pytest.warns(RuntimeWarning, match=DRIFT_MESSAGE):
        first = render_clip(tmp_path, on_rescale="warn", **options)
    assert first.workers == workers
    assert first.chunks == 4
    folder = tmp_path / ".clip.chunks"
    sidecar = folder / "chunk-3.warn"
    if damage == "missing":
        sidecar.unlink()
    elif damage == "truncated":
        sidecar.write_bytes(b"")
    elif damage == "invalid_utf8":
        sidecar.write_bytes(b"\xff")
    elif damage == "corrupted":
        sidecar.write_bytes(
            sidecar.read_bytes().replace(b"data scales", b"xxxx scales")
        )
    elif damage == "mismatched":
        sidecar.write_bytes((folder / "chunk-0.warn").read_bytes())
    else:
        # Same render identity and chunk range, but different raster bytes.
        donor = tmp_path / "donor"
        render_clip(
            donor,
            on_rescale="raise",
            **dict(options, build=partial(build_rescale_clip, fixed=True)),
        )
        sidecar.write_bytes((donor / ".clip.chunks" / "chunk-3.warn").read_bytes())
    original = first.path.read_bytes()
    with pytest.raises(ScaleDriftError, match=rf"(?s){DRIFT_MESSAGE}.*frame 7"):
        render_clip(tmp_path, on_rescale="raise", **options)
    assert first.path.read_bytes() == original
    with pytest.warns(RuntimeWarning, match=DRIFT_MESSAGE):
        rebuilt = render_clip(tmp_path, on_rescale="warn", **options)
    assert rebuilt.resumed == first.chunks - 1


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_checked_no_drift_chunks_require_sidecars(tmp_path, workers, damage):
    options = dict(
        workers=workers,
        key="fixed-pixels",
        keep_chunks=True,
        build=partial(build_rescale_clip, fixed=True),
        on_rescale="raise",
    )
    first = render_clip(tmp_path, **options)
    folder = tmp_path / ".clip.chunks"
    assert len(list(folder.glob("chunk-*.warn"))) == first.chunks
    assert render_clip(tmp_path, **options).resumed == first.chunks
    sidecar = folder / "chunk-0.warn"
    if damage == "missing":
        sidecar.unlink()
    else:
        sidecar.write_bytes(b"")
    assert render_clip(tmp_path, **options).resumed == first.chunks - 1


def test_ignore_cache_cannot_hide_drift(tmp_path):
    options = dict(workers=1, key="same-pixels", keep_chunks=True)
    render_clip(tmp_path, on_rescale="ignore", **options)
    assert not list(tmp_path.rglob("chunk-*.warn"))
    with pytest.raises(ScaleDriftError, match="frame 1"):
        render_clip(tmp_path, on_rescale="raise", **options)


def test_rescale_mode_is_not_a_content_key_input(tmp_path):
    build = partial(build_rescale_clip, fixed=True)
    options = dict(workers=1, key="same-pixels", keep_chunks=True, build=build)
    first = render_clip(tmp_path, on_rescale="raise", **options)
    second = render_clip(tmp_path, on_rescale="warn", **options)
    assert second.resumed == first.chunks
    spec = ClipSpec("scale", build_rescale_variant, 2, (Venue("doc"),))
    manifest = Manifest(tmp_path / "manifest.json")
    variant = Variant("doc", "light")
    first = render_variants(
        replace(spec, build=partial(build_rescale_variant, fixed=True)),
        [variant],
        tmp_path,
        manifest=manifest,
        workers=1,
    )[0]
    second = render_variants(
        replace(
            spec, build=partial(build_rescale_variant, fixed=True), on_rescale="raise"
        ),
        [variant],
        tmp_path,
        manifest=manifest,
        workers=1,
    )[0]
    assert second.key == first.key
    assert not second.rendered


@pytest.mark.parametrize("mode", ["invalid", None, []])
def test_invalid_mode_preflight(tmp_path, mode):
    marker = tmp_path / "built"
    with pytest.raises(ValueError, match="on_rescale"):
        render_clip(
            tmp_path,
            build=partial(build_rescale_clip, observe=marker),
            workers=1,
            on_rescale=mode,
        )
    assert not marker.exists()
    assert not (tmp_path / ".clip.chunks").exists()


def test_variants_forward_rescale_and_discard_failed_entry(tmp_path, monkeypatch):
    manifest = Manifest(tmp_path / "manifest.json")
    variant = Variant("doc", "light")
    spec = ClipSpec("scale", build_rescale_variant, 16, (Venue("doc"),))
    rendered = []

    def observe_render(*args, **kwargs):
        result = render_frames(*args, **kwargs)
        rendered.append(result)
        return result

    monkeypatch.setattr("eyepiece._render_variants.render_frames", observe_render)
    assert spec.on_rescale == "warn"
    with pytest.warns(RuntimeWarning, match=DRIFT_MESSAGE):
        result = render_variants(
            spec, [variant], tmp_path, manifest=manifest, workers=2
        )[0]
    assert rendered[0].workers == 2
    assert rendered[0].chunks == 2
    path = tmp_path / result.outputs[0]
    path.unlink()
    with pytest.raises(ScaleDriftError):
        render_variants(
            replace(spec, on_rescale="raise"),
            [variant],
            tmp_path,
            manifest=manifest,
            workers=2,
        )
    assert not path.exists()
    assert manifest.entry(spec.slug, variant.name) is None
    assert Manifest.load(manifest.path).entry(spec.slug, variant.name) is None
    marker = tmp_path / "built"
    with pytest.raises(ValueError, match="on_rescale"):
        render_variants(
            replace(
                spec,
                build=partial(build_rescale_variant, observe=marker),
                on_rescale="invalid",
            ),
            [variant],
            tmp_path,
            manifest=manifest,
            workers=1,
        )
    assert not marker.exists()
