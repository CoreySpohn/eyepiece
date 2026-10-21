---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Render a clip in parallel

A clip can render in parallel when `draw(fig, frame)` is a pure function of
the frame item. Each draw overwrites every artist property it uses. Keep
randomness seeded and avoid reading the wall clock.

This marker sets its position directly from the frame index. Check it with
`assert_frames_pure` before starting a pool. The check samples frames after
different drawing histories; it is a diagnostic, not a proof.

```{code-cell} python
import tempfile
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt

import eyepiece as ep

n, fps, dpi = 16, 16, 80


def draw(fig, frame, *, shift=0.0):
    fig.axes[0].lines[0].set_data([frame / (n - 1)], [0.5 + shift])


def build():
    fig, ax = plt.subplots(figsize=(2.4, 1.35))
    ax.plot([], [], "o", markersize=5)
    ax.set(xlim=(-0.1, 1.1), ylim=(0, 1), xticks=[], yticks=[])
    return fig, draw


ep.assert_frames_pure(build, n, dpi=dpi)
print("No sampled frame faults")
```

## Render it

Workers need a builder that is importable by name. A function defined in a
notebook cell or `__main__` cannot be rebuilt by a worker. Here the cell
writes a small module into a temporary directory, adds it to `sys.path`,
and imports it. Workers inherit `sys.path`.

The module also provides builders for the variants and stills below. Its
figure is 192 by 108 pixels at the chosen dpi. These examples need ffmpeg.

```{code-cell} python
import importlib
import textwrap

temporary = tempfile.TemporaryDirectory()
output = Path(temporary.name)
module_path = output / "parallel_clip_example.py"
module_path.write_text(
    textwrap.dedent('''\
    from functools import partial
    from pathlib import Path

    import matplotlib.pyplot as plt

    N = 16


    def draw(fig, frame, *, shift=0.0):
        fig.axes[0].lines[0].set_data([frame / (N - 1)], [0.5 + shift])


    def build():
        fig, ax = plt.subplots(figsize=(2.4, 1.35))
        ax.plot([], [], "o", markersize=5)
        ax.set(xlim=(-0.1, 1.1), ylim=(0, 1), xticks=[], yticks=[])
        return fig, draw


    def build_variant(venue, mode, *, data_path):
        shift = float(Path(data_path).read_text()) if mode == "dark" else 0.0
        fig, _ = build()
        return fig, partial(draw, shift=shift)


    def build_still():
        fig, draw = build()
        draw(fig, N // 2)
        return fig


    def timeline_frame(index, *, count):
        return index * (N - 1) / (count - 1)


    def build_legacy(venue, mode):
        return build()


    def build_current(venue, mode):
        fig, draw = build()
        fig.set_facecolor("0.85")
        return fig, draw
    '''),
    encoding="utf-8",
)
sys.path.insert(0, str(output))
importlib.invalidate_caches()
clip = importlib.import_module("parallel_clip_example")
ep.assert_frames_pure(clip.build, n, dpi=dpi)

path = output / "parallel.mp4"
result = ep.render_frames(clip.build, n, path, fps=fps, dpi=dpi, workers=2)
print(result._asdict())
assert isinstance(result, ep.RenderResult)

serial = ep.render_frames(
    clip.build, n, output / "serial.mp4", fps=fps, dpi=dpi, workers=1
)
assert serial.path.read_bytes() == path.read_bytes()
```

`RenderResult` reports `path`, `frames` (including holds), `chunks`,
`resumed` (reused chunks), `workers`, and elapsed `seconds`. `workers=1`
renders in the same process with the same code and gives the same bytes.
Pass style changes explicitly through `setup=` and `rc=` so every worker
receives them.

## Resume after a crash

Pass a `content_key` as `key=` and rerun with the same inputs, settings,
and `work_dir` after a crash. Complete chunks with matching identities and
checksums are reused. A failed render leaves the destination video untouched.
Change the key whenever an input changes.

Chunk files live in `work_dir`, which must be a dedicated empty folder or a
folder already owned by the driver. By default it is `.<stem>.chunks`
beside the video. This example keeps chunks after a successful render so
the next call can demonstrate reuse without interrupting a worker.

```{code-cell} python
settings = {
    "frames": n,
    "fps": fps,
    "dpi": dpi,
    "hold": (0, 0),
    "rc": {},
    "encoder": None,
    "extra_ffmpeg_args": None,
}
key = ep.content_key(modules=(clip,), settings=settings)
work_dir = output / "resume-chunks"
resume_args = dict(
    fps=fps, dpi=dpi, workers=1, chunk_frames=8, work_dir=work_dir, key=key
)
ep.render_frames(
    clip.build, n, output / "resumable.mp4", keep_chunks=True, **resume_args
)
recovered = ep.render_frames(
    clip.build, n, output / "resumable.mp4", **resume_args
)
assert recovered.resumed == recovered.chunks
print(f"Reused {recovered.resumed} complete chunks")
```

With `resume=False` or `key=None`, chunks are never reused and are deleted
as they are consumed. With a key and resume enabled, chunks remain until
the video succeeds, then are deleted unless `keep_chunks=True`.

## Name the content inputs

`content_key` hashes the script, clock, module sources, data, settings, and
package and ffmpeg versions. Include fps, dpi, encoder options, rc, holds,
and frame items or count in `settings`. Include the source and inputs of a
`frame_at` callable too, when using one.

A `Path` contributes file bytes; a string contributes its text. Packages
include all their Python files. Arrays contribute dtype, shape, and bytes.
Dictionary key order does not matter. The output path does not enter the
key. Stamps belong to the caller; pass stamp text in `extra` if it should
invalidate an output.

```{code-cell} python
key_inputs = dict(
    script=module_path,
    clock={"count": n, "fps": fps},
    modules=(clip,),
    data=([0.0, 0.5, 1.0],),
    settings=settings,
    extra=("Example stamp",),
)
key = ep.content_key(**key_inputs)
assert key == ep.content_key(**key_inputs)
changed_key = ep.content_key(**{**key_inputs, "data": ([0.0, 0.6, 1.0],)})
assert changed_key != key
print(key)
```

## Render variants and keep a manifest

`Venue` describes a destination's rc, dpi, fps, and optional silent frame
budget. `variants()` lists silent venue and mode pairs first, followed by
the requested narrated pairs. `ClipSpec.build(venue, mode)` supplies each
figure; `render_variants` applies the mode and venue rc before building it.

This builder reads a data file only in dark mode. The `data` callable
returns just the inputs each variant reads, so editing the file invalidates
only dark mode. List the builder module in `modules` to key its source too.

```{code-cell} python
from dataclasses import replace

data_path = output / "height.txt"
data_path.write_text("0.0", encoding="utf-8")


def variant_data(variant):
    return (data_path,) if variant.mode == "dark" else ()


venue = ep.Venue("doc", rc={"font.size": 7}, dpi=dpi, fps=fps, frame_budget=n)
selection = ep.variants((venue.name,), modes=("light", "dark"))
spec = ep.ClipSpec(
    slug="marker",
    build=partial(clip.build_variant, data_path=data_path),
    frames=n,
    venues=(venue,),
    modules=(clip,),
    data=variant_data,
)
variant_dir = output / "variants"
manifest_path = variant_dir / "manifest.json"
manifest = ep.Manifest.load(manifest_path)
first = ep.render_variants(spec, selection, variant_dir, manifest=manifest, workers=1)
assert all(item.rendered for item in first)

manifest = ep.Manifest.load(manifest_path)
second = ep.render_variants(spec, selection, variant_dir, manifest=manifest, workers=1)
assert not any(item.rendered for item in second)
print("Second call:", [(item.variant.name, item.rendered) for item in second])

data_path.write_text("0.2", encoding="utf-8")
changed = ep.render_variants(spec, selection, variant_dir, manifest=manifest, workers=1)
assert [item.variant.name for item in changed if item.rendered] == ["doc-dark"]
print("Changed data:", [(item.variant.name, item.rendered) for item in changed])
```

The manifest records keys, relative output paths, hashes, sizes, frame
counts, and render settings. Current entries skip rendering. The driver
saves it automatically; unchanged JSON keeps its modification time.
A `Manifest` file has a single writer. Serialize calls that write the same
file, even when their clips or output directories differ.

### Keep data scales fixed

`ClipSpec.on_rescale` defaults to `"warn"`: the driver reports changes in
axis limits or color normalization relative to frame zero. Exporters that
require fixed scales should use `"raise"`, which raises `ScaleDriftError`
and preserves the destination video on failure. Use `"ignore"` when changing
the scale is intentional. Pin limits over all frames before rendering when
the scale should stay fixed. This builder already pins its axis limits.

```{code-cell} python
assert spec.on_rescale == "warn"
strict_spec = replace(spec, slug="fixed-scale", frames=2, on_rescale="raise")
strict = ep.render_variants(
    strict_spec, selection[:1], variant_dir, manifest=manifest, workers=1
)
assert strict[0].rendered and strict[0].frames == 2
```

## Builders for a larger project

A project with many clips needs one module-level adapter rather than a
closure for each clip. A worker cannot unpickle a closure. Define
`build_variant(venue, mode, *, module, slug)` in an importable module. It
imports the named module, finds the clip definition by slug, and builds the
figure. Bind `module` and `slug` with `functools.partial` so `ClipSpec.build`
stays picklable. The partial's arguments must also be picklable.

This temporary module holds two clip definitions and the shared adapter.
The parent computes each frame list before submitting it. Frame items must
be picklable too; these lists contain ordinary numbers.

```{code-cell} python
project_module_path = output / "project_clips_example.py"
project_module_path.write_text(
    textwrap.dedent('''\
    import importlib
    from functools import partial
    from typing import NamedTuple

    import matplotlib.pyplot as plt


    def draw(fig, frame, *, height):
        fig.axes[0].lines[0].set_data([frame], [height])


    class Clip(NamedTuple):
        height: float
        positions: tuple

        def build(self, venue, mode):
            fig, ax = plt.subplots(figsize=(1.2, 0.7))
            ax.plot([], [], "o", markersize=3)
            ax.set(xlim=(-0.1, 1.1), ylim=(0, 1), xticks=[], yticks=[])
            return fig, partial(draw, height=self.height)


    CLIPS = {
        "low": Clip(0.25, (0.0, 0.5, 1.0)),
        "high": Clip(0.75, (1.0, 0.5, 0.0)),
    }


    def build_variant(venue, mode, *, module, slug):
        source = importlib.import_module(module)
        definition = source.CLIPS[slug]
        return definition.build(venue, mode)
    '''),
    encoding="utf-8",
)
importlib.invalidate_caches()
project_clips = importlib.import_module("project_clips_example")
project_dir = output / "project-clips"
project_manifest = ep.Manifest.load(project_dir / "manifest.json")

for slug, definition in project_clips.CLIPS.items():
    project_spec = ep.ClipSpec(
        slug=slug,
        build=partial(
            project_clips.build_variant, module=project_clips.__name__, slug=slug
        ),
        frames=list(definition.positions),
        venues=(venue,),
        modules=(project_clips,),
        on_rescale="raise",
    )
    rendered = ep.render_variants(
        project_spec, selection[:1], project_dir,
        manifest=project_manifest, workers=2,
    )
    assert rendered[0].rendered and rendered[0].frames == 3
    print(slug, rendered[0].outputs)
```

## Give narration its own timeline

Narrated variants require `audio=`, a callable from `Variant` to a WAV path.
The audio bytes enter their keys. Narrated variants are exempt from the
venue frame budget because speech sets the length of a narrated clip.

Use `frames_for(variant)` to supply a sequence or integer count for each
variant, overriding `frames`. Use `frame_at_for(variant)` to supply its
index-to-frame callable, overriding `frame_at`; returning `None` uses the
integer indices directly. `render_variants` calls both callbacks in the
parent before keying or building. The callbacks can close over parent
state, but the frame items and the callables returned by `frame_at_for`
must be picklable because they travel to workers. With a frame mapping,
`frames_for` must return an integer count. Set `frames=None` when
`frames_for` supplies every requested timeline.

Here a one-second sine wave stands in for speech. The narrated variant has
16 frames, while the silent variant has four. Both cover the same marker
positions. Include the mapping's source in `modules` and its configuration
in `clock` or other named key inputs, since a callable's identity alone
does not describe its behavior.

```{code-cell} python
import math
import struct
import wave

audio_path = output / "tone.wav"
rate = 16000
samples = b"".join(
    struct.pack("<h", round(3000 * math.sin(2 * math.pi * 440 * i / rate)))
    for i in range(rate)
)
with wave.open(str(audio_path), "wb") as track:
    track.setnchannels(1)
    track.setsampwidth(2)
    track.setframerate(rate)
    track.writeframes(samples)


def audio_for(variant):
    return audio_path


timeline_lengths = {"silent": 4, "narrated": n}


def frames_for(variant):
    return timeline_lengths["narrated" if variant.narrated else "silent"]


def frame_at_for(variant):
    return partial(clip.timeline_frame, count=frames_for(variant))


narrated_selection = ep.variants(
    (venue.name,), modes=("light",), narrated=(venue.name,)
)
narrated_spec = replace(
    spec,
    slug="timelines",
    frames=None,
    frames_for=frames_for,
    frame_at_for=frame_at_for,
    clock=timeline_lengths,
    audio=audio_for,
    venues=(venue._replace(frame_budget=4),),
)
narrated = ep.render_variants(
    narrated_spec, narrated_selection, variant_dir, manifest=manifest, workers=2
)
assert [item.frames for item in narrated] == [4, n]
assert all(item.rendered for item in narrated)
print([(item.variant.name, item.frames) for item in narrated])
```

## Keep a video made before content keys

For a deliberate migration, pass `adopt=True` to `render_variants` and set
`ClipSpec.adopt_build` to a picklable builder that reproduces the older
figure, including its stamp. It takes the same `(venue, mode)` arguments as
`build`. The driver renders that builder to a scratch video and compares
bytes with the existing target. An exact match keeps the target's bytes
and modification time and records it under the current key. A mismatch
renders the current `build` normally. `force=True` bypasses adoption.

This example records an older video without a manifest, then adopts it
using two module-level builders. `build_current` changes the figure
background; `build_legacy` reproduces the earlier background. Use the same
frame source, style, rc, and encoding settings as the older recording.
Adoption pays for a comparison render once; subsequent current entries
skip both builders. Requested posters use the current builder, and
`after` runs for adopted, rendered, and kept variants, so make it idempotent.

```{code-cell} python
import hwostyle

adopt_dir = output / "adopted"
adopt_dir.mkdir()
legacy_video = adopt_dir / "legacy-marker-doc-light.mp4"
ep.render_frames(
    partial(clip.build_legacy, venue, "light"), 3, legacy_video,
    fps=venue.fps, dpi=venue.dpi, workers=1,
    setup=partial(hwostyle.use, "light"), rc=venue.rc,
)
before = (legacy_video.read_bytes(), legacy_video.stat().st_mtime_ns)
adopt_manifest = ep.Manifest.load(adopt_dir / "manifest.json")
adopt_spec = ep.ClipSpec(
    slug="legacy-marker",
    build=clip.build_current,
    adopt_build=clip.build_legacy,
    frames=3,
    venues=(venue,),
    modules=(clip,),
)
adopted = ep.render_variants(
    adopt_spec, selection[:1], adopt_dir,
    manifest=adopt_manifest, workers=1, adopt=True,
)[0]
assert adopted.adopted and not adopted.rendered
assert (legacy_video.read_bytes(), legacy_video.stat().st_mtime_ns) == before
assert adopt_manifest.current(adopt_spec.slug, adopted.variant.name, adopted.key)
kept = ep.render_variants(
    adopt_spec, selection[:1], adopt_dir, manifest=adopt_manifest, workers=1
)[0]
assert not kept.rendered and not kept.adopted
print("Adopted:", adopted.outputs)
```

## Save posters and stills

`render_poster` freezes layout after frame zero, then draws the requested
frame index. Its PNG matches the worker raster at that index. Use the same
`dpi`, `setup`, and `rc` as the video.

`render_still` takes a builder returning a figure and saves it through
`save_fig` as PNG, PDF, or SVG. `same_picture` compares PNG channel levels
with a tolerance and can ignore a bottom band, such as a provenance stamp.

```{code-cell} python
poster = ep.render_poster(clip.build, n, n // 2, output / "poster.png", dpi=dpi)
poster_copy = ep.render_poster(
    clip.build, n, n // 2, output / "poster-copy.png", dpi=dpi
)
assert poster.read_bytes() == poster_copy.read_bytes()
assert ep.same_picture(poster, poster_copy, tol=2, ignore_bottom_px=8)

for suffix in ("png", "pdf", "svg"):
    still = ep.render_still(clip.build_still, output / f"still.{suffix}", dpi=dpi)
    before = still.read_bytes()
    ep.render_still(clip.build_still, still, dpi=dpi)
    assert still.read_bytes() == before
    print(still.name)

sys.path.remove(str(output))
sys.modules.pop(clip.__name__, None)
sys.modules.pop(project_clips.__name__, None)
temporary.cleanup()
```

The examples remove their temporary files after all cells finish.

The frame checks also live in the checking namespace:
`eyepiece.testing.frame_faults`, `eyepiece.testing.assert_frames_pure`, and
`eyepiece.testing.same_picture` are the same functions as the `eyepiece.render`
names. Neither path warns. These aliases load on access, so importing
`eyepiece.testing` alone does not load the render workers.
