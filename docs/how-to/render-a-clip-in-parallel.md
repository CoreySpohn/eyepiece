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

Narrated variants require `audio=`, a callable from `Variant` to a WAV path.
The audio bytes enter their keys. Narrated variants are exempt from the
venue's frame budget because speech determines their length. Here a
one-second sine wave stands in for speech, using one venue and mode.

```{code-cell} python
import math
import struct
import wave
from dataclasses import replace

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


narrated_selection = [
    variant
    for variant in ep.variants((venue.name,), modes=("light",), narrated=(venue.name,))
    if variant.narrated
]
narrated_spec = replace(
    spec, audio=audio_for, venues=(venue._replace(frame_budget=n - 1),)
)
narrated = ep.render_variants(
    narrated_spec, narrated_selection, variant_dir, manifest=manifest, workers=1
)
assert narrated[0].rendered
print(narrated[0].variant.name, narrated[0].outputs)
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
temporary.cleanup()
```

The examples remove their temporary files after all cells finish.
