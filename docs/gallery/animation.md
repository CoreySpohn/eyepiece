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

# Animation

An animation is a figure built once and mutated many times. `record` and
`animate` are the two entry points for that, and they differ only in who
owns the loop: `record` is a context manager that a caller drives from their
own iteration, and `animate` binds a draw function to a frame source and
renders when asked. Both open every requested output file together and grab
each rendered frame into all of them, so writing a gif and an embedded
player costs one pass rather than two.

The frames below come from a synthetic wavefront-control loop built in the
page: a bright core, a halo made of a handful of low-order modes that
weaken as the loop converges, and a companion that emerges from the halo as
it clears.

```{code-cell} python
import tempfile
from pathlib import Path

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece as ep

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small. A real figure script
# keeps the style library's 300 dpi print policy and omits this line.
plt.rcParams["savefig.dpi"] = 120

N = 96
PIXSCALE_LOD = 0.3
EXTENT = ep.extent_lod_from_pixels(N, PIXSCALE_LOD)
N_FRAMES = 20

u = (np.arange(N) - (N - 1) / 2.0) * PIXSCALE_LOD
x, y = np.meshgrid(u, u)
r = np.hypot(x, y)

rng = np.random.default_rng(3)
FREQ = rng.uniform(0.2, 1.1, (12, 2))
PHASE = rng.uniform(0.0, 2.0 * np.pi, 12)
RATE = rng.uniform(0.1, 0.35, 12)
CORE = np.exp(-0.5 * (r / 0.4) ** 2)
HALO = 2e-6 * (1.0 + r) ** -2.5


def loop_image(k):
    """The focal plane at iteration `k` of the control loop."""
    done = k / (N_FRAMES - 1)
    modes = sum(
        np.cos(FREQ[m, 0] * x + FREQ[m, 1] * y + PHASE[m] + RATE[m] * k)
        for m in range(len(FREQ))
    )
    speckle = HALO * (0.05 + 6.0 * (1.0 - 0.9 * done) * (modes / len(FREQ)) ** 2)
    companion = 3e-7 * done * np.exp(-0.5 * (np.hypot(x - 3.2, y + 1.8) / 0.4) ** 2)
    return CORE + speckle + companion
```

## Streaming with `record`

`record` is the form to reach for when the frames come from a loop that
already exists, a simulation stepping forward or a control loop iterating,
because it never asks for a frame index and never holds onto frame data. The
figure is built once outside the `with` block, and each pass through the
loop mutates the artists and calls `frame()`, which grabs the figure as it
stands into every open sink. `hold(n)` repeats the current state without
redrawing, which is how a reveal is given time to land at the end.

Both files below are written from the same pass. The suffix picks the
writer, so a `.gif` goes through Pillow and a `.html` becomes a
self-contained player with its frames embedded as base64 PNGs, and neither
needs anything installed beyond matplotlib.

```{code-cell} python
fig, ax = plt.subplots(figsize=(3.8, 2.8), layout="constrained")
result = ep.imshow_log(loop_image(0), ax=ax, extent=EXTENT, floor=1e-11, vmax=1.0)
ep.label_lod(ax)

with tempfile.TemporaryDirectory() as tmp_dir:
    gif = Path(tmp_dir) / "loop.gif"
    player = Path(tmp_dir) / "loop.html"

    with ep.record(fig, gif, player, fps=10, dpi=100) as rec:
        for k in range(N_FRAMES):
            result.update(loop_image(k))
            rec.frame()
        rec.hold(6)

    embedded = player.read_text().count("data:image/png;base64")
    print(f"gif    {gif.stat().st_size / 1024:7.0f} kB")
    print(f"player {player.stat().st_size / 1024:7.0f} kB, {embedded} frames")
```

Nothing in that loop clears anything. `result.update` re-applies the log
floor the first draw established and calls `set_data` on the existing image,
so the artist count is fixed for the whole recording and the norm stays the
one the first frame built. Redraw-style animation is still available, and it
is nothing more than calling `ax.clear()` inside the loop, but update mode
is faster and it keeps the axis limits and the color scale from drifting
between frames.

## The draw-function form

`animate` is the same recording with the loop supplied for you. It binds a
figure, a `draw(fig, ctx)` callable, and a frame source, and renders nothing
until one of `.save`, `.jshtml`, or `.video` is called. The frame source is
an integer here, which becomes `range(20)`, so `ctx` is the frame index; any
re-iterable works, and a one-shot generator is detected and refuses a second
render rather than silently writing an empty file.

`.jshtml` returns a self-contained player as a string, which is what an
executed documentation page wants: the frames travel with the page, so
nothing has to be installed on the machine that builds it and nothing has to
be served alongside it. The dpi is passed to the writer explicitly, because
the style library pins `savefig.dpi` for print figures and savefig's dpi
wins over the figure's, which makes setting `fig.dpi` a non-fix.

```{code-cell} python
fig, ax = plt.subplots(figsize=(3.8, 2.8), layout="constrained")
result = ep.imshow_log(loop_image(0), ax=ax, extent=EXTENT, floor=1e-11, vmax=1.0)
ep.label_lod(ax)
title = ax.set_title("iteration 0")


def draw(fig, k):
    result.update(loop_image(k))
    title.set_text(f"iteration {k}")


anim = ep.animate(fig, draw, N_FRAMES, fps=10)
print("frames:", anim.n_frames)
HTML(anim.jshtml(dpi=100))
```

An embedded frame costs tens of kilobytes, so a page holding several
animations grows quickly, and roughly twenty to thirty frames at dpi 100 on
a small figure is the range worth staying inside. Frame count is the first
thing to cut, figure size the second.

## Writing a video

`.save` takes as many paths as there are destinations and renders one pass
into all of them, and `.video` writes one file and hands back something a
notebook can play. Both are the right call for a talk or a repository asset
rather than for a page that has to carry its own frames, and the block below
is shown rather than executed for that reason.

```python
anim.save("loop.mp4", "loop.gif", dpi=120)
video = anim.video("loop.mp4", dpi=120)

with ep.record(fig, "loop.mp4", **ep.PRESETS["talk"]) as rec:
    for k in range(N_FRAMES):
        result.update(loop_image(k))
        rec.frame()
```

An mp4 sink needs an ffmpeg binary, and a documentation builder usually has
none. The facade resolves one from `rcParams["animation.ffmpeg_path"]`,
then from `PATH`, then from the `imageio-ffmpeg` package, checking each
candidate before handing it to a subprocess, and raises an error naming both
fixes when every candidate fails. Installing `imageio-ffmpeg` is the fix
that needs no system package manager, which is why it sits in this library's
`docs` extra.

Two mechanics of mp4 output are worth knowing because they are handled for
you. The h264 encoder rejects odd pixel dimensions, which a figure size in
inches times a dpi hits often, so every mp4 sink is given a crop filter that
truncates both dimensions to even numbers. A `-vf` passed through
`extra_ffmpeg_args` replaces that filter rather than adding to it, since
ffmpeg honors only the last one, so a caller who needs their own filter
chain has to include the crop in it.

## Presets

`PRESETS` holds measured frame rate and dpi pairs for the usual
destinations. An entry splats straight into `record` and into `.jshtml`,
both of which take an `fps` and a `dpi`. An animation otherwise carries the
frame rate it was bound with, so `.save` and `.video` take a dpi alone and a
preset reaches them one value at a time.

```{code-cell} python
for name, settings in ep.PRESETS.items():
    print(f"{name:>7}: {settings}")
```

A sink given no explicit dpi takes the default for its own suffix, so a
`.gif` is rasterized at the `gif` preset dpi, a `.html` at the `jshtml` one,
and an `.mp4` at the `talk` one, which is usually what mixed output wants.
Passing `dpi` to `record` or to `.save` overrides that for every sink at
once, which is what the recording above did to put the gif and the player on
the same rasterization.
