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

# One preparation, three outputs

A single prepared sequence below feeds a still, a strip of selected epochs,
and a movie. The science happens once, in the preparation block; the three
outputs only choose which samples to show and how. The data are synthetic: a
speckle field that decorrelates as its phases drift, a faint companion, an
occulted core marked invalid, one dead pixel, and irregular timestamps. The
{doc}`prepared views guide <../prepared-views>` explains the records used
here.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece.mpl as mpl
from eyepiece.prepared import (
    ArrayChannel,
    AxisSpec,
    Clock,
    CurveView,
    ImageView,
    Label,
    PanelGroup,
    Path,
    PathWindow,
    Points,
    Region,
    Scale,
    Sequence,
    resolve_bounds,
)
from eyepiece.style import SourceCast, snapshot_profile

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small. A real figure script
# keeps the style library's 300 dpi print policy and omits this line.
plt.rcParams["savefig.dpi"] = 120

# One cast and one profile for every output on this page.
CAST = SourceCast(["annulus"])
PROFILE = snapshot_profile(
    text_size_pt=plt.rcParams["font.size"],
    stroke_width_pt=plt.rcParams["lines.linewidth"],
)
```

## Preparing the sequence

The preparation owns every scientific decision: the time samples, the
invalid pixels, the annulus, and the trace, which is the mean over the
annulus's valid pixels computed from the scientific values before any display
clipping. It also fixes the display scale once, from the valid samples of the
whole sequence, so every output shares it.

```{code-cell} python
rng = np.random.default_rng(0)
N_SAMPLES, N_PIX, PIXSCALE_LOD = 30, 48, 0.5
HALF = N_PIX * PIXSCALE_LOD / 2.0
centers = (np.arange(N_PIX) + 0.5) * PIXSCALE_LOD - HALF
x, y = np.meshgrid(centers, centers)
r = np.hypot(x, y)

# Irregular sampling: the gaps between exposures vary.
times_s = np.concatenate([[0.0], np.cumsum(rng.uniform(4.0, 16.0, N_SAMPLES - 1))])

freq = rng.uniform(-1.2, 1.2, (40, 2))
phase0 = rng.uniform(0.0, 2.0 * np.pi, 40)
drift = rng.normal(0.0, 0.02, 40)  # rad per second


def speckle_contrast(t_s):
    arg = (
        freq[:, 0, None, None] * x
        + freq[:, 1, None, None] * y
        + (phase0 + drift * t_s)[:, None, None]
    )
    field = np.exp(1j * arg).sum(axis=0) / np.sqrt(len(freq))
    halo = 1e-9 * np.abs(field) ** 2 * (1.0 + r) ** -1.5
    companion = 3e-10 * np.exp(-0.5 * (np.hypot(x - 4.0, y + 3.0) / 0.6) ** 2)
    return halo + companion


cube = np.stack([speckle_contrast(t) for t in times_s]).astype(np.float32)
cube[:, 8, 40] = np.nan  # a dead pixel
occulted = np.zeros(cube.shape, dtype=bool)
occulted[:, r < 2.5] = True
cube = np.ma.masked_array(cube, mask=occulted)

in_annulus = (r > 4.0) & (r < 6.0)
trace = np.array([np.nanmean(frame.data[in_annulus]) for frame in cube])
trace_xy = np.column_stack([times_s, trace])

_, vmax = resolve_bounds(
    ((frame.data, ~frame.mask) for frame in cube), kind="log"
)
scale = Scale("log", vmin=vmax * 1e-3, vmax=vmax, floor=vmax * 1e-3)

image = ImageView(
    "image",
    cube[0],
    AxisSpec("x (λ/D)", "y (λ/D)", (-HALF, HALF), (-HALF, HALF)),
    scale,
    quantity="contrast",
    marks=(
        Region("annulus", np.array([0.0, 0.0]), outer_radius=6.0, inner_radius=4.0),
        Label("clock", "", (0.04, 0.95)),
    ),
)
trace_view = CurveView(
    "trace",
    AxisSpec(
        "time (s)",
        "annulus mean contrast",
        (0.0, float(times_s[-1])),
        (0.0, 1.2 * float(trace.max())),
        aspect="auto",
    ),
    marks=(
        Path("history", trace_xy, source_id="annulus", label="annulus mean"),
        Points("head", trace_xy[:1], source_id="annulus"),
    ),
)
sequence = Sequence(
    PanelGroup("speckles", (image, trace_view)),
    times=times_s,
    time_unit="s",
    channels=(
        ArrayChannel("image", "data", cube),
        PathWindow("history"),
        ArrayChannel("head", "xy", trace_xy[:, None, :]),
        Clock("clock", fmt="t = {value:.0f} {unit}"),
    ),
)
print(f"{len(sequence.times)} samples over {times_s[-1]:.0f} s, scale {scale.vmin:.2g} to {scale.vmax:.2g}")
```

## A still

`sequence.frame(i)` is the complete state at one sample. Rendering it gives a
paper figure on the full-sequence scale: the occulted core and the dead pixel
read as invalid, not as faint, and the annulus is an outline so the pixels
under it keep their colors.

```{code-cell} python
fig, axes = plt.subplots(
    1, 2, figsize=(7.4, 3.0), width_ratios=[1.0, 1.25], layout="constrained"
)
still = mpl.render(sequence.frame(12), axes=axes, cast=CAST, profile=PROFILE)
```

## A strip of selected epochs

A comparison between two non-adjacent times is a strip rather than a movie.
`sequence.strip(indices)` returns one slot per index with its IDs prefixed by
slot position and a time label added to each slot's first panel. The clock
the sequence drives is then redundant, so the strip hides it through its
part.

```{code-cell} python
picks = [0, 15, 29]
fig, grid = plt.subplots(
    2, 3, figsize=(8.0, 4.6), height_ratios=[1.6, 1.0], layout="constrained"
)
strip = mpl.render(
    sequence.strip(picks), axes=grid.T.ravel(), cast=CAST, profile=PROFILE
)
for slot in range(len(picks)):
    strip.parts[f"{slot}/clock"].set_visible(False)
```

## A movie

`mpl.animate` renders frame 0, then plays the shared schedule: thirty output
frames here, evenly spaced in physical time from the first sample through
the last, each showing the sample acquired at or before its time. The image,
the trace head, and the clock always name the same sample, and the image is
mapped again only when the held sample changes. Because the timestamps are
irregular, a long gap between exposures holds one image for several output
frames rather than inventing states that were never observed.

```{code-cell} python
fig, axes = plt.subplots(
    1, 2, figsize=(6.4, 2.6), width_ratios=[1.0, 1.25], layout="constrained"
)
movie = mpl.animate(
    sequence, run_time=3.0, fps=10, axes=axes, cast=CAST, profile=PROFILE
)
print("frames:", movie.n_frames)
HTML(movie.jshtml(dpi=100))
```

The same sequence plays in a talk through `eyepiece.manim.animate`, with the
same cast and a talk profile; see {doc}`../manim`.
