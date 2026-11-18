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

# Profiles

A profile collapses an image, or a performance estimate, onto separation
from the star, and the two shapes it takes are a radial profile of an image
and a contrast curve. Neither primitive on this page computes anything: both
take arrays the caller already has, because the averaging scheme, the bin
edges, and the units belong to whatever produced the numbers. The arrays
below are built in the page with NumPy for exactly that reason.

Every figure is drawn in the dark style mode, activated once in the
preamble, because a documentation page bakes its images at build time and
cannot respond to the mode a reader picks later.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep
from optixstuff.grids import Frame, Image, Measure, PixelGrid
from optixstuff.radial import radial_profile

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small. A real figure script
# keeps the style library's 300 dpi print policy and omits this line.
plt.rcParams["savefig.dpi"] = 120

N = 128
PIXSCALE_LOD = 0.25

grid = PixelGrid.focal(N, PIXSCALE_LOD, Frame(500, 1, "circumscribed"))
r = np.asarray(grid.radius())

rng = np.random.default_rng(19)
core = np.exp(-0.5 * (r / 0.35) ** 2)
halo = 3e-6 * (1.0 + r) ** -2.5
intensity = core + halo * rng.chisquare(2, size=r.shape)
image = Image(intensity, grid, (), Measure.intensive(), "intensity")
```

## A radial profile

`plot_radial` draws precomputed separations against precomputed values and
sets no axis label of its own, since it cannot know whether the separations
are in lambda over D, arcseconds, astronomical units, or pixels. The labels
below are therefore passed explicitly, and the log scale is asked for
through `log=True` rather than set afterward.

The profile is the azimuthal mean of the intensity sampled at the pixel
centers. optixstuff assigns samples to rings on the image grid and eyepiece
draws the resulting curve.

```{code-cell} python
sep, profile, _ = radial_profile(image, nbins=48)

fig, ax = plt.subplots(figsize=(5.4, 3.2), layout="constrained")
prof = ep.plot.plot_radial(
    sep,
    profile,
    ax=ax,
    log=True,
    label="azimuthal mean",
    xlabel=r"$r$ [$\lambda/D$]",
    ylabel="mean intensity",
)
ax.legend()
```

## A contrast curve with working angles and floors

`plot_contrast_curve` adds the annotations that make a contrast curve
readable. Everything inside `iwa` and outside `owa` is shaded, because those
regions are not measurements, and the shading is a rectangle anchored to the
axes edge rather than to the limits of the moment, so it still reaches the
edge after a later curve widens the x range. Each entry in `floors` is a
reference limit drawn as its own dashed curve in the next palette color,
which is how a curve is read against the noise sources that set it.

```{code-cell} python
sep_curve = np.linspace(1.0, 30.0, 200)
raw = 3e-8 * (sep_curve / 5.0) ** -1.6 + 4e-11
photon = 6e-11 * np.ones_like(sep_curve)
detector = 2e-11 * (sep_curve / 5.0) ** -0.4

floors = [
    (sep_curve, photon, "photon noise"),
    (sep_curve, detector, "detector noise"),
]

fig, ax = plt.subplots(figsize=(5.8, 3.4), layout="constrained")
curve = ep.plot.plot_contrast_curve(
    sep_curve,
    raw,
    ax=ax,
    iwa=3.0,
    owa=25.0,
    floors=floors,
    label="raw contrast",
    xlabel=r"$r$ [$\lambda/D$]",
    ylabel=r"$5\sigma$ contrast",
)
ax.set_xlim(1.0, 30.0)
ax.legend(loc="upper right")
```

## Comparing two curves on one axes

Comparing curves is the normal way this function is used, and the comparison
is made by calling it twice with the same `ax`. The second call repeats
`iwa`, `owa`, and `floors` verbatim, and neither the shading nor the floor
curves are duplicated: each kind of annotation is tracked per axes and
redrawn only when it is genuinely missing, so the second call contributes
its curve and nothing else.

Colors come from the same per-axes counter, which is why the second curve is
a different color from the first without either call naming one. The floor
curves drawn by the first call took the two colors after it, so the
post-processed curve below is the fourth color of the palette rather than
the second.

```{code-cell} python
processed = raw / 9.0 + 5e-12

fig, ax = plt.subplots(figsize=(5.8, 3.4), layout="constrained")
first = ep.plot.plot_contrast_curve(
    sep_curve,
    raw,
    ax=ax,
    iwa=3.0,
    owa=25.0,
    floors=floors,
    label="raw contrast",
    xlabel=r"$r$ [$\lambda/D$]",
    ylabel=r"$5\sigma$ contrast",
)
second = ep.plot.plot_contrast_curve(
    sep_curve,
    processed,
    ax=ax,
    iwa=3.0,
    owa=25.0,
    floors=floors,
    label="after subtraction",
)
ax.set_xlim(1.0, 30.0)
ax.legend(loc="upper right")
```

The two results say which call drew what. The first carries the shading
rectangles, their labels, and the floor lines alongside its own curve, and
the second carries only its curve, so a caller reaching for an annotation
always finds it on the call that actually made it.

```{code-cell} python
print("first call:", sorted(first.artists))
print("second call:", sorted(second.artists))
print("floor curves:", len(first.artists["lines"]))
```

## Separate curve-end labels

Lay out direct labels once after choosing the limits. `min_gap` is the gap
between rendered label edges in points. Labels keep their vertical order
and stay within the vertical axes bounds. A set too tall to fit raises.

```python
import matplotlib.pyplot as plt
import eyepiece as ep

fig, ax = plt.subplots()
ax.set(xlim=(0, 1), ylim=(0, 1))
labels = []
for name, y in [("a", 0.50), ("b", 0.51), ("c", 0.52)]:
    (line,) = ax.plot([0, 0.8], [0.2, y])
    labels.append(ax.text(0.8, y, name, color=line.get_color(), va="center"))
ep.plot.spread_labels(labels, min_gap=3)
```
