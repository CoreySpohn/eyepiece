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

hwostyle.use("dark")
plt.rcParams["savefig.dpi"] = 120

N = 128
PIXSCALE_LOD = 0.25

u = (np.arange(N) - (N - 1) / 2.0) * PIXSCALE_LOD
x, y = np.meshgrid(u, u)
r = np.hypot(x, y)

rng = np.random.default_rng(19)
core = np.exp(-0.5 * (r / 0.35) ** 2)
halo = 3e-6 * (1.0 + r) ** -2.5
image = core + halo * rng.chisquare(2, size=r.shape)


def radial_profile(values, radius, nbins=48):
    """Mean of `values` in `nbins` equal-width annuli, as (centers, means)."""
    edges = np.linspace(0.0, radius.max(), nbins + 1)
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, nbins - 1)
    total = np.bincount(index, weights=values.ravel(), minlength=nbins)
    count = np.bincount(index, minlength=nbins)
    return 0.5 * (edges[:-1] + edges[1:]), total / count
```

## A radial profile

`plot_radial` draws precomputed separations against precomputed values and
sets no axis label of its own, since it cannot know whether the separations
are in lambda over D, arcseconds, astronomical units, or pixels. The labels
below are therefore passed explicitly, and the log scale is asked for
through `log=True` rather than set afterward.

The profile itself is the azimuthal mean of the image, computed above in
eight lines of NumPy, which is the division of labor the library keeps
everywhere: the caller owns the reduction, and the primitive owns the
drawing.

```{code-cell} python
sep, profile = radial_profile(image, r)

fig, ax = plt.subplots(figsize=(5.4, 3.2), layout="constrained")
prof = ep.plot_radial(
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
curve = ep.plot_contrast_curve(
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
first = ep.plot_contrast_curve(
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
second = ep.plot_contrast_curve(
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

## Profiling an image directly

`radial_profile_plot` is the one function in this module that computes
anything, and it does so by calling `hwoutils.radial.radial_profile` rather
than reimplementing the reduction. It lives behind the `eyepiece[hwo]`
extra, which supplies `hwoutils` and, with it, the unit conversions behind
`extent_arcsec`, `extent_au`, and `Frame.extent_arcsec`. Because that extra
is not part of the documentation environment, the call below is shown rather
than executed, and without `hwoutils` installed it raises an `ImportError`
naming the extra.

```python
result = ep.radial_profile_plot(image, PIXSCALE_LOD, log=True, label="azimuthal mean")
```

Unlike `plot_radial`, this function is told its units, since the pixel scale
is an argument, so it defaults the x label to a lambda over D separation
label. Everything else is handed straight to `plot_radial`, and the same
`PlotResult` comes back.
