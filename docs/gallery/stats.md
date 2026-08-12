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

# Distributions

Sampled distributions get looked at in four ways: a triangle plot of every
pair of parameters, the same triangle with a second dataset laid over it, a
single marginal against the analytic form it should match, and a confidence
region drawn as an ellipse. The primitives on this page cover those four,
and the samples below are drawn from seeded generators inside the page, so
the figures are reproducible and nothing here reads a file.

Every figure is drawn in the dark style mode, activated once in the
preamble, because a documentation page bakes its images at build time and
cannot respond to the mode a reader picks later.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small. A real figure script
# keeps the style library's 300 dpi print policy and omits this line.
plt.rcParams["savefig.dpi"] = 120

PARAMS = ["period", "amplitude", "phase"]
LABELS = {
    "period": r"$P$ [d]",
    "amplitude": r"$K$ [m s$^{-1}$]",
    "phase": r"$\phi$ [rad]",
}
TRUTHS = {"period": 12.4, "amplitude": 3.1, "phase": 0.8}

MEAN = np.array([12.4, 3.1, 0.8])
COV = np.array(
    [
        [0.090, 0.045, -0.020],
        [0.045, 0.062, 0.008],
        [-0.020, 0.008, 0.035],
    ]
)


def draw(seed, n=4000, shift=0.0, cov=COV):
    """A correlated three-parameter sample set, as a dict of 1D arrays."""
    rng = np.random.default_rng(seed)
    values = rng.multivariate_normal(MEAN + shift, cov, size=n)
    return dict(zip(PARAMS, values.T, strict=True))


posterior = draw(0)
```

## A corner plot with truths

`corner` puts a one-dimensional histogram on each diagonal cell and a
two-dimensional density below it, hides the upper triangle, and labels only
the outer edge of the grid so the interior stays readable. The dashed guides
are the `truths` dict, drawn vertically on the diagonal and both vertically
and horizontally below it, which is how a recovered posterior is checked
against the values that generated it. A parameter absent from `truths` is
simply drawn without a guide.

Handing `corner` an existing grid is the ax-first path, and it is the reason
`title` is rejected in that case: a caller who built the figure may have put
other content in it, and a suptitle would land on top of that content. The
title here is therefore set on the figure the caller owns.

```{code-cell} python
n = len(PARAMS)
fig, axes = plt.subplots(n, n, figsize=(6.2, 5.6), layout="constrained")

tri = ep.corner(
    posterior,
    PARAMS,
    truths=TRUTHS,
    labels=LABELS,
    bins=28,
    axes=axes,
)
fig.suptitle("Recovered posterior against the input values")
```

## Two datasets on one triangle

`corner_overlay` draws several sample sets into the same grid, each in its
own palette color, as step histograms on the diagonal and scatter below it.
The two sets below differ by a small shift in every parameter and by a
wider covariance, which is the shape a comparison of two inference runs
usually takes. Density gives way to scatter here because two filled meshes
stacked on one cell hide each other, while two point clouds do not.

The legend needs somewhere to go, and the upper triangle of a corner plot is
empty by construction, so the top right cell is turned back on with its
ticks and spines removed and the dataset legend is parked there. Nothing
outside the grid is consumed.

```{code-cell} python
wide = COV * 2.4
overlay = ep.corner_overlay(
    [posterior, draw(1, shift=0.35, cov=wide)],
    PARAMS,
    labels=LABELS,
    names=["baseline", "inflated errors"],
    bins=28,
)
overlay.fig.set_size_inches(6.2, 5.6)
```

## A histogram against its analytic form

`hist_vs_pdf` normalizes the histogram and evaluates a callable over the
sample range, which makes it the fastest check that a generator produces
what its derivation says it should. The samples below are the modulus of a
circular complex Gaussian, whose distribution is Rayleigh with the same
scale as the real and imaginary parts, and the curve is that Rayleigh
density evaluated directly. Agreement across the whole range, rather than
only near the mode, is what the figure is for, so the y scale is
logarithmic.

Both the histogram and the curve take their labels through the arguments
that reach `ax.hist` and `ax.plot`, and the legend is drawn by the caller on
the axes the result hands back.

```{code-cell} python
rng = np.random.default_rng(5)
sigma = 1.4
amplitude = np.abs(rng.normal(0.0, sigma, 20000) + 1j * rng.normal(0.0, sigma, 20000))


def rayleigh_pdf(a):
    return a / sigma**2 * np.exp(-0.5 * (a / sigma) ** 2)


fig, ax = plt.subplots(figsize=(5.4, 3.2), layout="constrained")
res = ep.hist_vs_pdf(
    amplitude,
    rayleigh_pdf,
    ax=ax,
    bins=60,
    log=True,
    label="samples",
    line_kw={"label": "Rayleigh density", "lw": 2},
)
ax.set_xlabel("amplitude")
ax.set_ylabel("probability density")
ax.set_ylim(1e-4, 1.0)
ax.legend()
```

## Covariance ellipses

`cov_ellipse` turns a two by two covariance matrix into the ellipse it
describes, at whatever multiple of a standard deviation is asked for. The
ellipse is a patch added to the axes and nothing else is drawn, so the
scatter underneath it, the limits, and the labels stay the caller's to set.
Drawing the one and two sigma contours from the sample covariance of the
points below is the usual check that a fitted uncertainty actually matches
the spread of the samples it came from.

```{code-cell} python
pair = np.vstack([posterior["period"], posterior["amplitude"]])
mean = pair.mean(axis=1)
cov = np.cov(pair)

fig, ax = plt.subplots(figsize=(4.6, 3.8), layout="constrained")
ax.scatter(pair[0], pair[1], s=3, alpha=0.25, edgecolors="none")

one_sigma_kw = {"lw": 2, "label": r"$1\sigma$"}
two_sigma_kw = {"lw": 2, "ls": "--", "label": r"$2\sigma$"}

one = ep.cov_ellipse(mean, cov, ax=ax, n_sigma=1, ellipse_kw=one_sigma_kw)
two = ep.cov_ellipse(mean, cov, ax=ax, n_sigma=2, ellipse_kw=two_sigma_kw)
ax.set_xlabel(LABELS["period"])
ax.set_ylabel(LABELS["amplitude"])
ax.legend(loc="upper left")
```

The patches come back under the `ellipse` key, which is what makes a later
adjustment a matter of setting a property rather than redrawing the figure.

```{code-cell} python
print(sorted(one.artists), type(one.artists["ellipse"]).__name__)
```
