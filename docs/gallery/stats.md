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

## Samples converging on a reference

`convergence` draws a sequence of samples as dots beside their running mean
(or, with `running="sum"`, their running sum), with each reference value as
a labeled horizontal line and an optional tolerance band shaded around it.
The dots are a neutral tone and the running line carries the color, so the
eye follows the estimate settling while the scatter shows how noisy each
sample is. The reference labels sit at the right edge on backing boxes, and
the right limit leaves room for them past the last sample.

The samples below are the intensities of one speckle in independent frames,
which follow an exponential distribution: any one frame can be several times
the mean, and the running mean still closes on it rather than on the median,
the value half the frames fall below.

```{code-cell} python
rng = np.random.default_rng(3)
looks = rng.exponential(1.0, 240)

fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
ax.set(xlabel="frames averaged", ylabel="intensity / mean")
res = ep.convergence(
    looks,
    ax=ax,
    refs=[1.0, np.log(2.0)],
    ref_labels=["mean", "median"],
    ref_linestyles=["--", ":"],
    band=[0.1, None],
)
```

The limits are fixed from the full data on the first draw, and
`update(k)` reveals only the first `k` samples and the line up to them,
creating no artist, so an animation that adds one frame at a time keeps one
scale from start to finish. The three panels below are one call each, frozen
at three counts, as three frames of that animation would be.

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6), layout="constrained",
                         sharey=True)
for ax, k in zip(axes, [10, 60, 240]):
    frame = ep.convergence(looks, ax=ax, refs=1.0, ref_labels="mean", band=0.1)
    frame.update(k)
    ax.set_title(f"{k} frames")
```

`update` also takes `values=` and `refs=`, which replace the samples and move
the references, their labels, and their bands, for a figure whose parameter
changes between frames. With `running=None` the line is the values as given,
for a curve the caller already accumulated, and a second call on the same
axes widens the limits to cover both, so two curves converging on two
references share one panel.

## A trace about a level

`signed_trace` draws a sequence against a reference level and shades the
area between them in two colors, one where the trace is above the level and
one where it is below, with the crossings interpolated. The two colors are
the ends of the diverging colormap a signed map is drawn with, so a trace
beside a residual image shares its key. The limits are fixed from the full
trace on the first draw, and `update(k)` shows only the first `k` samples,
redrawing the two fills in the same `"fill"` list, so an animation keeps one
scale and one set of artists.

Below, a pixel's brightness wanders over ten hours about its static value:
brighter where the field it adds lines up with the static one, dimmer where
it opposes it.

```{code-cell} python
rng = np.random.default_rng(5)
t_hours = np.linspace(0.0, 10.0, 241)
drift = np.cumsum(rng.normal(0.0, 0.05, t_hours.size))
drift -= np.linspace(0.0, drift[-1], t_hours.size)
brightness = np.abs(1.0 + 0.35 * drift * np.exp(1j * 0.6 * t_hours)) ** 2

fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.8), layout="constrained",
                         sharey=True)
for ax, k in zip(axes, [90, 241]):
    trace = ep.signed_trace(brightness, ax=ax, x=t_hours, level=1.0,
                            level_label="static")
    trace.update(k)
    ax.set(xlabel="time (h)", title=f"{t_hours[k - 1]:.1f} h")
axes[0].set_ylabel("brightness / static")
print(sorted(trace.artists), len(trace.artists["fill"]), "fills")
```

`fill_colors=(above, below)` sets the two colors outright, `cmap=` samples
another diverging map, and `show_level=True` draws the level line without a
label.

## A histogram that fills

`hist_fill` counts samples into fixed bins and pins the y axis on the first
draw to the final counts, or to the expected counts of a law when one is
given, so `update(k)` counts only the first `k` samples into the same bars
and the histogram grows into a frame that never rescales. A frame that
rescaled to its tallest bar would make five samples look as settled as five
hundred. The law is drawn as expected counts, the number of samples times
the bin width times the density, which is what the bars converge to.

A count of one is a sliver on that frame, so `rug=` draws the first few
samples as dots on the baseline, one at each sample's value. The rug marks
where samples fell, not how many, and `update(k, rug_alpha=...)` fades it as
the bars take over. The law's curve is the expected count for every sample,
so `update(..., show_law=False)` keeps it hidden until the last one is in.
The speckle intensities below follow the exponential law of a fully
developed speckle field.

```{code-cell} python
looks = np.random.default_rng(8).exponential(1.0, 400)
edges = np.arange(0.0, 6.01, 0.25)

fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6), layout="constrained",
                         sharey=True)
for ax, k in zip(axes, [6, 60, 400]):
    hist = ep.hist_fill(looks, edges, ax=ax, law=lambda v: np.exp(-v),
                        law_label="exponential", law_label_x=1.4, rug=12)
    hist.update(k, rug_alpha=max(0.0, 1.0 - k / 60.0), show_law=k == 400)
    ax.set(xlabel="intensity / mean", title=f"{k} samples")
axes[0].set_ylabel("samples per bin")
print(sorted(hist.artists), axes[0].get_ylim())
```

`bins` may also be a count of equal bins spanning the samples. A law needs
equal bins, since expected counts per bin are a density times one width.
