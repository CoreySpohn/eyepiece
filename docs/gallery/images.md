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

# Images

The image primitives all solve the same underlying problem: a simulated
focal plane spans many decades of intensity, contains pixels that are
exactly zero, and is only interesting next to another image of the same
scene. This page draws each of them on synthetic arrays built in the page
itself, so every figure below is reproducible from the code beside it and
nothing here depends on a simulation package.

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

N = 128
PIXSCALE_LOD = 0.25
EXTENT = ep.extent_lod_from_pixels(N, PIXSCALE_LOD)

u = (np.arange(N) - (N - 1) / 2.0) * PIXSCALE_LOD
x, y = np.meshgrid(u, u)
r = np.hypot(x, y)


def point(px, py, flux):
    """A single unresolved source, in arbitrary intensity units."""
    return flux * np.exp(-0.5 * (np.hypot(x - px, y - py) / 0.35) ** 2)


def speckles(seed, level=1.0):
    """A power-law halo modulated by a speckle field of the given strength."""
    rng = np.random.default_rng(seed)
    return 3e-6 * (1.0 + r) ** -2.5 * (0.2 + level * rng.chisquare(2, size=r.shape))


CORE = point(0.0, 0.0, 1.0)

psf = CORE + speckles(11)
psf[:, 47] = 0.0
psf[90:96, 20:26] = 0.0
```

## The log floor

`imshow_log` clips its data to `floor` before the norm is built, which is
what allows the array above to be displayed at all: the dead column and the
dead pixel block are exactly zero, and a `LogNorm` autoscaled to those
values raises instead of drawing anything. Because the clip runs
first, the zeros are lifted to `floor` and land at the bottom of the
colormap along with everything else too faint to matter.

The floor therefore sets the displayed dynamic range as well, which the two
panels below demonstrate on the identical array. A low floor shows the
faint halo down to the level where the speckle field runs out, and a floor
raised five decades keeps only the core and the brightest speckles. Both
panels carry the extent from `extent_lod_from_pixels` and the axis labels
from `label_lod`, so the reader sees separations rather than pixel indices.

```{code-cell} python
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), layout="constrained")

low = ep.imshow_log(psf, ax=axes[0], extent=EXTENT, floor=1e-12)
high = ep.imshow_log(psf, ax=axes[1], extent=EXTENT, floor=1e-7)

ep.label_lod(axes[0])
axes[1].set_xlabel(r"$x$ [$\lambda/D$]")
axes[0].set_title("floor = 1e-12")
axes[1].set_title("floor = 1e-7")
```

The result carries the artists it drew, so anything the primitive did not
expose is still reachable, and an animation over a cube costs one call per
frame through `.update`, which re-applies the same floor to the new data.

```{code-cell} python
print(sorted(low.artists))
print(type(low.artists["image"]).__name__, type(low.artists["cbar"]).__name__)
```

## Bounds from the data

Left to itself, a log image runs from the data minimum to the data maximum,
and a single bright core sets the top of that range. The usual reflex is to
reference the floor to the peak, a fixed number of decades below `vmax`.
On a frame whose interesting structure sits near the background level, that
choice hides exactly what the figure was made to show.

`display_limits` derives the pair from the data instead. It returns plain
floats rather than a norm object, so the result drops into any primitive
here. `low_scale` anchors the floor to a fraction of a percentile, which is
what a rate map needs: its median IS the background, so half a median puts
the background mid-scale and leaves the structure above it visible.

```{code-cell} python
rate = speckles(3, level=0.6) + 2.0e-6

peak_lo, peak_hi = float(np.nanmax(rate)) * 1e-3, float(np.nanmax(rate))
data_lo, data_hi = ep.display_limits(rate, low=50.0, low_scale=0.5, high=99.9)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), layout="constrained")
ep.imshow_log(
    rate, ax=axes[0], vmin=peak_lo, vmax=peak_hi, extent=EXTENT, cbar_label="rate"
)
ep.imshow_log(
    rate, ax=axes[1], vmin=data_lo, vmax=data_hi, extent=EXTENT, cbar_label="rate"
)
axes[0].set_title("floor referenced to the peak")
axes[1].set_title("floor referenced to the median")
for ax in axes:
    ep.label_lod(ax)
```

The two floors differ by orders of magnitude on the same array.

```{code-cell} python
print(f"peak-referenced floor: {peak_lo:.3e}")
print(f"data-derived floor:    {data_lo:.3e}")
```

Before a log scale, pass `positive=True` so that zero and negative samples
are dropped before the percentile is taken. A masked frame is where this
matters: a dark-zone cut or a bad-pixel map leaves a large fraction of the
array at exactly zero, and a percentile taken over the whole frame then
lands on zero rather than on the data. Data with nothing left to measure
after that filtering returns an ordered pair rather than raising, so a
blank frame partway through an animation cannot stop a render.

```{code-cell} python
masked = np.where(r < 3.0, rate, 0.0)  # a dark-zone cut zeroes most of the frame

print(f"fraction at zero:   {np.mean(masked == 0.0):.1%}")
print("with zeros kept:   ", ep.display_limits(masked, low=1.0, high=99.9))
print("with zeros dropped:", ep.display_limits(masked, low=1.0, high=99.9, positive=True))
print("nothing positive:  ", ep.display_limits(np.zeros((8, 8)), positive=True))
```

## One norm across a row

`compare_row` builds a single norm from the minimum and maximum across all
of the images and passes that same object to every panel, so the three
frames below are directly comparable and not merely rescaled to look alike.
The images are three speckle levels of the same scene with a companion at a
fixed position and a fixed flux, and that companion emerges as the speckles
around it fall rather than because a panel was stretched differently.

Pinning `vmin` keeps the low end of the shared norm at a chosen value rather
than letting the faintest panel set it, and `cbar_label` names the quantity
once for the whole row. Because the row was handed axes it did not create,
the shared colorbar is an inset just outside the last panel, which leaves
the caller's three-panel layout exactly as it was assembled.

```{code-cell} python
companion = point(2.6, 1.4, 4e-6)
levels = [8.0, 2.0, 0.5]
frames = [CORE + speckles(3, lev) + companion for lev in levels]

fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), layout="constrained")
row = ep.compare_row(
    frames,
    titles=[f"speckles x {lev:g}" for lev in levels],
    axes=axes,
    extent=EXTENT,
    vmin=1e-9,
    cbar_label="intensity",
)
ep.label_lod(axes[0])
for ax in axes[1:]:
    ax.set_xlabel(r"$x$ [$\lambda/D$]")
```

### A colorbar slot the caller placed

`compare_grid` is the two-dimensional form of the same idea, and both take
`cax=` for a figure whose colorbar slot the caller has already placed: one
tall slot beside every row, say, instead of an inset off the last panel.
The shared colorbar is drawn into that axes and nothing else in the layout
moves.

```{code-cell} python
fig, slots = plt.subplot_mosaic(
    [["a", "b", "c", "bar"], ["d", "e", "f", "bar"]],
    width_ratios=[1, 1, 1, 0.06],
    figsize=(7.4, 3.8),
    layout="constrained",
)
panels = np.array([[slots[k] for k in "abc"], [slots[k] for k in "def"]])
grid = ep.compare_grid(
    [frames, [CORE + speckles(5, lev) for lev in levels]],
    titles=[[f"x {lev:g}" for lev in levels], ["", "", ""]],
    axes=panels,
    norm="log",
    extent=EXTENT,
    vmin=1e-9,
    cax=slots["bar"],
    cbar_label="intensity",
)
```

## Circles on an image

An aperture, a dark ring, or a working angle drawn over an image crosses
pixels from one end of the colormap to the other, so a circle in any single
color disappears somewhere along its length. `overlay_circle` draws a light
dash over a thin, solid, dark underlay, taking the two tones from the
style's background and text colors, so the circle stays visible over the
bright core and over the dark halo alike. It adds the circle without
touching the data limits: a working-angle circle larger than the view does
not widen it.

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
result = ep.imshow_log(frames[1], ax=ax, extent=EXTENT, vmin=1e-9)
ep.overlay_circle(ax, (2.6, 1.4), 0.7)  # a photometric aperture on the companion
ep.overlay_circle(ax, (0.0, 0.0), 1.22)  # the first dark ring of the core
ep.overlay_circle(ax, (0.0, 0.0), 20.0)  # an outer working angle beyond the view
ep.label_lod(ax)
```

The underlay is 1 pt wider than the dash by default, enough to outline a
thin dash at print size. A thicker dash, or a figure scaled up for a
slide, can want a different margin, so `underlay_kw` reaches the underlay
circle the way `circle_kw` reaches the dash, and a `lw` there sets its width
outright.

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
result = ep.imshow_log(frames[1], ax=ax, extent=EXTENT, vmin=1e-9)
ep.overlay_circle(ax, (2.6, 1.4), 0.7, circle_kw={"lw": 1.5},
                  underlay_kw={"lw": 3.6})
ep.label_lod(ax)
```

`ls` sets the line style of the light line, so a solid circle can mark a
hard edge, the rim of a pupil say, while a dashed one inside it marks a stop
or a working angle; the underlay stays solid either way. `label` writes a
name on the circle at `label_angle`, in degrees counterclockwise from +x,
just outside the circle and aligned away from its center, on a backing box
in the background color so it reads over any pixel. The label comes back
under `text`.

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
pupil_image = (r <= 10.0) * (1.0 + 0.2 * np.cos(x))
ax.imshow(pupil_image, extent=EXTENT, origin="lower", cmap="magma",
          interpolation="nearest")
ep.overlay_circle(ax, (0.0, 0.0), 10.0, ls="-", label="rim", label_angle=40.0)
stop = ep.overlay_circle(ax, (0.0, 0.0), 8.0, label="stop", label_angle=-135.0)
ax.set_axis_off()
print(sorted(stop.artists))
```

## A ruler over an image

`ruler` draws a double-headed dimension arrow between two points in data
coordinates, with its label at the midpoint on a backing box. Over an image
the arrow is the text color on a wider line in the background color, the
same ink-on-paper pair as the label, so it reads across the bright core and
the dark halo alike; `backing=False` drops the line and the box for a ruler
on a plain plot. The heads are sized in points like a phasor's, and a ruler
shorter than its two heads shrinks them to fit, so an animated ruler can
grow from nothing. Like `overlay_circle`, it leaves the limits alone.

Below, the ruler spans the first dark ring of the core, 2.44 lambda over D
across, and a second ruler runs from the star to the companion.

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
ep.imshow_log(frames[1], ax=ax, extent=EXTENT, vmin=1e-9)
width = ep.ruler(ax, (-1.22, -2.4), (1.22, -2.4), r"$2.44\,\lambda/D$",
                 side="below")
ep.overlay_circle(ax, (0.0, 0.0), 1.22)
ep.ruler(ax, (0.0, 0.0), (2.6, 1.4), "companion", side="left")
ax.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
ep.label_lod(ax)
```

`update(p0=None, p1=None, text=None)` moves the same arrow, backing line,
and label, so a frame loop can grow a ruler toward its final length.

```{code-cell} python
n_artists = len(ax.get_children())
for end in np.linspace(-1.22, 1.22, 6):
    width.update(p1=(end, -2.4), text=f"{end + 1.22:.2f}")
print(len(ax.get_children()) == n_artists, width.artists["text"].get_text())
```

## A cut against wavelength

A cut through an image, stacked row by row against a second variable, is a
kymograph: one row per time, wavelength, or roll angle, one column per
sample along the cut. A feature fixed on the sky stands as a vertical
stripe, and one that moves with the second variable leans. `kymograph`
draws it as a log image from the pixel centers of both axes, with no
interpolation and an `"auto"` aspect, since the two axes carry different
units. It is `imshow_log` with `centers=(x, y)`, an alternative to `extent`
that every image primitive takes, which builds the pixel-edge extent half a
step past the outer centers and refuses an unevenly spaced axis, which
`imshow` cannot draw.

The cut below runs outward from a star through a planet at 6 lambda over D,
across a band from 450 to 650 nm. A speckle is a diffraction feature, so its
distance from the star grows in proportion to the wavelength and it leans
outward, while the planet stays put. `overlay_line` draws a guide over the
image in the same light dash on a dark underlay as `overlay_circle`, so it
reads over the bright stripes and the dark gaps alike, with its label on a
backing box at a fraction of the way along the path.

```{code-cell} python
cut = np.linspace(0.0, 10.0, 161)
wavelength_nm = np.linspace(450.0, 650.0, 81)
lam = wavelength_nm[:, None] / 550.0


def spot(center, width=0.35):
    return np.exp(-0.5 * ((cut[None, :] - center) / width) ** 2)


halo = 1e-6 * (1.0 + cut[None, :]) ** -2.5 * np.ones_like(lam)
speckle_r = (2.5, 4.2, 7.6, 8.8)
cuts = halo + sum(3e-7 * spot(r0 * lam) for r0 in speckle_r) + 2e-7 * spot(6.0)

fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
kymo = ep.kymograph(cuts, cut, wavelength_nm, ax=ax, vmin=1e-9, vmax=1e-6,
                    cbar_label="brightness")
ep.overlay_line(ax, 7.6 * wavelength_nm / 550.0, wavelength_nm, ls=":",
                label=r"speckle, $r \propto \lambda$", label_at=0.85)
ep.overlay_line(ax, [6.0, 6.0], wavelength_nm[[0, -1]], label="planet",
                label_at=0.5)
ax.set(xlabel=r"distance from the star along the cut ($\lambda/D$)",
       ylabel="wavelength (nm)")
print(kymo.artists["image"].get_extent())
```

The guide's `update(x, y)` moves the same lines and label along a new path,
which may have a different number of points, as a cut swept around a star
during a roll does; `kymo.update(new_cuts)` redraws the kymograph under the
floor and norm of its first draw.

## Ratio and residual

`triptych` draws A and B through `compare_row`, so the first two panels
still share one norm and one colorbar, then adds a third panel that compares
them. The pair below is one speckle field before and after a correction that
suppresses it inside nine lambda over D, with a companion present in the
corrected frame.

With `mode="ratio"` the third panel is `b / a` on a diverging norm centered
on one, so the suppressed region and the companion sit on opposite sides of
the center color and everything left alone by the correction sits at the
center itself. The clip is the 99th percentile of the departure from unity,
which keeps the handful of pixels where A is nearly zero from washing out
the structure worth seeing.

```{code-cell} python
halo = speckles(7, 6.0)
dark_hole = 0.08 + 0.92 * (1.0 - np.exp(-((r / 9.0) ** 2)))

before = CORE + halo
after = CORE + halo * dark_hole + companion

fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9), layout="constrained")
ratio = ep.triptych(
    before,
    after,
    mode="ratio",
    titles=["uncorrected", "corrected", "ratio"],
    axes=axes,
    extent=EXTENT,
)
ep.label_lod(axes[0])
```

The same pair with `mode="residual"` puts `b - a` in the third panel on the
symmetric-about-zero norm that `imshow_diverging` builds. A ratio answers
"by what factor did this pixel change" and gives the faint outer region as
much weight as the bright center, while a residual answers "by how much" and
is dominated by wherever the intensity was largest, which is why the
suppressed inner halo takes over the panel and the companion nearly
disappears from it.

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9), layout="constrained")
residual = ep.triptych(
    before,
    after,
    mode="residual",
    titles=["uncorrected", "corrected", "difference"],
    axes=axes,
    extent=EXTENT,
)
residual.artists["image"][2].set_clim(-2e-6, 2e-6)
ep.label_lod(axes[0])
```

Tightening that panel's norm is the second line of the cell above, and it is
worth reading as an example of the contract rather than as a detail of this
figure. The comparison panel's bound is not a `triptych` parameter, so it is
set afterward on the artist the result handed back, which is the documented
move whenever a primitive does not expose something: route it through a
keyword dict if one reaches the right call, and otherwise reach through the
artists.

A signed map on its own goes through `imshow_diverging` directly, which is
the same third panel without the two reference panels beside it. Its `vlim`
pins the symmetric bound rather than deriving it from the data, so a bound
chosen to keep the companion visible is honored even though the brightest
suppressed pixels saturate at it, and two maps drawn from separate calls
under the same `vlim` remain comparable.

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.2, 3.4), layout="constrained")
diff = ep.imshow_diverging(
    after - before,
    ax=ax,
    extent=EXTENT,
    vlim=3e-6,
    cbar_label="difference",
)
ep.label_lod(ax)
```

## A complex field

`show_field` lays a complex array out as real and imaginary over amplitude
and phase. Each of the first three panels is scaled to its own power of ten
and says so in its title, which is what keeps a panel readable when its
values sit many decades below the field's peak. A panel far enough down to
be numerical noise, the imaginary part of a real and symmetric array for
instance, is labeled as machine zero rather than presented as structure.

Phase is undefined where there is no light, so it is masked rather than
drawn as numerical noise. The mask defaults to wherever the amplitude
clears a small fraction of its peak, and here it is passed explicitly as the
pupil itself. Given a figure through `fig=`, the primitive builds its own
two by two grid inside that figure and touches nothing else.

```{code-cell} python
radius = 0.55 * u[-1]
rho = r / radius
pupil = rho <= 1.0
amplitude = pupil * (1.0 - 0.4 * np.clip(rho, 0.0, 1.0) ** 2)
phase = 5.0 * (x**2 - y**2) / radius**2 + 1.2 * x / radius
field = amplitude * np.exp(1j * phase)

fig = plt.figure(figsize=(6.6, 5.4), layout="constrained")
panels = ep.show_field(field, fig=fig, mask=pupil, label="Aberrated pupil")
```

The returned `MosaicResult` holds the four panels in
`[Real, Imaginary, Amplitude, Phase]` order, together with the images,
colorbars, and titles it made, so a caller can adjust any one of them
without re-deriving which artist came from where.

```{code-cell} python
print(panels.axes.shape, sorted(panels.artists))
print([t.get_text().splitlines()[0] for t in panels.artists["title"]])
```
