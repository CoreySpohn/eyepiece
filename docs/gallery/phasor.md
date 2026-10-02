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

# Phasors

`phasor` draws complex numbers as arrows on the complex plane: each from a
shared origin, or chained tip to tail so the chain ends at the sum. It is the
picture behind interference, where a field at one point is a sum of
contributions that each carry an amplitude and a phase, and the brightness is
the squared length of the resultant. The inputs are a plain complex array, so
the page builds every example from closed-form expressions and seeded random
draws.

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
```

## Arrows and the phase key

Each value is an arrow from `origin`, on an equal-aspect plane with thin Re
and Im axes and no frame. `ring=True` draws the unit circle colored by the
same cyclic phase colormap a phase map is drawn with, so the direction of an
arrow and the color of a pixel in a phase map read against one key.

```{code-cell} python
amplitude = np.array([1.0, 0.8, 0.55])
phase = np.array([0.4, 2.3, -2.0])
res = ep.phasor(
    amplitude * np.exp(1j * phase),
    colors=[ep._style.color(i) for i in range(3)],
    ring=True,
    lim=1.3,
)
```

## Two waves, and arrows that lie on one another

`starts` gives each arrow its own start, one per vector, so arrows from the
origin and arrows placed elsewhere share one call. Arrow i runs from
`starts[i]` to `starts[i] + vectors[i]`, and the resultant of `show_sum`
still starts at `origin`. It is for a plane that is not chained, and
combining it with `chain=True` raises.

Two waves in step add along one line, and two waves half a turn apart fold
back along it, so their arrows lie on one another. `separate` draws such
arrows apart. Taking the arrows in the returned order, with the resultant
last, it moves each arrow that lies on the same line as an arrow before it
and shares a length with it, at their unshifted positions, by that fraction
of the plane's half-width, perpendicular to itself and clockwise from its
own direction. The resultant of the in-step pair therefore runs just
beneath the pair, and in the folded pair the second wave runs back above the
first while the resultant runs below. Arrows that meet at a single point,
as consecutive links of a chain or two arrows from one origin do, never
move. The automatic limits are fitted to the unshifted arrows.

`text_kw` styles the axis labels, here larger and on a backing box in the
page color.

```{code-cell} python
e1, e2 = 1.0, 0.6
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), layout="constrained")
turns = [0.0, 0.2, 0.5]
titles = ["in step", "a fifth of a turn apart", "half a turn apart"]
for ax, turn, title in zip(axes, turns, titles):
    wave2 = e2 * np.exp(2j * np.pi * turn)
    ep.phasor(
        [e1, wave2],
        ax=ax,
        starts=[0.0, e1],
        show_sum=True,
        colors=[ep._style.color(0), ep._style.color(2)],
        lim=(-0.3, 1.9, -0.9, 1.3),
        separate=0.08,
        text_kw={
            "fontsize": "medium",
            "bbox": {"facecolor": plt.rcParams["axes.facecolor"], "edgecolor": "none"},
        },
    )
    ax.set_title(title)
```

## A slit as a chain of wavelets

A slit seen from a point on a distant screen is a row of wavelets whose phases
advance steadily across it, so its field is a chain of equal arrows that turn
by a fixed angle each. On the axis the arrows line up and the resultant is as
long as it can be. Off axis the chain curls, and at the first dark fringe the
two edges of the slit differ by one wavelength: the chain winds through a full
turn, closes on itself, and the resultant vanishes. `show_sum=True` draws the
resultant from the start of the chain to its end, in the text color and
thicker. It is listed last among the returned arrows but layered beneath the
chain, so a straight chain lying along its own resultant stays visible on top.

The heads are sized in points and capped at the length of each arrow, so the
24 links of a chain never overshoot their own tails. A head scale below one
keeps them from crowding the chain, and thins the default shafts with them.

```{code-cell} python
n = 24
frame = (-0.2, 1.05, -0.1, 1.05)
turns = [0.0, 0.5, 1.0]
titles = ["on axis", "halfway to the first dark fringe", "first dark fringe"]
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), layout="constrained")
for ax, turn, title in zip(axes, turns, titles):
    steps = np.exp(2j * np.pi * turn * (np.arange(n) + 0.5) / n) / n
    ep.phasor(
        steps, ax=ax, chain=True, show_sum=True, head_scale=0.4, lim=frame
    )
    ax.set_title(title)
```

## Dials beside image features

`at=(x, y)` draws the plane as a small transparent inset, a dial, centered on
a point of another panel, with `size` its width in that panel's data units.
The dial is the returned `result.ax`. It follows its point if the parent's
limits change, and it takes no room from the parent, whose position and limits
stay as they were. A dial omits the axis labels by default.

Below, three speckles in a simulated focal plane each carry a dial showing the
complex field at its peak: the brightness is the squared length of the arrow,
and its direction, read against the phase ring, is the phase.

```{code-cell} python
rng = np.random.default_rng(0)
x = np.linspace(-6.0, 6.0, 241)
xx, yy = np.meshgrid(x, x)
centers = np.array([(-3.0, 2.0), (2.5, 3.0), (1.0, -3.0)])
fields = np.array([0.9 * np.exp(0.6j), 0.6 * np.exp(2.6j), 1.0 * np.exp(-1.9j)])
image = sum(
    abs(f) ** 2 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.8)
    for (cx, cy), f in zip(centers, fields)
)
image += 1e-3 * rng.random(image.shape)

res = ep.imshow_log(
    image, extent=(-6, 6, -6, 6), floor=1e-3, cbar_label="intensity"
)
for (cx, cy), field in zip(centers, fields):
    ep.phasor(
        [field],
        ax=res.ax,
        at=(cx + 1.8, cy),
        size=2.6,
        ring=True,
        lim=1.25,
        cross=False,
    )
```

`size_units="axes"` gives `size` instead as a fraction of the parent axes'
shorter side on screen, which suits a dial that should read at one size on
panels of different scales. The square is worked out from the parent's box
each time the figure is drawn, so it keeps its share of the panel through
layout and resizing, just as the dial keeps to its data point. Below, the
dials on a wide panel are each a third of its height, whatever its data
units.

```{code-cell} python
fig, ax = plt.subplots(figsize=(8.0, 2.6), layout="constrained")
t = np.linspace(0.0, 10.0, 400)
ax.plot(t, np.cos(2.0 * np.pi * t / 4.0), color=ep._style.neutral(0.6))
ax.set(xlim=(0.0, 10.0), ylim=(-1.6, 1.6), xlabel="time (s)")
for t0 in (1.0, 4.5, 8.0):
    ep.phasor(
        [np.exp(2j * np.pi * t0 / 4.0)],
        ax=ax,
        at=(t0, 0.0),
        size=0.33,
        size_units="axes",
        ring=True,
        lim=1.25,
        cross=False,
    )
```

## Chains standing over a curve

`curve_insets` stands a small square inset above each marked point of a
curve, joined to its point by a thin connector, with the point marked by a
dot. Given `chains`, one complex array per mark, each inset holds that
mark's chain of arrows and its resultant, drawn by `phasor` on one frame
fitted around every chain so arrow lengths compare between insets. The curve
below is a slit's brightness against angle, and above each mark is the chain
whose squared resultant is that brightness: straight on the axis, curled
halfway to the first dark fringe, and closed at it. `heights` stands the
insets on a level row instead of on their points, here lowered past the
first fringe where the curve is nearly dark.

The insets are placed each time the figure is drawn, from the parent's data
transform, so they keep to their points through layout and resizing as a
dial does, and they come back in `result.insets`, a tuple in mark order;
the dots and connectors are the result's `scatter` and `lines`.

```{code-cell} python
theta = np.linspace(0.0, 2.6, 400)
brightness = np.sinc(theta) ** 2
marks = [0.0, 0.5, 1.0, 1.5, 2.0]
n_arrows = 16
chains = [
    np.exp(2j * np.pi * m * (np.arange(n_arrows) + 0.5) / n_arrows) / n_arrows
    for m in marks
]

fig, ax = plt.subplots(figsize=(8.0, 3.4), layout="constrained")
ax.plot(theta, brightness, color=ep._style.color(0))
ax.set(xlim=(-0.25, 2.8), ylim=(-0.05, 1.9), xlabel=r"$\sin\theta$ ($\lambda/b$)",
       ylabel="brightness")
stand = ep.curve_insets(
    ax,
    theta,
    brightness,
    marks,
    heights=[1.2, 1.2, 1.2, 0.45, 0.45],
    above=0.03,
    chains=chains,
    phasor_kw={"head_scale": 0.35},
)
print(len(stand.insets), sorted(stand.artists))
```

Without `chains`, `build(inset_ax, i)` fills each inset however the figure
needs, and `size_units="data"` sizes the insets in the parent's x units
rather than as a fraction of its shorter side. Below, each inset is the
point source a curve of throughput was read from, at that separation.

```{code-cell} python
sep = np.linspace(0.0, 6.0, 300)
throughput = 1.0 - np.exp(-0.5 * (sep / 1.5) ** 2)
grid = np.linspace(-4.0, 4.0, 33)
gx, gy = np.meshgrid(grid, grid)
shown = [1.0, 2.5, 4.5]


def spot(inset, i):
    """A point source offset by the i-th separation, drawn in the inset."""
    image = np.exp(-0.5 * (np.hypot(gx - shown[i] + 2.0, gy) / 0.5) ** 2)
    inset.imshow(image, origin="lower", cmap="magma", interpolation="nearest")
    inset.set_xticks([])
    inset.set_yticks([])


fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
ax.plot(sep, throughput, color=ep._style.color(0))
ax.set(xlim=(0.0, 6.0), ylim=(0.0, 2.2), xlabel="separation", ylabel="throughput")
res = ep.curve_insets(
    ax, sep, throughput, shown, size=0.9, size_units="data", above=0.05, build=spot
)
```

## The phase key alone

`phase_ring` draws the ring by itself, a circle of `radius` about `center`
colored by the phase colormap, onto any axes, and `phasor(ring=True)` draws
its ring through it. On a given axes it adds the one collection and touches
nothing else: not the aspect, the ticks, the spines, or limits that were
set. Called without an axes, it makes a new one with an equal aspect so the
ring is round.

```{code-cell} python
fig, ax = plt.subplots(figsize=(3.4, 3.4), layout="constrained")
ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal")
ax.set_axis_off()
ring = ep.phase_ring(ax, radius=1.0, width=6.0)
marks = {"0": 0.0, r"$\pi/2$": np.pi / 2, r"$\pm\pi$": np.pi, r"$-\pi/2$": -np.pi / 2}
for label, phi in marks.items():
    ax.text(1.22 * np.cos(phi), 1.22 * np.sin(phi), label, ha="center", va="center")
```

## Moving arrows in an animation

The returned `.update(vectors, origin=None, starts=None)` moves the same
arrow patches to new values, re-chaining them, recomputing the resultant,
and separating overlapping arrows as the first draw did, and keeps every
color, line style, and width. An `origin` or `starts` of None keeps the last
one. A frame loop therefore
mutates the figure rather than clearing it. The number of vectors is fixed by
the first draw, and a different count raises. Below, the slit chain is drawn
once on the axis and then stepped to the first dark fringe; in a recorded
animation each step would be one `rec.frame()`.

```{code-cell} python
def steps_for(turn):
    """The slit chain's wavelets for a phase winding of `turn` full turns."""
    return np.exp(2j * np.pi * turn * (np.arange(n) + 0.5) / n) / n


res = ep.phasor(
    steps_for(0.0), chain=True, show_sum=True, head_scale=0.4, lim=frame
)
for turn in np.linspace(0.0, 1.0, 11):
    res.update(steps_for(turn))
```
