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

## Moving arrows in an animation

The returned `.update(vectors, origin=None)` moves the same arrow patches to
new values, re-chaining them and recomputing the resultant as the first draw
did, and keeps every color, line style, and width. A frame loop therefore
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
