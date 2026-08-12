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

# One scene, several views

The figure this library exists for is one scene shown several ways at once:
a trajectory beside the focal plane it lands in, beside whatever measurement
the analysis actually turns on. Such a figure is only readable when the
panels agree with each other, which means a given source is the same color
and the same marker everywhere it appears, every spatial panel covers the
same field of view, and, once the figure moves, every panel is showing the
same instant.

Agreement of that kind is usually arranged by accident and maintained by
hand, and it breaks quietly. A panel that takes the next palette color gets
a different color as soon as a panel is added ahead of it, a panel that
derives its own half width from its own array disagrees with its neighbor by
half a pixel or by the whole field, and an animation whose panels each hold
their own time base drifts apart over a long run. `SourceStyles` and `Frame`
exist so that none of those are per-panel decisions: the caller makes each
one once and hands the result to every panel.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece as ep

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small. A real figure script
# keeps the style library's 300 dpi print policy and omits this line.
plt.rcParams["savefig.dpi"] = 120
```

## The scene

The scene is a star with two companions on inclined circular orbits,
generated in the page from closed-form expressions. `position` gives a
source's sky-plane offset at a time, `detector_image` renders the whole
scene into a focal plane at that same time, and `separation` reduces it to
the one number a detection threshold is usually written against. Those three
are three views of one state, which is exactly the situation the page is
about.

```{code-cell} python
N = 96
HALF_FOV_LOD = 14.0

ORBITS = {
    "b": {"radius": 5.6, "period": 1.0, "tilt": 0.35, "phase": 0.7, "flux": 1.5e-5},
    "c": {"radius": 10.4, "period": 2.6, "tilt": 0.75, "phase": 2.3, "flux": 6.0e-6},
}

u = (np.arange(N) - (N - 1) / 2.0) * (2.0 * HALF_FOV_LOD / N)
x, y = np.meshgrid(u, u)
r = np.hypot(x, y)
PSF_SIGMA_LOD = 0.7
CORE = np.exp(-0.5 * (r / PSF_SIGMA_LOD) ** 2)
HALO = 4e-7 * (1.0 + r) ** -2.2


def position(name, t):
    """Sky-plane offset of one source at time `t`, in lambda/D."""
    if name == "star":
        return 0.0, 0.0
    orbit = ORBITS[name]
    angle = 2.0 * np.pi * t / orbit["period"] + orbit["phase"]
    radius, tilt = orbit["radius"], orbit["tilt"]
    return radius * np.cos(angle), radius * np.sin(angle) * tilt


def detector_image(t):
    """The whole scene rendered into a focal plane at time `t`."""
    image = CORE + HALO
    for name, orbit in ORBITS.items():
        px, py = position(name, t)
        image = image + orbit["flux"] * np.exp(
            -0.5 * (np.hypot(x - px, y - py) / PSF_SIGMA_LOD) ** 2
        )
    return image


def track(name, n=400):
    """One source's full path, as an `(n, 2)` array."""
    times = np.linspace(0.0, ORBITS[name]["period"], n)
    return np.array([position(name, t) for t in times])


def separation(name, times):
    """One source's projected separation at each of `times`."""
    return np.array([np.hypot(*position(name, t)) for t in times])
```

## The three objects every panel gets

`SourceStyles` assigns each name a palette color and a marker in declaration
order, so `b` is one color and one marker in every panel of every figure
built from this object. The star is declared even though it is drawn with a
star marker, because it is the declaration order that fixes the assignment,
and dropping it would shift both companions by one color.

`Frame` holds one field-of-view choice and hands out the extent that goes
with it. It is frozen, so a panel cannot widen the field for its neighbors,
and it carries the pixel scale alongside the half width, so the panel that
maps an array index onto the sky reads it from the same object that gave the
image panel its extent.

The clock is an ordinary array. Nothing in the library owns it, and that is
the point: a frame index means the same instant in every panel because every
panel indexes the same array.

```{code-cell} python
STYLES = ep.SourceStyles(["star", "b", "c"])
FRAME = ep.Frame(half_fov_lod=HALF_FOV_LOD, pixscale_lod=2.0 * HALF_FOV_LOD / N)
EPOCHS = np.linspace(0.0, 1.0, 20)

print("b:", STYLES["b"], " c:", STYLES["c"])
print("extent:", FRAME.extent_lod(), " pixel scale:", round(FRAME.pixscale_lod, 4))
```

## Building the views

`build_views` takes the figure and the three coordination objects and
returns the artists that will move. Every panel reads its color from the
styles it was handed and its limits from the frame it was handed, and no
panel reaches for a module-level default or for whatever the previous panel
happened to leave behind.

The third panel is worth noticing: it is a plain pair of `ax.plot` calls,
with no eyepiece primitive involved, and it still belongs to the same figure
because it reads the same styles object. Coordination through data does not
require every panel to be drawn by this library.

```{code-cell} python
def build_views(fig, styles, frame, epochs):
    """Three views of the scene, all reading the same styles, frame, and clock."""
    ax_sky, ax_image, ax_sep = fig.subplots(1, 3)
    extent = frame.extent_lod()
    markers = {"sky": {}, "image": {}, "sep": {}}

    for name in ORBITS:
        style = styles[name]
        ep.fading_track(track(name), ax=ax_sky, color=style["color"])
        (markers["sky"][name],) = ax_sky.plot(
            [], [], ls="none", marker=style["marker"], color=style["color"], ms=7
        )
    ax_sky.plot([0], [0], ls="none", marker="*", ms=13,
                color=styles["star"]["color"])
    ax_sky.set_xlim(extent[0], extent[1])
    ax_sky.set_ylim(extent[2], extent[3])
    ax_sky.set_aspect("equal")
    ax_sky.set_title("sky plane")
    ep.label_lod(ax_sky)

    image = ep.imshow_log(
        detector_image(epochs[0]), ax=ax_image, extent=extent,
        floor=1e-9, vmax=1.0, colorbar=False,
    )
    for name in ORBITS:
        style = styles[name]
        (markers["image"][name],) = ax_image.plot(
            [], [], ls="none", marker=style["marker"], mfc="none",
            mec=style["color"], ms=12, mew=1.4,
        )
    ax_image.set_title("focal plane")
    ax_image.set_xlabel(r"$x$ [$\lambda/D$]")

    for name in ORBITS:
        style = styles[name]
        ax_sep.plot(epochs, separation(name, epochs), color=style["color"],
                    lw=1.5, label=name)
        (markers["sep"][name],) = ax_sep.plot(
            [], [], ls="none", marker=style["marker"], color=style["color"], ms=7
        )
    ax_sep.set_xlim(epochs[0], epochs[-1])
    ax_sep.set_ylim(0.0, frame.half_fov_lod)
    ax_sep.set_xlabel("time [orbits of b]")
    ax_sep.set_ylabel(r"separation [$\lambda/D$]")
    ax_sep.set_title("separation")
    ax_sep.legend(loc="lower right")

    return {"image": image, "markers": markers}


def show(views, epochs, k):
    """Advance every panel of `views` to epoch `k`."""
    t = epochs[k]
    views["image"].update(detector_image(t))
    for name in ORBITS:
        px, py = position(name, t)
        views["markers"]["sky"][name].set_data([px], [py])
        views["markers"]["image"][name].set_data([px], [py])
        views["markers"]["sep"][name].set_data([t], [np.hypot(px, py)])
```

## One epoch

Building the figure and advancing it to a single epoch gives the static
version. The companion drawn as a square in the sky panel is the square ring
in the focal plane and the square on the separation curve, and the two
spatial panels reach the same fourteen lambda over D in every direction
because both took their limits from `FRAME`.

```{code-cell} python
fig = plt.figure(figsize=(9.2, 3.0), layout="constrained")
views = build_views(fig, STYLES, FRAME, EPOCHS)
show(views, EPOCHS, 8)
```

## Why the coordination is data

The obvious alternative is state: a module-level color registry a primitive
consults, or a style that is activated and then read implicitly by whatever
draws next. Every version of that makes a panel's appearance depend on what
else has run in the process and in what order, which is the failure this
library avoids everywhere else by resolving style at call time and holding
no state of its own. A registry would put the state back.

Passing the object instead has three consequences worth stating. The
dependency is visible at the call site, so a reader can see that two panels
agree because they were handed the same thing rather than because a global
happened to hold still. Each primitive stays a pure function of its
arguments, so a panel can be drawn on its own, in any order, into any
figure, and come out the same. And the assignment is reproducible across
processes, because `SourceStyles` resolves colors from the palette in
declaration order at construction, so two instances built from the same
names agree exactly and a figure built by one script matches a figure built
by another with no synchronization between them.

```{code-cell} python
elsewhere = ep.SourceStyles(["star", "b", "c"])
print(elsewhere["b"] == STYLES["b"], elsewhere["c"] == STYLES["c"])
```

## The same views, animated

The animated version calls the same builder on a fresh figure and hands
`animate` a draw function that advances every panel together. One draw pass
feeds every sink, and here the sink is an embedded player, so the frames
travel with the page and the build needs no ffmpeg.

Nothing is cleared and nothing is rebuilt between frames. `show` mutates the
image through the `update` that `imshow_log` returned, which re-applies the
log floor from the first draw, and moves each marker with `set_data`, so the
artist count is fixed for the whole render and the color scale cannot drift.
The frame index reaches all three panels, which is the last piece of the
coordination: the panels advance together because they share the clock, not
because they are refreshed at the same rate.

```{code-cell} python
fig = plt.figure(figsize=(7.2, 2.4), layout="constrained")
views = build_views(fig, STYLES, FRAME, EPOCHS)

anim = ep.animate(fig, lambda fig, k: show(views, EPOCHS, k), len(EPOCHS), fps=8)
HTML(anim.jshtml(dpi=100))
```

## What to copy

Build the coordination objects once, at the top, before any figure exists.
Take them as parameters of whatever builds the panels rather than reading
them from module scope inside it, because a builder that reads a global is
back to being order-dependent and cannot be handed a different palette or a
different field of view without being edited. Hand the builder a figure and
let it own only the panels it makes, so the same builder serves a static
figure and an animated one. Return the artists that will move, and mutate
those, rather than redrawing a panel from scratch. And keep the clock a
single array that every panel indexes, so that advancing the figure is one
integer rather than one integer per panel.
