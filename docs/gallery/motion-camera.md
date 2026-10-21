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

# Motion cameras and views

```{code-cell} python
import tempfile
from pathlib import Path

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.colors import LogNorm

import eyepiece as ep

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small.
plt.rcParams["savefig.dpi"] = 110

N = 96
PIXSCALE_LOD = 0.3
EXTENT = ep.plot.extent_lod_from_pixels(N, PIXSCALE_LOD)
u = (np.arange(N) - (N - 1) / 2.0) * PIXSCALE_LOD
x, y = np.meshgrid(u, u)
r, theta = np.hypot(x, y), np.arctan2(y, x)

# Leftover starlight with twelve spokes, and a companion at 8 lambda/D.
HALO = 1e-7 * (1.0 + r) ** -3 * (1.0 + 0.8 * np.cos(12 * theta) ** 2)
PLANET_XY = (8.0 * np.cos(2.4), 8.0 * np.sin(2.4))
PLANET = 3e-9 * np.exp(-0.5 * (np.hypot(x - PLANET_XY[0], y - PLANET_XY[1]) / 0.6) ** 2)
FIELD = HALO + PLANET
NORM = LogNorm(1e-12, 1e-7)
```

```{code-cell} python
FULL = (0.0, 0.0, 28.0)
CLOSE = (*PLANET_XY, 4.0)
path, length = ep.motion.zoom_path(FULL, CLOSE)

tl = ep.clock.Timeline({"title": ""})
tl.read("The whole field, star at the center", name="intro",
        to={"title": "The whole field"})
tl.hold(1.0, name="cue", to={"title": "Down to the companion"})
tl.move("zoom", 4.0)
tl.hold(2.0, name="close")
```

## A data camera

A data camera changes an axes' limits, so the data magnify while text and
ticks keep their size. `zoom_path` gives the van Wijk and Nuij path between
two views: it spreads the zoom evenly over the move and, when the target is
farther than the view is wide, zooms out in transit so both ends stay in
context. `view_limits` turns a view into limits, `overview_box` marks it on
an overview panel, and `record(free_limits=...)` lets those limits move
without tripping the recorder's scale-drift guard, which still checks the
color scale. The player below resolves the same timeline at 8 frames per
second instead of 24, which only changes where the span boundaries fall.

```{code-cell} python
fig, (ax, over) = plt.subplots(1, 2, figsize=(5.0, 2.6), layout="constrained",
                               width_ratios=(1.6, 1.0))
for target in (ax, over):
    ep.plot.imshow_log(FIELD, ax=target, extent=EXTENT, vmin=NORM.vmin, vmax=NORM.vmax,
                  floor=NORM.vmin, colorbar=False)
ep.plot.label_lod(ax)
over.set_xticks([])
over.set_yticks([])
over.set_title("overview", fontsize="small")
box = ep.motion.overview_box(FULL, ax=over)

with tempfile.TemporaryDirectory() as tmp_dir:
    player = Path(tmp_dir) / "zoom.html"
    player_plan = tl.at_fps(8)
    with ep.render.record(fig, player, fps=8, dpi=56, free_limits=[ax]) as rec:
        for frame in player_plan.frames:
            view = path(player_plan.progress(frame["index"], "zoom"))
            xlim, ylim = ep.motion.view_limits(view)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            box.update(view)
            ax.set_title(frame["title"], fontsize="small")
            rec.frame()
    html = player.read_text()
plt.close(fig)
HTML(html)
```

The same plan drives the limits, the overview box, and the title, so they
never disagree about which instant is on screen.

## A page camera and a spotlight

A page camera moves over a finished figure as a whole. `PageCamera.render`
crops the figure to a view at the zoomed dpi, so a magnified view re-renders
its text and pixels rather than enlarging a picture of them. With
`overview=True` each frame keeps a margin holding the whole page, with the
current view boxed, so a zoomed viewer keeps their place.

```{code-cell} python
page, panels = plt.subplots(1, 3, figsize=(8.0, 3.0), layout="constrained")
titles = ("starlight", "companion", "both")
for axis, image, title in zip(panels, (HALO, PLANET + 1e-13, FIELD), titles, strict=True):
    ep.plot.imshow_log(image, ax=axis, extent=EXTENT, vmin=NORM.vmin, vmax=NORM.vmax,
                  floor=NORM.vmin, colorbar=False)
    axis.set_title(title)

with ep.motion.PageCamera(page, (640, 360), overview=True) as cam:
    frames = [cam.render(cam.page), cam.render(cam.fit_axes(panels[1]))]

shown, cells = plt.subplots(1, 2, figsize=(7.5, 2.2), layout="constrained")
for cell, frame in zip(cells, frames, strict=True):
    cell.imshow(frame, interpolation="nearest")
    cell.set_axis_off()
plt.show()
```

`spotlight` is the fixed-camera alternative: the figure stays where it is
and a veil in its background color dims everything outside one or more holes,
given in figure inches, the same units as a camera's views. Its `update`
moves the holes and fades the veil from one stop to the next.

```{code-cell} python
box = panels[1].get_tightbbox().transformed(page.dpi_scale_trans.inverted())
light = ep.motion.spotlight(page, [(box.x0, box.y0, box.x1, box.y1)])
with ep.motion.PageCamera(page, (640, 240)) as still:
    spot = still.render((4.0, 1.5, 8.0))

shown, cell = plt.subplots(figsize=(6.0, 2.3), layout="constrained")
cell.imshow(spot, interpolation="nearest")
cell.set_axis_off()
plt.show()
```

Frames from a page camera do not come from grabbing one figure, so they go
to a `RawSink` rather than to `record`. It writes raw frames to an H.264 mp4
with fixed encoder settings and no metadata, so the same frames always give
the same file, and it moves the file into place only when it finishes:

```python
with ep.motion.PageCamera(fig, (1920, 1080), overview=True) as cam:
    path, _ = ep.motion.zoom_path(cam.page, cam.fit_axes(axes[1]))
    with ep.render.RawSink("tour.mp4", cam.size_px, fps=30) as sink:
        for t in np.linspace(0.0, 1.0, 60):
            sink.write(cam.render(path(ep.clock.ease(t))))
```

## Warp: the plane moves

`warp` applies a 2x2 matrix or a pointwise map to artists that are already
drawn, in the data coordinates of their axes, so the viewer watches a shape
become its image. Nothing is replaced: the same lines, patches, arrows,
collections and texts are reshaped in place, and the move is at its start
when it is made. A matrix is blended as ``I + u (M - I)``, which shrinks a
rotation through the origin; ``path="rotation"`` splits the matrix into a
rotation and a stretch and turns the rotation through its angle instead.
A pointwise map first subdivides each path (keeping its corners, using the
same arc-length sampling as `morph`) and moves every point along the line to
its image. `warp_grid` draws the coordinate grid that usually carries the
move. Arrows from `ep.motion.arrow` are `FancyArrowPatch` artists and are accepted
once that function is merged.

```{code-cell} python
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(8.0, 2.6))
ax.set(xlim=(-3.2, 3.2), ylim=(-1.2, 1.2), xticks=[], yticks=[])
ax.set_aspect("equal")
fig.set_layout_engine("none")

grid = ep.motion.warp_grid(ax, extent=(-8, 8, -4, 4), color=ep.style.neutral(0.5))
phi = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
(disk,) = ax.plot(0.8 * np.cos(phi), 0.8 * np.sin(phi), color="C1", lw=2.0)
arrow = ax.add_patch(FancyArrowPatch((0, 0), (0.8, 0), color="C0", mutation_scale=12))
tag = ax.text(0.9, 0.9, "(0.8, 0)", color="C0")

shear = np.array([[1.0, 1.0], [0.0, 1.0]])
turn = ep.motion.warp([grid, disk, arrow, tag], shear, path="linear")

shown, cells = plt.subplots(1, 4, figsize=(9.0, 1.7), layout="constrained")
for cell, u in zip(cells, (0.0, 0.33, 0.66, 1.0), strict=True):
    turn.update(u)
    fig.canvas.draw()
    cell.imshow(np.asarray(fig.canvas.buffer_rgba()).copy(), interpolation="nearest")
    cell.set_title(f"u = {u:g}")
    cell.set_axis_off()
plt.show()
```

A swirl is a pointwise map, here a rotation whose angle falls with distance;
the straight grid lines bend into spirals on the way because they were
subdivided before they moved.

```{code-cell} python
fig, ax = plt.subplots(figsize=(8.0, 2.6))
ax.set(xlim=(-3.2, 3.2), ylim=(-1.2, 1.2), xticks=[], yticks=[])
ax.set_aspect("equal")
fig.set_layout_engine("none")

grid = ep.motion.warp_grid(ax, extent=(-4, 4, -2, 2), color=ep.style.neutral(0.5))


def swirl(p):
    r = np.hypot(p[:, 0], p[:, 1])
    a = 1.2 * np.exp(-0.5 * r**2)
    return np.c_[
        np.cos(a) * p[:, 0] - np.sin(a) * p[:, 1],
        np.sin(a) * p[:, 0] + np.cos(a) * p[:, 1],
    ]


twist = ep.motion.warp(grid, swirl)

shown, cells = plt.subplots(1, 4, figsize=(9.0, 1.7), layout="constrained")
for cell, u in zip(cells, (0.0, 0.33, 0.66, 1.0), strict=True):
    twist.update(u)
    fig.canvas.draw()
    cell.imshow(np.asarray(fig.canvas.buffer_rgba()).copy(), interpolation="nearest")
    cell.set_title(f"u = {u:g}")
    cell.set_axis_off()
plt.show()
```

## Arrows, angle arcs and a compass

`arrow`, `angle_arc` and `compass` are marks that an explanation builds up
rather than redraws. Each is one artist, so `update` lengthens an arrow, sweeps
an arc, or turns a compass without making anything new, and at ``u = 1`` the
frame is the figure as drawn.

An arrow has three kinds. A vector grows from its tail, a double arrow and a
dimension grow outward from their middle, and a dimension puts its label on the
line in a backing box so that it reads as a measure of a gap. An angle arc runs
between two directions about a vertex, with its head tangent to the arc at every
point of the sweep, so the reader sees which way the angle is measured. A
compass draws north and east from one point, and the caller says which side east
is on: ``east="left"`` is the sky as an observer looking up sees it, and
``east="right"`` is a chart drawn from outside. There is no default, because the
wrong side is a mirrored sky.

```{code-cell} python
fig, ax = plt.subplots(figsize=(8.0, 2.8), dpi=200)
ax.set(xlim=(-0.5, 7.5), ylim=(-1.4, 1.6), xticks=[], yticks=[])
ax.set_aspect("equal")
fig.canvas.draw()
fig.set_layout_engine("none")

gap = ep.motion.arrow((0.0, -1.0), (2.4, -1.0), ax=ax, kind="dimension", label="d")
force = ep.motion.arrow((0.0, 0.0), (2.4, 0.6), ax=ax, label="v")
turn = ep.motion.angle_arc((4.2, -0.6), 0.0, 55.0, 1.1, ax=ax, label=r"$\theta$")
sky = ep.motion.compass((6.6, 0.4), 0.8, ax=ax, east="left", angle=20.0)

shown, cells = plt.subplots(2, 2, figsize=(9.0, 4.6), layout="constrained")
for cell, u in zip(cells.flat, (0.25, 0.5, 0.75, 1.0), strict=True):
    for move in (gap, force, turn, sky):
        move.update(u)
    sky.update(u, angle=20.0 + 50.0 * u)
    fig.canvas.draw()
    cell.imshow(np.asarray(fig.canvas.buffer_rgba()).copy(), interpolation="nearest")
    cell.set_title(f"u = {u:g}")
    cell.set_axis_off()
plt.show()
```

The moves take new ends as well as a progress: ``force.update(tail=..., head=...)``
slides or lengthens the arrow, ``turn.update(end_angle=...)`` opens the angle
further, and ``sky.update(angle=...)`` turns the compass as a roll would. An arrow
or an arc shorter than its head shrinks the head to fit, and one too short to
see draws nothing.

## Following zoom

`follow_zoom` hangs a magnified inset on a panel and keeps its window on a
moving point, as a camera operator holds a planet in a field of speckle. The
inset is not a second plot: it draws the panel's own artists again through a
smaller window, so anything that moves or changes in the panel moves in the
inset on the same draw. A frame box on the panel marks the region the window
covers, and two connectors join the box to the inset. All three are placed
each time the figure is drawn, so the target can be an artist whose position
changes every frame, an `Anchor`, or a function of a time that ``update``
sets, and nothing needs calling per frame to make the window follow.

The magnification is exact on both axes and the box and the window always agree.
``update(u)`` fades the inset, the box and the connectors in; ``level`` is their
opacity and never changes where the window sits. The connectors are audited as
a `link` is, so ``ep.testing.audit`` names one that crosses data, and they are hidden
while the box overlaps the inset, so give an inset a corner the target does
not reach, or ``loc="auto"``, which keeps it in the corner farthest from the
target and moves it only when another corner is farther by a quarter of its
side, so a target that hovers where two corners tie does not flicker it. The
box, the inset frame and the connectors carry a halo in the panel's background
color, so they read over a dark image like this one and a light one alike. The panel keeps the whole field in view, so the viewer never loses the place
while the inset reads at five times.

```{code-cell} python
import eyepiece as ep

rng = np.random.default_rng(7)
yy, xx = np.mgrid[-4:4:160j, -6:6:240j]
rings = np.cos(3.0 * np.hypot(xx, yy)) * np.exp(-0.15 * np.hypot(xx, yy) ** 2)
field = rings + 0.15 * rng.standard_normal(rings.shape)

fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=100)
ax.imshow(field, extent=(-6, 6, -4, 4), cmap=hwostyle.cmaps.residual, vmin=-0.6, vmax=1.0)
ax.set(xlim=(-6, 6), ylim=(-4, 4), xticks=[], yticks=[])
ax.set_title("A planet crossing the speckle field", fontsize=16)
(planet,) = ax.plot([], [], "o", ms=5, mfc="none", mec="white", mew=1.2)


def orbit(t):
    return 3.4 * np.cos(2.2 * t - 1.0), 2.1 * np.sin(2.2 * t - 1.0)


planet.set_data(*([v] for v in orbit(0.0)))
fig.canvas.draw()
fig.set_layout_engine("none")

zoom = ep.motion.follow_zoom(ax, planet, zoom=5.0, loc="auto", size=0.36)


def pose(t):
    planet.set_data(*([v] for v in orbit(t)))
    zoom.update(min(1.0, t / 0.4))


N_ZOOM = 60


def draw_zoom(fig, k):
    pose(2.8 * k / (N_ZOOM - 1))

ep.testing.assert_readable(fig, at=pose, times=[0.0, 0.5, 1.4, 2.8])
ep.testing.audit(fig)
HTML(ep.render.animate(fig, draw_zoom, N_ZOOM, fps=20).jshtml(dpi=80))
```

The target above is the marker itself, so nothing but its position is updated
per frame. With a callable target, ``zoom.update(1.0, time=t)`` poses the window
at time ``t`` and keeps that time until it is changed.
