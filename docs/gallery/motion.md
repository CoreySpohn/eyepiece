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

# Motion and cameras

An explanatory animation is a sequence of beats: the viewer reads while
nothing moves, then something moves while no new text arrives. This page
builds the three tools that kind of animation needs on one synthetic
coronagraph field. A `Timeline` decides when each beat happens. A camera
decides where the viewer looks, either by changing an axes' limits or by
moving over a finished figure. A morph changes how the data are drawn while
every pixel keeps its identity, so the viewer can follow it from one
representation into the next.

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
EXTENT = ep.extent_lod_from_pixels(N, PIXSCALE_LOD)
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

## A timeline

A `Timeline` declares its beats in seconds and the presentation knobs they
set or move. A hold may change knobs when it starts, a new title for
instance; a ramp moves numeric knobs (linearly, or in log space for widths
and other scales) and changes nothing else, so text never arrives during
motion. `read` sizes a hold from the words on screen. Resolving the timeline
for a frame rate gives a `Plan`, whose frames are plain dicts, one per
frame, and whose `progress` reads any beat's clock. Here the zoom beat moves
no knob: its clock drives a camera path in the next section.

```{code-cell} python
FULL = (0.0, 0.0, 28.0)
CLOSE = (*PLANET_XY, 4.0)
path, length = ep.zoom_path(FULL, CLOSE)

tl = ep.Timeline(knobs={"title": ""})
tl.read("The whole field, star at the center", name="intro", min_s=1.5,
        set={"title": "The whole field"})
tl.hold(1.0, name="cue", set={"title": "Down to the companion"})
tl.ramp(4.0, name="zoom")
tl.hold(2.0, name="close")

plan = tl.at_fps(24)
for beat in plan.beats:
    print(f"{beat.name:6s} {beat.kind:5s} frames {beat.start:3d}-{beat.stop - 1:3d}")
k = plan.mid("zoom")
s = plan.progress(k, "zoom")
print(f"frame {k}: zoom progress {s:.2f}, view width {path(s)[2]:.2f} lambda/D")
```

Every frame of a ramp has moved: frame `i` of an `n`-frame ramp reaches
progress `(i + 1) / n`, so the last frame shows the target exactly, and
`mid` is the frame at progress one half. The easing curves all fix both ends
and pass through one half at the midpoint; they differ in how gently they
start and stop (the gray line is linear, for reference).

```{code-cell} python
s = np.linspace(0.0, 1.0, 101)
kinds = ("linear", "cosine", "smoothstep", "smootherstep")
fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.0), layout="constrained",
                         sharey=True)
for axis, kind in zip(axes, kinds, strict=True):
    axis.plot(s, s, color=ep.blend(plt.rcParams["text.color"], 0.3), lw=1.0)
    axis.plot(s, ep.ease(s, kind), color=hwostyle.palette[0])
    axis.set_title(kind, fontsize="small")
    axis.set_xlabel("progress")
axes[0].set_ylabel("eased progress")
plt.show()
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
second instead of 24, which only changes where the beat boundaries fall.

```{code-cell} python
fig, (ax, over) = plt.subplots(1, 2, figsize=(5.0, 2.6), layout="constrained",
                               width_ratios=(1.6, 1.0))
for target in (ax, over):
    ep.imshow_log(FIELD, ax=target, extent=EXTENT, vmin=NORM.vmin, vmax=NORM.vmax,
                  floor=NORM.vmin, colorbar=False)
ep.label_lod(ax)
over.set_xticks([])
over.set_yticks([])
over.set_title("overview", fontsize="small")
box = ep.overview_box(FULL, ax=over)

with tempfile.TemporaryDirectory() as tmp_dir:
    player = Path(tmp_dir) / "zoom.html"
    player_plan = tl.at_fps(8)
    with ep.record(fig, player, fps=8, dpi=56, free_limits=[ax]) as rec:
        for frame in player_plan.frames:
            view = path(player_plan.progress(frame["index"], "zoom"))
            xlim, ylim = ep.view_limits(view)
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

## A morph

`quad_image` draws an image as one quadrilateral per pixel, filled with the
pixel's own color, and `update` moves every pixel at once. The corners come
from `pixel_quads`, here moved by an unrolling map: each ring around the star
becomes a vertical column, separation along the bottom and angle up the side,
so the leftover light of every ring can be compared on one axis.

```{code-cell} python
half = PIXSCALE_LOD / 2.0
edges = np.concatenate([u - half, [u[-1] + half]])
ex, ey = np.meshgrid(edges, edges)
corners = np.stack([ex, ey], axis=-1)
keep = r <= 13.0
quads = ep.pixel_quads(corners, keep)

# Angles of each pixel's corners taken on its own center's branch, so pixels
# straddling the cut at 180 degrees are not torn across the panel.
center_angle = theta[keep][:, None]
qr = np.hypot(quads[..., 0], quads[..., 1])
qt = center_angle + np.angle(np.exp(1j * (np.arctan2(quads[..., 1], quads[..., 0]) - center_angle)))


def unroll(t):
    """Rings opened into columns at x = r, angle up the side, at progress t."""
    if t >= 1.0:
        return np.stack([qr, qt * 13.0 / np.pi], axis=-1)
    a = t / (1.0 - t)
    phi = qt / (1.0 + a)
    xs = qr * ((1.0 + a) * np.cos(phi) - a)
    ys = qr * (1.0 + a) * np.sin(phi) * ((1.0 - t) + t * 13.0 / (np.pi * np.maximum(qr, 1e-9)))
    return np.stack([xs, ys], axis=-1)


fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), layout="constrained")
for axis, t in zip(axes, (0.0, 0.55, 1.0), strict=True):
    res = ep.quad_image(FIELD, corners, ax=axis, keep=keep, norm=NORM)
    res.update(quads=unroll(t))
    axis.set_xlim(-14 if t < 1 else 0, 14)
    axis.set_ylim(-14, 14)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"unroll {t:.0%}", fontsize="small")
plt.show()
```

`brush` is how the image and its unrolled copy show that they hold the same
pixels without an arrow drawn between them. It veils both panels in their
background color and redraws the chosen pixels above the veil in each, so one
ring lights as a circle on the left and as a column on the right.

```{code-cell} python
fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 3.0), layout="constrained")
ep.imshow_log(FIELD, ax=left, extent=EXTENT, vmin=NORM.vmin, vmax=NORM.vmax,
              floor=NORM.vmin, colorbar=False)
unrolled = ep.quad_image(FIELD, corners, ax=right, keep=keep, norm=NORM)
unrolled.update(quads=unroll(1.0))
right.set_xlim(0, 13)
right.set_ylim(-13, 13)
right.set_xlabel(r"separation [$\lambda/D$]")
right.set_ylabel("angle around the star")
right.set_yticks([])
ep.label_lod(left)

linked = ep.brush([left, right], [quads, unroll(1.0)], FIELD[keep], norm=NORM)
linked.update(np.abs(r[keep] - 8.0) <= half)
plt.show()
```

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
    ep.imshow_log(image, ax=axis, extent=EXTENT, vmin=NORM.vmin, vmax=NORM.vmax,
                  floor=NORM.vmin, colorbar=False)
    axis.set_title(title)

with ep.PageCamera(page, (640, 360), overview=True) as cam:
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
light = ep.spotlight(page, [(box.x0, box.y0, box.x1, box.y1)])
with ep.PageCamera(page, (640, 240)) as still:
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
with ep.PageCamera(fig, (1920, 1080), overview=True) as cam:
    path, _ = ep.zoom_path(cam.page, cam.fit_axes(axes[1]))
    with ep.RawSink("tour.mp4", cam.size_px, fps=30) as sink:
        for t in np.linspace(0.0, 1.0, 60):
            sink.write(cam.render(path(ep.ease(t))))
```
