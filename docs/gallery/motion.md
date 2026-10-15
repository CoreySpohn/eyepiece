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
motion. `read` sizes a hold from the words on screen by the reading rule
of `Tempo`. Resolving the timeline for a frame rate gives a `Plan`, whose
frames are plain dicts, one per frame, and whose `progress` reads any
beat's clock. Here the zoom beat moves no knob: its clock drives a camera
path in the next section.

```{code-cell} python
FULL = (0.0, 0.0, 28.0)
CLOSE = (*PLANET_XY, 4.0)
path, length = ep.zoom_path(FULL, CLOSE)

tl = ep.Timeline(knobs={"title": ""})
tl.read("The whole field, star at the center", name="intro",
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

## Attention: a flight, a callout, a leader

A beat has to say where to look. Three moves do it without touching the
data. `flight` carries a copy of a mark from one panel onto another, so the
viewer sees a value read in one place become a point somewhere else.
`callout` writes a caption of a few words beside the thing it names.
`leader` joins a caption to its mark when the two cannot touch, and refuses
to draw one that breaks the connector rules of the
{doc}`evidence guide <../evidence>`: it must end on the mark it names, run
shorter than a quarter of its panel, cross no data and no other text, and
take a neutral rather than a data color. Drawing never raises, so a still
or a PDF save is never aborted half way; `faults()` on the leader lists what
it breaks, and `audit(fig)` checks every leader and link of a figure at once
and raises `AttentionFault` for those made with `strict=True` (the default).
A recording calls it on the composite.

Every position is an `Anchor`, a data point (`Anchor.from_data`) or a part
of a drawn artist (`Anchor.from_artist`), resolved in display space when the
figure is drawn. The moves' artists belong to the figure, not to an axes,
so they never change data limits. Each result's `update(u)` sets progress
absolutely, and each has a rest state: a flight is absent, a callout is
fully drawn.

```{code-cell} python
grid = np.linspace(-0.2, 0.8, 200)
height = np.exp(-0.5 * ((0.35 - 0.25) / 0.1) ** 2) / (0.1 * np.sqrt(2.0 * np.pi))

fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.0, 3.6), layout="constrained")
top.plot(grid, np.exp(-0.5 * ((grid - 0.25) / 0.1) ** 2) / (0.1 * np.sqrt(2.0 * np.pi)))
top.axvline(0.35, lw=0.8, color=ep.neutral(0.5))
(read,) = top.plot([0.35], [height], "o", ms=6)
(landed,) = bottom.plot([0.25], [height], "o", ms=6, color=read.get_color())
top.set(xlim=(-0.2, 0.8), ylim=(0.0, 4.6), ylabel="density")
bottom.set(xlim=(-0.2, 0.8), ylim=(0.0, 4.6), xlabel="mean", ylabel="likelihood")
# Lay the panels out once, then freeze them: the moves are figure artists
# outside the layout, and an animation's panels must not shift between frames.
fig.canvas.draw()
fig.set_layout_engine("none")

carry = ep.flight(read, ep.Anchor.from_artist(landed), fig=fig, arc=0.15)
note = ep.callout(read, "its height is the likelihood")


def grab(flight_u, landed_shown):
    carry.update(flight_u)
    landed.set_visible(landed_shown)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


frames = [grab(0.0, False), grab(0.5, False), grab(1.0, True)]
shown, cells = plt.subplots(1, 3, figsize=(9.0, 2.2), layout="constrained")
for cell, frame, title in zip(cells, frames, ("read", "carried", "landed"), strict=True):
    cell.imshow(frame, interpolation="nearest")
    cell.set_title(title)
    cell.set_axis_off()
plt.show()
```

In an animation each frame's state sets the progress of each move. A
caption of a few words should start to appear just before the event it
names, so its `update` runs from 0 to 1 over a few frames ending then; the
flight's runs over the event itself; and `update(1.0, level=0.35)` dims a
caption that has been read, so the next event has the eye to itself. A
recording's scale-drift check never sees the moves, since none of them is
inside an axes.

Three limits follow from the moves living on the figure. A callout's
`side="auto"` is judged once, when it is made, so lay the figure out
first. A ``bbox_inches="tight"`` save does not grow to hold a caption
outside the axes, and a blitting loop does not repaint the moves. A
`crossfade` label must keep the opacity it lands at: the flight hides it
until the end of the move and restores it there.

## Staging: build an axis, dim what was read, gather a row

An explanation that opens on a full figure leaves the viewer to find the
subject. Three more moves let a clip open nearly empty and bring in one
thing at a time. `build_axes` draws an axis on in front of the viewer: the
spine from the origin, each tick as the spine reaches it, its number just
after, and the axis name last. `Stage` sets named groups to absolute levels,
present or absent, full or dimmed toward the background, so what the viewer
has already read recedes. `collapse` gathers a row of cells into a bar whose
length is their sum, and can carry the bar to the panel that shows the sums.
Each has an absolute `update` and a rest state equal to the finished figure.

```{code-cell} python
albedo = np.linspace(0.1, 0.7, 25)
row = np.exp(-0.5 * ((albedo - 0.47) / 0.05) ** 2)
edges = np.linspace(0.0875, 0.7125, 26)
cells = np.stack([np.c_[edges[:-1], [0.8] * 25], np.c_[edges[1:], [0.8] * 25]], 1)

fig, (left, right) = plt.subplots(
    1, 2, figsize=(6.0, 2.6), sharey=True, layout="constrained"
)
left.set(xlim=(0.05, 0.75), ylim=(0.5, 2.0), xlabel="albedo", ylabel="radius")
right.set(xlim=(0.0, 8.0), xlabel="row total")
(curve,) = left.plot(albedo, 0.3 / albedo**0.5 + 0.2, color=ep.neutral(0.7))
fig.canvas.draw()
fig.set_layout_engine("none")

build = ep.build_axes(right)
stage = ep.Stage({"scenery": [curve]})
start = ep.Anchor.from_data(left, (0.0875, 0.8))
end = ep.Anchor.from_data(left, (0.4, 0.8))
landing = (
    ep.Anchor.from_data(right, (0.0, 0.8)),
    ep.Anchor.from_data(right, (row.sum(), 0.8)),
)
gather = ep.collapse(
    left, cells, row, start, end, fig=fig, colors=plt.cm.Greys(row), color="C3",
    to=landing,
)


def grab(axis_u, dim_u, gather_u, carry_u):
    build.update(axis_u)
    stage.set(dimmed=["scenery"], u=dim_u)
    gather.update(gather_u, carry=carry_u)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


frames = [grab(0.4, 0.0, 0.0, 0.0), grab(1.0, 1.0, 0.6, 0.0), grab(1.0, 1.0, 1.0, 0.6)]
shown, panels = plt.subplots(1, 3, figsize=(9.0, 1.9), layout="constrained")
titles = ("axis drawn on", "row gathered", "bar carried")
for panel, frame, title in zip(panels, frames, titles, strict=True):
    panel.imshow(frame, interpolation="nearest")
    panel.set_title(title)
    panel.set_axis_off()
plt.show()
```

In a clip the frame's time sets each move's progress, so any frame draws
itself from its state alone, and the last frame, with every move at rest,
is the composite still.

## Emphasis: pulse, outline, term row, turn, link

Five more moves say "this one" for a thing already on screen, or join two
panels. `indicate` is a pulse: for a moment the artist's line width, marker
size, or font size grows by 1.15 and its colors move toward the text color,
then it returns, so a frame at the start or the end is the frame without the
move. It adds no hue, and it pulses from wherever the artist is, so on an
artist a `Stage` holds it swells from the stage's level (set the stage, then
the pulse, in each frame). `circumscribe` draws an outline around an artist
or an anchor, clockwise from the top, and lets it fade. `term_row` lays an
equation out as one text per term, so each term can be the
target of an anchor, a stage group, a flight, or a pulse of its own; a term
whose text changes pushes its neighbors over at the next draw. `flight` turns
what it carries, a segment or a bar, by `turn_deg`, so a stub beside the star
can be carried and turned a quarter turn onto a velocity axis. `link` grows
a line from a mark in one panel to a mark in another; the quarter-panel rule
of a leader does not apply, and `faults()` instead checks where the link
touches data, in every panel it passes through.

```{code-cell} python
fig, (star, speed) = plt.subplots(
    1, 2, figsize=(8.0, 2.8), gridspec_kw={"wspace": 0.35}
)
star.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), xlabel="x", ylabel="y")
speed.set(xlim=(0.0, 1.0), ylim=(-1.0, 1.0), xlabel="phase", ylabel="velocity")
star.set_aspect("equal")
orbit = np.linspace(0.0, 2.0 * np.pi, 200)
star.plot(0.6 * np.cos(orbit), 0.6 * np.sin(orbit), color=ep.neutral(0.4))
(stub,) = star.plot([0.6, 0.6], [0.0, 0.3], lw=3.0, color="C1", solid_capstyle="butt")
(orbiter,) = star.plot([0.6], [0.0], "o", ms=7, color="C0")
speed.plot(orbit / (2.0 * np.pi), 0.5 * np.sin(orbit), color=ep.neutral(0.4))
fig.canvas.draw()
fig.set_layout_engine("none")

row = ep.term_row(
    ep.Anchor.from_data(speed, (0.05, -0.8)), [r"$v_r$", "=", "0.30"], ha="left"
)
ring = ep.circumscribe(orbiter, shape="ellipse", pad_pt=5.0)
pulse = ep.indicate(stub)
turn = ep.flight(
    stub, ep.Anchor.from_data(speed, (0.0, 0.3)), fig=fig, turn_deg=90.0, arc=0.2
)
tie = ep.link(orbiter, ep.Anchor.from_data(speed, (0.0, 0.0)))


def grab(ring_u, pulse_u, turn_u, link_u):
    ring.update(ring_u)
    pulse.update(pulse_u)
    turn.update(turn_u)
    tie.update(link_u)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


frames = [grab(0.55, 0.0, 0.0, 0.0), grab(0.0, 0.5, 0.0, 0.0), grab(0.0, 0.0, 0.5, 0.5)]
shown, cells = plt.subplots(1, 3, figsize=(9.0, 1.8), layout="constrained")
titles = ("outlined", "pulsed", "turned and linked")
for cell, frame, title in zip(cells, frames, titles, strict=True):
    cell.imshow(frame, interpolation="nearest")
    cell.set_title(title)
    cell.set_axis_off()
plt.show()
```

A term is rewritten by setting its text in the loop, for example
``row.artists["text"][2].set_text("0.45")``, and `Anchor.from_artist` of a term
follows it. Use `term_row` for terms whose text you choose, and `readout` (the
next section) for a number that counts, because a rewritten term changes width
and pushes its neighbors. Call `audit(fig)` once the composite is laid out to
vet every leader and link of the clip.

## Morph: one path becomes another

`morph` carries one path into another so the viewer can follow the shape
rather than see one vanish and another appear. Both paths, a `Line2D`, a
`PathPatch`, or an array of vertices, are resampled to the same number of
points evenly spaced in arc length, and a ghost blends the two samplings
under the same easing as every other move, along with stroke color, fill
color, opacity, and line width. For closed paths the start point and the
direction of the second are chosen so that corresponding points are as near
as possible, so a ring that is merely drawn the other way round does not
turn inside out on the way. Several subpaths are paired by size, and a
subpath with no partner shrinks to a point or grows from one. As with
`flight`, the ghost shows only between the ends, so the first and second
path are the frames at either end.

```{code-cell} python
fig, ax = plt.subplots(figsize=(8.0, 2.6))
ax.set(xlim=(-1.6, 1.6), ylim=(-1.2, 1.2), xticks=[], yticks=[])
ax.set_aspect("equal")
fig.canvas.draw()
fig.set_layout_engine("none")

phi = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
ring = np.c_[np.cos(phi), np.sin(phi)]
box = np.array([[0.9, 0.9], [-0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]])
turn = ep.morph(
    ring,
    box[::-1],
    ax=ax,
    closed=True,
    a_kw={"edgecolor": "C0", "facecolor": "none"},
    b_kw={"edgecolor": "C1", "facecolor": "C1", "alpha": 0.4},
)

shown, cells = plt.subplots(1, 5, figsize=(9.0, 1.9), layout="constrained")
for cell, u in zip(cells, (0.1, 0.3, 0.5, 0.7, 0.9), strict=True):
    turn.update(u)
    fig.canvas.draw()
    cell.imshow(np.asarray(fig.canvas.buffer_rgba()).copy(), interpolation="nearest")
    cell.set_title(f"u = {u:g}")
    cell.set_axis_off()
plt.show()
```

The ring is drawn counterclockwise and the box clockwise, and the frames show
the ring growing into the box without a twist. The ghost is hidden at
``u = 0`` and ``u = 1``, so in a clip the caller shows the ring before the
move and the box after it.

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
move. Arrows from `ep.arrow` are `FancyArrowPatch` artists and are accepted
once that function is merged.

```{code-cell} python
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(8.0, 2.6))
ax.set(xlim=(-3.2, 3.2), ylim=(-1.2, 1.2), xticks=[], yticks=[])
ax.set_aspect("equal")
fig.set_layout_engine("none")

grid = ep.warp_grid(ax, extent=(-8, 8, -4, 4), color=ep.neutral(0.5))
phi = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
(disk,) = ax.plot(0.8 * np.cos(phi), 0.8 * np.sin(phi), color="C1", lw=2.0)
arrow = ax.add_patch(FancyArrowPatch((0, 0), (0.8, 0), color="C0", mutation_scale=12))
tag = ax.text(0.9, 0.9, "(0.8, 0)", color="C0")

shear = np.array([[1.0, 1.0], [0.0, 1.0]])
turn = ep.warp([grid, disk, arrow, tag], shear, path="linear")

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

grid = ep.warp_grid(ax, extent=(-4, 4, -2, 2), color=ep.neutral(0.5))


def swirl(p):
    r = np.hypot(p[:, 0], p[:, 1])
    a = 1.2 * np.exp(-0.5 * r**2)
    return np.c_[
        np.cos(a) * p[:, 0] - np.sin(a) * p[:, 1],
        np.sin(a) * p[:, 0] + np.cos(a) * p[:, 1],
    ]


twist = ep.warp(grid, swirl)

shown, cells = plt.subplots(1, 4, figsize=(9.0, 1.7), layout="constrained")
for cell, u in zip(cells, (0.0, 0.33, 0.66, 1.0), strict=True):
    twist.update(u)
    fig.canvas.draw()
    cell.imshow(np.asarray(fig.canvas.buffer_rgba()).copy(), interpolation="nearest")
    cell.set_title(f"u = {u:g}")
    cell.set_axis_off()
plt.show()
```

## Readability check

A clip can draw without error and still be unreadable. `eyepiece.testing`
draws the figure at each sample time and lists what a viewer could not read.
Here a third label hangs four pixels into the second row of a step list, which
is the kind of collision no exception reports.

```{code-cell} python
import eyepiece.testing as ept

fig = plt.figure(figsize=(6.0, 2.4), dpi=100)
fig.text(0.05, 0.8, "1  Steer the beam", fontsize=14)
row = fig.text(0.05, 0.62, "2  Hold the star on the mask", fontsize=14)
fig.canvas.draw()
low = row.get_window_extent().y0
label = fig.text(0.05, (low + 4.0) / fig.bbox.height, "axis label", fontsize=14, va="top")
for fault in ept.readability_faults(fig):
    print(fault)
```

Pass `at` and `times` to sample a clip at its beat ends. `assert_readable`
raises `ReadabilityError` with the same report, for a test suite.

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

gap = ep.arrow((0.0, -1.0), (2.4, -1.0), ax=ax, kind="dimension", label="d")
force = ep.arrow((0.0, 0.0), (2.4, 0.6), ax=ax, label="v")
turn = ep.angle_arc((4.2, -0.6), 0.0, 55.0, 1.1, ax=ax, label=r"$\theta$")
sky = ep.compass((6.6, 0.4), 0.8, ax=ax, east="left", angle=20.0)

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

## Readout: a number that counts

A value that changes on screen, such as the time of a simulation or a
probability, should change the way the eye can follow: the digits run to the
new value while everything else holds still. `readout` is that number. Its
format is fixed (a fixed-point spec, a prefix, a suffix and a unit), and so is
its layout: the decimal point stays at one place for the life of the readout,
the integer part is right-aligned to it and the fraction left-aligned from it,
and each slot is as wide as the widest string it will hold. The label
therefore never jitters, in a font with tabular digits or one without.
`update(u)` shows the value `u` of the way from the old value to the new one
under the same easing as every other move, rounded to the precision of the
format; rounding is monotone, so a count never goes back, and a value that
rounds to zero shows no minus sign. `set(value)` starts the next move from the
value last set, so a readout chains: out, and back.

```{code-cell} python
fig, ax = plt.subplots(figsize=(7.0, 2.8), layout="constrained")
ax.set(xlim=(0.0, 12.5), ylim=(-1.2, 1.2), xlabel="time [h]", ylabel="offset")
ax.plot([0.0, 12.5], [-1.0, 1.0], color=ep.neutral(0.4))
(dot,) = ax.plot([0.0], [-1.0], "o", ms=8, color="C1")
fig.canvas.draw()
fig.set_layout_engine("none")

clock = ep.readout(
    ep.Anchor.from_data(ax, (2.4, 0.78)), 0.0, prefix="t = ", unit="h", ha="point"
)
offset = ep.readout(
    ep.Anchor.from_data(ax, (2.4, 0.52)),
    -1.0,
    prefix="offset = ",
    fmt="+.2f",
    ha="point",
)


def run(start, stop):
    """Chain a move of both readouts, and the dot with them, from `start` to `stop`."""
    clock.set(12.5 * stop)
    offset.set(2.0 * stop - 1.0)

    def at(u):
        clock.update(u)
        offset.update(u)
        e = ep.ease(u)
        x = 12.5 * (start + e * (stop - start))
        dot.set_data([x], [x / 12.5 * 2.0 - 1.0])

    return at


out = run(0.0, 1.0)
frames = []
for u in (0.0, 0.25, 0.5, 0.75, 1.0):
    out(u)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
back = run(1.0, 0.0)
back(0.5)
fig.canvas.draw()
frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

shown, cells = plt.subplots(2, 3, figsize=(9.0, 4.0), layout="constrained")
titles = ("u = 0", "u = 0.25", "u = 0.5", "u = 0.75", "u = 1", "and back, u = 0.5")
for cell, frame, title in zip(cells.flat, frames, titles, strict=True):
    cell.imshow(frame, interpolation="nearest")
    cell.set_title(title, fontsize="small")
    cell.set_axis_off()
plt.show()
```

The offset crosses zero in the middle of the move, and the point, the sign and
the unit all stay where they are. The readout shows the old value until
`update` is called, and `update(1.0)` shows the new one exactly; a still that
never calls `update` shows the value the readout was made with. `set` widens
the slots once, when it is called, if the new value needs more digits or a
sign, never during a count; pass `digits` and `signed` to reserve room for a
later value, so that nothing moves at all. The parts are figure-level texts,
so they are anchors for a `callout` and members of a `Stage` group, and
``update(u, level)`` dims the whole readout once it has been read. Prefer it
to a `term_row` term whose text is set in the loop: a term changes width as
its digits change and pushes its neighbors, where a readout keeps its place.
