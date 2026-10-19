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

# Motion clocks and readouts

```{code-cell} python
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

A `Timeline` declares named spans in seconds. A span is the one primitive:
a hold, a subject entering or leaving, a move, a caption arriving, a pulse,
a ring. Each span writes the knobs it names, and `Timeline.at_fps` derives
every knob on every frame from them, so a draw function never sees time. A
hold may step knobs when it starts, a new title for instance; a move ramps
numeric knobs (linearly, or in log space for widths and other scales) and
changes nothing else, so text never arrives during motion. `read` sizes a
hold from the words on screen by the reading rule of `Tempo`. Resolving the
timeline for a frame rate gives a `Plan`, whose frames are plain dicts, one
per frame, and whose `progress` reads any span's clock. Here the zoom span
writes no knob: its clock drives a camera path in the {doc}`camera catalog <motion-camera>`.

```{code-cell} python
FULL = (0.0, 0.0, 28.0)
CLOSE = (*PLANET_XY, 4.0)
path, length = ep.zoom_path(FULL, CLOSE)

tl = ep.Timeline({"title": ""})
tl.read("The whole field, star at the center", name="intro",
        to={"title": "The whole field"})
tl.hold(1.0, name="cue", to={"title": "Down to the companion"})
tl.move("zoom", 4.0)
tl.hold(2.0, name="close")

plan = tl.at_fps(24)
for span in plan.spans.values():
    print(f"{span.name:6s} {span.kind:5s} frames {span.k0:3d}-{span.k1 - 1:3d}")
k = plan.frame_of("zoom", "mid")
s = plan.progress(k, "zoom")
print(f"frame {k}: zoom progress {s:.2f}, view width {path(s)[2]:.2f} lambda/D")
```

Every frame of a span has moved: a span from `s` to `e` seconds covers frames
`frame_count(s)` up to `max(k0 + 1, frame_count(e))`, frame `i` of its `n`
frames reaches progress `(i + 1) / n` before easing, and the last frame shows
the target exactly. The middle frame of a span is `k0 + (n - 1) // 2`. The four plain easing curves all fix
both ends and pass through one half at the midpoint; they differ in how gently
they start and stop (the gray line is linear, for reference). The last two
curves, `back_out` and `running_start`, also fix both ends exactly but leave
the interval between them: the first overshoots its target by a tenth of the
move and settles back, and the second pulls back by a tenth before it goes.
Each takes a `strength`, the size of that excursion as a fraction of the move.

```{code-cell} python
s = np.linspace(0.0, 1.0, 101)
kinds = ("linear", "cosine", "smoothstep", "smootherstep", "back_out",
         "running_start")
fig, axes = plt.subplots(1, 6, figsize=(9.0, 2.0), layout="constrained",
                         sharey=True)
for axis, kind in zip(axes, kinds, strict=True):
    axis.plot(s, s, color=ep.blend(plt.rcParams["text.color"], 0.3), lw=1.0)
    axis.plot(s, ep.ease(s, kind), color=hwostyle.palette[0])
    axis.set_title(kind, fontsize="small")
    axis.set_xlabel("progress")
axes[0].set_ylabel("eased progress")
plt.show()
```

## One clock for a whole clip

The same `Timeline` places everything timed in a clip, narration included.
Each verb returns a `Span` handle, so `at=slide.end + 0.2` places a span
after another without float drift (a `Time` adds exactly). Without `at=`, a
span starts at the cursor `now` and moves it on; with `at=` it starts where
told and the cursor stays. `together` starts a group at `now` and `stagger`
spaces its targets. A subject span writes the knob of its own name, so
`enter("star")` ramps `star` from 0 to 1, `pulse("speckle")` ramps
`pulse.speckle`, and a caption writes `caption.<name>`.

Given a `Narration`, the same function lays each span on the word that names
it. A mark `{show:star}` starts `enter("star")` a tenth of a second before
the word, `{indicate:speckle}` centers the pulse on its word, and
`{circumscribe:halo}` starts the ring on it. Lines open in script order with
`line`, and a line lasts until its speech and padding end or its last
moving span ends plus 0.6 s, whichever is later. Run without a narration, the
function builds the silent clip: sections replace lines, a caption leaves once
it has been read, and a pulse or ring that only a cue places is skipped and
recorded in `plan.faults` as information. The plan also lists its events
(starts, middles, ends and cues, each with the frame that shows it) and the
frames a contact sheet should draw.

```{code-cell} python
from eyepiece import narration as nr

SCRIPT = nr.Script(
    (
        nr.Line(
            "find",
            "The coronagraph blocks the {show:star}star, and what leaks "
            "through is {indicate:speckle}speckle.",
        ),
        nr.Line(
            "reveal",
            "A faint {show:planet}planet stands out of it, "
            "{circumscribe:halo}circled here.",
        ),
    )
)


def clip(narration=None):
    tl = ep.Timeline(tempo=ep.Tempo(), narration=narration)
    part = tl.line if narration is not None else tl.section
    part("find")
    tl.enter("star")
    tl.pulse("speckle")
    part("reveal")
    tl.enter("planet", caption="a planet, 1e-10 of the star")
    tl.ring("halo")
    return tl


silent = clip().at_fps(30)
voiced = clip(nr.Narration.estimated(SCRIPT)).at_fps(30)

print(f"silent:   {silent.n_frames} frames, {len(silent.spans)} spans")
print(f"narrated: {voiced.n_frames} frames, {len(voiced.spans)} spans")
for fault in silent.faults:
    print(f"  silent {fault.severity}: {fault.message}")
for event in voiced.events(what="cue"):
    print(f"  {event.name:28s} {event.t_s:5.2f} s  frame {event.frame:3d}")
print("review frames:", voiced.review_frames())

fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6), layout="constrained", sharex=True)
for ax, plan, title in ((axes[0], silent, "silent"), (axes[1], voiced, "narrated")):
    t = np.arange(plan.n_frames) / plan.fps
    for color, knob in zip(hwostyle.palette, plan.knobs, strict=False):
        trace = [plan.value(k, knob) for k in range(plan.n_frames)]
        ax.plot(t, trace, label=knob, color=color)
    for event in plan.events(what="cue"):
        ax.axvline(event.t_s, color="0.5", lw=0.8, ls=":")
    ax.set_title(title, fontsize="small", loc="left")
axes[0].legend(ncol=5, fontsize="x-small", loc="lower right",
               bbox_to_anchor=(1.0, 1.0))
axes[1].set_xlabel("seconds")
plt.show()
```

Two writers on one knob in the same frames always raise, naming both spans
and the knob. The other checks are faults with a severity. An error raises
when the timeline is strict, the default, and is listed in `plan.faults`
otherwise: a span that rounds to no frames, text arriving while something
moves, a caption leaving before it has been read, a subject leaving while its
own caption arrives, two subjects entering at once, a mark no verb fires and
a verb whose mark the line lacks. An information fault, such as a skipped
pulse or a span clamped to the end of the clip, only records what happened.
`with_=` names an overlap that a clip allows.

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

## Photons on an exposure clock

An image that develops from its own photons shows what an exposure is: each
frame is a draw of the detector, and the eye watches a planet emerge from
the shot noise beside the leftover starlight. `arrivals` draws the photons as one seeded Poisson
process per pixel on an exposure clock, `photons` draws them as a count
image on one fixed log scale with the newest arrivals glowing as they land,
and `aperture_counts` reads the counts, the expected counts and the S/N in a
declared aperture off the image the mark shows. Here the expected images are
a charge-6 vortex coronagraph computed with physicaloptix: the star's
leftover light and a planet 3.8 lambda/D out, both in units of the
unocculted star peak, through an unobscured circular pupil and an ideal
vortex, not a mission design. The rates are illustrative: the planet's
contrast (its peak over the star's, before the coronagraph) is set high
enough to read on screen in seconds of clip, so here the planet outshines
the leak around it.

```{code-cell} python
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
from physicaloptix.elements.vortex import build_multiscale_vortex, vortex_forward
from physicaloptix.transforms.cmft import cmft_fwd

jax.config.update("jax_enable_x64", True)
NPIX, HALF = 64, 8.0  # +-8 lambda/D at 0.25 lambda/D per pixel
SEP = (3.5, 1.5)  # the planet, in lambda/D
grid, levels = build_multiscale_vortex(
    charge=6, npup=192, q=128, scaling_factor=4, window_size=16
)
X, Y = jnp.meshgrid(grid, grid)
pupil = (jnp.hypot(X, Y) <= 0.5).astype(jnp.complex128)
lyot = (jnp.hypot(X, Y) <= 0.45).astype(jnp.float64)
focal = jnp.linspace(-HALF, HALF, NPIX + 1)[:-1] + HALF / NPIX


def through(field):
    """The focal-plane intensity behind the vortex and the Lyot stop."""
    behind = vortex_forward(field, grid, levels) * lyot
    return np.asarray(jnp.abs(cmft_fwd(behind, grid, focal)) ** 2)


peak = float(jnp.abs(cmft_fwd(pupil, grid, jnp.array([0.0]))[0, 0]) ** 2)
star = through(pupil) / peak
planet = through(pupil * jnp.exp(2j * jnp.pi * (X * SEP[0] + Y * SEP[1]))) / peak
fx, fy = np.meshgrid(np.asarray(focal), np.asarray(focal))
ring = np.hypot(fx - SEP[0], fy - SEP[1]) <= 1.0
print(f"leak: max {star.max():.1e}, median {np.median(star):.1e} of the star peak")
```

The clock starts at zero exposure, an empty detector, and runs on a log
scale to 1000 seconds in 24 draws over five seconds, about five a second.
The readout is read off the mark on every frame, so it changes on the draws
and never disagrees with the image. It counts the photons in the ring, one
lambda/D in radius, beside the counts expected there, planet and leak
together, and the expected leak alone. Its S/N is that of the
signal estimate, the counts less the expected leak over the square root of
the expected counts: photon noise only, with the leak known exactly. A real
leak is speckle that drifts and must be calibrated, which adds noise this
clip does not show. The grain in the image is shot noise, not speckle. At a
few counts the S/N is not yet meaningful, and a seed can show a run of early
counts well above the expectation; this page uses seed 0, the first seed
tried, whose counts stay within two standard deviations of the expectation
throughout. The player ends on the still, the plain image of the final
counts.

```{code-cell} python
STAR_RATE, CONTRAST = 1e6, 1e-7  # photons per second at the star peak; illustrative
leak = STAR_RATE * star
clock = np.r_[0.0, np.geomspace(1.0, 1e3, 24)]
channel = ep.arrivals(leak + STAR_RATE * CONTRAST * planet, clock, seed=0)

fig = plt.figure(figsize=(6.4, 3.6))
image_ax = fig.add_axes([0.03, 0.06, 0.5, 0.88])
image_ax.set_axis_off()
mark = ep.photons(channel, ax=image_ax, extent=(-HALF, HALF, -HALF, HALF), cmap="magma")
circle = np.linspace(0.0, 2.0 * np.pi, 121)
image_ax.plot(SEP[0] + np.cos(circle), SEP[1] + np.sin(circle), color="C0", lw=1.5)
backing = {"facecolor": "black", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}
label = {"fontsize": 10, "color": "C0", "va": "center", "bbox": backing}
image_ax.text(SEP[0] + 1.3, SEP[1] + 0.9, "planet", **label)
image_ax.text(0.0, -2.6, "star leak", ha="center", **label)
mono = {"family": "DejaVu Sans Mono", "fontsize": 11, "va": "top"}
clock_text = fig.text(0.58, 0.9, "", **mono)
fig.text(0.58, 0.82, "log clock", fontsize=9, va="top", color=ep.neutral(0.6))
fig.text(0.58, 0.68, "photons in the ring", fontsize=10, va="top")
read_text = fig.text(0.58, 0.61, "", **mono)
fig.text(
    0.58,
    0.36,
    "S/N = (counted - leak) / sqrt(expected)\nphoton noise only, leak known exactly",
    fontsize=9,
    va="top",
)
fig.text(
    0.58,
    0.2,
    f"charge-6 vortex (physicaloptix)\nplanet contrast {CONTRAST:.0e}, illustrative",
    fontsize=9,
    va="top",
    color=ep.neutral(0.6),
)

tl = ep.Timeline({"u": 0.0})
tl.hold(0.4, name="open")
tl.run("draw", 5.0, to={"u": 1.0})
tl.hold(1.2, name="rest")


def draw_photons(k, plan):
    mark.update(plan.value(k, "u"))
    read = ep.aperture_counts(mark, ring, leak=leak)
    clock_text.set_text("t = " + ep.exposure_text(read.exposure, pad=True))
    snr = f"{read.snr:>6.1f}" if read.expected > 0 else f"{'--':>6}"
    read_text.set_text(
        f"counted  {read.counts:>6,d}\n"
        f"expected {read.expected:>6,.0f}\n"
        f"leak     {read.leak:>6,.0f}\n"
        f"S/N      {snr}"
    )


# An embedded mp4 at 1280 x 720: the photon grain does not compress as frames.
plan = tl.at_fps(30)
draw_photons(plan.n_frames - 1, plan)
movie = ep.animate(fig, lambda f, k: draw_photons(k, plan), plan.n_frames, fps=30)
with tempfile.TemporaryDirectory() as tmp_dir:
    # The player embeds the file when it renders, so render it before cleanup.
    player = HTML(movie.video(Path(tmp_dir) / "photons.mp4", dpi=200)._repr_html_())
player
```

## Overshoot and anticipation

An overshoot ease is for a move whose arrival should read as a landing. Only
the input progress is clipped, never the eased value, so a move that places a
mark with it can carry the mark past its target before it settles. A move
that uses the eased value as an opacity, a fraction of a line, or a color
blend clips it first, so `link`, `circumscribe`, `Stage`, `collapse` and
`morph` accept these kinds and still draw valid colors and lengths. The
player below flies a dot from one panel to the next under `back_out`; the dot
passes the landing mark and comes back to it, and the last frame shows the
landed state exactly.

```{code-cell} python
fig, (left, right) = plt.subplots(1, 2, figsize=(5.6, 2.2), layout="constrained")
for axis in (left, right):
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xticks=[], yticks=[])
(read,) = left.plot([0.5], [0.5], "o", ms=9)
(landed,) = right.plot([0.5], [0.5], "o", ms=9, color=read.get_color())
fig.canvas.draw()
fig.set_layout_engine("none")

bounce = ep.flight(read, ep.Anchor.from_artist(landed), fig=fig, ease="back_out")
N_FLIGHT = 28


def draw_flight(fig, k):
    u = k / (N_FLIGHT - 1)
    bounce.update(u)
    landed.set_visible(u >= 1.0)


draw_flight(fig, 0)
HTML(ep.animate(fig, draw_flight, N_FLIGHT, fps=14).jshtml(dpi=80))
```
