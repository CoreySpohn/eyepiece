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

import eyepiece as ep

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small.
plt.rcParams["savefig.dpi"] = 110

N = 96
PIXSCALE_LOD = 0.3
EXTENT = ep.plot.extent_lod_from_pixels(N, PIXSCALE_LOD)
u = (np.arange(N) - (N - 1) / 2.0) * PIXSCALE_LOD
x, y = np.meshgrid(u, u)

# A companion at 8 lambda/D, the target of the zoom below.
PLANET_XY = (8.0 * np.cos(2.4), 8.0 * np.sin(2.4))
```

## A timeline

A `Timeline` declares named spans in seconds. A span is the one primitive: a
hold, a subject entering or leaving, a move, a caption arriving, an
indication, an outline. Each span writes the knobs it names, and
`Timeline.at_fps` derives every knob on every frame from them, so a draw
function never sees time. A hold may step knobs when it starts, a new title
for instance; a move ramps numeric knobs (linearly, or in log space for widths
and other scales) and changes nothing else, so text never arrives during
motion. `read` sizes a hold from the words on screen by the reading rule of
`Tempo`. Resolving the timeline for a frame rate gives a `Plan`, whose frames
are plain dicts, one per frame, and whose `progress` reads any span's clock.
Here the zoom span writes no knob: its clock drives a camera path in the
{doc}`camera catalog <motion-camera>`.

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

plan = tl.at_fps(24)
for span in plan.spans.values():
    print(f"{span.name:6s} {span.kind:5s} frames {span.k0:3d}-{span.k1 - 1:3d}")
k = plan.frame_of("zoom", "mid")
s = plan.progress(k, "zoom")
print(f"frame {k}: zoom progress {s:.2f}, view width {path(s)[2]:.2f} lambda/D")
```

Every frame of a span has moved: a span from `s` to `e` seconds covers frames
`ep.clock.frame_count(s)` up to `max(k0 + 1, ep.clock.frame_count(e))`, frame `i` of its `n`
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
    axis.plot(s, s, color=ep.motion.blend(plt.rcParams["text.color"], 0.3), lw=1.0)
    axis.plot(s, ep.clock.ease(s, kind), color=hwostyle.palette[0])
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
`ep.clock.Timeline.enter("star")` ramps `star` from 0 to 1, `ep.clock.Timeline.indicate("speckle")` ramps
`indicate.speckle`, and a caption writes `caption.<name>`.

Given a `Narration`, the same function lays each span on the word that names
it. A mark `{show:star}` starts `ep.clock.Timeline.enter("star")` a tenth of a second before the
word, `{indicate:speckle}` centers the indication on its word, and
`{circumscribe:halo}` starts the outline on it. Lines open in script order
with `line`, and a line lasts until its speech and padding end or its last
moving span ends plus 0.6 s, whichever is later. Run without a narration, the
function builds the silent clip: sections replace lines, a caption leaves once
it has been read, and an indication or outline that only a cue places is
skipped and recorded in `plan.faults` as information. The plan also lists its
events (starts, middles, ends and cues, each with the frame that shows it) and
the frames a contact sheet should draw.

```{code-cell} python
import eyepiece as ep

SCRIPT = ep.narration.Script(
    (
        ep.narration.Line(
            "find",
            "The coronagraph blocks the {show:star}star, and what leaks "
            "through is {indicate:speckle}speckle.",
        ),
        ep.narration.Line(
            "reveal",
            "A faint {show:planet}planet stands out of it, "
            "{circumscribe:halo}circled here.",
        ),
    )
)


def clip(narration=None):
    tl = ep.clock.Timeline(tempo=ep.clock.Tempo(), narration=narration)
    part = tl.line if narration is not None else tl.section
    part("find")
    tl.enter("star")
    tl.indicate("speckle")
    part("reveal")
    tl.enter("planet", caption="a planet, 1e-10 of the star")
    tl.circumscribe("halo")
    return tl


silent = clip().at_fps(30)
voiced = clip(ep.narration.Narration.estimated(SCRIPT)).at_fps(30)

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

## Hold until a computed time

Some holds end at a time the clip computes rather than a duration it states,
such as the moment a label has been read. `wait_until` moves the cursor to
that time without placing a span, so no knob is written and the open section
simply lasts until then. Here the label lands at 0.4 s, `Tempo.read_s` sizes
its reading time from its words, and the next section starts once the label
has been read.

```{code-cell} python
tempo = ep.clock.Tempo(factor=1.0)
tl = ep.clock.Timeline({"label": 0.0, "next": 0.0}, tempo=tempo)
tl.section("label")
label = tl.text("label")
tl.wait_until(label.end + tempo.read_s("Down to the companion"))
tl.section("next")
tl.enter("next")

plan = tl.at_fps(24)
for span in plan.spans.values():
    print(f"{span.name:6s} {span.start_s:5.2f} to {span.end_s:5.2f} s")
print(f"{plan.seconds:.2f} s in {plan.n_frames} frames")
```

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
ax.plot([0.0, 12.5], [-1.0, 1.0], color=ep.style.neutral(0.4))
(dot,) = ax.plot([0.0], [-1.0], "o", ms=8, color="C1")
fig.canvas.draw()
fig.set_layout_engine("none")

clock = ep.motion.readout(
    ep.motion.Anchor.from_data(ax, (2.4, 0.78)), 0.0, prefix="t = ", unit="h", ha="point"
)
offset = ep.motion.readout(
    ep.motion.Anchor.from_data(ax, (2.4, 0.52)),
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
        e = ep.clock.ease(u)
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

# Double precision: the leak sits eight decades below the star peak.
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
clip does not show. The grain in the image is shot noise, not speckle.

The seed is chosen on two criteria: an endpoint near the expectation and a
trajectory that does not wander against the claim. Seed 7, tried first, put
6 photons in the ring after one second against 1.1 expected, a false early
detection at 4.7 standard deviations that the cumulative counts carried
forward. Seed 0, tried next, ends at S/N 26.8 against 27.3 expected, and its
ring counts stay within two standard deviations of the expectation at every
draw. The final counts depend only on the seed and the final exposure, but
the earlier frames also depend on the clock samples, so a seed pins the whole
clip only for a fixed clock. The player ends on the still, the plain image of
the final counts.

```{code-cell} python
STAR_RATE, CONTRAST = 1e6, 1e-7  # photons per second at the star peak; illustrative
leak = STAR_RATE * star
clock = np.r_[0.0, np.geomspace(1.0, 1e3, 24)]
channel = ep.motion.arrivals(leak + STAR_RATE * CONTRAST * planet, clock, seed=0)

fig = plt.figure(figsize=(6.4, 3.6))
image_ax = fig.add_axes([0.03, 0.06, 0.5, 0.88])
image_ax.set_axis_off()
mark = ep.motion.photons(channel, ax=image_ax, extent=(-HALF, HALF, -HALF, HALF), cmap="magma")
circle = np.linspace(0.0, 2.0 * np.pi, 121)
image_ax.plot(SEP[0] + np.cos(circle), SEP[1] + np.sin(circle), color="C0", lw=1.5)
backing = {"facecolor": "black", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}
label = {"fontsize": 10, "color": "C0", "va": "center", "bbox": backing}
image_ax.text(SEP[0] + 1.3, SEP[1] + 0.9, "planet", **label)
image_ax.text(0.0, -2.6, "star leak", ha="center", **label)
mono = {"family": "DejaVu Sans Mono", "fontsize": 11, "va": "top"}
clock_text = fig.text(0.58, 0.9, "", **mono)
fig.text(0.58, 0.82, "log clock", fontsize=9, va="top", color=ep.style.neutral(0.6))
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
    color=ep.style.neutral(0.6),
)

tl = ep.clock.Timeline({"u": 0.0})
tl.hold(0.4, name="open")
tl.run("draw", 5.0, to={"u": 1.0})
tl.hold(1.2, name="rest")


def draw_photons(k, plan):
    mark.update(plan.value(k, "u"))
    read = ep.motion.aperture_counts(mark, ring, leak=leak)
    clock_text.set_text("t = " + ep.motion.exposure_text(read.exposure, pad=True))
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
movie = ep.render.animate(fig, lambda f, k: draw_photons(k, plan), plan.n_frames, fps=30)
with tempfile.TemporaryDirectory() as tmp_dir:
    # The player embeds the file when it renders, so render it before cleanup.
    player = HTML(movie.video(Path(tmp_dir) / "photons.mp4", dpi=200)._repr_html_())
player
```

## Orbit draws on the sky

A fitted orbit is uncertain, and a single nominal ellipse hides how
uncertain. `hop` shows the fit as what it is, a set of equally likely orbits
for one planet: it plays draws of the orbit one after another at a steady
rate, the current draw in full and the draws before it as ghosts that dim
with age, and it ends on a tally of one question asked of every draw. Here
the question is where the planet will be at t, one year after the element
epoch T0 (the epoch at which the mean anomaly is given): the dot on each
orbit is its position at t, the same date for every draw (each orbit is
drawn for one of its own periods starting at t, so the dot is where its
curve begins), and the tally is the fraction of draws that put
the planet within 30 mas of the nominal position at t (the cross, inside the
circle). The nominal orbit (dashed) is the orbit of the mean elements, not a
fitted best: the draws' own median position at t sits about 2 mas from it.

The fit is a toy, and its numbers are illustrative: a planet at 1.2 AU
around a solar-mass star 10 pc away, with a diagonal Gaussian covariance on
the six elements (semi-major axis, eccentricity, node, inclination, argument
of periapsis and mean anomaly at the reference epoch). A real fit has
correlated elements and a non-Gaussian posterior; the mark shows whatever
draws it is given, and the fraction it ends on depends entirely on the
invented spreads, so it is no forecast for any real planet. Offsets are in
mas because the star's distance turns AU into angle (1 AU at 10 pc is 100
mas). Every orbit, the draws, the nominal orbit and the reference, is computed with orbix's `KeplerianOrbit`, vmapped over the draws. Draws
with an eccentricity outside [0, 1) are left out: none of the 40 in the
clip, 2 of the 20,000 in the reference.

The figure follows orbix's sky convention, which is not the astrometric one.
An astrometric fit measures the node angle from north through east, and its
Thiele-Innes projection gives the planet's offsets north (Dec) and east (RA).
For the same elements, the coordinate orbix's `position_arcsec` returns as RA
is that projection's Dec, and the one it returns as Dec is its RA: orbix's
(RA, Dec) is the astrometric (Dec, RA). That is a reflection about the
diagonal, not a rotation, so the drawn orbit's orientation and its sense of
motion both differ from those of the orbit a published fit describes, and
elements taken from such a fit need converting before they are drawn this
way. The tally does not depend on it: a reflection keeps every distance, so
the fraction of draws within 30 mas is the same in either convention. RA
increases to the left, as orbix's sky plots draw it.

```{code-cell} python
from hwoutils.constants import G, Msun2kg
from matplotlib.transforms import Bbox
from orbix.equations.orbit import mean_motion, period_n
from orbix.orbit import KeplerianOrbit

jax.config.update("jax_enable_x64", True)  # Julian dates need double precision
DEG = np.pi / 180.0
# The toy fit: a (AU), e, W, i, w and M0 (radians) at T0, and their spreads.
MEAN = np.array([1.2, 0.15, 40 * DEG, 55 * DEG, 110 * DEG, 200 * DEG])
SIGMA = np.array([0.04, 0.04, 4 * DEG, 4 * DEG, 12 * DEG, 10 * DEG])
MS_KG, DIST_PC, T0 = Msun2kg, 10.0, 2461000.0
EPOCH = T0 + 365.25  # t, the epoch asked about: one year after T0
# Seed 2: of seeds 0 to 11, the first whose final fraction is within one draw
# of the 20,000-draw answer and whose interval holds it at every draw (the
# rule was set after seeing all twelve runs). It takes the first such seed,
# not the best: seed 4 ends closer, 0.6 points under, and wanders less.
WITHIN_MAS, N_DRAWS, SEED = 30.0, 40, 2


def sample(n, seed):
    """Element sets from the toy covariance, those with e outside [0, 1) left out."""
    rng = np.random.default_rng(seed)
    elements = MEAN + SIGMA * rng.standard_normal((n, 6))
    return elements[(elements[:, 1] >= 0.0) & (elements[:, 1] < 1.0)]


def at_epoch(elements):
    """Sky positions (RA, Dec offsets in mas) at EPOCH, one row per orbit."""
    orbits = KeplerianOrbit(*elements.T, np.full(len(elements), T0))
    ra, dec = orbits.position_arcsec(
        t_jd=jnp.array([EPOCH]), Ms_kg=MS_KG, dist_pc=DIST_PC
    )
    return 1e3 * np.column_stack([ra[:, 0], dec[:, 0]])


def one_orbit(elements, t_jd):
    orbit = KeplerianOrbit(*elements, T0)
    ra, dec = orbit.position_arcsec(t_jd=t_jd, Ms_kg=MS_KG, dist_pc=DIST_PC)
    return 1e3 * ra[0], 1e3 * dec[0]


def sky_orbits(elements):
    """Each orbit over one of its own periods from EPOCH, shaped (n, 181, 2)."""
    period = period_n(mean_motion(jnp.asarray(elements[:, 0]), G * MS_KG))
    t_jd = EPOCH + period[:, None] * jnp.linspace(0.0, 1.0, 181)
    ra, dec = jax.vmap(one_orbit)(jnp.asarray(elements), t_jd)
    return np.stack([np.asarray(ra), np.asarray(dec)], axis=-1)


nominal = at_epoch(MEAN[None])[0]
nominal_orbit = sky_orbits(MEAN[None])[0]
reference = sample(20_000, seed=12345)
p_ref = float(np.mean(np.hypot(*(at_epoch(reference) - nominal).T) <= WITHIN_MAS))
elements = sample(N_DRAWS, seed=SEED)
orbits = sky_orbits(elements)
within = np.hypot(*(orbits[:, 0] - nominal).T) <= WITHIN_MAS
print(f"{len(reference):,} reference draws: {p_ref:.1%} within {WITHIN_MAS:g} mas")
print(f"{len(elements)} clip draws: {within.mean():.1%} within {WITHIN_MAS:g} mas")
```

The draws play at five a second, slow enough to see each orbit as its own
answer, and the tally beside them is the running fraction with its 95%
Wilson interval at the current count, against the fraction over the 19,998
reference draws kept of 20,000. The tally counts every draw played, not only the seven on screen, so
the current dot can sit outside the circle while the fraction stays above
half; the first two draws both miss it, so the line starts flat at zero.
At one draw the interval spans most of the axis, as it should: one
draw says almost nothing. With 40 draws it is still about thirty points wide: the clip
shows how the tally settles, and the reference line is the answer a longer
run would give.

The seed is chosen on two criteria: an endpoint near the expectation and a
trajectory that does not wander against the claim. Of seeds 0 to 11, the rule
takes the first whose final fraction is within one draw (1/40) of the
reference and whose running interval holds the reference at every draw, a
rule set after seeing all twelve runs. Seeds 0 and 1 end at 60%, 1.8 draws
above the reference of 55.6%, so seed 2, at 57.5%, is taken. It is the first
that passes, not the best: seed 4 ends 0.6 points under the reference and
strays less on the way, and only seed 8 ever leaves the reference outside its
interval.
The player ends on the still, the final tally.

```{code-cell} python
fig = plt.figure(figsize=(6.4, 3.6))
sky = fig.add_axes([0.12, 0.16, 0.40, 0.63])
panel = fig.add_axes([0.59, 0.30, 0.25, 0.34])
foot = fig.add_axes([0.59, 0.02, 0.25, 0.08])  # holds the draw counter
foot.set_axis_off()
foot.set_xlim(0.0, 1.0)
foot.set_ylim(0.0, 1.0)
grey = ep.style.neutral(0.45)
note = {"color": ep.style.neutral(0.7), "fontsize": 8}
sky.plot(0.0, 0.0, "*", color=grey, ms=9)
sky.plot(*nominal_orbit.T, color=grey, lw=0.8, ls="--")
circle = np.linspace(0.0, 2.0 * np.pi, 121)
sky.plot(*(nominal + WITHIN_MAS * np.column_stack([np.cos(circle), np.sin(circle)])).T,
         color=grey, lw=1.0)  # fmt: skip
sky.plot(*nominal, "+", color=grey, ms=11, mew=1.4)
sky.set_aspect("equal")
sky.invert_xaxis()  # RA increasing to the left, as orbix draws the sky
sky.set_xlabel("RA offset (mas)")
sky.set_ylabel("Dec offset (mas)")
sky.set_title("one planet, 40 orbits it could be on, the last 7 shown\n"
              "(bright: this draw, faint: earlier ones)\n"
              "dot: where each orbit puts the planet at t, one date for all:\n"
              "a year after T0, the date of the orbital elements",
              fontsize=8)  # fmt: skip
panel.set_xlabel("draws\nline: running fraction within 30 mas\nbar: 95% interval of that fraction",
                 fontsize=8)  # fmt: skip
panel.set_ylabel(f"fraction within {WITHIN_MAS:g} mas", fontsize=8)
mark = ep.plot.hop(
    orbits,
    ax=sky,
    rate=5.0,
    ghosts=6,
    ghost_alpha=0.6,
    line_kw={"marker": "o", "markevery": [0], "ms": 5, "lw": 1.2},
    counter_at=ep.motion.Anchor.from_data(foot, (0.0, 0.5)),
    tally=within,
    tally_ax=panel,
    ref=p_ref,
    tally_fmt=lambda p, lo, hi, k: (
        f"position on the sky at t:\nwithin {WITHIN_MAS:g} mas of the nominal (+)\n"
        f"in {p:.1%} of draws\n"
        f"95% interval {lo:.0%} to {hi:.0%}"
        + (
            f"\n{'holds' if lo <= p_ref <= hi else 'misses'} the dashed answer, {p_ref:.1%}"
            if k == N_DRAWS
            else ""
        )
    ),
    text_kw={"fontsize": 9},
)
# A thin interval, so at the first draws it reads as an error bar, not a bar.
mark.tally_artists["lines"][0].set_linewidth(1.0)
# Scenery labels, placed from the computed geometry so they track the data.
sky.annotate("star", (0.0, 0.0), xytext=(6, -10), textcoords="offset points", **note)
# The labels sit inside the orbits, where no draw passes, with arrows out.
arrow = {"arrowstyle": "-", "color": note["color"], "lw": 0.6}
left = nominal_orbit[np.argmax(nominal_orbit[:, 0])]
sky.annotate("nominal orbit\n(dashed)", left, xytext=(35.0, 35.0), ha="center",
             va="center", arrowprops=arrow, **note)  # fmt: skip
sky.annotate(f"circle: {WITHIN_MAS:g} mas\naround +, the\nnominal at t",
             nominal + [WITHIN_MAS, 0.0], xytext=(80.0, -112.0), ha="center",
             va="center", arrowprops=arrow, **note)  # fmt: skip
panel.annotate(f"dashed:\nanswer from\n{len(reference):,}\ndraws: {p_ref:.1%}", (1.0, p_ref),
               xycoords=("axes fraction", "data"), xytext=(8, 0),
               textcoords="offset points", va="center", **note)  # fmt: skip
fig.text(0.98, 0.97, "toy covariance, illustrative", ha="right", va="top", **note)
# The counter and the panel's key are stacked from their drawn extents, not
# fixed offsets, so they cannot meet in whatever font the reader has: the counter
# sits one pad above the bottom edge, and the panel's foot rises to clear it.
fig.draw_without_rendering()
to_fig = fig.transFigure.inverted()
pad = 6.0 / 72.0 / fig.get_figheight()  # 6 pt, as a figure fraction
count = Bbox.union([t.get_window_extent() for t in mark.artists["text"]]).transformed(to_fig)
foot.set_position(foot.get_position().translated(0.0, pad - count.y0))
key_drop = panel.get_position().y0 - panel.get_tightbbox().transformed(to_fig).y0
top = panel.get_position().y1
bottom = 2.0 * pad + count.height + key_drop
panel.set_position([0.59, bottom, 0.25, top - bottom])
# The sky's title and axis label are measured too: the sky takes the height left
# between them and the edges, and its equal-aspect box shrinks if a face needs it.
box = sky.get_position()  # the drawn box, after the equal aspect
words = sky.get_tightbbox().transformed(to_fig)
low = pad + box.y0 - words.y0
high = 1.0 - pad - (words.y1 - box.y1)
sky.set_position([0.12, low, 0.40, high - low])

tl = ep.clock.Timeline({"u": 0.0})
tl.hold(0.5, name="open")
tl.run("hop", mark.duration, to={"u": 1.0})
tl.hold(1.5, name="rest")


def draw_orbits(k, plan):
    mark.update(plan.value(k, "u"))


plan = tl.at_fps(30)
draw_orbits(plan.n_frames - 1, plan)
movie = ep.render.animate(fig, lambda f, k: draw_orbits(k, plan), plan.n_frames, fps=30)
with tempfile.TemporaryDirectory() as tmp_dir:
    # The player embeds the file when it renders, so render it before cleanup.
    player = HTML(movie.video(Path(tmp_dir) / "orbits.mp4", dpi=200)._repr_html_())
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

bounce = ep.motion.flight(read, ep.motion.Anchor.from_artist(landed), fig=fig, ease="back_out")
N_FLIGHT = 28


def draw_flight(fig, k):
    u = k / (N_FLIGHT - 1)
    bounce.update(u)
    landed.set_visible(u >= 1.0)


draw_flight(fig, 0)
HTML(ep.render.animate(fig, draw_flight, N_FLIGHT, fps=14).jshtml(dpi=80))
```

## Declare an instant cut

A cut changes state on one frame and consumes no time. It may be placed by
`at` or an explicit narration `cue`. `check_clip` reads its declaration from
the plan, so the jump is accepted with the other checks still active.

```python
import eyepiece as ep

tl = ep.clock.Timeline({"panel": 0})
tl.hold(0.5)
tl.cut("switch", to={"panel": 1})
tl.hold(0.5)
plan = tl.at_fps(30)
assert plan.value(14, "panel") == 0
assert plan.value(15, "panel") == 1
assert plan.step_frames() == (15,)
```
