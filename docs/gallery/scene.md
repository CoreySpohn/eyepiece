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

# Scenes

The primitives on this page draw geometry rather than intensity: where a
source went, where it might have gone, and which plane of an instrument a
panel is showing. Their inputs are position arrays and plain `(label, glyph)`
pairs, so nothing here knows what an orbit or an optical element is, and the
positions below are generated in the page from closed-form expressions and
seeded random draws.

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


def loop(a, ecc, inc_deg, node_deg, n=240):
    """A closed inclined ellipse, as an `(n, 3)` array of positions in AU."""
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    radius = a * (1.0 - ecc**2) / (1.0 + ecc * np.cos(theta))
    inc, node = np.deg2rad(inc_deg), np.deg2rad(node_deg)
    xp, yp = radius * np.cos(theta), radius * np.sin(theta)
    xi, yi, zi = xp, yp * np.cos(inc), yp * np.sin(inc)
    return np.column_stack(
        [
            xi * np.cos(node) - yi * np.sin(node),
            xi * np.sin(node) + yi * np.cos(node),
            zi,
        ]
    )
```

## A path, flat and in depth

`trail` draws a connected line through an `(N, 2)` or `(N, 3)` array of
positions and puts a marker on every point. The two panels below are the
same inclined ellipse: the left one hands over its first two columns and
gets a flat projection with every marker the same size, and the right one
hands over all three and gets the depth cues that a three-dimensional path
needs to be readable at all.

A closed curve in 3D is ambiguous on its own: nothing about the outline says
which half is nearer the camera, and for an orbit that half is the difference
between a planet in front of its star and behind it. `depth` picks how the
path answers that.

The default `"hidden"` borrows the engineering-drawing convention. The whole
path is drawn dim and dashed, and the near half is overdrawn solid, so the
boundary falls exactly where the path crosses the plane containing the line
of sight. `"markers"` is the older treatment: a marker on every point, scaled
by `(1 + cos(viewer_angle)) / 2`, so the far side nearly vanishes. `"none"`
draws the bare path.

The default changed because the markers were measured against the
alternative on a three-planet system: they cost 2.6 times the ink and hid
22.6 percent of the orbits drawn behind them, against 9.1 percent, for the
same one fact. Reach for `"markers"` when the per-point sampling is itself
the subject -- an uneven observing cadence, say -- rather than as the way to
show depth.

Either way the camera angles come from the axes itself, which is why a 3D
path passed to a plain axes raises rather than silently flattening: `trail`
would have no viewing direction to work from. Because those angles are read
when the call is made, set the view first.

The same path under all three, with two orbits so the occlusion cost is
visible rather than asserted:

```{code-cell} python
path = loop(1.4, 0.25, 55.0, 20.0, n=36)
inner = loop(0.8, 0.10, 50.0, 60.0, n=36)

# Both names from ONE SourceStyles: a fresh one restarts the palette, so two
# separate calls would hand back the same colour twice.
pair = ep.SourceStyles(["outer", "inner"])

fig = plt.figure(figsize=(11.4, 3.6), layout="constrained")
for i, mode in enumerate(["hidden", "markers", "none"]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.view_init(elev=30, azim=10)
    ep.trail(inner, ax=ax, depth=mode, marker_scale=60.0, style=pair["inner"])
    ep.trail(path, ax=ax, depth=mode, marker_scale=80.0, style=pair["outer"])
    ax.set_title(f'depth="{mode}"')
```

Under `"markers"` the sizes are on the returned collection, so the cue stays
inspectable rather than baked into a rendering pass; a 2D path has no depth
and takes `marker_scale` flat.

```{code-cell} python
fig = plt.figure(figsize=(7.8, 3.4), layout="constrained")
ax_flat = fig.add_subplot(1, 2, 1)
ax_deep = fig.add_subplot(1, 2, 2, projection="3d")
ax_deep.view_init(elev=30, azim=10)

flat = ep.trail(path[:, :2], ax=ax_flat, marker_scale=14.0)
deep = ep.trail(path, ax=ax_deep, depth="markers", marker_scale=80.0)

ax_flat.set_aspect("equal")
ax_flat.set_xlabel("$x$ [AU]")
ax_flat.set_ylabel("$y$ [AU]")
ax_flat.set_title("(N, 2) positions")
ax_deep.set_title('(N, 3), depth="markers"')

sizes = deep.artists["scatter"].get_sizes()
print(f"3D marker sizes: {sizes.min():.1f} to {sizes.max():.1f}")
print("2D marker sizes:", np.unique(flat.artists["scatter"].get_sizes()))
```

### Naming the source instead of the color

`style` takes a color, but it also takes a `SourceStyles` entry, which is
what a trajectory usually wants: the entry carries the marker as well, so a
source drawn here matches how it is drawn in every other panel of the figure
without the call site pulling the pair apart. That is the whole linked-views
idea applied to one primitive -- see [one scene, N views](one-scene-n-views)
for the full case.

```{code-cell} python
styles = ep.SourceStyles(["star", "b", "c"])

fig = plt.figure(figsize=(4.2, 3.6), layout="constrained")
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=22, azim=-125)
for name, tilt in (("b", 40.0), ("c", 70.0)):
    ep.trail(loop(1.2, 0.2, tilt, 15.0, n=30), ax=ax,
             style=styles[name], marker_scale=45.0)
ax.set_title("one entry, color and marker together")
```

The color and the marker both come from the entry, so this panel and any
other panel built from the same `SourceStyles` agree on what planet b looks
like.

```{code-cell} python
for name in ("b", "c"):
    print(name, styles[name])
```

## A fan of candidate tracks

`sky_fan` draws many candidate paths through the same sky plane and fades
each one by its weight, which is how a set of tracks consistent with a few
measured positions is shown without committing to one of them. Each weight
below is the exponential of minus half the mean squared standardized
residual of that candidate against the three plotted epochs, so the tracks
that actually pass through the measurements are the ones that stay visible
and the rest recede toward the background.

Passing one color for every track, rather than letting each take the next
palette color, is what makes weight the only visual variable in the figure.
The color here is read from a `SourceStyles` built on the source name, which
is the same mechanism the {doc}`linked views <one-scene-n-views>` page uses
to keep several panels agreeing about a source. The `iwa` disk marks the
region a coronagraph cannot see into, and `data` overlays the observed
epochs as symmetric errorbars, both drawn in neutral tones taken from the
active rcParams so the scenery stays legible in either mode.

```{code-cell} python
rng = np.random.default_rng(4)
truth = loop(1.4, 0.25, 55.0, 20.0, n=400)
epoch_index = np.array([20, 120, 250])
sigma = 0.06
observed = truth[epoch_index, :2]

tracks, weights = [], []
for _ in range(60):
    candidate = loop(
        1.4 + rng.normal(0.0, 0.18),
        np.clip(0.25 + rng.normal(0.0, 0.08), 0.0, 0.6),
        55.0 + rng.normal(0.0, 10.0),
        20.0 + rng.normal(0.0, 10.0),
        n=400,
    )
    residual = (candidate[epoch_index, :2] - observed) / sigma
    tracks.append((candidate[:, 0], candidate[:, 1]))
    weights.append(float(np.exp(-0.5 * np.mean(residual**2))))

color = ep.SourceStyles(["candidate"])["candidate"]["color"]

fig, ax = plt.subplots(figsize=(4.8, 4.4), layout="constrained")
fan = ep.sky_fan(
    tracks,
    ax=ax,
    colors=[color] * len(tracks),
    weights=weights,
    iwa=0.45,
    data=(observed[:, 0], observed[:, 1], np.full(len(observed), sigma)),
    fan_kw={"lw": 1.0},
)
ax.set_xlabel(r"$\Delta$RA [AU]")
ax.set_ylabel(r"$\Delta$Dec [AU]")
```

Each kind of scenery comes back under its own key, so a caller can restyle
the working-angle disk or the errorbars without redrawing the fan.

```{code-cell} python
print(sorted(fan.artists), len(fan.artists["lines"]), "tracks")
print("alpha range:", round(min(ln.get_alpha() for ln in fan.artists["lines"]), 4),
      "to", round(max(ln.get_alpha() for ln in fan.artists["lines"]), 4))
```

## A track that fades into its own past

`fading_track` draws one path whose opacity ramps from `alpha_range[0]` at
the tail to `alpha_range[1]` at the head, which is how a moving object shows
where it has just been without the whole history competing with its current
position. The ramp is built as an explicit per-segment RGBA array rather
than deferred to a colormap, so the alpha a caller reads back off the
`LineCollection` is the alpha that was drawn.

The path below spirals inward over three turns, and the head marker is the
caller's own, drawn on the axes the result hands back.

```{code-cell} python
t = np.linspace(0.0, 6.0 * np.pi, 500)
radius = np.exp(-0.06 * t)
spiral = np.column_stack([radius * np.cos(t), radius * np.sin(t)])

fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
fade = ep.fading_track(spiral, ax=ax, alpha_range=(0.05, 1.0),
                       collection_kw={"lw": 2.0})
ax.plot(spiral[-1, 0], spiral[-1, 1], marker="o", ms=6,
        color=fade.artists["collection"].get_colors()[-1][:3])
ax.set_aspect("equal")
ax.set_xlabel("$x$ [AU]")
ax.set_ylabel("$y$ [AU]")
```

```{code-cell} python
alphas = fade.artists["collection"].get_colors()[:, 3]
print(f"{len(alphas)} segments, alpha {alphas[0]:.2f} at the tail to "
      f"{alphas[-1]:.2f} at the head")
```

## The glyph vocabulary

`rail` builds a miniature optical train from a list of `(label, glyph)`
pairs, where the glyph is one of the eight names in `GLYPHS`. The names are
this library's own vocabulary, chosen to say what to draw rather than to
match any simulation package's class names, and the mapping they carry is
whether the beam is wide at that plane or pinched. The envelope in every
rail is read straight off that mapping, so a train opens after a pupil and
closes into a focus without the caller positioning anything.

```{code-cell} python
print({name: "pupil-like" if wide else "image-like"
       for name, wide in sorted(ep.GLYPHS.items())})
```

The rail below is assembled to show all eight glyphs once rather than to
describe a real instrument. A lens is drawn after every plane but the last,
the planes are spaced evenly because no `positions` were given, and the
train is capped with a detector block only when it ends on a `focal` plane,
which this one does not.

```{code-cell} python
vocabulary = [
    ("Source", "source"),
    ("Pupil", "pupil"),
    ("Apodizer", "apodizer"),
    ("FPM", "fpm"),
    ("Lyot", "lyot"),
    ("Mask", "mask"),
    ("Focal", "focal"),
    ("Detector", "detector"),
]

fig, ax = plt.subplots(figsize=(9.2, 2.2), layout="constrained")
glyphs = ep.rail(vocabulary, ax=ax, highlight="FPM")
```

## You are here

The point of the rail is the highlight, because a figure that shows a field
at some plane of an instrument otherwise leaves the reader to work out which
plane that is. `highlight` takes a plane's label, matched
case-insensitively, and draws that plane's marker, glyph, and label in the
accent color, leaving everything else neutral. A label that is not in the
train raises rather than quietly matching nothing.

The four panels below are one Lyot coronagraph propagated with NumPy FFTs, a
circular pupil to its focal plane, an opaque occulter of radius three lambda
over D, back to the Lyot pupil, through an undersized stop, and on to the
final focal plane. Each panel carries the same preset rail underneath it
with its own plane picked out, so the reader never has to hold the train in
their head.

```{code-cell} python
M = 256
D_PIX = 64.0
PIX_PER_LOD = M / D_PIX

q = (np.arange(M) - M // 2) / (D_PIX / 2.0)
qx, qy = np.meshgrid(q, q)
rho = np.hypot(qx, qy)
pupil = (rho <= 1.0).astype(float)

s = (np.arange(M) - M // 2) / PIX_PER_LOD
sx, sy = np.meshgrid(s, s)
sep = np.hypot(sx, sy)


def to_focal(field):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field))) / M


def to_pupil(field):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field))) * M


focal = to_focal(pupil)
lyot_plane = to_pupil(focal * (sep > 3.0))
lyot_stop = (rho <= 0.85).astype(float)
final = to_focal(lyot_plane * lyot_stop)

peak = float((np.abs(focal) ** 2).max())
print(f"on-axis suppression: {float((np.abs(final) ** 2).max()) / peak:.2e}")
```

```{code-cell} python
def crop(a, half):
    c = M // 2
    return a[c - half : c + half, c - half : c + half]


PUP_HALF, FOC_HALF = 44, 60

panels = [
    ("Pupil", crop(np.abs(pupil), PUP_HALF), False),
    ("FPM", crop(np.abs(focal) ** 2 / peak, FOC_HALF), True),
    ("Lyot", crop(np.abs(lyot_plane * lyot_stop), PUP_HALF), False),
    ("Focal", crop(np.abs(final) ** 2 / peak, FOC_HALF), True),
]

fig, axes = plt.subplots(
    4, 2, figsize=(8.2, 7.0), height_ratios=[3, 1.5, 3, 1.5], layout="constrained"
)
slots = [(axes[0, 0], axes[1, 0]), (axes[0, 1], axes[1, 1]),
         (axes[2, 0], axes[3, 0]), (axes[2, 1], axes[3, 1])]

for (plane, data, is_focal), (ax_img, ax_rail) in zip(panels, slots, strict=True):
    if is_focal:
        # no extent, so imshow_log drops the index ticks on its own
        ep.imshow_log(data, ax=ax_img, floor=1e-8, vmax=1.0, colorbar=False)
    else:
        ax_img.imshow(data, origin="lower", interpolation="nearest", cmap="magma")
        ax_img.set_xticks([])  # a raw imshow still needs asking
        ax_img.set_yticks([])
    ax_img.set_title(plane)
    ep.schematic("coronagraph", ax=ax_rail, highlight=plane)
```

`schematic` is the preset wrapper used above, and it is nothing more than
`rail` called with a stored plane list and hand-tuned positions for the two
trains that come up constantly. An imager is a pupil and a focal plane, and
a coronagraph adds the focal-plane mask and the Lyot pupil between them.
Anything else is a list of pairs and a call to `rail`.

```{code-cell} python
fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.0), layout="constrained")
imager = ep.schematic("imager", ax=axes[0], highlight="focal")
corona = ep.schematic("coronagraph", ax=axes[1], highlight="lyot")
print([t.get_text() for t in imager.artists["text"]])
print([t.get_text() for t in corona.artists["text"]])
```

The rail returns its per-plane markers and labels in plane order under
`lines` and `text`, alongside the beam envelope under `fill`, so a caller
who wants a second plane emphasized, or a label reworded, sets it on the
artist instead of rebuilding the train.
