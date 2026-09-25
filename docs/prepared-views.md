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

# Prepared views

A prepared view is a scene written down as plain data: frozen records that
carry borrowed NumPy arrays, units, labels, display scales, and stable
element IDs, and nothing else. A simulation library prepares one once, from
its own objects, and every output then reads the same record: a paper still,
a strip of selected epochs, a Matplotlib movie, and a native Manim clip for a
talk. None of those outputs recomputes the science, so they cannot disagree
about it.

The layer has four parts, each in its own namespace.

| Namespace | What it holds | What it imports |
|---|---|---|
| `eyepiece.prepared` | The records, the display mapping, and `Sequence` | The standard library and NumPy only |
| `eyepiece.style` | `SourceCast`, `RenderProfile`, and `snapshot_profile` | hwostyle and Matplotlib |
| `eyepiece.mpl` | `render` and `animate` for Matplotlib | Matplotlib |
| `eyepiece.manim` | `render` and `animate` for Manim Community | Manim, through the `[manim]` extra |

Importing `eyepiece.prepared` loads neither renderer, so a library can build
records in its base install. The renderers are documented here and on the
{doc}`Manim page <manim>`; the {doc}`prepared gallery <gallery/prepared>`
shows a still, a strip, and a movie from one preparation.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece.mpl as mpl
from eyepiece.prepared import (
    ArrayChannel,
    AxisSpec,
    Clock,
    ImageView,
    Label,
    Path,
    Points,
    Region,
    Scale,
    Sequence,
    TrackView,
    find_element,
    map_rgba,
    normalize_values,
    replace_elements,
    resolve_bounds,
    weight_opacity,
)
from eyepiece.style import SourceCast, snapshot_profile

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small. A real figure script
# keeps the style library's 300 dpi print policy and omits this line.
plt.rcParams["savefig.dpi"] = 120
```

## The records

Views hold marks, and a `PanelGroup` holds views. Building a mark or a view
validates it, including the axis specification and scale a view carries, and
every error names the offending element's ID, so a renderer can trust what it
is handed.

| Record | Fields and meaning |
|---|---|
| `AxisSpec` | `x_label`, `y_label`, `x_limits`, `y_limits`, `aspect="equal"`, `x_reverse=False`, `show_ticks=True` |
| `Scale` | `kind` (linear, symmetric, or log), `vmin`, `vmax`, `cmap_role="intensity"`, `floor=None` |
| `ImageView` | `id`, `data` shaped (Y, X), `axes`, `scale`, `quantity`, `valid=None`, `marks=()` |
| `CurveView` | `id`, `axes`, `marks=()`: a curve panel, such as a time trace |
| `TrackView` | `id`, `axes`, `marks=()`: a 2D coordinate panel, such as a sky track |
| `PanelGroup` | `id`, `views`, `direction="row"` (or `"column"`) |
| `Path` | `id`, `xy` shaped (P, 2), `source_id=None`, `weight=1.0`, `visible=(0, None)`, `label=""` |
| `Points` | `id`, `xy` shaped (P, 2), `source_id=None`, `xerr=None`, `yerr=None`, `label=""` |
| `Region` | `id`, `center` shaped (2,), `outer_radius`, `inner_radius=0.0`, `role="reference"`, `label=""` |
| `ReferenceLine` | `id`, `axis` (`"x"` or `"y"`), `value`, `role="reference"`, `label=""` |
| `Label` | `id`, `text`, `xy`, `space="panel"` (or `"data"`) |

An image has no separate extent field: its pixel-edge extent is the axis
limits, `(axes.x_limits, axes.y_limits)`. Row 0 of `data` is the bottom row
of the displayed image. `x_reverse=True` flips the display once, which is how
right ascension increases to the left; the data, the limits, and the image
storage are never reversed, so every mark on the panel stays in the same data
coordinates as the pixels under it.

Arrays are borrowed rather than copied or promoted. A float32 cube, a
read-only memory map, or a strided slice is stored as given, which means the
caller keeps that storage alive and unchanged for as long as any output made
from it is in use. Mutating it afterwards requires a fresh preparation.

## Validity, clipping, and scales

Validity has two sources: a mask and a nonfinite value. A masked array handed
to `ImageView` has its mask captured before the data is coerced, an explicit
`valid=` array is combined with it by logical and, and any NaN or infinity is
invalid whatever the mask says.

```{code-cell} python
data = np.ma.masked_array(
    np.array([[1e-9, 2e-9], [np.nan, 5e-12]]),
    mask=[[False, True], [False, False]],
)
axes = AxisSpec("x (pixel)", "y (pixel)", (0.0, 2.0), (0.0, 2.0))
scale = Scale("log", vmin=1e-10, vmax=1e-8, floor=1e-10)
view = ImageView("demo", data, axes, scale, quantity="contrast")
print("valid from the mask:\n", view.valid)
print("display values:\n", normalize_values(view.data, valid=view.valid, scale=scale))
```

The masked pixel and the NaN are both invalid, and both display as masked.
The valid pixel at `5e-12` sits below the display floor, so it is clipped to
the bottom of the scale rather than masked: a clipped value is still data,
drawn in the colormap's first color, while an invalid one is drawn in the
profile's `bad_rgba`, a saturated magenta that no colormap contains. Clipping
is a display decision. A scientific floor, such as a coronagraph's residual
added to a model, belongs to the preparation and is applied before any of
this.

`map_rgba` is the one mapping from values to colors. It indexes the
profile's 256-entry table with the same bins a Matplotlib colorbar built from
that table uses, so an image pixel and the colorbar swatch at the same value
are the same entry.

`resolve_bounds` makes one pass over `(data, valid)` pairs and returns the
full-sequence bounds for a scale kind, ignoring invalid samples. A still made
from one frame should use those bounds, not its own range, so that the still
and the movie share a scale. It raises when no frame has a valid sample, and
when the data are constant, which needs explicit bounds instead.

```{code-cell} python
frames = [(np.array([1.0, np.nan]), None), (np.array([-3.0, 2.0]), None)]
print("linear   ", resolve_bounds(frames, kind="linear"))
print("symmetric", resolve_bounds(frames, kind="symmetric"))
try:
    resolve_bounds([(np.array([np.nan, np.inf]), None)], kind="linear")
except ValueError as err:
    print("error:", err)
```

A symmetric scale is linear with `vmin == -vmax`, for a signed quantity such
as a difference between two intensities, and wants a diverging colormap role
so that zero is visible. A log scale needs a positive `floor` and `vmin`.

## Identity

Every view and mark has an `id` unique within its tree. Renderers expose
their output by these IDs, and a sequence edits a tree by them. A `source_id`
is a different thing: it names the scientific source a mark belongs to, so
forty candidate paths can all be one planet, drawn in that planet's color,
while each path remains separately addressable.

`find_element` looks an element up by ID, and `replace_elements` returns a
new tree with named elements replaced. It validates every replacement before
rebuilding anything, and every branch it did not touch is the same object
afterwards, arrays included.

```{code-cell} python
star = Points("star", np.array([[0.0, 0.0]]))
path = Path("path", np.array([[0.0, 0.0], [1.0, 1.0]]), source_id="planet b")
track = TrackView(
    "sky", AxisSpec("x", "y", (-1.0, 2.0), (-1.0, 2.0)), marks=(star, path)
)
moved = replace_elements(
    track, {"star": Points("star", np.array([[0.5, 0.5]]))}
)
print(find_element(moved, "path") is path, find_element(moved, "star") is star)
```

## Sequences and physical time

A `Sequence` is a fixed-topology template view, one physical timestamp per
sample, and the channels that say how the template varies:

- `ArrayChannel(element_id, field, values)` drives an image's `"data"` or
  `"valid"`, or a `Points` mark's `"xy"`, from an array with one leading slot
  per sample. A masked cube has its mask captured once.
- `PathWindow(element_id, history=None)` reveals a stored `Path` up to the
  current sample: a growing prefix, or a trailing window of `history`
  vertices. The path has one vertex per sample and is stored once.
- `Clock(element_id, fmt="{value:g} {unit}")` sets a `Label` to the
  acquisition time of the sample being shown. It reports the sampled time,
  never an interpolated one.

Every frame is a complete state, not a patch on the one before it, so
seeking backwards reconstructs the same trail as playing forwards. Times must
be finite and strictly increasing, and they are never sorted for the caller.
Only instantaneous samples are supported: `sample_kind` other than
`"instantaneous"`, such as exposure-integrated frames, raises.

`Sequence.at(t)` is left sample-and-hold. It shows sample `i` on
`[t_i, t_(i+1))`, the final sample from the final timestamp on, and clamps
requests outside the sampled range. A request within a relative `1e-9` below
a sample time counts as that sample time, so a time recovered through a unit
conversion does not fall back one sample. With samples at 0, 1, and 10, time
5 shows the sample acquired at 1.

```{code-cell} python
cube = np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2)
template = ImageView(
    "image",
    cube[0],
    AxisSpec("x", "y", (0.0, 2.0), (0.0, 2.0)),
    Scale("linear", 0.0, 11.0),
    quantity="counts",
    marks=(Label("clock", "", (0.03, 0.95)),),
)
sequence = Sequence(
    template,
    times=[0.0, 1.0, 10.0],
    time_unit="s",
    channels=(ArrayChannel("image", "data", cube), Clock("clock")),
)
for t in (-1.0, 0.0, 0.5, 1.0, 5.0, 10.0, 12.0):
    sample = sequence.at(t)
    text = find_element(sample.view, "clock").text
    print(f"t = {t:5.1f}: sample {sample.index}, clock reads {text!r}")
```

Presentation duration and frame rate are separate from physical time.
`schedule(run_time, fps)` returns `N = ceil(run_time * fps)` physical times,
at least two for a sequence that changes, evenly spaced from the first sample
through the last with both ends included. The encoded duration is `N / fps`,
and the final sample is always on screen for at least one output frame. A
product within float noise of a whole number counts as that number, so
`run_time = 7 / 25` at 25 fps gives seven frames. Both renderers play from
this schedule.

```{code-cell} python
times = sequence.schedule(run_time=1.0, fps=4)
print("output times:", np.round(times, 2))
print("held samples:", [sequence.at(t).index for t in times])
print("frames for 0.28 s at 25 fps:", len(sequence.schedule(0.28, 25)))
```

`frame(i)` returns the state at sample `i`, and `strip(indices)` returns a
`PanelGroup` with one slot per index. Each slot's IDs are prefixed with its
position (`"0/image"`, `"1/image"`), so a repeated or out-of-order selection
still has unique IDs. Each slot carries exactly one time label. When the
sequence has a `Clock`, that label is the clock's own label (`"0/clock"`),
restamped with the slot's acquisition time in the clock's format, so a strip
never shows two time labels and a clock label may itself be named `"time"`.
Without a `Clock`, each slot gains a `"<slot>/time"` label in the default
format.

## Appearance: cast and profile

A renderer takes two appearance inputs, and both are explicit.

A `SourceCast` fixes which palette slot and marker each source name gets,
from its position in the declared list. Declare one per document, as a
module-level list shared by every script whose figures appear together, and
pass it to every render. Left out, a renderer builds one from the tree's
source IDs in encounter order, which is only stable within one tree.

A `RenderProfile` is a read-only snapshot of the style: palette colors,
sampled colormap tables, the bad-sample color, text and background colors,
the reference color, font family, text size, and stroke width.
`snapshot_profile()` reads the current hwostyle state at the moment it is
called and never changes global state; call `hwostyle.use(...)` first to
snapshot a particular mode. A profile never changes after construction, so
switching modes cannot recolor a result that already exists, and two profiles
taken in two modes can render the same view for a paper and for a talk.

The renderers differ in one default. Without a `profile`, `eyepiece.mpl`
snapshots with the Matplotlib rc sizes (`font.size` and `lines.linewidth`),
sized for a paper figure, while `eyepiece.manim` uses the `snapshot_profile`
defaults of 24 pt text and 1.5 pt strokes, sized for a slide. Either way the
result keeps its profile for every later update.

Region and reference-line colors come from the profile's reference role,
which is the only role currently supported; any other `role` raises. Paths
and points take their source's palette color, and a mark with no `source_id`
takes the text color.

## Rendering with Matplotlib

`eyepiece.mpl.render(view, *, ax=None, axes=None, cast=None, profile=None)`
draws a tree and returns an `MplResult`. Without axes it creates a
constrained-layout figure, nesting panel groups as rows and columns; with
`ax=` or `axes=` (one Axes per panel, depth first) it draws into exactly what
it was handed.

The track below shows the conventions a coordinate panel carries: right
ascension increasing to the left, candidate paths whose opacity follows
their weights, observations with error bars, a filled inner-working-angle
region, and a missing observation given as a NaN row.

```{code-cell} python
phase = np.linspace(0.0, 2.0 * np.pi, 120)
radii = [70.0, 85.0, 100.0]
weights = [0.2, 1.0, 0.5]
paths = tuple(
    Path(
        f"candidate/{k}",
        np.column_stack([a * np.cos(phase), 0.6 * a * np.sin(phase)]),
        source_id="planet b",
        weight=w,
    )
    for k, (a, w) in enumerate(zip(radii, weights, strict=True))
)
observed = np.array([[80.0, 25.0], [np.nan, np.nan], [-60.0, 40.0]])
observations = Points(
    "observations",
    observed,
    source_id="planet b",
    xerr=np.array([6.0, 0.0, 6.0]),
    yerr=np.array([6.0, 0.0, 6.0]),
    label="observed",
)
sky = TrackView(
    "sky",
    AxisSpec(
        "RA offset (mas)", "Dec offset (mas)", (-120.0, 120.0), (-90.0, 90.0),
        x_reverse=True,
    ),
    marks=(
        Region("iwa", np.array([0.0, 0.0]), outer_radius=40.0, label="IWA"),
        *paths,
        observations,
        Points("star", np.array([[0.0, 0.0]])),
    ),
)
cast = SourceCast(["planet b"])
profile = snapshot_profile(
    text_size_pt=plt.rcParams["font.size"],
    stroke_width_pt=plt.rcParams["lines.linewidth"],
)
fig, ax = plt.subplots(figsize=(4.4, 3.4), layout="constrained")
result = mpl.render(sky, ax=ax, cast=cast, profile=profile)
print(sorted(result.parts))
print("path opacities:", [round(result.parts[p.id].get_alpha(), 3) for p in paths])
```

`result.parts` maps every element ID to its artist: an image to its
`AxesImage`, a curve or track view to its Axes, a path or reference line to
a `Line2D`, points to a `PathCollection`, a region to a `Circle` or
`Annulus`, and a label to a `Text`. Derived artists take suffixed keys:
`"<image id>/colorbar"`, and `"<points id>/xerr"` and `"<points id>/yerr"`
for error-bar segments. `result.axes` maps each drawn view's ID to its Axes,
`result.ax` is the single Axes of a one-panel result, and `result.fig` is the
figure. The artists are ordinary Matplotlib objects, so a caller adjusts one
through its handle, which is how a renderer without routed keyword
dictionaries is customized.

Path opacity is `0.15 + 0.65 * weight / max_weight` over the paths on one
panel, from `weight_opacity`. It is relative emphasis, not probability: a
candidate of zero weight stays visible at 0.15, the heaviest is drawn at
0.80, and a lone path is 0.80 whatever its weight. Points and moving heads are not
weighted. A figure whose claim depends on the weights themselves still owes
them a quantitative axis.

```{code-cell} python
print(weight_opacity([0.2, 1.0, 0.5]), weight_opacity([3.0]))
```

A few drawing rules hold in both renderers:

- A row of `xy` with any nonfinite component is a gap. A path breaks there,
  and a point and its error bars are hidden, while the handle keeps one entry
  per sample so a later finite update shows it again.
- A `Region` on an `ImageView` is an outline only, so a fill never changes
  the colors of the measured pixels under it. On a curve or track panel its
  interior is shaded at 0.2 opacity.
- `show_ticks=False` hides the tick labels. The tick marks and the axis
  labels remain.

On a figure the renderer created, constrained layout keeps the colorbar and
its label on the canvas. On caller axes the colorbar hangs in an inset beside
the image and no layout engine is installed, so a caller figure without one
can clip the colorbar label at the figure edge. Create the figure with
`layout="constrained"`, as above, or leave room on the right.

### Updates

`MplResult.update(view)` shows a new state of the same tree. It accepts new
values, path visible intervals, point, region, and reference-line positions,
label text, and path weights. It rejects a changed topology: different IDs or
kinds, image shapes, scales, axes, point counts, error-bar presence, or
source IDs. The whole new tree is validated, and every new image mapped,
before any artist changes, so a rejected update leaves the figure exactly as
it was. An image whose data and validity are the same borrowed arrays as the
ones displayed is not mapped again. Updates never change an artist's
visibility, so a part the caller hid stays hidden.

```{code-cell} python
still = sequence.frame(0)
fig, ax = plt.subplots(figsize=(3.2, 2.6), layout="constrained")
panel = mpl.render(still, ax=ax, profile=profile)
panel.update(sequence.frame(2))
print(find_element(panel.view, "clock").text)

rescaled = replace_elements(
    still,
    {"image": ImageView("image", cube[0], still.axes, Scale("linear", 0.0, 99.0),
                        quantity="counts", marks=still.marks)},
)
try:
    panel.update(rescaled)
except ValueError as err:
    print("rejected:", err)
plt.close(fig)
```

`eyepiece.mpl.animate(sequence, *, run_time, fps=30, ...)` renders frame 0
and plays the schedule through `update`, returning an `eyepiece.Animation`
that writes through `.save`, `.jshtml`, or `.video`, like any other eyepiece
animation.

## Where each piece lives

The records hold no simulation objects, Matplotlib artists, or Manim
mobjects, and a renderer never receives one. The division of work follows
from that.

- A simulation library's preparation functions own extraction, units,
  coordinate conventions, scientific floors, masks, and traces. They return
  records or sequences, import only `eyepiece.prepared`, and run once.
- Eyepiece owns the records, the display mapping, and the renderers, and
  computes no science during rendering or updates.
- A consumer script owns the story: which preparations to combine, the cast,
  the profile, layout, and, for a talk, the slides.

A consumer that wants to drive native Manim geometry straight from a
scientific model, evaluating it at each playback time, does so in its own
code and needs no viz import; the {doc}`Manim page <manim>` shows that
recipe beside the prepared renderer.
