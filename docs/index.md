# eyepiece

`eyepiece` is a small library of plotting and animation primitives for
astronomical imaging simulations: ax-first stateless figure functions for
images, comparisons, profiles, and distributions, a multi-sink animation
recorder, and styled figure saving, all built on top of matplotlib and NumPy.

Every primitive takes plain arrays in and returns a small result object
holding the axes it drew on and the artists it made, so a caller can keep
working on a figure without hunting down which image or line object came
from where. Nothing in the library imports a simulation package, and
`import eyepiece` never activates a style or writes to matplotlib's
rcParams, so a primitive behaves the same whether it is called from a
notebook, a figure script, or another library's plotting module.

The same arrays can also be written down once as a prepared view: plain
records of images, curves, tracks, and their marks, with units, masks,
display scales, and stable IDs. The {doc}`prepared views <explanation/prepared-views>`
guide renders one preparation as a paper still, a strip of epochs, and a
Matplotlib movie through `eyepiece.mpl`, and
{doc}`Rendering in Manim <explanation/rendering-in-manim>` plays the same preparation as native Manim objects in a talk through
`eyepiece.manim`.

Because the same behavior holds for every function, the rules are worth
reading once rather than rediscovering per call. The
{doc}`contract <explanation/contract>` page states them, and
{doc}`viz-convention <viz-convention>` describes how a simulation library
ships plotting for its own types on top of these primitives.
{doc}`evidence <explanation/evidence>` is the reasoning underneath both: proportionality,
data ink, layering by value, small multiples, labels on the data, and showing
mechanism rather than outcome.

## Installation

```bash
pip install eyepiece
```

The base install pulls in matplotlib, NumPy, and
[hwostyle](https://github.com/HabitableWorldsObservatory/hwostyle), which
supplies the colormaps, palettes, and savefig policy that primitives read at
call time.

```bash
pip install eyepiece[hwo]
```

The `[hwo]` extra adds
[hwoutils](https://github.com/CoreySpohn/hwoutils), which provides the unit
conversions behind `extent_arcsec`, `extent_au`, and `Frame.extent_arcsec`,
and the profile computation behind `radial_profile_plot`. Those four are the
only names that need it, and each raises an `ImportError` naming the extra
when it is missing. Everything else works on the base install.

```bash
pip install "eyepiece[manim]"
pip install "eyepiece[slides]"
```

The `[manim]` extra adds Manim Community for `eyepiece.manim`, and
`[slides]` adds Manim Slides beside it. Manim needs system Cairo, Pango,
pkg-config, and ffmpeg. Nothing outside `eyepiece.manim` imports either.

## Quickstart

A primitive called with no axes creates its own figure, draws into it, and
hands back both. Layout helpers supply the pixel-edge extent and the
matching axis labels, and `save_fig` writes the result using the active
mode's savefig policy.

```python
import eyepiece as ep

result = ep.imshow_log(psf, extent=ep.extent_lod_from_pixels(psf.shape[0], 0.25))
ep.label_lod(result.ax)
ep.save_fig(result.fig, "psf")
```

The same call given an axes draws into that axes instead, which is what
makes a primitive usable as one panel of a larger figure the caller
assembled. Multi-panel primitives take an `axes=` sequence in place of
`ax=`, and return the array of panels they drew on.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, layout="constrained")
row = ep.compare_row(
    [before, after, model], titles=["Before", "After", "Model"], axes=axes
)
```

Because a result carries the artists it made, animating a figure is a matter
of mutating them rather than redrawing. Image primitives that transform their
data before display also return an `.update` that re-applies the same
transform, so a frame costs one `set_data` call.

```python
result = ep.imshow_log(cube[0])


def draw(fig, k):
    result.update(cube[k])


ep.animate(result.fig, draw, len(cube), fps=10).save("run.mp4", "run.gif")
```

`animate` binds a figure, a draw function, and a frame source, and renders
one pass into every sink named in `.save`. When the frames come from a loop
the caller already owns, a simulation stepping forward for instance, use
`record` instead and take a frame whenever there is something to show.

```python
with ep.record(result.fig, "run.mp4", "run.gif", fps=10) as rec:
    for state in simulation:
        result.update(state.image)
        rec.frame()
    rec.hold(8)
```

## What is in it

Every public name is importable straight from `eyepiece`; the submodules
that implement them are internal organization. The full signatures live in
the API reference, and the groups are:

- **Images.** `imshow_log`, `imshow_diverging`, `show_field` for a complex
  field as real, imaginary, amplitude, and phase panels, `compare_row` for
  several images under one shared norm and colorbar, and `triptych` for A,
  B, and a ratio or residual panel, plus `overlay_circle` for an aperture
  or ring drawn over an image, with an optional label, and `ruler` for a
  labeled dimension arrow.
- **Distributions.** `corner`, `corner_overlay`, `hist_vs_pdf`,
  `cov_ellipse`, and `convergence` for samples and their running mean or sum
  against labeled references.
- **Profiles.** `plot_radial`, `plot_contrast_curve`, and
  `radial_profile_plot`.
- **Scenes.** `trail`, `sky_fan`, and `fading_track`.
- **Phasors.** `phasor` for complex numbers as arrows on the complex plane,
  separate or chained tip to tail with their resultant, with a phase-colored
  ring as the key, or as a small dial inset beside a feature of another panel,
  and `phase_ring` for that key drawn alone on any axes. `curve_insets`
  stands small insets, chains of arrows or anything a callback draws, over
  marked points of a curve.
- **Emphasis.** `fade`, which blends any drawn artists toward their
  background so an element carried from an earlier figure reads as already
  seen, `capture`, which collects what a block of code drew so it can be
  faded as one group, and `blend`, the same color blend for a single color.
- **Schematics.** `rail` for an optical train built from a plain
  `(label, glyph)` list over the `GLYPHS` vocabulary, in its own axes or in
  the caller's data coordinates, `schematic`, a preset wrapper over it, and
  `rail_panels` for a strip of axes hung under the planes of a drawn rail.
- **Motion.** `Timeline`, which declares an animation's named spans (holds,
  entries, moves, captions, pulses, the lines of a narration) in seconds and
  resolves them for any frame rate into a `Plan` of per-frame knob values and
  `Event`s, with `Time` handles that add exactly, `Span` and `Fault` records,
  `ease` (including `back_out` and `running_start`, which overshoot and pull
  back), `stagger`, and `frame_count`.
- **Cameras and morphs.** `zoom_path`, the optimal pan-and-zoom path between
  two views, `view_limits` for a data camera and `overview_box` to mark its
  view on an overview panel; `PageCamera`, which renders any view of a
  finished figure at a fixed size and stays sharp when it magnifies, and
  `spotlight`, which dims a figure outside chosen rectangles; `quad_image`,
  an image drawn as one movable shape per pixel from `pixel_quads`, and
  `brush`, which lights the same objects in several panels at once.
- **Attention.** `Anchor`, a data point or a part of a drawn artist
  resolved in display space at draw time; `flight`, which carries a copy of
  a mark from one panel onto another; `callout`, a caption of a few words
  beside the thing it names, its key term in its mark's color; `leader`, a
  connector that refuses to break the evidence guide's rules; `Stage`,
  which shows, dims, or hides named groups at absolute levels; `build_axes`,
  which draws an axis on (spine, ticks, numbers, then its name);
  `collapse`, which gathers a row of cells into a bar whose length is their
  sum and can carry the bar to another panel; `indicate`, a pulse that
  returns the artist to where it was, `circumscribe`, an outline that draws
  on and fades, `term_row`, an equation with one text per term,
  `readout`, a number that counts to a new value in a fixed layout,
  `words`, one line of text with one text per word to color or reveal alone,
  `link`, a line that grows between panels, `morph`, a ghost that carries one
  path into another keeping its identity, `write`, which traces a text's
  glyph outlines and then fills them, `equation`, math as glyph outlines with named
  terms, `transform`, which carries one equation into another glyph by glyph, `warp`, which carries drawn
  artists through a matrix or a pointwise map of the plane (with `warp_grid`
  for the grid), `arrow`, `angle_arc` and `compass`, marks that grow, sweep
  and turn on the same artists, `follow_zoom`, a magnified inset that
  follows a moving point, and `audit`, which checks the
  leaders and links of a figure; and `Tempo`, which sets a
  clip's pace in one place: how long moves and settles last, and how long
  each caption holds for its number of words.
  The submodule `eyepiece.testing` checks a figure, or a clip sampled at
  chosen times, for overlapping text, text clipped by the frame, text too
  small for its venue, and text on marks of its own color; `motion_faults`
  samples every frame of a move and reports jumps, dead moves, drift, torn
  groups, overlaps, and artists that leave the frame; `speech_rate`
  reports the words per minute of each narrated line.
- **Layout.** `extent_lod`, `extent_lod_from_pixels`, `extent_arcsec`,
  `extent_au`, the matching `label_lod`, `label_arcsec`, and `label_au`, plus
  `Frame` and `SourceStyles` for keeping several panels of one scene
  consistent.
- **Output.** `save_fig`, the `record` context manager, `animate` and the
  `Animation` it returns, `RawSink` for frames from anywhere into a
  reproducible mp4, and `PRESETS` of measured fps and dpi pairs.
- **Vocabularies.** `ARTIST_KEYS`, the key set a result's `artists` dict
  draws from, alongside `PlotResult` and `MosaicResult` themselves.

Prepared views live in their own namespaces rather than at the top level:

- **`eyepiece.prepared`.** The records (`ImageView`, `CurveView`,
  `TrackView`, `PanelGroup`, and the `Path`, `Points`, `Region`,
  `ReferenceLine`, and `Label` marks, with `AxisSpec` and `Scale`),
  `Sequence` with its `ArrayChannel`, `PathWindow`, and `Clock` channels,
  and the display mapping (`normalize_values`, `map_rgba`, `resolve_bounds`,
  `weight_opacity`). It imports only NumPy.
- **`eyepiece.style`.** `SourceCast`, `RenderProfile`, and
  `snapshot_profile`.
- **`eyepiece.mpl`.** `render`, returning an `MplResult`, and `animate`,
  returning an `Animation`.
- **`eyepiece.manim`.** `render`, returning a `ManimResult`, `animate`,
  returning a `ManimClip`, `Units` for point sizes, and `ensure_font`.

## Where to go next

Read {doc}`contract <explanation/contract>` to predict what any primitive will
do with the arguments you hand it and what you get back. Read
{doc}`prepared views <explanation/prepared-views>` to prepare a scene once for figures,
movies, and talks, and
{doc}`Rendering in Manim <explanation/rendering-in-manim>` to present one. Read
{doc}`viz-convention <viz-convention>` if you maintain a simulation library
and want its own types to plot themselves without eyepiece ever learning
about them. Read {doc}`the clock <explanation/clock>` for timing and motion rules,
{doc}`narration design <explanation/narration-design>` for speech and cues, and
{doc}`Eyepiece and Manim <explanation/eyepiece-and-manim>` for the motion
comparison. The API reference documents every signature.

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/still
tutorials/build-up
tutorials/narrated
```

```{toctree}
:maxdepth: 2
:caption: How-to

how-to/bring-a-panel-in
how-to/draw-a-line-on
how-to/bind-a-move-to-a-knob
how-to/declare-a-cast
how-to/check-a-clip
how-to/step-through-discrete-values
how-to/keep-the-scale-fixed
how-to/stills-from-the-encode
how-to/write-a-viz-module
how-to/burn-captions-into-a-band
how-to/estimate-length-before-speaking
how-to/speak-a-script
how-to/write-numbers-for-speech
how-to/check-narration-and-motion
how-to/publish-a-scrubber-page
```

```{toctree}
:maxdepth: 2
:caption: Reference

autoapi/index
gallery/images
gallery/stats
gallery/intervals
gallery/marks
gallery/values
gallery/diagnostics
gallery/profiles
gallery/scene
gallery/phasor
gallery/animation
gallery/motion
gallery/one-scene-n-views
gallery/prepared
gallery/motifs
```

```{toctree}
:maxdepth: 2
:caption: Explanation

explanation/contract
explanation/clock
explanation/evidence
explanation/narration-design
explanation/prepared-views
explanation/eyepiece-and-manim
explanation/rendering-in-manim
viz-convention
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

autoapi/index
```
