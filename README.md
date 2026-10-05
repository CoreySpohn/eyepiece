# eyepiece

Plotting and animation primitives for astronomical imaging simulations.

## What eyepiece is

`eyepiece` provides ax-first, stateless figure functions for the plots that
recur across direct-imaging work: log-scaled and diverging image displays,
side-by-side and A/B comparison panels, radial profiles and contrast curves,
corner plots, orbit and sky-track scenes, optical-train schematics, and a
multi-sink animation recorder that grabs frames once and writes them to any
combination of gif, mp4, and html. Every primitive takes plain arrays in and
returns a small result object holding the axes and the artists it drew, so
callers can update or extend a figure without re-deriving which line or
image object came from where.

For a scene that has to appear more than once, eyepiece also offers prepared
views: plain records of images, curves, tracks, and their marks, carrying
borrowed arrays, units, masks, display scales, and stable IDs. A simulation
library prepares a scene once, and the same records render as a paper still
or a movie through `eyepiece.mpl`, and as native Manim objects for a talk
through `eyepiece.manim`, without recomputing any science.

Figure style (color, colormaps, and mode) is resolved through
[hwostyle](https://github.com/HabitableWorldsObservatory/hwostyle) at call
time, not at import time. `import eyepiece` never activates a style or
touches matplotlib's rcParams. Without an active hwostyle mode, primitives
fall back to a fixed light-mode palette, so eyepiece also works as a
standalone plotting library.

## What eyepiece is not

- **Not a simulation library.** eyepiece takes arrays and returns figures; it
  never imports a simulation, orbit, or optics package.
- **Not a style engine.** Color, colormap, and mode definitions live in
  hwostyle; eyepiece only reads them at call time.
- **Not a data pipeline.** Aggregating or transforming simulation output
  before plotting is the caller's job.

## What is in it

Every name below is importable straight from `eyepiece`; the submodules that
implement them are internal organization.

- **Images.** `imshow_log` (log scale clipped to a floor, so a zero-valued
  pixel cannot break the norm), `imshow_diverging` (symmetric norm about
  zero), both placed by `extent` or by pixel `centers`, `kymograph` (a cut
  stacked against time or wavelength), `show_field` (amplitude and phase
  panels of a complex field), `compare_row` (several images sharing one norm
  and colorbar), `compare_grid` (the same on a 2D layout with empty cells
  allowed), and `triptych` (A, B, and a ratio or residual comparison panel,
  side by side).
- **Annotations.** `overlay_circle` and `overlay_line` (dashed circles and
  paths that read over bright and dark pixels), `ruler` (a dimension arrow),
  and `bracket` (a labeled span).
- **Distributions and sequences.** `corner`, `corner_overlay` (a second
  sample set laid over an existing triangle plot), `hist_vs_pdf`,
  `cov_ellipse`, and three that reveal samples one at a time:
  `convergence` (a running mean or sum), `signed_trace` (a trace shaded
  above and below a level), and `hist_fill` (a histogram filling on a
  pinned scale).
- **Profiles.** `plot_radial` (a precomputed radial profile line),
  `plot_contrast_curve` (a contrast curve with inner/outer working angle
  shading and reference floor curves, drawn once per axes even across
  repeated calls, with each curve taking the next palette color so two
  calls on one axes are distinguishable), and `radial_profile_plot`
  (computes the profile via `hwoutils` and plots it in one call, under the
  `[hwo]` extra).
- **Scenes.** `trail` (a 2D or 3D trajectory with depth-cued markers),
  `sky_fan` (weighted candidate sky tracks with an inner-working-angle disk),
  and `fading_track`.
- **Schematics.** `rail` (an optical-train diagram built from a plain
  `(label, glyph)` element list, over the `GLYPHS` vocabulary), and
  `schematic`, a preset wrapper over `rail` for the imager and coronagraph
  trains that come up constantly.
- **Motion.** `Timeline` (an animation's holds and ramps declared in
  seconds and resolved for any frame rate into a `Plan` of per-frame knob
  values), with `ease`, `stagger`, and `frame_count`.
- **Cameras and morphs.** `zoom_path` (the optimal pan-and-zoom path between
  two views), `view_limits` and `overview_box` for a camera that zooms an
  axes, `PageCamera` (any view of a finished figure, re-rendered sharp at a
  fixed output size, with an optional overview margin), `spotlight` (a veil
  that dims a figure outside chosen rectangles), `quad_image` (an image drawn
  as one movable shape per pixel, from `pixel_quads`), and `brush` (the same
  objects lit in several panels at once).
- **Layout.** Pixel-edge extent helpers (`extent_lod`, `extent_arcsec`,
  `extent_au`, ...) with matching axis labelers, plus `Frame` and
  `SourceStyles` for keeping several panels of one scene consistent.
- **Output.** `save_fig` for a styled write to disk, `record` (a context
  manager that opens every sink at once and takes frames from a loop the
  caller drives) and `animate` (which binds a figure, a draw function, and a
  frame source into the public `Animation` type it returns), `RawSink` (raw
  frames from anywhere, such as a `PageCamera`, into a reproducible mp4), and
  `PRESETS` of measured fps/dpi pairs.

Prepared views live in namespaces of their own:

- **`eyepiece.prepared`.** The frozen records (`ImageView`, `CurveView`,
  `TrackView`, `PanelGroup`, and the `Path`, `Points`, `Region`,
  `ReferenceLine`, and `Label` marks), `Sequence` for replayable states over
  physical time, and the shared display mapping. It imports only NumPy.
- **`eyepiece.style`.** `SourceCast` for fixed source colors and markers, and
  `snapshot_profile`, which captures the active hwostyle appearance as a
  read-only `RenderProfile`.
- **`eyepiece.mpl`.** `render` and `animate` for Matplotlib stills and movies.
- **`eyepiece.manim`.** `render` and `animate` for native Manim Community
  objects, usable in a Manim scene or a Manim Slides deck, with `Units` for
  point sizes and `ensure_font` for Pango fonts.

## Usage

```python
import eyepiece as ep

result = ep.imshow_log(psf, extent=ep.extent_lod_from_pixels(psf.shape[0], 0.5))
ep.label_lod(result.ax)
ep.save_fig(result.fig, "psf")
```

Every primitive returns a small result object carrying the axes it drew on
and the artists it made (keyed by the `ARTIST_KEYS` vocabulary), so a caller
can keep working on the figure without hunting for the objects again. An
image primitive also returns an `.update` that redraws with new data through
the same transform, which is what makes animation a few lines:

```python
frames = [cube[k] for k in range(len(cube))]
result = ep.imshow_log(frames[0])


def draw(fig, k):
    result.update(frames[k])


ep.animate(result.fig, draw, len(frames), fps=10).save("run.mp4", "run.gif")
```

A prepared sequence gives a still, a strip of selected epochs, and a movie
from one preparation:

```python
import eyepiece.mpl as mpl
from eyepiece.style import SourceCast, snapshot_profile

cast = SourceCast(["planet b"])
still = mpl.render(sequence.frame(10), cast=cast)
strip = mpl.render(sequence.strip([0, 10, 20]), cast=cast)
movie = mpl.animate(sequence, run_time=8, fps=30, cast=cast)
movie.save("run.mp4")
```

The same sequence plays in a talk through Manim:

```python
import eyepiece.manim as em

clip = em.animate(sequence, cast=cast, profile=snapshot_profile())
self.add(clip.mobject)  # inside a Manim Scene or Slide
self.play(clip.playback(run_time=8))
```

`rail` draws a miniature optical-train diagram from a plain element list, so
a physics panel can sit beside a reminder of which plane it shows:

```python
import eyepiece as ep

result = ep.rail(
    [("Pupil", "pupil"), ("FPM", "fpm"), ("Lyot", "lyot"), ("Focal", "focal")],
    highlight="FPM",
)
ep.save_fig(result.fig, "coronagraph_rail")
```

## Status

eyepiece is young: the primitives above are implemented and tested, and the
public API may still shift before 1.0.

## Installation

```bash
pip install eyepiece
```

Unit conversions used by a small number of layout helpers (arcsecond and AU
extents) and `radial_profile_plot`'s profile computation are optional and
pull in [hwoutils](https://github.com/CoreySpohn/hwoutils):

```bash
pip install eyepiece[hwo]
```

The Manim renderer is optional too. `[manim]` adds Manim Community, and
`[slides]` adds Manim Slides beside it; Manim needs system Cairo, Pango,
pkg-config, and ffmpeg:

```bash
pip install "eyepiece[manim]"
pip install "eyepiece[slides]"
```

## License

MIT
