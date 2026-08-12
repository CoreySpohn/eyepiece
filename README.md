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
  zero), `show_field` (amplitude and phase panels of a complex field),
  `compare_row` (several images sharing one norm and colorbar), and
  `triptych` (A, B, and a ratio or residual comparison panel, side by side).
- **Distributions.** `corner`, `corner_overlay` (a second sample set laid
  over an existing triangle plot), `hist_vs_pdf`, and `cov_ellipse`.
- **Profiles.** `plot_radial` (a precomputed radial profile line),
  `plot_contrast_curve` (a contrast curve with inner/outer working angle
  shading and reference floor curves, drawn once per axes even across
  repeated calls), and `radial_profile_plot` (computes the profile via
  `hwoutils` and plots it in one call, under the `[hwo]` extra).
- **Scenes.** `trail` (a 2D or 3D trajectory with depth-cued markers),
  `sky_fan` (weighted candidate sky tracks with an inner-working-angle disk),
  `fading_track`, and `schematic` (a miniature optical-train rail with one
  plane picked out).
- **Schematics.** `rail` (an optical-train diagram built from a plain
  `(label, glyph)` element list, over the `GLYPHS` vocabulary), and
  `schematic`, a preset wrapper over `rail` for the imager and coronagraph
  trains that come up constantly.
- **Layout.** Pixel-edge extent helpers (`extent_lod`, `extent_arcsec`,
  `extent_au`, ...) with matching axis labelers, plus `Frame` and
  `SourceStyles` for keeping several panels of one scene consistent.
- **Output.** `save_fig` for a styled write to disk, `record` and `animate`
  for animation, the public `Animation` type they both return, and
  `PRESETS` of measured fps/dpi pairs.

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

## License

MIT
