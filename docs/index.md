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

Because the same behavior holds for every function, the rules are worth
reading once rather than rediscovering per call. The
{doc}`contract <contract>` page states them, and
{doc}`viz-convention <viz-convention>` describes how a simulation library
ships plotting for its own types on top of these primitives.

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
row = ep.compare_row([before, after, model], titles=["Before", "After", "Model"], axes=axes)
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
  B, and a ratio or residual panel.
- **Distributions.** `corner`, `corner_overlay`, `hist_vs_pdf`, and
  `cov_ellipse`.
- **Profiles.** `plot_radial`, `plot_contrast_curve`, and
  `radial_profile_plot`.
- **Scenes.** `trail`, `sky_fan`, and `fading_track`.
- **Schematics.** `rail` for an optical train built from a plain
  `(label, glyph)` list over the `GLYPHS` vocabulary, and `schematic`, a
  preset wrapper over it.
- **Layout.** `extent_lod`, `extent_lod_from_pixels`, `extent_arcsec`,
  `extent_au`, the matching `label_lod`, `label_arcsec`, and `label_au`, plus
  `Frame` and `SourceStyles` for keeping several panels of one scene
  consistent.
- **Output.** `save_fig`, the `record` context manager, `animate` and the
  `Animation` it returns, and `PRESETS` of measured fps and dpi pairs.
- **Vocabularies.** `ARTIST_KEYS`, the key set a result's `artists` dict
  draws from, alongside `PlotResult` and `MosaicResult` themselves.

## Where to go next

Read {doc}`contract <contract>` to predict what any primitive will do with
the arguments you hand it and what you get back. Read
{doc}`viz-convention <viz-convention>` if you maintain a simulation library
and want its own types to plot themselves without eyepiece ever learning
about them. The API reference documents every signature.

```{toctree}
:maxdepth: 2
:caption: Guides

contract
viz-convention
```

```{toctree}
:maxdepth: 2
:caption: Gallery

gallery/images
gallery/stats
gallery/profiles
gallery/scene
gallery/animation
gallery/one-scene-n-views
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

autoapi/index
```
