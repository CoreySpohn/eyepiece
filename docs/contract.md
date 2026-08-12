# The primitive contract

Every plotting primitive in eyepiece obeys the same rules about what it
accepts, what it draws where, and what it hands back. The rules exist so a
reader can predict a function's behavior from its signature alone, and so a
figure assembled from several primitives comes out looking like one figure
rather than a collage. This page states the contract; the API reference
states the arguments.

## Arrays in, plain floats for scalars

An array argument is anything `numpy.asarray` accepts, and it is converted
on entry. A NumPy array, a nested list, a JAX array, a masked array, or a
memory-mapped slice are all valid, and none of them are given special
treatment. No primitive accepts a simulation-library type, checks for one,
or imports a package that defines one, which is what keeps the library
usable from any code that can produce a number.

Scalars are plain Python floats. There is no `Quantity` type anywhere in the
API, no unit registry, and no attempt to infer units from an array's
metadata.

## Units live in argument names

A scalar that carries physical units says so in its own name:
`pixscale_mas` is milliarcseconds per pixel, `pixscale_lod` is lambda over D
per pixel, `wavelength_nm` is nanometers, `distance_pc` is parsecs, and
`diameter_m` is meters. Passing a number in the wrong unit is therefore a
visible mistake at the call site rather than a silent one inside the
function.

The convention has one consequence worth stating plainly: eyepiece does not
convert units it was not asked to convert. `plot_radial` takes `r` and
values with no unit suffix at all, and it deliberately sets no axis label,
because the separation units belong to the caller. Where a conversion is
genuinely needed, it goes through the `[hwo]` extra rather than being
reimplemented, as in `extent_arcsec` and `extent_au`.

## Who owns the figure

Single-panel primitives take `ax=None`, and multi-panel primitives take
`axes=None`. The default means the primitive creates its own figure, always
through `plt.subplots(..., layout="constrained")`, and returns it. Passing
an axes, or a sequence of axes, means the primitive draws into exactly what
it was handed and creates nothing.

This is the whole of the ax-first idea, and it is what lets the same
function serve a quick look and a publication panel. A caller who wants a
figure gets one for free; a caller who has already built a mosaic hands over
the slots and gets them filled.

A few multi-panel primitives also accept a `fig=` argument, which is a
`Figure` or a `SubFigure` to build their own panel grid inside. `show_field`
works this way: given `axes` it uses them, given `fig` it subdivides that
figure, and given neither it creates its own.

## What comes back

No primitive returns a bare matplotlib object, and none of them return
`None`. A single-axes primitive returns a `PlotResult`, a multi-panel one
returns a `MosaicResult`, and both are frozen dataclasses, so an attribute
cannot be reassigned out from under the figure. To change what is drawn,
call the primitive again or use the `update` described below.

`PlotResult` carries `.ax`, `.artists`, `.update`, and a `.fig` property
that reads the figure off the axes. `MosaicResult` carries `.axes` (the
panel array, as `plt.subplots` returns it), the same `.artists` and
`.update`, and a `.fig` property that reads the figure off the first panel.

### The artist vocabulary

`.artists` is a plain dict whose keys are drawn from `ARTIST_KEYS`, a fixed
vocabulary exported at the top level. It is a convention rather than an
enforced schema: a primitive populates only the keys for artists it actually
drew, so a missing key means "not drawn here", never "drawn and hidden".

Each key holds either a single matplotlib artist or a list of them. A list
usually means one entry per panel, in panel order, from a multi-panel
primitive drawing the same kind of artist in each. `lines` is the standing
exception, defined below as several artists on one axes, and `fill` and
`text` are read the same way when a single-axes primitive draws several of
them. A key names the kind of artist, not how many axes are involved.

- `image`: the `AxesImage` from `imshow`, or a list of one per panel.
- `cbar`: the `Colorbar` attached to an image or scalar mappable, or a list
  of one per panel.
- `line`: a single `Line2D`, or a list of one per panel.
- `lines`: a list of `Line2D` artists drawn together on one axes, as in a
  multi-curve plot, as distinct from `line`'s one-per-panel list.
- `fill`: the `PolyCollection` from `fill_between` or `fill_betweenx`, or a
  `Rectangle` for a shaded band. A list holds either one per panel or several
  drawn together on one axes.
- `hist`: the `BarContainer` from a filled `hist`, or the patch list a
  `histtype="step"` call returns, or a list of one per panel.
- `scatter`: the `PathCollection` from `scatter`, or a list of one per panel.
- `ellipse`: an `Ellipse`, or another `Patch`, marking a region, or a list of
  one per panel.
- `collection`: a `Collection` artist not covered by a more specific key
  above, such as an errorbar's `LineCollection` or the `QuadMesh` that
  `pcolormesh` and `hist2d` draw, or a list of one per panel.
- `text`: a `Text` artist placed as an annotation, never the title. A list
  holds either one per panel or several drawn together on one axes.
- `title`: the `Text` artist returned by `set_title`, or a list of one per
  panel.

### The update slot

`.update` is an optional callable that redraws with new data, reusing
whatever transform the primitive applied on the first draw. It is `None`
when there is no such transform to reuse.

The reason it exists is animation. `imshow_log` clips its data to a floor
before building the norm, so a frame loop that called `imshow` itself would
have to re-derive that clip and would eventually get it wrong. Instead
`result.update(new_image)` re-applies the floor and calls `set_data` on the
existing `AxesImage`, creating no new artist. `imshow_diverging`, by
contrast, has no stateful transform to reapply and so returns no `update`.

An `update` reuses the norm built from the first draw. Values outside that
norm are not an error, they render clipped to the colormap's end colors, and
the norm is not rescaled. When the data range is expected to move, pin it up
front with `vmin` and `vmax`, or call the primitive again.

## Routed keyword arguments

A parameter the primitive owns semantically is a real keyword argument:
`floor`, `vlim`, `iwa`, `owa`, `highlight`, `norm`, `mode`. Everything else
is routed through a per-target dict named for the matplotlib call it reaches:
`imshow_kw` for `ax.imshow`, `cbar_kw` for `fig.colorbar`, `line_kw` for
`ax.plot`, and so on for the targets a given primitive draws. Each dict is
merged last, so it also overrides the primitive's own defaults for that
call.

There is no `**kwargs` anywhere in the public API, and its absence is
deliberate. A flat `**kwargs` cannot say which matplotlib call an argument
was meant for, silently swallows a misspelling, and turns every parameter
matplotlib ever adds into part of this library's signature.

When something is not exposed, there are three moves, in order of
preference. Route it through the relevant `_kw` dict, which is what those
dicts are for. Failing that, reach through the returned artists and set the
property directly, since `result.artists["image"].set_clim(...)` and
`result.ax.set_xlim(...)` are exactly as valid as anything the primitive
does. Failing that, the need is a signature gap, and a parameter that keeps
coming up is a candidate for promotion into the real keyword list.

## Colorbars and geometry

A primitive that was handed an axes never alters geometry outside the slots
it was handed. It does not resize a neighbor, does not steal gridspec space,
and does not re-solve the caller's layout. A caller who assembled a mosaic
gets back the mosaic they assembled.

Colorbars are the place this rule bites, because `fig.colorbar(ax=...)`
takes its space out of the axes it is attached to, which visibly shrinks an
equal-aspect image panel and pushes everything beside it. The default
colorbar is therefore an in-slot inset, `ax.inset_axes([1.02, 0.0, 0.04,
1.0])`, which sits just outside the panel's right edge and consumes none of
the panel's own space.

Only a primitive that owns the whole figure, meaning it created that figure
in this call, may take gridspec space. `compare_row` is the worked example:
called with `axes=None` it creates the row and attaches one shared colorbar
across it with `fig.colorbar(ax=axes)`, and called with `axes=` it draws the
identical row but puts the shared colorbar in an inset off the last panel.
The figure looks the same either way, and the caller's layout survives.

## Style resolves at call time

Color, colormap, and savefig policy come from hwostyle, and every lookup
happens inside the call, not at import. Importing eyepiece activates no
style and touches no rcParams. Switching modes between two calls is
therefore always reflected in the second figure, and switching modes between
building a figure and saving it is honored by `save_fig`, which fetches the
policy fresh every time.

The practical rule this comes from is that a style library rebinds its own
module globals when a mode is activated, so a reference captured at import
time silently freezes whichever mode happened to be active then. Nothing in
eyepiece holds such a reference.

A primitive called with no style applied at all still produces a sensible
figure. Colormaps fall back to the light-mode definitions, and palette
colors follow the matplotlib environment the caller is already in: a
customized `axes.prop_cycle` is used as-is, and only an untouched factory
default falls back to the light palette. Working on bare matplotlib is
supported, not merely tolerated.

## Image display defaults to nearest

Every `imshow` call defaults to `interpolation="nearest"` and
`origin="lower"`. Interpolating simulated detector data misrepresents the
pixels, smearing a single hot pixel into a plausible-looking blob and hiding
the sampling of the very grid the simulation computed on, so the default is
raw pixels. A figure that genuinely wants smoothing can ask for it through
`imshow_kw`, which merges last.

## What earns a place in this library

eyepiece grows by accretion, not by anticipation. A candidate primitive
qualifies when all of the following hold.

- **Two or more independent hand-rolled instances in real use.** Not two
  imagined callers and not the same figure copied twice, but two places that
  separately solved the same drawing problem. One instance is a script, and
  three are a primitive.
- **Arrays and floats only.** Inputs are `numpy.asarray`-able arrays and
  plain numbers, per the rules above.
- **No other library's identifiers.** Neither the argument names, the
  returned data, nor the drawn labels may name another package's classes,
  attributes, or vocabulary. A primitive that would have to track another
  library's API is that library's to ship, not this one's.
- **A fit inside this contract.** Ax-first, stateless, `PlotResult` or
  `MosaicResult` out, routed kwargs, no geometry outside the handed slots.
  A device that cannot be expressed this way is evidence about the device,
  not about the contract.
- **A sunset clause.** Fewer than two consumers after two release cycles
  means the primitive is deprecated. Being in the library is not permanent
  tenure.

Two shapes are explicitly out of scope. A function that owns and lays out a
whole figure, deciding panel counts and inter-panel annotation for one
specific analysis, belongs in the consumer's own plotting module, or stays a
script. A function that renders a particular library's types belongs in that
library, built on these primitives, which is the subject of the
{doc}`viz module convention <viz-convention>`.
