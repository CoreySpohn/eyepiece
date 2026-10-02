# The primitive contract

Every plotting primitive in eyepiece obeys the same rules about what it
accepts, what it draws where, and what it hands back. The rules exist so a
reader can predict a function's behavior from its signature alone, and so a
figure assembled from several primitives comes out looking like one figure
rather than a collage. This page states the contract; the API reference
states the arguments.

eyepiece has two ways to draw. The ax-first primitives at the top level
(`imshow_log`, `compare_row`, `sky_fan`, and the rest) take arrays and draw
them in one call. Prepared views, in `eyepiece.prepared`, write a scene down
as data once, and the renderers in `eyepiece.mpl` and `eyepiece.manim` draw
that data as a Matplotlib figure or as native Manim objects. Most of this page
applies to both; the {ref}`prepared-contract` section states the rules
specific to the second path.

## Arrays in, plain floats for scalars

An array argument is anything `numpy.asarray` accepts, and it is converted
on entry. A NumPy array, a nested list, a JAX array, a masked array, or a
memory-mapped slice are all valid, and none of them are given special
treatment. No primitive accepts a simulation-library type, checks for one,
or imports a package that defines one, which is what keeps the library
usable from any code that can produce a number.

The same firewall holds for prepared views. A record holds borrowed numeric
arrays and plain metadata, never a live simulation object, and neither
renderer ever receives one. The records borrow rather than convert: a
float32 cube or a read-only memory map is stored as given, without a dtype
promotion or a copy.

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

The prepared renderers return their own result types, described under
{ref}`prepared-contract`: `eyepiece.mpl.render` returns an `MplResult`,
`eyepiece.mpl.animate` an `eyepiece.Animation`, `eyepiece.manim.render` a
`ManimResult`, and `eyepiece.manim.animate` a `ManimClip`. A preparation
function returns a record or a `Sequence`, not a figure.

### The artist vocabulary

`.artists` is a plain dict whose keys are drawn from `ARTIST_KEYS`, a fixed
vocabulary exported at the top level. It is a convention rather than an
enforced schema: a primitive populates only the keys for artists it actually
drew, so a missing key means "not drawn here", never "drawn and hidden".

Each key holds either a single matplotlib artist or a list of them. A list
usually means one entry per panel, in panel order, from a multi-panel
primitive drawing the same kind of artist in each. `lines` is the standing
exception, defined below as several artists on one axes, and `fill`, `text`,
`ellipse`, and `arrow` are read the same way when a single-axes primitive
draws several of them. A key names the kind of artist, not how many axes are involved.

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
- `ellipse`: an `Ellipse`, or another `Patch`, marking a region. A list holds
  either one per panel or several drawn together on one axes, as in a rail's
  lenses.
- `arrow`: a `FancyArrowPatch` drawn as a vector, such as a phasor on the
  complex plane. A list holds either one per panel or several drawn together
  on one axes, in the order they were given, as in a chain of phasors.
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
existing `AxesImage`, creating no new artist. `imshow_diverging` returns an
`update` as well, which converts the new data to float and calls `set_data`
under the symmetric norm of the first draw, so every frame keeps the same
zero and the same scale. `rail` returns an `update(highlight=...)` that
relights its planes by restyling the existing artists, keeping the tones of
the first draw, so a frame loop moves the highlight without clearing the
axes.

An `update` reuses the norm built from the first draw. Values outside that
norm are not an error, they render clipped to the colormap's end colors, and
the norm is not rescaled. When the data range is expected to move, pin it up
front with `vmin` and `vmax`, or call the primitive again.

This is deliberate, and the reason is integrity rather than convenience. A scale
that follows the data redraws every frame at full height, so a quantity that
falls by an order of magnitude reads as one that never moved. Reusing the first
draw's norm makes an animation's default behavior honest and makes rescaling
something the caller has to ask for. The same reasoning applies to axis limits,
which no primitive rescales after its own call: an animated figure's data scales
are the caller's contract, and pinning them from the union over all frames is
part of building the figure rather than part of the frame loop. `record` warns
when one drifts anyway.

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

`eyepiece.mpl.render` follows the same rule. On caller axes its colorbar is
an inset beside the image and no layout engine is installed on the caller's
figure, so on a figure created without a layout engine the colorbar label can
fall outside the canvas. A figure created with `layout="constrained"` keeps
it on the canvas.

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

The prepared renderers resolve style at an explicit moment instead of at
each draw. `eyepiece.style.snapshot_profile()` copies the current palette,
colormap tables, text settings, and reference color into a read-only
`RenderProfile`, and a rendered result keeps that profile for every later
update. A mode switch therefore changes the next snapshot and never an
existing result, which is what lets a movie or a slide play to the end in
the colors it started with.

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

## An image without an extent has no ticks

Pass an `extent` and the axes carry real coordinates, so they get ticks.
Leave it out and the axes are raw array indices, which are almost never what
the reader is meant to measure: they frame the picture in numbers whose units
live in the colorbar instead. So every image primitive -- `imshow_log`,
`imshow_diverging`, `compare_row`, `triptych`, `show_field` -- drops the tick
labels when `extent` is None, and restores them the moment you pass one.

The rule matters most for the panel that is *nearly* the same as its
neighbour. A row of images where one panel happens to carry an extent and the
rest do not should not read as two kinds of figure, and answering "no extent"
differently per primitive is how that happens. Ask for indices back with
`ax.set_xticks(range(0, n, step))` after the call if you genuinely want them.

## A size that means a number needs a key

Marker size, opacity, and line width become quantitative channels the moment a
caller drives them from data, and a reader measures them whether or not anyone
intended it. Three rules follow.

A size argument names its unit, like every other physical scalar here.
Matplotlib's `scatter` takes `s` as an AREA in points squared while `plot` takes
`ms` as a DIAMETER in points, so a helper producing one and a parameter
consuming the other will square the encoding silently and the figure will still
render. Naming the parameter for its unit makes the mistake visible at the call
site.

A channel carries one quantity. Where a second cue has to share it, such as a
depth swell over a size that already encodes a physical radius, the cue's full
dynamic range stays a small fraction of the smallest gap the encoded quantity has
to show, and that fraction is computed rather than estimated.

A non-linear encoding is declared. A geometric size map is often the only legible
choice, and it stays legitimate exactly as long as the figure carries a key from
which the reader can recover it. Without one the reader assumes linear.

## Curves are labeled at the data, not in a corner

A legend asks the reader to hold a color in working memory, cross the figure, and
come back. Below roughly five curves that trade is not worth making, so a
primitive drawing a small fixed set of named curves labels them in place: at each
curve's right end where the curves separate there, inside a band where the
primitive draws filled regions, or at a caller-chosen x otherwise. Labels take
the curve's own color, which is what makes the word and the mark one object
rather than two.

A primitive that labels inline exposes the legend as the opt-out, so a caller
drawing many curves, or composing a small multiple around one shared key, can
still ask for one. It never solves label placement per draw: a label that moves
between animation frames is worse than the legend it replaced, which is the same
reason a "best" legend location is not used.

A label is hidden, not shrunk and not displaced, when its curve's local
separation falls below about two text heights.

## A whole and its parts are not three peers

When a primitive draws a quantity alongside the components it decomposes into,
the total is not a third measurement of equal standing. Drawn as three peer lines
it asserts three independent quantities, and when the components converge the
lines overprint into a color belonging to none of them. Draw the total as a wide,
pale envelope beneath its own components: the components stay legible on top, and
a component sitting inside the envelope is the additive identity made visible.

(prepared-contract)=
## Prepared views

A prepared view is a frozen tree of records in `eyepiece.prepared`: images,
curve and track panels, panel groups, and the paths, points, regions,
reference lines, and labels drawn on them, plus `Sequence` for replayable
states over physical time. The {doc}`prepared views guide <prepared-views>`
documents the records; the rules they follow are these.

### Preparation is pure

A preparation module turns scientific objects into records and does nothing
else. It imports `eyepiece.prepared`, which loads only the standard library
and NumPy, and it never imports a renderer, activates a style, or touches
Matplotlib state. It performs every scientific step once: extraction, unit
choice, coordinate conventions such as right ascension increasing to the
left, scientific floors and normalization, masks, and traces computed from
scientific values before any display clipping. It also owns the transfer from
a device array to host memory. The records it returns are then the only
input a renderer sees, and no update recomputes any of it.

### Masks and validity

A sample is invalid when a mask marks it, when an explicit `valid` array
marks it, or when it is not finite. A masked array's mask is captured before
the data is coerced, and a mask and an explicit `valid` combine by logical
and. Invalid samples contribute to no bounds or statistics, and they are
drawn in the profile's bad-sample color, which is distinct from every
colormap entry. A valid sample outside the display range is clipped to the
nearest end of the scale and stays visible. Masking and clipping are never
the same appearance.

Display bounds come from the valid samples of the whole sequence, or from
limits the caller supplies, and stay fixed for every still, strip, and frame
made from it. An all-invalid sequence and a constant one both raise rather
than invent a scale, and an unsupported scale kind or invalid bounds fail
before anything is drawn.

### Return types and named parts

A renderer's result exposes what it drew by the element IDs of the records.
`MplResult.parts` maps each ID to its Matplotlib artist, and
`ManimResult.parts` maps each ID to its mobject. Derived parts take suffixed
keys: `"<image id>/colorbar"` in both renderers, `"<points id>/xerr"` and
`"<points id>/yerr"` for error bars in both, and `"<view id>/frame"` and
`"<view id>/axes"` in the Manim renderer only, since only there are the axes
box and the axis decoration separate objects. An element ID that collides
with a derived key is rejected before drawing.

A `source_id` is not an element ID. It names the scientific source a mark
belongs to, so many independently addressable marks can share one source's
color and marker.

A result's `update(view)` takes a new state of the same topology, validates
the whole of it, and prepares every change before applying any, so an
invalid update leaves the output exactly as it was. Adding or removing an
element, or changing a shape, scale, axis specification, point count, or
source, needs a new render.

### Ownership is renderer specific

The two renderers share the records, the display mapping, the schedule, and
the topology rules, and nothing else. Each returns objects native to its own
toolkit, and the consumer customizes them there: through the Matplotlib
artists in `MplResult.parts`, or through the mobjects in `ManimResult.parts`.
There are no routed keyword dictionaries on the prepared renderers and no
shared abstraction over artists and mobjects. Exact figure dimensions,
slide placement, camera motion, and reveal order stay with the consumer.

Updates never change what the consumer owns. A Matplotlib update never
changes an artist's visibility, and a Manim update never sets a part's
opacity, so a part the consumer hid stays hidden until the consumer shows it.

### Cast and profile are explicit

Appearance enters a render through two arguments. A `SourceCast` fixes each
source name's palette slot and marker from its position in a declared list,
independent of the order in which sources are drawn. A `RenderProfile` is a
read-only snapshot of the style, taken when `snapshot_profile` is called and
never changed afterwards. When either is omitted, the renderer builds it once
at render time and keeps it for every update: a cast from the tree's source
IDs in encounter order, and a profile sized for a paper figure from the
Matplotlib rc settings in `eyepiece.mpl`, or with 24 pt text and 1.5 pt
strokes in `eyepiece.manim`. A document with more than one figure declares
its cast once and passes it everywhere.

### Time is sampled, not interpolated

A `Sequence` holds instantaneous samples at finite, strictly increasing
physical times, and rejects any other declared sample kind. Every frame is a
complete state. Evaluation is left sample-and-hold, and a clock label reports
the acquisition time of the sample shown. Presentation duration and frame
rate choose an output schedule, `ceil(run_time * fps)` times spanning the
first sample through the last, that both renderers use; neither renderer
interpolates between samples or reevaluates a simulation.

### Direct model binding stays in the consumer

A consumer that animates native Manim geometry straight from a scientific
model, evaluating the model at each playback time, writes that binding in its
own script or in the library that owns the model. It needs no import from
eyepiece or from any viz module, and eyepiece does not inspect model objects
to support it. The restriction that updates compute no science applies to the
prepared renderers, not to that explicit consumer code.

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
