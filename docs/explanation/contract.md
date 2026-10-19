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

`PlotResult` carries `.ax`, `.artists`, `.update`, `.insets`, and a `.fig`
property that reads the figure off the axes. `MosaicResult` carries `.axes`
(the panel array, as `plt.subplots` returns it), the same `.artists` and
`.update`, and a `.fig` property that reads the figure off the first panel.

`.insets` is a tuple of the axes a primitive created beside the one it drew
on, for the caller to fill: the insets `curve_insets` stands over the marks
of a curve, or the panels `rail_panels` hangs under the planes of a rail.
They are axes, not artists, so they never go under an artist key, and the
tuple is empty for every primitive that creates none. A dial drawn by
`phasor(at=...)` is different: there the inset is the axes the phasor draws
on, so it is the result's `.ax`.

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
axes. `convergence`, `signed_trace`, and `hist_fill` return an `update(k)`
that reveals the first `k` samples under the limits the full data set on the
first draw; `ruler` an `update(p0, p1, text)` that moves its arrow and label;
`overlay_line` an `update(x, y)` that moves its path; `fading_track` an
`update(xy)` that redraws its ramp along a new path; and `step_list` an
`update(current, done)` that moves the reader's place. `kymograph` returns
the `update` of `imshow_log`, which it draws through.

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

See {doc}`clock` for the motion contract and timing rules.

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

## Saves are deterministic

`save_fig` writes no timestamp. SVG omits its `Date`, PDF omits `CreationDate`
and `Producer`, PNG carries no time chunk, and the SVG hash salt is fixed, so
two saves of one figure are byte-identical and a PDF does not change when
matplotlib is upgraded. A different figure gives a different file. An explicit
`metadata=` still wins key by key, and the provenance keys a stamp writes stay.
`stamp(fig, reserve=True)` reserves the stamp's band in the layout and returns
a `PlotResult`; its date defaults to the commit date of the script's
repository, else today.

## Moving geometry is not pixel-snapped

Agg snaps axis-aligned lines and patch edges to whole pixels, which makes a
still crisp and turns a slow move into one-pixel jumps. A move that changes
geometry therefore sets `snap=False` on the artists it moves for the frames
strictly between its ends, and restores each artist's own setting at `u = 0`
and `u = 1`, so outside a recording the rest state is pixel-exact against
the snapped still. This holds for `warp`, `arrow`, `angle_arc` and
`compass`; the ghosts of `flight`, `collapse` and `morph` are drawn only
mid-move and are never snapped. `Stage` changes opacity and color only, so
it has nothing to unsnap.

Inside `record`, which by default (`snap=False`) turns `path.snap` off for
the whole recording, every frame is drawn at sub-pixel positions, so a mark
on a slow tail glides instead of stalling, then jumping a pixel. The rest
frame of a move then matches a snapped still of the same state only to
within half a pixel, and a held frame's axes and spines are drawn unsnapped
too. `record(snap=True)` restores the per-move snapping above, and with it
the pixel-exact rest state.

## Signature moves keep a motion contract

Each signature move states what it lets move, what it must not touch, how it
arrives at rest, and the rest state it ends in. The tests pin these
contracts rather than the mechanism behind them.

- `head_draw`: only the drawn prefix of each path changes. It grows from the
  start of the path at the eased progress, and its tip moves at constant
  speed in display arc length for a linear ease, whatever the vertex spacing
  or axis scale. The last tenth of the drawn length brightens toward the text
  ink with a dot on the tip; over the last eighth of the path the head
  dissolves into the body color, so the move arrives at the plain line with
  no step in color or width. The source line's data, style and markers are
  never changed. At `u = 0` nothing is drawn and at `u = 1` the source draws
  itself, so the rest frame equals the plain still (pixel-exact outside a
  recording, within half a pixel inside an unsnapped one). Several lines
  stagger with `lag=`, as Manim's `lag_ratio`.
- `ruler_readout`: one knob drives a `head_draw` head, a ruler and a
  counter. The ruler and the counter are read off the head after it is posed,
  never interpolated beside it, so on every frame the counter shows the value
  at the drawn tip and the ruler's far end sits at the quantity of the tip.
  Only the head's prefix, the ruler's far end and the ruler's label move; the
  ruler's near end and the counter's slots stay still (the slots are reserved
  for every value along the path when it is bound). At `u = 1` the head is at
  the path's end, the ruler at full length and the counter at the exact end
  value: the still of the final quantities.
- `Stage.weigh(..., lift_pt=)`: a label arrives attached to its mark. Only
  the lifted texts and their boxes move, straight up into place, by at most
  10 display pixels over the whole arrival, on the same eased presence as
  their fade, so they land as they finish fading in. Their colors are their
  own, the ink of the mark they name. At presence 1 every label is exactly in
  place, so the rest state is the still. Labels and callouts only: a group
  holding a data mark raises, because a data mark never drifts.
- `write` and `equation(...).update(reveal="write")`: ink is laid down glyph
  by glyph in reading order. Each glyph first traces its outline with a thin
  pen (0.75 pt by default), then fills; progress is linear inside each
  glyph's window, the easing lives only in when the windows open, and a
  short front of four to seven glyphs is in flux at once. The filled glyphs
  sit on the text as Agg lays it out (hinted advances), so the hand-off into
  the real text is a cross-fade with no frame lighter than both its ends.
  Nothing outside the text moves. At `u = 1` the text, or the equation as
  made, draws itself: the rest state is the still.
- `circumscribe(style="flash")`: a comet `time_width` of the perimeter long
  passes once round the outline, from the top and clockwise, on the eased
  progress. Its head runs from the start of the outline to `time_width` past
  the end, so it grows out of the top, flies round at constant length, and
  shrinks back into the top, leaving nothing behind. It tapers from full
  width and opacity at the head to 40% width at the tail and adds no hue.
  Only the comet moves; the target is never touched. At `u = 0` and `u = 1`
  the frame is the still.
- `indicate(shape="plateau")`: the pulse rises on the ease over the first
  35% of the progress, holds its peak through the middle 30%, and falls
  symmetrically, so the emphasis is read at its peak. Only widths, sizes and
  the tint toward the text ink move, never positions or hue, and at both ends
  the artist is exactly as it was. Lines, patches and collections swell
  smoothly; a text swells in whole-pixel steps, because Agg hints and places
  text on the pixel grid.
- `click` and `Timeline.click`: a composite lands in one beat of 0.6 s on
  screen at any tempo, followed by at least 1.5 s of stillness the tempo
  never shortens. Over the beat the geometry's width swells by 1.55 on a
  plateau (up over 35%, held over 30%, down over 35%), the end dot grows from
  nothing over the first half, the number turns bold over a fifth of the beat
  from the start of the plateau (a cross-fade to a bold copy in place, never
  a step), and the ruler recedes from there to the end. Nothing changes
  position and no hue is added. At `u = 0` the figure is as it was; at
  `u = 1` it is the landed still, whatever path the knob took: the geometry
  at its own width, the dot set, the number bold and the ruler gone.

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

Image colormaps follow the same rule. A primitive picks its colormap by the
role of the quantity it draws, through `hwostyle.cmaps.<role>`, and never by a
map name, so a change of style reaches every figure at once. The roles are
`intensity` (idealized model images), `readouts` (detector readouts and processed
data products), `residual` (signed difference images), `opd` (signed wavefront
error), `statistic` (signed statistics and SNR maps), `phase` (cyclic phase),
`probability` (posterior and scatter density), `pupil` (pupil transmission), and
`mask` (validity masks and binary apertures). The sequential roles are oriented by
the page, with low values at the page end: dark on a dark page, and pale on a
white page for `probability` and `pupil`. `pupil` and `probability` share their
colors, so they never appear in the same figure: pupil belongs to optics figures
and probability to inference figures. `cmaps.high_dynamic_range` and
`roles.envelope` are deprecated and warn, so new code uses `intensity` or
`readouts`. When the installed hwostyle predates a role, eyepiece falls back to a
plain matplotlib map for `statistic` and `pupil`, and nothing else.

Video gets one derived variant of the dark roles. `eyepiece.use_video()`, after
`hwostyle.use("dark")`, makes `role` and the motif colors resolve from the
``"dark-video"`` table (`eyepiece.video_roles()`), and `motif_faults` accepts the
mode by name. The table is computed from the dark roles, never picked: each role
keeps its Oklab hue, its BT.709 chroma (the length of Cb and Cr in 8-bit code
values, the part 4:2:0 encoding halves in resolution) is capped at 75, and a dim
role is lifted to a contrast of 5 on black. `alert` keeps a cap of 85 and `disk`
and `prior` a floor of 4.5, because at the plain limits they fall within the
`motif_faults` fault distance of `posterior` and `positive` for a color-vision
deficiency; the tests hold the variant to adding no motif fault or warning that
the dark mode does not have. The palette cycle, the colormaps, the neutrals and
the light and paper modes are unchanged, and nothing changes until
`use_video` is called.

A primitive called with no style applied at all still produces a sensible
figure. Colormaps fall back to the light-mode definitions, and palette
colors follow the matplotlib environment the caller is already in: a
customized `axes.prop_cycle` is used as-is, and only an untouched factory
default falls back to the light palette. Working on bare matplotlib is
supported, not merely tolerated.

(motifs)=
### Motifs

An object that recurs across figures (the star, a planet, a detector, a
reference plane) is a `Motif`, and a `MotifSet` holds motifs by key, never by
position. The key ``"planet"`` is the same planet whatever else a set lists,
which an order-based assignment cannot promise across the scripts of one talk.
A motif stores a hwostyle role name or a neutral rank and never a hex value;
`motif.color` resolves it through `eyepiece.role` at the moment it is read, so
a mode switch reaches every figure, and a saved figure state keeps the motif
key and not a color that goes stale. Beside the color a motif carries its
marker, line style, hatch, arrow style and font style: the non-color channels
that tell it apart when its color is shared, dimmed or lost to color vision.
`signature()` returns them, `line_kw`, `marker_kw` and `patch_kw` turn the
motif into matplotlib arguments, and `derive(**changes)` makes a chapter's own
variant. `MotifSet.default()` reads the table from `hwostyle.motifs`, and an
older hwostyle gets a smaller stand-in table, as a missing role does.

A figure spends at most four hues, and hwostyle owns that budget with the
table. A fifth or sixth hue needs a second channel, a seventh is never used,
and the pairs hwostyle declares never to co-occur (exozodi with prior, local
zodi with reference, and the image roles that share a map family) stay out of
one clip, because viewers remember a color across a clip and not only across a
figure. On an image a motif is its glyph in overlay ink and not its hue. The
cue that marks the subject of a step is a form (a ring, a band or an underlay)
drawn in the text color. It spends no hue, it never recolors the motif it
marks, and it is never a line, a marker or text. The contrast a dimmed step
keeps is 0.35 in dark mode and 0.5 in light and paper mode, which `Stage`
resolves when `dim` is None; an explicit `dim` wins.

`eyepiece.testing.motif_faults(motifs, mode=...)` checks the subset a figure or
clip draws with (`MotifSet.only`), not the whole default table. It reports two
motifs that share a color and every other channel, a never-together pair in
one set, hues past the budget without a second channel, hues that fall close
in CAM02-UCS under normal vision or a color-vision deficiency, hues that fall
close once dimmed together (advisory), and a cue that is not a form in the
text color or that matches the color of a motif in the set. `motif_margins`
returns the distances behind these. `assert_motifs` raises `MotifError`. The
checks need `colorspacious`, which the `test` extra installs.

See {ref}`which-cast` for the distinction between a role cast and a prepared source cast.

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
{doc}`viz module convention <../viz-convention>`.
