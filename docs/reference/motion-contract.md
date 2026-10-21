# Motion contract

This page lists `MotionContract` fields and the two span types used to place and
check moves. See {doc}`../explanation/clock` for the reasoning.

## MotionContract

`MotionContract` is a frozen record of what a move promises about its own
artists. `PlotResult.contract` and `MosaicResult.contract` hold it, or None
when the primitive declares no motion contract. The new `contract` argument is
keyword-only, preserving positional arguments in result subclasses.
`Stage.contract` describes its staged artists. The artists are the ones the
figure draws, not copies made for the checker. Pass a result directly as
`ep.testing.motion_faults(fig, at, times, moves=[result])`.

- `moves` is a tuple of artists whose geometry may move or change shape;
  the checker also accepts the older name `artists` in a declaration.
- `rest` is a tuple of artists whose geometry stays still; opacity and color
  may change.
- `groups` is a tuple of artist tuples that move rigidly, or dicts of the
  form `{"artists": (...), "scales": True}` for uniform scaling.
- `may_overlap` is a tuple of artist pairs whose overlap is intentional.
- `settles` is always True and cannot be set by the caller: a move must be
  continuous into its rest state.
- `arrives_after_motion` is a tuple of text artists that fade in after the
  geometry has settled. Path glyphs are not text artists.
- `rectilinear` is a tuple of bars, rules or boxes that stay axis-aligned
  rectangles at every progress, including intermediate states.
- `draws` is the draw group of a run of exchangeable draws: the artists that
  change in a cut at each draw by design (`hop` declares all of its artists).

## Declaration scope

Declare only what is true. Empty tuples make no promise. A warp may deform its
inputs, a growing compass is not rigid, and a replacement equation term may
stretch by different factors on its two axes, so none earns a rigid group
merely by sharing one progress.

A transform declares matched glyphs as rigid only when their relative layout
stays the same; its rules remain rectilinear even when their lengths change.

A word reveal by letters changes its visible extent, and `reveal` can choose
that mode on each call, so a word contract allows changing extents across both
reveal modes.

A write declares nothing: its real text is hidden until the end and its
renderer-only ghost has no independently traceable geometry, so no drift check
could compare anything. An inset's spines and virtual layout artists likewise
do not become traceable marks by being declared.

These declarations describe drawing and do not change it. The checker reads
`moves`, `rest`, `groups`, `may_overlap`, `rectilinear` and `draws`; `settles`
and `arrives_after_motion` also state promises that a caller can check
directly.

## A span and a check window

`eyepiece.clock.Span` is the plan span returned by `Timeline` verbs such as
`enter` and `move`. It carries a name, start and end times in seconds, the
knobs it writes, and, once resolved in a `Plan`, frame bounds `k0` and `k1`.
Use it to place another span or to select a settled still from the plan.

`eyepiece.testing.Window(start, stop)` is the checker's declaration of a move's
time interval in seconds. Use it to sample every frame with `motion_faults`
or to declare when an artist moves. It carries no knobs or frame bounds.
`ep.testing.check_clip(moves=...)` accepts either a timeline span **by its name** or an
`eyepiece.testing.Window` value, for example `moves={curve: "reveal"}` or
`moves={curve: ep.testing.Window(0.0, 1.0)}`. An `eyepiece.clock.Span` object itself is not
accepted there; pass its `.name`. See {doc}`../how-to/check-a-clip` for the
complete check and {doc}`../how-to/stills-from-the-encode` for still selection.
