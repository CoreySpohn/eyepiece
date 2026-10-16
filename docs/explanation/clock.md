# The clock

A clip has physical samples and a presentation schedule. Keeping them separate
lets the viewer follow a move without changing the data it shows. The
{doc}`primitive contract <contract>` describes the arrays, artists, and scales
that those moves use.

## Time is sampled, not interpolated

A `Sequence` holds instantaneous samples at finite, strictly increasing
physical times, and rejects any other declared sample kind. Every frame is a
complete state. Evaluation is left sample-and-hold, and a clock label reports
the acquisition time of the sample shown. Presentation duration and frame
rate choose an output schedule, `ceil(run_time * fps)` times spanning the
first sample through the last, that both renderers use; neither renderer
interpolates between samples or reevaluates a simulation.

## Presentation time is a separate clock

A `Timeline` interpolates how a figure is shown, never what was simulated. Its
knobs are presentation quantities: a title, an angle a construction has turned
through, the width of a camera's view, the progress of a morph. A move may
change them smoothly between holds, and the frames it produces drive a draw
function or a camera. It never interpolates between the samples of a
`Sequence` or stands in for a physical time, so the two clocks stay apart: the
`Sequence` says what was measured when, and the timeline says when the viewer
sees it. A camera's moving limits are declared to `record` with
`free_limits=`, which exempts only those limits from the drift check and
still checks that axes' color scale.

## One clock places everything timed

The named span is the only timed primitive: a hold, a subject entering or
leaving, a move, a caption, a pulse, a ring, a line of narration and a
section are all spans of one `Timeline`, and the knobs a span writes are all a
draw function reads, through `Plan.value(k, knob)`. A draw function never sees
time. Times are exact sums, so `Time` handles place one span after another
without drift. One rule turns seconds into frames everywhere: a span from `s`
to `e` seconds covers frames `k0 = frame_count(s)` up to
`k1 = max(k0 + 1, frame_count(e))`, its progress on frame `k` is
`ease((k - k0 + 1) / n)` and exactly 1 from `k1` on, and a time that falls on
half a frame rounds up. Two spans that write one knob on the same frame raise.
The other checks are faults (text arriving over motion, a caption leaving
before it is read, a span that rounds to no frames); a strict timeline raises
them and any other records them in `Plan.faults`, with information-level
entries for what the clock did on its own (a skipped pulse, a span truncated
at the end of the clip).

Names live in two places. Span and knob names share one namespace, and a
repeat raises. Sections and lines share a second one, so a span may carry the
name of the line it sits in (the `square` line may hold the `square` entry),
but two sections or lines may not share a name. No user-given name may hold
`:`, because event names and cue marks use it as a separator.

A span truncated at the end of the clip keeps its speed: it plays the frames
it has and the rest are never drawn, and the information-level fault says by
how many frames. A span that a cue places can start later than its word when
the cursor is still busy; the wait is an information-level `late` fault up to
`Timeline(max_late_s=0.5)` seconds and an error beyond it, which a strict
timeline raises, naming the span, the cue word and the seconds. In a narrated
clip speech reads the captions, so a caption that the next caption replaces
leaves without a read-time check; a silent clip keeps it.

Narration is placement only. `Narration.place` lays one line on the clock and
is pure, a cue is the start of its word with no lead, and the lead of each
kind of move (a subject arrives 0.1 s ahead of its word, a pulse is centered
on it, a ring starts on it) belongs to the clock, applied once. A line lasts
until its speech and padding end or its last extending span ends plus 0.6 s,
solved in one pass as the spans are placed; `Timeline.narration` returns the
`Narration` with those starts and ends for `audio_track`, `mux` and the
captions. The same function builds the silent clip: a pulse or ring placed
only by a cue is skipped and recorded rather than placed at a guess.

## Stills come from the encode

`record(stills=...)` writes chosen frames as PNGs while the sinks encode, from
the figure exactly as the sinks grabbed it at that frame, so a still is the
same pixels as that frame and costs no second pass. A time is converted to a
frame by the same rule as every span boundary (`frame_count`, half a frame
rounds up), a span name resolves through the `Plan` to the span's settled
last frame, and a request the recording never reaches is an error rather than
a missing file. `on_rescale` is one of `"warn"`, `"raise"` and `"ignore"`;
`"raise"` raises `ScaleDriftError` and removes every output the call began, so
a failed run leaves nothing that could pass for a finished one. The last
frame of a span is `(span, "last")`, never `span.end`: the end time is the
first instant after the span, and a time that rounds to the clip's frame count
is past the last frame and is the "never written" error. A recording given its
`plan` also warns once, as it closes, about every knob the draw function never
read and every knob that never changes (the always-present `pulse.*` and
`ring.*` knobs excepted), because a clip about motion needs a knob that moves.

## Attention moves sit on the figure

`flight`, `callout`, and `leader` say where to look; none of them is data.
Their positions are `Anchor`s resolved in display space at draw time, and
their artists are children of the figure rather than of an axes, so they
never join autoscaling and a recording's drift check never sees them. Each
returns a result whose `update(u)` sets progress absolutely and whose rest
state, the state it is made in, is the one a still shows: a flight is
absent, a callout, a leader and a link are fully drawn. Drawing never
raises, so a still or a PDF save is not aborted half way: a leader or link
that breaks the connector rules of the evidence guide is found by its
`faults()` or by `audit(fig)`, which raises `AttentionFault` for the strict
ones.

`Stage`, `build_axes`, and `collapse` follow the same rule. A stage sets
each named group to an absolute presence and contrast computed from the
colors and opacities the artists had when the stage was made, so levels
never compound and setting a state twice draws what setting it once does; a
dimmed group blends toward the background it sits on, which reads the same
on a light figure and a dark one. An axis build only clips and fades what
the axes already draws, so it moves no limit. A collapse is a figure-level
copy, absent at rest. `indicate` and `circumscribe` are transient: at both
ends of their progress the frame equals the frame without them, and
`indicate` pulses from the artist's colors as they are when each update
starts, so it composes with a stage that is set first. `term_row` places one
text per term at draw time, so its terms are anchors and stage members, and
a changed term re-lays the row. `readout` shows a running number as an
integer part right-aligned to a fixed decimal point and a fraction left-aligned
from it, each in a slot as wide as the widest string it holds, so a count
never moves the label; its shown value is the interpolated value rounded to the
format's precision (monotone, with no negative zero), its slots grow only when
`set` is called with a value that needs more room, and at ``u = 1`` it shows
the exact formatted value last set. The eases `back_out` and `running_start`
leave [0, 1] between their exact ends, and a move may take any ease as a kind
name or as a callable from progress to eased progress, for example
`functools.partial(ep.ease, kind="back_out", strength=0.2)`. A move clips only
the progress it is given. Where an eased factor would become an opacity, a
level, a fraction of a path, a color blend, or a line width, the move clips the
factor itself before blending, so a position or a warp may overshoot and
nothing else can go invalid. A move snaps to its ends on the progress, never
on the eased value. A `readout` clips its eased value to [0, 1] before it
interpolates, so a count passes neither its old value nor its new one. `words` cuts one line of text into one
text per word, laid out once as a single string with each word's extent taken
from the renderer, so its words are anchors and stage members too, math in
``$...$`` stays whole, a color is given by word index or by the word itself (a string key colors
every occurrence), and the joined line at rest is the single text, pixel
identical; its reveal is by opacity or by letters in place, never by motion. A flight can carry a segment or a bar and
turn it. `morph` is a figure-level ghost like a flight, visible only
between its ends, whose geometry at the ends is the two paths resampled by
arc length (and, for a closed pair, the second rolled and possibly reversed
to lie on the first). `equation` lays mathtext out as glyph outlines, one figure-level patch per glyph or rule, in
display pixels, with named groups (``{{name:...}}``, found by hiding the group with ``\phantom``
and reading what disappears, and raising when the hidden group would move any other glyph); a group is an
artist with the box of its glyphs, laid out where the equation is, so anchors, callouts,
outlines, pulses and a `Stage` act on it unchanged. Its ``update(u, level=1.0)`` writes the
terms on, left to right, and at ``u = 1`` the equation is whole; `level` is an opacity and never
geometry. `transform` carries one equation into another, owns the visibility of both, rests at
``u = 1`` (the second equation shown, the first hidden) and draws the first alone at ``u = 0``,
each pixel-identical to that equation as placed. Matching is by glyph identity (font and
glyph, never the character code): inside a tag both equations share (or a `key_map` pair), in
left-to-right order, then across glyphs in no tag, and a tagged glyph never pairs outside its tag;
equations that share no tag pair every glyph by order. `align` (default ``"auto"``) places the
second equation so that the sideways shift most groups share is removed and those glyphs keep their place in the first, which moves the second
equation's layout at every `u`, the rest state included. Matched glyphs move by group, never
one by one: all the matched glyphs of one tag, or one run of neighbouring glyphs in no tag, share
one eased progress, staggered left to right by the group's place in the first equation, so a word
moves rigidly. A group arcs on an optional arc, up when it moves right and down when it moves left, scaled by its sideways travel
(``min(1, travel / fontsize)``), so two groups that swap places pass each other and a group that does not travel sideways does not arc. A tag both
equations have whose glyphs do not all match is a replacement: what is left of it in the first is
carried as one box that moves and scales onto the box of what is left in the second, fading out
while the second, drawn inside the moving box, fades in. `replace="morph"` (the default; `"fade"` keeps
the box) changes the outlines instead, aligned as ManimCE's `Transform` aligns paths: every segment is
a cubic, the glyphs of each side pair in reading order and the side with fewer is padded with
stacked copies of its own glyphs (which start or end invisible), each pair gets the same contours
(a missing contour is a null contour at the last point of the last one, so a hole grows from or
shrinks to a point) and the same curve count (curves split evenly in their parameter), and the
control points, fill color and opacity mix linearly in display space on the group's eased
progress. Both modes rest at the two equations as drawn, take the motion part of the beat, and
leave matched glyphs and unmatched glyphs as they are. Contours pair by index, as in Manim, so a
morph that pairs an outline with a hole flips that contour's winding part way and the nonzero
fill can briefly close a hole. Glyphs with no counterpart at all
fade out in the first `fade` of the beat or fade in in the last, so when anything fades in
the motion takes the first ``1 - fade``, and new text arrives after the motion has settled. A
fraction bar stretches as a rectangle, each corner moving to its partner, so its ends stay square. A transform's narration cue is ``{move:name}``, whose time
starts its ramp. Not supported: ``usetex``, several math spans in one string, escaped braces
(they raise), and the size animation of ``\left ... \right``.
`write` is a figure-level ghost of a text, visible only before ``u = 1``: it
hides the real text until the end, where the text is shown with the
visibility it had, so the rest state is the text as drawn. Its outlines are
laid out as the text lays itself out, its stroke and fill take the text's
color and opacity at draw time, and its eased tracing and fill are clamped
to [0, 1]. `arrow`, `angle_arc` and `compass` are drawn marks, not
overlays: each is one patch (an arrow, an arc whose head is tangent to the
arc, or a north and an east arrow) added without joining autoscaling, and
`update` sets its growth `u`, its ends or angles, and a `level` opacity on the
same artists. An arrow grows from its tail, or from its middle when it has
two heads; an arc sweeps from its start angle; and a compass has no default
side for east, because a compass drawn with the wrong side is a mirrored sky.
`follow_zoom` is a figure-level inset: an axes in the panel that draws the
panel's own artists again through a window centered on its target, a frame box
on the figure, and two connectors audited as links. Its geometry is recomputed
at every draw, so following needs no per-frame call; the window is exactly
`zoom` times closer on both axes and is never clamped to the panel. Its
`update(u, level)` fades all of it in with `u`, `level` is an opacity only, and
at ``u = 0`` the figure is pixel for pixel the one without the zoom. The parent's
artists are borrowed for each draw and handed back as they were.
`loc="auto"` holds the corner farthest from the target, switching only when another is farther by a quarter of the inset's side; the frame, box and connectors carry a halo in the panel's background so they read on dark and light images.
The rest state of each, the one a still shows, is the
figure as drawn: every group at full weight and every axis finished.

`warp` differs from the figure-level moves: it reshapes the artists it is
given, in place and in data coordinates, so it creates nothing and an artist
keeps its place, style and stage membership. Its rest state is the start,
``u = 0``, the geometry the artists had when the move was made (a path that a
pointwise map subdivided is put back as drawn); at ``u = 1`` the geometry is
exactly ``M p`` or ``f(p)``. Its `level` changes nothing: the artists'
opacity belongs to the caller or a `Stage`, so dimming a warped group never
undoes the warp. It leaves axis limits alone.

`eyepiece.testing` is the readability check for such a clip, a submodule
imported by name rather than a top-level export, because it belongs to a test
suite and not to a figure. `readability_faults(fig, at=pose, times=...)`
poses the figure at each sample time, draws it, and returns a list of faults
in output pixels: two visible texts, or a text and another's backing box,
whose extents overlap by more than a tolerance; text or marks the figure edge
clips; text under the venue's minimum height; and text drawn over a data mark
of its own color. It does not repeat `audit`, which owns the connectors, and
a label with an opaque backing box is not reported over marks. The result is
deterministic and the figure keeps its dpi. `assert_readable` raises
`ReadabilityError` with the report.

`meaning_faults(fig, at=pose, times=..., keys={gid_prefix: meaning})` checks
what the colors say, which `readability_faults` does not. Artists are tied to a
meaning by `gid` prefix (longest prefix wins, and an artist with no `gid`
inherits its nearest ancestor's). Colors are measured as drawn, after artist
alpha and any `Stage` dim, over the background, in CAM02-UCS under normal
vision and the three Machado types at severity 100. It reports five kinds:
`same-frame` (two meanings in one color within `fault_de`, or within `warn_de`
as a warning, downgraded when a marker, line style or hatch also differs),
`rebound` (one color carrying different meanings at different times, or across
the clips of `meaning_faults_chain`, unless declared with `rebind`; clustered
on the colors before opacity, and a `sharing` pair seen in one clip is always a
fault), `cue-form`
(a key named in `cue_keys` drawn as a line, as text, as an opaque patch over
its subject, or in any hue; the cue is a band, ring or underlay in the text
color), `unkeyed` (a chromatic artist with no meaning, advisory unless
`strict`), and `dimmed` (an advisory: two keys that share a hue only once both
are dimmed toward the background). `rebind` takes times, in `meaning_faults`
and in `meaning_faults_chain` alike. `sharing` pairs may share a color but not
a clip. A finding is waived only by naming it in `waive` with a reason, and
stays in `meaning_report`, whose all-pass wording is "no tier 1 or 2 findings".
`assert_meaning` raises `MeaningError`. It needs `colorspacious`, in the
`test` extra.

`motion_faults(fig, at, times, moves=..., rest=..., groups=...)` is the same
check for the path between two states rather than the states themselves.
Given an `eyepiece.testing.Span(start, stop)` it poses the figure at every frame of the venue's
frame rate (30 per second for each venue; `fps=` overrides), always ending
exactly on `stop`, draws it, and traces every visible artist, equation glyph
paths and collections included, with the rendered frame when a pixel check is
asked for. It returns `MotionFault` records, sorted by time, of eight kinds.
`jump` is a frame step far above the typical one and not echoed by its
neighbours, a step into the rest state (the end of a declared move, else the
last step that changes anything) that is both above half the ratio and twice
the step before it, or, for a clip too still to have a typical step, one
artist that moves a large share of its size in one frame. `dead` is a declared
move whose region never changes, or whose artists are never visible. `drift`
is an artist listed in `rest` whose box or ink centroid moves by more than
`tol_px`. `tear` is a `groups` set whose members, by their ink centroids,
change their spacing, so a stagger or a drift of one member shows and a rigid
rotation does not; a group declared with `scales=True` may scale as a whole.
`overlap` is two artists, text or declared in `moves`, whose visible parts
cover each other beyond a tolerance while both travel (judged over a tenth of
a second, so the answer does not depend on the frame rate) and are not listed
in `may_overlap`. `outside` is any traced artist, a glyph path included, past
the frame, and any text or declared artist cut by its clip box or clipped away
entirely; a clip box that only widens, and ends by uncovering the artist, is a
reveal and not a cut. `spike` is a hop off the path, or a one-frame excursion,
or a kink in speed (measured over elapsed time) on an artist that keeps its
shape; it is off unless `spikes=True` or `kinds` names it, because a small hop
is sometimes the intended motion. `shape-class` is a patch or line listed in
`rectilinear` (a fraction bar, a rule, a box) whose drawn outline stops being an
axis-aligned rectangle, by corner count or by a side slanted more than `tol_px`;
a rotated rectangle is reported, and a straight line is a rectangle with no
thickness that passes while it is horizontal or vertical. A move is a dict, or any object whose
`contract` is a dict or has the same fields, with the keys `name`, `span`,
`artists` (or `moves`), `rest`, `groups`, `may_overlap` and `rectilinear`; a
move's `rest`, `groups`, `may_overlap` and `rectilinear` hold only over its
`span`, `may_overlap` also over the tenth of a second on either side in which
the span's motion already counts as moving. The call restores the figure's dpi and
canvas and is deterministic. It costs the draw time of each frame plus a
little; the frames are kept only for `jump` and `dead`. `assert_motion` raises
`MotionError` with the report.

`caption_line(fig, at=pose, times=...)` uses the same extents to place
captions: it returns the WebVTT `line:` percentage of the lowest strip, two
caption lines tall, that no visible text of the figure touches at any sampled
time, and raises when there is none.

`speech_rate(narration, ceiling=160)` checks the voice instead of the picture.
For each line of a `Narration` it returns the words per minute, spoken words
over the span from the start of the first word to the end of the last in the
cached word times, so padding and the pauses between lines do not count and a
pause inside a line does. The result lists the lines strictly above the
ceiling; `assert_speech_rate` raises `SpeechRateError` naming them.

A `Tempo` holds the pace of such a clip in one place. Moves and holds are
written in nominal seconds and multiplied by its `factor`; a caption holds,
after it has finished arriving, for `factor * (base_s + word_s * words)`,
a symbol in math counting as two words (`read_s`); and after an axis build
or a new subject the frame rests for `factor * settle_s` (`rest_s`). At the
defaults a caption holds about 1 s plus 0.4 s per word, so slowing a clip
is one change. `Timeline.read` holds by the same rule.

## Motion contract

`MotionContract` is a frozen record of what a move promises about its own
artists. `PlotResult.contract` and `MosaicResult.contract` hold it, or None
when the primitive declares no motion contract. The new `contract` argument is
keyword-only, preserving positional arguments in result subclasses.
`Stage.contract` describes its staged artists. The artists are the ones the
figure draws, not copies made for the checker. Pass a result directly as
`motion_faults(fig, at, times, moves=[result])`.

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

Declare only what is true. Empty tuples make no promise. A warp may deform
its inputs, a growing compass is not rigid, and a replacement equation term
may stretch by different factors on its two axes, so none earns a rigid
group merely by sharing one progress. A transform declares matched glyphs
as rigid only when their relative layout stays the same; its rules remain
rectilinear even when their lengths change. A word reveal by letters changes
its visible extent, and `reveal` can choose that mode on each call, so a word
contract allows changing extents across both reveal modes. A write declares
nothing: its real text is hidden until the end and its renderer-only ghost has
no independently traceable geometry, so no drift check could compare anything. An inset's spines and virtual
layout artists likewise do not become traceable marks by being declared.
These declarations describe drawing and do not change it. The checker reads
`moves`, `rest`, `groups` and `may_overlap`; `settles`, `arrives_after_motion`
and `rectilinear` also state promises that a caller can check directly.

## Two spans, two uses

`eyepiece.Span` is the plan span returned by `Timeline` verbs such as
`enter` and `move`. It carries a name, start and end times in seconds, the
knobs it writes, and, once resolved in a `Plan`, frame bounds `k0` and `k1`.
Use it to place another span or to select a settled still from the plan.

`eyepiece.testing.Span(start, stop)` is the checker's declaration of a move's
time interval in seconds. Use it to sample every frame with `motion_faults`
or to declare when an artist moves. It carries no knobs or frame bounds.
`check_clip(moves=...)` accepts either a timeline span **by its name** or an
`eyepiece.testing.Span` value, for example `moves={curve: "reveal"}` or
`moves={curve: ept.Span(0.0, 1.0)}`. An `eyepiece.Span` object itself is not
accepted there; pass its `.name`. See {doc}`../how-to/check-a-clip` for the
complete check and {doc}`../how-to/stills-from-the-encode` for still selection.
