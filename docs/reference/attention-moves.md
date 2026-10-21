# Attention moves

This page lists attention moves, their rest states, and their behavior. See
{doc}`../explanation/clock` for the reasoning.

## Anchors and progress

`flight`, `callout`, and `leader` say where to look; none of them is data.
Their positions are `Anchor`s resolved in display space at draw time, and their
artists are children of the figure rather than of an axes, so they never join
autoscaling and a recording's drift check never sees them. Each returns a
result whose `update(u)` sets progress absolutely and whose rest state, the
state it is made in, is the one a still shows: a flight is absent, a callout, a
leader and a link are fully drawn. Drawing never raises, so a still or a PDF
save is not aborted half way: a leader or link that breaks the connector rules
of the evidence guide is found by its `faults()` or by `ep.testing.audit(fig)`, which
raises `AttentionFault` for the strict ones.

`Stage`, `build_axes`, and `collapse` follow the same rule.

## flight

A flight is absent at rest.

A flight can carry a segment or a bar and turn it.

## callout

A callout is fully drawn at rest.

## leader

A leader is fully drawn at rest.

## link

A link is fully drawn at rest.

## Stage

A stage sets each named group to an absolute presence and contrast computed
from the colors and opacities the artists had when the stage was made, so
levels never compound and setting a state twice draws what setting it once
does; a dimmed group blends toward the background it sits on, which reads the
same on a light figure and a dark one.

The rest state of a stage, the one a still shows, is the figure as drawn: every
group at full weight.

## build_axes

An axis build only clips and fades what the axes already draws, so it moves no
limit.

The rest state of an axis build, the one a still shows, is the figure as drawn:
every axis finished.

## collapse

A collapse is a figure-level copy, absent at rest.

## indicate

`indicate` is transient: at both ends of its progress the frame equals the
frame without it, and `indicate` pulses from the artist's colors as they are
when each update starts, so it composes with a stage that is set first.

## circumscribe

`circumscribe` is transient: at both ends of its progress the frame equals the
frame without it.

## term_row

`term_row` places one text per term at draw time, so its terms are anchors and
stage members, and a changed term re-lays the row.

## readout

`readout` shows a running number as an integer part right-aligned to a fixed
decimal point and a fraction left-aligned from it, each in a slot as wide as
the widest string it holds, so a count never moves the label; its shown value
is the interpolated value rounded to the format's precision (monotone, with no
negative zero), its slots grow only when `set` is called with a value that
needs more room, and at ``u = 1`` it shows the exact formatted value last set.

A `readout` clips its eased value to [0, 1] before it interpolates, so a count
passes neither its old value nor its new one.

## Eases

The eases `back_out` and `running_start` leave [0, 1] between their exact ends,
and a move may take any ease as a kind name or as a callable from progress to
eased progress, for example `functools.partial(ep.clock.ease, kind="back_out",
strength=0.2)`. A move clips only the progress it is given. Where an eased
factor would become an opacity, a level, a fraction of a path, a color blend,
or a line width, the move clips the factor itself before blending, so a
position or a warp may overshoot and nothing else can go invalid. A move snaps
to its ends on the progress, never on the eased value.

## words

`words` cuts one line of text into one text per word, laid out once as a single
string with each word's extent taken from the renderer, so its words are
anchors and stage members too, math in ``$...$`` stays whole, a color is given
by word index or by the word itself (a string key colors every occurrence), and
the joined line at rest is the single text, pixel identical; its reveal is by
opacity or by letters in place, never by motion.

## morph

`morph` is a figure-level ghost like a flight, visible only between its ends,
whose geometry at the ends is the two paths resampled by arc length (and, for a
closed pair, the second rolled and possibly reversed to lie on the first).

## equation

`equation` lays mathtext out as glyph outlines, one figure-level patch per
glyph or rule, in display pixels, with named groups (``{{name:...}}``, found by
hiding the group with ``\phantom`` and reading what disappears, and raising
when the hidden group would move any other glyph); a group is an artist with
the box of its glyphs, laid out where the equation is, so anchors, callouts,
outlines, pulses and a `Stage` act on it unchanged. Its ``update(u,
level=1.0)`` writes the terms on, left to right, and at ``u = 1`` the equation
is whole; `level` is an opacity and never geometry.

## transform

`transform` carries one equation into another, owns the visibility of both,
rests at ``u = 1`` (the second equation shown, the first hidden) and draws the
first alone at ``u = 0``, each pixel-identical to that equation as placed.

Matching is by glyph identity (font and glyph, never the character code):
inside a tag both equations share (or a `key_map` pair), in left-to-right
order, then across glyphs in no tag, and a tagged glyph never pairs outside its
tag; equations that share no tag pair every glyph by order.

`align` (default ``"auto"``) places the second equation so that the sideways
shift most groups share is removed and those glyphs keep their place in the
first, which moves the second equation's layout at every `u`, the rest state
included.

Matched glyphs move by group, never one by one: all the matched glyphs of one
tag, or one run of neighbouring glyphs in no tag, share one eased progress,
staggered left to right by the group's place in the first equation, so a word
moves rigidly. A group arcs on an optional arc, up when it moves right and down
when it moves left, scaled by its sideways travel (``min(1, travel /
fontsize)``), so two groups that swap places pass each other and a group that
does not travel sideways does not arc.

A tag both equations have whose glyphs do not all match is a replacement: what
is left of it in the first is carried as one box that moves and scales onto the
box of what is left in the second, fading out while the second, drawn inside
the moving box, fades in.

`replace="morph"` (the default; `"fade"` keeps the box) changes the outlines
instead, aligned as ManimCE's `Transform` aligns paths: every segment is a
cubic, the glyphs of each side pair in reading order and the side with fewer is
padded with stacked copies of its own glyphs (which start or end invisible),
each pair gets the same contours (a missing contour is a null contour at the
last point of the last one, so a hole grows from or shrinks to a point) and the
same curve count (curves split evenly in their parameter), and the control
points, fill color and opacity mix linearly in display space on the group's
eased progress. Both modes rest at the two equations as drawn, take the motion
part of the beat, and leave matched glyphs and unmatched glyphs as they are.

Contours pair by index, as in Manim, so a morph that pairs an outline with a
hole flips that contour's winding part way and the nonzero fill can briefly
close a hole.

Glyphs with no counterpart at all fade out in the first `fade` of the beat or
fade in in the last, so when anything fades in the motion takes the first ``1 -
fade``, and new text arrives after the motion has settled. A fraction bar
stretches as a rectangle, each corner moving to its partner, so its ends stay
square.

A transform's narration cue is ``{move:name}``, whose time starts its ramp. Not
supported: ``usetex``, several math spans in one string, escaped braces (they
raise), and the size animation of ``\left ... \right``.

## write

`write` is a figure-level ghost of a text, visible only before ``u = 1``: it
hides the real text until the end, where the text is shown with the visibility
it had, so the rest state is the text as drawn. Its outlines are laid out as
the text lays itself out, its stroke and fill take the text's color and opacity
at draw time, and its eased tracing and fill are clamped to [0, 1].

## arrow, angle_arc, and compass

`arrow`, `angle_arc` and `compass` are drawn marks, not overlays: each is one
patch (an arrow, an arc whose head is tangent to the arc, or a north and an
east arrow) added without joining autoscaling, and `update` sets its growth
`u`, its ends or angles, and a `level` opacity on the same artists. An arrow
grows from its tail, or from its middle when it has two heads; an arc sweeps
from its start angle; and a compass has no default side for east, because a
compass drawn with the wrong side is a mirrored sky.

## follow_zoom

`follow_zoom` is a figure-level inset: an axes in the panel that draws the
panel's own artists again through a window centered on its target, a frame box
on the figure, and two connectors audited as links. Its geometry is recomputed
at every draw, so following needs no per-frame call; the window is exactly
`zoom` times closer on both axes and is never clamped to the panel. Its
`update(u, level)` fades all of it in with `u`, `level` is an opacity only, and
at ``u = 0`` the figure is pixel for pixel the one without the zoom. The
parent's artists are borrowed for each draw and handed back as they were.
`loc="auto"` holds the corner farthest from the target, switching only when
another is farther by a quarter of the inset's side; the frame, box and
connectors carry a halo in the panel's background so they read on dark and
light images.

## warp

`warp` differs from the figure-level moves: it reshapes the artists it is
given, in place and in data coordinates, so it creates nothing and an artist
keeps its place, style and stage membership. Its rest state is the start, ``u =
0``, the geometry the artists had when the move was made (a path that a
pointwise map subdivided is put back as drawn); at ``u = 1`` the geometry is
exactly ``M p`` or ``f(p)``. Its `level` changes nothing: the artists' opacity
belongs to the caller or a `Stage`, so dimming a warped group never undoes the
warp. It leaves axis limits alone.

## Tempo

A `Tempo` holds the pace of such a clip in one place. Moves and holds are
written in nominal seconds and multiplied by its `factor`; a caption holds,
after it has finished arriving, for `factor * (base_s + word_s * words)`, a
symbol in math counting as two words (`read_s`); and after an axis build or a
new subject the frame rests for `factor * settle_s` (`rest_s`). At the defaults
a caption holds about 1 s plus 0.4 s per word, so slowing a clip is one change.
`Timeline.read` holds by the same rule.
