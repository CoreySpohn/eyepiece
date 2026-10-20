# Clip checks

This page lists the readability, meaning, motion, caption, speech, timing and grounding
checks in `eyepiece.testing`. See {doc}`../explanation/clock` for the reasoning.

## readability_faults

`eyepiece.testing` is the readability check for such a clip, a submodule
imported by name rather than a top-level export, because it belongs to a test
suite and not to a figure.

`readability_faults(fig, at=pose, times=...)` poses the figure at each sample
time, draws it, and returns a list of faults in output pixels: two visible
texts, or a text and another's backing box, whose extents overlap by more than
a tolerance; text or marks the figure edge clips; text under the venue's
minimum height; and text drawn over a data mark of its own color. It does not
repeat `audit`, which owns the connectors, and a label with an opaque backing
box is not reported over marks. The result is deterministic and the figure
keeps its dpi. `assert_readable` raises `ReadabilityError` with the report.

## meaning_faults

`meaning_faults(fig, at=pose, times=..., keys={gid_prefix: meaning})` checks
what the colors say, which `readability_faults` does not. Artists are tied to a
meaning by `gid` prefix (longest prefix wins, and an artist with no `gid`
inherits its nearest ancestor's).

Colors are measured as drawn, after artist
alpha and any `Stage` dim, over the background, in CAM02-UCS under normal
vision and the three Machado types at severity 100.

It reports five kinds:
`same-frame` (two meanings in one color within `fault_de`, or within `warn_de`
as a warning, downgraded when a marker, line style or hatch also differs),
`rebound` (one color carrying different meanings at different times, or across
the clips of `meaning_faults_chain`, unless declared with `rebind`; clustered
on the colors before opacity, and a `sharing` pair seen in one clip is always a
fault), `cue-form` (a key named in `cue_keys` drawn as a line, as text, as an
opaque patch over its subject, or in any hue; the cue is a band, ring or
underlay in the text color), `unkeyed` (a chromatic artist with no meaning,
advisory unless `strict`), and `dimmed` (an advisory: two keys that share a hue
only once both are dimmed toward the background).

`rebind` takes times, in `meaning_faults` and in `meaning_faults_chain` alike.
`sharing` pairs may share a color but not a clip. A finding is waived only by
naming it in `waive` with a reason, and stays in `meaning_report`, whose all-pass wording is "no tier 1 or 2 findings". `assert_meaning` raises
`MeaningError`. It needs `colorspacious`, in the `test` extra.

## motion_faults

`motion_faults(fig, at, times, moves=..., rest=..., groups=...)` is the same
check for the path between two states rather than the states themselves. Given
an `eyepiece.testing.Span(start, stop)` it poses the figure at every frame of
the venue's frame rate (30 per second for each venue; `fps=` overrides), always
ending exactly on `stop`, draws it, and traces every visible artist, equation
glyph paths and collections included, with the rendered frame when a pixel
check is asked for. It returns `MotionFault` records, sorted by time, of eight
kinds.

## jump

`jump` is a frame step far above the typical one and not echoed by its
neighbours, a step into the rest state (the end of a declared move, else the
last step that changes anything) that is both above half the ratio and twice
the step before it, or, for a clip too still to have a typical step, one artist
that moves a large share of its size in one frame.

## dead

`dead` is a declared move whose region never changes, or whose artists are
never visible.

## drift

`drift` is an artist listed in `rest` whose box or ink centroid moves by more
than `tol_px`.

## tear

`tear` is a `groups` set whose members, by their ink centroids, change their
spacing, so a stagger or a drift of one member shows and a rigid rotation does
not; a group declared with `scales=True` may scale as a whole.

## overlap

`overlap` is two artists, text or declared in `moves`, whose visible parts
cover each other beyond a tolerance while both travel (judged over a tenth of a
second, so the answer does not depend on the frame rate) and are not listed in
`may_overlap`.

## outside

`outside` is any traced artist, a glyph path included, past the frame, and any
text or declared artist cut by its clip box or clipped away entirely; a clip
box that only widens, and ends by uncovering the artist, is a reveal and not a
cut.

## spike

`spike` is a hop off the path, or a one-frame excursion, or a kink in speed
(measured over elapsed time) on an artist that keeps its shape; it is off
unless `spikes=True` or `kinds` names it, because a small hop is sometimes the
intended motion.

## shape-class

`shape-class` is a patch or line listed in `rectilinear` (a fraction bar, a
rule, a box) whose drawn outline stops being an axis-aligned rectangle, by
corner count or by a side slanted more than `tol_px`; a rotated rectangle is
reported, and a straight line is a rectangle with no thickness that passes
while it is horizontal or vertical.

## Move declarations

A move is a dict, or any object whose `contract` is a dict or has the same
fields, with the keys `name`, `span`, `artists` (or `moves`), `rest`, `groups`,
`may_overlap` and `rectilinear`; a move's `rest`, `groups`, `may_overlap` and
`rectilinear` hold only over its `span`, `may_overlap` also over the tenth of a
second on either side in which the span's motion already counts as moving.

## Declared steps and draw groups

A declared step (`steps`, in seconds) excuses a jump that ends at it. On its
own it excuses the whole picture, so an artist that cuts by mistake on the same
frame as an intended cut goes unseen.

A draw group (`draws=`, or a move's
`draws`) narrows that: at a step where the group changes, the jump check
measures the picture outside the group's boxes, before and after the step,
padded by two pixels. The group's cut is excused; any other artist that cuts on
that frame is reported, against the clip's pixel baseline or, in a clip too
still to have one, as an artist that jumps. The group's artists may overlap one
another, and every other check holds for them as for any artist.

A step where
the group does not change, such as a scene cut, still excuses the whole
picture.

Inside the group's boxes, which for a run of curves can cover most of
an axes, the pixel baseline is blind at the group's cuts, so there every artist
outside the group is followed on its own and one that jumps on the cut is
reported.

What stays unseen is a change there that moves no artist's box, such
as an artist outside the group that changes color or alpha in place, or a line
redrawn within its old extent (a test marks this gap).

The call restores the
figure's dpi and canvas and is deterministic. It costs the draw time of each
frame plus a little; the frames are kept only for `jump` and `dead`.
`assert_motion` raises `MotionError` with the report.

## caption_line

`caption_line(fig, at=pose, times=...)` uses the same extents to place
captions: it returns the WebVTT `line:` percentage of the lowest strip, two
caption lines tall, that no visible text of the figure touches at any sampled
time, and raises when there is none.

## speech_rate

`speech_rate(narration, ceiling=160)` checks the voice instead of the picture.
For each line of a `Narration` it returns the words per minute, spoken words
over the span from the start of the first word to the end of the last in the
cached word times, so padding and the pauses between lines do not count and a
pause inside a line does. The result lists the lines strictly above the
ceiling; `assert_speech_rate` raises `SpeechRateError` naming them.

## timing_faults

`timing_faults(narration, plan=None, source=None, at=None, readouts=None)`
applies the timing rules of a narrated clip in one call and returns
`TimingFault` records sorted by time; `assert_timing` raises `TimingError`.
Each finding has one of the `TIMING_KINDS`:

- `speech-rate`: a line faster than 160 words per minute, or 166 for a line of
  at most 10 spoken words, compared in whole words per minute.
- `line-gap`: less than 0.4 s of heard silence between the last word of one
  spoken line and the first word of the next.
- `late-first-cue`, `early-last-cue`, `undeclared-silence`, `multiple-hero`
  and `speech-in-hero`: the findings of `eyepiece.narration.check_silence`,
  passed through unchanged.
- `still-speech` (with `source`, a video path or a callable returning an
  image): a still run, in which fewer than 0.16 percent of the pixels change
  against the run's first frame, under more than 2 s of speech; a silence of
  2 s or more splits the speech.
- `number-late` and `number-contradicts` (with `readouts` and `at`): a readout
  does not show the spoken number by the start of its word (plus
  `settle_tol_s`, at least one frame with a plan), or, once it has, shows a
  different number while the line is still said. A readout shows a number when
  its first whole number equals it and its ink is at least 90 percent of the
  most it reaches during the line; a blank or hidden readout contradicts
  nothing.

## review_sheet

`review_sheet(path, source, narration=None, plan=None, events=None,
audio=None)` writes one PNG of a clip at its event times: every cue, every beat
end and, with a plan, the middle and end of every move. With a narration each
row is one beat,
with its text, a frame per event, the caption under each frame and, with
`audio`, the waveform with a tick at each event. `events=` takes any
`(seconds, label)` list instead.

## grounding_faults

`grounding_faults(equations, at=None, times=None)` checks that each colored term
of an `equation` is grounded in the mark it names, the binding made with
`eq.bind(name, mark, motif=...)`. It returns `GroundingFault` records, one per
term and kind at its first time; `assert_grounded` raises `GroundingError`.
A term is colored when its ink is neither a gray nor a tone of the ramp from
the axes background to the text color that `neutral` draws. The kinds are:

- `no-referent`: a colored term on screen that is unbound, or whose mark is not
  on screen.
- `color-mismatch`: a bound term not in its motif's label ink (else its mark's
  color), or a mark off its motif's color.
- `symbol-first`: a bound term on screen before its mark, sampled over `times`.
