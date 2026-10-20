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

The cursor moves only where the author says. `wait_until(t)` moves it without
placing a span, to hold the picture until a time computed elsewhere, such as
the moment a caption has been read: the open section or line ends no earlier
than that time, with no tail, and the clip lasts at least that long, but no
knob is written, so a wait shorter than a frame is not a fault. A span placed
with `at=` does not move the cursor itself, but by default it holds its
section open, and closing the section moves the cursor to its end; give it
`extends=False` so that what follows keeps its time however late the placed
span ends.

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
state, the state it is made in, is the one a still shows.

A stage computes absolute levels from the colors and opacities the artists
had when it was made, so levels never compound and setting a state twice
draws what setting it once does. Transient moves, `indicate` and
`circumscribe`, equal the frame without them at both ends of their progress.
Drawing never raises, so a still or a PDF save is not aborted half way:
a leader or link that breaks the connector rules of the evidence guide is
found by its `faults()` or by `audit(fig)`, which raises `AttentionFault` for
the strict ones. See {doc}`../reference/attention-moves` for each move's rest
state and behavior, eases, and `Tempo`.

## Why clips are checked

`eyepiece.testing` belongs to a test suite and not to a figure. Readability
checks find overlapping, clipped, small, or same-color text; meaning checks
ask what the colors say; motion checks follow the path between states rather
than the states themselves. See {doc}`../reference/clip-checks` for the
checks and their findings.

## Why a move declares a contract

A motion contract records what a move promises about its own artists, the
ones the figure draws rather than copies made for the checker. Declare only
what is true: empty tuples make no promise, and sharing one progress does
not earn a rigid group. These declarations describe drawing and do not
change it. See {doc}`../reference/motion-contract` for the fields and the
two span types used to place and check moves.
