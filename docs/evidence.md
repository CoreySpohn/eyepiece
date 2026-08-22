# Showing evidence

Most of the rules in this library are not about plotting. They are about not
lying with a picture. A reader cannot separate a change in the drawing from a
change in the data, so every property of the drawing is a claim, whether or not
anyone intended to make it. This page is the reasoning; the
[contract](contract.md) is the API rules that follow from it.

The vocabulary is Edward Tufte's. The specific failures below are ones this
library has actually shipped and then fixed, which is why they are written down.

## Proportionality

A quantity's extent on the page is a measurement the reader takes: pixels,
points, degrees of arc, opacity. It has to stay proportional to the quantity,
across every frame and every panel where that quantity appears.

The working test is the lie factor, `(size of effect shown) / (size of effect in
the data)`. Compute it rather than eyeballing it. Anything outside roughly
0.95 to 1.05 is either a defect or a deliberate compression, and a deliberate
compression has to be labeled on the figure.

Three consequences the library enforces or documents:

**Scales are pinned across an animation, explicitly.** An axis that follows the
data redraws every frame at full height, so a quantity that falls by an order of
magnitude reads as one that never moved. `record` fingerprints every axes'
limits and color norms on the first frame, compares on each later frame, and
warns when one drifts, naming the ratio. Pass `allow_rescale=True` when the
rescaling is the point. A limit that merely happens not to move is not pinned:
`Line2D.set_data` does not request an autoscale, so such a figure is one added
artist away from the defect.

**A moving camera must not change how big a fixed object is drawn.** Sweeping a
3D view through raw azimuth changes the projected extent of an object that is
not moving, which is the same failure as an unpinned axis with a rotation matrix
in front of it. Where a camera has to move, move it on a path that leaves the
projected extent invariant. For a planar track, sweeping about the plane normal
holds the projected area exactly constant, because it is `pi * a * b * cos(tilt)`
with the tilt fixed, and the object turns in the page plane instead of inflating.

**Two panels of one object share one page scale.** Two views of the same thing
drawn at two page lengths per physical unit is the same lie as an unpinned axis,
spread sideways instead of forward in time. Give them a common scale, matched
scale bars, or an explicit conversion on the figure.

## Data ink

Ink that does not carry information competes with ink that does. The library
already applies this in one place and states the reasoning there: an image drawn
without an `extent` gets no ticks, because axis numbers in raw array indices
frame the picture in units that do not mean anything to the reader.

The same argument governs a 3D view. A projected coordinate cage offers no
orthogonal drop lines, so a reader cannot recover a point's value from it however
many tick numbers sit beside it, and under a moving camera those numbers
re-solve their own layout every frame. The ticks promise a measurement the panel
cannot deliver. A 3D panel drawn for shape and orientation wants one reference
plane and a labeled scale, not three ruled planes and fifteen numbers.

Grids fall under the same test. A grid exists so a reader can recover a number
from a mark. Where the message is a shape, a trend, or a comparison, it is
furniture, and it competes directly with dense data such as an ensemble of
curves or a speckle field.

## Layering by value

Elements must be visually distinguishable, and the layering has to be clear.
Rank is set by value, meaning distance from the background, not by hue:

| Rank | What lives here |
|---|---|
| Furniture | Coordinate cages, gridlines, panes, spines, shaded exclusion regions |
| Scenery | Reference geometry, origin markers, leader lines and their labels |
| Data | The thing the figure is about |
| Answer | At most one mark per panel: the decision, the threshold crossing |

`eyepiece.SourceStyles` and the neutral ramp exist to make this mechanical:
scenery takes a neutral resolved against the live background, and data takes a
palette slot. Scenery never takes a palette slot, and the text color is reserved
for type and for the Answer rank. On a dark background this is the commonest
failure, because pure white is the default and costs nothing to write.

Two rules about color follow:

**One meaning, one color, for the whole document.** A talk is one document and a
paper is one document. Declare the cast once and thread it through every figure,
including figures built by different scripts.

**Two views of one entity stay in one hue and separate by value.** A posterior
ensemble and the truth it brackets are not two different kinds of thing.
Separating them by line width alone, with the names left in a panel title, asks
the reader to read a difference the palette says is not there.

An ensemble is drawn so that its darkness encodes its density. Set the per-curve
alpha from the count rather than by eye, and check that the saturation carries
information: a bundle at full saturation in every frame would look identical with
five draws or five hundred, so the ink has stopped reporting the spread. Opacity
that a reader interprets as probability has to be the probability.

## Small multiples

Small multiples resemble the frames of a movie: the same combination of
variables, indexed by changes in another variable. Once the reader has decoded
one panel they have decoded them all, and the comparison happens in the eye
rather than in memory.

Before building an animation, name the variable the frames are indexed by, and
ask whether the reader needs to compare two non-adjacent values of it. If so,
the answer is a strip of panels. Motion earns its place when the viewpoint is
what moves, when a rate is the concept and equal-time tick marks along the path
cannot carry it, or when a speaker is narrating live.

Two mechanical tests settle most cases. A sequence that holds every frame, or
runs at two frames per second or fewer, is already a strip of stills wearing a
movie's encoding. And a claim that is a ratio between the first frame and the
last is a strip, because a reader cannot divide across a cut.

No new primitive is needed to build one. The ax-first contract is what pays for
it: `plt.subplots(rows, cols)` and then the primitives drawing into the axes they
are handed. Four chores are the whole discipline, and every strip gets one of
them wrong the first time. Pin identical limits across panels by setting them,
never through `sharex` or `sharey`, which are banned on caller axes anyway. Strip
the interior tick labels so the frame repeats but the numbers do not. Put the
indexing variable's value on every panel in one consistent place. Draw one legend
for the grid rather than one per panel.

## Words belong on the data

Words, numbers and images belong together. A legend that makes the eye shuttle
between a key and the data is a design failure at small series counts, and a name
in a title is a legend with the swatches removed.

Distinguish two kinds of text, because they behave oppositely:

A **label** names a mark the panel already draws. It adds no content; it
discharges a debt the panel incurred the moment it drew a distinguishable glyph.
Labels are required and are not budgeted. A mark drawn and left unnamed is ink
spent for nothing.

An **assertion** states something the marks do not show. Assertions compete for
attention and belong under a small fixed cap, with the rest in speaker notes.

A primitive drawing a small fixed set of named curves should therefore label them
in place rather than build a legend: at each curve's right end where the curves
separate there, inside a band where the primitive fills regions, or at a
caller-chosen x otherwise, in the curve's own color. Hide a label when its
curve's local separation falls below about two text heights rather than shrinking
or displacing it. Two curves converging until one label disappears is usually the
figure's message.

## Mechanism, not outcome

A figure showing a score change has shown an outcome. Ask what the score is
computed from, and check that a reader could read each of those ingredients off a
panel. If not, the physical panel is context rather than mechanism, and a second
panel showing the ingredients is what the figure is missing.

A score about a rate cannot be companioned by a panel showing only a shape.
Period, phase and speed are invisible on a static closed curve.

Any measurement uncertainty entering the score as a denominator is drawn at true
scale on the physical panel, and no marker is drawn larger than the error it
represents.

Finally, ask "compared to what?". A figure claiming a decision rule works shows
at least one comparison case run to the same depth. A single arm shows that
something happened, never that the rule caused it. If the comparison is not going
to be built, the figure's text must not imply the claim.
