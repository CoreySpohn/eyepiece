# Showing evidence

Most of the rules in this library are not about plotting. They are about not
lying with a picture. A reader cannot separate a change in the drawing from a
change in the data, so every property of the drawing is a claim, whether or not
anyone intended to make it. This page is the reasoning; the
[contract](contract.md) is the API rules that follow from it.

The vocabulary is Edward Tufte's. The specific failures below are ones this
library has actually shipped and then fixed, which is why they are written down.

## Provenance

A graphic is an argument, and an argument without a source cannot be checked. A
credible figure says who made it, when, from what data, at what scale, and by
what means. A reader who cannot answer those questions is left to judge the
picture by how good it looks, which is the one criterion a careless figure and a
sound one share.

Simulated data is the case that matters most, because it is the case where the
reader cannot tell. A rendering of a simulated measurement and a rendering of a
real one are the same picture. Nothing about a clean curve, a tight ensemble, or
a plausible error bar reveals that a generator produced it from parameters
somebody typed. The figure has to say so, in words, on the figure. "Simulated"
is one word and it is load bearing, so it is never delegated to a caption that
travels separately from the image, to a slide the speaker skipped, or to a file
name.

A number that chose the picture is documented even when it is not plotted.
Search ranges, priors, mixture orders, candidate windows, iteration counts and
clipping thresholds each select among the pictures that could have been drawn.
The test is mechanical: list the parameters that would visibly change the figure
if they moved by a factor of two, and put every one of them on the figure or in
its caption. A figure about aliasing that hides its search interval has withheld
exactly the thing a skeptic would ask for first.

Where a figure is scored against a known truth, that truth is provenance rather
than result. Give its value, so a reader can check the panel against it instead
of accepting the conclusion and the answer key together.

A figure built from a seeded generator shows one draw out of many, and that draw
was chosen whether or not anybody chose it deliberately. Pin a seed per figure,
and say how many were examined and on what basis one was kept. Pinning buys
reproducibility; it is not evidence of typicality.

A decision drawn at the edge of a searched window is a property of the window.
When an optimum lands on the boundary of the range that was searched, the
boundary is the result, and the figure says so rather than presenting an
interior optimum it did not find.

The stamp itself is furniture: the smallest legible type, the furniture rank in
the value ordering, one fixed corner, never competing with the data. The same
fields go into the file metadata as well, because the visible line is lost to a
crop and the metadata is lost to a screenshot, and a figure outlives the
conversation that explained it. `stamp` writes both channels: the visible line,
and a structured payload that `save_fig` embeds in the file. A figure that has
been cropped, renamed and pasted into a document still answers what produced it.
Nothing is written for an unstamped figure, since a file claiming no provenance
is honest and one claiming invented provenance is not.

Constants are imported, not typed. A physical constant written as a literal
inside a plotting script is untraceable by construction, and two scripts that
type the same constant to different precision disagree in ways no reader can
see.

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

## Time on an axis

A closed curve shows where a thing goes and hides when it is there. Position is
on the page; period, phase and speed are not. Motion restores them only to a
viewer who is watching, and only as an impression: a body visibly speeds up, and
nobody watching can say by what factor.

Giving time a spatial axis makes those readable as geometry. Horizontal distance
between repeats is the period. Slope is the rate. A crossing of two curves is a
coincidence between two things. The shape of one cycle is the departure from
uniform motion.

One question settles the choice, asked before an animation is written. Name the
quantity the figure is about. If it is a rate, a period, a phase, a delay or a
coincidence, it is a time quantity and the figure owes it an axis.

A shared clock is not a shared axis. Several objects animated on one clock show
one instant per frame, and the ratios between their periods and the times at
which they coincide are properties of the whole run that appear on the page at
no instant.

An elapsed-time readout is not a time axis and does not discharge this. A number
with no denominator cannot be placed in its cycle, so where a readout is all the
room allows, give it the denominator in the same string.

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


Parallel form is a claim, and the claim is about coordinates. Panels drawn at
one size, in one frame, in one row assert that their axes mean the same thing.
Panels in a row therefore share an origin, not only a label: an axis re-zeroed
per panel gives every panel an identical label and a different zero, so heights
stay comparable while feature positions do not. A row that shares no axis with
the rows above it must not be drawn as though it does, which means its own tick
labels, its own axis label, and a visible gap. And a pinned scale that leaves
most panels occupying a fifth of their axes has stopped serving comparison: a
logarithmic axis usually resolves that without weakening the pin.

A process that loops is drawn as a grid, not as a line. A procedure that
repeats carries two indices, the stage within an iteration and the iteration
itself. Give the stages the rows and the iterations the columns.

## How many variables the panel carries

A panel is a budget and the unit is the variable. Count the distinct quantities
a reader can recover from one still frame and name the channel each one uses.
Position on a flat surface costs two channels and counts two. A scalar printed
in a title counts one. Opacity, marker size, marker shape, tick spacing along a
path, and the length of an axis spine are each a channel.

Two things do not count. A cue that re-encodes a quantity the panel already
draws adds nothing. A landmark derived from the drawn geometry is a reading aid
on a quantity already present.

Four to six is the working range for a scientific panel. A panel carrying two is
under-loaded, and the usual reason is that the missing variables are being
supplied by the reader from memory or from the caption. Adding one back normally
makes the panel easier to read, because context is what makes a mark
interpretable. A quantity the analysis already computes and then discards is the
first candidate.

**Dimensions are not variables.** A picture does not become multivariate by
acquiring axes. A planar object drawn in three dimensions has one scalar of
depth content, and a projected cage cannot deliver even that, because there are
no orthogonal drop lines to read against. Before drawing a third axis, name the
variable it buys and say what a flat panel would buy for the same space. A track
plotted against time carries distance, phase, period and rate on two axes, which
is four variables where an oblique view of the same track carries two.

**Multifunctioning elements are the cheapest variable a figure can buy.** A path
drawn with marks at equal time intervals carries rate as well as shape: close
marks are fast, wide marks are slow, and the bunching at the slow end is the
mechanism drawn rather than asserted. An axis drawn only over the range the data
occupies reports that range, which turns furniture into a measurement. An axis
can carry the events that drive the curve above it.

Multifunctioning is one step from ambiguity, and the test is whether each
function stays separately readable. A marker size carrying a physical magnitude
and a depth cue at once passes only when the cue's full range is small against
the smallest gap the physical encoding must show, computed against the smallest
gap rather than a convenient larger one, and the panel carries a key. Where the
ordering survives and the magnitude does not, the honest description is that the
channel ranks rather than measures, and the figure says so.

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

## Links

A drawn connector is the most overloaded glyph in technical illustration. One
line is asked to mean causes, is followed by, flows into, belongs to,
corresponds to, and is transformed into, and nothing in the drawing says which.

The rule is not that links are forbidden. It is that an unnamed link is
forbidden. Three forms satisfy it, in order of preference.

**Repetition.** A panel that reappears unchanged is the strongest link
available, because the reader can verify it. Draw the second appearance from the
same function, in the same units, with the same limits, so identity holds by
construction. Nobody can check an arrow; anybody can check that two pictures
match.

**Alignment.** Two quantities in a causal relation go on a common axis, and
successive states of one system go in the columns of one grid. Know what
alignment can say and what it cannot: it establishes which things are related
and does not establish how, so a figure still has to settle that in words.

**A named connector.** Where the link is genuinely a line in the geometry, a
radius standing for a distance, draw it and name it. The line is the quantity,
and refusing to draw it costs the reader a measurement they are already trying
to take.

A leader line is a connector and is audited as one. Move the words to the mark
where you can. Where a leader cannot be avoided, it terminates on the mark it
names and on no other, runs shorter than a quarter of the panel, crosses no
data, and takes a neutral rather than a color that already carries meaning
there. A leader drawn in the data's own value is read as data, and a plural
label attached to one member of a class names that member rather than the class.

## Shape, type, and color

**Aspect ratio is either a free choice or a data value.** When the two axes
carry different quantities in different units, the aspect is free and should
tend toward the horizontal, roughly 1.4 to 1.8: the eye judges departures from a
horizontal baseline well, reading runs left to right, and horizontal type fits a
wide frame. When both axes carry the same physical quantity in the same units,
the aspect is a data value, and setting it to anything but one multiplies every
angle and every eccentricity in the panel by the distortion. Measure the data
rectangle rather than the canvas, because the canvas carries margins and
siblings and its ratio says nothing.

**Type is a design element of the graphic, not a caption applied afterward.**
Set titles, labels and annotations in sentence case, and keep capitalization
consistent within a figure. Spell the words out: a count is "3 epochs" rather
than "n = 3". Run type horizontally where the layout allows. Reserve space for
type before drawing, because text placed into whatever margin is left over is
text that will collide, and a collision is measurable.

**Color does four separate jobs**, and confusing them is the ordinary failure.
It labels, distinguishing nominal categories. It measures, carrying an ordered
value. It represents, standing for the real appearance of a thing. And it
enlivens. A palette serving as a document's nominal alphabet is spent: a panel
drawing exactly one series takes a neutral or a pinned role color, never the next
slot off the cycle, because taking a slot makes a nominal claim the panel has no
categories to support and spends a letter another figure needs. A diverging scale
needs its neutral midpoint visible against the plot background, since a scale
running through the background renders zero as a hole in the data.

**Color a reader can actually see.** Between five and ten percent of readers
have a color vision deficiency, so a figure whose meaning rests on hue alone
fails for about one viewer in twenty. That is checkable rather than a matter of
taste: simulate the palette under each deficiency, compute pairwise perceptual
distance, and read off the minimum. A palette of six hues chosen under normal
vision typically resolves to three or four distinguishable groups, so a figure
needing more than three simultaneous nominal categories cannot get them from hue
and has to recruit position, direct labels, or dashing. A palette produced by
desaturating a bright one inherits its collisions and adds new ones, because
desaturation moves every color toward a common gray.

**Dark backgrounds invert the operation, not the rule.** Data sits a modest
distance above the ground and strong contrast is reserved for accents. On white,
quieting a color means lightening it toward the page; on a dark ground,
lightening is what makes a color loud, so quieting means darkening. A palette of
bright saturated hues on black is a palette of accents used as defaults, and the
symptom is an author reaching past the palette to hand-darken a series. A
saturated bright stroke on a dark ground also blooms, so per-curve opacity wants
roughly half its light-mode value at the same count. Nothing about hue changes:
color deficiency is a property of the observer, not the background.

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

## Claims a figure cannot cash

A title is read as a finding. It is read first, it carries more weight than any
label on the page, and it is what a viewer repeats afterwards, so it is held to
the evidence in the panels beneath it and to nothing else.

**A single arm.** A figure claiming that a rule works shows at least one
comparison case run to the same depth. One arm shows that something happened,
never that the rule caused it. Where the comparison is not going to be built,
the title states what was run rather than what it achieved.

**A credited decision that any decision would have made.** A panel showing that
the chosen option scored highest has not shown that the choice mattered. Compare
the chosen option against the distribution over the options not chosen.

**A derived statistic standing in for the quantity it is named after.** A
statistic computed by selecting one component of a fitted model, by nearest
match, by rank, or by any rule containing a tie is not the quantity in the name
unless the rule is stated and the tie is impossible. Where several components
have converged on the same answer, a rule that picks one of them reports how the
labels fell rather than what the model believes, and the resulting sequence will
move while the belief stands still. State a tolerance, sum inside it, and put
the tolerance on the panel.

**A sequence smoothed into a trend.** A run that goes up, then down, then up is
either the message or a defect, and the difference is worth the time it takes to
find out. Cropping it, rescaling until it flattens, or narrating past it is the
rage to conclude.

**Words carry links too, and an audit of the ink will miss them.** A title
saying that one thing broke another is an arrow made of type, and it meets the
same test as a drawn one: name the relation and show the reader where to read it
off the panels.

## Content is the deciding factor

Nothing above can rescue a figure that has nothing to say. Layering, labeling,
proportionality and documentation raise the ceiling on how much a reader can
extract; the analytic quality of the material sets where that ceiling is. A
figure with a real finding drawn badly is worth repairing. A figure with no
finding drawn beautifully is worth deleting, and it is the more dangerous of the
two, because it looks finished.

The practical form is a question asked before any drawing begins: what would a
reader know after this figure that they did not know before? A shape they could
have guessed is not an answer. A capability demonstration is not an answer. A
quantity that moved, with no account of what moved it, is not an answer.

The demonstration genre is the exception and the worst available template. A
gallery figure exists to show what a drawing tool can render, and its content is
legitimately the tool. It must not be used as a model for an analysis figure,
because everything that makes it good, the clean single subject and the absence
of comparison, is what would make an analysis figure empty.

Ask what the informed skeptic will say, and answer it on the figure. The
strongest objection to a good figure is usually specific, short, and already
known to its author: the window was too narrow, the prior did the work, the seed
was kind, the comparison was not run. An objection answered on the figure costs
one line.
