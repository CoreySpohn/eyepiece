---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Motion attention and emphasis

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small.
plt.rcParams["savefig.dpi"] = 110
```

## Attention: a flight, a callout, a leader

A beat has to say where to look. Three moves do it without touching the
data. `flight` carries a copy of a mark from one panel onto another, so the
viewer sees a value read in one place become a point somewhere else.
`callout` writes a caption of a few words beside the thing it names.
`leader` joins a caption to its mark when the two cannot touch, and refuses
to draw one that breaks the connector rules of the
{doc}`evidence guide <../explanation/evidence>`: it must end on the mark it names, run
shorter than a quarter of its panel, cross no data and no other text, and
take a neutral rather than a data color. Drawing never raises, so a still
or a PDF save is never aborted half way; `faults()` on the leader lists what
it breaks, and `audit(fig)` checks every leader and link of a figure at once
and raises `AttentionFault` for those made with `strict=True` (the default).
A recording calls it on the composite.

Every position is an `Anchor`, a data point (`Anchor.from_data`) or a part
of a drawn artist (`Anchor.from_artist`), resolved in display space when the
figure is drawn. The moves' artists belong to the figure, not to an axes,
so they never change data limits. Each result's `update(u)` sets progress
absolutely, and each has a rest state: a flight is absent, a callout is
fully drawn.

```{code-cell} python
grid = np.linspace(-0.2, 0.8, 200)
height = np.exp(-0.5 * ((0.35 - 0.25) / 0.1) ** 2) / (0.1 * np.sqrt(2.0 * np.pi))

fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.0, 3.6), layout="constrained")
top.plot(grid, np.exp(-0.5 * ((grid - 0.25) / 0.1) ** 2) / (0.1 * np.sqrt(2.0 * np.pi)))
top.axvline(0.35, lw=0.8, color=ep.neutral(0.5))
(read,) = top.plot([0.35], [height], "o", ms=6)
(landed,) = bottom.plot([0.25], [height], "o", ms=6, color=read.get_color())
top.set(xlim=(-0.2, 0.8), ylim=(0.0, 4.6), ylabel="density")
bottom.set(xlim=(-0.2, 0.8), ylim=(0.0, 4.6), xlabel="mean", ylabel="likelihood")
# Lay the panels out once, then freeze them: the moves are figure artists
# outside the layout, and an animation's panels must not shift between frames.
fig.canvas.draw()
fig.set_layout_engine("none")

carry = ep.flight(read, ep.Anchor.from_artist(landed), fig=fig, arc=0.15)
note = ep.callout(read, "its height is the likelihood")


def grab(flight_u, landed_shown):
    carry.update(flight_u)
    landed.set_visible(landed_shown)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


frames = [grab(0.0, False), grab(0.5, False), grab(1.0, True)]
shown, cells = plt.subplots(1, 3, figsize=(9.0, 2.2), layout="constrained")
for cell, frame, title in zip(cells, frames, ("read", "carried", "landed"), strict=True):
    cell.imshow(frame, interpolation="nearest")
    cell.set_title(title)
    cell.set_axis_off()
plt.show()
```

In an animation each frame's state sets the progress of each move. A
caption of a few words should start to appear just before the event it
names, so its `update` runs from 0 to 1 over a few frames ending then; the
flight's runs over the event itself; and `update(1.0, level=0.35)` dims a
caption that has been read, so the next event has the eye to itself. A
recording's scale-drift check never sees the moves, since none of them is
inside an axes.

Three limits follow from the moves living on the figure. A callout's
`side="auto"` is judged once, when it is made, so lay the figure out
first. A ``bbox_inches="tight"`` save does not grow to hold a caption
outside the axes, and a blitting loop does not repaint the moves. A
`crossfade` label must keep the opacity it lands at: the flight hides it
until the end of the move and restores it there.

### Draw a curve on

`draw_on` reveals one line or a list of lines along their paths. Each
`update(u)` draws the first fraction of the current display-space arc
length, so log axes and data replaced with `set_data` keep working.
`update(u, level=0.35)` dims the line without changing its revealed length.
Nothing is drawn at zero; at one the original line draws with its exact
markers and dashes. During the reveal, markers include the cut tip and
dashes start at the beginning of each connected run.

```{code-cell} python
fig, panels = plt.subplots(1, 3, figsize=(8.0, 2.4), layout="constrained")
x = np.geomspace(1, 100, 120)
for ax, progress in zip(panels, (0.0, 0.5, 1.0), strict=True):
    (curve,) = ax.plot(x, np.sin(np.log(x)), lw=2)
    ax.set(xscale="log", xlim=(1, 100), ylim=(-1.2, 1.2),
           xlabel="x", title=f"progress {progress:.1f}")
    reveal = ep.draw_on(curve)
    reveal.update(progress)
plt.show()
```

## Emphasis: pulse, outline, term row, turn, link

Five more moves say "this one" for a thing already on screen, or join two
panels. `indicate` is a pulse: for a moment the artist's line width, marker
size, or font size grows by 1.15 and its colors move toward the text color,
then it returns, so a frame at the start or the end is the frame without the
move. It adds no hue, and it pulses from wherever the artist is, so on an
artist a `Stage` holds it swells from the stage's level (set the stage, then
the pulse, in each frame). `circumscribe` draws an outline around an artist
or an anchor, clockwise from the top, and lets it fade. `term_row` lays an
equation out as one text per term, so each term can be the
target of an anchor, a stage group, a flight, or a pulse of its own; a term
whose text changes pushes its neighbors over at the next draw. `flight` turns
what it carries, a segment or a bar, by `turn_deg`, so a stub beside the star
can be carried and turned a quarter turn onto a velocity axis. `link` grows
a line from a mark in one panel to a mark in another; the quarter-panel rule
of a leader does not apply, and `faults()` instead checks where the link
touches data, in every panel it passes through.

```{code-cell} python
fig, (star, speed) = plt.subplots(
    1, 2, figsize=(8.0, 2.8), gridspec_kw={"wspace": 0.35}, layout="constrained"
)
star.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), xlabel="x", ylabel="y")
speed.set(xlim=(0.0, 1.0), ylim=(-1.0, 1.0), xlabel="phase", ylabel="velocity")
star.set_aspect("equal")
orbit = np.linspace(0.0, 2.0 * np.pi, 200)
star.plot(0.6 * np.cos(orbit), 0.6 * np.sin(orbit), color=ep.neutral(0.4))
(stub,) = star.plot([0.6, 0.6], [0.0, 0.3], lw=3.0, color="C1", solid_capstyle="butt")
(orbiter,) = star.plot([0.6], [0.0], "o", ms=7, color="C0")
speed.plot(orbit / (2.0 * np.pi), 0.5 * np.sin(orbit), color=ep.neutral(0.4))
fig.canvas.draw()
fig.set_layout_engine("none")

row = ep.term_row(
    ep.Anchor.from_data(speed, (0.05, -0.8)), [r"$v_r$", "=", "0.30"], ha="left"
)
ring = ep.circumscribe(orbiter, shape="ellipse", pad_pt=5.0)
pulse = ep.indicate(stub)
turn = ep.flight(
    stub, ep.Anchor.from_data(speed, (0.0, 0.3)), fig=fig, turn_deg=90.0, arc=0.2
)
tie = ep.link(orbiter, ep.Anchor.from_data(speed, (0.0, 0.0)))


def grab(ring_u, pulse_u, turn_u, link_u):
    ring.update(ring_u)
    pulse.update(pulse_u)
    turn.update(turn_u)
    tie.update(link_u)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


frames = [grab(0.55, 0.0, 0.0, 0.0), grab(0.0, 0.5, 0.0, 0.0), grab(0.0, 0.0, 0.5, 0.5)]
shown, cells = plt.subplots(1, 3, figsize=(9.0, 1.8), layout="constrained")
titles = ("outlined", "pulsed", "turned and linked")
for cell, frame, title in zip(cells, frames, titles, strict=True):
    cell.imshow(frame, interpolation="nearest")
    cell.set_title(title)
    cell.set_axis_off()
plt.show()
```

A term is rewritten by setting its text in the loop, for example
``row.artists["text"][2].set_text("0.45")``, and `Anchor.from_artist` of a term
follows it. Use `term_row` for terms whose text you choose, and {doc}`readout <motion-clock>` for a number that counts, because a rewritten term changes width
and pushes its neighbors. Call `audit(fig)` once the composite is laid out to
vet every leader and link of the clip.

## Write: a label put down by hand

`write` is for a label or a short equation that should arrive the way a hand
would put it on a board. A thin pen traces the outline of each glyph in reading
order, then the glyph fills while the pen thins away; each glyph starts before
the one before it has finished, so a short front of a few glyphs is in flux at
once. The outlines come from the font the text itself uses, placed where the
renderer lays the text out, so the ghost sits on the text, and mathtext works
through the mathtext parser: every symbol is a glyph, and a fraction bar counts
as one. The filled glyphs cross-fade into the real text after `draw` of the
progress, so the last frame is the text as drawn. `lag` sets how much the
glyphs overlap one another and `lw` the pen width.
`equation(...).update(u, reveal="write")` writes an equation the same way.

```{code-cell} python
import eyepiece.testing as ept

fig = plt.figure(figsize=(9.0, 1.2), dpi=100)
note = fig.text(0.04, 0.4, r"Contrast $C = \frac{F_p}{F_s}$", fontsize=26)
fig.canvas.draw()
scribe = ep.write(note, lag=0.6)

shown, cells = plt.subplots(1, 5, figsize=(9.0, 1.0), layout="constrained")
for cell, u in zip(cells, (0.15, 0.3, 0.5, 0.75, 1.0), strict=True):
    scribe.update(u)
    ept.assert_readable(fig)
    fig.canvas.draw()
    cell.imshow(np.asarray(fig.canvas.buffer_rgba()).copy(), interpolation="nearest")
    cell.set_title(f"u = {u:g}", fontsize="small")
    cell.set_axis_off()
plt.show()
```

The ghost is hidden at ``u = 0``, where nothing is drawn, so a clip starts on
the empty figure and ends on the text. A `Stage` that owns the text weighs the
ghost too, because the ghost reads the text's color and opacity as it draws.

## Caption placement and wrapping

A caption at the player's default position, the bottom of the picture, lands
on the figure's own bottom line. `eyepiece.testing.caption_line` finds the
`line:` setting of a WebVTT cue that keeps the captions clear of the text the
figure shows at the times the captions are on screen. The cue itself is wrapped
at a clause boundary, 42 characters to a line and two lines to a cue.

```{code-cell} python
import eyepiece.testing as ept
from eyepiece import narration as nr
from matplotlib.patches import Rectangle

fig = plt.figure(figsize=(7.0, 3.6), dpi=100)
fig.text(0.5, 0.88, "Two kinds of world", fontsize=16, ha="center")
fig.text(0.5, 0.5, "band present: 0.24 of the sky", fontsize=14, ha="center")
fig.text(0.5, 0.05, "what is left: area 0.24 and 0.07", fontsize=14, ha="center")

line = ept.caption_line(fig)

sentence = (
    "Water worlds with the band cover 0.24, and dry worlds with the band "
    "cover 0.07."
)
words = sentence.split()
line_of_text = nr.Line("a", sentence, pad_after_s=0.0)
spoken = nr.Spoken(
    "x", 5.0, tuple(nr.Word(w, i * 5 / 14, (i + 1) * 5 / 14) for i, w in enumerate(words))
)
cues = nr.captions(nr.Narration(nr.Script((line_of_text,)), {"a": spoken}))

top = 1.0 - line / 100.0
fig.add_artist(Rectangle((0.1, top - 0.14), 0.8, 0.14, transform=fig.transFigure,
                         fc="black", alpha=0.75, ec="none"))
fig.text(0.5, top - 0.07, cues[0][2], color="white", ha="center", va="center",
         fontsize=11)
print(f"line:{line:g}%")
print(cues[0][2])
plt.show()
```

`write_vtt(narration, path, line=line)` writes the setting on every cue, and
`write_captions` writes the WebVTT and the SRT file together. A line of the
script may carry `display=` text, so the caption reads `Pr(H|E)` while the voice
says "P of H given E"; the cue marks stay on the spoken words. When no position
is clear, `caption_line` raises, and the clip reserves a caption band in its
layout instead.

## Words: color and reveal inside one line

`words` sets a line of text as one text per word, so a word of a sentence can
carry its own color, be the target of an anchor, or come in on its own beat.
The line is laid out once, as a single string, and each word's place comes
from the renderer, so the spacing is that of one `ax.text` and, without math,
the joined line is pixel for pixel the same picture. A word in ``$...$`` is
kept whole, and a line with math is measured and drawn as mathtext throughout,
as matplotlib draws it, which matches the single text to within a pixel. For
the choice between `words`, `term_row` and `readout`, see the `term_row`
docstring.

The reveal never moves anything. ``reveal(u, by="word")`` fades the words in
from left to right, each where it will stay; ``by="letter"`` spells each word
out the same way. At ``u = 1`` the line is the one that was drawn, so a still
shows the whole sentence.

```{code-cell} python
fig, ax = plt.subplots(figsize=(8.0, 0.7))
ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
ax.set_axis_off()
fig.canvas.draw()
fig.set_layout_engine("none")

line = ep.words(
    ep.Anchor.from_data(ax, (0.03, 0.3)),
    r"the prior times the likelihood is the $p(\theta \mid d)$",
    fontsize=15,
    colors={1: "C0", 4: "C1"},
)

shown, cells = plt.subplots(2, 2, figsize=(9.0, 2.0), layout="constrained")
for row, by in zip(cells, ("word", "letter"), strict=True):
    for cell, u in zip(row, (0.55, 1.0), strict=True):
        line.reveal(u, by=by)
        fig.canvas.draw()
        cell.imshow(
            np.asarray(fig.canvas.buffer_rgba()).copy(), interpolation="nearest"
        )
        cell.set_title(f"by {by}, u = {u:g}", fontsize=8)
        cell.set_axis_off()
plt.show()
```

Each word is an artist of the figure, so ``line.artists["text"][1]`` can be
given to `Anchor.from_artist`, `Stage`, `indicate`, or `callout`, and the
colors pick out the two factors the sentence names. In a clip the reveal runs
before the caption's words are spoken, and the words that stay on screen are
the same texts that the later moves point at.

## Underline a word

`underline` follows a single text or one artist from `words` at draw time.
The line draws on, holds, and fades, leaving the still unchanged.

```python
import matplotlib.pyplot as plt
import eyepiece as ep

fig, ax = plt.subplots()
label = ax.text(0.2, 0.5, "the prior", fontsize=20)
mark = ep.underline(label)
mark.update(0.5)
fig.canvas.draw()
```
