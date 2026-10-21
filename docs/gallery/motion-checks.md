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

# Motion checks and captions

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small.
plt.rcParams["savefig.dpi"] = 110
```

## Readability check

A clip can draw without error and still be unreadable. `eyepiece.testing`
draws the figure at each sample time and lists what a viewer could not read.
Here a third label hangs four pixels into the second row of a step list, which
is the kind of collision no exception reports.

```{code-cell} python
import eyepiece as ep

fig = plt.figure(figsize=(6.0, 2.4), dpi=100)
fig.text(0.05, 0.8, "1  Steer the beam", fontsize=14)
row = fig.text(0.05, 0.62, "2  Hold the star on the mask", fontsize=14)
fig.canvas.draw()
low = row.get_window_extent().y0
label = fig.text(0.05, (low + 4.0) / fig.bbox.height, "axis label", fontsize=14, va="top")
for fault in ep.testing.readability_faults(fig):
    print(fault)
```

Pass `at` and `times` to sample a clip at its span ends. `assert_readable`
raises `ReadabilityError` with the same report, for a test suite.

## Motion check

`readability_faults` looks at states. `eyepiece.testing.motion_faults` looks at
the path between them: it poses the clip at every frame of a span and reports
what a viewer would see as a snap, a stall, a drifting label, a torn row or a
term that leaves the frame. Here a slider eases toward its rest position, but
its last frame lands 60 pixels past the end of the ease, and a second label
that should stay put creeps by 3 pixels.

```{code-cell} python
import eyepiece as ep
from matplotlib.patches import Rectangle
from matplotlib.transforms import IdentityTransform

fig = plt.figure(figsize=(6.0, 2.4), dpi=100)
fig.patch.set_facecolor("white")
slider = Rectangle((30, 90), 40, 30, transform=IdentityTransform(), color="k")
anchor = fig.text(0.05, 0.15, "reference label", fontsize=14, color="black")
fig.add_artist(slider)


def at(t):
    u = min(1.0, t / 2.0)
    u = u * u * (3.0 - 2.0 * u)
    slider.set_x(30 + 400 * u + (60.0 if t >= 2.0 else 0.0))
    anchor.set_x(0.05 + 0.005 * u)


for fault in ep.testing.motion_faults(fig, at, ep.testing.Window(0.0, 2.0), rest=[anchor]):
    print(fault)
```

The report names each fault by kind, time and artist. `assert_motion` raises
`MotionError` with the same report, for a test suite. Declare a move's
artists, its `rest` set, its `groups` and the pairs that `may_overlap` to
let the check separate a deliberate motion from a defect.

## Check a whole clip

`check_clip` combines readability, meaning, motion, plan, unread-knob and
speech-rate checks in one `ClipReport`. Readability and meaning sample the
plan's review frames, including cues; motion samples every frame. The draw
callable takes a frame index and sets the figure from the plan, just as in
a recording loop. Checking does not write any files.

```{code-cell} python
import matplotlib.pyplot as plt
import eyepiece as ep
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform

fig = plt.figure(figsize=(4.0, 2.0), dpi=100)
dot = Line2D([40], [80], marker="o", color=ep.style.neutral(1.0), linestyle="none",
             transform=IdentityTransform())
fig.add_artist(dot)
timeline = ep.clock.Timeline({"x": 40.0})
timeline.hold(0.3)
timeline.run("slide", 0.8, to={"x": 48.0})
timeline.hold(0.3)
plan = timeline.at_fps(10)


def draw(k):
    dot.set_xdata([plan.value(k, "x")])


report = ep.testing.assert_clip(fig, draw, plan, moves={dot: "slide"})
print(report)
plt.show()
```

Use `check_clip` to inspect `report.faults`, a list of `(kind, detail)` pairs.
`assert_clip` returns the report when clean and raises `ClipError` naming every
finding otherwise. Pass `motifs=motifs` to check the meanings attached by
`Motif.line_kw()`, or pass an explicit `keys` mapping. With neither argument,
the meaning check is skipped.

### Declare intentional steps

A discrete sweep can change a marker position or a label at each hold.
`plan.step_frames()` returns the sorted unique frames where a step write
changes a knob value. A write that repeats the held value does not count.
Frame `k` is the first frame showing the new value, including frame zero
when it differs from the initial value.

`check_clip` always declares these frames. For changes made outside the plan,
pass `steps=[frame_index, "span_name"]`; a span name declares its first frame
`k0`, and an unknown name raises `KeyError`. These declarations add to the
automatic steps. Only jumps ending at a declared frame are omitted, including
steps into rest. All other checks still run. When calling `motion_faults`
directly, `steps` contains times in seconds instead of frame indices. Matching
uses an absolute tolerance of `1e-9` seconds; a jump one sample away is still
reported.

```{code-cell} python
import matplotlib.pyplot as plt
import eyepiece as ep
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform

fig = plt.figure(figsize=(4.0, 2.0), dpi=100)
dot = Line2D([40], [80], marker="o", color=ep.style.neutral(1.0), linestyle="none",
             transform=IdentityTransform())
fig.add_artist(dot)
timeline = ep.clock.Timeline({"x": 40.0}, tempo=ep.clock.Tempo(factor=1.0))
timeline.hold(0.5)
timeline.hold(0.5, name="next_position", to={"x": 160.0})
timeline.hold(0.5, to={"x": 280.0})
plan = timeline.at_fps(10)


def draw(k):
    dot.set_xdata([plan.value(k, "x")])


print(plan.step_frames())  # (5, 10)
print(ep.testing.assert_clip(fig, draw, plan))
# Additional declarations can use either form, even for the same frame.
print(ep.testing.assert_clip(fig, draw, plan, steps=[5, "next_position"]))
plt.show()
```

## Numbers for the voice

Use the figure's value and format specification with `ep.narration.say` so the spoken
digits keep the same precision. Exponents, percentages and units become
words the voice reads correctly; captions can retain the compact notation.
`ep.narration.check_script_speech` checks spoken text before synthesis and can also be
called without a speech engine. A fault raises `ep.narration.SpeechTextError` with the
line key and offending token. `ep.narration.speech_faults` inspects text directly.

```{code-cell} python
import eyepiece as ep

contrast = 3e-8
spoken = f"The contrast is {ep.narration.say(contrast, '.0e')}."
line = ep.narration.Line("contrast", spoken, display=f"The contrast is {contrast:.0e}.")
script = ep.narration.Script((line,))
ep.narration.check_script_speech(script)
print(line.spoken)
print(line.display)
print(ep.narration.say(-0.126, ".1%"))
print(ep.narration.say(0.943, ".3f"))
print(ep.narration.say(0.55, ".2f", unit="um"))
print(ep.narration.speech_faults(line.spoken))
```

## Captions at a playback time

`ep.narration.caption_at(cues, t)` returns the text to show at time `t` in seconds.
A cue includes its start and excludes its end; gaps and times outside the
cues return an empty string. Generate the cues once with `ep.narration.captions` and
look up the text as the playback clock advances. Cue and line breaks keep
bracketed text together and avoid either side of a token with no letters or
digits, such as `/` or `-`. If every break is forbidden, the usual split
still lets a long sentence fit.

```{code-cell} python
import eyepiece as ep

line = ep.narration.Line("ratio", "Compute a / b - 1; cap +/-1% applies.")
script = ep.narration.Script((line,))
narration = ep.narration.Narration(script, {"ratio": ep.narration.estimate(line)})
cues = ep.narration.captions(narration, max_chars=14)
print(ep.narration.caption_at(cues, cues[0][0]))
print(repr(ep.narration.caption_at(cues, cues[-1][1])))
```
