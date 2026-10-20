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

# Check the timing of a narrated clip

Use `eyepiece.testing.timing_faults` to check speech rate (160 words per
minute, with a 166 words per minute allowance for lines of at most 10 words),
the heard gap between lines, and the cue, undeclared silence, and hero span
rules of `nr.check_silence`. Pass `source` to find a still screen under speech,
or `readouts` and `at` to check that a number lands on its word.
Findings are returned as `TimingFault` objects; `assert_timing` takes the same
arguments and raises `TimingError` when there are findings.

## Check the voice

Build a narration with hand-made word times, measured from each line's audio
start. The first line is deliberately fast, and its 0.15 s trailing pad leaves
less than the default 0.4 s heard gap before the next line.

```{code-cell} python
from eyepiece import narration as nr
from eyepiece.testing import timing_faults

script = nr.Script((
    nr.Line("quick", "Watch {show:panel}this change.", pad_after_s=0.15),
    nr.Line("measured", "We measured {show:count}8 epochs."),
))
narration = nr.Narration(script, {
    "quick": nr.Spoken("quick", 0.75, (
        nr.Word("Watch", 0.0, 0.25),
        nr.Word("this", 0.25, 0.5),
        nr.Word("change.", 0.5, 0.75),
    )),
    "measured": nr.Spoken("measured", 2.0, (
        nr.Word("We", 0.0, 0.5),
        nr.Word("measured", 0.5, 1.0),
        nr.Word("8", 1.0, 1.5),
        nr.Word("epochs.", 1.5, 2.0),
    )),
})
for fault in timing_faults(narration):
    print(fault.kind, fault.key, fault.message)
```

## Check a still screen

This source changes once, after a still run containing more than the default
2 s of speech.

```{code-cell} python
import numpy as np


def source(t):
    shade = 60 if t < 3.0 else 180
    return np.full((54, 96, 3), shade, dtype=np.uint8)


for fault in timing_faults(narration, source=source):
    if fault.kind == "still-speech":
        print(fault.kind, fault.key, fault.message)
```

Pass a video path as `source` instead to sample an encoded clip, and use
`ignore_bottom=0.12` to exclude the bottom 12 percent for a burned-in caption band.

## Check that a number lands on its word

Map the line key and zero-based word index to the text artist, and let `at(t)`
pose it on the clip clock. Here the spoken 8 starts at 1.9 s, but appears at
2.4 s.

```{code-cell} python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(2.4, 1.0))
text = fig.text(0.5, 0.5, "", ha="center", va="center")


def at(t):
    text.set_text("18 epochs" if t < 2.4 else "8 epochs")


for fault in timing_faults(narration, at=at, readouts={("measured", 2): text}):
    if fault.kind == "number-late":
        print(fault.kind, fault.key, fault.message)
plt.close(fig)
```

Matching uses the first whole numeric token: "18 epochs" does not show 8.
For other narration and motion checks, see {doc}`check-narration-and-motion`.

## Write a review sheet

Write frames, captions, and a waveform to one PNG; this example uses a
synthetic tone in place of recorded speech.

```{code-cell} python
import tempfile
from pathlib import Path

from IPython.display import Image, display

from eyepiece.testing import review_sheet

rate = 8000
times = np.arange(int(narration.total_s * rate)) / rate
samples = 0.2 * np.sin(2 * np.pi * 220 * times)
with tempfile.TemporaryDirectory() as directory:
    path = review_sheet(
        Path(directory) / "timing.png",
        source,
        narration=narration,
        audio=(samples, rate),
        columns=2,
        width=200,
    )
    display(Image(filename=str(path)))
```

Default events include every cue, every beat end, and the middle and end of
every move when given a `plan`; pass your own `(seconds, label)` list as `events=`.
