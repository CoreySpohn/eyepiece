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

# Publish a scrubber page

Export a clip as one HTML page whose slider is a knob of the plan.

A slider beats a clip when the claim is a function of the knob: a sweep of
contrast, of separation, or of exposure time. A viewer drags back and forth
and compares the first sample with the last, which a cut from one frame to
the next cannot offer. A clip still wins for motion whose timing is the
point, and for narration that has to be heard.

The page is also the way to publish a clip where video does not play. Some
page viewers, such as a private Claude Artifact page, do not play H.264, and
a frame stack needs nothing but images and a few lines of script.

```{code-cell} python
import tempfile
from pathlib import Path

import hwostyle
import matplotlib.pyplot as plt

import eyepiece as ep

hwostyle.use("dark")
tl = ep.clock.Timeline({"height": 1.0}, tempo=ep.clock.Tempo(factor=1.0))
tl.hold(0.5, name="open")
tl.run("rise", 1.0, to={"height": 3.0})
tl.hold(0.5, name="settled")
plan = tl.at_fps(10)
fig, ax = plt.subplots(figsize=(3.0, 2.2), layout="constrained")
(line,) = ax.plot([0, 1], [0, 1], color=ep.style.neutral(0.85))
ax.set(xlim=(0, 1), ylim=(0, 4), xlabel="x", ylabel="height")
fig.canvas.draw()
fig.set_layout_engine("none")


def draw(k):
    line.set_ydata([0, plan.value(k, "height")])


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "rise.html"
    page = ep.render.to_scrubber(fig, draw, plan, "height", [1.0, 1.5, 2.0, 2.5, 3.0],
                          path, dpi=80, unit="m")
    print(f"{path.name}: {path.stat().st_size} bytes, 5 samples")
plt.close(fig)
```

Each sample is a value of the knob, and the page shows the frame whose value
is nearest it. The last sample here is the knob's final value, so the image a
reader sees before any script runs is the settled figure. The frames are the
PNG stills that `record` writes for the same frames at the same dpi, so the
page and the clip agree frame for frame.

The page loads nothing from the network: every frame is an inline data URI,
and two renders of the same inputs give the same bytes. Pass `max_bytes` to
change the 8 MB budget; a page over it raises with its size rather than
shrinking the frames, and `format="jpeg"` with a `quality`, fewer samples,
or a lower dpi brings it under. Use `"t"` as the knob to scrub time itself.

With a narration, each span a cue targets becomes a chapter button and the
spoken lines are listed under the figure, with the line at the slider
highlighted. See {doc}`the narrated tutorial <../tutorials/narrated>` for
such a plan, and {doc}`stills from the encode <stills-from-the-encode>` for
the stills the frames come from.
