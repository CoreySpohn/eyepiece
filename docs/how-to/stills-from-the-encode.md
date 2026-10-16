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

# Stills from the encode

Write selected PNG frames during the same pass that records the clip.

```{code-cell} python
import tempfile
from pathlib import Path

import hwostyle
import matplotlib.pyplot as plt

import eyepiece as ep

hwostyle.use("dark")
tl = ep.Timeline({"height": 1.0}, tempo=ep.Tempo(factor=1.0))
tl.hold(0.5, name="open")
tl.run("rise", 0.5, to={"height": 3.0})
plan = tl.at_fps(4)
fig, ax = plt.subplots(figsize=(3.0, 2.2), layout="constrained")
(line,) = ax.plot([0, 1], [0, 1], color=ep.neutral(0.85))
ax.set(xlim=(0, 1), ylim=(0, 4))
fig.canvas.draw()
fig.set_layout_engine("none")

with tempfile.TemporaryDirectory() as directory:
    output = Path(directory)
    stills = {output / "open.png": "open", output / "middle.png": ("rise", "mid")}
    with ep.record(fig, output / "clip.html", fps=plan.fps, dpi=80,
                   plan=plan, stills=stills, on_rescale="raise") as rec:
        for k in range(plan.n_frames):
            line.set_ydata([0, plan.value(k, "height")])
            rec.frame()
    print(sorted(path.name for path in output.glob("*.png")))
```

A span name selects its settled last frame, a `(span, "mid")` tuple selects
its middle frame, and a number selects a time in seconds.
The `.html` file is only the first sink of the recording, here a stand-in for
your video path; `on_rescale="raise"` stops the recording if a draw function
moves an axis limit. Each still matches the frame captured by the first sink,
and this example removes its temporary output when the context ends.
See {doc}`the animation catalog <../gallery/animation>` for other sinks.
