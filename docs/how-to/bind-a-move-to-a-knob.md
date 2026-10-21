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

# Bind a move to a knob

Drive a marker pulse from the plan rather than from elapsed time.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt

import eyepiece as ep

hwostyle.use("dark")
fig, ax = plt.subplots(figsize=(4.0, 2.5), layout="constrained")
(marker,) = ax.plot([0.5], [0.5], "o", color=hwostyle.palette[0], ms=8)
ax.set(xlim=(0, 1), ylim=(0, 1))
pulse = ep.motion.indicate(marker, scale=1.5)
tl = ep.clock.Timeline(tempo=ep.clock.Tempo(factor=1.0))
tl.section("Marker")
tl.indicate("marker", at=tl.now)
tl.hold(1.5, name="watch")
plan = tl.at_fps(10)


def draw(k):
    pulse.update(plan.value(k, "indicate.marker"))


draw(plan.frame_of("indicate.marker", "mid"))
plt.show()
```

Place a silent pulse with `at=` and provide a hold long enough to contain it.
Use `scale=` for a width bump on a line, or a size bump on a marker. Any knob
reads the same way: `plan.value(k, name)` returns its value on frame `k`, and a
move's `update` takes the raw value.
See {doc}`the attention catalog <../gallery/motion-attention>` and
{doc}`the clock catalog <../gallery/motion-clock>`.
