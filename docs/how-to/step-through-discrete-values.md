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

# Step through discrete values

Hold each profile sample index as a discrete value in the plan.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep
import eyepiece.testing as ept

hwostyle.use("dark")
x = np.array([0.2, 0.5, 0.8])
y = np.array([0.3, 0.7, 0.4])
fig, ax = plt.subplots(figsize=(4.6, 2.8), dpi=100, layout="constrained")
ax.plot(x, y, color=ep.neutral(0.6))
(marker,) = ax.plot([x[0]], [y[0]], "o", color=hwostyle.palette[0], ms=8)
ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
fig.canvas.draw()
fig.set_layout_engine("none")
tl = ep.Timeline({"sample": 0}, tempo=ep.Tempo(factor=1.0))
for index in range(len(x)):
    tl.hold(0.5, name=f"sample_{index}", to={"sample": index})
plan = tl.at_fps(10)


def draw(k):
    index = int(plan.value(k, "sample"))
    marker.set_data([x[index]], [y[index]])


print(plan.step_frames())
print(ept.assert_clip(fig, draw, plan))
plt.show()
```

Each `hold(seconds, to=...)` changes its value at the first frame and keeps it
until the next write, so these steps are declared automatically to the clip
checker.
See {doc}`the clock catalog <../gallery/motion-clock>` for timing and
{doc}`the checks catalog <../gallery/motion-checks>` for step declarations.
