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

# Draw a line on

Reveal a profile along its displayed path with `ep.draw_on`.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

hwostyle.use("dark")
x = np.geomspace(1.0, 100.0, 200)
fig, ax = plt.subplots(figsize=(4.6, 2.8), layout="constrained")
(curve,) = ax.plot(x, 1.0 / x, color=ep.neutral(0.85))
ax.set(xscale="log", xlim=(1, 100), ylim=(0, 1), xlabel="separation",
       ylabel="relative intensity")
fig.canvas.draw()
fig.set_layout_engine("none")
reveal = ep.draw_on(curve)
reveal.update(0.6)
plt.show()
```

The reveal follows the path as drawn. A two-point line on a log axis draws
straight in log space, so a line that stands for a linear rule needs dense
samples or a linear axis.

`update(0.0)` hides the line and `update(1.0)` restores its full path, markers,
and dashes.

For the signature reveal of a talk, `ep.head_draw` draws at the same constant
arc-length speed with a bright head on the tip: the last tenth of the drawn
length brightens toward the text color with a dot on the tip, and the head
dissolves into the line's own color as the path completes, so the last frame is
the plain line. Its default ease is the library's reveal ease, and `lag=`
staggers several lines.

```{code-cell} python
reveal.update(1.0)
head = ep.head_draw(curve)
head.update(0.6)
plt.show()
```

See {doc}`the attention catalog <../gallery/motion-attention>` for other reveals.
