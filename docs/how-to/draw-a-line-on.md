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
See {doc}`the attention catalog <../gallery/motion-attention>` for other reveals.
