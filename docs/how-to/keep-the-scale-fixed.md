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

# Keep the scale fixed

Choose shared image bounds and fixed axis limits before animating a PSF.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece as ep

hwostyle.use("dark")
x, y = np.meshgrid(np.linspace(-3, 3, 32), np.linspace(-3, 3, 32))
cube = np.array([amplitude * np.exp(-(x**2 + y**2))
                 for amplitude in (1.0, 0.5, 0.25, 0.5)])
result = ep.imshow_log(cube[0], vmin=1e-5, vmax=1.0, floor=1e-5, colorbar=False)
result.fig.set_size_inches(3.0, 2.6)
result.ax.set(xlim=(-0.5, 31.5), ylim=(-0.5, 31.5), title="PSF")
result.fig.canvas.draw()
result.fig.set_layout_engine("none")


def draw(fig, k):
    result.update(cube[k])


anim = ep.animate(result.fig, draw, len(cube), fps=4)
HTML(anim.jshtml(dpi=80, on_rescale="raise"))
```

Updating the image data preserves the norm, so the same intensity keeps the
same color in every frame.
`on_rescale="raise"` rejects axis or norm drift and removes partial files,
while `"warn"` reports drift and `"ignore"` permits a deliberate rescale.
See {doc}`the animation catalog <../gallery/animation>` for the recording policy.
