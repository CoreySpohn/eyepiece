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

# Make a still figure

Start with a synthetic coronagraph field: leftover starlight and one faint
companion. Choose a style explicitly, then give the image a pixel-edge extent
so that its coordinates and axis labels describe the same scale.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

hwostyle.use("dark")
n = 96
pixel_scale = 0.3
coordinate = (np.arange(n) - (n - 1) / 2.0) * pixel_scale
x, y = np.meshgrid(coordinate, coordinate)
radius = np.hypot(x, y)
angle = np.arctan2(y, x)
halo = 1e-7 * (1.0 + radius) ** -3 * (1.0 + 0.8 * np.cos(12 * angle) ** 2)
planet_x, planet_y = 8.0 * np.cos(2.4), 8.0 * np.sin(2.4)
planet = 3e-9 * np.exp(-0.5 * (np.hypot(x - planet_x, y - planet_y) / 0.6) ** 2)

result = ep.imshow_log(
    halo + planet,
    extent=ep.extent_lod_from_pixels(n, pixel_scale),
    vmin=1e-12,
    vmax=1e-7,
    floor=1e-12,
    colorbar=False,
)
ep.label_lod(result.ax)
result.ax.set_title("Synthetic coronagraph field")
plt.show()
```

The result holds the figure, axes, and image artist. Use `result.ax` to add
labels and marks. To save the finished still, pass `result.fig` to `save_fig`:

```python
ep.save_fig(result.fig, "coronagraph-field")
```

Continue with {doc}`building up a figure <../how-to/build-up>` to reveal a
finished axes in stages, or browse the {doc}`image catalog <../gallery/images>`
for other image displays.
