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

# Build up a figure

Finish the limits, ticks, labels, and layout before making a build. Draw the
canvas once and freeze the layout so that a reveal keeps every mark in place.
Set `title=True` to bring in the axes titles during the final label stage.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

hwostyle.use("dark")
separation = np.linspace(1.0, 12.0, 100)
contrast = 1e-7 * (1.0 + separation) ** -3
fig, ax = plt.subplots(figsize=(5.0, 3.0), layout="constrained")
(curve,) = ax.plot(separation, contrast)
ax.set(
    xlim=(1.0, 12.0),
    ylim=(1e-11, 2e-8),
    yscale="log",
    xlabel=r"separation [$\lambda/D$]",
    ylabel="contrast",
    title="Synthetic starlight profile",
)
fig.canvas.draw()
fig.set_layout_engine("none")
build = ep.build_axes(ax, title=True)
stage = ep.Stage({"profile": [curve]})

build.update(1.0)
stage.rest()
plt.show()
```

In a draw function, call `build.update(u)` with progress from zero to one.
Use `stage.set(shown=[])` while the axes arrive, then `stage.rest()` to show
the profile. Calling `build.update(1.0)` restores the axes as made, including
the titles. Omit `title=True` to keep titles visible throughout the build.

Keep artists that the draw function recolors every frame out of Stage groups,
because the stage overwrites their colors. See the
{doc}`staging catalog <../gallery/motion-staging>` for a row gathered into a
bar, and the {doc}`clock catalog <../gallery/motion-clock>` for timing a clip.
