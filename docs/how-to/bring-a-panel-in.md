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

# Bring a panel in

Fade a finished axes with `Stage.weigh`, then build its axes and title.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

hwostyle.use("dark")
x = np.linspace(1.0, 12.0, 100)
fig, ax = plt.subplots(figsize=(5.0, 3.0), layout="constrained")
(curve,) = ax.plot(x, 1e-7 * (1.0 + x) ** -3, color=ep.neutral(0.85))
ax.set(xlim=(1, 12), ylim=(1e-11, 2e-8), yscale="log",
       xlabel="separation [lambda/D]", ylabel="contrast", title="Starlight profile")
fig.canvas.draw()
fig.set_layout_engine("none")
stage = ep.Stage({"panel": ax})
content = ep.Stage({"profile": [curve]})
build = ep.build_axes(ax, title=True)


def draw(panel_u, axes_u, profile_u):
    stage.weigh({"panel": panel_u})
    build.update(axes_u)
    content.weigh({"profile": profile_u})


stage.set(shown=[])
draw(1.0, 1.0, 0.0)
content.rest()
stage.rest()
build.update(1.0)
plt.show()
```

Freeze the layout after setting limits, ticks, labels, and titles, then apply
Stage before the build in every frame.
`build.update(1.0)` restores the axes as made, and omitting `title=True` keeps
titles visible throughout. When the panel group of a Stage holds the axes, it
also dims the titles, and the build restores their opacity once it finishes, so
dim the panel only while the build is still running or keep titles out of the
Stage group.
Keep artists recolored on every frame outside Stage groups, since Stage owns
their colors, and see {doc}`the staging catalog <../gallery/motion-staging>`
and {doc}`the clock catalog <../gallery/motion-clock>` for more examples.
