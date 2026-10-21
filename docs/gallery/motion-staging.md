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

# Motion staging

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small.
plt.rcParams["savefig.dpi"] = 110
```

## Staging: build an axis, dim what was read, gather a row

An explanation that opens on a full figure leaves the viewer to find the
subject. Three more moves let a clip open nearly empty and bring in one
thing at a time. `build_axes` draws an axis on in front of the viewer: the
spine from the origin, each tick as the spine reaches it, its number just
after, and the axis name last. `Stage` sets named groups to absolute levels,
present or absent, full or dimmed toward the background, so what the viewer
has already read recedes. `collapse` gathers a row of cells into a bar whose
length is their sum, and can carry the bar to the panel that shows the sums.
Each has an absolute `update` and a rest state equal to the finished figure.

```{code-cell} python
albedo = np.linspace(0.1, 0.7, 25)
row = np.exp(-0.5 * ((albedo - 0.47) / 0.05) ** 2)
edges = np.linspace(0.0875, 0.7125, 26)
cells = np.stack([np.c_[edges[:-1], [0.8] * 25], np.c_[edges[1:], [0.8] * 25]], 1)

fig, (left, right) = plt.subplots(
    1, 2, figsize=(6.0, 2.6), sharey=True, layout="constrained"
)
left.set(xlim=(0.05, 0.75), ylim=(0.5, 2.0), xlabel="albedo", ylabel="radius")
right.set(xlim=(0.0, 8.0), xlabel="row total")
(curve,) = left.plot(albedo, 0.3 / albedo**0.5 + 0.2, color=ep.neutral(0.7))
fig.canvas.draw()
fig.set_layout_engine("none")

build = ep.build_axes(right)
stage = ep.Stage({"scenery": [curve]})
start = ep.Anchor.from_data(left, (0.0875, 0.8))
end = ep.Anchor.from_data(left, (0.4, 0.8))
landing = (
    ep.Anchor.from_data(right, (0.0, 0.8)),
    ep.Anchor.from_data(right, (row.sum(), 0.8)),
)
gather = ep.collapse(
    left, cells, row, start, end, fig=fig, colors=plt.cm.Greys(row), color="C3",
    to=landing,
)


def grab(axis_u, dim_u, gather_u, carry_u):
    build.update(axis_u)
    stage.set(dimmed=["scenery"], u=dim_u)
    gather.update(gather_u, carry=carry_u)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


frames = [grab(0.4, 0.0, 0.0, 0.0), grab(1.0, 1.0, 0.6, 0.0), grab(1.0, 1.0, 1.0, 0.6)]
shown, panels = plt.subplots(1, 3, figsize=(9.0, 1.9), layout="constrained")
titles = ("axis drawn on", "row gathered", "bar carried")
for panel, frame, title in zip(panels, frames, titles, strict=True):
    panel.imshow(frame, interpolation="nearest")
    panel.set_title(title)
    panel.set_axis_off()
plt.show()
```

In a clip the frame's time sets each move's progress, so any frame draws
itself from its state alone, and the last frame, with every move at rest,
is the composite still.

## Mark a carried panel

Use a small relay note when a panel returns from the preceding clip. The
tag uses the active neutral ramp and stays a fixed distance from the top
corner. It fades with the panel, is not lifted as an arriving label, and
is furniture to meaning and text-size checks. Overlap and clipping checks
still apply.

```python
import matplotlib.pyplot as plt
import eyepiece as ep

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
tag = ep.carried_tag(ax, "(from clip 1)", corner="upper right")
stage = ep.Stage({"panel": ax})
stage.set(dimmed=["panel"])
stage.rest()
```
