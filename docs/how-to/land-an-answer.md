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

# Measure and land an answer

Use `ruler_readout` when a curve, a ruler and a counter must agree on every
frame. Finalize the layout before binding them: the counter reserves its
digit slots by sampling the drawn path.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece as ep

hwostyle.use("dark")
plt.rcParams.update({"figure.dpi": 90, "savefig.dpi": 90})


def measurement():
    fig, (image, ax) = plt.subplots(
        1, 2, figsize=(6.4, 2.8), layout="constrained"
    )
    image.set(xlim=(-0.1, 1.1), ylim=(-0.4, 0.4), xlabel="radius")
    image.set_yticks([])
    ax.set(xlim=(-0.04, 1.05), ylim=(0, 115), xlabel="radius", ylabel="count")
    radius = np.linspace(0, 1, 120)
    (curve,) = ax.plot(radius, 100 * (1 - np.exp(-3 * radius)))
    count = ep.readout(
        ep.Anchor.from_data(ax, (0.05, 103)), 0, fmt="d", digits=3
    )
    rule = ep.ruler(image, (0, 0), (0, 0), "r = 0.00", head_scale=0)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    head = ep.head_draw(curve)
    measured = ep.ruler_readout(
        head, readout=count, value=lambda r, n: n,
        ruler=rule, ruler_to=lambda r, n: (r, 0),
        ruler_text=lambda r, n: f"r = {r:.2f}",
    )
    return fig, curve, count, rule, measured


fig, curve, count, rule, measured = measurement()
measured.update(0.6)
plt.show()
```

`measured.update(u)` poses the head first and reads the displayed tip back.
The ruler uses its radius and the counter uses its count. Interpolating the
count separately would disagree with the head on this curved path.

Use `click` to land the completed measurement: ink swells, an end dot grows,
the number turns bold and the ruler recedes. Here is the middle of that beat.

```{code-cell} python
fig, curve, count, rule, measured = measurement()
measured.update(1.0)
landing = ep.click(
    curve, dot=ep.Anchor.from_data(curve.axes, curve.get_xydata()[-1]),
    readout=count, ruler=rule,
)
landing.update(0.5)
plt.show()
```

`Timeline.click` schedules a 0.6 second beat and at least 1.5 seconds of
stillness afterwards, even at a fast tempo. Drive its knob directly and let
the plan keep that rest. This small player has 30 frames at 10 fps.

```{code-cell} python
fig, curve, count, rule, measured = measurement()
landing = ep.click(
    curve, dot=ep.Anchor.from_data(curve.axes, curve.get_xydata()[-1]),
    readout=count, ruler=rule,
)
tl = ep.Timeline({"measure": 0.0}, tempo=ep.Tempo(factor=1.0))
tl.run("measure", 0.9, to={"measure": 1.0})
tl.click("answer")
plan = tl.at_fps(10)
print(plan.spans_json())


def draw(fig, k):
    landing.update(0.0)
    measured.update(plan.value(k, "measure"))
    landing.update(plan.value(k, "click.answer"))


HTML(ep.animate(fig, draw, plan.n_frames, fps=plan.fps).jshtml(dpi=90))
```

For binding a move to a plan, see {doc}`bind-a-move-to-a-knob`.
