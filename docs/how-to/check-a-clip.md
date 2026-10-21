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

# Check a clip

Inspect a plan-driven clip and assert that its declared motion is clean.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform

import eyepiece as ep

hwostyle.use("dark")
fig = plt.figure(figsize=(4.0, 2.0), dpi=100)
marker = Line2D([40], [80], marker="o", color=ep.style.neutral(1.0),
                linestyle="none", transform=IdentityTransform())
fig.add_artist(marker)
tl = ep.clock.Timeline({"x": 40.0}, tempo=ep.clock.Tempo(factor=1.0))
tl.hold(0.3, name="open")
tl.run("slide", 0.8, to={"x": 48.0})
tl.hold(0.5, name="next", to={"x": 160.0})
plan = tl.at_fps(10)


def draw(k):
    marker.set_xdata([plan.value(k, "x")])


report = ep.testing.check_clip(fig, draw, plan, moves={marker: "slide"}, steps=["next"])
print(report.faults)
print(ep.testing.assert_clip(fig, draw, plan, moves={marker: "slide"}, steps=["next"]))
plt.show()
```

Use plan span names in `moves` to declare the interval for each moving artist.
Hold writes declare steps automatically, while `steps` adds intended changes
made outside the plan as frame indices or span names.
See {doc}`the checks catalog <../gallery/motion-checks>` for every check.
