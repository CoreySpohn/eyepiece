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

# Weighted draws

Use `weighted_draws` for an importance-weighted ensemble or a particle
filter. Twice the weight means twice the marker area, including very small
weights. The alpha encoding uses one marker size and scales opacity instead.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece as ep

hwostyle.use("dark")
plt.rcParams.update({"figure.dpi": 90, "savefig.dpi": 90})
rng = np.random.default_rng(4)
points = rng.normal(size=(60, 2))
weights = np.exp(-((points[:, 0] - 0.8)**2 + points[:, 1]**2) / 0.6)
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), layout="constrained")
for ax, encode in zip(axes, ("area", "alpha"), strict=True):
    mark = ep.plot.weighted_draws(points, weights, ax=ax, encode=encode, max_size=180)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="parameter x",
           ylabel="parameter y", title=encode)
print(f"Effective sample size: {mark.ess:.1f} of {len(points)}")
plt.show()
```

Pass a seed to play systematic resampling. Positions stay fixed, and each
parent ends with area proportional to its offspring count. Parents with no
offspring vanish. The weighted and resampled sets share one area scale, so
the change in size can be compared throughout the beat.

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.6, 2.8), layout="constrained")
mark = ep.plot.weighted_draws(points, weights, ax=ax, resample=3, max_size=180)
ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="parameter x", ylabel="parameter y")
fig.canvas.draw()
fig.set_layout_engine("none")
print("Offspring counts:", mark.counts[:12])
print("Final weights:", np.round(mark.weights(1.0)[:12], 3))


def draw(fig, k):
    mark.update(ep.clock.ease(k / 29))


HTML(ep.render.animate(fig, draw, 30, fps=10).jshtml(dpi=90))
```

Without `resample`, `update(u)` holds the original weights. To show separate
unweighted draws over time, use `hop` in {doc}`motion-clock`.
