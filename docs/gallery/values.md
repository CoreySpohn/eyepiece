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

# Points colored by values

`scatter_values` pairs a scalar-colored point cloud with an inset colorbar.
Role names follow the active style at call time; `statistic` uses PuOr until
the style supplies its own mapping.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import eyepiece as ep

hwostyle.use("dark")
plt.rcParams["savefig.dpi"] = 100
rng = np.random.default_rng(0)
x, y = rng.normal(size=(2, 60))
fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
ep.scatter_values(x, y, np.exp(-(x*x + y*y)), ax=axes[0],
                  cmap="probability", vmin=0, vmax=1, cbar_label="probability")
ep.scatter_values(x, y, x - y, ax=axes[1], cmap="statistic",
                  vmin=-3, vmax=3, cbar_label="signed statistic")
for ax in axes:
    ax.set(xlabel="x", ylabel="y")
```

The colorbars sit to the right of their panels without changing the caller's
subplot slots. The result exposes the points under `scatter`, the colorbar
under `cbar`, and its axes in `insets`.
