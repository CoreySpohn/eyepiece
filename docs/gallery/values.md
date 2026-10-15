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

## A stamp that reserves its band

`stamp(fig, reserve=True)` puts the provenance line along the bottom edge and
shrinks the layout above it, so the stamp never lands on a tick label and the
caller does not edit the layout rectangle by hand. The date defaults to the
commit date of the script's repository, else today, so rebuilding an unchanged
figure repeats the same stamp. `save_fig` then writes the file with no
timestamp: two saves of one figure are byte-identical in PNG, SVG and PDF.

```{code-cell} python
fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
ax.plot(x, y, ".")
ax.set(xlabel="x", ylabel="y")
result = ep.stamp(fig, note="simulated", seed=0, reserve=True)
```

The result is a `PlotResult` with the stamp under `text`; its `update(u,
level)` scales the stamp's opacity.
