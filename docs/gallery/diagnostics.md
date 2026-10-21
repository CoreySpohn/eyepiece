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

# Calibration and sample strips

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import eyepiece as ep

hwostyle.use("dark")
plt.rcParams["savefig.dpi"] = 100
rng = np.random.default_rng(0)
```

## Stated against realized: `calibration`

`calibration` puts both quantities on identical limits with an identity line.
Use a linear scale for probabilities, or a log scale and an explicit display
floor when zeros must remain visible.

```{code-cell} python
fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
stated = np.linspace(0.1, 0.9, 9)
realized = stated + rng.normal(0, 0.025, len(stated))
ep.plot.calibration(stated, realized, ax=axes[0], scale="linear",
               lo=realized - 0.04, hi=realized + 0.04,
               hollow=stated > 0.7, tolerance=0.05)
ep.plot.calibration([0.001, 0.01, 0.1], [0, 0.012, 0.08], ax=axes[1],
               floor=1e-4, hollow=[True, False, False])
for ax in axes:
    ax.set(xlabel="stated", ylabel="realized")
```

The band is an absolute tolerance around identity. Hollow markers encode the
supplied mask; zero-valued points on the log panel occupy the row labeled zero.

## Jittered groups: `strip`

`strip` spreads each group's samples across a seeded, fixed-width column.
Values at or above a cap appear as upward triangles at that cap.

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
ep.plot.strip({"baseline": rng.lognormal(-2, 0.8, 40),
          "variant": rng.lognormal(-1.5, 0.9, 40)},
         ax=ax, capped=0.6, seed=7)
ep.plot.ref_line(ax, 0.6, "cap")
ax.set(yscale="log", ylabel="error")
```

Repeated calls with the same seed reproduce the jitter without changing
NumPy's global random state. Width is the total width of each column.
