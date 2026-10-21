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

# Reference marks

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import eyepiece as ep

hwostyle.use("dark")
plt.rcParams["savefig.dpi"] = 100
```

## Excluded regions: `shade_intervals`

`shade_intervals` shades the runs where a boolean mask is True, placing the
boundaries halfway between samples. Set `log=True` for geometric midpoints, or
supply explicit `intervals` whose infinite ends clip to the current limits.

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
x = np.arange(4)
ax.plot(x, [0.2, 0.6, 0.8, 0.9])
in_range = np.array([True, False, False, True])
ep.shade_intervals(x, ~in_range, ax=ax)
ax.set(xlabel="position", ylabel="response")
```

The shading covers [0.5, 2.5], retaining the same boundaries even when the
curve between samples is steep.

## Labeled thresholds: `ref_line`

`ref_line` anchors a label to the axes edge at a reference value.

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
ax.plot([0, 1, 2], [0.5, 1.3, 0.9])
ep.ref_line(ax, 1, "threshold", text_kw={"fontsize": 9})
ax.set(xlabel="position", ylabel="response")
```

Changing the data limits keeps the label at the right edge; `orient="v"`
places it above a vertical reference instead.

## A zero row: `log_zero_floor`

`log_zero_floor` decorates a log axis with a zero label and spine break.
The caller maps zero values to the positive display floor.

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
values = np.array([0.1, 0.02, 0.0, 0.005])
floor = 1e-4
ax.plot(np.arange(4), np.where(values == 0, floor, values), "o")
ep.log_zero_floor(ax, floor)
ax.set(xlabel="sample", ylabel="error")
```

The tick at the floor reads zero while ticks above it retain logarithmic labels.
