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

# Points and intervals

`interval_points` draws supplied endpoints without estimating uncertainty.
Zero values have hollow faces; a missing endpoint stops the interval at the
point. Horizontal orientation exchanges the position and value axes.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import eyepiece as ep

hwostyle.use("dark")
plt.rcParams["savefig.dpi"] = 100
fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
ep.interval_points([0, 1, 2], [0, 2, 3], lo=[0, 1.5, 2.4],
                   hi=[0.4, 2.8, 3.5], ax=axes[0])
ep.interval_points([0, 1, 2], [0, 2, 3], hi=[0.4, 2.8, 3.5],
                   orient="h", ax=axes[1])
axes[0].set(xlabel="position", ylabel="value", title="Two endpoints")
axes[1].set(xlabel="value", ylabel="position", title="Upper endpoint only")
```

The `scatter` collection preserves input order. The `lines` list contains
one interval per point, making the supplied bounds available as line data.
