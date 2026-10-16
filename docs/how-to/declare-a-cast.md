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

# Declare a cast

Give recurring profiles and markers stable meaning keys through a `MotifSet`.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt

import eyepiece as ep
import eyepiece.testing as ept

hwostyle.use("dark")
cast = ep.MotifSet([
    ep.Motif("profile", "profile", role="truth"),
    ep.Motif("marker", "marker", role="planet", marker="o", linestyle="none"),
])
fig, ax = plt.subplots(figsize=(4.6, 2.6), layout="constrained")
ax.plot([0, 1], [0.2, 0.8], **cast["profile"].line_kw(gid="profile:row"))
ax.plot([0.5], [0.5], **cast["marker"].marker_kw(size=10))
ax.text(0.5, 0.85, "marker", **cast["marker"].label_kw())
ax.set(xlim=(0, 1), ylim=(0, 1))
keys = cast.meaning_keys()
print(ept.meaning_report(ept.meaning_faults(fig, keys=keys)))
plt.show()
```

Role kwargs such as `line_kw`, `marker_kw`, `patch_kw`, and `label_kw` resolve
colors in the active mode and attach `gid`, whose prefix identifies the motif.
`style.SourceCast` is a different, prepared-view cast.
See {doc}`the motifs catalog <../gallery/motifs>` for the full vocabulary.
