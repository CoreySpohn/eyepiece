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

hwostyle.use("dark")
cast = ep.motifs.MotifSet([
    ep.motifs.Motif("profile", "profile", role="truth"),
    ep.motifs.Motif("marker", "marker", role="planet", marker="o", linestyle="none"),
])
fig, ax = plt.subplots(figsize=(4.6, 2.6), layout="constrained")
ax.plot([0, 1], [0.2, 0.8], **cast["profile"].line_kw(gid="profile:row"))
ax.plot([0.5], [0.5], **cast["marker"].marker_kw(size=10))
ax.text(0.5, 0.85, "marker", **cast["marker"].label_kw())
ax.set(xlim=(0, 1), ylim=(0, 1))
keys = cast.meaning_keys()
print(ep.testing.meaning_report(ep.testing.meaning_faults(fig, keys=keys)))
plt.show()
```

Role kwargs such as `line_kw`, `marker_kw`, `patch_kw`, and `label_kw` resolve
colors in the active mode and attach `gid`, whose prefix identifies the motif.
`style.SourceCast` is a different, prepared-view cast.
See {doc}`the motifs catalog <../gallery/motifs>` for the full vocabulary.

## Color a label apart from its mark

When a motif's hue is too faint to read as words, name its role in
`label_color`. The text takes that role's label ink, a darker or lighter
version of the hue made for text. `Motif.resolve_label(mode)` returns the
ink for a named mode; the `label` property returns it in the active mode.
Pass `label_kw()` to `ax.text` to use that color with the motif's `gid` and
its font style, when set.

Compare the mark and label colors in dark mode, with and without `label_color`.

```{code-cell} python
hwostyle.use("dark")
labels = [
    ep.motifs.Motif("ink", "label ink", role="zodi", label_color="zodi", fontstyle="italic"),
    ep.motifs.Motif("plain", "mark color", role="zodi"),
]
fig, ax = plt.subplots(figsize=(4.6, 1.6), layout="constrained")
for y, motif in enumerate(labels):
    mark, ink = motif.resolve("dark"), motif.resolve_label("dark")
    print(motif.key, "mark:", mark, "label:", ink)
    ax.plot([0.05, 0.25], [y, y], **motif.line_kw())
    ax.text(0.35, y, motif.noun, va="center", **motif.label_kw())
ax.set(xlim=(0, 1), ylim=(-0.5, 1.5))
ax.axis("off")
plt.show()
```

If the installed hwostyle has no label ink for the role, mode or palette
family, the label falls back to the mark color. hwostyle 1.6.0 and earlier
have no label inks, and labels there take the mark color.

## Accept a close pair with a second channel

Some role pairs that appear together stay below the usual color separation
in one mode. `MotifSet.default().accepted` lists them as `AcceptedPair`
records: the roles `a` and `b`, the `modes`, the `channels` that must differ,
and the measured CAM02-UCS distance `de`, taken over normal vision and the
three color-vision deficiencies at full severity.

Use `MotifSet.accepts(a, b, mode)` with motif keys. It returns true only when
their roles match a record for that mode, in either order, and they differ
in every listed channel. For a pair below the `fault_de` threshold,
`eyepiece.testing.motif_faults` drops the `cvd` color-distance finding only
when this check accepts it. Other checks, including dimmed-color findings,
still run. The recorded `de` is information, not a threshold used by the check.

Keep the reference and negative roles distinct in paper mode with different
markers. Give them the same marker and the `cvd` finding returns, even though
the record stays in the set.

```{code-cell} python
hwostyle.use("paper")
reference = ep.motifs.Motif("ref", "reference", role="reference", marker="o", linestyle="none")
pair = ep.motifs.MotifSet.default().only("negative").add(reference)
same_marker = pair.derive("ref", marker="s")
fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.6), layout="constrained")
for ax, name, subset in zip(
    axes, ("different markers", "same marker"), (pair, same_marker), strict=True
):
    cvd = [f for f in ep.testing.motif_faults(subset, mode="paper") if f.kind == "cvd"]
    print(
        name, "accepted:", subset.accepts("ref", "negative", "paper"), "cvd:", len(cvd)
    )
    for y, motif in enumerate(subset.values()):
        ax.plot([0.15], [y], **motif.marker_kw(size=9))
        ax.text(0.3, y, motif.key, va="center", **motif.label_kw())
    ax.set(title=name, xlim=(0, 1), ylim=(-0.5, 1.5))
    ax.axis("off")
plt.show()
```

Pass your own records through `ep.motifs.MotifSet(..., accepted=...)` for a custom set.
Import `AcceptedPair` from `eyepiece.motifs` and supply its roles, modes,
channels and measured `de`. A record is a permission per mode, never a
verdict: the channel check is what accepts the pair.
