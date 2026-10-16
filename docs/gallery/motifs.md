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

# Motifs

A motif is one recurring object (the star, a planet, a detector) drawn the same way
in every figure. `MotifSet.default()` is the vocabulary hwostyle ships. The sheet
below draws every motif with its own `line_kw`, `marker_kw` or `patch_kw`, so what
you see is what a figure gets, and switching the mode changes only the color.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import eyepiece as ep

plt.rcParams["savefig.dpi"] = 100
motifs = ep.MotifSet.default()


def draw_sample(ax, motif, y):
    """Draw one motif as it is meant to be drawn, at height y."""
    if motif.hatch:
        ax.add_patch(
            mpatches.Rectangle((0.0, y - 0.3), 1.0, 0.6, **motif.patch_kw())
        )
    elif motif.arrowstyle:
        ax.annotate(
            "",
            xy=(1.0, y),
            xytext=(0.0, y),
            arrowprops={
                "arrowstyle": motif.arrowstyle,
                "color": motif.color,
                "lw": motif.linewidth,
            },
        )
    elif motif.marker:
        if motif.linestyle != "none":
            ax.plot([0.0, 1.0], [y, y], **motif.line_kw())
        ax.plot([0.5], [y], **motif.marker_kw(size=9))
    elif motif.fill_alpha:
        ax.add_patch(
            mpatches.Rectangle((0.0, y - 0.3), 1.0, 0.6, **motif.patch_kw())
        )
    else:
        ax.plot([0.0, 1.0], [y, y], **motif.line_kw())


def motif_sheet(mode, motifs):
    """One mode's sheet: a sample and the noun for every motif."""
    hwostyle.use(mode)
    fig, ax = plt.subplots(
        figsize=(4.6, 0.3 * len(motifs) + 0.5), layout="constrained"
    )
    for i, motif in enumerate(motifs.values()):
        y = len(motifs) - i
        draw_sample(ax, motif, y)
        color = motif.color if motif.is_hue else plt.rcParams["text.color"]
        ax.text(1.15, y, motif.noun, va="center", color=color, fontstyle=motif.fontstyle or "normal")
    ax.set_xlim(-0.1, 3.0)
    ax.set_ylim(0.3, len(motifs) + 0.7)
    ax.axis("off")
    ax.set_title(f"{mode} mode", loc="left")
    return fig
```

## Light, dark and paper

```{code-cell} python
fig = motif_sheet("light", motifs)
```

```{code-cell} python
fig = motif_sheet("dark", motifs)
```

```{code-cell} python
fig = motif_sheet("paper", motifs)
```

## Under color-vision deficiency

Each row shows the hue motifs as a viewer with normal vision and with each
deficiency sees them (full severity). Hues that look alike here must differ in
marker, line style or hatch, which is what `signature()` records and
`eyepiece.testing.motif_faults` checks.

```{code-cell} python
def vision_rows(mode, motifs):
    """Hue swatches as seen by normal vision and three deficiencies."""
    from hwostyle.colors import simulate_cvd

    hwostyle.use(mode)
    hues = [m for m in motifs.values() if m.is_hue]
    kinds = [None, "deuteranopia", "protanopia", "tritanopia"]
    fig, ax = plt.subplots(figsize=(0.45 * len(hues) + 1.6, 1.6), layout="constrained")
    for r, kind in enumerate(kinds):
        for c, motif in enumerate(hues):
            color = motif.color if kind is None else simulate_cvd(motif.color, kind)
            ax.add_patch(mpatches.Rectangle((c, -r), 0.9, 0.8, color=color))
        ax.text(-0.2, -r + 0.4, kind or "normal", ha="right", va="center")
    for c, motif in enumerate(hues):
        ax.text(c + 0.45, 1.0, motif.key, rotation=60, ha="left", va="bottom", fontsize=7)
    ax.set_xlim(-3.0, len(hues) + 1)
    ax.set_ylim(-3.2, 2.2)
    ax.axis("off")
    return fig


fig = vision_rows("dark", motifs)
```

```{code-cell} python
fig = vision_rows("light", motifs)
```

## Checking the set a clip uses

A clip draws with a few motifs, so check that subset. The check reports the hue
count, the closest pair under every vision type, and how close the pair falls
once the set is dimmed to the mode's level (0.35 in dark mode, 0.5 in light and
paper).

```{code-cell} python
import eyepiece.testing as ept

hwostyle.use("dark")
clip = motifs.only("star", "planet", "disk", "model")
print(ept.motif_margins(clip, mode="dark"))
for fault in ept.motif_faults(clip, mode="dark"):
    print(fault.kind, fault.severity, fault.keys, fault.detail)
```
