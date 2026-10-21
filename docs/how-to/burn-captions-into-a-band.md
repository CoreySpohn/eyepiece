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

# Burn captions into a band

Reserve a strip below the axes before drawing the first frame. Keep one text
artist in that strip and update it with `ep.narration.caption_at(cues, t)`. Caption
starts are included and ends are excluded, so the same call clears gaps.

This small example uses explicit cue times. For a narrated clip, obtain them
with `cues = ep.narration.captions(tl.narration)` after solving the timeline.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
from IPython.display import HTML

import eyepiece as ep

hwostyle.use("dark")
fig = plt.figure(figsize=(5, 3), dpi=80)
ax = fig.add_axes([0.16, 0.40, 0.78, 0.50])
ax.plot([0, 1, 2], [0, 1, 0], color=ep.style.neutral(0.85))
ax.set(xlim=(0, 2), ylim=(0, 1.1), xticks=[0, 1, 2], yticks=[0, 1])
band = fig.add_axes([0, 0, 1, 0.25], facecolor=ep.style.neutral(0.08))
band.set(xticks=[], yticks=[])
for spine in band.spines.values():
    spine.set_visible(False)
caption = band.text(0.5, 0.5, "", ha="center", va="center", fontsize=17,
                    color=ep.style.neutral(0.95))
cues = [(0.0, 1.0, "The profile rises."), (1.2, 2.0, "Then it falls.")]


def draw(fig, k):
    caption.set_text(ep.narration.caption_at(cues, k / 10))


plt.close(fig)
anim = ep.render.animate(fig, draw, 20, fps=10)
HTML(anim.jshtml(dpi=80))
```

The text becomes part of each recorded frame, so an exported movie displays
it without a caption track. See {doc}`../tutorials/narrated` for cues loaded
from committed word timings and a readability check of the complete clip.
