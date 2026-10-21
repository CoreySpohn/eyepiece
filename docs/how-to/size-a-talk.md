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

# Size a figure for a talk

Use `talk_rc` before creating the figure and its artists. It sizes text and
strokes as fractions of the output height. Match `height_px` and `dpi` to
the actual canvas: here 360 pixels at 90 dpi gives a four inch high frame.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

hwostyle.use("dark")
height_px, dpi = 360, 90
with plt.rc_context(ep.talk_rc(height_px, dpi=dpi)):
    plt.rcParams["savefig.dpi"] = dpi
    fig, ax = plt.subplots(figsize=(6.4, height_px / dpi), dpi=dpi,
                           layout="constrained")
    x = np.linspace(0, 1, 100)
    ax.plot(x, 1 - np.exp(-4 * x))
    ax.set(xlabel="exposure", ylabel="fraction detected",
           title="More exposure, fewer new detections", xlim=(0, 1), ylim=(0, 1))
    print("Body fraction:", ep.TALK_FRACTIONS["body"])
    print("Body size (pt):", plt.rcParams["font.size"])
    print("Video cap style:", ep.VIDEO_RC["lines.solid_capstyle"])
    plt.show()
```

`talk_rc` includes `VIDEO_RC` so moving lines have butt caps and round
joins. The context restores rcParams afterwards. Entering it only around
the recording does not resize artists that already exist.
