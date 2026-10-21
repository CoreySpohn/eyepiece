---
mystnb:
  render_markdown_format: myst
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

# Eyepiece and Manim

Eyepiece's Matplotlib moves cover a small vocabulary of reveals, emphasis,
and equation changes. Each move sets an absolute state from progress; the
caller owns the figure and the frame loop. A `Timeline` owns the named spans,
cues, and conversion from seconds to frames. It supplies knobs to a draw
function and never discovers or animates artists itself. See {doc}`clock`
for that separation and {doc}`rendering-in-manim` for rendering prepared views
as native Manim objects in a consumer-owned scene.

The table compares purposes, with limits in the last column. Every eyepiece
call it lists runs below on a small Agg figure at progress 0, 0.5, and 1.
These are checks of eyepiece behavior, not pixel comparisons with Manim.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from matplotlib.backends.backend_agg import FigureCanvasAgg

import eyepiece as ep

hwostyle.use("dark")


def small_figure():
    fig, ax = plt.subplots(figsize=(3.2, 1.8), dpi=80)
    FigureCanvasAgg(fig)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, ax


def sample(fig, update):
    for u in (0.0, 0.5, 1.0):
        update(u)
        fig.canvas.draw()
        assert np.asarray(fig.canvas.buffer_rgba()).shape == (144, 256, 4)


# FadeIn: the caller's artist stays on its axes; Stage owns its opacity.
fig, ax = small_figure()
(line,) = ax.plot([0.1, 0.9], [0.3, 0.7], color=ep.style.neutral(0.8))
original_alpha = line.get_alpha()
stage = ep.motion.Stage({"mark": line})
sample(fig, lambda u: stage.weigh({"mark": u}, ease="smoothstep"))
assert line.get_alpha() == original_alpha
plt.close(fig)

# Create: a temporary line draws the cut path, then the source returns.
fig, ax = small_figure()
(line,) = ax.plot([0.1, 0.5, 0.9], [0.2, 0.8, 0.4], color=ep.style.neutral(0.8))
original = np.column_stack(line.get_data()).copy()
reveal = ep.motion.draw_on(line, ease="linear")
sample(fig, reveal.update)
assert line.get_visible()
assert np.array_equal(np.column_stack(line.get_data()), original)
plt.close(fig)

# Indicate: size and tint return to the caller's original style.
fig, ax = small_figure()
(line,) = ax.plot([0.2, 0.8], [0.5, 0.5], color=ep.style.neutral(0.8), lw=2)
original_color = line.get_color()
pulse = ep.motion.indicate(line)
sample(fig, pulse.update)
assert line.get_linewidth() == 2
assert line.get_color() == original_color
plt.close(fig)

# Circumscribe: the figure-level outline is absent at both endpoints.
fig, ax = small_figure()
note = ax.text(0.5, 0.5, "signal", ha="center", color=ep.style.neutral(0.8))
fig.canvas.draw()
original = np.asarray(fig.canvas.buffer_rgba()).copy()
ring = ep.motion.circumscribe(note, color=ep.style.neutral(0.8))
sample(fig, ring.update)
assert np.array_equal(np.asarray(fig.canvas.buffer_rgba()), original)
plt.close(fig)

# Write: the ghost traces glyphs and hands back the original Text.
fig, ax = small_figure()
note = ax.text(0.5, 0.5, "signal", ha="center", color=ep.style.neutral(0.8))
fig.canvas.draw()
original = np.asarray(fig.canvas.buffer_rgba()).copy()
scribe = ep.motion.write(note)
sample(fig, scribe.update)
assert note.get_visible()
assert np.array_equal(np.asarray(fig.canvas.buffer_rgba()), original)
plt.close(fig)

# Transform: only an equation replacement is claimed here.
fig, ax = small_figure()
a = ep.motion.equation(ax, r"${{term:x}}$", fontsize=20, colors={"term": ep.style.neutral(0.8)})
b = ep.motion.equation(ax, r"${{term:y}}$", fontsize=20, colors={"term": ep.style.neutral(0.8)})
change = ep.motion.transform(a, b, replace="morph")
sample(fig, change.update)
assert not any(g.get_visible() for g in a.artists["glyphs"])
assert all(g.get_visible() for g in b.artists["glyphs"])
plt.close(fig)

# TransformMatchingTex: named groups constrain glyph-identity matching.
fig, ax = small_figure()
a = ep.motion.equation(ax, r"${{x:x}}+{{y:y}}$", fontsize=20, colors={"x": ep.style.neutral(0.8), "y": ep.style.neutral(0.8)})
b = ep.motion.equation(ax, r"${{y:y}}+{{x:x}}$", fontsize=20, colors={"x": ep.style.neutral(0.8), "y": ep.style.neutral(0.8)})
change = ep.motion.transform(a, b, arc=0.4)
sample(fig, change.update)
assert change.pairs
assert not any(g.get_visible() for g in a.artists["glyphs"])
assert all(g.get_visible() for g in b.artists["glyphs"])
plt.close(fig)

# Rate functions: shapes are explicit; a custom callable covers other curves.
curves = ("linear", "smoothstep", "smootherstep", "cosine", "back_out", "running_start")
for kind in curves:
    fig, ax = small_figure()
    (line,) = ax.plot([0.1, 0.9], [0.3, 0.7], color=ep.style.neutral(0.8))
    reveal = ep.motion.draw_on(line, ease=kind)
    sample(fig, reveal.update)
    assert ep.clock.ease(0.0, kind=kind) == 0.0
    assert np.isfinite(ep.clock.ease(0.5, kind=kind))
    assert ep.clock.ease(1.0, kind=kind) == 1.0
    plt.close(fig)

fig, ax = small_figure()
(line,) = ax.plot([0.1, 0.9], [0.3, 0.7], color=ep.style.neutral(0.8))
reveal = ep.motion.draw_on(line, ease=lambda u: u**2)
sample(fig, reveal.update)
assert line.get_visible()
plt.close(fig)

rows = [
    (
        "FadeIn",
        '`stage.weigh({"mark": u}, ease="smoothstep")`',
        "Opacity only, no travel. The caller owns the artist; Stage owns its "
        "color and opacity. The clock supplies presence.",
    ),
    (
        "Create",
        '`ep.motion.draw_on(line, ease="linear")`',
        "Line2D paths only, no general mobject creation. The caller's source "
        "is hidden behind a temporary cut line until the end; the clock supplies progress.",
    ),
    (
        "Indicate",
        "`ep.motion.indicate(line)`",
        "Pulses width, marker or font size and tint, rather than scaling arbitrary "
        "geometry. Borrows the artist and returns its style; the clock supplies pulse progress.",
    ),
    (
        "Circumscribe",
        '`ep.motion.circumscribe(note, color=ep.style.neutral(0.8))`',
        "Draws and fades a figure-owned outline around the borrowed target's "
        "display box. The clock supplies ring progress; no target geometry moves.",
    ),
    (
        "Write",
        "`ep.motion.write(note)`",
        "Traces Matplotlib Text glyphs with a figure-owned ghost, then restores "
        "the caller's Text. No general mobject writing; the clock supplies progress.",
    ),
    (
        "Transform",
        '`ep.motion.transform(a, b, replace="morph")`',
        "Morphs replaced equation outlines, moves matches and fades unpaired "
        "glyphs. Owns both equations' visibility, ending on b. The clock supplies "
        "progress; there is no arbitrary mobject Transform counterpart.",
    ),
    (
        "TransformMatchingTex",
        "`ep.motion.transform(a, b, arc=0.4)`",
        "Matches font and glyph identities within shared term tags, then untagged "
        "runs. Owns both equations' visibility; the clock supplies progress. Uses "
        "mathtext, not TeX token matching; no usetex counterpart.",
    ),
    (
        "rate functions (smooth, linear, and the named ease family)",
        "`ep.clock.ease(u, kind=kind)` and `ep.motion.draw_on(line, ease=...)`",
        "linear, smoothstep, smootherstep, cosine, back_out and running_start "
        "are available; smooth is not a named alias. No complete Manim rate-function "
        "catalog or aliases such as ease_in_sine. A callable such as u**2 gives another "
        "curve. The caller owns the line, draw_on its temporary copy. The clock "
        "owns frame timing; ease shapes progress and must be applied only once.",
    ),
]
header = "| Manim call | eyepiece call | the differences |\n|---|---|---|\n"
display(Markdown(header + "\n".join("| " + " | ".join(row) + " |" for row in rows)))
```

The table's `u` is uneased progress. A move with its own `ease=` shapes it
internally. `Stage.weigh` shapes presence only, and clips it to a valid
opacity; `draw_on` clips an eased path fraction. Overshoot can therefore
shape a position without making opacity or a path fraction invalid.

The clock can place an entry and a reveal in sequence. Here it supplies the
presence and path-progress knobs for the same two calls checked above.

```{code-cell} python
fig, ax = small_figure()
(line,) = ax.plot([0.1, 0.9], [0.3, 0.7], color=ep.style.neutral(0.8))
(marker,) = ax.plot([0.1], [0.3], "o", color=ep.style.neutral(0.8))
original_alpha = marker.get_alpha()
stage = ep.motion.Stage({"mark": marker})
reveal = ep.motion.draw_on(line)
tl = ep.clock.Timeline({"path": 0.0})
tl.enter("mark", 0.5)
tl.move("reveal", 0.5, to={"path": 1.0}, ease="linear")
plan = tl.at_fps(6)
for k in range(plan.n_frames):
    stage.weigh({"mark": plan.value(k, "mark")})
    reveal.update(plan.value(k, "path"))
    fig.canvas.draw()
assert line.get_visible()
assert marker.get_alpha() == original_alpha
plt.close(fig)
```

The {doc}`attention catalog <../gallery/motion-attention>`,
{doc}`staging catalog <../gallery/motion-staging>`, and
{doc}`equation catalog <../gallery/motion-equations>` show these moves in
larger figures. A native Manim scene can use its own animation vocabulary;
Eyepiece does not provide a scene graph, scene scheduler, or a general
replacement for that vocabulary.
