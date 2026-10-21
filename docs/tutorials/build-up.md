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

# Tutorial 2: a silent build-up

Build an eight-second clip that reveals a synthetic coronagraph field and its
central row profile. Start with the finished figure, give each span a section
label, then let a plan set every frame.

## Make the finished figure

Create the image and profile together so that the profile samples the image
shown beside it. Set the limits and layout before creating the moves.

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece as ep

hwostyle.use("dark")
plt.rcParams.update({
    "font.size": 17, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.labelsize": 17, "axes.titlesize": 18, "savefig.dpi": 80,
})
n = 64
coordinate = np.linspace(-12.0, 12.0, n)
x, y = np.meshgrid(coordinate, coordinate)
radius = np.hypot(x, y)
angle = np.arctan2(y, x)
halo = 1e-7 * (1.0 + radius) ** -3 * (1.0 + 0.8 * np.cos(12 * angle) ** 2)
planet = 3e-9 * np.exp(-0.5 * (np.hypot(x + 6.0, y - 5.0) / 0.6) ** 2)
field = halo + planet
profile = field[n // 2] / field[n // 2].max()

fig, (image_ax, profile_ax) = plt.subplots(
    1, 2, figsize=(7.5, 4.2), dpi=80, layout="constrained"
)
result = ep.plot.imshow_log(
    field, ax=image_ax, extent=ep.plot.extent_lod_from_pixels(n, 24.0 / (n - 1)),
    vmin=1e-12, vmax=1e-7, floor=1e-12, colorbar=False,
)
image_ax.set(xlabel="x [lambda/D]", ylabel="y [lambda/D]", title="Field",
             xticks=[-10, 0, 10], yticks=[-10, 0, 10])
(curve,) = profile_ax.plot(coordinate, profile, color=ep.style.neutral(0.85), lw=2)
(marker,) = profile_ax.plot([1.5], [np.interp(1.5, coordinate, profile)],
                            "o", color=hwostyle.palette[0], ms=7)
profile_ax.set(xlim=(-12, 12), ylim=(0, 1.1), xticks=[-10, 0, 10],
               yticks=[0, 1], xlabel="x [lambda/D]", ylabel="relative intensity",
               title="Row profile")
fig.canvas.draw()
fig.set_layout_engine("none")
labels = ep.motion.Stage({"labels": [ax.xaxis.label for ax in (image_ax, profile_ax)]
                             + [ax.yaxis.label for ax in (image_ax, profile_ax)]})
builds = [ep.motion.build_axes(ax, label=False, title=True)
          for ax in (image_ax, profile_ax)]
reveal = ep.motion.draw_on(curve)
emphasis = ep.motion.indicate(marker, scale=1.5)
plt.close(fig)
```

## Name the spans

Use one section for each span so that review frames have useful labels.
The emphasis run writes an indication knob from zero to one, which makes `indicate`
grow and return to rest.

```{code-cell} python
tl = ep.clock.Timeline({"panel": 0.0, "image": 0.0, "profile": 0.0,
                  "indicate.marker": 0.0}, tempo=ep.clock.Tempo(factor=1.0))
tl.section("Panel")
tl.run("panel_in", 1.5, to={"panel": 1.0})
tl.section("Image")
tl.run("image_in", 1.5, to={"image": 1.0})
tl.section("Profile")
tl.run("profile_on", 2.0, to={"profile": 1.0})
tl.section("Marker")
tl.run("emphasize", 1.0, to={"indicate.marker": 1.0})
tl.section("Hold")
tl.hold(2.0, name="settled")
plan = tl.at_fps(10)
print(f"{plan.n_frames} frames at {plan.fps:g} fps")
```

## Draw from the plan

Fade the axis labels with Stage while `build_axes` draws the spines, ticks,
and titles. Set absolute values in `draw(k)` so that the checker can revisit any frame.
Adapt that function to the two-argument animation callback and embed the clip
at 80 dpi.

```{code-cell} python
def draw(k):
    labels.weigh({"labels": plan.value(k, "panel")})
    for build in builds:
        build.update(plan.value(k, "panel"))
    result.artists["image"].set_alpha(plan.value(k, "image"))
    reveal.update(plan.value(k, "profile"))
    marker.set_alpha(plan.value(k, "profile"))
    emphasis.update(plan.value(k, "indicate.marker"))


def draw_frame(fig, k):
    draw(k)


anim = ep.render.animate(fig, draw_frame, plan.n_frames, fps=plan.fps)
HTML(anim.jshtml(dpi=80))
```

## Check the clip

Readability checks text size, overlap, clipping, and text on marks, while
motion checks every frame for jumps, dead moves, drift, torn groups,
collisions, departures, spikes, and invalid shape changes under the supplied
declarations. The same call also reports timing faults and unread knobs from the
plan, compares declared artist meanings when you supply them, and measures
narration rate when the plan has a narration.

See the {doc}`checks catalog <../gallery/motion-checks>` for declarations and
reports, and the {doc}`clock catalog <../gallery/motion-clock>` for other spans.
This silent clip has no narration or declared meaning keys.

```{code-cell} python
report = ep.testing.assert_clip(fig, draw, plan, moves={curve: "profile_on"})
print(report)
```
