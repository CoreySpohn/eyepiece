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

# Signature moves

These small examples show the moves in the
{doc}`primitive contract <../explanation/contract>` and the still each move
returns to. Progress `u` runs from 0 to 1. Strips sample that progress; they
do not assign equal times to separate moves. Only the passing flash below
is an animation.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece as ep

plt.rcParams.update({"figure.dpi": 90, "savefig.dpi": 90, "font.size": 11})


def board(figsize=(4.0, 2.1)):
    fig, ax = plt.subplots(figsize=figsize, dpi=90)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return fig, ax


def show_strip(fig, update, values, titles=None):
    frames = []
    for u in values:
        update(u)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
    # Display only the strip, not the working figure as well.
    plt.close(fig)
    height = 8.4 / len(frames) * frames[0].shape[0] / frames[0].shape[1]
    shown, cells = plt.subplots(
        1, len(frames), figsize=(8.4, height + 0.35),
        squeeze=False, layout="constrained",
    )
    labels = titles if titles is not None else [f"u = {u:g}" for u in values]
    for cell, frame, label in zip(cells[0], frames, labels, strict=True):
        cell.imshow(frame, interpolation="hanning")
        cell.set_title(label, fontsize=10)
        cell.set_axis_off()
    plt.show()
    plt.close(shown)
```

## Head draw with a backing line

Use `ep.motion.head_draw(..., behind=)` when a curve needs a halo that must stay
behind the bright drawing head.

```{code-cell} python
fig, ax = board()
x = np.linspace(0.08, 0.92, 120)
y = 0.25 + 0.5 * np.sin(np.pi * (x - 0.08) / 1.68)
(halo,) = ax.plot(x, y, color="0.8", lw=9, solid_capstyle="round")
(curve,) = ax.plot(x, y, color="C0", lw=2.5)
reveal = ep.motion.head_draw(curve, behind={curve: halo}, ease="linear")
show_strip(fig, reveal.update, (0.25, 0.6, 1.0))
```

The halo follows the drawn body, and the final frame is the original curve
and halo with no bright head.

## Ruler and counter read the head

Use `ruler_readout` when a measurement and a number must agree with the
drawn tip on every frame.

```{code-cell} python
fig, ax = board()
x = np.linspace(0.1, 0.9, 100)
(curve,) = ax.plot(x, 0.25 + 0.5 * x**2, color="C0", lw=2)
count = ep.motion.readout(ep.motion.Anchor.from_data(ax, (0.12, 0.82)), 0.0,
                   fmt=".2f", digits=3, fontsize=18)
rule = ep.plot.ruler(ax, (0.1, 0.14), (0.1, 0.14), "x = 0.10", head_scale=0)
head = ep.motion.head_draw(curve, ease="linear")
measure = ep.motion.ruler_readout(
    head, readout=count, value=lambda x, y: y,
    ruler=rule, ruler_to=lambda x, y: (x, 0.14),
    ruler_text=lambda x, y: f"x = {x:.2f}",
)
show_strip(fig, measure.update, (0.2, 0.6, 1.0))
```

The number reports `y` at the head while the ruler reports `x`; neither has
its own interpolation. Bind the knob to a plan as in
{doc}`../how-to/bind-a-move-to-a-knob`.

## A label arrives attached

Use `Stage.weigh(..., lift_pt=)` when a label should fade and settle into
place without moving the data mark it names.

```{code-cell} python
fig, ax = board()
(mark,) = ax.plot([0.42], [0.42], "o", color="C0", ms=8)
label = ax.annotate("planet b", (0.42, 0.42), xytext=(12, 14),
                    textcoords="offset points", color=mark.get_color(), fontsize=16)
stage = ep.motion.Stage({"label": label})


def arrive(u):
    stage.weigh({"label": u}, ease="smooth", lift_pt=6)


show_strip(fig, arrive, (0.15, 0.5, 1.0))
```

At 90 dpi the six-point lift is 7.5 pixels, within the ten-pixel limit.
Only the label belongs to the lifted group.

## Write a label and an equation

Use `write` or an equation's `reveal="write"` when a short phrase or formula
should arrive in reading order, with each glyph traced before it fills.

```{code-cell} python
fig, ax = board()
label = ax.text(0.08, 0.67, "planet signal", color="C0", fontsize=20)
formula = ep.motion.equation(ep.motion.Anchor.from_data(ax, (0.48, 0.32)),
                      r"$C = F_p / F_s$", fontsize=25)
fig.canvas.draw()
scribe = ep.motion.write(label)


def write_both(u):
    scribe.update(u)
    formula.update(u, reveal="write")


show_strip(fig, write_both, (0.2, 0.55, 1.0))
```

Both end as the original text with no pen left over.

## A passing flash

Use `ep.motion.circumscribe(style="flash")` to point out a subject once and leave its
appearance unchanged.

```{code-cell} python
fig, ax = board()
label = ax.text(0.5, 0.5, "the answer", ha="center", va="center", fontsize=22)
flash = ep.motion.circumscribe(label, style="flash", time_width=0.25, pad_pt=8, lw=2.5)


def draw_flash(fig, k):
    flash.update(k / 29)


movie = HTML(ep.render.animate(fig, draw_flash, 30, fps=15).jshtml(dpi=90))
plt.close(fig)
movie
```

The tapered comet runs clockwise from the top and disappears at the end.

## Hold an emphasis at its peak

Use `ep.motion.indicate(shape="plateau")` when the viewer needs time to read an
emphasis before it returns to the original appearance.

```{code-cell} python
fig, ax = board()
(line,) = ax.plot([0.15, 0.85], [0.5, 0.5], "o-", color="C0", lw=2, ms=6)
pulse = ep.motion.indicate(line, shape="plateau", scale=1.8)
show_strip(fig, pulse.update, (0.0, 0.35, 0.65, 1.0))
```

The middle two snapshots hold the same peak; positions stay fixed.

## Land an answer with a click

Use `click` when the geometry, an end dot and a number should settle
together into an answer the viewer can read.

```{code-cell} python
fig, ax = board()
(line,) = ax.plot([0.12, 0.85], [0.35, 0.6], color="C0", lw=2)
count = ep.motion.readout(ep.motion.Anchor.from_data(ax, (0.16, 0.78)), 42,
                   fmt="d", digits=2, fontsize=20)
rule = ep.plot.ruler(ax, (0.12, 0.18), (0.85, 0.18), "measured", head_scale=0)
land = ep.motion.click(line, dot=ep.motion.Anchor.from_data(ax, (0.85, 0.6)),
                readout=count, ruler=rule)
tl = ep.clock.Timeline()
beat = tl.click("answer")
plan = tl.at_fps(10)
show_strip(fig, land.update, (0.0, 0.5, 1.0),
           ("before", "landing", "landed and held"))
print(f"Beat: {float(beat.end - beat.start):.1f} s; "
      f"clip including hold: {plan.n_frames / plan.fps:.1f} s")
```

`Timeline.click` reserves 0.6 seconds for the beat and at least 1.5 seconds
for the still answer, independent of tempo. In a frame loop, update `land`
with `plan.value(k, "click.answer")` after the moves it settles.

## Grow an arrow at one shape

Use `ep.motion.arrow(grow="scaled")` when a vector should grow from its tail with
its stroke and head scaled together.

```{code-cell} python
fig, ax = board()
vector = ep.motion.arrow((0.15, 0.25), (0.85, 0.75), ax=ax, grow="scaled",
                   color="C0", lw=3, head_size=1.6)
show_strip(fig, vector.update, (0.2, 0.55, 1.0))
```

## Build axes in small events

Use `ep.motion.build_axes(style="staccato")` when the viewer should read the spine,
ticks, numbers and axis label as separate events.

```{code-cell} python
fig, ax = plt.subplots(figsize=(3.6, 1.35), dpi=90)
fig.subplots_adjust(left=0.1, bottom=0.5, right=0.94, top=0.9)
ax.set(xlim=(0, 3), ylim=(0, 1), xticks=[0, 1, 2, 3], yticks=[],
       xlabel="exposure [s]")
ax.tick_params(axis="x", labelsize=13, width=1.5, length=5)
ax.xaxis.label.set_fontsize(14)
ax.spines["bottom"].set_linewidth(1.5)
for name in ("left", "right", "top"):
    ax.spines[name].set_visible(False)
fig.canvas.draw()
build = ep.motion.build_axes(ax, axis="x", style="staccato")
show_strip(fig, build.update, (0.2, 0.55, 0.8, 1.0))
```

The limits stay fixed, and the last frame is the axes as made.

## Drift unmatched equation glyphs

Use `ep.motion.transform(fade_toward=True)` when removed and added material should
fade toward each other while the matching equation terms travel.

```{code-cell} python
fig, ax = board()
at = ep.motion.Anchor.from_data(ax, (0.5, 0.5))
eq_a = ep.motion.equation(at, r"${{l:x}}+{{r:y}}-c$", fontsize=29)
eq_b = ep.motion.equation(at, r"${{l:x}}+{{r:y}}=d^2$", fontsize=29)
change = ep.motion.transform(eq_a, eq_b, fade_toward=True)
show_strip(fig, change.update, (0.0, 0.4, 0.7, 1.0))
```

The tags preserve `x` and `y`; the unmatched `-c` and `=d^2` drift and fade.

## Develop an image from photons

Use `photons` when an image should develop from seeded photon arrivals on
one fixed count scale.

```{code-cell} python
fig, ax = board(figsize=(3.0, 2.5))
yy, xx = np.mgrid[-4:5, -4:5]
rate = 0.2 + 3 * np.exp(-(xx**2 + yy**2) / 3)
exposure = np.linspace(0, 2, 9)
channel = ep.motion.arrivals(rate, exposure, seed=7)
mark = ep.motion.photons(channel, ax=ax, norm="linear", cmap="magma")
ax.set(xlim=(-0.5, 8.5), ylim=(-0.5, 8.5))
fig.colorbar(mark.artists["image"], ax=ax, label="counts", shrink=0.8)
show_strip(fig, mark.update, (0.0, 0.55, 1.0))
```

The first sample is empty; new arrivals glow and settle into the final
counts. Drive the knob linearly to keep the samples evenly spaced in time,
and declare `mark.steps(knob_values)` as `steps=` to `check_clip` for their
deliberate cuts.

## Hop through exchangeable draws

Use `hop` when equally likely curves should appear one at a time with
aging ghosts and a tally of what the set says.

```{code-cell} python
fig, (curves, tally) = plt.subplots(1, 2, figsize=(6.2, 2.6), dpi=90)
fig.subplots_adjust(left=0.1, right=0.96, bottom=0.22, top=0.76, wspace=0.65)
x = np.linspace(0, 1, 40)
slopes = np.random.default_rng(8).normal(0.5, 0.3, 12)
draws = np.stack([np.column_stack((x, slope * x)) for slope in slopes])
mark = ep.plot.hop(draws, ax=curves, rate=4, ghosts=3,
              tally=slopes > 0.5, tally_ax=tally, kind="fraction")
curves.set(xlabel="time", ylabel="position")
tally.set(xlabel="draws", ylabel="fraction above 0.5")
show_strip(fig, mark.update, (0.25, 1.0))
```

The tally uses only the draws shown so far, with a Wilson interval.
Play `mark.duration` seconds with a linear knob and pass
`mark.steps(knob_values)` to `check_clip` as `steps=` for the draw cuts.

## Resample weighted draws

Use `weighted_draws` when posterior or particle weights must remain
visible in marker area, including during seeded systematic resampling.

```{code-cell} python
fig, ax = plt.subplots(figsize=(3.6, 2.5), dpi=90)
fig.subplots_adjust(left=0.18, bottom=0.22, right=0.95, top=0.95)
points = np.random.default_rng(2).normal(size=(24, 2))
weights = np.exp(-0.5 * ((points[:, 0] - 0.6)**2 + points[:, 1]**2) / 0.5)
mark = ep.plot.weighted_draws(points, weights, ax=ax, max_size=160, resample=4)
ax.set(xlim=(-2.2, 2.4), ylim=(-2.6, 2.1), xlabel="parameter a", ylabel="parameter b")
show_strip(fig, mark.update, (0.0, 0.5, 1.0),
           ("weighted", "resampling", "offspring counts / n"))
```

Positions stay fixed; parents with no offspring vanish, while each surviving
parent shows the total weight of its offspring. For opacity rather than
area, pass `encode="alpha"`; that encoding keeps marker size fixed.
