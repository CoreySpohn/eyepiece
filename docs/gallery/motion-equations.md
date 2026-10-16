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

# Motion equations and morphs

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.colors import LogNorm

import eyepiece as ep

hwostyle.use("dark")
# Docs-build only, to keep the baked page images small.
plt.rcParams["savefig.dpi"] = 110

N = 96
PIXSCALE_LOD = 0.3
EXTENT = ep.extent_lod_from_pixels(N, PIXSCALE_LOD)
u = (np.arange(N) - (N - 1) / 2.0) * PIXSCALE_LOD
x, y = np.meshgrid(u, u)
r, theta = np.hypot(x, y), np.arctan2(y, x)

# Leftover starlight with twelve spokes, and a companion at 8 lambda/D.
HALO = 1e-7 * (1.0 + r) ** -3 * (1.0 + 0.8 * np.cos(12 * theta) ** 2)
PLANET_XY = (8.0 * np.cos(2.4), 8.0 * np.sin(2.4))
PLANET = 3e-9 * np.exp(-0.5 * (np.hypot(x - PLANET_XY[0], y - PLANET_XY[1]) / 0.6) ** 2)
FIELD = HALO + PLANET
NORM = LogNorm(1e-12, 1e-7)
```

## A morph

`quad_image` draws an image as one quadrilateral per pixel, filled with the
pixel's own color, and `update` moves every pixel at once. The corners come
from `pixel_quads`, here moved by an unrolling map: each ring around the star
becomes a vertical column, separation along the bottom and angle up the side,
so the leftover light of every ring can be compared on one axis.

```{code-cell} python
half = PIXSCALE_LOD / 2.0
edges = np.concatenate([u - half, [u[-1] + half]])
ex, ey = np.meshgrid(edges, edges)
corners = np.stack([ex, ey], axis=-1)
keep = r <= 13.0
quads = ep.pixel_quads(corners, keep)

# Angles of each pixel's corners taken on its own center's branch, so pixels
# straddling the cut at 180 degrees are not torn across the panel.
center_angle = theta[keep][:, None]
qr = np.hypot(quads[..., 0], quads[..., 1])
qt = center_angle + np.angle(np.exp(1j * (np.arctan2(quads[..., 1], quads[..., 0]) - center_angle)))


def unroll(t):
    """Rings opened into columns at x = r, angle up the side, at progress t."""
    t = np.clip(t, 0.0, 1.0)
    remaining = 1.0 - t
    phi = qt * remaining
    # sinc takes its limiting value at zero, including at the origin and rest.
    xs = qr * (1.0 - 0.5 * qt**2 * remaining * np.sinc(phi / (2.0 * np.pi))**2)
    ys = qt * np.sinc(phi / np.pi) * (qr * remaining + t * 13.0 / np.pi)
    return np.stack([xs, ys], axis=-1)


fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), layout="constrained")
for axis, t in zip(axes, (0.0, 0.55, 1.0), strict=True):
    res = ep.quad_image(FIELD, corners, ax=axis, keep=keep, norm=NORM)
    res.update(quads=unroll(t))
    axis.set_xlim(-14 if t < 1 else 0, 14)
    axis.set_ylim(-14, 14)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"unroll {t:.0%}", fontsize="small")
plt.show()
```

`brush` is how the image and its unrolled copy show that they hold the same
pixels without an arrow drawn between them. It veils both panels in their
background color and redraws the chosen pixels above the veil in each, so one
ring lights as a circle on the left and as a column on the right.

```{code-cell} python
fig, (left, right) = plt.subplots(1, 2, figsize=(6.0, 3.0), layout="constrained")
ep.imshow_log(FIELD, ax=left, extent=EXTENT, vmin=NORM.vmin, vmax=NORM.vmax,
              floor=NORM.vmin, colorbar=False)
unrolled = ep.quad_image(FIELD, corners, ax=right, keep=keep, norm=NORM)
unrolled.update(quads=unroll(1.0))
right.set_xlim(0, 13)
right.set_ylim(-13, 13)
right.set_xlabel(r"separation [$\lambda/D$]")
right.set_ylabel("angle around the star")
right.set_yticks([])
ep.label_lod(left)

linked = ep.brush([left, right], [quads, unroll(1.0)], FIELD[keep], norm=NORM)
linked.update(np.abs(r[keep] - 8.0) <= half)
plt.show()
```

## Morph: one path becomes another

`morph` carries one path into another so the viewer can follow the shape
rather than see one vanish and another appear. Both paths, a `Line2D`, a
`PathPatch`, or an array of vertices, are resampled to the same number of
points evenly spaced in arc length, and a ghost blends the two samplings
under the same easing as every other move, along with stroke color, fill
color, opacity, and line width. For closed paths the start point and the
direction of the second are chosen so that corresponding points are as near
as possible, so a ring that is merely drawn the other way round does not
turn inside out on the way. Several subpaths are paired by size, and a
subpath with no partner shrinks to a point or grows from one. As with
`flight`, the ghost shows only between the ends, so the first and second
path are the frames at either end.

```{code-cell} python
fig, ax = plt.subplots(figsize=(8.0, 2.6))
ax.set(xlim=(-1.6, 1.6), ylim=(-1.2, 1.2), xticks=[], yticks=[])
ax.set_aspect("equal")
fig.canvas.draw()
fig.set_layout_engine("none")

phi = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
ring = np.c_[np.cos(phi), np.sin(phi)]
box = np.array([[0.9, 0.9], [-0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]])
turn = ep.morph(
    ring,
    box[::-1],
    ax=ax,
    closed=True,
    a_kw={"edgecolor": "C0", "facecolor": "none"},
    b_kw={"edgecolor": "C1", "facecolor": "C1", "alpha": 0.4},
)

shown, cells = plt.subplots(1, 5, figsize=(9.0, 1.9), layout="constrained")
for cell, u in zip(cells, (0.1, 0.3, 0.5, 0.7, 0.9), strict=True):
    turn.update(u)
    fig.canvas.draw()
    cell.imshow(np.asarray(fig.canvas.buffer_rgba()).copy(), interpolation="nearest")
    cell.set_title(f"u = {u:g}")
    cell.set_axis_off()
plt.show()
```

The ring is drawn counterclockwise and the box clockwise, and the frames show
the ring growing into the box without a twist. The ghost is hidden at
``u = 0`` and ``u = 1``, fading in over the first fifth of the move and out
over the last fifth. In a clip the caller shows the ring before the move
and the box after it.

## Equations: name the terms, then transform

`term_row` sets one text per term, which suits a readout whose text changes
while its structure stays. `equation` is for math that has structure, and for
a derivation that rearranges. It reads the glyph outlines and positions that
mathtext computes, and a marker ``{{name:...}}`` in the string groups the
glyphs of one piece into a term. A term is an artist with the box of its
glyphs, so `callout`, `indicate`, `circumscribe`, `Stage` and
`Anchor.from_artist` take it as they take any artist. The groups are found
by difference (the string is parsed again with the group hidden by
``\phantom``), so fractions, scripts and rules work, and a layout that the
hidden group would change raises instead of drawing a wrong group.

`transform` carries one equation into another. The second equation is placed
so that the glyphs most groups agree on keep their place, so unchanged material
(the left side of Bayes' rule) does not slide; `align` names another tag, or
None to leave both on their own anchors. A glyph that is the same character in
both moves from its place in the first to its place in the second, and it moves
with its group: a tag, or a run of neighbours in no tag, takes one eased
progress, so a word travels whole, and the groups start left to right. A tag
that both equations have but whose glyphs do not match, such as ``num`` becoming
another ``num``, morphs: the outlines of the old glyphs turn into the outlines
of the new ones, as ManimCE's `Transform` does (the next clip and the
comparison below show it). Glyphs with
no counterpart at all fade out where they stand, in the first part of the beat,
or fade in where they will stand, in the last part, after everything has
settled. A tagged glyph pairs only inside its tag, and `key_map` pairs two
differently named tags. The bar of a fraction stretches as a rectangle with
square ends, and a group arcs on `arc` (up going right, down going left) only as far as it
travels sideways. At ``u = 1`` the figure shows the second equation, as every
move rests, and ``update(0)`` shows the first. The first clip turns Bayes' rule
into the numbers of a worked example: the left side stays, the bar stretches,
the numerator and denominator become their numbers, and the result arrives last.

```{code-cell} python
import eyepiece.testing as ept

BAYES_A = (
    r"${{lhs:\mathrm{Pr}(H|E)=}}\frac{{num:\mathrm{Pr}(E|H)\,\mathrm{Pr}(H)}}"
    r"{{den:\mathrm{Pr}(E)}}$"
)
BAYES_B = (
    r"${{lhs:\mathrm{Pr}(H|E)=}}\frac{{num:0.24}}{{den:0.24+0.07}}{{res:=0.77}}$"
)
TERM_COLORS = {"lhs": "C0", "num": "C1", "den": "C3", "res": "C2"}

fig = plt.figure(figsize=(7.6, 2.4))
board = fig.add_axes([0, 0, 1, 1])
board.set(xlim=(0, 1), ylim=(0, 1))
board.axis("off")
left = ep.Anchor.from_data(board, (0.06, 0.5))
bayes_a = ep.equation(left, BAYES_A, fontsize=34, colors=TERM_COLORS, ha="left")
bayes_b = ep.equation(left, BAYES_B, fontsize=34, colors=TERM_COLORS, ha="left")
bayes = ep.transform(bayes_a, bayes_b, stagger=0.55)

# A beat of A, the move, a beat of B; the move's progress is uneased.
N_BAYES = 60


def draw_bayes(fig, k):
    bayes.update(min(1.0, max(0.0, (k - 12) / 36)))


for k in (0, N_BAYES - 1):
    draw_bayes(fig, k)
    ept.assert_readable(fig)
HTML(ep.animate(fig, draw_bayes, N_BAYES, fps=20).jshtml(dpi=80))
```

A replaced tag changes shape by default. The outlines of the old glyphs turn
into the outlines of the new ones, aligned as ManimCE's `Transform` aligns
paths: glyph for glyph in reading order, the side with fewer glyphs gaining
stacked copies of its own, a hole that one glyph lacks growing from a point, and
every outline split until both have the same curves. ``replace="fade"`` is the
option that keeps the older behavior: the old glyphs ride a box that moves and
scales onto the box of the new ones, fading out as the new ones fade in inside
it. Below, the same Bayes move twice, the fade first and the morph second. The
thirteen glyphs of the numerator outnumber the four of ``0.24``, so several
outlines gather on each new glyph and the middle of the morph is crowded, as it
is in Manim.

```{code-cell} python
from IPython.display import display


def bayes_move(replace):
    fig = plt.figure(figsize=(7.6, 2.4))
    board = fig.add_axes([0, 0, 1, 1])
    board.set(xlim=(0, 1), ylim=(0, 1))
    board.axis("off")
    left = ep.Anchor.from_data(board, (0.06, 0.5))
    eq_a = ep.equation(left, BAYES_A, fontsize=34, colors=TERM_COLORS, ha="left")
    eq_b = ep.equation(left, BAYES_B, fontsize=34, colors=TERM_COLORS, ha="left")
    return fig, ep.transform(eq_a, eq_b, stagger=0.55, replace=replace)


for replace in ("fade", "morph"):
    fig, move = bayes_move(replace)

    def draw(fig, k, move=move):
        move.update(min(1.0, max(0.0, (k - 12) / 36)))

    display(HTML(ep.animate(fig, draw, N_BAYES, fps=20).jshtml(dpi=80)))
```

Most rearrangements keep most glyphs, and then the move is mostly travel.
Below, ``a^2 + b^2 = c^2`` becomes ``a^2 = c^2 - b^2``: every ``a``, ``c`` and
square stays a glyph, ``a^2`` stays where it is, ``b^2`` crosses the equals sign
rising on an arc while ``c^2`` dips under it, and only the plus and the minus fade. Without the tag on ``b^2`` the squares would
pair by their order and the wrong ``2`` would travel, which is the reason to
tag a piece that must stay together; a term of either equation is also an
anchor, so a pulse can name the piece that moves.

```{code-cell} python
fig = plt.figure(figsize=(6.4, 2.0))
board = fig.add_axes([0, 0, 1, 1])
board.set(xlim=(0, 1), ylim=(0, 1))
board.axis("off")
middle = ep.Anchor.from_data(board, (0.5, 0.5))
pyth_a = ep.equation(middle, r"$a^2+{{b:b^2}}=c^2$", fontsize=40, colors={"b": "C1"})
pyth_b = ep.equation(middle, r"$a^2=c^2-{{b:b^2}}$", fontsize=40, colors={"b": "C1"})
pyth = ep.transform(pyth_a, pyth_b, stagger=0.7, arc=0.7)
N_PYTH = 60


def draw_pyth(fig, k):
    pyth.update(min(1.0, max(0.0, (k - 12) / 36)))


for k in (0, N_PYTH - 1):
    draw_pyth(fig, k)
    ept.assert_readable(fig)
HTML(ep.animate(fig, draw_pyth, N_PYTH, fps=20).jshtml(dpi=80))
```

Unsupported in this version: ``usetex``, several math spans in one string, and
the size change of ``\left ... \right``. Escaped braces raise.
