"""Continuity regressions for the moves demonstrated in the motion gallery."""

import ast
import re
from pathlib import Path

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _gallery_cell(marker):
    page = Path(__file__).resolve().parents[1] / "docs/gallery/motion.md"
    cells = re.findall(r"```\{code-cell\} python\n(.*?)```", page.read_text(), re.S)
    return next(code for code in cells if marker in code)


def test_collapse_copy_is_continuous():
    hwostyle.use("dark")
    ns = {"np": np, "plt": plt, "ep": ep}
    exec(_gallery_cell("gather = ep.collapse("), ns)
    ept.assert_motion(ns["fig"], lambda t: ns["gather"].update(t / 2), ept.Span(0, 2))
    ept.assert_motion(
        ns["fig"], lambda t: ns["gather"].update(1, carry=t / 2), ept.Span(0, 2)
    )
    without_landing = ep.collapse(
        ns["left"],
        ns["cells"],
        ns["row"],
        ns["start"],
        ns["end"],
        fig=ns["fig"],
        colors="C3",
    )
    ept.assert_motion(
        ns["fig"], lambda t: without_landing.update(t / 2), ept.Span(0, 2)
    )


def test_morph_ghost_fades_continuously_at_both_ends():
    fig, ax = plt.subplots(figsize=(4, 2.6), dpi=80)
    ax.set(xlim=(-1.6, 1.6), ylim=(-1.2, 1.2), xticks=[], yticks=[], aspect="equal")
    phi = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    ring = np.c_[np.cos(phi), np.sin(phi)]
    box = np.array([[0.9, 0.9], [-0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]])
    move = ep.morph(
        ring,
        box[::-1],
        ax=ax,
        closed=True,
        a_kw={"edgecolor": "C0", "facecolor": "none"},
        b_kw={"edgecolor": "C1", "facecolor": "C1", "alpha": 0.4},
    )
    ept.assert_motion(fig, lambda t: move.update(t / 2), ept.Span(0, 2))


def test_follow_zoom_halos_fade_continuously():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    ax.imshow(np.zeros((20, 20)), extent=(-6, 6, -4, 4), vmin=-1, vmax=1)
    ax.set(xlim=(-6, 6), ylim=(-4, 4), xticks=[], yticks=[])
    move = ep.follow_zoom(ax, (-2, -1), size=0.36)
    ept.assert_motion(fig, lambda t: move.update(t / 2), ept.Span(0, 2))


def test_gallery_emphasis_labels_stay_inside_frame():
    hwostyle.use("dark")
    ns = {"np": np, "plt": plt, "ep": ep}
    exec(_gallery_cell("ring = ep.circumscribe(orbiter"), ns)
    fig = ns["fig"]
    ept.assert_motion(fig, lambda t: ns["pulse"].update(t / 2), ept.Span(0, 2))


def test_gallery_quad_image_unroll_settles_continuously():
    grid = np.array([-8.0, 0.0, 8.0])
    x, y = np.meshgrid(grid, grid)
    # A corner at the origin approached along the negative x axis keeps pi.
    x[x == 0] = -0.0
    corners = np.stack([x, y], axis=-1)
    quads = ep.pixel_quads(corners)
    centers = quads.mean(axis=1)
    center_angle = np.arctan2(centers[:, 1], centers[:, 0])[:, None]
    qt = center_angle + np.angle(
        np.exp(1j * (np.arctan2(quads[..., 1], quads[..., 0]) - center_angle))
    )
    ns = {"np": np, "qr": np.hypot(quads[..., 0], quads[..., 1]), "qt": qt}
    tree = ast.parse(_gallery_cell("def unroll(t):"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "gallery-unroll", "exec"),
        ns,
    )
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    ax.set(xlim=(-14, 14), ylim=(-14, 14), xticks=[], yticks=[])
    move = ep.quad_image(np.arange(4).reshape(2, 2), corners, ax=ax)
    ept.assert_motion(
        fig, lambda t: move.update(quads=ns["unroll"](t / 2)), ept.Span(0, 2)
    )
