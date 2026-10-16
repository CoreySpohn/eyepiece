"""Shared figure, word-timed plan, and drawing for tutorial 3."""

import math

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import eyepiece as ep
from eyepiece import narration as nr


def caption_steps(plan, cues):
    """Declare the first frame at every caption replacement or clearing."""
    return sorted(
        {
            math.ceil(t * plan.fps)
            for start, end, _ in cues
            for t in (start, end)
            if 0 <= math.ceil(t * plan.fps) < plan.n_frames
        }
    )


def make_timeline(narration):
    """Place the tutorial 2 reveals on the marked spoken words."""
    tl = ep.Timeline(narration=narration)
    tl.line("panel")
    tl.enter("panel", 1.5)
    tl.line("image")
    tl.enter("image", 1.5)
    tl.line("profile")
    tl.enter("profile", 2.0)
    tl.line("marker")
    tl.pulse("marker")
    tl.line("close")
    return tl


class Scene:
    """Tutorial 2's field and row profile with a reserved caption band."""

    def __init__(self):
        """Build the finished figure and freeze its panel layout."""
        hwostyle.use("dark")
        plt.rcParams.update(
            {
                "font.size": 17,
                "xtick.labelsize": 17,
                "ytick.labelsize": 17,
                "axes.labelsize": 17,
                "axes.titlesize": 18,
                "savefig.dpi": 80,
            }
        )
        n = 64
        coordinate = np.linspace(-12.0, 12.0, n)
        x, y = np.meshgrid(coordinate, coordinate)
        radius = np.hypot(x, y)
        angle = np.arctan2(y, x)
        halo = 1e-7 * (1.0 + radius) ** -3 * (1.0 + 0.8 * np.cos(12 * angle) ** 2)
        planet = 3e-9 * np.exp(-0.5 * (np.hypot(x + 6.0, y - 5.0) / 0.6) ** 2)
        field = halo + planet
        profile = field[n // 2] / field[n // 2].max()

        self.fig, (image_ax, profile_ax) = plt.subplots(
            1, 2, figsize=(7.5, 5.4), dpi=80, layout="constrained"
        )
        self.fig.get_layout_engine().set(rect=(0, 0.24, 1, 0.76))
        self.result = ep.imshow_log(
            field,
            ax=image_ax,
            extent=ep.extent_lod_from_pixels(n, 24.0 / (n - 1)),
            vmin=1e-12,
            vmax=1e-7,
            floor=1e-12,
            colorbar=False,
        )
        image_ax.set(
            xlabel="x [lambda/D]",
            ylabel="y [lambda/D]",
            title="Field",
            xticks=[-10, 0, 10],
            yticks=[-10, 0, 10],
        )
        (self.curve,) = profile_ax.plot(
            coordinate, profile, color=ep.neutral(0.85), lw=2
        )
        (self.marker,) = profile_ax.plot(
            [1.5],
            [np.interp(1.5, coordinate, profile)],
            "o",
            color=hwostyle.palette[0],
            ms=7,
        )
        profile_ax.set(
            xlim=(-12, 12),
            ylim=(0, 1.1),
            xticks=[-10, 0, 10],
            yticks=[0, 1],
            xlabel="x [lambda/D]",
            ylabel="relative intensity",
            title="Row profile",
        )
        self.fig.add_artist(
            Rectangle(
                (0, 0),
                1,
                0.22,
                transform=self.fig.transFigure,
                facecolor=ep.neutral(0.08),
                edgecolor="none",
                zorder=0,
            )
        )
        self.caption = self.fig.text(
            0.5,
            0.11,
            "",
            ha="center",
            va="center",
            fontsize=17,
            color=ep.neutral(0.95),
            linespacing=1.3,
        )
        self.fig.canvas.draw()
        self.fig.set_layout_engine("none")
        self.labels = ep.Stage(
            {
                "labels": [ax.xaxis.label for ax in (image_ax, profile_ax)]
                + [ax.yaxis.label for ax in (image_ax, profile_ax)]
            }
        )
        self.builds = [
            ep.build_axes(ax, label=False, title=True) for ax in (image_ax, profile_ax)
        ]
        self.reveal = ep.draw_on(self.curve)
        self.emphasis = ep.indicate(self.marker, scale=1.5)
        self.moves = {self.curve: "profile"}
        plt.close(self.fig)

    def draw(self, k, plan, cues):
        """Set all artist states from one frame, including its caption."""
        self.labels.weigh({"labels": plan.value(k, "panel")})
        for build in self.builds:
            build.update(plan.value(k, "panel"))
        self.result.artists["image"].set_alpha(plan.value(k, "image"))
        self.reveal.update(plan.value(k, "profile"))
        self.marker.set_alpha(plan.value(k, "profile"))
        self.emphasis.update(plan.value(k, "pulse.marker"))
        self.caption.set_text(nr.caption_at(cues, k / plan.fps))
