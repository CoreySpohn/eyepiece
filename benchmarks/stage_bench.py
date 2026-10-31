"""Deterministic, data-free Stage benchmark; run this file for CSVs and findings."""

import argparse
import csv
import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import eyepiece as ep


def scene():
    """Make 2100 artists in 300 groups, with 90 percent absent each frame."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=80)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    groups = {}
    for i in range(300):
        x, y = rng.uniform(0.05, 0.9, size=2)
        color = (0.1, 0.2 + (i % 3) * 0.2, 0.5)
        texts = [
            ax.text(
                x,
                y + j * 0.015,
                f"{i}:{j}",
                color=color,
                fontsize=5,
                in_layout=(j == 0),
            )
            for j in range(4)
        ]
        (line,) = ax.plot(
            [x, x + 0.02],
            [y, y + 0.01],
            color=color,
            marker="o",
            markersize=2,
            markeredgecolor="red",
            fillstyle="left",
            markerfacecoloralt="green",
            alpha=0.8 if i % 2 else None,
        )
        patch = Rectangle((x, y), 0.01, 0.01, facecolor=color, edgecolor="black")
        ax.add_patch(patch)
        dots = ax.scatter([x], [y], s=2, facecolor=color, edgecolor="red")
        groups[str(i)] = [*texts, line, patch, dots]
    stage = ep.motion.Stage(groups)

    def draw(k):
        phase = k % 10
        stage.weigh({str(i): (1.0 if i % 10 == phase else 0.0) for i in range(300)})

    return fig, stage, draw


def findings(fig, stage, draw):
    """Report readability and motion findings at deterministic review frames."""
    lines = []
    for k in (0, 3, 9, 0):
        draw(k)
        faults = ep.testing.readability_faults(fig)
        lines.append(f"frame {k}: readability {len(faults)}")
        lines.extend(str(f) for f in faults)
    faults = ep.testing.motion_faults(
        fig,
        lambda t: stage.weigh(dict.fromkeys(stage.names, t)),
        [0, 0.5, 1],
        moves=[stage],
        kinds=("drift",),
    )
    lines.append(f"motion {len(faults)}")
    lines.extend(str(f) for f in faults)
    return "\n".join(lines) + "\n"


def main():
    """Time updates and canvas draws, hash each frame, and write check findings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=150)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig, stage, draw = scene()
    draw(0)
    fig.canvas.draw()
    rows = []
    for k in np.random.default_rng(0).integers(0, 1000, args.n):
        t0 = time.perf_counter()
        draw(int(k))
        t1 = time.perf_counter()
        fig.canvas.draw()
        t2 = time.perf_counter()
        digest = hashlib.sha256(
            np.asarray(fig.canvas.buffer_rgba()).tobytes()
        ).hexdigest()
        rows.append((k, (t1 - t0) * 1000, (t2 - t1) * 1000, digest))
    with (args.out / "draw.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "scene_draw_ms", "canvas_draw_ms", "rgba_sha256"])
        writer.writerows(rows)
    (args.out / "checks.txt").write_text(findings(fig, stage, draw))
    print("median ms scene, canvas:", np.median([r[1:3] for r in rows], axis=0))
    plt.close(fig)


if __name__ == "__main__":
    main()
