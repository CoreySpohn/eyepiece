"""Offline render benchmark: 2100 artists in 300 mostly absent groups.

Run with this checkout on PYTHONPATH::

    python benchmarks/render_sparse.py --out .codex-bench/sparse --workers 2

Both runs write the same frames. The second uses an explicit state key to
skip repeated states. Times include building, rasterization, and encoding.
"""

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import eyepiece as ep


def frame_key(item):
    """One absolute scene state for each six-frame hold."""
    return item // 6


def build(*, calls):
    """Build 300 groups, each with three labels, two lines, and two patches."""
    fig = plt.figure(figsize=(3.2, 1.8), dpi=100)
    groups = {}
    for index in range(300):
        x = 0.04 + (index % 20) * 0.046
        y = 0.04 + (index // 20) * 0.062
        artists = []
        for offset in range(3):
            artists.append(fig.text(x, y + offset * 0.012, str(index), fontsize=4))
        for offset in range(2):
            line = Line2D(
                [x, x + 0.03],
                [y + offset * 0.01] * 2,
                transform=fig.transFigure,
                linewidth=0.5,
                marker=".",
            )
            fig.add_artist(line)
            artists.append(line)
            patch = Rectangle(
                (x, y + offset * 0.01),
                0.015,
                0.01,
                transform=fig.transFigure,
                facecolor="0.5",
                edgecolor="0.2",
            )
            fig.add_artist(patch)
            artists.append(patch)
        groups[str(index)] = artists
    stage = ep.motion.Stage(groups)
    return fig, partial(draw, stage=stage, calls=calls)


def draw(fig, item, *, stage, calls):
    """Set every group absolutely; at least 299 groups have presence zero."""
    active = frame_key(item) % 300
    stage.weigh({str(index): float(index == active) for index in range(300)})
    with Path(calls).open("a") as stream:
        stream.write(f"{item}\n")


def main():
    """Compare complete and explicitly deduplicated renders, including bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    videos = []
    for name, key in (("full", None), ("dedup", frame_key)):
        calls = args.out / f"{name}-draws.txt"
        calls.unlink(missing_ok=True)
        path = args.out / f"{name}.mp4"
        result = ep.render.render_frames(
            partial(build, calls=calls),
            args.frames,
            path,
            fps=30,
            dpi=100,
            workers=args.workers,
            frame_key=key,
        )
        count = len(calls.read_text().splitlines())
        print(f"{name}: {result.seconds:.3f} s, {count} draws")
        videos.append(path)
    assert videos[0].read_bytes() == videos[1].read_bytes()
    print("MP4 bytes identical")


if __name__ == "__main__":
    # Workers import builders by module name rather than the launcher's main.
    import render_sparse

    render_sparse.main()
