"""Importable builders for spawned render processes."""

import os
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def build_crashing_clip(*, killed=False, exitcode=7):
    fig, _ = build_clip()
    return fig, partial(draw_crashing_clip, killed=killed, exitcode=exitcode)


def draw_crashing_clip(fig, frame, *, killed, exitcode):
    if frame == 1:
        print("deliberate worker crash", file=sys.stderr, flush=True)
        if killed:
            os.kill(os.getpid(), 9)
        os._exit(exitcode)
    draw_clip(fig, frame)


def build_waiting_clip(*, folder):
    (folder / f"pid-{os.getpid()}").write_text(str(os.getpid()))
    fig, _ = build_clip()
    return fig, partial(draw_waiting_clip, folder=folder)


def draw_waiting_clip(fig, frame, *, folder):
    if frame in (5, 7):
        (folder / f"blocked-{frame}").touch()
        while not (folder / "continue").exists():
            time.sleep(0.02)
    draw_clip(fig, frame)


def build_tiny_clip():
    return plt.figure(figsize=(0.02, 0.02)), draw_tiny_clip


def draw_tiny_clip(fig, frame):
    fig.set_facecolor("white" if frame == 0 else "black")


def frame_at(index):
    return index / 23


def setup_style():
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def build_clip(
    *,
    odd=False,
    accumulating=False,
    failure_dir=None,
    observe_dir=None,
    wait_for_chunks=True,
):
    if observe_dir is not None:
        folder = Path(observe_dir)
        lock = next(folder.glob("*.lock"))
        (folder / "lock-name.txt").write_text(lock.name)
    fig, ax = plt.subplots(
        figsize=(1.93 if odd else 1.92, 1.09 if odd else 1.08),
        dpi=73,
        layout="constrained",
    )
    ax.set(xlim=(-0.1, 1.1), ylim=(-1, 1))
    (line,) = ax.plot([], [], "o")
    ax.set_title("Moving marker", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    fig._render_line = line
    return fig, partial(
        draw_clip,
        accumulating=accumulating,
        failure_dir=failure_dir,
        wait_for_chunks=wait_for_chunks,
    )


def draw_clip(
    fig,
    frame,
    *,
    accumulating=False,
    failure_dir=None,
    wait_for_chunks=True,
):
    if failure_dir is not None and frame == 12:
        folder = Path(failure_dir)
        marker = folder / "failed-once"
        if not marker.exists():
            marker.touch()
            deadline = time.monotonic() + 30
            while wait_for_chunks and len(list(folder.glob("chunk-*.bin"))) < 3:
                if time.monotonic() > deadline:
                    raise TimeoutError("other chunks did not complete")
                time.sleep(0.02)
            raise OSError("deliberate draw failure")
    line = fig._render_line
    x = frame if isinstance(frame, float) else frame_at(frame)
    if accumulating:
        old = list(line.get_xdata())
        line.set_data([*old, x], [0] * (len(old) + 1))
    else:
        line.set_data([x], [0])
