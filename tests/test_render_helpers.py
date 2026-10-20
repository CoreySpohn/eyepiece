"""Importable builders for spawned render processes."""

import os
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def build_variant_clip(
    venue, mode, *, markers, fail_mode=None, data=None, failure_root=None
):
    """Count builds on disk, including builds in spawned workers."""
    folder = Path(markers)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{venue.name}-{mode}").open("a") as stream:
        stream.write(f"{os.getpid()}\n")
    if mode == fail_mode:
        raise RuntimeError("deliberate variant failure")
    fig = plt.figure(figsize=(0.96, 0.54))
    if data is not None:
        fig.set_facecolor(data[mode])
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set(xlim=(0, 7), ylim=(-1, 1), xticks=[], yticks=[])
    (fig._variant_marker,) = ax.plot([], [], "o", markersize=3)
    return fig, partial(
        draw_variant_clip,
        failure_dir=None if failure_root is None else failure_root / mode,
    )


def build_style_variant(venue, mode):
    """Read both the active background and semantic video-style colors."""
    import eyepiece as ep

    fig = plt.figure(figsize=(0.96, 0.54))
    ax = fig.add_axes((0.25, 0.25, 0.5, 0.5))
    ax.set_facecolor(ep.role("planet"))
    ax.set(xticks=[], yticks=[])
    return fig, draw_style_variant


def draw_style_variant(fig, frame):
    """Keep the style probe static across frames."""


def setup_video_style():
    """Explicitly enable video colors for either rendering path."""
    import hwostyle

    import eyepiece as ep

    hwostyle.use("dark")
    ep.use_video(True)


def draw_variant_clip(fig, frame, *, failure_dir=None):
    if failure_dir is not None and frame == 3:
        failure_dir.mkdir(parents=True, exist_ok=True)
        marker = failure_dir / "failed"
        if not marker.exists():
            marker.touch()
            raise RuntimeError("deliberate chunk failure")
    fig._variant_marker.set_data([frame], [0])


def variant_frame_at(index):
    return index / 2


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
