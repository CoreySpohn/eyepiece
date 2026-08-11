"""Write a figure to disk with mode-aware defaults and a resolved directory.

The style library owns savefig policy (dpi, transparency, bbox handling) as
data, keyed by the active mode; this module owns the mechanism of getting a
figure onto disk with that policy applied. The policy is fetched fresh on
every call rather than cached, so a mode switch between import time and save
time is always honored.

A figure's own facecolor overrides the mode's declared facecolor: the mode
default describes what a *new* figure should look like, but a figure already
carries the rcParams that were in force when it was created, which can
predate a later mode switch. Saving with the mode's facecolor instead of the
figure's own would silently mismatch the plot as rendered on screen from the
file written to disk.
"""

import os
from pathlib import Path

import hwostyle


def save_fig(fig, name, *, dir=None, anchor=None, **overrides):
    """Save a figure using the active hwostyle mode's savefig policy.

    Directory resolution, in order: the explicit `dir` argument; else
    `Path(anchor).resolve().parent / "output"`; else the `EYEPIECE_OUT`
    environment variable; else `Path.cwd() / "output"`. The chosen directory
    is created if it does not already exist.

    The savefig kwargs are built in three layers, later layers winning:
    `hwostyle.save_defaults()` for the active mode, then `facecolor`
    replaced with `fig.get_facecolor()` (the figure's own background always
    beats the mode's declared one -- see module docstring), then the
    caller's `overrides`, so an explicit `facecolor=` or `dpi=` always wins.

    Args:
        fig: A matplotlib Figure to save.
        name: Output filename. A name with no suffix gets ".png" appended;
            a name that already has a suffix (e.g. "plot.pdf") keeps it.
        dir: Explicit output directory. Highest-priority resolution branch.
        anchor: A path (typically a calling script's `__file__`) whose
            parent directory anchors an "output" subdirectory. Used only
            when `dir` is not given.
        **overrides: Extra kwargs passed to `fig.savefig`, applied last.

    Returns:
        The `pathlib.Path` the figure was written to.
    """
    if dir is not None:
        out_dir = Path(dir)
    elif anchor is not None:
        out_dir = Path(anchor).resolve().parent / "output"
    elif "EYEPIECE_OUT" in os.environ:
        out_dir = Path(os.environ["EYEPIECE_OUT"])
    else:
        out_dir = Path.cwd() / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / name
    if not path.suffix:
        path = path.with_suffix(".png")

    kwargs = hwostyle.save_defaults()
    kwargs["facecolor"] = fig.get_facecolor()
    kwargs.update(overrides)

    fig.savefig(path, **kwargs)
    return path
