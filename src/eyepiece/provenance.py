"""Provenance stamps: what a reader needs to rebuild or challenge a figure.

A figure is a claim, and a claim nobody can check is a claim that has to be
taken on trust. The conditions a figure was produced under are the half of
the evidence that vanishes first: the script, the date, the code version, the
random seed, and whether the data was simulated at all. None of that is
recoverable from the picture, and a picture outlives the conversation that
explained it.

The stamp is deliberately furniture: the smallest legible type, a neutral
resolved against the live background, one fixed corner. Furniture a reader
can ignore costs almost nothing. Provenance that is absent costs the figure
its standing.

Two channels, because either one alone is lost. The visible line survives a
screenshot and is lost to a crop; the file metadata survives a crop and is
lost to a screenshot. Writing both is cheap.
"""

import datetime as _datetime
import subprocess
from pathlib import Path

from eyepiece import _style


def _git_short_sha(anchor):
    """Short commit SHA of the repository containing `anchor`, or None."""
    if anchor is None:
        return None
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(Path(anchor).resolve().parent),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def provenance_text(*, script=None, seed=None, note=None, date=None, sha=None):
    """Assemble the one-line stamp, skipping fields that are not supplied.

    Args:
        script: Path to the generating script, typically ``__file__``. Only
            the basename is shown; the full path is noise to a reader.
        seed: Random seed, if anything was sampled. A figure drawn from a
            generator shows one draw out of many, and the seed is what makes
            that draw recoverable.
        note: Free text. This is where the word "simulated" belongs when the
            data came from a generator, because no property of the drawing
            reveals it and a reader will otherwise assume a measurement.
        date: Date string. None uses today.
        sha: Code version. None derives it from the script's repository.

    Returns:
        The assembled single-line string, possibly empty.
    """
    parts = []
    if script is not None:
        parts.append(Path(script).name)
        if sha is None:
            sha = _git_short_sha(script)
    if sha:
        parts.append(sha)
    parts.append(date or _datetime.date.today().isoformat())
    if seed is not None:
        parts.append(f"seed {seed}")
    if note:
        parts.append(str(note))
    return "  |  ".join(parts)


def stamp(fig, *, script=None, seed=None, note=None, date=None, sha=None, size=6.0):
    """Write a provenance line along the bottom edge of a figure.

    Args:
        fig: The Figure to stamp.
        script: Generating script path, typically ``__file__``.
        seed: Random seed, if anything was sampled.
        note: Free text; say "simulated" here when it is.
        date: Date string. None uses today.
        sha: Code version. None derives it from the script's repository.
        size: Font size in points. The default is deliberately small: this
            is furniture and must never compete with the data.

    Returns:
        The `matplotlib.text.Text` artist, or None when there is nothing to
        say.
    """
    text = provenance_text(script=script, seed=seed, note=note, date=date, sha=sha)
    if not text:
        return None
    return fig.text(
        0.005,
        0.004,
        text,
        ha="left",
        va="bottom",
        fontsize=size,
        color=_style.neutral(0.45),
        zorder=0.1,
    )
