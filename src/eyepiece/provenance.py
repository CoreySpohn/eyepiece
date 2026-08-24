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
import json
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


# Where `stamp` leaves the structured payload for `save_fig` to embed. The
# visible line and the file metadata are the same facts in two channels, so
# a caller should not have to supply them twice.
PAYLOAD_ATTR = "_eyepiece_provenance"

# The gid marking the stamp's Text artist, so a figure can be asked whether
# it carries provenance without parsing what it says.
STAMP_GID = "eyepiece-provenance"

# Each writer accepts a different key vocabulary: PNG takes arbitrary tEXt
# keys, PDF only its fixed infodict set (an unknown key warns), and the SVG
# writer RAISES on one. So the payload is routed per format rather than
# handed to every backend unchanged.
_METADATA_KEYS = {
    ".png": ("Software", "Comment"),
    ".pdf": ("Creator", "Subject"),
    ".svg": ("Creator", "Description"),
}


def provenance_fields(*, script=None, seed=None, note=None, date=None, sha=None):
    """The stamp's contents as structured data, for the file-metadata channel.

    The same facts as `provenance_text`, unjoined, so a reader parsing a
    file does not have to split a display string back apart.

    Returns:
        A dict with the supplied fields only; absent fields are omitted
        rather than set to None, so a consumer can distinguish "no seed"
        from "seed recorded as null".
    """
    fields = {}
    if script is not None:
        fields["script"] = Path(script).name
        if sha is None:
            sha = _git_short_sha(script)
    if sha:
        fields["sha"] = sha
    fields["date"] = date or _datetime.date.today().isoformat()
    if seed is not None:
        fields["seed"] = seed
    if note:
        fields["note"] = str(note)
    return fields


def file_metadata(fields, suffix):
    """Map a provenance payload onto one format's metadata key vocabulary.

    Args:
        fields: The dict from `provenance_fields`.
        suffix: The output file suffix, including the dot.

    Returns:
        A dict suitable for `savefig(metadata=...)` for that format, or an
        empty dict when the format has no supported key vocabulary. An
        unsupported format is not an error: the visible stamp is still on
        the figure, and only the second channel is unavailable.
    """
    keys = _METADATA_KEYS.get(str(suffix).lower())
    if not keys or not fields:
        return {}
    software_key, payload_key = keys
    return {software_key: "eyepiece", payload_key: json.dumps(fields, sort_keys=True)}


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

    Writes both channels the module docstring describes: the visible line
    here, and the structured payload stashed for `save_fig` to embed in the
    file's metadata.

    Returns:
        The `matplotlib.text.Text` artist, or None when there is nothing to
        say.
    """
    text = provenance_text(script=script, seed=seed, note=note, date=date, sha=sha)
    if not text:
        return None
    # The structured form rides along on the figure so `save_fig` can write
    # the file-metadata channel without the caller repeating themselves.
    setattr(
        fig,
        PAYLOAD_ATTR,
        provenance_fields(script=script, seed=seed, note=note, date=date, sha=sha),
    )
    artist = fig.text(
        0.005,
        0.004,
        text,
        ha="left",
        va="bottom",
        fontsize=size,
        color=_style.neutral(0.45),
        zorder=0.1,
    )
    artist.set_gid(STAMP_GID)
    return artist
