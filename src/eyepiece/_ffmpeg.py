"""Locate an ffmpeg binary, without importing matplotlib.

`eyepiece.anim` and `eyepiece.narration` both run ffmpeg; this module holds
the search they share so that the narration layout half stays free of
matplotlib. It imports only the standard library.
"""

import shutil


def resolve_ffmpeg(configured="ffmpeg"):
    """Locate an ffmpeg binary.

    Resolution order: the ``configured`` name or path, then ``ffmpeg`` on
    PATH, then the binary shipped with `imageio_ffmpeg` (an optional
    dependency, imported here and only here so that plain ``import eyepiece``
    never requires it).

    Every candidate is checked with `shutil.which` before it is returned,
    the configured one included: a stale configured path would otherwise be
    handed straight to `subprocess`, turning an actionable error into a bare
    FileNotFoundError from a failed launch.

    Args:
        configured: A name or path tried first, such as matplotlib's
            ``animation.ffmpeg_path``.

    Returns:
        The path or bare name to run ffmpeg with.

    Raises:
        RuntimeError: If no ffmpeg can be found, naming both fixes.
    """
    if shutil.which(configured):
        return configured
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError(
            "mp4 export needs ffmpeg, and none was found: install the "
            "imageio-ffmpeg package (pip install imageio-ffmpeg), which "
            "ships a binary, or put ffmpeg on PATH -- a documentation "
            "builder can install one with the build apt_packages option "
            "(build.apt_packages: [ffmpeg])."
        ) from None
    return imageio_ffmpeg.get_ffmpeg_exe()
