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


ENCODER_PROFILE = {
    "crf": 18,
    "tune": "animation",
    "keyframe_s": 2.0,
    "colorspace": "bt709",
    "faststart": True,
}
"""The H.264 settings every eyepiece movie is encoded with.

RGB frames are converted with the BT.709 matrix into limited-range
yuv420p and the stream is tagged BT.709, because players assume that matrix
for untagged HD video while ffmpeg converts with BT.601 by default, which
moves a saturated brand color by up to 19 levels. ``crf`` and ``tune`` set
quality; ``keyframe_s`` puts a keyframe at least that often so a reader
scrubbing a clip lands quickly; ``faststart`` moves the index to the front
so playback starts before the download ends. ``colorspace`` is ``"bt709"``
or None, which leaves the conversion and tags to ffmpeg. Override any key
with ``encoder={...}`` on `record`, `RawSink` or `narration.mux`.
"""

_PROFILE_KEYS = frozenset(ENCODER_PROFILE)


def _merged_profile(encoder):
    """`ENCODER_PROFILE` with the keys of `encoder` replaced."""
    encoder = {} if encoder is None else dict(encoder)
    unknown = sorted(set(encoder) - _PROFILE_KEYS)
    if unknown:
        raise ValueError(
            f"unknown encoder keys {unknown}; "
            f"the profile keys are {sorted(_PROFILE_KEYS)}"
        )
    return {**ENCODER_PROFILE, **encoder}


def with_caller_pixels(encoder, extra_args):
    """Drop the profile's color handling when the caller picks the pixel format.

    A caller who passes ``-pix_fmt`` (a lossless rgb encode, say) owns the
    pixel format, and a BT.709 yuv conversion or tag would corrupt it.

    Returns:
        `encoder` with ``colorspace`` set to None when `extra_args` holds a
        ``-pix_fmt``, else `encoder` unchanged.
    """
    if extra_args and "-pix_fmt" in list(extra_args):
        return {**(encoder or {}), "colorspace": None}
    return encoder


def convert_filter(encoder=None):
    """The ffmpeg filter that converts RGB frames to the profile's matrix, or "".

    Returns:
        A filter string to chain after any crop, empty when the profile
        leaves the conversion to ffmpeg.
    """
    if _merged_profile(encoder)["colorspace"] == "bt709":
        return "scale=out_color_matrix=bt709:out_range=tv"
    return ""


def encoder_args(fps, encoder=None):
    """Build the ffmpeg output arguments for the profile at a frame rate.

    The conversion filter is not included: see `convert_filter`.

    Args:
        fps: Frames per second of the movie, which sets the keyframe spacing.
        encoder: Mapping of `ENCODER_PROFILE` keys to replace, or None.

    Returns:
        A list of ffmpeg arguments, quality and tags, then the movie flags.

    Raises:
        ValueError: If `encoder` has a key the profile does not.
    """
    profile = _merged_profile(encoder)
    args = ["-crf", str(profile["crf"])]
    if profile["tune"]:
        args += ["-tune", profile["tune"]]
    if profile["keyframe_s"]:
        args += ["-g", str(max(1, round(float(fps) * profile["keyframe_s"])))]
    if profile["colorspace"] == "bt709":
        args += [
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-color_range", "tv",
        ]  # fmt: skip
    if profile["faststart"]:
        args += ["-movflags", "+faststart"]
    return args
