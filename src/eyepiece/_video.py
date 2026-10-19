"""A video-safe variant of the dark role colors, derived rather than picked.

Video is stored as 4:2:0 YCbCr: color is kept at half the resolution of
brightness, so a thin line in a strongly saturated color bleeds and loses its
edge. The pure-RGB roles of the dark palette (the cyan planet, the yellow
star, the red alert) sit far out in Cb and Cr. Two dark roles are also dim
for text on black: `disk` and `reference` fall below a contrast of 5.

The ``"dark-video"`` table takes each dark role color and

1. holds its Oklab hue angle, so the role keeps its identity (Oklab keeps
   hue along a line of falling chroma better than CIELAB, which turns a
   desaturated red toward orange);
2. lowers its Oklab chroma at fixed lightness until its BT.709 chroma
   (the length of (Cb, Cr) in 8-bit studio code values) is at most
   `CHROMA_CAP`;
3. raises its lightness, if needed, until its WCAG contrast against the dark
   background is at least `MIN_CONTRAST`, keeping chroma within the cap and
   the sRGB gamut.

Three roles take a relaxed limit (`RELAXED`), because the plain limits put
them within the `motif_faults` fault distance of another role for a viewer
with a color-vision deficiency: `alert` keeps a cap of 85, and `disk` and
`prior` (one color) a contrast floor of 4.5, the WCAG AA level for text.

A color already inside both limits is returned unchanged. The dark, light
and paper palettes are untouched; the table is used only while
`eyepiece.use_video` is on in dark mode, or when a check names the mode.
"""

import functools
import warnings

import numpy as np
from matplotlib.colors import to_hex, to_rgb

VIDEO_MODE = "dark-video"
VIDEO_BASE = "dark"
CHROMA_CAP = 75.0  # BT.709 |(Cb, Cr)| in 8-bit studio code values
MIN_CONTRAST = 5.0  # WCAG contrast against the background
_MARGIN = 0.02  # contrast headroom so the rounded hex still clears the floor
_CAP_MARGIN = 0.5  # chroma headroom so the rounded hex stays under the cap
# Roles whose cap or floor `motif_faults` forces open: at the plain limits the
# capped red falls within 10 CAM02-UCS of the brown posterior, and the lifted
# purple within 10 of the positive blue, under a color-vision deficiency.
# Each entry is (cap, floor); None keeps the plain limit.
RELAXED = {
    "alert": (85.0, None),
    "disk": (None, 4.5),
    "prior": (None, 4.5),
}

# Oklab (Ottosson 2020): linear sRGB to LMS, and LMS cube roots to Lab.
_RGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LMS_TO_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_LUMINANCE = np.array([0.2126729, 0.7151522, 0.0721750])  # sRGB row of XYZ Y


def _linear(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _encoded(lin):
    lin = np.asarray(lin, dtype=float)
    return np.where(
        lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1.0 / 2.4) - 0.055
    )


def _lch(rgb):
    """Oklab lightness (0 to 1), chroma and hue angle (radians) of sRGB."""
    lightness, a, b = _LMS_TO_LAB @ np.cbrt(_RGB_TO_LMS @ _linear(rgb))
    return float(lightness), float(np.hypot(a, b)), float(np.arctan2(b, a))


def _rgb(lightness, chroma, hue):
    """The sRGB of an Oklab LCh color; components may fall outside [0, 1]."""
    lab = np.array([lightness, chroma * np.cos(hue), chroma * np.sin(hue)])
    lms = np.linalg.solve(_LMS_TO_LAB, lab) ** 3
    return _encoded(np.linalg.solve(_RGB_TO_LMS, lms))


def _in_gamut(rgb):
    return bool(np.all(rgb >= -1e-9) and np.all(rgb <= 1.0 + 1e-9))


def ycc_chroma(rgb):
    """BT.709 chroma of an sRGB color: |(Cb, Cr)| in 8-bit studio code values."""
    r, g, b = np.asarray(rgb, dtype=float)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return float(np.hypot((b - luma) / 1.8556 * 224.0, (r - luma) / 1.5748 * 224.0))


def contrast(color, background):
    """WCAG contrast ratio of two colors."""
    lum = [float(_linear(to_rgb(c)) @ _LUMINANCE) for c in (color, background)]
    return (max(lum) + 0.05) / (min(lum) + 0.05)


def _largest(ok, hi, steps=50):
    """The largest c in [0, hi] with ok(c), for ok true at 0 and monotone."""
    if ok(hi):
        return hi
    lo = 0.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def _chroma_at(lightness, chroma, hue, cap):
    """The chroma kept at a lightness: within the cap, the gamut and `chroma`."""

    def ok(c):
        rgb = _rgb(lightness, c, hue)
        return _in_gamut(rgb) and ycc_chroma(rgb) <= cap - _CAP_MARGIN

    return _largest(ok, chroma)


def derive(color, background, *, cap=CHROMA_CAP, floor=MIN_CONTRAST):
    """The video-safe version of `color` on `background`, as a hex string.

    Args:
        color: Any matplotlib color.
        background: The background the color is read on.
        cap: Largest BT.709 chroma.
        floor: Least contrast against `background`.

    Returns:
        `color` itself when its BT.709 chroma is within `cap` and its
        contrast at least `floor`, else the nearest color of the same hue
        that is.
    """
    rgb = np.asarray(to_rgb(color))
    if ycc_chroma(rgb) <= cap and contrast(color, background) >= floor:
        return to_hex(color)
    lightness, chroma, hue = _lch(rgb)
    chroma = _chroma_at(lightness, chroma, hue, cap)
    target = floor + _MARGIN

    def clears(lift):
        lit = lightness + lift
        c = _chroma_at(lit, chroma, hue, cap)
        return contrast(to_hex(np.clip(_rgb(lit, c, hue), 0, 1)), background) >= target

    if not clears(0.0):
        lo, hi = 0.0, 1.0 - lightness
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if clears(mid) else (mid, hi)
        lightness += hi
        chroma = _chroma_at(lightness, chroma, hue, cap)
    return to_hex(np.clip(_rgb(lightness, chroma, hue), 0.0, 1.0))


@functools.cache
def _tables():
    import hwostyle.styles
    from hwostyle.palettes import ALL_ROLE_NAMES, Roles

    roles = Roles(VIDEO_BASE)
    back = hwostyle.styles.MODE_RC[VIDEO_BASE]["axes.facecolor"]
    with warnings.catch_warnings():
        # Deprecated roles still need an entry while hwostyle carries them.
        warnings.simplefilter("ignore", DeprecationWarning)
        marks, labels = {}, {}
        for name in ALL_ROLE_NAMES:
            cap, floor = RELAXED.get(name, (None, None))
            limits = {
                "cap": CHROMA_CAP if cap is None else cap,
                "floor": MIN_CONTRAST if floor is None else floor,
            }
            marks[name] = derive(roles[name], back, **limits)
            labels[name] = derive(roles.label_ink(name), back, **limits)
    return marks, labels


def video_roles():
    """The ``"dark-video"`` role table: role name to hex, derived from dark.

    Returns:
        A new dict of every hwostyle role and its video-safe color.

    Example::

        import eyepiece as ep
        ep.video_roles()["planet"]  # the dark cyan, chroma capped for 4:2:0
    """
    return dict(_tables()[0])


def video_label_inks():
    """Label inks of the ``"dark-video"`` mode, derived like the marks."""
    return dict(_tables()[1])


def base_mode(mode):
    """The hwostyle mode a mode name stands on (``"dark"`` for the video mode)."""
    return VIDEO_BASE if mode == VIDEO_MODE else mode
