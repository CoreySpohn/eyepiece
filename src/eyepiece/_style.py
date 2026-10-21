"""Call-time resolution of hwostyle state.

Never bind `cmaps` or `palette` by name off of the hwostyle module:
hwostyle.use() rebinds those module globals, so a captured reference
silently freezes the mode a caller saw at import time. Every lookup here
goes through the `hwostyle` module attribute at call time, so a later
`hwostyle.use()` switch is always seen.

This module serves the Matplotlib primitives; the prepared-view renderers
take their appearance from an explicit snapshot instead (`eyepiece.style`).
"""

import hwostyle
import matplotlib
from matplotlib.colors import to_hex, to_rgb, to_rgba

_LIGHT_CMAPS = None
_VIDEO = False
_LIGHT_PALETTE = None
_LIGHT_ROLES = None
# Stand-ins for roles an older hwostyle lacks; "pupil" is pale to dark like
# the role it replaces on a light page.
_CMAP_FALLBACKS = {"statistic": "PuOr", "pupil": "Oranges"}
# Palette slot standing in for a role when the installed hwostyle lacks it;
# slots follow the canonical cyan, pink, yellow, green, red, purple order.
_ROLE_FALLBACKS = {
    "star": 2,
    "planet": 0,
    "disk": 5,
    "measured": 0,
    "data": 0,
    "model": 1,
    "predicted": 1,
    "reference": 3,
    "zodi": 3,
    "envelope": 5,
    "alert": 4,
    "highlight": 1,
    "prior": 5,
    # hwostyle gives these roles fixed colors rather than palette slots, so the
    # stand-in is its light-mode hex.
    "positive": "#1565C0",
    "negative": "#E65100",
    "posterior": "#662506",
    "truth": "#757575",
}


def mode():
    """Active mode name.

    The hwostyle mode, "light" when none is active, or "dark-video" while
    `use_video` is on in dark mode. "dark-video" is not an hwostyle mode:
    pass the name through `eyepiece._video.base_mode` before looking it up
    in a table keyed by hwostyle mode (such as `MODE_RC`).
    """
    name = hwostyle.current_mode() or "light"
    if _VIDEO and name == "dark":
        from eyepiece._video import VIDEO_MODE

        return VIDEO_MODE
    return name


def use_video(on=True):
    """Resolve roles from the video-safe dark table while in dark mode.

    The ``"dark-video"`` table is derived from the dark roles (see
    `video_roles`): each role keeps its hue, its BT.709 chroma is capped
    at 75 so thin lines survive 4:2:0 encoding, and a dim role is lifted
    to a contrast of at least 5 on the dark background. It applies to
    `role` and to motif colors; the palette cycle, colormaps and neutrals
    are unchanged, and so are the light and paper modes. Turn it on for a
    clip that is encoded to video, after ``hwostyle.use("dark")``.

    Args:
        on: Whether video-safe roles are used.

    Returns:
        Whether they were used before the call.

    Example::

        hwostyle.use("dark")
        ep.use_video()
        ep.role("planet")  # the capped cyan
    """
    global _VIDEO
    was, _VIDEO = _VIDEO, bool(on)
    return was


def dim_level(mode_name=None):
    """Contrast a dimmed group keeps: 0.35 in dark mode, 0.5 otherwise.

    Dimmed hues lose separation faster on a white page than on black, so the
    light and paper levels dim less. hwostyle owns the numbers
    (`hwostyle.motifs.DIM_LEVELS`); an older hwostyle without them gets the
    same values from here.

    Args:
        mode_name: A mode name, or None for the active mode.

    Returns:
        A float in [0, 1].
    """
    from eyepiece._video import base_mode

    name = base_mode(mode_name or mode())
    try:
        from hwostyle.motifs import DIM_LEVELS as levels
    except ImportError:
        levels = {"dark": 0.35}
    return levels.get(name, 0.5)


def _light_cmaps():
    global _LIGHT_CMAPS
    if _LIGHT_CMAPS is None:
        from hwostyle.colormaps import Colormaps

        _LIGHT_CMAPS = Colormaps("light")
    return _LIGHT_CMAPS


def _light_palette():
    global _LIGHT_PALETTE
    if _LIGHT_PALETTE is None:
        from hwostyle.palettes import Palette

        _LIGHT_PALETTE = Palette("light")
    return _LIGHT_PALETTE


def _light_roles():
    global _LIGHT_ROLES
    if _LIGHT_ROLES is None:
        from hwostyle.palettes import Roles

        _LIGHT_ROLES = Roles("light")
    return _LIGHT_ROLES


def _as_colormap(value):
    """Normalize a hwostyle colormap value to a matplotlib Colormap.

    hwostyle.Colormaps.__getattr__ returns a plain registered-name string
    for most semantic names (e.g. "magma") but a Colormap object for others
    (e.g. "mask"). Callers need a consistent type.
    """
    if isinstance(value, str):
        return matplotlib.colormaps[value]
    return value


def cmap(name, override=None):
    """Semantic colormap by name ("intensity", "phase", "residual", ...).

    Args:
        name: Semantic name defined by hwostyle.
        override: A matplotlib Colormap, registered cmap name, or semantic
            role; wins when not None. A missing "statistic" or "pupil" role
            (an hwostyle older than the role) falls back to PuOr or
            Oranges.

    Returns:
        A matplotlib Colormap instance.
    """
    if override is not None and not isinstance(override, str):
        return override
    source = hwostyle.cmaps if hwostyle.current_mode() else _light_cmaps()
    role = name if override is None else override
    try:
        value = getattr(source, role)
    except AttributeError:
        if role in _CMAP_FALLBACKS:
            value = _CMAP_FALLBACKS[role]
        elif override is not None:
            value = override
        else:
            raise
    return _as_colormap(value)


def role(name, override=None, *, mode=None):
    """Semantic brand color by role ("alert", "model", "reference", ...).

    The role resolves under the active hwostyle mode, or the light mode when
    none is active, through the same call-time lookup as `cmap`. When the
    installed hwostyle does not carry the role, a known role name falls back
    to a stable stand-in: its palette slot in the canonical cyan, pink,
    yellow, green, red, purple order, or, for the roles hwostyle colors
    outside the palette (positive, negative, posterior, truth), hwostyle's
    light-mode hex.

    A neutral rank (``"hardware"``, ``"text"``, ...) names a gray, the same
    tone `neutral` gives for that rank's level, so a `Motif` role, which is
    either a color role or a rank, resolves through this one function.
    `neutral` takes the level as a number, for a tone between the ranks.

    Args:
        name: Role name, such as "alert", "model", "planet" or "reference",
            or a neutral rank such as "hardware" or "text".
        override: A color; wins when not None.
        mode: A hwostyle mode name, such as "dark" or "paper", to resolve
            the color in that mode without activating it; None uses the
            active mode.

    Returns:
        A hex color string.

    Raises:
        KeyError: If the name is neither a known role nor a neutral rank.

    Example::

        import eyepiece as ep
        ax.axvline(0.5, color=ep.role("alert"))
    """
    if override is not None:
        return override
    # motifs imports this module, so its rank table is read at call time.
    from eyepiece import motifs

    ranks = motifs._ranks()
    if ranks.get(name) is not None:
        return motifs._neutral_in(ranks[name], mode)
    if mode is not None:
        return motifs._role_in(name, mode)
    if name not in _ROLE_FALLBACKS:
        raise KeyError(
            f"No role {name!r}. Available: {sorted(_ROLE_FALLBACKS)} and the "
            f"neutral ranks {sorted(n for n, v in ranks.items() if v is not None)}"
        )
    return _active_role(name)


def _active_role(name):
    """Hex color of a known role under the active mode, with its fallbacks."""
    from eyepiece._video import VIDEO_MODE, video_roles

    if mode() == VIDEO_MODE:
        table = video_roles()
        if name in table:
            return table[name]
    source = getattr(hwostyle, "roles", None) if hwostyle.current_mode() else None
    if source is None:
        try:
            source = _light_roles()
        except ImportError:
            source = None
    try:
        return to_hex(source[name])
    except (KeyError, TypeError):
        fallback = _ROLE_FALLBACKS[name]
        return to_hex(fallback) if isinstance(fallback, str) else color(fallback)


def _factory_default_cycle():
    """True when the active property cycle is still matplotlib's factory default.

    Cyclers compare by value, so a user cycle that happens to equal the
    factory default (same colors, same order) reads as "not customized"
    here; that is treated as acceptable, since a cycle indistinguishable
    from the default should get the same fallback as never touching it.
    """
    return (
        matplotlib.rcParams["axes.prop_cycle"]
        == matplotlib.rcParamsDefault["axes.prop_cycle"]
    )


def _cycle_colors():
    """Hex colors of the active property cycle, or [] if it defines none.

    A cycler need not carry a "color" key at all -- matplotlib documents
    linestyle-only and marker-only cycles -- and even a "color" key can be
    set to an empty list. Both mean the same thing: no color preference.
    """
    colors = matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", [])
    return [to_hex(c) for c in colors]


def color(i, override=None):
    """Palette color by cycle index, honoring the active mode.

    The index wraps around the color source: a palette or property cycle
    holds only a handful of colors, while a caller can hand a primitive an
    unbounded number of tracks or datasets, so index `i` resolves to
    `i % len(source)` rather than running off the end.

    With a hwostyle mode active, the source is the mode's palette, as
    always. With no mode active, the source follows the matplotlib
    environment the caller is already in, the same call-time philosophy
    `neutral()` uses for rcParams: if `axes.prop_cycle` has been
    customized (via `plt.style.use(...)`, a direct rcParams edit, and so
    on), those colors are used. Only when the property cycle is still
    matplotlib's untouched factory default does this fall back to the
    hwostyle light palette -- "zero style" on bare matplotlib should still
    produce brand-quality colors, and matplotlib's default `C0`-`C9` cycle
    is banned by the brand rules even in this fallback case. A customized
    cycle that carries no colors at all (e.g. one that only cycles
    `linestyle` or `marker`, or a `color` key explicitly set to an empty
    list) expresses no color preference either, so it takes the same
    light-palette fallback as the factory default.

    Args:
        i: Index into the active color source. Indices at or past the
            source's length wrap back to its start.
        override: A color; wins when not None.

    Returns:
        A hex color string.
    """
    if override is not None:
        return override
    if hwostyle.current_mode():
        source = hwostyle.palette
        return source[i % len(source)]
    if _factory_default_cycle():
        source = _light_palette()
        return source[i % len(source)]
    source = _cycle_colors() or _light_palette()
    return source[i % len(source)]


def neutral(level):
    """Neutral tone `level` of the way from the background to the text color.

    Scenery that is not data -- a beam envelope, an inner-working-angle
    disk, a central-star marker -- wants a tone that reads against the plot
    background rather than a palette color. Both ends of the ramp come from
    `matplotlib.rcParams` at call time (`axes.facecolor` and `text.color`),
    so the tone inverts with the mode instead of staying a fixed gray that
    vanishes on a dark background. In light mode, where the background is
    white and the text black, `level` reproduces the gray of value
    `1 - level` exactly. The named neutral ranks (``"hardware"``,
    ``"text"``, ...) are fixed levels on this ramp; `role` takes a rank by
    name and returns its tone as a hex string.

    Args:
        level: Position on the ramp, from 0.0 (the background itself) to
            1.0 (the text color).

    Returns:
        An `(r, g, b)` tuple of floats.

    Example::

        import eyepiece as ep
        ax.axhline(0, color=ep.neutral(0.5))
    """
    back = to_rgb(matplotlib.rcParams["axes.facecolor"])
    front = to_rgb(matplotlib.rcParams["text.color"])
    frac = float(level)
    return tuple(b + (f - b) * frac for b, f in zip(back, front, strict=True))


# Padding of the backing box around a label, as a fraction of its font size.
BACKING_PAD = 0.15


def backing(alpha=0.85):
    """A `bbox` dict for a label that sits over marks or pixels.

    A rounded box in the plot background color, mostly opaque and without
    an edge, so a label written on top of a line, a band, or an image reads
    without a stroked halo around its glyphs. The color is
    `rcParams["axes.facecolor"]` at call time, so the box follows the mode.

    Args:
        alpha: Opacity of the box face.

    Returns:
        A dict for the `bbox` argument of `ax.text` or `Text.set_bbox`.

    Example::

        import eyepiece as ep
        ax.text(0.5, 0.5, "reference", bbox=ep.backing())
    """
    face = to_rgba(matplotlib.rcParams["axes.facecolor"], alpha)
    return {
        "boxstyle": f"round,pad={BACKING_PAD},rounding_size=0.25",
        "facecolor": face,
        "edgecolor": "none",
    }


def quiet_spines(ax, *, keep=("left", "bottom"), level=None):
    """Keep selected axes spines and optionally soften their neutral tone.

    This styles one axes without changing rcParams. A repeated call gives
    the same visibility, including restoring any kept spine previously hidden.

    Args:
        ax: The axes to style.
        keep: Names of spines to show. The others are hidden.
        level: None preserves kept colors; a value in [0, 1] uses `neutral`.

    Example::

        ep.quiet_spines(ax, level=0.4)
    """
    keep = set(keep)
    unknown = keep.difference(ax.spines)
    if unknown:
        raise ValueError(f"unknown spine names: {sorted(unknown)}")
    if level is not None and not 0 <= float(level) <= 1:
        raise ValueError("spine level must be in [0, 1]")
    color = None if level is None else neutral(level)
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)
        if name in keep and color is not None:
            spine.set_color(color)
