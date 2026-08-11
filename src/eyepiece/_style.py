"""Call-time resolution of hwostyle state.

Never bind `cmaps` or `palette` by name off of the hwostyle module:
hwostyle.use() rebinds those module globals, so a captured reference
silently freezes the mode a caller saw at import time. Every lookup here
goes through the `hwostyle` module attribute at call time, so a later
`hwostyle.use()` switch is always seen.
"""

import hwostyle
import matplotlib
from matplotlib.colors import to_hex, to_rgb

_LIGHT_CMAPS = None
_LIGHT_PALETTE = None


def mode():
    """Active hwostyle mode, or "light" when no mode has been activated."""
    return hwostyle.current_mode() or "light"


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
        override: A matplotlib Colormap or registered cmap name; wins when
            not None.

    Returns:
        A matplotlib Colormap instance.
    """
    if override is not None:
        return _as_colormap(override)
    source = hwostyle.cmaps if hwostyle.current_mode() else _light_cmaps()
    return _as_colormap(getattr(source, name))


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
    return [to_hex(c) for c in matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]]


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
    is banned by the brand rules even in this fallback case.

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
    source = _cycle_colors()
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
    `1 - level` exactly.

    Args:
        level: Position on the ramp, from 0.0 (the background itself) to
            1.0 (the text color).

    Returns:
        An `(r, g, b)` tuple of floats.
    """
    back = to_rgb(matplotlib.rcParams["axes.facecolor"])
    front = to_rgb(matplotlib.rcParams["text.color"])
    frac = float(level)
    return tuple(b + (f - b) * frac for b, f in zip(back, front, strict=True))
