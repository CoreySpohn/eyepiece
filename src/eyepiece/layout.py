"""Axis labels and coordinated source styles."""

from eyepiece import _style

_MARKERS = ("o", "s", "D", "^", "v", "P")


def label_lod(ax):
    """Label both axes of `ax` in lambda/D units.

    Args:
        ax: A matplotlib Axes.
    """
    ax.set_xlabel(r"$x$ [$\lambda/D$]")
    ax.set_ylabel(r"$y$ [$\lambda/D$]")


def label_arcsec(ax):
    """Label both axes of `ax` in arcseconds.

    Args:
        ax: A matplotlib Axes.
    """
    ax.set_xlabel(r"$x$ [arcsec]")
    ax.set_ylabel(r"$y$ [arcsec]")


def label_au(ax):
    """Label both axes of `ax` in AU.

    Args:
        ax: A matplotlib Axes.
    """
    ax.set_xlabel(r"$x$ [AU]")
    ax.set_ylabel(r"$y$ [AU]")


class SourceStyles:
    """Stable per-source color and marker assignment.

    Each name is assigned a color from the active palette
    (``_style.color(i)``, in declaration order) and a marker cycling
    through a fixed sequence, so a caller that hands the same
    ``SourceStyles`` to several panels draws a given source identically in
    every one of them. Colors resolve through ``_style`` at construction
    time, so two instances built from the same names -- even across
    separate calls -- agree exactly; nothing here depends on set or dict
    iteration order or any other source of nondeterminism.

    This is the Matplotlib-side assignment. The prepared-view renderers take
    an `eyepiece.style.SourceCast`, which fixes the same slots and markers
    but leaves the color to a `RenderProfile` at render time.

    Args:
        names: Source names, in the order colors and markers are
            assigned.

    Example::

        styles = SourceStyles(["star", "b", "c"])
        ax.scatter(x, y, color=styles["b"]["color"], marker=styles["b"]["marker"])
    """

    def __init__(self, names):
        """Assign each name its color and marker, in declaration order.

        Args:
            names: Source names, in the order colors and markers are
                assigned.
        """
        self._styles = {
            name: {"color": _style.color(i), "marker": _MARKERS[i % len(_MARKERS)]}
            for i, name in enumerate(names)
        }

    def __getitem__(self, name):
        """Style dict for `name`.

        Args:
            name: A name passed to the constructor.

        Returns:
            A `{"color": str, "marker": str}` dict.
        """
        return self._styles[name]
