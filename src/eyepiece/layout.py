"""Pixel-edge extent helpers and matching axis labelers.

matplotlib's ``imshow(..., extent=(left, right, bottom, top))`` places pixel
CENTERS at integer-spaced coordinates by default, so the array's outer edge
sits half a pixel beyond the outermost sample. Every extent helper here
returns that pixel-EDGE extent, not the center-to-center span: a 4-pixel
axis at a 1.0 pixel scale covers ``(-2.0, 2.0)``, not ``(-1.5, 1.5)``, because
the edge of pixel 0 is half a step below its center and the edge of pixel 3
is half a step above its. Get this wrong and a figure is subtly
misregistered -- the data lines up with the wrong tick.

Two extent families are provided:

- ``extent_lod`` / ``extent_lod_from_pixels`` work in units of lambda/D and
  depend only on numpy.
- ``extent_arcsec`` / ``extent_au`` convert through the optional
  ``hwoutils`` dependency (the ``eyepiece[hwo]`` extra); import it lazily
  inside the function body so plain ``import eyepiece`` never pulls in jax.

``SourceStyles`` and ``Frame`` are the coordination-as-data mechanisms for
the library's driving use case: several panels of the same scene (a
trajectory, a detector image, a spectrum) staying visually consistent
because the caller passes the same small object to each primitive, rather
than each panel re-deriving its own colors or extent.
"""

from dataclasses import dataclass, field

import numpy as np

from eyepiece import _style

_MARKERS = ("o", "s", "D", "^", "v", "P")


def extent_lod(u_lod):
    """Pixel-edge extent from a vector of pixel-center coordinates.

    Args:
        u_lod: 1D array of pixel-center coordinates, evenly spaced, in
            lambda/D.

    Returns:
        A ``(left, right, bottom, top)`` tuple of floats padded half a step
        beyond the first and last centers.
    """
    u_lod = np.asarray(u_lod, dtype=float)
    step = u_lod[1] - u_lod[0]
    half = step / 2.0
    left = float(u_lod[0] - half)
    right = float(u_lod[-1] + half)
    return (left, right, left, right)


def extent_lod_from_pixels(n_pix, pixscale_lod):
    """Pixel-edge extent from a pixel count and scale, centered on zero.

    Args:
        n_pix: Number of pixels along the axis.
        pixscale_lod: Pixel scale in lambda/D per pixel.

    Returns:
        A ``(left, right, bottom, top)`` tuple of floats.
    """
    half_width = n_pix * pixscale_lod / 2.0
    return (-half_width, half_width, -half_width, half_width)


def extent_arcsec(n_pix, pixscale_mas):
    """Pixel-edge extent in arcseconds from a pixel count and mas scale.

    Args:
        n_pix: Number of pixels along the axis.
        pixscale_mas: Pixel scale in milliarcseconds per pixel.

    Returns:
        A ``(left, right, bottom, top)`` tuple of floats, in arcseconds.

    Raises:
        ImportError: If ``hwoutils`` is not installed. Install the
            ``eyepiece[hwo]`` extra to enable this function.
    """
    try:
        from hwoutils import conversions
    except ImportError:
        raise ImportError(
            "extent_arcsec needs hwoutils: pip install eyepiece[hwo]"
        ) from None
    half_width_mas = n_pix * pixscale_mas / 2.0
    half_width = float(conversions.mas_to_arcsec(half_width_mas))
    return (-half_width, half_width, -half_width, half_width)


def extent_au(n_pix, pixscale_mas, distance_pc):
    """Pixel-edge extent in AU from a pixel count, mas scale, and distance.

    Args:
        n_pix: Number of pixels along the axis.
        pixscale_mas: Pixel scale in milliarcseconds per pixel.
        distance_pc: Distance to the target system in parsecs.

    Returns:
        A ``(left, right, bottom, top)`` tuple of floats, in AU.

    Raises:
        ImportError: If ``hwoutils`` is not installed. Install the
            ``eyepiece[hwo]`` extra to enable this function.
    """
    try:
        from hwoutils import conversions
    except ImportError:
        raise ImportError(
            "extent_au needs hwoutils: pip install eyepiece[hwo]"
        ) from None
    half_width_mas = n_pix * pixscale_mas / 2.0
    half_width_arcsec = conversions.mas_to_arcsec(half_width_mas)
    half_width = float(conversions.arcsec_to_au(half_width_arcsec, distance_pc))
    return (-half_width, half_width, -half_width, half_width)


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


@dataclass(frozen=True)
class Frame:
    """Shared field-of-view extent for panels of the same scene.

    Several panels of one scene (a trajectory view, a detector image, a
    schematic) each call their own extent method against the same
    `Frame`, so a single field-of-view choice stays visually consistent
    between them instead of each panel deriving its own half-width.

    Attributes:
        half_fov_lod: Half the field of view, in lambda/D, measured to the
            pixel edge -- the same edge convention `extent_lod_from_pixels`
            uses, not a pixel center.
        pixscale_lod: Optional pixel scale in lambda/D per pixel, carried
            for callers that need to map an array index onto this frame.
            Not used by `extent_lod` or `extent_arcsec` themselves.
    """

    half_fov_lod: float
    pixscale_lod: float | None = field(default=None, kw_only=True)

    def extent_lod(self):
        """Pixel-edge extent in lambda/D.

        Returns:
            A `(left, right, bottom, top)` tuple of floats:
            `(-half_fov_lod, half_fov_lod, -half_fov_lod, half_fov_lod)`.
        """
        half = float(self.half_fov_lod)
        return (-half, half, -half, half)

    def extent_arcsec(self, wavelength_nm, diameter_m):
        """Pixel-edge extent in arcseconds at a wavelength and diameter.

        Args:
            wavelength_nm: Wavelength in nanometers.
            diameter_m: Telescope diameter in meters.

        Returns:
            A `(left, right, bottom, top)` tuple of floats, in arcseconds.

        Raises:
            ImportError: If `hwoutils` is not installed. Install the
                `eyepiece[hwo]` extra to enable this method.
        """
        try:
            from hwoutils import conversions
        except ImportError:
            raise ImportError(
                "Frame.extent_arcsec needs hwoutils: pip install eyepiece[hwo]"
            ) from None
        half = float(
            conversions.lambda_d_to_arcsec(self.half_fov_lod, wavelength_nm, diameter_m)
        )
        return (-half, half, -half, half)
