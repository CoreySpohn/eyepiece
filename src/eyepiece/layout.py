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
"""

import numpy as np


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
