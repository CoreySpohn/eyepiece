---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Write a viz module

Put plotting for library types in a `viz.py` module with an optional plotting
dependency and imports inside the function.

```python
# mylib/viz.py
def plot_psf(psf_or_array, *, ax=None, imshow_kw=None):
    """Draw a PSF array or an object exposing its intensity array.

    Args:
        psf_or_array: Intensity array or object with an intensity attribute.
        ax: Optional destination axes.
        imshow_kw: Optional image drawing properties.

    Returns:
        The eyepiece PlotResult for the image.
    """
    import eyepiece as ep

    array = getattr(psf_or_array, "intensity", psf_or_array)
    return ep.imshow_log(array, ax=ax, imshow_kw=imshow_kw)
```

```python
# In a consumer script after installing mylib with its viz extra:
import hwostyle
from mylib import viz

hwostyle.use("dark")
result = viz.plot_psf([[0.1, 1.0], [0.01, 0.1]])
```

Keep style activation in the consumer, accept bare arrays and caller axes,
and return the primitive result so the caller can add marks.
Declare eyepiece in the library `viz` extra and keep plotting imports out of
the package initializer.
See {doc}`../viz-convention` for packaging and preparation rules and
{doc}`the image catalog <../gallery/images>` for image primitives.
