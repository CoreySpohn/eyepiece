"""One-time layout of direct curve labels in display space."""

import math

import numpy as np
from matplotlib.text import Annotation, Text

from eyepiece._result import PlotResult


def spread_labels(labels, *, min_gap=2.0):
    """Separate curve-end labels vertically with the least squared movement.

    Call once after setting the axes limits and figure layout. Labels keep
    their vertical order and horizontal position. Their rendered boxes stay
    inside the vertical axes bounds with at least `min_gap` between edges.
    Nothing moves if the labels already fit. This is a still layout helper,
    so an animation does not recompute it on every frame.

    Args:
        labels: Nonempty iterable of plain Text artists on one axes.
        min_gap: Minimum vertical edge-to-edge gap in points, nonnegative.

    Returns:
        A PlotResult with the original labels in ``artists["text"]``.

    Raises:
        ValueError: If labels span axes, the gap is invalid, or they cannot fit.
        TypeError: If a label is not plain Text (annotations are unsupported).

    Example::

        labels = [ax.text(0.9, y, name, va="center")
                  for y, name in [(0.5, "a"), (0.51, "b"), (0.52, "c")]]
        ep.spread_labels(labels, min_gap=3)
    """
    labels = list(labels)
    gap = float(min_gap)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("min_gap must be finite and nonnegative")
    if not labels:
        raise ValueError("spread_labels needs at least one label")
    if any(not isinstance(t, Text) or isinstance(t, Annotation) for t in labels):
        raise TypeError("spread_labels needs plain Text artists")
    ax = labels[0].axes
    if ax is None or any(t.axes is not ax for t in labels):
        raise ValueError("spread_labels needs labels on one axes")
    fig = ax.get_figure(root=True)
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in labels]
    order = sorted(range(len(labels)), key=lambda i: (boxes[i].y0 + boxes[i].y1) / 2)
    heights = np.array([boxes[i].height for i in order])
    centers = np.array([(boxes[i].y0 + boxes[i].y1) / 2 for i in order])
    gap_px = gap * fig.dpi / 72
    offsets = np.r_[0, np.cumsum((heights[:-1] + heights[1:]) / 2 + gap_px)]
    low = ax.bbox.y0 + heights[0] / 2
    high = ax.bbox.y1 - heights[-1] / 2 - offsets[-1]
    if high < low:
        raise ValueError("labels and min_gap do not fit inside the axes")
    # Pool adjacent violating blocks: the least-squares isotonic projection.
    blocks = []
    for i, value in enumerate(centers - offsets):
        blocks.append([i, i + 1, float(value), 1])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            b = blocks.pop()
            a = blocks.pop()
            weight = a[3] + b[3]
            blocks.append([a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / weight, weight])
    shifted = np.empty(len(labels))
    for start, end, mean, _ in blocks:
        shifted[start:end] = np.clip(mean, low, high)
    targets = shifted + offsets
    for i, delta in zip(order, targets - centers, strict=True):
        label = labels[i]
        transform = label.get_transform()
        position = transform.transform(label.get_unitless_position())
        label.set_position(
            transform.inverted().transform(position + np.array([0, delta]))
        )
    return PlotResult(ax=ax, artists={"text": labels})
