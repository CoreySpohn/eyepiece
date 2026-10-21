"""Scheduling, progress shapes, and timing records without Matplotlib.

Example::

    import eyepiece as ep

    tl = ep.clock.Timeline()

Exports load on first access and preserve their implementation objects.
"""

from typing import TYPE_CHECKING

from eyepiece import _LAZY_EXPORTS, _lazy_lookup

if TYPE_CHECKING:
    from eyepiece._clock import (
        Event,
        Fault,
        Plan,
        Span,
        Time,
        Timeline,
    )
    from eyepiece._motion import (
        Tempo,
        ease,
        frame_count,
        pulse_shape,
        reading_words,
        stagger,
    )

__all__ = [
    "Event",
    "Fault",
    "Plan",
    "Span",
    "Tempo",
    "Time",
    "Timeline",
    "ease",
    "frame_count",
    "pulse_shape",
    "reading_words",
    "stagger",
]

_EXPORTS = {name: _LAZY_EXPORTS[name] for name in __all__}


def __getattr__(name):
    """Load one public task export on first access."""
    return _lazy_lookup(name, _EXPORTS, globals())


def __dir__():
    """Include lazy task exports in discovery."""
    return sorted(set(globals()) | set(__all__))
