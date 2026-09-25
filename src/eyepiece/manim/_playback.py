"""Deterministic playback of a prepared `Sequence` through a Manim scene.

A `ManimClip` pairs a rendered `ManimResult` with the `Sequence` it shows.
The sequence lives on the clip, never on a mobject, so Manim's deep copies
of the clip's mobject (`copy()`, the starting state of other animations)
never duplicate the sequence storage.

`ManimClip.playback(run_time)` returns a `manim.Animation` over the shared
output schedule, `sequence.schedule(run_time=run_time, fps=fps)` at the
scene frame rate: N physical times from the first sample through the last.
The animation lasts exactly N frames, and encoded frame k shows schedule
time k, so the final encoded frame is the final physical sample, not a
state reached only after the last frame is written. Its rate function is
linear, and every `begin` redraws schedule time 0, so a replay (a looping
slide, a second `play`) always starts from the same state.
"""

import manim
import numpy as np

from eyepiece.manim._render import render
from eyepiece.prepared import Sequence

# Manim's scene loop writes one frame at every t in
# `numpy.arange(0, run_time, 1 / fps)`, and hands the animation
# alpha = t / run_time. With run_time = N / fps exactly, float rounding
# can add an (N + 1)th tick and can put alpha * N just below an integer.
# Running for N / fps shortened by this relative margin always gives
# exactly N ticks, and tick k lands at alpha * N = k (1 + ~1e-9), which a
# floor maps to k; `_TICK_EPSILON` absorbs the remaining float noise.
_TICK_MARGIN = 1e-9
_TICK_EPSILON = 1e-6


class ManimClip:
    """A rendered sequence: its result, mobject, parts, and the sequence.

    Attributes:
        result: The `ManimResult` every playback and seek updates.
        sequence: The prepared `Sequence` being shown.
    """

    def __init__(self, result, sequence):
        """Bind a result to its sequence; built by `animate`, not directly."""
        self.result = result
        self.sequence = sequence

    @property
    def mobject(self):
        """The root mobject (the result's), to add to a scene."""
        return self.result.mobject

    @property
    def parts(self):
        """The result's parts by element ID."""
        return self.result.parts

    def seek(self, physical_time):
        """Show the sequence's left sample-and-hold state at `physical_time`.

        Args:
            physical_time: A physical time in the sequence's time unit;
                clamped to the sampled range like `Sequence.at`.

        Returns:
            The `Sample` now displayed.
        """
        sample = self.sequence.at(physical_time)
        self.result.update(sample.view)
        return sample

    def playback(self, run_time):
        """An animation playing the whole sequence at the scene frame rate.

        Args:
            run_time: Presentation duration in seconds, quantized up to a
                whole number of output frames at `manim.config.frame_rate`.

        Returns:
            A `manim.Animation` over this clip's mobject.
        """
        return _SequencePlayback(self, run_time)


class _SequencePlayback(manim.Animation):
    """Maps each rendered output tick to its schedule index; see module doc."""

    def __init__(self, clip, run_time):
        fps = float(manim.config.frame_rate)
        output_times = clip.sequence.schedule(run_time=run_time, fps=fps)
        count = len(output_times)
        duration = count / fps
        if count > 1:
            duration *= 1.0 - _TICK_MARGIN
        super().__init__(clip.mobject, run_time=duration, rate_func=manim.linear)
        self._clip = clip
        self._output_times = output_times

    def create_starting_mobject(self):
        # Every frame is replayed from the sequence, never interpolated from
        # a starting copy, so no copy of the clip's mobject is made.
        return self.mobject

    def interpolate_mobject(self, alpha):
        alpha = min(max(float(alpha), 0.0), 1.0)
        count = len(self._output_times)
        index = min(count - 1, int(np.floor(alpha * count + _TICK_EPSILON)))
        self._clip.seek(float(self._output_times[index]))


def animate(sequence, *, cast=None, profile=None):
    """Render a prepared `Sequence` for replayable Manim playback.

    The clip shows frame 0 until played or seeked. No scientific
    calculation happens after this call: playback and seeking only index
    the sequence's prepared arrays and update the rendered mobjects.

    Args:
        sequence: A prepared `Sequence`.
        cast: As for `render`.
        profile: As for `render`; snapshotted once when None.

    Returns:
        A `ManimClip`.

    Raises:
        TypeError: If `sequence` is not a `Sequence`.
    """
    if not isinstance(sequence, Sequence):
        raise TypeError(
            f"animate takes a prepared Sequence, got {type(sequence).__name__}"
        )
    result = render(sequence.frame(0), cast=cast, profile=profile)
    return ManimClip(result, sequence)
