"""Native Manim Community (Cairo) renderer for prepared views and sequences.

`render(view)` builds a `ManimResult` (`.mobject`, `.parts`, `.update`);
`animate(sequence)` builds a `ManimClip` whose `.playback(run_time)` is a
`manim.Animation` over the sequence's shared output schedule. Both take
`eyepiece.prepared` records only, and compute no science.

Manim is an optional dependency (`pip install "eyepiece[manim]"`, which
needs system Cairo, Pango, pkg-config, and ffmpeg; no TeX). This namespace
never imports Manim Slides: a deck uses its `Slide` class directly and
plays these animations inside it.

Render scenes that use these clips with Manim's caching disabled
(`--disable_caching`, or `config.disable_caching = True`). Manim's
partial-movie hash walks each animation's attributes, so it would hash
the whole prepared sequence on every `play`, and it summarizes array
subclasses such as memory maps from a truncated copy; cache invalidation
for these clips is untested.
"""

try:
    import manim  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised in a subprocess
    raise ImportError(
        "eyepiece.manim needs Manim Community: install the optional extra "
        'with `pip install "eyepiece[manim]"` (system Cairo, Pango, '
        "pkg-config, and ffmpeg are required)"
    ) from exc

from eyepiece.manim._playback import ManimClip, animate
from eyepiece.manim._render import ManimResult, render

__all__ = ["ManimClip", "ManimResult", "animate", "render"]
