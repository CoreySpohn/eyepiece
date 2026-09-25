# Manim and slides

`eyepiece.manim` renders the same prepared views and sequences as
`eyepiece.mpl`, as native Manim Community objects for talks and explainer
videos. A consumer adds the returned mobject to an ordinary Manim `Scene`, or
to a Manim Slides `Slide`, and plays the returned animations there; eyepiece
provides no scene class, slide scheduler, or deck template of its own.

The code on this page is shown rather than executed. Rendering Manim media
needs system libraries a documentation builder does not have, and it is
exercised by the library's render tests instead.

## Installation

```bash
pip install "eyepiece[manim]"    # Manim Community, Cairo renderer
pip install "eyepiece[slides]"   # Manim plus Manim Slides
```

Manim needs system Cairo, Pango, pkg-config, and ffmpeg. The renderer uses
no TeX: every label, tick, and colorbar bound is set with Pango text. A
Matplotlib mathtext label such as `$\lambda/D$` therefore appears literally,
so a view meant for both renderers spells its labels in plain text or
Unicode.
Importing `eyepiece`, `eyepiece.prepared`, or `eyepiece.mpl` never imports
Manim or Manim Slides, and `eyepiece.manim` never imports Manim Slides.
Importing `eyepiece.manim` without Manim installed raises an `ImportError`
naming the extra.

## Rendering a view

`render(view, *, cast=None, profile=None)` returns a `ManimResult`. Its
`mobject` is a Manim `Group` centered at the origin, holding raster images
and vector marks together; move and scale it like any mobject.

```python
import eyepiece.manim as em
from eyepiece.style import SourceCast, snapshot_profile

CAST = SourceCast(["annulus"])
TALK = snapshot_profile()  # 24 pt text and 1.5 pt strokes by default

panel = em.render(sequence.frame(0), cast=CAST, profile=TALK)
panel.mobject.scale(0.9).to_edge(manim.LEFT)
```

A profile's `text_size_pt` is Manim's `font_size` and its `stroke_width_pt`
is Manim's `stroke_width`. The Matplotlib renderer defaults to paper sizes
from the rc settings, so a talk and a paper normally take two profiles of the
same mode, or of two modes, with the same cast.

`result.parts` maps element IDs to mobjects:

| Key | Mobject |
|---|---|
| an `ImageView` ID | the `ImageMobject` of its pixels |
| a `CurveView` or `TrackView` ID | the panel `Group` |
| a `PanelGroup` ID | the arranged `Group` |
| a `Path` or `ReferenceLine` ID | a `VMobject` |
| a `Region` ID | a `VMobject`; an annulus has a hole |
| a `Points` ID | a `VGroup` with one marker per sample |
| a `Label` ID | a `VGroup` wrapper whose single child holds the glyph outlines |
| `"<view id>/frame"` | the axes box, whose corners are the axis limits |
| `"<view id>/axes"` | the tick marks, tick labels, and axis labels |
| `"<image id>/colorbar"` | the colormap strip, its bounds, and its quantity |
| `"<points id>/xerr"`, `"<points id>/yerr"` | one error-bar segment per sample |

The `frame` and `axes` keys exist only in the Manim renderer.
`result.panels` maps each leaf view's ID to its panel `Group`.

Every mark is placed through its panel's frame. An update reads the frame's
current corners, so a panel the caller has moved or uniformly scaled keeps
the same data-coordinate mapping. The display conventions match the
Matplotlib renderer: row 0 of an image is its bottom row, `x_reverse` flips
the display once, a nonfinite coordinate row is a gap whose marker, error
bar, or path segment has no points, a region on an image is an outline, and
a region on a curve or track panel is shaded. Tick labels are fixed point
when the largest tick magnitude lies between `1e-5` and `1e6`; outside that
range every tick on the axis shares one power of ten written after its
mantissa, such as `2.5e-13`. `show_ticks=False` hides the tick labels and
keeps the tick marks.

### Background and fonts

A Manim scene has a black background unless told otherwise, and a profile
taken outside dark mode has dark text that vanishes against it. Set the
background from the profile the scene renders with:

```python
import manim
from matplotlib.colors import to_hex


class SpeckleScene(manim.Scene):
    def construct(self):
        self.camera.background_color = manim.ManimColor(to_hex(TALK.background_color))
        ...
```

Manim lays text out through Pango, which resolves `profile.font_family`
independently of Matplotlib. Check that the family is installed where the
scene renders, pass `font_family=` to `snapshot_profile`, or register a font
file with `manimpango.register_font` before rendering.

### Updates and ownership

`result.update(view)` accepts a new state of the same tree under the same
topology rules as `MplResult.update`, validates the whole view, and computes
every new pixel buffer, geometry array, and label outline before any mobject
changes. A rejected update leaves every part exactly as it was. Every part
keeps its identity: geometry is rewritten in place, and a label's glyph
outlines are rewritten inside its stable wrapper, so a scene that is playing
redraws them.

Opacity belongs to the consumer. An update never sets a part's opacity, so a
part hidden before a slide begins stays hidden through seeking, playback,
moving, and scaling, until the consumer reveals it. A new path weight
rescales the path's current opacity rather than replacing it.

```python
annulus = clip.parts["annulus"]
annulus.set_stroke(opacity=0.0)  # hidden until its beat
...
self.play(annulus.animate.set_stroke(opacity=1.0))
```

Adding or removing an element needs a new render; changing values, or the
visible interval of an existing path, does not.

## Playing a sequence

`animate(sequence, *, cast=None, profile=None)` returns a `ManimClip`
showing frame 0. Its `mobject` and `parts` are its result's, `seek(t)` shows
the left sample-and-hold state at physical time `t` and returns the `Sample`,
and `playback(run_time)` returns a `manim.Animation` over the whole sequence.

```python
clip = em.animate(sequence, cast=CAST, profile=TALK)
self.add(clip.mobject)
self.play(clip.playback(run_time=8))
self.wait(1)
```

Playback follows the sequence's shared schedule at the scene frame rate:
`N = ceil(run_time * fps)` physical times from the first sample through the
last, one per encoded frame, so encoded frame k shows schedule time k and the
last encoded frame is the last sample. The same schedule drives
`eyepiece.mpl.animate`, so a movie and a slide at the same duration and frame
rate show the same samples. Every playback begins from schedule time 0, which
makes a replay deterministic, and a sequence that is not periodic shows its
replay boundary as a jump rather than a blend.

Pass the duration to `playback`, never to `play`. `Scene.play` applies its
own keywords to each animation, so `self.play(clip.playback(8), run_time=4)`
would change the quantized duration; the playback detects this and raises a
`ValueError` when it begins.

The sequence lives on the clip, not on any mobject, so Manim's copies of the
mobject never copy the cube. No scientific calculation happens after
`animate` returns: playback and seeking index prepared arrays and rewrite
mobjects.

Render scenes that use clips with Manim's caching disabled, through
`--disable_caching` or `config.disable_caching = True`. Manim's partial-movie
hash walks each animation's attributes, so it would hash the whole prepared
sequence on every `play`, and it summarizes array subclasses such as memory
maps from a truncated copy. Cache invalidation for these clips is untested.

## Slides

Manim Slides plays prerendered segments, and a deck uses its `Slide` class
directly. Options given to `next_slide` configure the segment that follows
the call, so a looping playback is enclosed between an introduction and a
held state:

```python
import manim
from manim_slides import Slide


class SpeckleTalk(Slide):
    def construct(self):
        self.camera.background_color = manim.ManimColor(to_hex(TALK.background_color))
        clip = em.animate(sequence, cast=CAST, profile=TALK)
        self.play(manim.FadeIn(clip.mobject))
        self.next_slide(loop=True, notes="The halo decorrelates as it drifts.")
        self.play(clip.playback(run_time=8))
        self.next_slide(notes="Hold the final state and read the trace.")
        self.wait(1)
```

```bash
manim-slides render --disable_caching talk.py SpeckleTalk
manim-slides present SpeckleTalk
manim-slides convert SpeckleTalk talk.html --one-file --offline
```

`--one-file` embeds the videos and `--offline` stores the presentation
framework's remote assets locally; a single portable HTML file needs both.
Test the moved file in a browser with networking blocked before relying on
it. Native presentation needs a Qt binding, which `manim-slides[pyside6]`
provides.

Build each reusable piece of a deck as an ordinary function that takes the
scene, the prepared inputs, the cast, and the profile, and have every scene
import the same cast declaration. Visual identity then holds across
independently rendered scenes, while object identity holds within each one.

## Driving native Manim from a scientific model

Prepared views are optional. A consumer that wants its own geometry can
evaluate a scientific model at each playback time and distribute that one
state to native mobjects with Manim's own update mechanisms. This code lives
in the consumer's script, or in the scientific library that owns the model,
and imports nothing from eyepiece:

```python
import manim
import numpy as np

UNITS_PER_MAS = 0.02


def sky_position_mas(t_days):
    """A circular orbit as seen on the sky, standing in for a model object."""
    phase = 2.0 * np.pi * t_days / 540.0
    return 120.0 * np.cos(phase), 80.0 * np.sin(phase)


class DirectOrbit(manim.Scene):
    def construct(self):
        star = manim.Dot(manim.ORIGIN, radius=0.1)
        planet = manim.Dot(radius=0.07)
        radius = manim.Line(manim.ORIGIN, manim.ORIGIN)
        marks = manim.VGroup(planet, radius)

        def update(group, alpha):
            planet_mob, radius_mob = group
            # One model evaluation feeds every mark, so they share one instant.
            ra_mas, dec_mas = sky_position_mas(alpha * 540.0)
            # Right ascension increases to the left.
            point = np.array([-ra_mas, dec_mas, 0.0]) * UNITS_PER_MAS
            planet_mob.move_to(point)
            radius_mob.put_start_and_end_on(manim.ORIGIN, point)

        self.add(star, planet, radius)
        self.play(
            manim.UpdateFromAlphaFunc(marks, update),
            rate_func=manim.linear,
            run_time=4,
        )
```

Here the consumer owns every convention a prepared renderer would otherwise
apply, such as the direction of right ascension and the mapping to scene
units. Cheap deterministic evaluations can run during rendering; an expensive
one is sampled ahead of time and indexed by `alpha`. When such a scene sits
beside a prepared panel, both should read the same sampled time, so the two
never show two different instants.

## Testing

The render tests skip when Manim is not installed. Setting
`EYEPIECE_REQUIRE_MANIM=1` makes Manim import at session start and turns any
skipped Manim test into a failed session, which is how a render job proves
those tests ran:

```bash
EYEPIECE_REQUIRE_MANIM=1 pytest tests -rs
```
