# eyepiece

Plotting and animation primitives for astronomical imaging simulations.

## What eyepiece is

`eyepiece` provides ax-first, stateless figure functions for the plots that
recur across direct-imaging work: log-scaled and diverging image displays,
side-by-side and A/B comparison panels, radial profiles and contrast curves,
corner plots, orbit and sky-track scenes, optical-train schematics, and a
multi-sink animation recorder that draws each frame once and writes it to any
combination of gif, mp4, and html. Every primitive takes plain arrays in and
returns a small result object holding the axes and the artists it drew, so
callers can update or extend a figure without re-deriving which line or
image object came from where.

On top of the figures sits a layer for explanatory animation. A `Timeline`
lays a clip out in seconds and resolves it into a per-frame plan, the moves
(draw-ons, pulses, callouts, staged dimming, camera paths, equation
transforms) are driven by that plan, `eyepiece.narration` places spoken
lines and their cue words on the same clock, `eyepiece.testing` checks the
finished clip for motion, readability, color-meaning and timing faults, and
`eyepiece.render` renders its venue, mode and narration variants, each
with parallel frame workers.

For a scene that has to appear more than once, eyepiece also offers prepared
views: plain records of images, curves, tracks, and their marks, carrying
borrowed arrays, units, masks, display scales, and stable IDs. A simulation
library prepares a scene once, and the same records render as a paper still
or a movie through `eyepiece.mpl`, and as native Manim objects for a talk
through `eyepiece.manim`, without recomputing any science.

Figure style (color, colormaps, and mode) is resolved through
[hwostyle](https://github.com/HabitableWorldsObservatory/hwostyle) at call
time, not at import time. `import eyepiece` never activates a style or
touches matplotlib's rcParams. Without an active hwostyle mode, primitives
fall back to a fixed light-mode palette, so eyepiece also works as a
standalone plotting library.

## What eyepiece is not

- **Not a simulation library.** eyepiece takes arrays and returns figures; it
  never imports a simulation, orbit, or optics package.
- **Not a style engine.** Color, colormap, and mode definitions live in
  hwostyle; eyepiece only reads them at call time.
- **Not a data pipeline.** Aggregating or transforming simulation output
  before plotting is the caller's job.

## What is in it

Use `import eyepiece as ep` everywhere, then call through a task namespace,
for example `ep.plot.imshow_log(...)` or `ep.motion.indicate(...)`.

| Namespace | Task |
|---|---|
| `ep.plot` | Still and updatable data displays, marks, and layouts |
| `ep.clock` | Scheduling, progress shapes, and timing records |
| `ep.motion` | Moves, attention, and camera geometry |
| `ep.style` | Appearance, source styles, and render profiles |
| `ep.render` | Recording, files, render workers, and provenance |
| `ep.testing` | Clip, timing, readability, and frame checks |
| `ep.narration` | Speech, captions, chapters, and cue sounds |
| `ep.motifs`, `ep.prepared`, `ep.mpl`, `ep.manim` | Recurring styles, scene records, and prepared renderers |

The root keeps ten stable shortcuts: `Timeline`, `Plan`, `Tempo`,
`PlotResult`, `MosaicResult`, `MotionContract`, `animate`, `record`,
`save_fig`, and `__version__`. All former root paths still work silently in
0.16, while star imports and discovery favor the small root. Deprecated
aliases keep their existing warnings. The
[capability catalog](https://github.com/CoreySpohn/eyepiece/blob/main/docs/reference/catalog.md)
lists every public name with its canonical path and a worked example.

### Make a still

- **Images.** `ep.plot.imshow_log`, `ep.plot.imshow_diverging` and `ep.plot.display_limits`,
  `ep.plot.compare_row` and `ep.plot.compare_grid` (several images on one norm and
  colorbar), `ep.plot.triptych` (A, B, and their ratio or residual), `ep.plot.show_field`
  (a complex field), and `ep.plot.kymograph` (a cut stacked against time or
  wavelength).
- **Marks and annotations.** `ep.plot.overlay_circle` and `ep.plot.overlay_line` (dashed
  marks that read over any pixels), `ep.plot.ruler`, `ep.plot.bracket`, `ep.plot.ref_line` (a
  threshold with its label pinned to an axes edge), `ep.plot.shade_intervals`
  (excluded intervals), `ep.plot.log_zero_floor`, and `ep.style.backing` (a plate behind a label).
- **Distributions and diagnostics.** `ep.plot.corner`, `ep.plot.corner_overlay`,
  `ep.plot.hist_vs_pdf`, `ep.plot.cov_ellipse`, `ep.plot.convergence`, `ep.plot.signed_trace`, `ep.plot.hist_fill`,
  `ep.plot.interval_points` (including one-sided intervals), `ep.plot.calibration`, `ep.plot.strip`,
  and `ep.plot.scatter_values`.
- **Profiles.** `ep.plot.plot_radial`, `ep.plot.plot_contrast_curve`, and
  `ep.plot.plot_radial_image` (under the `[hwo]` extra).
- **Scenes, schematics and phasors.** `ep.plot.trail`, `ep.plot.sky_fan`, `ep.plot.fading_track`,
  `ep.plot.step_list`; `ep.plot.rail`, `ep.plot.rail_panels`, `ep.plot.schematic` and the `ep.plot.GLYPHS`
  vocabulary for optical trains; `ep.plot.phasor`, `ep.plot.phase_ring`, and `ep.plot.curve_insets`.
- **Layout and color.** `ep.plot.extent_lod`, `ep.plot.extent_lod_from_pixels`,
  `ep.plot.extent_arcsec`, `ep.plot.extent_au`, the matching `ep.plot.label_lod`, `ep.plot.label_arcsec`,
  and `ep.plot.label_au`, `ep.plot.Frame` and `ep.style.SourceStyles` for panels of one scene, and
  `ep.style.role` (a named role or neutral rank, in any mode) and `ep.style.neutral` for
  brand and scenery colors.

### Put a clip on one clock

- **Timing.** `ep.clock.Timeline` declares named spans in seconds (`enter`, `leave`,
  `text`, `move`, `run`, `hold`, `wait_until`, `read`, `settle`,
  `together`, `stagger`, `indicate`, `circumscribe`, `click`, `caption`, `section`,
  `hero`) and `ep.clock.Timeline.at_fps` resolves them into an `ep.clock.Plan` that a draw
  function reads frame by frame (`value`, `progress`, `frame_of`, `events`,
  `spans_json`). `ep.clock.Tempo` sets the pace of the whole clip in one place.
- **Helpers.** `ep.clock.ease`, `ep.clock.stagger`, `ep.clock.Time`, `ep.clock.frame_count`, `ep.clock.reading_words`,
  and the `ep.clock.Span`, `ep.clock.Event` and `ep.testing.Fault` records a plan carries.

### Move the viewer's eye

- **Draw on.** `ep.motion.head_draw` (the signature arc-length reveal with a bright
  head), `ep.motion.draw_on`, `ep.motion.build_axes`, `ep.motion.arrow`, `ep.motion.angle_arc`, `ep.motion.compass`,
  `ep.motion.write` (a label traced as if by hand), and `ep.motion.words`.
- **Point at the subject.** `ep.motion.indicate` (a pulse that returns exactly),
  `ep.motion.circumscribe`, `ep.motion.callout`, `ep.motion.leader`, `ep.motion.link`, `ep.motion.flight`, `ep.motion.spotlight`,
  `ep.motion.brush`, `ep.motion.follow_zoom`, `ep.motion.Anchor`, and `ep.testing.audit` with its `ep.testing.AttentionFault`
  for the connector rules.
- **Dim, bring in, and settle.** `ep.motion.Stage` (groups shown, dimmed, or absent,
  at absolute levels), `ep.motion.fade`, `ep.style.blend`, `ep.motion.capture`, and `ep.motion.collapse`.
- **Land the answer.** `ep.motion.click` (ink swells, the end dot grows, the number
  sets bold, the ruler recedes), `ep.motion.ruler_readout`, and `ep.motion.readout`.
- **Equations and numbers.** `ep.motion.term_row`, `ep.motion.equation`, `ep.motion.transform` (one
  equation carried into another glyph by glyph), `ep.motion.exposure_text`,
  `ep.motion.aperture_counts`, `ep.motion.arrivals`, and `ep.motion.photons`.
- **Uncertainty.** `ep.plot.weighted_draws` (a posterior as true-scale weighted
  draws) and `ep.plot.hop` (draws played one at a time with aging ghosts and a
  running tally).
- **Cameras and morphs.** `ep.motion.PageCamera`, `ep.motion.zoom_path`, `ep.motion.view_limits`,
  `ep.motion.overview_box`, `ep.motion.warp`, `ep.motion.warp_grid`, `ep.motion.morph`, `ep.motion.quad_image`, and
  `ep.motion.pixel_quads`.

### Keep color and scale honest

- **Motifs.** `ep.motifs.Motif` and `ep.motifs.MotifSet` give each recurring quantity one
  drawing style across a clip, with `AcceptedPair` and `is_hue_role` in
  `eyepiece.motifs`.
- **Scale for a talk.** `ep.style.talk_rc` (type and strokes sized from the frame
  height, from `ep.style.TALK_FRACTIONS` and `ep.style.VIDEO_RC`), `ep.style.use_video` and
  `ep.style.video_roles` for video-safe dark colors, and `ep.style.unsnapped`.

### Narrate it (`eyepiece.narration`)

- **Script and speech.** `ep.narration.Script`, `ep.narration.Line`, `ep.narration.Pause`, `ep.narration.Cue` and `ep.narration.parse_cues`
  for cue marks on words, `ep.narration.estimate` for a draft length, `ep.narration.say` for numbers
  written for the voice, `ep.narration.speech_faults`, `ep.narration.lint_script`, and
  `ep.narration.check_script_speech`. `python -m eyepiece.narration.synth` speaks the
  lines into a cache under the `[narration]` extra.
- **On the clock.** `ep.narration.Narration` loads the measured speech and places each
  line, beat, and cue (`ep.clock.Timeline(narration=...)` and `ep.clock.Timeline.line` use
  it), with `ep.narration.check_number_sync`, `ep.narration.check_silence`, and `ep.narration.sentences`.
- **Captions, chapters and sound.** `ep.narration.captions`, `ep.narration.caption_at`,
  `ep.narration.write_captions`, `ep.narration.write_vtt`, `ep.narration.write_srt`, `ep.narration.write_chapters`, cue sounds
  (`ep.narration.CueEvents`, `ep.narration.cue_track`, `ep.narration.count_tone`, `ep.narration.pixel_ticks`, `ep.narration.click_sting`),
  `ep.narration.audio_track`, and `ep.narration.mux` for a loudness-normalized narrated mp4.

### Check the clip (`eyepiece.testing`)

`ep.testing.check_clip` re-renders a plan-driven clip without writing files and
collects its readability, motion and meaning findings, the plan's own faults,
unread knobs, and speaking pace in one `ep.testing.ClipReport`. Each check also stands
alone: `ep.testing.motion_faults` (jumps, dead moves, drift, torn groups, overlaps,
departures) against the moves a clip declares, or the `ep.motion.MotionContract` a
move's result carries,
`ep.testing.readability_faults`, `ep.testing.meaning_faults` and `ep.testing.meaning_faults_chain` (one
meaning per hue, within and across clips), `ep.testing.motif_faults` and
`ep.testing.motif_margins`, `ep.testing.timing_faults` (pace, gaps, still screens, numbers said
before they are shown), `ep.testing.speech_rate`, `ep.testing.grounding_faults` (equation terms
without a referent), and `ep.testing.motion_ink` (how much of the motion is the
quantity). The fault checks have `assert_*` twins that raise, `ep.narration.caption_line`
places captions clear of the figure's text, and `ep.testing.review_sheet` draws a
contact sheet at the cues and events that matter.

### Record and render

- **Record.** `ep.render.save_fig`, `ep.render.record` (a context manager that takes frames from
  a loop the caller drives, with `stills=` from the same pass and
  `on_rescale=` against scale drift), `ep.render.animate` and its `ep.render.Animation`,
  `ep.render.RawSink`, `ep.render.PRESETS`, `ep.render.ENCODER_PROFILE`, and `ep.render.to_scrubber` (an offline
  HTML page whose slider scrubs one knob).
- **Render in parallel.** `ep.render.ClipSpec`, `ep.render.Venue`, `ep.render.Variant` and `ep.render.variants`
  describe a clip's venue, mode, and narration variants; `ep.render.render_variants`
  renders each one keyed by `ep.render.content_key` into an `ep.render.Manifest`, resumably;
  `ep.render.render_frames` renders one clip's frames in worker processes; `ep.render.render_poster`,
  `ep.render.render_still`, and `ep.testing.same_picture` make and compare deterministic stills;
  `ep.testing.frame_faults` and `ep.testing.assert_frames_pure` check that frames do not depend
  on draw order.
- **Provenance.** `ep.render.stamp`, `ep.render.provenance_text`, `ep.render.provenance_fields`, and
  `ep.render.file_metadata`.

### Prepare a scene once

- **`eyepiece.prepared`.** The frozen records (`ImageView`, `CurveView`,
  `TrackView`, `PanelGroup`, and the `Path`, `Points`, `Region`,
  `ReferenceLine`, and `Label` marks, with `AxisSpec` and `Scale`),
  `Sequence` for replayable states over physical time with the
  `ArrayChannel`, `PathWindow`, and `Clock` channels that change them, and
  the display mapping. It imports only NumPy.
- **`eyepiece.style`.** `SourceCast` for fixed source colors and markers, and
  `snapshot_profile`, which captures the active hwostyle appearance as a
  read-only `RenderProfile`.
- **`eyepiece.mpl`.** `ep.mpl.render` and `ep.mpl.animate` for Matplotlib stills and movies.
- **`eyepiece.manim`.** `ep.manim.render` and `ep.manim.animate` for native Manim Community
  objects, usable in a Manim scene or a Manim Slides deck, with `Units` for
  point sizes and `ensure_font` for Pango fonts.

## Usage

```python
import eyepiece as ep

result = ep.plot.imshow_log(psf, extent=ep.plot.extent_lod_from_pixels(psf.shape[0], 0.5))
ep.plot.label_lod(result.ax)
ep.render.save_fig(result.fig, "psf")
```

Every primitive returns a small result object carrying the axes it drew on
and the artists it made (keyed by the common `ARTIST_KEYS` vocabulary), so a
caller can keep working on the figure without hunting for the objects again.
`imshow_log` and `imshow_diverging` also return an `.update` that redraws
with new data through the same transform, which is what makes animation a
few lines:

```python
frames = [cube[k] for k in range(len(cube))]
result = ep.plot.imshow_log(frames[0])


def draw(fig, k):
    result.update(frames[k])


ep.render.animate(result.fig, draw, len(frames), fps=10).save("run.mp4", "run.gif")
```

An explanatory clip puts every move on one clock. The timeline below draws a
curve on, pulses a marker, and holds; the plan sets every frame, the checks
run on the same draw function, and `record` writes the movie and a still
from one pass:

```python
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

plt.rcParams.update(ep.style.talk_rc(720, dpi=100))
x = np.linspace(0.0, 10.0, 200)
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100, layout="constrained")
(curve,) = ax.plot(x, np.sin(x))
(marker,) = ax.plot([7.85], [1.0], "o")
reveal = ep.motion.head_draw(curve, ease="linear")
emphasis = ep.motion.indicate(marker, scale=1.5)

tl = ep.clock.Timeline({"curve": 0.0, "emphasis": 0.0})
tl.move("draw", 2.0, to={"curve": 1.0})
tl.run("emphasize", 1.0, to={"emphasis": 1.0})
tl.hold(1.5)
plan = tl.at_fps(30)


def draw(k):
    reveal.update(plan.value(k, "curve"))
    emphasis.update(plan.value(k, "emphasis"))


ep.testing.assert_clip(fig, draw, plan, moves={curve: "draw"})
with ep.render.record(fig, "clip.mp4", fps=plan.fps, plan=plan, stills={"clip.png": "draw"}) as rec:
    for k in range(plan.n_frames):
        draw(k)
        rec.frame()
```

### Callback conventions

`check_clip` and `assert_clip` take `draw(k)`, where `k` is an integer
frame index. `animate` and render workers take `draw(fig, item)`, where
`item` comes from the supplied frame iterable. Lower-level checks such as
`motion_faults` and `readability_faults` take `at(t)`, where `t` is in
seconds. Adapt the plan-driven `draw` above as follows:

```python
def draw_frame(fig, item):
    draw(item)


def at(t):
    k = min(plan.n_frames - 1, max(0, round(t * plan.fps)))
    draw(k)


movie = ep.render.animate(fig, draw_frame, range(plan.n_frames), fps=plan.fps)
faults = ep.testing.readability_faults(fig, at=at, times=[0.0, 0.5, 1.0])
```

For render workers, create the figure and these callbacks inside the worker
builder and return `(fig, draw_frame)`. In the other direction, a frame-based
callback can be passed to `check_clip` through `lambda k: draw_frame(fig, k)`,
and a seconds-based callback through `lambda k: at(k / plan.fps)`.
`tl.move` already eases its knob, so its reveal uses `ease="linear"` to
apply that eased value once.

A prepared sequence gives a still, a strip of selected epochs, and a movie
from one preparation:

```python
import eyepiece as ep

cast = ep.style.SourceCast(["planet b"])
still = ep.mpl.render(sequence.frame(10), cast=cast)
strip = ep.mpl.render(sequence.strip([0, 10, 20]), cast=cast)
movie = ep.mpl.animate(sequence, run_time=8, fps=30, cast=cast)
movie.save("run.mp4")
```

The same sequence plays in a talk through Manim:

```python
import manim
from matplotlib.colors import to_hex

import eyepiece as ep

profile = ep.style.snapshot_profile()
clip = ep.manim.animate(sequence, cast=cast, profile=profile)
# Inside a Manim Scene or Slide; the background follows the profile so its
# text stays readable.
self.camera.background_color = manim.ManimColor(to_hex(profile.background_color))
self.add(clip.mobject)
self.play(clip.playback(run_time=8))
```

`rail` draws a miniature optical-train diagram from a plain element list, so
a physics panel can sit beside a reminder of which plane it shows:

```python
import eyepiece as ep

result = ep.plot.rail(
    [("Pupil", "pupil"), ("FPM", "fpm"), ("Lyot", "lyot"), ("Focal", "focal")],
    highlight="FPM",
)
ep.render.save_fig(result.fig, "coronagraph_rail")
```

## Status

eyepiece is pre-1.0. The figure primitives, the timeline and its moves, the
narration and check layers, the parallel render driver, and the prepared
views are implemented and tested, and the public API may still shift
between minor versions before 1.0. The
[changelog](https://github.com/CoreySpohn/eyepiece/blob/main/CHANGELOG.md)
records each change.

Deprecated in 0.16, with a warning that names
the replacement: `eyepiece.testing.Span` (now `Window`), `spans` (now
`shade_intervals`), `eyepiece.motifs.resolve_role` (now `role`, which
takes `mode=`), `Timeline.pulse` and `Timeline.ring` (now `Timeline.indicate` and
`Timeline.circumscribe`), and `PRESETS["talk_60"]` (now `PRESETS["talk"]`).
The arguments `ep.motion.write(overlap=)`, `record(allow_rescale=)` and
`ep.narration.Narration(min_s=)` are gone; use `on_rescale=` for `record` and `ends=` for
`Narration`.

## Installation

```bash
pip install eyepiece
```

Unit conversions used by a small number of layout helpers (arcsecond and AU
extents) and `plot_radial_image`'s profile computation are optional and
pull in [hwoutils](https://github.com/CoreySpohn/hwoutils):

```bash
pip install eyepiece[hwo]
```

mp4 export needs ffmpeg; `pip install "eyepiece[video]"` adds the
`imageio-ffmpeg` package, which ships a binary, if none is on PATH.

The Manim renderer is optional too. `[manim]` adds Manim Community, and
`[slides]` adds Manim Slides beside it; Manim needs system Cairo, Pango,
pkg-config, and ffmpeg:

```bash
pip install "eyepiece[manim]"
pip install "eyepiece[slides]"
```

Speech synthesis for narrated clips is the `[narration]` extra (Kokoro and
soundfile). Kokoro pulls in torch and supports Python 3.10 to 3.12, so run
the synthesis step in an environment of its own; loading cached speech,
captions, and the narration checks need only the base install. The meaning
and motif checks in `eyepiece.testing` measure color distances with
`colorspacious`, which the `[test]` extra installs.

```bash
pip install "eyepiece[narration]"
```

## License

MIT
