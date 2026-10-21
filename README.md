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

The groups below follow what an author sets out to do. Names without a
prefix are importable straight from `eyepiece`; the rest live in the named
submodule. The
[capability catalog](https://github.com/CoreySpohn/eyepiece/blob/main/docs/reference/catalog.md)
is the full table, one row per task, with the page that shows each name.

### Make a still

- **Images.** `imshow_log`, `imshow_diverging` and `display_limits`,
  `compare_row` and `compare_grid` (several images on one norm and
  colorbar), `triptych` (A, B, and their ratio or residual), `show_field`
  (a complex field), and `kymograph` (a cut stacked against time or
  wavelength).
- **Marks and annotations.** `overlay_circle` and `overlay_line` (dashed
  marks that read over any pixels), `ruler`, `bracket`, `ref_line` (a
  threshold with its label pinned to an axes edge), `shade_intervals`
  (excluded intervals), `log_zero_floor`, and `backing` (a plate behind a label).
- **Distributions and diagnostics.** `corner`, `corner_overlay`,
  `hist_vs_pdf`, `cov_ellipse`, `convergence`, `signed_trace`, `hist_fill`,
  `interval_points` (including one-sided intervals), `calibration`, `strip`,
  and `scatter_values`.
- **Profiles.** `plot_radial`, `plot_contrast_curve`, and
  `radial_profile_plot` (under the `[hwo]` extra).
- **Scenes, schematics and phasors.** `trail`, `sky_fan`, `fading_track`,
  `step_list`; `rail`, `rail_panels`, `schematic` and the `GLYPHS`
  vocabulary for optical trains; `phasor`, `phase_ring`, and `curve_insets`.
- **Layout and color.** `extent_lod`, `extent_lod_from_pixels`,
  `extent_arcsec`, `extent_au`, the matching `label_lod`, `label_arcsec`,
  and `label_au`, `Frame` and `SourceStyles` for panels of one scene, and
  `role` (a named role or neutral rank, in any mode) and `neutral` for
  brand and scenery colors.

### Put a clip on one clock

- **Timing.** `Timeline` declares named spans in seconds (`enter`, `leave`,
  `text`, `move`, `run`, `hold`, `wait_until`, `read`, `settle`,
  `together`, `stagger`, `indicate`, `circumscribe`, `click`, `caption`, `section`,
  `hero`) and `Timeline.at_fps` resolves them into a `Plan` that a draw
  function reads frame by frame (`value`, `progress`, `frame_of`, `events`,
  `spans_json`). `Tempo` sets the pace of the whole clip in one place.
- **Helpers.** `ease`, `stagger`, `Time`, `frame_count`, `reading_words`,
  and the `Span`, `Event` and `Fault` records a plan carries.

### Move the viewer's eye

- **Draw on.** `head_draw` (the signature arc-length reveal with a bright
  head), `draw_on`, `build_axes`, `arrow`, `angle_arc`, `compass`,
  `write` (a label traced as if by hand), and `words`.
- **Point at the subject.** `indicate` (a pulse that returns exactly),
  `circumscribe`, `callout`, `leader`, `link`, `flight`, `spotlight`,
  `brush`, `follow_zoom`, `Anchor`, and `audit` with its `AttentionFault`
  for the connector rules.
- **Dim, bring in, and settle.** `Stage` (groups shown, dimmed, or absent,
  at absolute levels), `fade`, `blend`, `capture`, and `collapse`.
- **Land the answer.** `click` (ink swells, the end dot grows, the number
  sets bold, the ruler recedes), `ruler_readout`, and `readout`.
- **Equations and numbers.** `term_row`, `equation`, `transform` (one
  equation carried into another glyph by glyph), `exposure_text`,
  `aperture_counts`, `arrivals`, and `photons`.
- **Uncertainty.** `weighted_draws` (a posterior as true-scale weighted
  draws) and `hop` (draws played one at a time with aging ghosts and a
  running tally).
- **Cameras and morphs.** `PageCamera`, `zoom_path`, `view_limits`,
  `overview_box`, `warp`, `warp_grid`, `morph`, `quad_image`, and
  `pixel_quads`.

### Keep color and scale honest

- **Motifs.** `Motif` and `MotifSet` give each recurring quantity one
  drawing style across a clip, with `AcceptedPair` and `is_hue_role` in
  `eyepiece.motifs`.
- **Scale for a talk.** `talk_rc` (type and strokes sized from the frame
  height, from `TALK_FRACTIONS` and `VIDEO_RC`), `use_video` and
  `video_roles` for video-safe dark colors, and `unsnapped`.

### Narrate it (`eyepiece.narration`)

- **Script and speech.** `Script`, `Line`, `Pause`, `Cue` and `parse_cues`
  for cue marks on words, `estimate` for a draft length, `say` for numbers
  written for the voice, `speech_faults`, `lint_script`, and
  `check_script_speech`. `python -m eyepiece.narration.synth` speaks the
  lines into a cache under the `[narration]` extra.
- **On the clock.** `Narration` loads the measured speech and places each
  line, beat, and cue (`Timeline(narration=...)` and `Timeline.line` use
  it), with `check_number_sync`, `check_silence`, and `sentences`.
- **Captions, chapters and sound.** `captions`, `caption_at`,
  `write_captions`, `write_vtt`, `write_srt`, `write_chapters`, cue sounds
  (`CueEvents`, `cue_track`, `count_tone`, `pixel_ticks`, `click_sting`),
  `audio_track`, and `mux` for a loudness-normalized narrated mp4.

### Check the clip (`eyepiece.testing`)

`check_clip` re-renders a plan-driven clip without writing files and
collects its readability, motion and meaning findings, the plan's own faults,
unread knobs, and speaking pace in one `ClipReport`. Each check also stands
alone: `motion_faults` (jumps, dead moves, drift, torn groups, overlaps,
departures) against the moves a clip declares, or the `MotionContract` a
move's result carries,
`readability_faults`, `meaning_faults` and `meaning_faults_chain` (one
meaning per hue, within and across clips), `motif_faults` and
`motif_margins`, `timing_faults` (pace, gaps, still screens, numbers said
before they are shown), `speech_rate`, `grounding_faults` (equation terms
without a referent), and `motion_ink` (how much of the motion is the
quantity). The fault checks have `assert_*` twins that raise, `caption_line`
places captions clear of the figure's text, and `review_sheet` draws a
contact sheet at the cues and events that matter.

### Record and render

- **Record.** `save_fig`, `record` (a context manager that takes frames from
  a loop the caller drives, with `stills=` from the same pass and
  `on_rescale=` against scale drift), `animate` and its `Animation`,
  `RawSink`, `PRESETS`, `ENCODER_PROFILE`, and `to_scrubber` (an offline
  HTML page whose slider scrubs one knob).
- **Render in parallel.** `ClipSpec`, `Venue`, `Variant` and `variants`
  describe a clip's venue, mode, and narration variants; `render_variants`
  renders each one keyed by `content_key` into a `Manifest`, resumably;
  `render_frames` renders one clip's frames in worker processes; `render_poster`,
  `render_still`, and `same_picture` make and compare deterministic stills;
  `frame_faults` and `assert_frames_pure` check that frames do not depend
  on draw order.
- **Provenance.** `stamp`, `provenance_text`, `provenance_fields`, and
  `file_metadata`.

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
- **`eyepiece.mpl`.** `render` and `animate` for Matplotlib stills and movies.
- **`eyepiece.manim`.** `render` and `animate` for native Manim Community
  objects, usable in a Manim scene or a Manim Slides deck, with `Units` for
  point sizes and `ensure_font` for Pango fonts.

## Usage

```python
import eyepiece as ep

result = ep.imshow_log(psf, extent=ep.extent_lod_from_pixels(psf.shape[0], 0.5))
ep.label_lod(result.ax)
ep.save_fig(result.fig, "psf")
```

Every primitive returns a small result object carrying the axes it drew on
and the artists it made (keyed by the common `ARTIST_KEYS` vocabulary), so a
caller can keep working on the figure without hunting for the objects again.
`imshow_log` and `imshow_diverging` also return an `.update` that redraws
with new data through the same transform, which is what makes animation a
few lines:

```python
frames = [cube[k] for k in range(len(cube))]
result = ep.imshow_log(frames[0])


def draw(fig, k):
    result.update(frames[k])


ep.animate(result.fig, draw, len(frames), fps=10).save("run.mp4", "run.gif")
```

An explanatory clip puts every move on one clock. The timeline below draws a
curve on, pulses a marker, and holds; the plan sets every frame, the checks
run on the same draw function, and `record` writes the movie and a still
from one pass:

```python
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep
import eyepiece.testing as ept

plt.rcParams.update(ep.talk_rc(720, dpi=100))
x = np.linspace(0.0, 10.0, 200)
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100, layout="constrained")
(curve,) = ax.plot(x, np.sin(x))
(marker,) = ax.plot([7.85], [1.0], "o")
reveal = ep.head_draw(curve, ease="linear")
emphasis = ep.indicate(marker, scale=1.5)

tl = ep.Timeline({"curve": 0.0, "emphasis": 0.0})
tl.move("draw", 2.0, to={"curve": 1.0})
tl.run("emphasize", 1.0, to={"emphasis": 1.0})
tl.hold(1.5)
plan = tl.at_fps(30)


def draw(k):
    reveal.update(plan.value(k, "curve"))
    emphasis.update(plan.value(k, "emphasis"))


ept.assert_clip(fig, draw, plan, moves={curve: "draw"})
with ep.record(fig, "clip.mp4", fps=plan.fps, plan=plan, stills={"clip.png": "draw"}) as rec:
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


movie = ep.animate(fig, draw_frame, range(plan.n_frames), fps=plan.fps)
faults = ept.readability_faults(fig, at=at, times=[0.0, 0.5, 1.0])
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
import eyepiece.mpl as mpl
from eyepiece.style import SourceCast, snapshot_profile

cast = SourceCast(["planet b"])
still = mpl.render(sequence.frame(10), cast=cast)
strip = mpl.render(sequence.strip([0, 10, 20]), cast=cast)
movie = mpl.animate(sequence, run_time=8, fps=30, cast=cast)
movie.save("run.mp4")
```

The same sequence plays in a talk through Manim:

```python
import manim
from matplotlib.colors import to_hex

import eyepiece.manim as em

profile = snapshot_profile()
clip = em.animate(sequence, cast=cast, profile=profile)
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

result = ep.rail(
    [("Pupil", "pupil"), ("FPM", "fpm"), ("Lyot", "lyot"), ("Focal", "focal")],
    highlight="FPM",
)
ep.save_fig(result.fig, "coronagraph_rail")
```

## Status

eyepiece is pre-1.0. The figure primitives, the timeline and its moves, the
narration and check layers, the parallel render driver, and the prepared
views are implemented and tested, and the public API may still shift
between minor versions before 1.0. The
[changelog](https://github.com/CoreySpohn/eyepiece/blob/main/CHANGELOG.md)
records each change.

Deprecated in 0.16, with a warning that names the replacement, and removed in
0.17: `eyepiece.testing.Span` (now `Window`), `spans` (now
`shade_intervals`), `eyepiece.motifs.resolve_role` (now `role`, which
takes `mode=`), `Timeline.pulse` and `Timeline.ring` (now `Timeline.indicate` and
`Timeline.circumscribe`), and `PRESETS["talk_60"]` (now `PRESETS["talk"]`).
The arguments `write(overlap=)`, `record(allow_rescale=)` and
`Narration(min_s=)` are gone; use `on_rescale=` for `record` and `ends=` for
`Narration`.

## Installation

```bash
pip install eyepiece
```

Unit conversions used by a small number of layout helpers (arcsecond and AU
extents) and `radial_profile_plot`'s profile computation are optional and
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
