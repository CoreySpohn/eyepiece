# eyepiece

`eyepiece` is a small library of plotting and animation primitives for
astronomical imaging simulations: ax-first stateless figure functions for
images, comparisons, profiles, and distributions, a multi-sink animation
recorder, and styled figure saving, all built on top of matplotlib and NumPy.

The same primitives drive explanatory animation. A `Timeline` puts every move
of a clip on one clock, `eyepiece.narration` places spoken lines and their cue
words on that clock, `eyepiece.testing` checks the finished clip for motion,
readability, color-meaning and timing faults, and `eyepiece.render` renders
its venue, mode and narration variants in parallel. To find the function for
something a figure or clip should do, start from the
{doc}`capability catalog <reference/catalog>`, which maps each task to its
function and the page that shows it.

Every primitive takes plain arrays in and returns a small result object
holding the axes it drew on and the artists it made, so a caller can keep
working on a figure without hunting down which image or line object came
from where. Nothing in the library imports a simulation package, and
`import eyepiece` never activates a style or writes to matplotlib's
rcParams, so a primitive behaves the same whether it is called from a
notebook, a figure script, or another library's plotting module.

The same arrays can also be written down once as a prepared view: plain
records of images, curves, tracks, and their marks, with units, masks,
display scales, and stable IDs. The {doc}`prepared views <explanation/prepared-views>`
guide renders one preparation as a paper still, a strip of epochs, and a
Matplotlib movie through `eyepiece.mpl`, and
{doc}`Rendering in Manim <explanation/rendering-in-manim>` plays the same preparation as native Manim objects in a talk through
`eyepiece.manim`.

Because the same behavior holds for every function, the rules are worth
reading once rather than rediscovering per call. The
{doc}`contract <explanation/contract>` page states them, and
{doc}`viz-convention <viz-convention>` describes how a simulation library
ships plotting for its own types on top of these primitives.
{doc}`evidence <explanation/evidence>` is the reasoning underneath both: proportionality,
data ink, layering by value, small multiples, labels on the data, and showing
mechanism rather than outcome.

## Installation

```bash
pip install eyepiece
```

The base install pulls in matplotlib, NumPy, and
[hwostyle](https://github.com/HabitableWorldsObservatory/hwostyle), which
supplies the colormaps, palettes, and savefig policy that primitives read at
call time.

```bash
pip install eyepiece[hwo]
```

The `[hwo]` extra adds
[hwoutils](https://github.com/CoreySpohn/hwoutils), which provides the unit
conversions behind `extent_arcsec`, `extent_au`, and `Frame.extent_arcsec`,
and the profile computation behind `radial_profile_plot`. Those four are the
only names that need it, and each raises an `ImportError` naming the extra
when it is missing. Everything else works on the base install.

```bash
pip install "eyepiece[manim]"
pip install "eyepiece[slides]"
```

The `[manim]` extra adds Manim Community for `eyepiece.manim`, and
`[slides]` adds Manim Slides beside it. Manim needs system Cairo, Pango,
pkg-config, and ffmpeg. Nothing outside `eyepiece.manim` imports either.

## Quickstart

A primitive called with no axes creates its own figure, draws into it, and
hands back both. Layout helpers supply the pixel-edge extent and the
matching axis labels, and `save_fig` writes the result using the active
mode's savefig policy.

```python
import eyepiece as ep

result = ep.imshow_log(psf, extent=ep.extent_lod_from_pixels(psf.shape[0], 0.25))
ep.label_lod(result.ax)
ep.save_fig(result.fig, "psf")
```

The same call given an axes draws into that axes instead, which is what
makes a primitive usable as one panel of a larger figure the caller
assembled. Multi-panel primitives take an `axes=` sequence in place of
`ax=`, and return the array of panels they drew on.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, layout="constrained")
row = ep.compare_row(
    [before, after, model], titles=["Before", "After", "Model"], axes=axes
)
```

Because a result carries the artists it made, animating a figure is a matter
of mutating them rather than redrawing. Image primitives that transform their
data before display also return an `.update` that re-applies the same
transform, so a frame costs one `set_data` call.

```python
result = ep.imshow_log(cube[0])


def draw(fig, k):
    result.update(cube[k])


ep.animate(result.fig, draw, len(cube), fps=10).save("run.mp4", "run.gif")
```

`animate` binds a figure, a draw function, and a frame source, and renders
one pass into every sink named in `.save`. When the frames come from a loop
the caller already owns, a simulation stepping forward for instance, use
`record` instead and take a frame whenever there is something to show.

```python
with ep.record(result.fig, "run.mp4", "run.gif", fps=10) as rec:
    for state in simulation:
        result.update(state.image)
        rec.frame()
    rec.hold(8)
```

## What is in it

The groups below follow what an author sets out to do. Names without a
prefix are importable straight from `eyepiece`; the rest live in the named
submodule. The {doc}`capability catalog <reference/catalog>` is the full
table, one row per task, with the page that shows each name, and the API
reference documents every signature.

### Make a still

- **Images.** `imshow_log`, `imshow_diverging` and `display_limits`,
  `compare_row` and `compare_grid` (several images on one norm and
  colorbar), `triptych` (A, B, and their ratio or residual), `show_field`
  (a complex field), and `kymograph` (a cut stacked against time or
  wavelength).
- **Marks and annotations.** `overlay_circle` and `overlay_line` (dashed
  marks that read over any pixels), `ruler`, `bracket`, `ref_line` (a
  threshold with its label pinned to an axes edge), `spans` (excluded
  intervals), `log_zero_floor`, and `backing` (a plate behind a label).
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
  `role` and `neutral` for brand and scenery colors.

### Put a clip on one clock

- **Timing.** `Timeline` declares named spans in seconds (`enter`, `leave`,
  `text`, `move`, `run`, `hold`, `wait_until`, `read`, `settle`,
  `together`, `stagger`, `pulse`, `ring`, `click`, `caption`, `section`,
  `hero`) and `Timeline.at_fps` resolves them into a `Plan` that a draw
  function reads frame by frame (`value`, `progress`, `frame_of`, `events`,
  `spans_json`). `Tempo` sets the pace of the whole clip in one place.
- **Helpers.** `ease`, `stagger`, `Time`, `frame_count`, `reading_words`,
  and the `Span`, `Event` and `Fault` records a plan carries.

### Move the viewer's eye

- **Draw on.** `head_draw` (the signature constant-speed reveal with a
  bright head), `draw_on`, `build_axes`, `arrow`, `angle_arc`, `compass`,
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
  drawing style across a clip, with `AcceptedPair`, `resolve_role`, and
  `is_hue_role` in `eyepiece.motifs`.
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
without a referent), and `motion_ink`. Every check has an `assert_*` twin,
`caption_line` places captions clear of the figure's text, and
`review_sheet` draws a contact sheet at the cues and events that matter.

### Record and render

- **Record.** `save_fig`, `record` (a context manager that takes frames from
  a loop the caller drives, with `stills=` from the same pass and
  `on_rescale=` against scale drift), `animate` and its `Animation`,
  `RawSink`, `PRESETS`, `ENCODER_PROFILE`, and `to_scrubber` (an offline
  HTML page whose slider scrubs one knob).
- **Render in parallel.** `ClipSpec`, `Venue`, `Variant` and `variants`
  describe a clip's venue, mode, and narration variants; `render_variants`
  renders them keyed by `content_key` into a `Manifest`, resumably;
  `render_frames` renders one clip in worker processes; `render_poster`,
  `render_still`, and `same_picture` make and compare deterministic stills;
  `frame_faults` and `assert_frames_pure` check that frames do not depend
  on draw order.
- **Provenance.** `stamp`, `provenance_text`, `provenance_fields`, and
  `file_metadata`.

### Prepare a scene once

- **`eyepiece.prepared`.** The frozen records (`ImageView`, `CurveView`,
  `TrackView`, `PanelGroup`, and the `Path`, `Points`, `Region`,
  `ReferenceLine`, and `Label` marks, with `AxisSpec` and `Scale`),
  `Sequence` for replayable states over physical time, and the display
  mapping. It imports only NumPy.
- **`eyepiece.style`.** `SourceCast` for fixed source colors and markers, and
  `snapshot_profile`, which captures the active hwostyle appearance as a
  read-only `RenderProfile`.
- **`eyepiece.mpl`.** `render` and `animate` for Matplotlib stills and movies.
- **`eyepiece.manim`.** `render` and `animate` for native Manim Community
  objects, usable in a Manim scene or a Manim Slides deck, with `Units` for
  point sizes and `ensure_font` for Pango fonts.

## Where to go next

Read {doc}`contract <explanation/contract>` to predict what any primitive will
do with the arguments you hand it and what you get back. Read
{doc}`prepared views <explanation/prepared-views>` to prepare a scene once for figures,
movies, and talks, and
{doc}`Rendering in Manim <explanation/rendering-in-manim>` to present one. Read
{doc}`viz-convention <viz-convention>` if you maintain a simulation library
and want its own types to plot themselves without eyepiece ever learning
about them. Read {doc}`the clock <explanation/clock>` for timing and motion rules,
{doc}`narration design <explanation/narration-design>` for speech and cues, and
{doc}`Eyepiece and Manim <explanation/eyepiece-and-manim>` for the motion
comparison. The API reference documents every signature.

The tutorials build {doc}`a still <tutorials/still>`,
{doc}`a silent build-up <tutorials/build-up>` and
{doc}`a narrated clip <tutorials/narrated>` from start to finish. The motion
galleries show each family of moves running:
{doc}`clocks and readouts <gallery/motion-clock>`,
{doc}`cameras and zooms <gallery/motion-camera>`,
{doc}`attention moves <gallery/motion-attention>`,
{doc}`staging <gallery/motion-staging>`,
{doc}`equations and morphs <gallery/motion-equations>` and
{doc}`clip checks <gallery/motion-checks>`. The how-to guides each solve one
task: {doc}`bring a panel in <how-to/bring-a-panel-in>`,
{doc}`draw a line on <how-to/draw-a-line-on>`,
{doc}`bind a move to a knob <how-to/bind-a-move-to-a-knob>`,
{doc}`declare a cast <how-to/declare-a-cast>`,
{doc}`check a clip <how-to/check-a-clip>`,
{doc}`step through discrete values <how-to/step-through-discrete-values>`,
{doc}`keep the scale fixed <how-to/keep-the-scale-fixed>`,
{doc}`take stills from the encode <how-to/stills-from-the-encode>`,
{doc}`write a viz module <how-to/write-a-viz-module>`,
{doc}`burn captions into a band <how-to/burn-captions-into-a-band>`,
{doc}`estimate length before speaking <how-to/estimate-length-before-speaking>`,
{doc}`speak a script <how-to/speak-a-script>`,
{doc}`write numbers for speech <how-to/write-numbers-for-speech>`,
{doc}`check narration and motion <how-to/check-narration-and-motion>`,
{doc}`check timing <how-to/check-timing>`,
{doc}`ground a symbol <how-to/ground-a-symbol>`,
{doc}`publish a scrubber page <how-to/publish-a-scrubber-page>`,
{doc}`add cue sounds <how-to/add-cue-sounds>` and
{doc}`render a clip in parallel <how-to/render-a-clip-in-parallel>`.

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/still
tutorials/build-up
tutorials/narrated
```

```{toctree}
:maxdepth: 2
:caption: How-to

how-to/bring-a-panel-in
how-to/draw-a-line-on
how-to/bind-a-move-to-a-knob
how-to/declare-a-cast
how-to/check-a-clip
how-to/step-through-discrete-values
how-to/keep-the-scale-fixed
how-to/stills-from-the-encode
how-to/write-a-viz-module
how-to/burn-captions-into-a-band
how-to/estimate-length-before-speaking
how-to/speak-a-script
how-to/write-numbers-for-speech
how-to/check-narration-and-motion
how-to/check-timing
how-to/ground-a-symbol
how-to/publish-a-scrubber-page
how-to/add-cue-sounds
how-to/render-a-clip-in-parallel
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/catalog
autoapi/index
reference/to-scrubber
reference/attention-moves
reference/clip-checks
reference/motion-contract
gallery/images
gallery/stats
gallery/intervals
gallery/marks
gallery/values
gallery/diagnostics
gallery/profiles
gallery/scene
gallery/phasor
gallery/animation
gallery/motion
gallery/one-scene-n-views
gallery/prepared
gallery/motifs
```

```{toctree}
:maxdepth: 2
:caption: Explanation

explanation/contract
explanation/clock
explanation/evidence
explanation/narration-design
explanation/prepared-views
explanation/eyepiece-and-manim
explanation/rendering-in-manim
viz-convention
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

autoapi/index
```
