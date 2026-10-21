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
and the profile computation behind `plot_radial_image`. Those four are the
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

result = ep.plot.imshow_log(psf, extent=ep.plot.extent_lod_from_pixels(psf.shape[0], 0.25))
ep.plot.label_lod(result.ax)
ep.render.save_fig(result.fig, "psf")
```

The same call given an axes draws into that axes instead, which is what
makes a primitive usable as one panel of a larger figure the caller
assembled. Multi-panel primitives take an `axes=` sequence in place of
`ax=`, and return the array of panels they drew on.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, layout="constrained")
row = ep.plot.compare_row(
    [before, after, model], titles=["Before", "After", "Model"], axes=axes
)
```

Because a result carries the artists it made, animating a figure is a matter
of mutating them rather than redrawing. Image primitives that transform their
data before display also return an `.update` that re-applies the same
transform, so a frame costs one `set_data` call.

```python
result = ep.plot.imshow_log(cube[0])


def draw(fig, k):
    result.update(cube[k])


ep.render.animate(result.fig, draw, len(cube), fps=10).save("run.mp4", "run.gif")
```

`animate` binds a figure, a draw function, and a frame source, and renders
one pass into every sink named in `.save`. When the frames come from a loop
the caller already owns, a simulation stepping forward for instance, use
`record` instead and take a frame whenever there is something to show.

```python
with ep.render.record(result.fig, "run.mp4", "run.gif", fps=10) as rec:
    for state in simulation:
        result.update(state.image)
        rec.frame()
    rec.hold(8)
```

## What is in it

Use `import eyepiece as ep`, then call through the task namespace that
matches what you want to do. The {doc}`capability catalog <reference/catalog>`
is the full table, one row per task, with the page that shows each name,
and the API reference documents every signature.

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

- **Timing.** `ep.clock.Timeline` declares named spans in seconds (`ep.clock.Timeline.enter`, `ep.clock.Timeline.leave`,
  `ep.clock.Timeline.text`, `ep.clock.Timeline.move`, `ep.clock.Timeline.run`, `ep.clock.Timeline.hold`, `ep.clock.Timeline.wait_until`, `ep.clock.Timeline.read`, `ep.clock.Timeline.settle`,
  `ep.clock.Timeline.together`, `ep.clock.Timeline.stagger`, `ep.clock.Timeline.indicate`, `ep.clock.Timeline.circumscribe`, `ep.clock.Timeline.click`, `ep.clock.Timeline.caption`, `ep.clock.Timeline.section`,
  `ep.clock.Timeline.hero`) and `ep.clock.Timeline.at_fps` resolves them into an `ep.clock.Plan` that a draw
  function reads frame by frame (`ep.clock.Plan.value`, `ep.clock.Plan.progress`, `ep.clock.Plan.frame_of`, `ep.clock.Plan.events`,
  `ep.clock.Plan.spans_json`). `ep.clock.Tempo` sets the pace of the whole clip in one place.
- **Helpers.** `ep.clock.ease`, `ep.clock.stagger`, `ep.clock.Time`, `ep.clock.frame_count`, `ep.clock.reading_words`,
  and the `ep.clock.Span`, `ep.clock.Event` and `ep.clock.Fault` records a plan carries.

### Move the viewer's eye

- **Draw on.** `ep.motion.head_draw` (the signature arc-length reveal with a bright
  head), `ep.motion.draw_on`, `ep.motion.build_axes`, `ep.motion.arrow`, `ep.motion.angle_arc`, `ep.motion.compass`,
  `ep.motion.write` (a label traced as if by hand), and `ep.motion.words`.
- **Point at the subject.** `ep.motion.indicate` (a pulse that returns exactly),
  `ep.motion.circumscribe`, `ep.motion.callout`, `ep.motion.leader`, `ep.motion.link`, `ep.motion.flight`, `ep.motion.spotlight`,
  `ep.motion.brush`, `ep.motion.follow_zoom`, `ep.motion.Anchor`, and `ep.testing.audit` with its `ep.testing.AttentionFault`
  for the connector rules.
- **Dim, bring in, and settle.** `ep.motion.Stage` (groups shown, dimmed, or absent,
  at absolute levels), `ep.motion.fade`, `ep.motion.blend`, `ep.motion.capture`, and `ep.motion.collapse`.
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
  drawing style across a clip, with `ep.motifs.AcceptedPair` and `ep.motifs.is_hue_role` in
  `ep.motifs`.
- **Scale for a talk.** `ep.style.talk_rc` (type and strokes sized from the frame
  height, from `ep.style.TALK_FRACTIONS` and `ep.style.VIDEO_RC`), `ep.style.use_video` and
  `ep.style.video_roles` for video-safe dark colors, and `ep.style.unsnapped`.

### Narrate it (`ep.narration`)

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

### Check the clip (`ep.testing`)

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
quantity). The fault checks have `ep.testing.assert_*` twins that raise, `ep.narration.caption_line`
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

- **`ep.prepared`.** The frozen records (`ep.prepared.ImageView`, `ep.prepared.CurveView`,
  `ep.prepared.TrackView`, `ep.prepared.PanelGroup`, and the `ep.prepared.Path`, `ep.prepared.Points`, `ep.prepared.Region`,
  `ep.prepared.ReferenceLine`, and `ep.prepared.Label` marks, with `ep.prepared.AxisSpec` and `ep.prepared.Scale`),
  `ep.prepared.Sequence` for replayable states over physical time with the
  `ep.prepared.ArrayChannel`, `ep.prepared.PathWindow`, and `ep.prepared.Clock` channels that change them, and
  the display mapping. It imports only NumPy.
- **`ep.style`.** `ep.style.SourceCast` for fixed source colors and markers, and
  `ep.style.snapshot_profile`, which captures the active hwostyle appearance as a
  read-only `ep.style.RenderProfile`.
- **`ep.mpl`.** `ep.mpl.render` and `ep.mpl.animate` for Matplotlib stills and movies.
- **`ep.manim`.** `ep.manim.render` and `ep.manim.animate` for native Manim Community
  objects, usable in a Manim scene or a Manim Slides deck, with `ep.manim.Units` for
  point sizes and `ep.manim.ensure_font` for Pango fonts.

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
{doc}`signature moves <gallery/signature-moves>`,
{doc}`clocks and readouts <gallery/motion-clock>`,
{doc}`weighted draws <gallery/weighted-draws>`,
{doc}`cameras and zooms <gallery/motion-camera>`,
{doc}`attention moves <gallery/motion-attention>`,
{doc}`staging <gallery/motion-staging>`,
{doc}`equations and morphs <gallery/motion-equations>` and
{doc}`clip checks <gallery/motion-checks>`. The how-to guides each solve one
task: {doc}`bring a panel in <how-to/bring-a-panel-in>`,
{doc}`draw a line on <how-to/draw-a-line-on>`,
{doc}`measure and land an answer <how-to/land-an-answer>`,
{doc}`size a figure for a talk <how-to/size-a-talk>`,
{doc}`bind a move to a knob <how-to/bind-a-move-to-a-knob>`,
{doc}`schedule Timeline verbs <how-to/timeline-verbs>`,
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
how-to/land-an-answer
how-to/size-a-talk
how-to/bind-a-move-to-a-knob
how-to/timeline-verbs
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
gallery/signature-moves
gallery/weighted-draws
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
