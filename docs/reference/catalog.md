# Capability catalog

Find the eyepiece name for a thing you want an animation or figure to do. Each row starts from
the task as you would say it, names the function or class, says in one line what it does, and
links the page that shows it. A page shows a name when a code block on it calls the name.
"(prose)" means a page names it but has no code example, and "API reference" means only the API
reference covers it, with no tutorial, how-to, gallery or explanation page yet. The lists on the
front page and in this catalog do not count.

Use `import eyepiece as ep` and call through the task namespaces, for example
`ep.plot.imshow_log(...)` or `ep.motion.indicate(...)`. The tables omit `ep.`
for readability: `plot.` means `eyepiece.plot`, `clock.` means
`eyepiece.clock`, and the other task prefixes work the same way. `synth.` is
`eyepiece.narration.synth`. `tl` is a `clock.Timeline`, `plan` is its
`clock.Plan`, and `stage` is a `motion.Stage`.

## Legacy root paths

In 0.17, every former root path still works silently and returns the same
object. New examples use the task namespaces. Root star imports expose only
`Timeline`, `Plan`, `Tempo`, `PlotResult`, `MosaicResult`, `MotionContract`,
`animate`, `record`, `save_fig`, and `__version__`; these stable shortcuts may
also appear bare in the tables. Existing deprecated aliases keep warning.
The older render check paths and `testing.caption_line` remain compatible;
their canonical homes below are `testing` and `narration`, respectively.

## 1. Put the clip on one clock

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| lay out a clip in seconds, one clock for every move | `Timeline(knobs, tempo=, narration=)`, then `tl.at_fps(fps)` returns a `Plan` | Spans declared in seconds; the plan resolves them for one frame rate | [motion-clock](../gallery/motion-clock.md), [clock](../explanation/clock.md), [build-up](../tutorials/build-up.md) |
| open the next spoken line | `tl.line(key)` | Places the line's speech on the clock and closes the previous line | [narrated](../tutorials/narrated.md) (tutorial script) |
| bring a subject in on the word that names it | `tl.enter(name, seconds=0.6, cue=)` | Knob `name` ramps 0 to 1; with a cue it starts 0.1 s before the cue word | [motion-clock](../gallery/motion-clock.md), [narrated](../tutorials/narrated.md) |
| take a subject away | `tl.leave(name, seconds=0.3)` | Knob `name` ramps back to 0 | [timeline-verbs](../how-to/timeline-verbs.md) |
| bring text in, or a label as the move it names settles | `tl.text(name, names=)` | Text knob ramps in; with `names=` it may start over the named move's last quarter | [motion-clock](../gallery/motion-clock.md), [motion-attention](../gallery/motion-attention.md) |
| move a value to a target, eased | `tl.move(name, seconds, to=, space="log")` | Knobs move to targets on the ease, optionally in log space | [motion-clock](../gallery/motion-clock.md), [animation](../gallery/animation.md) |
| run at a constant rate (a physical clock, a sweep) | `tl.run(name, seconds, to=)` | Linear in time, never paced or eased | [build-up](../tutorials/build-up.md), [check-a-clip](../how-to/check-a-clip.md) |
| change state instantly at a declared cut | `tl.cut(name, to=, at=, cue=)` | Zero-duration span; strict builds and clip checks accept the step | [motion-clock](../gallery/motion-clock.md) |
| hold the picture for a time | `tl.hold(seconds, to=)` | A span in which nothing moves; `to` steps knobs at its start | [bind-a-move-to-a-knob](../how-to/bind-a-move-to-a-knob.md) |
| hold the picture until a time computed elsewhere | `tl.wait_until(*times)` | Extends the section to that time without placing a span | [motion-clock](../gallery/motion-clock.md), [clock](../explanation/clock.md) |
| hold long enough to read some text | `tl.read(text)`, `Tempo.read_s(text)` | Hold of `factor * (base_s + word_s * words)` | [motion-clock](../gallery/motion-clock.md), [motion-camera](../gallery/motion-camera.md) |
| rest after a new subject or an axis build | `tl.settle()`, `Tempo.rest_s` | Rest for the tempo's settle time | [timeline-verbs](../how-to/timeline-verbs.md) |
| start several moves at the same moment | `tl.together()` | Used as a `with` block: every span in it starts at `now` | [timeline-verbs](../how-to/timeline-verbs.md) |
| step through targets one after another | `tl.stagger(verb, targets, each_s=)` | Calls a verb per target, `each_s` apart | [timeline-verbs](../how-to/timeline-verbs.md) |
| stagger items inside one beat (rings, pixels) | `clock.stagger(u, rank, span=0.35)` | Per-item progress from one beat's progress | API reference; [motion-clock](../gallery/motion-clock.md) (prose) |
| find when a cue word is said | `tl.cue("kind:target")` | The word start of a cue mark, as a `clock.Time`; `tl.enter`, `tl.leave`, `tl.text`, `tl.move`, `tl.span`, `tl.caption`, `tl.indicate` and `tl.circumscribe` also take `cue=` | [timeline-verbs](../how-to/timeline-verbs.md) |
| declare the hero beat (stillness, one move, silence) | `tl.hero(span)`, `plan.hero_windows()` | Marks the span the silence and timing checks treat as deliberate | [timeline-verbs](../how-to/timeline-verbs.md) |
| mark sections of a clip | `tl.section(name)` | Opens a named section; the plan's frames carry it | [animation](../gallery/animation.md), [build-up](../tutorials/build-up.md) |
| a span the verbs do not cover | `tl.span(name, seconds, kind=, paced=, ease=, extends=)` | Every default stated explicitly | [timeline-verbs](../how-to/timeline-verbs.md) |
| read a knob on a frame | `plan.value(k, knob)` | The value of a knob at frame `k` | [bind-a-move-to-a-knob](../how-to/bind-a-move-to-a-knob.md), [motion-clock](../gallery/motion-clock.md) |
| read a span's progress on a frame | `plan.progress(k, span, eased=)` | 0 before the span, 1 from its end | [motion-clock](../gallery/motion-clock.md), [motion-camera](../gallery/motion-camera.md) |
| find the frame of a span's start, middle or end | `plan.frame_of(span, where=)` | Frame index of a span's state | [bind-a-move-to-a-knob](../how-to/bind-a-move-to-a-knob.md) |
| list the clip's moments | `plan.events()`, `plan.review_frames()` | Timed events; frames a contact sheet draws | [motion-clock](../gallery/motion-clock.md) |
| list frames where a knob steps | `plan.step_frames()` | Frames where a step write changes a knob | [step-through-discrete-values](../how-to/step-through-discrete-values.md) |
| find knobs the draw never reads | `plan.unused_knobs()` | Unread and never-changing knobs | [animation](../gallery/animation.md) |
| write span times to a manifest | `plan.spans_json(holds=)` | Each span's start and end in seconds, rounded to the millisecond; `holds=False` leaves out holds | [timeline-verbs](../how-to/timeline-verbs.md) |
| set the pace of the whole clip in one place | `Tempo(factor, base_s, word_s, settle_s)` | One factor scales moves, holds, reading and settles | [bind-a-move-to-a-knob](../how-to/bind-a-move-to-a-knob.md), [motion-clock](../gallery/motion-clock.md) |
| ease a progress value | `clock.ease(u, kind)` | Eased progress with fixed ends, for the kinds `"smooth"` (the sigmoid default), `"smoothstep"`, `"smootherstep"`, `"cosine"`, `"linear"`, `"back_out"` and `"running_start"` | [attention-moves](../reference/attention-moves.md), [motion-clock](../gallery/motion-clock.md) |
| add seconds exactly | `clock.Time` | A float that adds exactly | [clock](../explanation/clock.md) (prose) |
| count frames in a duration | `clock.frame_count(seconds, fps)` | Frames a duration lasts | [clock](../explanation/clock.md) (prose) |
| count the words a viewer reads | `clock.reading_words(text)` | Words, with a math symbol counting as two | API reference |
| inspect resolved spans and findings | `clock.Span`, `clock.Event`, `clock.Fault` (`plan.faults`) | Records the plan returns | [clock](../explanation/clock.md) (prose: `clock.Span`, `plan.faults`); `clock.Event` and `clock.Fault` API reference |

## 2. Draw things on

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| draw a curve on at constant speed (the signature reveal) | `motion.head_draw(lines, ease=, lag=, behind=)` | Arc-length draw (constant speed under a linear ease) with a bright head that dissolves at rest | [draw-a-line-on](../how-to/draw-a-line-on.md), [signature-moves](../gallery/signature-moves.md) |
| reveal a line along its path, no head | `motion.draw_on(lines, ease="linear")` | Steady display-space reveal | [draw-a-line-on](../how-to/draw-a-line-on.md), [motion-attention](../gallery/motion-attention.md) |
| keep a halo or backing line behind the drawing head | `motion.head_draw(..., behind={line: halo})` | Cuts the behind line to the drawn body | [signature-moves](../gallery/signature-moves.md) |
| build an axis in front of the viewer, or a number line | `motion.build_axes(ax, axis="x"/"y"/"both", style="sweep"/"staccato")` | Spine draws on, ticks and numbers arrive, label writes on | [bring-a-panel-in](../how-to/bring-a-panel-in.md), [motion-staging](../gallery/motion-staging.md), [build-up](../tutorials/build-up.md), [signature-moves](../gallery/signature-moves.md) |
| grow an arrow or a vector | `motion.arrow(tail, head, kind=, grow="length"/"scaled")` | Arrow that grows and moves without redraw | [motion-camera](../gallery/motion-camera.md), [signature-moves](../gallery/signature-moves.md) |
| sweep an angle | `motion.angle_arc(center, start, end, radius)` | Arc between two directions that can sweep | [motion-camera](../gallery/motion-camera.md) |
| show north and east turning together | `motion.compass(center, length, east=)` | Paired N and E arrows | [motion-camera](../gallery/motion-camera.md) |
| write a label as if by hand | `motion.write(text, draw=, lag=)` | Each glyph traced, then filled | [motion-attention](../gallery/motion-attention.md), [signature-moves](../gallery/signature-moves.md) |
| reveal or color one word of a line | `motion.words(at, string, by=)` | One text per word in a fixed layout | [motion-attention](../gallery/motion-attention.md) |
| a track that fades into its own past | `plot.fading_track(xy)` | Opacity ramps from tail to head | [scene](../gallery/scene.md) |
| a 3D trajectory with a depth cue | `plot.trail(xyz, depth=, marker_scale=)` | Connected path with a depth cue: the far side dashed, or markers sized by depth | [scene](../gallery/scene.md) |
| a fan of candidate tracks by weight | `plot.sky_fan(tracks, weights=)` | Tracks faded by weight | [scene](../gallery/scene.md) |
| an image that develops from its photons | `motion.arrivals(rate, exposure, seed=)`, `motion.photons(arrivals)` | Seeded Poisson arrivals; newest photons flash | [motion-clock](../gallery/motion-clock.md), [signature-moves](../gallery/signature-moves.md) |
| a histogram that fills as samples arrive | `plot.hist_fill(values, ymax=, law=)` | Fills on a pinned scale | [stats](../gallery/stats.md) |

## 3. Point at the subject

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| underline words for a beat | `motion.underline(text_artist, color=, ease=)`, placed by `tl.underline(target)` | Draws along the baseline, holds, then fades | [motion-attention](../gallery/motion-attention.md) |
| light the thing the narration names | `motion.indicate(artists, scale=, strength=, shape=)`, placed by `tl.indicate(target)` | Swells and tints toward the text ink, then returns exactly | [motion-attention](../gallery/motion-attention.md), [bind-a-move-to-a-knob](../how-to/bind-a-move-to-a-knob.md) |
| a plateau pulse (emphasis held at its peak) | `clock.pulse_shape(u, rise=, hold=, fall=, ease=)`, `motion.indicate(..., shape="plateau")` | One shared shape: up 35%, hold 30%, down 35% by default | [motion-attention](../gallery/motion-attention.md), [signature-moves](../gallery/signature-moves.md) |
| outline a thing, then let it go | `motion.circumscribe(target, shape=, style="trace")`, placed by `tl.circumscribe(target)` | Outline draws on, then fades | [motion-attention](../gallery/motion-attention.md) (`motion.circumscribe`), [motion-clock](../gallery/motion-clock.md) (`tl.circumscribe`) |
| pass a comet around a thing | `motion.circumscribe(target, style="flash")` | A tapered comet runs once around the outline | [signature-moves](../gallery/signature-moves.md) |
| put a no-hue cue behind the subject | `motifs.MotifSet.default()["cue"].patch_kw()` on a patch you draw | Text-ink band, ring or underlay behind the subject | [motifs](../gallery/motifs.md) (no function; hand-built patch) |
| a short caption beside the thing it names | `motion.callout(target, text, side=, rise_pt=)` | Caption with an optional leader, rising into place | [motion-attention](../gallery/motion-attention.md) |
| a leader from a text to its mark | `motion.leader(text, target)` | Leader line that obeys the connector rules | [attention-moves](../reference/attention-moves.md) (prose) |
| a line from a mark in one panel to a mark in another | `motion.link(source, target, fig=)` | Grows a connector across panels | [motion-attention](../gallery/motion-attention.md) |
| carry a copy of a mark across panels | `motion.flight(source, destination, fig=, arc=)` | A copy flies and crossfades | [motion-attention](../gallery/motion-attention.md) |
| dim the whole figure outside a region | `motion.spotlight(fig, rects, strength=)` | Background-colored veil with holes | [motion-camera](../gallery/motion-camera.md) |
| light the same objects in several panels | `motion.brush(axes, quads, values, veil_alpha=)` | Lights matching pixels; `veil_alpha` is veil opacity | [motion-equations](../gallery/motion-equations.md) |
| name a point of the figure (data point or part of an artist) | `motion.Anchor(ax=, xy=)`, `motion.Anchor.from_artist(artist, loc=)` | A point resolved in display space at draw time | [motion-attention](../gallery/motion-attention.md), [attention-moves](../reference/attention-moves.md) (prose) |
| check every leader and link | `testing.audit(fig)`, `testing.AttentionFault` | Connector-rule check | [motion-camera](../gallery/motion-camera.md); `testing.AttentionFault` in [motion-attention](../gallery/motion-attention.md) (prose) |
| a magnified inset that follows a moving point | `motion.follow_zoom(ax, target, zoom=)` | Haloed inset with connectors | [motion-camera](../gallery/motion-camera.md) |

## 4. Dim, bring in and settle

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| dim everything but the current subject | `motion.Stage(groups, dim=)`, `stage.set(shown=, dimmed=)` | Groups set to shown, dimmed or absent | [motion-staging](../gallery/motion-staging.md), [bring-a-panel-in](../how-to/bring-a-panel-in.md) |
| set some groups' presence and contrast, leave the rest | `stage.weigh(levels, ease=)` | Absolute levels per named group | [bring-a-panel-in](../how-to/bring-a-panel-in.md), [build-up](../tutorials/build-up.md) |
| settle the scene by presence | `stage.weigh(levels)` | Arrival by opacity, no motion, driven by an entry knob such as one `tl.enter` or `tl.run` writes | [build-up](../tutorials/build-up.md) |
| a label that arrives attached to its mark | `stage.weigh(levels, lift_pt=)` | Labels drift up at most 10 px as they fade in | [signature-moves](../gallery/signature-moves.md) |
| return every group to how it was drawn | `stage.rest()`, `stage.state()` | Rest state; the levels of a state | [bring-a-panel-in](../how-to/bring-a-panel-in.md) (`rest`); `state` API reference |
| fade artists toward the background, or by opacity | `motion.fade(target, level, by="color"/"alpha")`, `motion.blend(color, level)` | Contrast blend (or opacity) that works in both modes | [scene](../gallery/scene.md), [motion-clock](../gallery/motion-clock.md) |
| collect the artists a block adds | `motion.capture(container)` | Context manager listing new artists | [scene](../gallery/scene.md) |
| a list of steps with the current one bright | `plot.step_list(rows, current=)` | Current step bold, done steps dim, future hidden | [scene](../gallery/scene.md) |
| gather a row of cells into one bar | `motion.collapse(ax, segments, weights, start, end, fig=)` | Cells fly into a bar of their sum | [motion-staging](../gallery/motion-staging.md) |

## 5. Land the answer

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| land a composite in one beat (the click) | `motion.click(lines, dot=, readout=, ruler=, swell=1.55)`, placed by `tl.click(target)` | 0.6 s: ink swells on a plateau, end dot grows, number sets bold, ruler recedes; 1.5 s hold after | [land-an-answer](../how-to/land-an-answer.md), [signature-moves](../gallery/signature-moves.md) |
| a chime at the click | `narration.click_sting(t_end, duration_s=)` | Two-note sting ending with the swell | [add-cue-sounds](../how-to/add-cue-sounds.md) (prose) |

## 6. Numbers, readouts and equations

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| a readout that ticks with a mark | `motion.ruler_readout(head, readout=, value=, ruler=, ruler_to=)` | One knob drives the head, a ruler and a counter read off the tip | [land-an-answer](../how-to/land-an-answer.md), [signature-moves](../gallery/signature-moves.md) |
| a number that counts without jitter | `motion.readout(at, value, fmt=, digits=, signed=)` | Fixed digit slots; `set` chains the next value | [motion-clock](../gallery/motion-clock.md) |
| a dimension arrow with a label | `plot.ruler(ax, p0, p1, text)` | Dimension arrow with a backed label | [images](../gallery/images.md) |
| label an exposure time on a log clock | `motion.exposure_text(seconds)` | `2.8 h`, `11.6 d` | [motion-clock](../gallery/motion-clock.md) |
| read counts and S/N off a photon image | `motion.aperture_counts(mark, aperture, leak=)` | Numbers read from the drawn pixels | [motion-clock](../gallery/motion-clock.md) |
| lay an equation out as separate terms | `motion.term_row(at, terms)` | One text per term, each a target | [motion-attention](../gallery/motion-attention.md) |
| an equation as glyph outlines with named terms | `motion.equation(at, tex, fontsize=, colors=)` | Glyph groups per term; `update(reveal="write")` | [motion-equations](../gallery/motion-equations.md), [ground-a-symbol](../how-to/ground-a-symbol.md), [signature-moves](../gallery/signature-moves.md) |
| carry one equation into another | `motion.transform(eq_a, eq_b, key_map=, fade_toward=)` | Glyph-by-glyph transform | [motion-equations](../gallery/motion-equations.md), [signature-moves](../gallery/signature-moves.md) |

## 7. Reference marks and furniture

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| separate end labels of nearby curves | `plot.spread_labels(labels, min_gap=)` | Least squared vertical movement, order preserved, inside the axes | [profiles](../gallery/profiles.md) |
| a labeled threshold or reference line | `plot.ref_line(ax, value, label, orient=, where=)` | Line with its label pinned to an axes edge | [marks](../gallery/marks.md) |
| a reference line with its own reveal beat | `plot.ref_line(...)`, then `motion.draw_on(result.artists["line"])` and a `motion.Stage` group for its label, on a `tl.enter` span | No single move: compose the still with a reveal | [marks](../gallery/marks.md) (still only) |
| shade excluded intervals | `plot.shade_intervals(x, mask, intervals=)` | Masked runs as bands | [marks](../gallery/marks.md) |
| a zero row on a log axis | `plot.log_zero_floor(ax, floor)` | Zero row at a display floor with a break mark | [marks](../gallery/marks.md) |
| a labeled bracket | `plot.bracket(ax, x0, x1, y, text, depth=)` | Square bracket labeled on its far side | [scene](../gallery/scene.md) |
| mark a panel carried from a previous clip | `motion.carried_tag(ax, text, corner=)` | Fixed neutral italic furniture in a top corner | [motion-staging](../gallery/motion-staging.md) |
| a label that reads over marks or pixels | `style.backing(alpha=)` | `bbox` plate in the background color | [stats](../gallery/stats.md) |
| a dashed circle or path legible on any pixels | `plot.overlay_circle`, `plot.overlay_line` | Dashed mark with an underlay | [images](../gallery/images.md) |
| quiet the frame of a line or scatter panel | `style.quiet_spines(ax, keep=, level=)` | Hides unkept spines and optionally softens the rest | [stats](../gallery/stats.md) |
| a neutral scenery tone | `style.neutral(level)` | Gray between background and text ink | [stats](../gallery/stats.md) |
| the brand color of a role | `style.role(name, mode=)` | Hex color for a named role or a neutral rank, in the active or a given mode | [stats](../gallery/stats.md) |
| a pixel-edge extent and matching axis labels | `plot.extent_lod_from_pixels`, `plot.extent_lod`, `plot.extent_arcsec`, `plot.extent_au`, `plot.label_lod`, `plot.label_arcsec`, `plot.label_au` | Extents and labels for image axes | [images](../gallery/images.md) (`plot.extent_lod_from_pixels`, `plot.label_lod`), [profiles](../gallery/profiles.md) (prose: `plot.extent_arcsec`, `plot.extent_au`); the rest API reference |
| one field of view across panels | `plot.Frame(half_fov_lod, pixscale_lod=)` | Shared extent for panels of one scene | [one-scene-n-views](../gallery/one-scene-n-views.md) |

## 8. Cameras, morphs and warps

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| move over a finished figure at a fixed output size | `motion.PageCamera(fig, size_px)`, `motion.PageCamera.fit_axes`, `motion.PageCamera.render(view)` | Renders any view of a figure | [motion-camera](../gallery/motion-camera.md); `fit`, `refresh` and `release` API reference |
| pan and zoom along the optimal path | `motion.zoom_path(view0, view1)`, `motion.view_limits(view)` | Van Wijk and Nuij path; limits of a view | [motion-camera](../gallery/motion-camera.md) |
| show where the camera is | `motion.overview_box(view, ax=)` | The view as a box on an overview panel | [motion-camera](../gallery/motion-camera.md) |
| carry the plane through a map | `motion.warp(artists, mapping)`, `motion.warp_grid(ax)` | Continuous map of drawn artists and a grid | [motion-camera](../gallery/motion-camera.md) |
| turn one path into another | `motion.morph(a, b)` | Path morph keeping identity | [motion-equations](../gallery/motion-equations.md) |
| move every pixel of an image | `motion.quad_image(values, corners)`, `motion.pixel_quads(corners)` | One movable quadrilateral per pixel | [motion-equations](../gallery/motion-equations.md) |

## 9. Uncertainty and draws

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| show a posterior as weighted draws | `plot.weighted_draws(points, weights, encode="area"/"alpha", resample=)` | True-scale weights; optional resampling beat | [weighted-draws](../gallery/weighted-draws.md), [signature-moves](../gallery/signature-moves.md) |
| play draws one at a time with ghosts and a tally | `plot.hop(draws, rate=, ghosts=, tally=)` | Steady draw rate, aging ghosts, running tally | [motion-clock](../gallery/motion-clock.md), [signature-moves](../gallery/signature-moves.md) |
| samples converging on a reference | `plot.convergence(values, refs=, running=)` | Samples beside a running mean or sum | [stats](../gallery/stats.md) |
| a triangle plot | `plot.corner`, `plot.corner_overlay` | 1D and 2D marginals | [stats](../gallery/stats.md) |
| a covariance ellipse | `plot.cov_ellipse(mean2, cov22, n_sigma=)` | n-sigma ellipse | [stats](../gallery/stats.md) |
| a histogram against its law | `plot.hist_vs_pdf(samples, pdf)` | Normalized histogram and PDF | [stats](../gallery/stats.md) |
| a trace about a level | `plot.signed_trace(values, level=)` | Filled above and below | [stats](../gallery/stats.md) |
| points with intervals, including one-sided | `plot.interval_points(x, y, lo, hi)` | Absolute interval endpoints | [intervals](../gallery/intervals.md) |
| stated against realized values | `plot.calibration(stated, realized)` | Identical-limit comparison | [diagnostics](../gallery/diagnostics.md) |
| jittered groups | `plot.strip(groups)` | Reproducible jitter, capped values as triangles | [diagnostics](../gallery/diagnostics.md) |

## 10. Images, profiles and scenes

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| a log or diverging image | `plot.imshow_log`, `plot.imshow_diverging`, `plot.display_limits` | Floored log norm; symmetric norm; data bounds | [images](../gallery/images.md) |
| a row or grid on one norm | `plot.compare_row`, `plot.compare_grid` | Shared norm and colorbar | [images](../gallery/images.md) |
| A, B and their ratio or residual | `plot.triptych(a, b, mode=)` | Three panels | [images](../gallery/images.md) |
| a complex field | `plot.show_field(field)` | Real, imaginary, amplitude, phase | [images](../gallery/images.md) |
| a cut against a second variable | `plot.kymograph(values, x, y)` | Stacked cuts | [images](../gallery/images.md) |
| radial and contrast profiles | `plot.plot_radial`, `plot.plot_radial_image`, `plot.plot_contrast_curve` | `plot.plot_radial` plots existing arrays; `plot.plot_radial_image` computes a profile from an image; contrast profiles add working angles and floors | [profiles](../gallery/profiles.md) |
| points colored by value | `plot.scatter_values(x, y, values)` | Inset colorbar | [values](../gallery/values.md) |
| an optical train with chosen planes lit | `plot.rail`, `plot.rail_panels`, `plot.schematic(kind)`, `plot.GLYPHS` | Miniature train, panels under planes | [scene](../gallery/scene.md) |
| complex numbers as arrows | `plot.phasor`, `plot.phase_ring`, `plot.curve_insets` | Phasors, phase key, insets over a curve | [phasor](../gallery/phasor.md) |
| stable color and marker per named source | `style.SourceStyles(names)` (direct), `style.SourceCast(names)` (prepared) | Deterministic source styling | [one-scene-n-views](../gallery/one-scene-n-views.md), [scene](../gallery/scene.md), [prepared](../gallery/prepared.md) |

## 11. Color, motifs and scale

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| keep a quantity's color across a series of clips | `motifs.MotifSet.default().only(*keys)`, `motifs.Motif`, `motifs.AcceptedPair` | One drawing style per recurring object | [declare-a-cast](../how-to/declare-a-cast.md), [motifs](../gallery/motifs.md); `AcceptedPair` in [declare-a-cast](../how-to/declare-a-cast.md) (prose) |
| check a hue keeps one meaning in a clip, and across clips | `testing.meaning_faults(fig, at, times, keys=)`, `testing.meaning_faults_chain(clips)`, `testing.assert_meaning`, `testing.meaning_report` | Colors with two meanings; hue memory across clips; the findings as text | [declare-a-cast](../how-to/declare-a-cast.md) (`meaning_faults`, `meaning_report`), [clip-checks](../reference/clip-checks.md) (prose) |
| check a motif set can be read | `testing.motif_faults`, `testing.motif_margins`, `testing.assert_motifs` | Separation in each mode and under color-vision deficiency | [motifs](../gallery/motifs.md); `assert_motifs` in [contract](../explanation/contract.md) (prose) |
| whether a role spends a hue | `motifs.is_hue_role(role)` | Hue role, gray or rank | API reference |
| size type and strokes for a talk frame | `style.talk_rc(height_px, dpi=)`, `style.TALK_FRACTIONS`, `style.VIDEO_RC` | Fractions of frame height for title, body, ticks, strokes | [size-a-talk](../how-to/size-a-talk.md) |
| video-safe dark colors | `style.use_video(on)`, `style.video_roles()` | The dark-video role table | [contract](../explanation/contract.md) (prose) |
| let slow moves glide | `style.unsnapped(artists)`; `record(snap=False)` is the default | Pixel snapping off | [contract](../explanation/contract.md) (prose) |

## 12. Narration: script and speech

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| write a script with cue marks | `narration.Script(lines, voice=)`, `narration.Line(key, text, sentence_gap_s=)`, `narration.parse_cues(text)`, `narration.Cue` | Lines, voice, and cue marks on words | [narration-design](../explanation/narration-design.md), [speak-a-script](../how-to/speak-a-script.md); `Cue` (prose); `parse_cues` API reference |
| a silent beat in the script | `narration.Pause(key, duration_s, description)` | A beat with no speech | API reference |
| synthesize the speech | `synth.synthesize`, `synth.list_cache`, `synth.prune_cache` | Speaks uncached lines into a cache; run as `python -m eyepiece.narration.synth` | [speak-a-script](../how-to/speak-a-script.md) (the command); `list_cache` and `prune_cache` API reference |
| estimate the length before speaking | `narration.estimate(line)`, `narration.Narration.estimated(script)` | Rough timing for a draft | [estimate-length-before-speaking](../how-to/estimate-length-before-speaking.md) (`estimate`), [motion-clock](../gallery/motion-clock.md) (`estimated`) |
| lay the measured speech on the clock | `narration.Narration.load(script, cache_dir)`, `narration.Narration.place`, `narration.Narration.beat`, `narration.Narration.cue`, `narration.Narration.cues`, `narration.Narration.start`, `narration.Narration.end`, `narration.Narration.total_s` | Beats, words and cue times on one clock | [narrated](../tutorials/narrated.md), [check-timing](../how-to/check-timing.md); `place`, `beat` and `cue` (prose) |
| list what is heard and what is silent | `narration.Narration.audit()` | One row per beat | API reference |
| check every cue mark is handled | `narration.Narration.check(kinds=, targets=, used=)` | Raises on a cue the clip never fires | API reference |
| pair a scientific value on screen with speech | `narration.scientific(value, spec)` | Mathtext and spoken wording from one rounded value; both number gates read the full quantity | [write-numbers-for-speech](../how-to/write-numbers-for-speech.md) |
| write numbers for speech | `narration.say(value, spec, unit=)` | Notation expanded for the voice | [write-numbers-for-speech](../how-to/write-numbers-for-speech.md) |
| find text the voice will misread | `narration.speech_faults(text)`, `narration.check_script_speech(script)`, `narration.lint_script(script)` | Notation and wording findings | [write-numbers-for-speech](../how-to/write-numbers-for-speech.md), [check-narration-and-motion](../how-to/check-narration-and-motion.md) |
| land a counter on its spoken number | `narration.check_number_sync(narration, bindings, plan=)` | Counter reaches the number as it is said | [check-narration-and-motion](../how-to/check-narration-and-motion.md) |
| find silence nobody declared | `narration.check_silence(narration, plan, hero=)` | Undeclared silence and cue gaps | [check-timing](../how-to/check-timing.md) (prose) |
| list sentences with their times | `narration.sentences(narration)` | Sentence timing | API reference |

## 13. Captions, chapters and sound

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| captions in a band | `narration.captions(narration)`, `narration.caption_at(cues, t)` on a text in a band you reserve | Caption cues wrapped to reading limits | [narrated](../tutorials/narrated.md), [burn-captions-into-a-band](../how-to/burn-captions-into-a-band.md) (band built by hand) |
| place captions clear of the figure's text | `narration.caption_line(fig, at=, times=)` | WebVTT `line:` percentage | [motion-attention](../gallery/motion-attention.md) |
| an on-figure caption on the clip's clock | `tl.caption(name, text, leaves=)`, `tl.clear_caption()` | Caption knob that leaves once read or on the next | [timeline-verbs](../how-to/timeline-verbs.md) |
| caption files | `narration.write_captions(narration, stem)`, `narration.write_vtt`, `narration.write_srt` | WebVTT and SRT | [narrated](../tutorials/narrated.md) (`write_captions`); `write_vtt` and `write_srt` in [narration-design](../explanation/narration-design.md) (prose) |
| chapter markers | `narration.write_chapters(narration, path, titles=)` | WebVTT chapters, one cue per beat | [check-narration-and-motion](../how-to/check-narration-and-motion.md) |
| cue sounds | `narration.CueEvents`, `narration.cue_track`, `narration.count_tone`, `narration.pixel_ticks`, `narration.click_sting`, `narration.audio_track(..., cues=)` | Count tone, pixel ticks, click sting | [add-cue-sounds](../how-to/add-cue-sounds.md) (`CueEvents`, `cue_track`; the rest prose) |
| a speech track, and a narrated MP4 | `narration.audio_track(narration, audio_dir, path)`, `narration.mux(video, audio, path)` | WAV of the whole clip; loudness-normalized mux | [narrated](../tutorials/narrated.md) |

## 14. Check a clip

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| check a clip for motion, text and timing faults in one call | `testing.check_clip(fig, draw, plan)`, `testing.assert_clip`, `testing.ClipReport` | Re-renders the plan-driven clip without writing files and collects readability, motion and meaning findings, plan faults, unread knobs and speaking pace | [check-a-clip](../how-to/check-a-clip.md), [motion-checks](../gallery/motion-checks.md); `ClipReport` (prose) |
| check the path between states | `testing.motion_faults(fig, at, times, moves=, steps=, hops=)`, `testing.assert_motion`, `testing.Window` | Jumps, dead moves, drift, tears, overlap | [motion-checks](../gallery/motion-checks.md), [clip-checks](../reference/clip-checks.md); `assert_motion` (prose) |
| declare what a move may move | `MotionContract`, `PlotResult.contract` | The geometry a move promises | [motion-contract](../reference/motion-contract.md) (prose) |
| check text overlap, size and containment | `testing.readability_faults`, `testing.assert_readable` | Readability findings at sample times | [motion-checks](../gallery/motion-checks.md) (`readability_faults`), [motion-attention](../gallery/motion-attention.md) (`assert_readable`) |
| find text halos that should be backing boxes | `testing.halo_faults(fig)`, `testing.assert_no_halos` | Lists visible text with a stroked halo | [clip-checks](../reference/clip-checks.md#halo_faults) |
| check timing: pace, gaps, still screens, numbers | `testing.timing_faults(narration, plan=, source=, at=, readouts=)`, `testing.assert_timing` | Timing findings on the encode or a render callable | [check-timing](../how-to/check-timing.md); `assert_timing` (prose) |
| check speaking pace | `testing.speech_rate`, `testing.assert_speech_rate` | Words per minute per line | [clip-checks](../reference/clip-checks.md) (prose) |
| check equation terms are grounded | `testing.grounding_faults`, `testing.assert_grounded` | Missing referents, wrong colors, early symbols | [ground-a-symbol](../how-to/ground-a-symbol.md); `assert_grounded` (prose) |
| measure how much motion is the quantity | `testing.motion_ink(fig, at, times, driven)` | Share of changed pixels on driven artists | [check-narration-and-motion](../how-to/check-narration-and-motion.md) (through `check_clip`) |
| a contact sheet at the moments that matter | `testing.review_sheet(path, source, narration=, plan=, audio=)` | Frames at cues and events, with the waveform | [check-timing](../how-to/check-timing.md) |
| check frames do not depend on draw order | `testing.frame_faults`, `testing.assert_frames_pure` | Purity diagnostic on sampled frames | [render-a-clip-in-parallel](../how-to/render-a-clip-in-parallel.md); `testing.frame_faults` API reference |
| compare two PNGs | `testing.same_picture(a, b, tol=)` | RGBA comparison | [render-a-clip-in-parallel](../how-to/render-a-clip-in-parallel.md) |

## 15. Record and render

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| animate from a draw function | `animate(fig, draw, frames, fps=)`, `render.Animation.save/.video/.jshtml` | Bind figure, draw and frames, then write | [animation](../gallery/animation.md) |
| record frames from my own loop | `record(fig, *paths, fps=, plan=, stills=, on_rescale=, free_limits=)` | Multi-sink recorder with a frozen layout | [animation](../gallery/animation.md) |
| keep scale fixed | `record(..., on_rescale="raise", free_limits=)`, `render.ScaleDriftError` | Refuses an axis or norm that drifts; `free_limits` exempts the axes a camera moves | [keep-the-scale-fixed](../how-to/keep-the-scale-fixed.md) (through `render.Animation.jshtml`); `render.ScaleDriftError` in [clock](../explanation/clock.md) (prose) |
| a still from the encode | `record(..., plan=, stills={path: span or (span, "mid") or seconds})` | PNGs of chosen frames from the same pass | [stills-from-the-encode](../how-to/stills-from-the-encode.md) |
| raw frames of a fixed size to H.264 | `render.RawSink(path, size_px, fps)` | ffmpeg sink for camera renders | [motion-camera](../gallery/motion-camera.md) |
| pick venue settings and the encoder | `render.PRESETS`, `render.ENCODER_PROFILE` | Venue fps and dpi; H.264 profile | [animation](../gallery/animation.md) |
| render variants in parallel | `render.ClipSpec(target_for=)`, `render.Venue`, `render.Variant`, `render.variants`, `render.render_variants(spec, variants, out_dir, manifest=)`, `render.Manifest`, `render.content_key` | Keyed, chunked, resumable renders of each venue, mode and narration variant, each with parallel frame workers | [render-a-clip-in-parallel](../how-to/render-a-clip-in-parallel.md); `render.Variant` (prose) |
| render one clip in chunks | `render.render_frames(build, frames, path, workers=)`, `render.RenderResult` | Independent frames, encoded in order | [render-a-clip-in-parallel](../how-to/render-a-clip-in-parallel.md) |
| a poster frame or a deterministic still | `render.render_poster`, `render.render_still(..., **savefig_kwargs)`, `save_fig` | PNG of one frame; styled save | [render-a-clip-in-parallel](../how-to/render-a-clip-in-parallel.md), [still](../tutorials/still.md) |
| a scrubber page for one knob | `render.to_scrubber(fig, draw, plan, knob, samples)` | Self-contained HTML slider | [publish-a-scrubber-page](../how-to/publish-a-scrubber-page.md) |
| a provenance stamp | `render.stamp(fig, script=, seed=)`, `render.provenance_text`, `render.provenance_fields`, `render.file_metadata` | Footer line and file metadata | [values](../gallery/values.md); the last three are API reference |

## 16. One preparation, many outputs

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| describe data once for a still, a strip, a movie and Manim | `prepared.ImageView`, `prepared.CurveView`, `prepared.TrackView`, `prepared.PanelGroup`, `prepared.AxisSpec`, `prepared.Scale`, `prepared.Path`, `prepared.Points`, `prepared.ReferenceLine`, `prepared.Region`, `prepared.Label`, `prepared.View`, `prepared.Mark` | Plain records with units, masks and scales; `View` and `Mark` are the unions of the view and mark records | [prepared-views](../explanation/prepared-views.md), [prepared](../gallery/prepared.md); `View` and `Mark` API reference |
| a replayable sequence over physical time | `prepared.Sequence`, `prepared.ArrayChannel`, `prepared.PathWindow`, `prepared.Clock`, `prepared.Sample` | Fixed-topology sequence; acquisition-time clock label | [prepared](../gallery/prepared.md); `Sample` in [rendering-in-manim](../explanation/rendering-in-manim.md) (prose) |
| render it with Matplotlib | `mpl.render(view)`, `mpl.animate(sequence, run_time=, fps=)`, `mpl.MplResult` | Still and movie | [prepared](../gallery/prepared.md); `MplResult` in [prepared-views](../explanation/prepared-views.md) (prose) |
| render it in Manim | `manim.render(view)`, `manim.animate(sequence)`, `manim.ManimResult`, `manim.ManimClip`, `manim.Units`, `manim.ensure_font` | Native mobjects and playback | [rendering-in-manim](../explanation/rendering-in-manim.md); `ManimResult`, `ManimClip` and `ensure_font` (prose) |
| freeze the appearance per venue | `style.snapshot_profile()`, `style.RenderProfile` | Read-only snapshot of the style | [prepared](../gallery/prepared.md); `RenderProfile` in [prepared-views](../explanation/prepared-views.md) (prose) |
| edit or display a prepared tree | `prepared.find_element`, `prepared.replace_elements`, `prepared.resolve_bounds`, `prepared.normalize_values`, `prepared.map_rgba`, `prepared.weight_opacity` | Tree edits and display mapping | [prepared-views](../explanation/prepared-views.md); `map_rgba` (prose) |

## 17. Results, findings, errors and constants

Every primitive returns a `PlotResult` or a `MosaicResult` (`ax` or `axes`, `artists`, `update`
and `contract`), described in [contract](../explanation/contract.md). Each fault check returns a
list of fault records and has an `assert_*` twin that raises the matching error; `motion_ink`,
`motif_margins` and `speech_rate` are measurements and return values instead. The remaining public
names are the records those calls return and the constants they read.

| I want to... | Use | What it does | Shown in |
|---|---|---|---|
| read what a primitive drew | `PlotResult`, `MosaicResult`, `plot.ARTIST_KEYS` | The axes, the artists keyed by the common `plot.ARTIST_KEYS` vocabulary (some primitives add keys of their own), `update` and `contract` | [contract](../explanation/contract.md) (prose) |
| read photon counts and arrivals | `motion.ApertureCounts`, `motion.Arrivals` | Records returned by `motion.aperture_counts` and `motion.arrivals` | API reference; [motion-clock](../gallery/motion-clock.md) calls the functions that return them |
| read the output of a render | `render.RenderResult`, `render.VariantResult` | The encoded clip and its work; one rendered or kept variant | [render-a-clip-in-parallel](../how-to/render-a-clip-in-parallel.md) (`render.RenderResult`); `render.VariantResult` API reference |
| read the narration on the clock | `narration.NarrationBeat`, `narration.CueTime`, `narration.Spoken`, `narration.Word`, `narration.Cue` | One placed line; one placed cue; one synthesized line and its words; one parsed cue mark | [check-timing](../how-to/check-timing.md) (`Spoken`, `Word`); `Cue` in [narration-design](../explanation/narration-design.md) (prose); the rest API reference |
| read narration findings | `narration.SpeechFault`, `narration.ScriptFault`, `narration.NumberSyncFault`, `narration.SilenceFault` | Findings of `speech_faults`, `lint_script`, `check_number_sync` and `check_silence` | `SpeechFault` in [narration-design](../explanation/narration-design.md) (prose); the rest API reference |
| catch a narration error | `narration.SpeechTextError`, `narration.MissingSynthesisError`, `narration.MissingAudioError` | Misreadable notation; a line never synthesized; timing with no audio | `SpeechTextError` in [narration-design](../explanation/narration-design.md) (prose); the rest API reference |
| key the speech cache | `narration.line_hash`, `narration.KOKORO_MODEL`, `narration.KOKORO_VERSION` | Cache key of a spoken line; the speech model and its pinned version | API reference |
| read clip-check findings | `testing.ClipReport`, `testing.HaloFault`, `testing.MotionFault`, `testing.ReadabilityFault`, `testing.MeaningFault`, `testing.MotifFault`, `testing.GroundingFault`, `testing.TimingFault`, `testing.SpeechRate` | One record per finding; speaking pace per line | [clip-checks](../reference/clip-checks.md) (prose: `MotionFault`, `GroundingFault`, `TimingFault`); `ClipReport` in [motion-checks](../gallery/motion-checks.md) (prose); the rest API reference |
| catch a clip-check error | `testing.ClipError`, `testing.HaloError`, `testing.MotionError`, `testing.ReadabilityError`, `testing.MeaningError`, `testing.MotifError`, `testing.GroundingError`, `testing.TimingError`, `testing.SpeechRateError` | Raised by the `assert_*` twins | [clip-checks](../reference/clip-checks.md) (prose); `ClipError` in [motion-checks](../gallery/motion-checks.md) (prose); `MotifError` in [contract](../explanation/contract.md) (prose) |
| list the kinds a check reports | `testing.KINDS`, `testing.MOTION_KINDS`, `testing.MEANING_KINDS`, `testing.MOTIF_KINDS`, `testing.GROUNDING_KINDS`, `testing.TIMING_KINDS` | The fault kinds of the readability, motion, meaning, motif, grounding and timing checks | `TIMING_KINDS` in [clip-checks](../reference/clip-checks.md) (prose); the rest API reference |
| size text for a venue | `testing.VENUES`, `testing.VENUE_FPS` | Smallest readable text size in output pixels per venue; the rate a clip is sampled at when `times` is a `testing.Window` | API reference |
| catch a scale or connector problem | `render.ScaleDriftError`, `testing.AttentionFault` | Raised by `record(on_rescale="raise")`; raised by `testing.audit` when a strict leader or link breaks a rule | `render.ScaleDriftError` in [clock](../explanation/clock.md) (prose), `testing.AttentionFault` in [attention-moves](../reference/attention-moves.md) (prose) |
| check the installed version | `__version__` | The package version, available only at `ep.__version__` | API reference |

## Deprecated names

These names still work for one minor release and warn with a `DeprecationWarning` that names the
replacement. They are out of every `__all__`, and they go away in the next minor release.

| Deprecated | Use instead | Note |
|---|---|---|
| `testing.Span(start, stop)` | `testing.Window(start, stop)` | The same check window, renamed apart from the timeline record `clock.Span` |
| `radial_profile_plot(image, pixscale_lod)` | `plot.plot_radial_image(image, pixscale_lod)` | The same image-to-profile computation and plotting; [profiles](../gallery/profiles.md) |
| `spans(x, mask)` | `plot.shade_intervals(x, mask)` | The same function and signature |
| `motifs.resolve_role(role, mode)` | `style.role(name, mode=)` | `style.role` also takes the neutral ranks |
| `tl.pulse(target)` | `tl.indicate(target)` | The verb is named for the move it places; its knob is `indicate.<target>` |
| `tl.ring(target)` | `tl.circumscribe(target)` | The same, with the knob `circumscribe.<target>` |
| `tl.span(..., kind="pulse")`, `kind="ring"` | `kind="indicate"`, `kind="circumscribe"` | Span kinds follow the verbs |
| `render.PRESETS["talk_60"]` | `render.PRESETS["talk"]` | The same fps and dpi pair |

`motion.brush(dim=)` is a deprecated spelling of `motion.brush(veil_alpha=)`, with the same
opacity value. `motion.Stage(dim=)` continues to mean contrast retained.
