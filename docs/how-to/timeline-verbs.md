---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Schedule departures, groups, captions, and hero beats

Use these recipes when a clip needs more than sequential entries and moves.
Each example builds a small `Timeline`, resolves it at 10 frames per second,
and prints its schedule. Times in `plan.spans_json()` are seconds rounded to
the millisecond; `plan.spans` also carries frame bounds and span metadata.
The recipes share imports, but each starts a fresh timeline.

The clock schedules presentation knobs. Connect those knobs to existing
artists with {doc}`bind-a-move-to-a-knob`; keep the data and display scales
fixed as described in {doc}`the primitive contract <../explanation/contract>`.

## Let a subject settle, then leave

`settle()` adds a hold of `Tempo.rest_s`, after the subject has arrived.
`leave()` ramps its subject knob back to zero and names the span
`"<subject>.leave"`. Here the subject arrives over 0.6 s, rests for 0.5 s,
and leaves over 0.3 s. `also=` makes its label leave on the same clock.
The tempo factor scales all three durations.

```{code-cell} python
import json

import matplotlib.pyplot as plt

import eyepiece as ep

tl = ep.clock.Timeline(tempo=ep.clock.Tempo(factor=1.0, settle_s=0.5))
tl.enter("planet", also=("label",))
rest = tl.settle()
gone = tl.leave("planet", also=("label",))
plan = tl.at_fps(10)
print(plan.spans_json())
print("rest:", plan.spans[rest.name].kind,
      "last opacity:", plan.value(plan.frame_of(gone.name, "last"), "planet"))
```

## Start a group together

Inside `together()`, each child starts at the current cursor. On exit, the
cursor moves to the latest child end, so the following hold starts at 0.8 s.
Declare intentional overlap with `with_=`: grouping does not disable the
rule that subjects normally enter one at a time.

```{code-cell} python
tl = ep.clock.Timeline()
with tl.together():
    first = tl.enter("planet", 0.4)
    tl.enter("orbit", 0.8, with_=(first.name,))
tl.hold(0.4, name="inspect")
plan = tl.at_fps(10)
together_plan = plan
print(plan.spans_json())
print("cursor:", float(tl.now))
```

## Stagger whole spans

`tl.stagger()` calls a named verb once per target. `each_s` is the interval
between starts in seconds; `seconds` is the nominal duration passed to each
entry. Use an interval at least as long as the paced entry duration so the
subjects arrive one at a time. This group starts at 0.0, 0.5, and 1.0 s;
the following hold starts at the last entry's end, 1.4 s.

```{code-cell} python
tl = ep.clock.Timeline()
tl.stagger("enter", ["planet", "orbit", "label"], each_s=0.5, seconds=0.4)
tl.hold(0.4, name="inspect")
plan = tl.at_fps(10)
print(plan.spans_json())

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), dpi=90,
                         layout="constrained", sharex=True)
for ax, schedule, title in zip(
    axes, (together_plan, plan), ("Together", "Stagger"), strict=True
):
    spans = list(schedule.spans.values())
    for row, span in enumerate(spans):
        ax.barh(row, span.end_s - span.start_s, left=span.start_s,
                height=0.55, color=ep.style.role("data"))
    ax.set_yticks(range(len(spans)), [span.name for span in spans])
    ax.invert_yaxis()
    ax.set(xlabel="presentation time [s]", title=title, xlim=(0, 1.9))
plt.show()
```

`tl.stagger("indicate", ...)` and `tl.stagger("circumscribe", ...)` place
attention spans without advancing the cursor, since those spans do not
extend their section. Add a hold to keep them in the clip. The separate
function `ep.clock.stagger(u, rank, span=...)` divides the progress of one move
among its items; it does not place timeline spans.

## Read a cue's word time

`tl.cue("kind:target")` returns the word start as a `Time`, without a lead.
Open the line first. This example supplies word times directly, so it needs
no synthesis or audio files: the word "planet" starts at 1.0 s. Passing the
same mark to `enter(cue=...)` applies the entry lead once, starting the entry
at 0.9 s. `at=tl.cue(...)` instead would start exactly on the word.

```{code-cell} python
script = ep.narration.Script((ep.narration.Line("reveal", "Here is the {show:planet}planet."),))
narration = ep.narration.Narration(script, {
    "reveal": ep.narration.Spoken("reveal", 1.6, (
        ep.narration.Word("Here", 0.0, 0.3),
        ep.narration.Word("is", 0.3, 0.5),
        ep.narration.Word("the", 0.5, 1.0),
        ep.narration.Word("planet.", 1.0, 1.6),
    )),
})
tl = ep.clock.Timeline(narration=narration)
tl.line("reveal")
word = tl.cue("show:planet")
tl.enter("planet", cue="show:planet")
plan = tl.at_fps(10)
print("word:", float(word), "entry:", plan.spans_json()["planet"])
```

A cue must be in the current or an earlier line. In a silent timeline,
`tl.cue()` returns `now`; it does not invent word times. For a complete
spoken clip, see {doc}`../tutorials/narrated`.

## Declare the hero beat

Give the clip's single main move a name, then pass its handle or name to
`hero()`. Here the 1.2 s sweep is the hero window; the final 0.8 s hold is
outside it. The declaration adds no motion or silence. Narration and timing
checks use it to recognize deliberate silence, and report speech over the
hero or a second hero declaration. See {doc}`check-timing` for the checks.

```{code-cell} python
tl = ep.clock.Timeline({"reveal": 0.0})
sweep = tl.run("sweep", 1.2, to={"reveal": 1.0})
tl.hero(sweep)
tl.hold(0.8, name="answer")
plan = tl.at_fps(10)
print(plan.spans_json())
print("hero windows:", plan.hero_windows())
```

## State a custom span's defaults

Use `span()` when a named verb does not express the timing you need. This
one writes a presentation knob over exactly 1 s with a cosine ease.
`paced=False` keeps that duration even at tempo factor 2; `extends=True`
keeps its section open through its end. The following ordinary 0.5 s hold
is paced to 1 s. `kind` selects existing timeline rules; it is not a new
event type. A progress-only span can omit `to=` and use `plan.progress()`.

```{code-cell} python
tl = ep.clock.Timeline({"blend": 0.0}, tempo=ep.clock.Tempo(factor=2.0))
custom = tl.span("crossfade", 1.0, kind="move", to={"blend": 1.0},
                 paced=False, ease="cosine", extends=True)
tl.hold(0.5, name="inspect")
plan = tl.at_fps(10)
print(plan.spans_json())
print("ease:", plan.spans[custom.name].ease,
      "extends:", plan.spans[custom.name].extends)
```

## Give a silent caption time to be read

`caption()` creates a 0.4 s arrival and writes `caption.<name>`. In a
silent clip the default lifetime is `leaves="read"`: closing the section
waits for arrival plus `Tempo.read_s(text)`, or for its last extending
span, whichever is later, then schedules the leave. This two-word caption
gets 1 s to be read under the tempo stated here. The plan schedules text;
the draw function still owns the text artist and its opacity.

```{code-cell} python
tl = ep.clock.Timeline(tempo=ep.clock.Tempo(factor=1.0, base_s=0.5, word_s=0.25))
tl.caption("result", "Planet found")
plan = tl.at_fps(10)
print(plan.spans_json())
print("read seconds:", plan.spans["result"].read_s)
```

## Replace a caption, then clear it

`leaves="next"` keeps a caption pending until another caption arrives or
`clear_caption()` takes it down. The second caption begins as the first
leaves. Each `read()` below supplies the silent reading time before that
caption is removed. Clearing without `at=` advances the cursor through
the 0.3 s departure.

```{code-cell} python
tl = ep.clock.Timeline(tempo=ep.clock.Tempo(factor=1.0, base_s=0.5, word_s=0.25))
tl.caption("first", "Planet found", leaves="next")
tl.read("Planet found", name="read_first")
tl.caption("second", "Orbit measured", leaves="next")
tl.read("Orbit measured", name="read_second")
gone = tl.clear_caption()
plan = tl.at_fps(10)
print(plan.spans_json())
print("cleared:", gone.name, "cursor:", float(tl.now))
```

In a narrated clip `leaves="next"` is the default, and speech supplies
the reading time. `leaves="never"` keeps a caption for the rest of the
clip and does not queue it for `clear_caption()`; use `"next"` for a caption
you intend to clear. On-screen captions differ from subtitle tracks made
by `ep.narration.captions`; see {doc}`burn-captions-into-a-band` for subtitles.

## Land a click and keep its hold

`tl.click()` creates a linear `click.<target>` knob over 0.6 s and follows
it with a separate 1.5 s hold. Both durations stay fixed under the tempo,
so this clip lasts 2.1 s even at factor 0.5. Use the knob with the visual
`ep.motion.click` move, binding it as in {doc}`bind-a-move-to-a-knob`.

```{code-cell} python
tl = ep.clock.Timeline(tempo=ep.clock.Tempo(factor=0.5))
tl.click("answer")
plan = tl.at_fps(10)
print(plan.spans_json())
print("kinds:", {name: span.kind for name, span in plan.spans.items()})
```

With `at=`, the click and its following hold keep their relative timing,
and the cursor stays where it was, as for a placed `run()`.

## Export span times for a manifest

`spans_json()` returns a dictionary ready for JSON serialization. The
values preserve declared seconds rounded to milliseconds; frame bounds
remain in `plan.spans`. `holds=False` removes every span whose kind is
`"hold"`, including settles, reads, and the hold after a click, while
leaving the other times unchanged.

```{code-cell} python
tl = ep.clock.Timeline()
tl.enter("planet", 0.6333333)
tl.settle()
tl.move("measure", 0.7333333)
plan = tl.at_fps(10)
print(json.dumps(plan.spans_json(), indent=2))
print("without holds:", plan.spans_json(holds=False))
```

For entry, text, motion, holds, reading, sections, and frame queries in a
larger clip, continue with {doc}`../gallery/motion-clock`. For the time and
frame rules, see {doc}`../explanation/clock`.
