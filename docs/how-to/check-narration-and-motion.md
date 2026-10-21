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

# Check a narrated clip

Each check below reports; none changes the clip. The examples are short and
use `ep.narration` for speech and `ep.testing` for clip checks, after
`import eyepiece as ep`.

## Land a counter on its spoken number

`ep.narration.check_number_sync` ties a number word of the narration to the readout that
shows it. A number lands on or before its word. The values are the readout's
shown value at each frame, as `(time_s, value)` pairs, and the span may run
past the end of the line so a late counter is measured.

```python
faults = ep.narration.check_number_sync(
    narration,
    [{"word_key": ("ratio", 3), "span": "count", "values": samples}],
    plan=plan,
)
```

A fault has the code `late-settle` (it names the lag in seconds) or
`contradicts` (a final value that is not the spoken one, an overshoot, a
count that backs up, or a readout that leaves the value after reaching it).
With a plan the settle tolerance is floored at one frame.

## Lint a script before speaking it

`ep.narration.lint_script` returns findings for a line with several sentences, a
question after a statement, many numbers, or a number with no cue. It
never changes the script.

```python
for fault in ep.narration.lint_script(script):
    print(fault.code, fault.key, fault.message)
```

To refuse a script with findings, pass `rules=True` to
`ep.narration.check_script_speech` or to `synthesize`. Both default to off, so a script
written before the rules still synthesizes.

## Report how much of the motion is the quantity

`check_clip` takes the artists a knob drives and reports `motion_ink`, the
share of changed pixels (0 to 1) that lie on them, or None when none are
given. It is a number to read, never a fault.

```python
report = check_clip(fig, draw, plan, driven=[readout_text, needle])
print(report.motion_ink)
```

## Declare artists that redraw in cuts

A group of artists that redraws by design, such as exchangeable draws, is
declared with `hops=`. Its cuts are excused; a rate above 8 draws per second
is reported as `hop-rate`.

```python
motion_faults(fig, at, times, hops=[{"artists": [draws], "rate": 4.0}])
```

## Write chapters

`ep.narration.write_chapters` writes a WebVTT chapters file with one cue per beat, from
the beat's start to its end, titled by the line key unless `titles` maps the
key to a title. An empty title is refused.

```python
ep.narration.write_chapters(narration, "out/chapters.vtt", titles={"open": "The problem"})
```

`write_captions(..., chapters=True)` writes the same file beside the captions.

## Pause between sentences

`Line(sentence_gap_s=0.4)` speaks the line a sentence at a time and joins the
sentences with that much silence, each sentence cached on its own. The default
(None) speaks the line whole, with the keys and output it always had.

```python
ep.narration.Line("why", "It is faint. The star is bright.", sentence_gap_s=0.4)
```

To check pace, gaps, still screens and numbers in one call, see {doc}`check-timing`.
