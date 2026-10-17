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

# Tutorial 3: a narrated build-up

Rebuild the field and row profile from {doc}`build-up`, then place their
reveals on the words that introduce them. The committed word timings drive
the animation and captions, and the committed speech audio voices the clip,
so this page needs no speech engine. Its embedded clip is narrated, with
captions burned into a band.

## Rebuild the finished figure

The shared `Scene` builds the same synthetic field, central row profile,
marker, and reveal artists as tutorial 2. A taller canvas reserves the bottom
band for two caption lines. Both this page and the standalone renderer use
`clip.py`, so they draw the same frames.

MyST-NB executes this notebook in `docs/tutorials`, because `docs/conf.py`
leaves `nb_execution_in_temp` at its default of `False`. Resolve the helper
directory relative to that working directory, then import the scene and script.

```{code-cell} python
import base64
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from IPython.display import HTML

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece import narration as nr

tutorial_dir = Path("../narration/tutorial").resolve()
sys.path.insert(0, str(tutorial_dir))
from clip import Scene, caption_steps, make_timeline
from script import SCRIPT

scene = Scene()
fig = scene.fig
```

Here is the figure construction, including the reserved band and its text artist.

```{literalinclude} ../narration/tutorial/clip.py
:language: python
:pyobject: Scene.__init__
```

## Place the reveals on spoken words

`Narration.load` reads only the timing JSON in `cache`. Each `line(key)`
opens the next spoken line. The `enter` verbs use `show:panel`, `show:image`,
and `show:profile`, while `pulse` uses `indicate:marker`. Entries start
0.1 seconds before their marked word, and the pulse is centered on its word.
The final `close` line keeps the finished figure visible while it is spoken.

```{literalinclude} ../narration/tutorial/clip.py
:language: python
:pyobject: make_timeline
```

The entry durations retain tutorial 2's 1.5, 1.5, and 2 seconds. The pulse
uses the narration verb's 0.9-second duration. Speech and line padding set
the holds between these spans, replacing the silent clip's fixed holds.

```{code-cell} python
cache_dir = tutorial_dir / "cache"
narration = nr.Narration.load(SCRIPT, cache_dir)
tl = make_timeline(narration)
plan = tl.at_fps(10)
cues = nr.captions(tl.narration)
print(f"{plan.n_frames} frames at {plan.fps:g} fps")
```

Use `tl.narration` after solving the plan, since it carries the line starts
and ends chosen by the timeline. Captions and audio must share those times.

## Draw the captioned clip

Every artist is set from `plan.value`, and the caption text is selected at
`k / plan.fps`. `caption_at` includes each cue's start and excludes its end,
clearing the band between cues. Absolute frame state lets the checker
revisit frames in any order.

```{literalinclude} ../narration/tutorial/clip.py
:language: python
:pyobject: Scene.draw
```

Adapt the shared draw method to a function of the frame index.

```{code-cell} python
def draw(k):
    scene.draw(k, plan, cues)
```

Write separate caption files in a temporary directory. The WebVTT `line`
setting places optional player captions at the same band height.

```{code-cell} python
with TemporaryDirectory() as directory:
    vtt, srt = nr.write_captions(
        tl.narration, Path(directory) / "narrated", line=90
    )
    print(vtt.read_text().splitlines()[0])
    print(f"Wrote {vtt.suffix} and {srt.suffix} captions")
```

## Add speech and play the movie

The speech audio for each line is committed beside its timing JSON in
`cache`; it was regenerated with the narration tool, as described in
{doc}`../how-to/speak-a-script`. Record the silent frames to MP4 at 80 dpi,
assemble the audio track from the solved narration, and mux the two. This
cell runs the export in a temporary directory and embeds the result as an
HTML5 player. It requires ffmpeg.

```{code-cell} python
with TemporaryDirectory() as directory:
    work = Path(directory)
    with ep.record(fig, work / "silent.mp4", plan=plan, dpi=80) as rec:
        for k in range(plan.n_frames):
            draw(k)
            rec.frame()
    track = nr.audio_track(tl.narration, cache_dir, work / "track.wav")
    movie = nr.mux(work / "silent.mp4", track, work / "narrated.mp4")
    data = base64.b64encode(Path(movie).read_bytes()).decode("ascii")
HTML(
    '<video controls style="max-width: 100%" '
    f'src="data:video/mp4;base64,{data}"></video>'
)
```

Speech is synthesized with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(Apache-2.0), voice `bm_george`.

From the repository root, `python docs/narration/tutorial/render.py /tmp/tutorial-3`
runs the same export with the shared figure and writes the WebVTT and SRT files.

## Check the clip

Use the same readability, motion, and plan checks as tutorial 2. Declare the
curve's reveal span through `scene.moves`. This figure has no declared
meaning keys, and its plan now carries narration for the speech-rate check.
The report covers the caption text as part of figure readability. Declare
caption starts and ends as steps, since replacing text changes the picture
in a single frame.

```{literalinclude} ../narration/tutorial/clip.py
:language: python
:pyobject: caption_steps
```

The narration covers every line, so the checker also measures the speech rate
against its word limit. Any figure, motion, plan or speech fault stops the
build.

```{code-cell} python
report = ept.assert_clip(
    fig, draw, plan, moves=scene.moves, steps=caption_steps(plan, cues)
)
print(report)
```
