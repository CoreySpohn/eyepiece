# Narration

`eyepiece.narration` gives a clip a voice. A script of lines is spoken once,
the word times and the audio are kept beside the clip, and every later render
reads them without a speech model. A cue mark in the text fires an attention
move on the word that names its subject, and the length of each line sets the
length of the part of the clip that shows it.

The code on this page is shown rather than executed.

## The script

A `Script` is a tuple of `Line`. A line is one part of the clip, spoken as a
unit. A mark of the form `{kind:target}` sits in front of the word it names
and is removed before the text is spoken.

```python
from eyepiece import narration as nr

SCRIPT = nr.Script(
    (
        nr.Line("open", "The coronagraph blocks the {indicate:star}starlight."),
        nr.Line(
            "leak",
            "What leaks through is {indicate:speckle}speckle, and the "
            "{circumscribe:planet}planet must stand out.",
            pad_after_s=0.8,
        ),
    )
)
```

The kind is the move the clip runs (`show`, `indicate`, `circumscribe`,
`move`, or any name the clip understands) and the target names what it acts
on. Write the
numbers in a line as they appear on the figure, and read them from the
figure's source rather than typing them twice. The voice defaults to Kokoro
`bm_george`, a British English voice, with language code `"b"`.

## Speaking it

Kokoro pulls in torch, so the synthesis runs in an environment of its own and
writes a cache that belongs in the repository:

```bash
PYTHONPATH=<eyepiece>/src uv run --no-project --python 3.12 \
    --with kokoro==0.9.4 --with soundfile --with numpy \
    python -m eyepiece.narration.synth script.py cache_dir
```

`script.py` defines `SCRIPT`. For each line the command writes `<key>.opus`
(Ogg Opus, or FLAC when the installed libsndfile cannot write Opus) and
`<key>.json` into `cache_dir`, where the key hashes the spoken text, the
voice, the speed, the language, the engine and its version, and the model.
Marks are not part of the key, so moving a cue never speaks a line again, and
changing a word speaks only that line.

A line whose JSON and audio are both present is never spoken again, so the
committed pair stays the pair the cue times were measured on. A JSON whose
audio is missing is an error, since speaking it again can move its cues;
`--respeak` says that is intended. `--force` speaks every line again,
`--prune` deletes files that no line uses, and `--list` prints the cache.

The command sets `HF_HUB_DISABLE_XET=1` for itself. Without the extra it fails
with an `ImportError` that names it. Misaki, the phonemizer Kokoro uses, may
download a small language model on the first run and may need espeak-ng for
words outside its dictionary.

## Laying it out

`Narration.load` reads the cache and places the lines one after another. A
`Narration` is placement only: it knows where each line and each word falls on
the clip's clock and nothing about what the clip does with them.
`Narration.place(key, start_s, end_s=None)` lays one line on the clock, purely;
a cue is the start of its word, with no lead.

The clip places its moves on the words through `eyepiece.Timeline`. Open each
line with `line(key)`, in script order, and write the verbs of that line. A
verb with no `at=` starts on its mark: `enter`, `text` and `caption` on
`show:<name>` a tenth of a second ahead of the word, `pulse` on
`indicate:<name>` centered on the word, `ring` on `circumscribe:<name>`
starting on the word, and `move` on `move:<name>`. A mark in the current or an
earlier line can be named with `cue="kind:target"`, and a mark in a later line
cannot be waited on. A line lasts until its speech and padding end or until
its last moving span ends plus 0.6 s, whichever is later, and the next line
starts there; the pulses and rings of a line never hold it open. This replaces
the hand-typed holds, the per-line minimum lengths and the second layout pass.
A span whose mark falls while the cursor is still busy starts late; up to
`max_late_s` (0.5 s by default, a `Timeline` argument) that is noted, and
beyond it the timeline raises, naming the span, its cue word and the seconds.
Line keys and section names have a namespace of their own, so a line and a
span may share a name. A mid-clip still is the last frame of a span with
`stills={"x.png": (span, "last")}`, not `span.end`.

`Narration(min_s=)` and `Narration.load(..., min_s=)` still work for one
minor release and warn with a `DeprecationWarning`: the value is a duration
from the start of each line, and `ends=` (an absolute end time) or a
`Timeline` replaces it.

```python
import eyepiece as ep

narration = nr.Narration.load(SCRIPT, "narration/clip")
tl = ep.Timeline(narration=narration, tempo=ep.Tempo())
tl.line("open")
tl.pulse("star")
tl.line("leak")
tl.pulse("speckle")
tl.ring("planet")
plan = tl.at_fps(30)

for k, frame in enumerate(plan.frames):
    ...  # draw the frame as for any Plan

cue = plan.events(what="cue")[0]  # name, t_s and the frame that shows it
solved = tl.narration  # the narration with each line's solved start and end
```

`Timeline.narration` rebuilds the `Narration` with those starts and ends, and
that object is the one `audio_track`, `mux` and the captions take. A line must
be opened for every line of the script, and a mark that no verb fires is a
fault, so a retyped line cannot leave a cue silent. The same function with
`narration=None` builds the silent clip: `section(name)` replaces `line(key)`,
captions leave once read, and a pulse or ring placed only by a cue is skipped
and recorded in `plan.faults` as information. `cues` lists every cue, and
`check(kinds=..., targets=...)` raises on a mark the clip does not handle, and
with `used=True` on one it never asked for. Narration measures the pace of the
speech, so it takes the place of `Tempo` for the lines it covers. A transition
is eased and overlapped with the speech, for example with
`Stage.weigh(levels, ease="smoothstep")`.

## Audio and captions

```python
track = nr.audio_track(narration, "narration/clip", "track.wav")
nr.mux("silent.mp4", track, "narrated.mp4")
vtt, srt = nr.write_captions(narration, "narrated")
```

`audio_track` places each line's audio at its speech start and refuses a file
whose length differs from the duration its JSON records, which means the
audio and the cue times come from different syntheses. The lines are resampled
to 48 kHz (`rate`) and each line gets an 8 ms raised-cosine fade at both ends
(`fade_s`).

`mux` copies the picture and encodes the audio as AAC after a two-pass
`loudnorm`: the first pass measures the track, the second applies a linear
gain, so the speech keeps its dynamics. The target is `loudness=-16.0` LUFS
integrated with `true_peak=-1.0` dBTP, the usual level for spoken web audio
(Kokoro alone comes out near -25 LUFS). Pass another `loudness`, such as
-23.0 for broadcast, or None to skip the step. After the normalization `mux`
adds a seeded white-noise bed at `room_tone_db=-60.0` dBFS under the whole
track, so the voice does not cut to digital silence between lines and the
bed is at that level in the muxed file whatever gain the speech needed. The
same seed gives the same file; `room_tone_db=None` adds no bed. `audio_track`
carries no bed by default; with ``loudness=None`` give the bed to either
`audio_track` or `mux`, not both.

`captions` is the list of cues behind `write_srt` and `write_vtt`. A sentence
is one cue unless it does not fit. A cue holds at most 42 characters on a line
and two lines, and lasts at most 7 s of speech; a longer sentence splits at a
comma, colon, semicolon or dash, or at its most even word break when it has
none, and the lines of a cue break at a clause boundary when that leaves two
even lines. A cue is held for at least `min_s` (1 s) and long enough to read
at 20 characters per second, but never into the next cue. All the limits are
keywords. A browser reads WebVTT in a `<track kind="captions">` element, not
SRT.

A line may carry `display=` text, which the captions show in place of the
spoken words (`nr.Line("post", "It is P of H given E.", display="It is
Pr(H|E).")`). The cue marks, the cache key and the cue times belong to the
spoken text. When the display text has as many sentences as the speech, each
sentence keeps the span of its spoken sentence. Otherwise the whole line is
one run over the line's speech, and a split inside it is timed by character
count, not by the words.

Captions sit at the bottom of the picture unless a cue says otherwise. Where
the figure has text there, `eyepiece.testing.caption_line(fig, at=pose,
times=...)` returns the WebVTT `line:` percentage of a position that no visible
text touches at the sampled times, and `write_vtt(..., line=...)` writes it on
every cue. The alternative is a caption band reserved in the clip layout, a
strip no panel or label uses, with its percentage passed to `write_vtt`
directly. `caption_line` raises when no position is clear, which is the sign
that a band is needed. A narrated clip should not loop or start muted, and a
silent variant of the same content stays the default where the clip is not
designed to be spoken over.

## Licenses

Kokoro and its voices are used as published by their authors. Check the
current terms of the model weights, the voice packs, and the phonemizer
before a narrated clip is published.
