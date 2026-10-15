# Narration

`eyepiece.narration` gives a clip a voice. A script of lines is spoken once,
the word times and the audio are kept beside the clip, and every later render
reads them without a speech model. A cue mark in the text fires an attention
move on the word that names its subject, and the length of each line sets the
length of the beat that shows it.

The code on this page is shown rather than executed.

## The script

A `Script` is a tuple of `Line`. A line is one beat of the clip, spoken as a
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

The kind is the move the clip runs (`indicate`, `circumscribe`, or any name
the clip understands) and the target names what it acts on. Write the
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
beat lasts as long as its speech plus the line's padding, or `min_s` for the
motion that must finish inside it, whichever is longer. `min_s` replaces a
hold typed by hand.

```python
narration = nr.Narration.load(
    SCRIPT, "narration/clip", lead_s=0.15, min_s={"open": 3.0}
)
plan = narration.timeline({"title": "none"}).at_fps(30)

for k, frame in enumerate(plan.frames):
    ...  # draw the frame as for any Plan

k = narration.cue_frame(plan, "leak", "indicate", "speckle")
```

`Narration.timeline` returns an `eyepiece.Timeline` with one hold per line,
so the clip keeps its knobs and its `Plan`; `cue_frame` is the frame of that
plan on which a cue fires, kept inside its own beat. `cues` lists every cue,
and `check(kinds=..., targets=...)` raises on a mark the clip does not
handle, and with `used=True` on one it never asked for. Narration measures
the pace of the speech, so it takes the place of `Tempo` for the lines it
covers. A transition is eased and overlapped with the speech, for example with
`Stage.weigh(levels, ease="smoothstep")`.

## Audio and captions

```python
track = nr.audio_track(narration, "narration/clip", "track.wav")
nr.mux("silent.mp4", track, "narrated.mp4")
nr.write_vtt(narration, "narrated.vtt")
nr.write_srt(narration, "narrated.srt")
```

`audio_track` places each line's audio at its speech start and refuses a file
whose length differs from the duration its JSON records, which means the
audio and the cue times come from different syntheses. `mux` copies the
picture and encodes the audio as AAC. A browser reads WebVTT in a
`<track kind="captions">` element, not SRT. A narrated clip should not loop
or start muted, and a silent variant of the same content stays the default
where the clip is not designed to be spoken over.

## Licenses

Kokoro and its voices are used as published by their authors. Check the
current terms of the model weights, the voice packs, and the phonemizer
before a narrated clip is published.
