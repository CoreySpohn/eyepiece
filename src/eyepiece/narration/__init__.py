"""Narration: speech for an eyepiece clip, with attention moves on the words.

Two halves keep a render reproducible without a speech model installed.

Synthesis (`eyepiece.narration.synth`, needs the ``narration`` extra) speaks
a `Script` once and caches word times and compressed audio. Layout (this
package, pure Python) reads that cache and places each line as a beat of the
clip: a beat lasts as long as its speech plus padding, and each cue mark in
the text (``{indicate:speckle}speckle``) fires its move when the word is
said. A clip hands the `Narration` to ``eyepiece.Timeline(narration=...)``,
which places every move on its word, solves the length of each line in one
pass, and returns a `Narration` with those starts and ends as
``Timeline.narration``; `audio_track`, `mux`, `captions`, `write_srt` and
`write_vtt` then put the voice and the captions with the silent render.

Importing this package loads neither matplotlib nor Kokoro.
"""

from ._mux import (
    audio_track,
    caption_at,
    captions,
    mux,
    sentences,
    write_captions,
    write_srt,
    write_vtt,
)
from ._script import (
    KOKORO_MODEL,
    KOKORO_VERSION,
    Cue,
    Line,
    Script,
    line_hash,
    parse_cues,
)
from ._speech import (
    SpeechFault,
    SpeechTextError,
    check_script_speech,
    say,
    speech_faults,
)
from ._timeline import (
    CueTime,
    MissingAudioError,
    MissingSynthesisError,
    Narration,
    NarrationBeat,
    Spoken,
    Word,
    estimate,
)

__all__ = [
    "KOKORO_MODEL",
    "KOKORO_VERSION",
    "Cue",
    "CueTime",
    "Line",
    "MissingAudioError",
    "MissingSynthesisError",
    "Narration",
    "NarrationBeat",
    "Script",
    "SpeechFault",
    "SpeechTextError",
    "Spoken",
    "Word",
    "audio_track",
    "caption_at",
    "captions",
    "check_script_speech",
    "estimate",
    "line_hash",
    "mux",
    "parse_cues",
    "say",
    "sentences",
    "speech_faults",
    "write_captions",
    "write_srt",
    "write_vtt",
]
