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
`write_vtt` then put the voice and the captions with the silent render;
`write_chapters` writes one WebVTT chapter cue per beat.
`cue_track` synthesizes cue sounds (a count heard as pitch, one tick per
pixel read, a click sting) from numbers shown in the clip.

Importing this package loads neither matplotlib nor Kokoro.
"""

from ._cues import CueEvents, click_sting, count_tone, cue_track, pixel_ticks
from ._mux import (
    audio_track,
    caption_at,
    captions,
    mux,
    sentences,
    write_captions,
    write_chapters,
    write_srt,
    write_vtt,
)
from ._script import (
    KOKORO_MODEL,
    KOKORO_VERSION,
    Cue,
    Line,
    Pause,
    Script,
    line_hash,
    parse_cues,
)
from ._speech import (
    NumberSyncFault,
    ScriptFault,
    SilenceFault,
    SpeechFault,
    SpeechTextError,
    check_number_sync,
    check_script_speech,
    check_silence,
    lint_script,
    say,
    scientific,
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
    "CueEvents",
    "CueTime",
    "Line",
    "MissingAudioError",
    "MissingSynthesisError",
    "Narration",
    "NarrationBeat",
    "NumberSyncFault",
    "Pause",
    "Script",
    "ScriptFault",
    "SilenceFault",
    "SpeechFault",
    "SpeechTextError",
    "Spoken",
    "Word",
    "audio_track",
    "caption_at",
    "captions",
    "check_number_sync",
    "check_script_speech",
    "check_silence",
    "click_sting",
    "count_tone",
    "cue_track",
    "estimate",
    "line_hash",
    "lint_script",
    "mux",
    "parse_cues",
    "pixel_ticks",
    "say",
    "scientific",
    "sentences",
    "speech_faults",
    "write_captions",
    "write_chapters",
    "write_srt",
    "write_vtt",
]
