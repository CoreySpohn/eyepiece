"""Lay a narration script out in time from its synthesized word times.

The layout is pure Python and reads only the JSON that `synthesize` wrote
beside the clip, so a render never needs Kokoro. Each line is a beat. The
beat lasts as long as its speech plus the line's padding, or as long as the
motion declared for it, whichever is longer, and it replaces the fixed holds
a silent clip uses. A cue is the start of the word it precedes; the clip
decides how far ahead of the word each kind of move starts.

A `Narration` is placement only. `Narration.place` lays one line on the clock
and is pure; `eyepiece.Timeline(narration=...)` places the moves of a clip on
the words, solves the length of every line in one pass, and hands back a
`Narration` with those starts and ends for `audio_track`, `mux` and the
captions. Narration measures the pace of the speech, so it takes the place of
a `Tempo` for the lines it covers; the speech speed is `Script.speed`.

This module imports only the standard library.
"""

import json
from dataclasses import dataclass
from pathlib import Path

_LETTERS_PER_S = 14.0
_PAUSE_S = 0.25


class MissingSynthesisError(FileNotFoundError):
    """A line has no cached timing; the message says how to synthesize it."""


class MissingAudioError(FileNotFoundError):
    """A line has timing but no audio file beside it."""


@dataclass(frozen=True)
class Word:
    """A spoken word and when it is said.

    Attributes:
        text: The word with its punctuation.
        start_s: Start, in seconds from the start of the clip.
        end_s: End, in seconds from the start of the clip.
    """

    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Spoken:
    """The synthesis of one line: its words and length, from its start.

    Attributes:
        key: The line's cache key.
        duration_s: Length of the audio in seconds.
        words: One `Word` per spoken word, in seconds from the line's audio start.
        text: The spoken text the timing belongs to, or None when unknown.
        audio: Name of the audio file beside the timing, or None.
        rate: Sample rate of the audio in Hz, or None.
        samples: Number of samples of the audio before compression, or None.
    """

    key: str
    duration_s: float
    words: tuple
    text: str | None = None
    audio: str | None = None
    rate: int | None = None
    samples: int | None = None

    def to_json(self):
        """The record `synthesize` writes beside the clip."""
        record = {
            "key": self.key,
            "duration_s": self.duration_s,
            "words": [[w.text, w.start_s, w.end_s] for w in self.words],
        }
        for name in ("text", "audio", "rate", "samples"):
            if getattr(self, name) is not None:
                record[name] = getattr(self, name)
        return record

    @classmethod
    def from_json(cls, record):
        """Read a record written by `to_json`."""
        words = tuple(Word(t, float(a), float(b)) for t, a, b in record["words"])
        return cls(
            record["key"],
            float(record["duration_s"]),
            words,
            record.get("text"),
            record.get("audio"),
            record.get("rate"),
            record.get("samples"),
        )


def estimate(line):
    """A rough `Spoken` for a line that has not been synthesized.

    Words last in proportion to their letters, with a short pause after
    sentence and clause punctuation. It stands in for the cache in tests and
    in a first layout, never in a published render.
    """
    t = 0.0
    words = []
    for text in line.words:
        letters = sum(ch.isalnum() for ch in text)
        end = t + max(1, letters) / _LETTERS_PER_S
        words.append(Word(text, t, end))
        t = end + (_PAUSE_S if text[-1] in ".?!,;" else 0.0)
    return Spoken("estimate", t, tuple(words))


@dataclass(frozen=True)
class CueTime:
    """A cue placed on the clip's clock.

    Attributes:
        line: Key of the line the cue belongs to.
        kind: The attention move.
        target: What it acts on.
        at_s: When its word starts, in seconds from the start of the clip.
        word: The word it is on.
    """

    line: str
    kind: str
    target: str
    at_s: float
    word: Word


@dataclass(frozen=True)
class NarrationBeat:
    """One line placed on the clip's clock.

    Attributes:
        key: The line's key.
        start_s: Start of the beat.
        speech_start_s: When the audio starts.
        speech_end_s: When the audio ends.
        end_s: End of the beat, where the next one starts.
        words: The `Word` of the line, on the clip's clock.
        cues: The `CueTime` of the line, on the clip's clock.
    """

    key: str
    start_s: float
    speech_start_s: float
    speech_end_s: float
    end_s: float
    words: tuple
    cues: tuple


class Narration:
    """A script laid out on one clock.

    Lines run one after another: a line starts where the one before it ends,
    unless `starts` moves it later, and ends when its speech and padding do,
    unless `ends` holds it longer. A `Timeline` with a narration passes both,
    from the solved lengths of its lines.

    Args:
        script: The `Script`.
        spoken: Mapping from line key to the line's `Spoken`.
        starts: Mapping from line key to an absolute start in seconds, at or
            after the end of the line before it. The first line starts at 0
            by default.
        ends: Mapping from line key to the absolute time its beat must last
            until: the motion that must finish inside the line.

    Raises:
        KeyError: If `starts`, `ends` or `spoken` names a line the script
            lacks or omits one it has.
        ValueError: If a line's timing does not have its words, or a line
            starts before the one before it ends.
    """

    def __init__(self, script, spoken, *, starts=None, ends=None):
        """Place every line, in order, one after another."""
        starts = dict(starts or {})
        ends = dict(ends or {})
        keys = {line.key for line in script.lines}
        for name, given in (("starts", starts), ("ends", ends)):
            unknown = set(given) - keys
            if unknown:
                raise KeyError(f"{name} names unknown lines {sorted(unknown)}")
        self.script = script
        self.spoken = {}
        self.beats = []
        self._asked = set()
        for line in script.lines:
            if line.key not in spoken:
                raise KeyError(f"no timing for line {line.key!r}")
            timing = spoken[line.key]
            self.spoken[line.key] = timing
            if len(timing.words) != len(line.words):
                raise ValueError(
                    f"line {line.key!r}: {len(line.words)} words but the "
                    f"timing has {len(timing.words)}; synthesize it again"
                )
        clock = 0.0
        for line in script.lines:
            start = starts.get(line.key, clock)
            if start < clock - 1e-9:
                raise ValueError(
                    f"line {line.key!r} starts at {start} s, before the line "
                    f"before it ends at {clock} s"
                )
            beat = self.place(line.key, start, end_s=ends.get(line.key))
            self.beats.append(beat)
            clock = beat.end_s
        self._by_key = {b.key: b for b in self.beats}

    def place(self, key, start_s, *, end_s=None):
        """Lay one line on the clock, starting at `start_s`.

        Pure: it depends only on the line, its timing and the arguments. The
        speech starts after the line's ``pad_before_s``; each cue is the start
        of its word, with no lead.

        Args:
            key: The line's key.
            start_s: Absolute start of the beat, in seconds.
            end_s: Absolute time the beat must last until, or None. The beat
                ends at the later of this and its speech plus ``pad_after_s``.

        Returns:
            The `NarrationBeat`.
        """
        line = self.script.line(key)
        timing = self.spoken[key]
        start_s = float(start_s)
        on = start_s + line.pad_before_s
        words = tuple(Word(w.text, on + w.start_s, on + w.end_s) for w in timing.words)
        speech_end = on + timing.duration_s
        end = speech_end + line.pad_after_s
        if end_s is not None:
            end = max(end, float(end_s))
        cues = tuple(
            CueTime(line.key, c.kind, c.target, words[c.word].start_s, words[c.word])
            for c in line.cues
        )
        return NarrationBeat(line.key, start_s, on, speech_end, end, words, cues)

    @classmethod
    def load(cls, script, cache_dir, **kwargs):
        """Read the cached timing of every line of ``script``.

        Args:
            script: The `Script`.
            cache_dir: Directory `synthesize` wrote its JSON into.
            **kwargs: Passed to `Narration`.

        Raises:
            MissingSynthesisError: If a line has no cached timing.
            ValueError: If a file's key or text is not the line's.
        """
        spoken = {}
        for line in script.lines:
            key = script.line_key(line)
            path = Path(cache_dir) / f"{key}.json"
            if not path.exists():
                raise MissingSynthesisError(
                    f"no cached speech for line {line.key!r} ({path.name}); run "
                    "python -m eyepiece.narration.synth <script.py> <cache_dir> "
                    "in an environment with the narration extra"
                )
            timing = Spoken.from_json(json.loads(path.read_text()))
            if timing.key != key:
                raise ValueError(f"{path.name} holds the timing of {timing.key}")
            if timing.text is not None and timing.text != line.spoken:
                raise ValueError(
                    f"{path.name} was spoken from {timing.text!r}, not the "
                    f"text of line {line.key!r}"
                )
            spoken[line.key] = timing
        return cls(script, spoken, **kwargs)

    @classmethod
    def estimated(cls, script, **kwargs):
        """A narration from `estimate`, for tests and first layouts."""
        return cls(
            script, {line.key: estimate(line) for line in script.lines}, **kwargs
        )

    def beat(self, key):
        """The `NarrationBeat` named ``key``."""
        try:
            return self._by_key[key]
        except KeyError:
            raise KeyError(f"no line named {key!r}") from None

    def start(self, key):
        """Start of the beat named ``key``, in seconds."""
        return self.beat(key).start_s

    def end(self, key):
        """End of the beat named ``key``, in seconds."""
        return self.beat(key).end_s

    def cue(self, key, kind, target):
        """When the cue ``kind:target`` of line ``key`` fires, in seconds."""
        self.script.line(key).cue(kind, target)
        self._asked.add((key, kind, target))
        for c in self.beat(key).cues:
            if (c.kind, c.target) == (kind, target):
                return c.at_s
        raise KeyError((key, kind, target))

    def cues(self):
        """Every `CueTime` of the clip, in the order they fire."""
        every = [c for b in self.beats for c in b.cues]
        return sorted(every, key=lambda c: c.at_s)

    def check(self, *, kinds=None, targets=None, used=False):
        """Check the cues against what the clip handles.

        A mark the clip does not understand parses, lays out, and is then
        never fired; this turns that silence into an error.

        Args:
            kinds: The cue kinds the clip handles; None accepts any.
            targets: The cue targets the clip handles; None accepts any.
            used: Also require that `cue` has been asked for every cue, so a
                mark the clip never reads is reported. Call it after the
                clip has placed its moves.

        Raises:
            ValueError: Naming each offending cue with its line.
        """
        problems = []
        for c in self.cues():
            where = f"line {c.line!r}: {c.kind}:{c.target}"
            if kinds is not None and c.kind not in kinds:
                problems.append(
                    f"{where} has an unknown kind; use one of {sorted(kinds)}"
                )
            if targets is not None and c.target not in targets:
                problems.append(
                    f"{where} has an unknown target; use one of {sorted(targets)}"
                )
            if used and (c.line, c.kind, c.target) not in self._asked:
                problems.append(f"{where} is never used by the clip")
        if problems:
            raise ValueError("; ".join(problems))

    @property
    def total_s(self):
        """End of the last beat, which is the length of the clip."""
        return self.beats[-1].end_s if self.beats else 0.0

    def beat_at(self, t):
        """Key of the beat running at time ``t``.

        Before the start it is the first beat, and after the end the last.
        """
        key = self.beats[0].key
        for b in self.beats:
            if b.start_s <= t + 1e-9:
                key = b.key
        return key
