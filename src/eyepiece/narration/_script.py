"""A narration script: lines of speech whose words carry cue marks.

A line's text holds inline marks of the form ``{kind:target}``. A mark sits
in front of the word it names and fires at that word's start::

    What leaks through is {indicate:speckle}speckle.

The kind is the attention move to run (``indicate``, ``circumscribe``,
``link``, ``flight``, ``caption``, or any name the clip understands) and the
target names what it acts on. The marks are stripped before the text is
spoken. The cache key hashes only the spoken text, so moving a mark never
costs a new synthesis.

This module imports only the standard library.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field

_MARK = re.compile(r"\{(?P<kind>[a-z_][a-z0-9_]*):(?P<target>[^{}\s:]+)\}")
_STRAY = re.compile(r"[{}]")

KOKORO_MODEL = "Kokoro-82M"
KOKORO_VERSION = "0.9.4"


@dataclass(frozen=True)
class Cue:
    """One cue mark, resolved to the word it sits in front of.

    Attributes:
        kind: The attention move to run.
        target: What the move acts on.
        word: Index of the word the mark precedes, in the spoken text.
    """

    kind: str
    target: str
    word: int


def parse_cues(text):
    """Split marked text into the spoken text and its cues.

    Args:
        text: Narration text with ``{kind:target}`` marks.

    Returns:
        ``(spoken, cues)``: the text with marks removed and whitespace
        collapsed, and a tuple of `Cue` in reading order.

    Raises:
        ValueError: If a brace is not part of a well-formed mark, a mark does
            not follow a space, or a mark has no word after it.
    """
    cues = []
    pieces = []
    last = 0
    for match in _MARK.finditer(text):
        pieces.append(text[last : match.start()])
        before = "".join(pieces)
        if before and not before[-1].isspace():
            raise ValueError(f"cue mark must follow a space in {text!r}")
        cues.append(Cue(match["kind"], match["target"], len(before.split())))
        last = match.end()
    pieces.append(text[last:])
    rest = "".join(pieces)
    if _STRAY.search(rest):
        raise ValueError(f"malformed cue mark in {text!r}; use {{kind:target}}")
    spoken = " ".join(rest.split())
    count = len(spoken.split())
    for cue in cues:
        if cue.word >= count:
            raise ValueError(f"cue {cue.kind}:{cue.target} has no word after it")
    return spoken, tuple(cues)


@dataclass(frozen=True)
class Line:
    """One line of narration, spoken as a unit, with its padding.

    A line is one beat of the clip. The beat lasts as long as the speech plus
    the padding, or as long as the motion that must finish inside it,
    whichever is longer.

    Attributes:
        key: Name of the beat; unique within a script.
        text: The text with cue marks.
        pad_before_s: Silence before the speech starts, in seconds.
        pad_after_s: Silence after the speech ends, in seconds.
        display: Text the captions show instead of the spoken words, or None
            to show the spoken words. A line may say "P of H given E" while
            the caption reads "Pr(H|E)". It carries no cue marks and does
            not enter the cache key, and the cue marks stay bound to the
            spoken words.
        spoken: The text with marks removed (set from `text`).
        cues: The marks, as `Cue` (set from `text`).
    """

    key: str
    text: str
    pad_before_s: float = 0.0
    pad_after_s: float = 0.5
    display: str | None = None
    spoken: str = field(init=False, compare=False)
    cues: tuple = field(init=False, compare=False)

    def __post_init__(self):
        """Parse the marks once, and check the padding."""
        if self.pad_before_s < 0.0 or self.pad_after_s < 0.0:
            raise ValueError(f"line {self.key!r}: padding must not be negative")
        try:
            spoken, cues = parse_cues(self.text)
        except ValueError as err:
            raise ValueError(f"line {self.key!r}: {err}") from None
        if not spoken:
            raise ValueError(f"line {self.key!r} has no words")
        if self.display is not None:
            if not self.display.split():
                raise ValueError(f"line {self.key!r}: display text is empty")
            if _STRAY.search(self.display):
                raise ValueError(
                    f"line {self.key!r}: display text must not hold cue marks"
                )
        object.__setattr__(self, "spoken", spoken)
        object.__setattr__(self, "cues", cues)

    @property
    def words(self):
        """The spoken words, with their punctuation."""
        return tuple(self.spoken.split())

    def cue(self, kind, target):
        """The one cue of this line with this kind and target.

        Raises:
            KeyError: If there is none, or more than one.
        """
        found = [c for c in self.cues if (c.kind, c.target) == (kind, target)]
        if len(found) != 1:
            raise KeyError(f"line {self.key!r} has {len(found)} cues {kind}:{target}")
        return found[0]


@dataclass(frozen=True)
class Script:
    """The lines of one clip and the voice that speaks them.

    The voice, speed, language, engine, engine version and model all enter
    the cache key of every line, so a change to any of them speaks the
    lines again instead of reusing stale audio.

    Attributes:
        lines: The `Line` in order; their keys are unique.
        voice: Voice name of the engine. The default, ``bm_george``, is a
            British English Kokoro voice.
        speed: Speech speed factor (1.0 is the voice's natural rate).
        lang: Language code of the engine. Kokoro's ``"b"`` is British
            English, which the ``bm_`` voices need; ``"a"`` is American.
        engine: Name of the speech engine, a key of ``synth.ENGINES``.
        engine_version: Version of the engine package the audio came from;
            the synthesis refuses to run on another version.
        model: Name of the engine's model.
    """

    lines: tuple
    voice: str = "bm_george"
    speed: float = 1.0
    lang: str = "b"
    engine: str = "kokoro"
    engine_version: str = KOKORO_VERSION
    model: str = KOKORO_MODEL

    def __post_init__(self):
        """Check that the keys are unique and the speed is positive."""
        keys = [line.key for line in self.lines]
        if len(set(keys)) != len(keys):
            raise ValueError(f"line keys must be unique, got {keys}")
        if self.speed <= 0.0:
            raise ValueError(f"speed must be positive, got {self.speed}")

    def line(self, key):
        """The line named ``key``."""
        for line in self.lines:
            if line.key == key:
                return line
        raise KeyError(
            f"no line named {key!r}; lines are {[x.key for x in self.lines]}"
        )

    def line_key(self, line):
        """Cache key of one line: its spoken text and everything that voices it."""
        return line_hash(
            line.spoken,
            self.voice,
            self.speed,
            self.lang,
            engine=self.engine,
            engine_version=self.engine_version,
            model=self.model,
        )

    def cache_keys(self):
        """The cache key of every line, in order."""
        return tuple(self.line_key(line) for line in self.lines)


def line_hash(
    spoken,
    voice,
    speed,
    lang,
    *,
    engine="kokoro",
    engine_version=KOKORO_VERSION,
    model=KOKORO_MODEL,
):
    """Cache key of a spoken line, 16 hex digits.

    Args:
        spoken: The text as spoken (no cue marks).
        voice: Voice name.
        speed: Speed factor.
        lang: Language code.
        engine: Name of the speech engine.
        engine_version: Version of the engine package.
        model: Model name.

    Returns:
        The first 16 hex digits of the SHA-256 of the fields.
    """
    payload = json.dumps(
        [spoken, voice, float(speed), lang, engine, engine_version, model]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
