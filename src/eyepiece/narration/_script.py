"""A narration script: lines of speech whose words carry cue marks.

A line's text holds inline marks of the form ``{kind:target}``. A mark sits
in front of the word it names and fires at that word's start::

    What leaks through is {indicate:speckle}speckle.

The kind is the move to run (``show``, ``indicate``, ``circumscribe``,
``move``, or any name the clip understands) and the
target names what it acts on. The marks are stripped before the text is
spoken. The cache key hashes only the spoken text, so moving a mark never
costs a new synthesis.

This module imports only the standard library.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

_MARK = re.compile(r"\{(?P<kind>[a-z_][a-z0-9_]*):(?P<target>[^{}\s:]+)\}")
_STRAY = re.compile(r"[{}]")
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")
_ABBREVIATIONS = frozenset(
    "dr mr mrs ms prof st vs etc fig figs eq eqs approx cf al".split()
)
_INITIALS = re.compile(r"^(?:[A-Za-z]\.)+$")

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


def split_sentences(words):
    """Split spoken words into sentences, as ``(first, stop)`` word indices.

    A sentence ends at a word that ends in ``.``, ``?`` or ``!`` (with any
    closing quote or bracket after it). A period inside a number (``3.5``)
    never ends one, and neither does an abbreviation such as ``Dr.``, ``e.g.``
    or an initial.

    Args:
        words: The spoken words, with their punctuation.

    Returns:
        A tuple of ``(first, stop)`` pairs that tile ``range(len(words))``.
    """
    spans, first = [], 0
    for i, word in enumerate(words):
        if not _SENTENCE_END.search(word):
            continue
        bare = word.rstrip(".?!\"')]").lower()
        if word.rstrip("\"')]").endswith("."):
            if bare in _ABBREVIATIONS or _INITIALS.match(word.rstrip("\"')]")):
                if i + 1 < len(words):
                    continue
        spans.append((first, i + 1))
        first = i + 1
    if first < len(words):
        spans.append((first, len(words)))
    return tuple(spans)


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
        speed: Speech speed factor of this line, or None for the script's
            speed. It enters the line's cache key as the script speed does.
        sentence_gap_s: None speaks the line in one call. A number speaks it
            one sentence at a time (see `split_sentences`) and joins the
            sentences with that many seconds of silence, since the engine
            does not pause reliably at punctuation inside one call. Cue marks
            stay bound to their words, the timing reports each word on the
            joined clock, and each sentence is cached under its own key. It
            enters the line's cache key when set.
        spoken: The text with marks removed (set from `text`).
        cues: The marks, as `Cue` (set from `text`).
    """

    key: str
    text: str
    pad_before_s: float = 0.0
    pad_after_s: float = 0.5
    display: str | None = None
    speed: float | None = None
    sentence_gap_s: float | None = None
    spoken: str = field(init=False, compare=False)
    cues: tuple = field(init=False, compare=False)

    def __post_init__(self):
        """Parse the marks once, and check the padding."""
        if self.pad_before_s < 0.0 or self.pad_after_s < 0.0:
            raise ValueError(f"line {self.key!r}: padding must not be negative")
        if self.speed is not None and not self.speed > 0.0:
            raise ValueError(f"line {self.key!r}: speed must be positive")
        if self.sentence_gap_s is not None and not self.sentence_gap_s >= 0.0:
            raise ValueError(f"line {self.key!r}: sentence_gap_s must not be negative")
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

    @property
    def sentences(self):
        """The spoken sentences, each the words of one sentence joined."""
        words = self.words
        return tuple(" ".join(words[a:b]) for a, b in split_sentences(words))

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
class Pause:
    """A silent beat of the clip: it never speaks and has no audio.

    A pause sits in a `Script` among the lines. It enters the timing like a
    line with no audio, so the clip's moves run inside it; it adds nothing to
    the captions, and the audit lists it as a "silent" row naming what is on
    screen. Use it for the stillness the design declares, never to patch a gap.

    Attributes:
        key: Name of the beat; unique among the keys of the script.
        duration_s: Least length of the beat in seconds; motion that must
            finish inside it can hold it longer.
        description: One line saying what is on screen during the silence.
    """

    key: str
    duration_s: float
    description: str

    def __post_init__(self):
        """Check the duration and the description."""
        if not self.duration_s > 0.0 or self.duration_s == float("inf"):
            raise ValueError(
                f"pause {self.key!r}: duration must be finite and positive, "
                f"got {self.duration_s}"
            )
        if not self.description.strip():
            raise ValueError(f"pause {self.key!r}: description is empty")

    # The parts of a `Line` that the layout and the checks read.
    spoken = ""
    words = ()
    cues = ()
    display = None
    speed = None
    pad_before_s = 0.0
    pad_after_s = 0.0

    def cue(self, kind, target):
        """A pause has no cues.

        Raises:
            KeyError: Always.
        """
        raise KeyError(f"pause {self.key!r} has no cues")


@dataclass(frozen=True)
class Script:
    """The lines of one clip and the voice that speaks them.

    The voice, speed, language, engine, engine version and model all enter
    the cache key of every line, so a change to any of them speaks the
    lines again instead of reusing stale audio.

    Attributes:
        lines: The `Line` and `Pause` entries in order; their keys are unique.
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
        """Cache key of one line: its spoken text and everything that voices it.

        Raises:
            ValueError: If ``line`` is a `Pause`, which has no speech.
        """
        if isinstance(line, Pause):
            raise ValueError(f"pause {line.key!r} has no speech and no cache key")
        return line_hash(
            line.spoken,
            self.voice,
            self.speed_of(line),
            self.lang,
            engine=self.engine,
            engine_version=self.engine_version,
            model=self.model,
            gap=line.sentence_gap_s,
        )

    def sentence_keys(self, line):
        """Cache keys of the sentences of a line that is spoken sentence by sentence.

        Returns:
            One key per sentence, each that of the sentence spoken alone; an
            empty tuple for a line with no `sentence_gap_s`.
        """
        if isinstance(line, Pause) or line.sentence_gap_s is None:
            return ()
        return tuple(
            line_hash(
                text,
                self.voice,
                self.speed_of(line),
                self.lang,
                engine=self.engine,
                engine_version=self.engine_version,
                model=self.model,
            )
            for text in line.sentences
        )

    def speed_of(self, line):
        """The speed ``line`` is spoken at: its own, else the script's."""
        return self.speed if line.speed is None else line.speed

    def cache_keys(self):
        """The cache key of every spoken line, in order (pauses have none).

        A line spoken sentence by sentence is followed by its sentences' keys.
        """
        keys = []
        for line in self.lines:
            if not isinstance(line, Pause):
                keys.append(self.line_key(line))
                keys.extend(self.sentence_keys(line))
        return tuple(keys)

    def to_json(self, path):
        """Write this script to JSON without importing its defining module.

        Derived spoken text and cues are rebuilt from the marked text on
        load, preserving every line field and cache key.

        Args:
            path: Output JSON path, as a string or path-like object.

        Returns:
            The path written.
        """
        record = {f.name: getattr(self, f.name) for f in fields(self)}
        record["lines"] = [
            {
                f.name: getattr(line, f.name)
                for f in fields(line)
                if f.init
                and not (
                    f.name in ("speed", "sentence_gap_s")
                    and getattr(line, f.name) is None
                )
            }
            | ({"kind": "pause"} if isinstance(line, Pause) else {})
            for line in self.lines
        ]
        path = Path(path)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path):
        """Load a script written by `to_json`, including its voice settings.

        Args:
            path: Input JSON path, as a string or path-like object.

        Returns:
            The script with its derived spoken text and cues restored.
        """
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        record["lines"] = tuple(
            Pause(**{k: v for k, v in line.items() if k != "kind"})
            if line.get("kind") == "pause"
            else Line(**line)
            for line in record["lines"]
        )
        return cls(**record)


def line_hash(
    spoken,
    voice,
    speed,
    lang,
    *,
    engine="kokoro",
    engine_version=KOKORO_VERSION,
    model=KOKORO_MODEL,
    gap=None,
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
        gap: Seconds of silence between the sentences of a line spoken one
            sentence at a time, or None; it enters the key only when set.

    Returns:
        The first 16 hex digits of the SHA-256 of the fields.
    """
    fields_ = [spoken, voice, float(speed), lang, engine, engine_version, model]
    if gap is not None:
        fields_.append(["sentence_gap_s", float(gap)])
    payload = json.dumps(fields_)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
