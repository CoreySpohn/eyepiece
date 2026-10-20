"""Standard-library wording and notation checks for spoken narration."""

import math
import re
from dataclasses import dataclass

from ._script import Pause, Script

_UNITS = {
    "um": "microns",
    "nm": "nanometers",
    "pc": "parsecs",
    "AU": "AU",
    "arcsec": "arcseconds",
    "mas": "milliarcseconds",
    "mag/arcsec^2": "magnitudes per square arcsecond",
    "lambda/D": "lambda over D",
}
_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_FAULT = re.compile(
    rf"(?P<exponent>(?<!\w){_NUMBER}\.?[eE][+-]?\d+\b)"
    rf"|(?P<power>\w+\^(?:[+-]?{_NUMBER}|\([+-]?{_NUMBER}\)))"
    r"|(?P<plus_minus>\+/-)"
    r"|(?P<approximation>~)"
    r"|(?P<asterisk>\*)"
    r"|(?P<slash>/)"
    r"|(?P<minus>-)"
    r"|(?<=\d)\s*(?P<unit>[u\u03bc\u00b5]m)\b"
)
_DETAILS = {
    "exponent": "write the exponent as 'times ten to the' and its signed power",
    "power": "write powers and squared units in words",
    "plus_minus": "write 'plus or minus'",
    "approximation": "write 'about' or 'approximately'",
    "asterisk": "write the intended meaning of the asterisk in words",
    "slash": "write 'over' or 'per' as appropriate",
    "minus": "write 'minus' or 'to' as appropriate",
    "unit": "write 'microns'",
}


@dataclass(frozen=True)
class SpeechFault:
    """One notation form the speech engine can misread.

    Attributes:
        token: The offending text.
        kind: The notation category.
        detail: Suggested wording for speech.
        start: Character index of the token in the checked text.
    """

    token: str
    kind: str
    detail: str
    start: int


class SpeechTextError(ValueError):
    """A script contains notation that the speech engine can misread."""


def say(value, spec="g", *, unit=None, signed=False) -> str:
    """Format a finite number with notation expanded for speech.

    Args:
        value: The number to format.
        spec: Python's numeric format specification. Mantissa digits retain
            the precision chosen by ``format(value, spec)``.
        unit: Unit name: um, nm, pc, AU, arcsec, mas, mag/arcsec^2 or lambda/D.
        signed: Prefix nonnegative numbers with 'plus'. Negative numbers
            always begin with 'minus', including negative zero.

    Returns:
        The formatted digits with signs, exponents, percentages and units
        written for speech.

    Raises:
        ValueError: If the value is not finite, the format specification is
            invalid, or the unit is unknown.
    """
    if not math.isfinite(value):
        raise ValueError("speech needs a finite number")
    if unit is not None and unit not in _UNITS:
        raise ValueError(f"unknown unit {unit!r}; known units: {', '.join(_UNITS)}")
    text = format(value, spec).strip()
    sign = "plus " if signed else ""
    if text.startswith(("-", "+")):
        sign = "minus " if text[0] == "-" else "plus "
        text = text[1:]
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    exponent = re.fullmatch(r"(.+)[eE]([+-]?\d+)", text)
    if exponent:
        power = int(exponent[2])
        direction = "minus " if power < 0 else ""
        text = f"{exponent[1]} times ten to the {direction}{abs(power)}"
    text = sign + text
    if percent:
        text += " percent"
    if unit is not None:
        text += " " + _UNITS[unit]
    return text


def speech_faults(text: str) -> list[SpeechFault]:
    """Find notation that the speech engine can misread, in reading order.

    Args:
        text: Spoken text, with cue marks already removed.

    Returns:
        Faults with token offsets into ``text``. Plain numbers, leading
        signs, word hyphens and the known abbreviation S/N are accepted.
    """
    good_slashes = {match.start() + 1 for match in re.finditer(r"\bS/N\b", text)}
    faults = []
    for match in _FAULT.finditer(text):
        kind = match.lastgroup
        start, end = match.span(kind)
        if kind == "slash" and start in good_slashes:
            continue
        if kind == "minus":
            before, after = text[:start].rstrip(), text[end:].lstrip()
            if not before or not after:
                continue
            left_space = text[start - 1].isspace()
            right_space = text[end].isspace()
            if left_space and not right_space:
                if not re.search(r"(?:\d%?|\))$", before):
                    continue
            elif not left_space and not right_space:
                if not (
                    re.search(r"(?:\d%?|\))$", before)
                    and re.match(r"(?:\d|\.\d|\()", after)
                ):
                    continue
        faults.append(SpeechFault(match[kind], kind, _DETAILS[kind], start))
    return faults


_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = "hundred thousand million billion trillion".split()
_NUMBER_WORDS = frozenset(_ONES + _TENS + _SCALES)
_DIGITS = re.compile(r"^\W*[+-]?\d[\d,]*(?:\.\d+)?%?\W*$")
# "one" is a pronoun more often than a number; it is a number unless its
# neighbors say otherwise.
_ONE_NOT_AFTER = frozenset("the no any each every this that which a an".split())
_ONE_NOT_BEFORE = frozenset(
    "of at another by can could is was that who must should more way thing".split()
)


@dataclass(frozen=True)
class ScriptFault:
    """One finding of `lint_script`.

    Attributes:
        code: ``"multi-sentence"``, ``"question-after-statement"``,
            ``"many-numbers"`` or ``"number-without-cue"``.
        message: What was found and how to fix it.
        key: Key of the line.
        word: Index of the word involved, or None for the whole line.
    """

    code: str
    message: str
    key: str
    word: int | None = None


def _bare(word):
    return word.strip(".,;:?!\"'()[]").lower()


def _numbers(words):
    """The numbers a line says, as the index of the first word of each.

    A run of number words (``twenty one``, ``one hundred and five``,
    ``three point five``) is one number, as is each token of digits.
    """
    found, run = [], False
    for i, word in enumerate(words):
        bare = _bare(word)
        if _DIGITS.match(word):
            found.append(i)
            run = False
            continue
        is_number = bare in _NUMBER_WORDS
        if bare == "one" and (
            (i and _bare(words[i - 1]) in _ONE_NOT_AFTER)
            or (i + 1 < len(words) and _bare(words[i + 1]) in _ONE_NOT_BEFORE)
        ):
            is_number = False
        joins = run and (
            bare in ("and", "point")
            and i + 1 < len(words)
            and _bare(words[i + 1]) in _NUMBER_WORDS
        )
        if is_number and not run:
            found.append(i)
        run = is_number or joins
    return found


def lint_script(script: Script) -> list[ScriptFault]:
    """Find wording that is hard to hear or to time, line by line.

    The rules are: more than one sentence in a line without a
    ``sentence_gap_s`` (the engine runs sentences together); a question that
    follows a statement in the same line; more than one number in a line (a
    number is a token of digits or a run of number words, and "one" counts
    unless it reads as a pronoun); and a number with no cue mark within two
    words of it, so nothing on screen lands with it.

    Args:
        script: The narration script to check.

    Returns:
        The findings in line order; empty when the script is clean. Display
        text is never checked.
    """
    faults = []
    for line in script.lines:
        if isinstance(line, Pause):
            continue
        sentences = line.sentences
        if len(sentences) > 1 and line.sentence_gap_s is None:
            faults.append(
                ScriptFault(
                    "multi-sentence",
                    f"line {line.key!r} holds {len(sentences)} sentences with no "
                    "sentence_gap_s; set it, or split the line",
                    line.key,
                )
            )
        for i, text in enumerate(sentences[1:], start=1):
            if text.endswith("?") and not all(
                prior.endswith("?") for prior in sentences[:i]
            ):
                faults.append(
                    ScriptFault(
                        "question-after-statement",
                        f"line {line.key!r}: {text!r} asks a question after a "
                        "statement; give the question its own line",
                        line.key,
                    )
                )
                break
        numbers = _numbers(line.words)
        if len(numbers) > 1:
            faults.append(
                ScriptFault(
                    "many-numbers",
                    f"line {line.key!r} introduces {len(numbers)} numbers; "
                    "introduce one number per line",
                    line.key,
                )
            )
        for index in numbers:
            if not any(abs(cue.word - index) <= 2 for cue in line.cues):
                faults.append(
                    ScriptFault(
                        "number-without-cue",
                        f"line {line.key!r}: {line.words[index]!r} (word {index}) "
                        "has no cue mark within two words",
                        line.key,
                        index,
                    )
                )
    return faults


def check_script_speech(script: Script, *, rules: bool = False) -> None:
    """Check all spoken lines without loading a speech engine.

    Callers can use this before synthesis or timing estimation. Display
    text and cue targets are never checked.

    Args:
        script: The narration script to check.
        rules: Also raise for the findings of `lint_script`. Off by default,
            so a script written before those rules still synthesizes.

    Raises:
        SpeechTextError: If any spoken line contains a speech fault, or with
            `rules` a lint finding. The message names each line, offending
            token and suggested wording.
    """
    messages = [
        f"line {line.key!r}: {fault.token!r} at {fault.start}: {fault.detail}"
        for line in script.lines
        for fault in speech_faults(line.spoken)
    ]
    if rules:
        messages += [f"{f.code}: {f.message}" for f in lint_script(script)]
    if messages:
        raise SpeechTextError(
            "speech text needs wording changes; " + "; ".join(messages)
        )


@dataclass(frozen=True)
class SilenceFault:
    """One finding of `check_silence`.

    Attributes:
        code: ``"late-first-cue"``, ``"early-last-cue"``, ``"undeclared-silence"``,
            ``"multiple-hero"`` or ``"speech-in-hero"``.
        message: What was found.
        key: Key of the line involved, or None.
        start_s: Start of the stretch involved, in seconds, or None.
        end_s: End of the stretch involved, in seconds, or None.
    """

    code: str
    message: str
    key: str | None = None
    start_s: float | None = None
    end_s: float | None = None


def _minus(span, cuts):
    """The parts of ``span`` that no interval of ``cuts`` covers."""
    parts = [span]
    for c0, c1 in cuts:
        rest = []
        for a, b in parts:
            if c1 <= a or c0 >= b:
                rest.append((a, b))
                continue
            if c0 > a:
                rest.append((a, c0))
            if c1 < b:
                rest.append((c1, b))
        parts = rest
    return parts


def check_silence(
    narration, plan=None, *, hero=(), max_cue_gap_s=3.0, max_silence_s=2.0
):
    """Check a laid-out narration for silence nobody declared.

    Findings are returned, never raised, like `speech_faults`.

    * ``late-first-cue`` / ``early-last-cue``: a spoken line whose first cue
      comes more than ``max_cue_gap_s`` after its speech starts, or whose last
      cue comes more than that before its speech ends, so the screen waits on
      the voice or the voice runs on over a screen that has stopped.
    * ``undeclared-silence``: a stretch longer than ``max_silence_s`` with no
      speech that is neither a `Pause` nor inside a hero span.
    * ``multiple-hero``: more than one hero span.
    * ``speech-in-hero``: a line spoken during a hero span.

    Args:
        narration: A `Narration` (timing placed on the clip's clock).
        plan: A `eyepiece.Plan` whose `hero` annotations declare hero spans.
        hero: More hero windows as ``(start_s, end_s)``.
        max_cue_gap_s: Longest uncued stretch at either end of a line.
        max_silence_s: Longest silent stretch that needs no declaration.

    Returns:
        A list of `SilenceFault`, in clip order.
    """
    windows = list(plan.hero_windows()) if plan is not None else []
    windows += [(float(a), float(b)) for a, b in hero]
    pauses = [
        (b.start_s, b.end_s)
        for b in narration.beats
        if isinstance(narration.script.line(b.key), Pause)
    ]
    speech = [
        b
        for b in narration.beats
        if not isinstance(narration.script.line(b.key), Pause)
    ]
    out = []
    for b in speech:
        if not b.cues:
            continue
        first, last = b.cues[0], b.cues[-1]
        if first.at_s - b.speech_start_s > max_cue_gap_s:
            out.append(
                SilenceFault(
                    "late-first-cue",
                    f"line {b.key!r}: its first cue is "
                    f"{first.at_s - b.speech_start_s:.1f} s after the line starts",
                    b.key,
                    b.speech_start_s,
                    first.at_s,
                )
            )
        if b.speech_end_s - last.at_s > max_cue_gap_s:
            out.append(
                SilenceFault(
                    "early-last-cue",
                    f"line {b.key!r}: its last cue is "
                    f"{b.speech_end_s - last.at_s:.1f} s before the line ends",
                    b.key,
                    last.at_s,
                    b.speech_end_s,
                )
            )
    declared = pauses + windows
    edges = [0.0] + [t for b in speech for t in (b.speech_start_s, b.speech_end_s)]
    edges.append(float(narration.total_s))
    for a, b in zip(edges[0::2], edges[1::2], strict=True):
        for c, d in _minus((a, b), declared):
            if d - c > max_silence_s:
                out.append(
                    SilenceFault(
                        "undeclared-silence",
                        f"{d - c:.1f} s of silence from {c:.1f} s with no declared "
                        "pause or hero span",
                        None,
                        c,
                        d,
                    )
                )
    if len(windows) > 1:
        out.append(
            SilenceFault(
                "multiple-hero",
                f"{len(windows)} hero spans declared; a clip has at most one",
            )
        )
    for b in speech:
        for h0, h1 in windows:
            if b.speech_start_s < h1 - 1e-9 and b.speech_end_s > h0 + 1e-9:
                out.append(
                    SilenceFault(
                        "speech-in-hero",
                        f"line {b.key!r} is spoken during the hero span "
                        f"{h0:.1f} to {h1:.1f} s",
                        b.key,
                        h0,
                        h1,
                    )
                )
    return sorted(out, key=lambda f: (f.start_s is None, f.start_s or 0.0))
