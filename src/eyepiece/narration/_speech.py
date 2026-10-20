"""Standard-library wording and notation checks for spoken narration."""

import itertools
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
# A bare "one" is a determiner or a pronoun more often than a number, and a
# false alarm costs more than a miss: it counts only beside a word that makes
# it a quantity, and always inside a run such as "twenty one".
_ONE_AFTER = frozenset("step number plus minus times divided equals".split())
_ONE_BEFORE = frozenset("percent times plus minus over point".split())


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
        if bare == "one" and not run:
            after = i and _bare(words[i - 1]) in _ONE_AFTER
            before = i + 1 < len(words) and (
                _bare(words[i + 1]) in _ONE_BEFORE
                or _bare(words[i + 1]) in _NUMBER_WORDS
            )
            is_number = bool(after or before)
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


_VALUE_ONES = {w: float(i) for i, w in enumerate(_ONES)}
_VALUE_TENS = {w: 10.0 * (i + 2) for i, w in enumerate(_TENS)}
_VALUE_SCALES = {
    "hundred": 1e2,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}


def _spoken_value(words, index):
    """The value of the number that starts at ``words[index]``.

    Reads a token of digits (``96``, ``3.5``, ``1,000``, ``40%``) or a run of
    number words (``twenty one``, ``one hundred and five``, ``three point
    five``).

    Raises:
        ValueError: If the word does not start a number.
    """
    word = words[index]
    if _DIGITS.match(word):
        return float(word.strip(".,;:?!\"'()[]%").replace(",", ""))
    total = current = 0.0
    seen = False
    i = index
    while i < len(words):
        bare = _bare(words[i])
        if bare in _VALUE_ONES or bare in _VALUE_TENS:
            current += _VALUE_ONES.get(bare, _VALUE_TENS.get(bare))
        elif bare == "hundred":
            current = max(current, 1.0) * 100.0
        elif bare in _VALUE_SCALES:
            total += max(current, 1.0) * _VALUE_SCALES[bare]
            current = 0.0
        elif (
            seen
            and bare == "and"
            and i + 1 < len(words)
            and _bare(words[i + 1]) in _NUMBER_WORDS
        ):
            i += 1
            continue
        elif seen and bare == "point":
            digits = ""
            for later in words[i + 1 :]:
                got = _VALUE_ONES.get(_bare(later))
                if got is None or got > 9:
                    break
                digits += str(int(got))
            return total + current + (float("0." + digits) if digits else 0.0)
        else:
            break
        seen = True
        i += 1
    if not seen:
        raise ValueError(f"{word!r} is not a number")
    return total + current


@dataclass(frozen=True)
class NumberSyncFault:
    """One finding of `check_number_sync`.

    Attributes:
        code: ``"late-settle"`` or ``"contradicts"``.
        message: What was found.
        key: Key of the line.
        word: Index of the number word in the line.
        time_s: Time on the clip's clock the finding is at, or None.
    """

    code: str
    message: str
    key: str
    word: int
    time_s: float | None = None


def check_number_sync(
    narration, bindings, *, plan=None, settle_tol_s=0.0, value_tol=1e-9
) -> list[NumberSyncFault]:
    """Check that a counter or result lands on its spoken number.

    Each binding ties a number word of the narration to the readout that
    shows that number. A number lands on or before its word, so the readout
    must have settled on the spoken value by the time the word starts. The
    readout must also never show a value on the way that contradicts the
    spoken number: a final value that is not the spoken one, a pass-through
    that overshoots it, or a count that backs up.

    Args:
        narration: The `Narration`, with its measured per-word times.
        bindings: Dicts with ``word_key``, the number word as
            ``(line_key, word_index)`` or ``"line_key:word_index"``;
            ``span``, the window in which the readout counts, as ``(start_s,
            end_s)`` or the name of a plan span (with `plan`); and ``values``,
            the readout's shown value as ``(time_s, value)`` pairs, one per
            frame. Only samples inside the span count; the span may run past
            the line's beat, so a counter that settles late is measured.
        plan: A `Plan`, to resolve a span given by name.
        settle_tol_s: Seconds after the word's start by which the readout may
            settle. The default is zero: a number lands on or before its word.
            With a `plan` it is floored at one frame, so a readout drawn on
            the word's frame does not flag.
        value_tol: Largest difference, in the readout's units, between two
            values that count as equal.

    Returns:
        The findings, in binding order; empty when every readout lands.

    Raises:
        KeyError: If a binding names a line the narration lacks.
        ValueError: If a binding has no values, its word is not a number, or
            a named span needs a plan.
    """
    faults = []
    if plan is not None:
        settle_tol_s = max(settle_tol_s, 1.0 / plan.fps)
    for binding in bindings:
        key, index = _word_key(binding["word_key"])
        beat = narration.beat(key)
        words = narration.script.line(key).words
        spoken = _spoken_value(words, index)
        start = beat.words[index].start_s
        lo, hi = _span_of(binding["span"], plan)
        values = sorted(
            (float(t), float(v))
            for t, v in binding["values"]
            if lo - 1e-9 <= t <= hi + 1e-9
        )
        if not values:
            raise ValueError(
                f"binding for {key!r} word {index} has no values inside the span"
            )
        label = f"line {key!r} says {words[index]!r} ({spoken:g})"

        def off(a, b):
            return abs(a - b) > value_tol

        if off(values[-1][1], spoken):
            faults.append(
                NumberSyncFault(
                    "contradicts",
                    f"{label} but the readout ends on {values[-1][1]:g}",
                    key,
                    index,
                    values[-1][0],
                )
            )
        else:
            wrong = [t for t, v in values if off(v, spoken)]
            settled = values[0][0] if not wrong else _next_time(values, wrong[-1])
            if settled - start > settle_tol_s + 1e-9:
                faults.append(
                    NumberSyncFault(
                        "late-settle",
                        f"{label} at {start:.2f} s but the readout settles at "
                        f"{settled:.2f} s, {settled - start:.2f} s late",
                        key,
                        index,
                        settled,
                    )
                )
        faults += _path_faults(values, spoken, value_tol, label, key, index)
    return faults


def _next_time(values, t):
    return next(u for u, _ in values if u > t)


def _path_faults(values, spoken, tol, label, key, index):
    """A shown value that is past the spoken number or backs away from it."""
    gap = spoken - values[0][1]
    if gap == 0:
        for t, b in values[1:]:
            if abs(b - spoken) > tol:
                return [
                    NumberSyncFault(
                        "contradicts",
                        f"{label} but the readout leaves it for {b:g} at {t:.2f} s",
                        key,
                        index,
                        t,
                    )
                ]
        return []
    direction = math.copysign(1.0, gap)
    for (_, a), (t, b) in itertools.pairwise(values):
        if (b - spoken) * direction > tol:
            return [
                NumberSyncFault(
                    "contradicts",
                    f"{label} but the readout overshoots to {b:g} at {t:.2f} s",
                    key,
                    index,
                    t,
                )
            ]
        if (b - a) * direction < -tol:
            return [
                NumberSyncFault(
                    "contradicts",
                    f"{label} but the readout backs up from {a:g} to {b:g} "
                    f"at {t:.2f} s",
                    key,
                    index,
                    t,
                )
            ]
    return []


def _word_key(word_key):
    if isinstance(word_key, str):
        key, _, index = word_key.rpartition(":")
        return key, int(index)
    key, index = word_key
    return key, int(index)


def _span_of(span, plan):
    if isinstance(span, str):
        if plan is None:
            raise ValueError(f"the span {span!r} is a name: pass the plan")
        resolved = plan._span(span)
        return float(resolved.start_s), float(resolved.end_s)
    return float(span[0]), float(span[1])


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
