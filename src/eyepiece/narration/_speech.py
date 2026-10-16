"""Standard-library wording and notation checks for spoken narration."""

import math
import re
from dataclasses import dataclass

from ._script import Script

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


def check_script_speech(script: Script) -> None:
    """Check all spoken lines without loading a speech engine.

    Callers can use this before synthesis or timing estimation. Display
    text and cue targets are never checked.

    Args:
        script: The narration script to check.

    Raises:
        SpeechTextError: If any spoken line contains a speech fault. The
            message names each line, offending token and suggested wording.
    """
    messages = [
        f"line {line.key!r}: {fault.token!r} at {fault.start}: {fault.detail}"
        for line in script.lines
        for fault in speech_faults(line.spoken)
    ]
    if messages:
        raise SpeechTextError(
            "speech text needs wording changes; " + "; ".join(messages)
        )
