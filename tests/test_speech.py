"""Number wording and checks for notation in spoken text."""

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from eyepiece import narration as nr


@pytest.mark.parametrize(
    "value,spec,expected",
    [
        (0.943, ".3f", "0.943"),
        (-0.126, ".1%", "minus 12.6 percent"),
        (3e-8, ".0e", "3 times ten to the minus 8"),
        (1e-15, "g", "1 times ten to the minus 15"),
        (1.5e-10, "g", "1.5 times ten to the minus 10"),
        (1e-10, ".0e", "1 times ten to the minus 10"),
        (3e-8, ".2E", "3.00 times ten to the minus 8"),
        (1e8, ".0e", "1 times ten to the 8"),
        (-1.5e-10, "g", "minus 1.5 times ten to the minus 10"),
        (0.041, ".1%", "4.1 percent"),
        (-0.0, ".1f", "minus 0.0"),
        (1000, ",d", "1,000"),
        (Decimal("0.943"), ".3f", "0.943"),
    ],
)
def test_say_preserves_formatted_digits_and_expands_notation(value, spec, expected):
    text = nr.say(value, spec)
    assert text == expected
    assert nr.speech_faults(text) == []
    assert not any(symbol in text for symbol in ("^", "~", "*", "/", "%", "+", "-"))


@pytest.mark.parametrize(
    "value,spec,expected",
    [
        (4.1, ".1f", "plus 4.1"),
        (-4.1, ".1f", "minus 4.1"),
        (0, "g", "plus 0"),
        (0.041, ".1%", "plus 4.1 percent"),
        (3e-8, ".0e", "plus 3 times ten to the minus 8"),
    ],
)
def test_say_signed_numbers(value, spec, expected):
    assert nr.say(value, spec, signed=True) == expected


@pytest.mark.parametrize(
    "unit,words",
    [
        ("um", "microns"),
        ("nm", "nanometers"),
        ("pc", "parsecs"),
        ("AU", "AU"),
        ("arcsec", "arcseconds"),
        ("mas", "milliarcseconds"),
        ("mag/arcsec^2", "magnitudes per square arcsecond"),
        ("lambda/D", "lambda over D"),
    ],
)
def test_say_expands_units(unit, words):
    text = nr.say(0.55, ".2f", unit=unit)
    assert text == f"0.55 {words}"
    assert nr.speech_faults(text) == []


def test_say_unknown_unit_names_the_unit_and_choices():
    with pytest.raises(ValueError) as err:
        nr.say(1, unit="furlong")
    for unit in (
        "furlong",
        "um",
        "nm",
        "pc",
        "AU",
        "arcsec",
        "mas",
        "mag/arcsec^2",
        "lambda/D",
    ):
        assert unit in str(err.value)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_say_refuses_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite"):
        nr.say(value)


@pytest.mark.parametrize(
    "text,token,kind",
    [
        ("1e-15", "1e-15", "exponent"),
        ("3e-08", "3e-08", "exponent"),
        ("1.5e-10", "1.5e-10", "exponent"),
        ("1E+08", "1E+08", "exponent"),
        ("1.e+08", "1.e+08", "exponent"),
        ("10^-10", "10^-10", "power"),
        ("d^6.1", "d^6.1", "power"),
        ("arcsec^2", "arcsec^2", "power"),
        ("x^(2)", "x^(2)", "power"),
        ("x^(-2)", "x^(-2)", "power"),
        ("+/-1%", "+/-", "plus_minus"),
        ("~1%", "~", "approximation"),
        ("I*", "*", "asterisk"),
        ("a / b", "/", "slash"),
        ("1/10", "/", "slash"),
        ("lambda/D", "/", "slash"),
        ("a - 1", "-", "minus"),
        ("2% - 1%", "-", "minus"),
        ("2 -4", "-", "minus"),
        ("2%-1%", "-", "minus"),
        ("2-.4", "-", "minus"),
        ("(a)-(b)", "-", "minus"),
        ("(a) - (b)", "-", "minus"),
        ("2-4", "-", "minus"),
        ("0.55 um", "um", "unit"),
        ("1.6 \u03bcm", "\u03bcm", "unit"),
        ("1.6 \u00b5m", "\u00b5m", "unit"),
    ],
)
def test_speech_faults_identifies_notation_and_offsets(text, token, kind):
    text = "Read " + text + "."
    (fault,) = nr.speech_faults(text)
    assert isinstance(fault, nr.SpeechFault)
    assert (fault.token, fault.kind, fault.start) == (token, kind, text.index(token))
    assert fault.detail
    assert text[fault.start : fault.start + len(fault.token)] == token
    with pytest.raises((FrozenInstanceError, AttributeError)):
        fault.token = "changed"


@pytest.mark.parametrize(
    "text",
    [
        "0.943",
        ".943",
        "-12.6%",
        "+4.1%",
        "\u00b11%",
        "lambda over D",
        "3 times ten to the minus 8",
        "22 magnitudes per square arcsecond",
        "5 sigma",
        "S/N = 7",
        "2026",
        "1,000",
        "3rd",
        "6 m",
        "10 pc",
        "2.4 AU",
        "0.24 arcsec",
        "It changes by -12.6%.",
        "well-known",
        "A 10-year baseline.",
        "The 3-D view.",
        "a \u2014 b",
        "a \u2013 b",
        "The spectrum is blue.",
    ],
)
def test_speech_faults_accepts_readable_forms(text):
    assert nr.speech_faults(text) == []


def test_speech_faults_reports_every_occurrence_in_reading_order():
    text = "1e-15 and a / b - 1; 22 mag/arcsec^2; 0.55 um; ~1%; I*; +/-1%."
    faults = nr.speech_faults(text)
    assert [fault.token for fault in faults] == [
        "1e-15",
        "/",
        "-",
        "/",
        "arcsec^2",
        "um",
        "~",
        "*",
        "+/-",
    ]
    assert [fault.start for fault in faults] == sorted(fault.start for fault in faults)


def test_check_script_speech_checks_spoken_text_and_names_the_line():
    script = nr.Script(
        (
            nr.Line("clean", "A clean line."),
            nr.Line("contrast", "It is {indicate:value}1e-15."),
        )
    )
    assert issubclass(nr.SpeechTextError, ValueError)
    with pytest.raises(nr.SpeechTextError, match=r"contrast.*1e-15"):
        nr.check_script_speech(script)


def test_check_script_speech_ignores_display_text_and_cue_targets():
    script = nr.Script(
        (nr.Line("contrast", "It is {indicate:1e-15}one.", display="1e-15"),)
    )
    assert nr.check_script_speech(script) is None


def test_speech_helpers_are_public():
    for name in (
        "say",
        "speech_faults",
        "SpeechFault",
        "SpeechTextError",
        "check_script_speech",
    ):
        assert name in nr.__all__
        assert getattr(nr, name) is not None
