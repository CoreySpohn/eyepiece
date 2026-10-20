"""The script lint: wording that makes narration hard to hear or to time."""

import pytest

from eyepiece import narration as nr


def _codes(*lines, **kwargs):
    return [(f.code, f.key) for f in nr.lint_script(nr.Script(lines), **kwargs)]


def test_a_clean_line_has_no_findings():
    line = nr.Line("a", "The {indicate:star}star is faint.")
    assert nr.lint_script(nr.Script((line,))) == []


def test_several_sentences_in_one_call_are_flagged_unless_the_gap_is_set():
    text = "The star dims. The planet stays."
    assert _codes(nr.Line("a", text)) == [("multi-sentence", "a")]
    assert _codes(nr.Line("a", text, sentence_gap_s=0.4)) == []
    assert _codes(nr.Line("a", "Just a sentence here.")) == []


def test_a_question_after_a_statement_is_flagged_even_with_a_gap():
    line = nr.Line("a", "It is faint. Why is it faint?", sentence_gap_s=0.4)
    assert _codes(line) == [("question-after-statement", "a")]
    assert _codes(nr.Line("b", "Why is it faint? Because of the glare.")) == [
        ("multi-sentence", "b")
    ]


def test_more_than_one_number_in_a_line_is_flagged():
    line = nr.Line(
        "a", "{pulse:a}Sixteen times {pulse:b}6 is {pulse:c}96.", sentence_gap_s=0.4
    )
    assert _codes(line) == [("many-numbers", "a")]
    # Words of one number are one number.
    ok = nr.Line(
        "b", "It takes {pulse:a}twenty one hours and {pulse:b}three point five."
    )
    assert [c for c, _ in _codes(ok)] == ["many-numbers"]
    one = nr.Line("c", "It takes {pulse:a}twenty one hours.")
    assert _codes(one) == []
    assert _codes(nr.Line("d", "One of them is {pulse:a}6.")) == []
    # A bare "one" is a number only beside a word that makes it a quantity.
    assert _codes(nr.Line("e", "It picks out one separation of the profile.")) == []
    assert _codes(nr.Line("f", "Step one {pulse:a}starts here.")) == []
    assert _codes(nr.Line("g", "Step one starts here.")) == [
        ("number-without-cue", "g")
    ]


def test_a_number_needs_a_cue_within_two_words():
    assert _codes(nr.Line("a", "The ratio is 6.")) == [("number-without-cue", "a")]
    assert _codes(nr.Line("a", "The ratio is {pulse:x}6.")) == []
    assert _codes(nr.Line("a", "The ratio {pulse:x}is now six.")) == []
    far = nr.Line("a", "The {pulse:x}ratio is now at about six.")
    assert _codes(far) == [("number-without-cue", "a")]


def test_findings_name_the_line_and_a_message():
    (fault,) = nr.lint_script(nr.Script((nr.Line("a", "Some things. Done."),)))
    assert fault.key == "a" and "sentence" in fault.message


def test_check_script_speech_raises_on_the_rules_only_when_asked():
    script = nr.Script((nr.Line("a", "It is faint. Why is it faint?"),))
    nr.check_script_speech(script)
    with pytest.raises(nr.SpeechTextError, match="multi-sentence"):
        nr.check_script_speech(script, rules=True)
    clean = nr.Script((nr.Line("a", "It is faint."),))
    nr.check_script_speech(clean, rules=True)
