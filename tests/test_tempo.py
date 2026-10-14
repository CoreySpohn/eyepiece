"""Tempo: one knob for the pace of a staged clip."""

from itertools import pairwise

import pytest

import eyepiece as ep


def test_reading_hold_scales_with_words():
    tempo = ep.Tempo()
    holds = [tempo.read_s(" ".join(["word"] * n)) for n in range(9)]
    steps = [b - a for a, b in pairwise(holds)]
    assert steps == pytest.approx([tempo.factor * tempo.word_s] * 8)
    assert holds[0] == pytest.approx(tempo.factor * tempo.base_s)


def test_default_hold_is_about_one_second_plus_four_tenths_per_word():
    tempo = ep.Tempo()
    assert tempo.read_s("") == pytest.approx(1.0, abs=0.05)
    assert tempo.read_s("a b c d e") - tempo.read_s("") == pytest.approx(2.0, abs=0.05)


def test_default_pace_is_a_third_slower_than_factor_one():
    slow, brisk = ep.Tempo(), ep.Tempo(factor=1.0)
    caption = "add up this row of cells"
    for a, b in (
        (slow.read_s(caption), brisk.read_s(caption)),
        (slow.rest_s, brisk.rest_s),
        (slow.seconds(2.0), brisk.seconds(2.0)),
    ):
        assert 1.3 <= a / b <= 1.4


def test_factor_scales_every_duration():
    one, two = ep.Tempo(factor=1.0), ep.Tempo(factor=2.0)
    caption = r"the chance of $D$ given $q$"
    assert two.read_s(caption) == pytest.approx(2.0 * one.read_s(caption))
    assert two.rest_s == pytest.approx(2.0 * one.rest_s)
    assert two.seconds(1.5) == pytest.approx(3.0)


def test_a_symbol_in_math_reads_as_two_words():
    assert ep.reading_words("true q") == 2
    assert ep.reading_words(r"true $q$") == 3
    assert ep.reading_words(r"$\lambda \mid D$") == 4
    assert ep.reading_words(r"$x \pm 0.5$") == 4
    tempo = ep.Tempo()
    assert tempo.read_s(r"$q$") == pytest.approx(tempo.read_s("two words"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"factor": 0.0}, "factor"),
        ({"factor": float("nan")}, "factor"),
        ({"word_s": -0.1}, "word_s"),
        ({"settle_s": float("inf")}, "settle_s"),
    ],
)
def test_tempo_rejects_bad_durations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ep.Tempo(**kwargs)


def test_seconds_rejects_negative():
    with pytest.raises(ValueError, match="seconds"):
        ep.Tempo().seconds(-1.0)


def test_caption_pieces_read_as_their_text():
    pieces = ["add up ", ("the row", "tab:red"), (" of cells", None, "bold")]
    assert ep.reading_words(pieces) == ep.reading_words("add up the row of cells")
    assert ep.Tempo().read_s(pieces) == ep.Tempo().read_s("add up the row of cells")
