"""A counter or result lands on the number the narration says."""

import numpy as np
import pytest

from eyepiece import narration as nr

SCRIPT = nr.Script(
    (
        nr.Line("a", "The ratio is {pulse:x}96 today.", pad_after_s=0.2),
        nr.Line("b", "It takes {pulse:y}twenty one hours.", pad_after_s=0.2),
        nr.Line("c", "A small {pulse:z}three point five.", pad_after_s=0.2),
    )
)
TL = nr.Narration.estimated(SCRIPT)
WORD = TL.beat("a").words[3].start_s  # "96"
SPAN = (0.0, TL.beats[-1].end_s)


def _count(t0, t1, v0, v1, fps=30.0, until=None):
    """A readout counting linearly from v0 to v1 over [t0, t1], then holding."""
    until = TL.beat("a").end_s if until is None else until
    ts = np.arange(0.0, until, 1.0 / fps)
    return [(float(t), float(np.interp(t, [t0, t1], [v0, v1]))) for t in ts]


def _check(values, **kwargs):
    return nr.check_number_sync(
        TL, [{"word_key": ("a", 3), "span": SPAN, "values": values}], **kwargs
    )


def test_a_counter_that_lands_before_the_word_is_clean():
    assert _check(_count(0.0, WORD - 0.1, 0.0, 96.0)) == []


def test_a_counter_that_lands_after_the_word_is_late_by_how_much():
    (fault,) = _check(_count(0.0, WORD + 0.5, 0.0, 96.0))
    assert fault.code == "late-settle"
    assert fault.key == "a" and fault.word == 3
    assert "0.5" in fault.message


def test_the_settle_tolerance_is_a_keyword_defaulting_to_zero():
    values = _count(0.0, WORD + 0.15, 0.0, 96.0)
    assert [f.code for f in _check(values)] == ["late-settle"]
    assert _check(values, settle_tol_s=0.2) == []


def test_a_readout_that_lands_on_another_number_contradicts():
    faults = _check(_count(0.0, WORD - 0.1, 0.0, 69.0))
    assert [f.code for f in faults] == ["contradicts"]
    assert "96" in faults[0].message and "69" in faults[0].message


def test_an_overshoot_on_the_way_contradicts_the_spoken_number():
    values = _count(0.0, WORD - 0.3, 0.0, 120.0)
    values = [(t, min(v, 120.0) if t < WORD - 0.3 else 96.0) for t, v in values]
    faults = _check(values)
    assert [f.code for f in faults] == ["contradicts"]
    assert "overshoot" in faults[0].message


def test_a_path_that_backs_up_contradicts_it():
    values = _count(0.0, WORD - 0.1, 0.0, 96.0)
    values = [(t, v - 15.0 if 0.3 < t < 0.4 else v) for t, v in values]
    (fault,) = _check(values)
    assert fault.code == "contradicts" and "back" in fault.message


@pytest.mark.parametrize(
    ("key", "index", "number"),
    [("a", 3, 96.0), ("b", 2, 21.0), ("c", 2, 3.5)],
)
def test_the_spoken_number_is_read_from_digits_and_number_words(key, index, number):
    beat = TL.beat(key)
    start = beat.words[index].start_s
    values = [(start - 0.5 + 0.05 * i, number) for i in range(20)]
    binding = {"word_key": (key, index), "span": SPAN, "values": values}
    assert nr.check_number_sync(TL, [binding]) == []
    wrong = [(t, number + 1) for t, _ in values]
    binding["values"] = wrong
    assert [f.code for f in nr.check_number_sync(TL, [binding])] == ["contradicts"]


def test_a_word_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError, match="not a number"):
        nr.check_number_sync(
            TL, [{"word_key": ("a", 0), "span": SPAN, "values": [(0.0, 1.0)]}]
        )


def test_a_binding_for_no_line_or_with_no_values_is_refused():
    with pytest.raises(KeyError):
        nr.check_number_sync(
            TL, [{"word_key": ("zz", 0), "span": SPAN, "values": [(0.0, 1.0)]}]
        )
    with pytest.raises(ValueError, match="values"):
        nr.check_number_sync(TL, [{"word_key": ("a", 3), "span": SPAN, "values": []}])
