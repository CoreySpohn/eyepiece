"""A line may set its own speech speed; it enters that line's cache key."""

import pytest

from eyepiece import narration as nr


def test_a_line_speed_defaults_to_the_script_speed_and_keeps_the_key():
    plain = nr.Script((nr.Line("a", "A definition."),), speed=0.9)
    line = plain.line("a")
    assert line.speed is None
    assert plain.line_key(line) == nr.line_hash("A definition.", "bm_george", 0.9, "b")
    same = nr.Script((nr.Line("a", "A definition.", speed=0.9),), speed=0.9)
    assert same.cache_keys() == plain.cache_keys()


def test_the_line_speed_enters_the_cache_key():
    base = nr.Script((nr.Line("a", "A definition."),))
    slow = nr.Script((nr.Line("a", "A definition.", speed=0.8),))
    assert slow.cache_keys() != base.cache_keys()
    assert slow.cache_keys()[0] == nr.line_hash("A definition.", "bm_george", 0.8, "b")


def test_only_the_slowed_line_changes_key():
    lines = (nr.Line("a", "Plain."), nr.Line("b", "Definition.", speed=0.8))
    mixed = nr.Script(lines)
    flat = nr.Script((nr.Line("a", "Plain."), nr.Line("b", "Definition.")))
    assert mixed.cache_keys()[0] == flat.cache_keys()[0]
    assert mixed.cache_keys()[1] != flat.cache_keys()[1]


@pytest.mark.parametrize("speed", [0.0, -1.0, float("nan")])
def test_a_line_speed_must_be_positive(speed):
    with pytest.raises(ValueError, match="speed"):
        nr.Line("a", "Words.", speed=speed)


def test_the_script_names_the_speed_each_line_is_spoken_at():
    script = nr.Script(
        (nr.Line("a", "Words.", speed=0.7), nr.Line("b", "More.")), speed=1.1
    )
    assert [script.speed_of(line) for line in script.lines] == [0.7, 1.1]


def test_the_speed_round_trips_and_is_omitted_when_unset(tmp_path):
    script = nr.Script((nr.Line("a", "Words.", speed=0.7), nr.Line("b", "More.")))
    path = tmp_path / "s.json"
    script.to_json(path)
    text = path.read_text()
    assert text.count('"speed"') == 2  # the script's and line a's
    assert nr.Script.from_json(path) == script
