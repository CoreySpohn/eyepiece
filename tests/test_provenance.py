"""The stamp that says what a reader needs to rebuild or challenge a figure."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

import eyepiece as ep


def test_provenance_text_assembles_only_supplied_fields():
    text = ep.provenance_text(
        script="/a/b/make_fig.py",
        seed=3,
        note="simulated",
        date="2026-08-21",
        sha="abc1234",
    )
    assert "make_fig.py" in text
    assert "/a/b/" not in text  # the full path is noise to a reader
    assert "abc1234" in text
    assert "2026-08-21" in text
    assert "seed 3" in text
    assert "simulated" in text


def test_provenance_text_omits_absent_fields():
    text = ep.provenance_text(date="2026-08-21")
    assert text == "2026-08-21"


def test_stamp_writes_furniture_not_data():
    """The stamp must never compete with the data it documents."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    text = ep.stamp(fig, script="s.py", seed=1, note="simulated", date="2026-08-21")
    assert text in fig.texts
    assert text.get_fontsize() <= 7.0
    assert text.get_zorder() < 1.0
    # a neutral, not the text color: this is scenery
    assert text.get_color() != plt.rcParams["text.color"]
    plt.close(fig)


def test_stamp_returns_none_with_nothing_to_say():
    fig, _ = plt.subplots()
    assert ep.stamp(fig, date="") is not None  # date defaults to today
    plt.close(fig)


@pytest.mark.parametrize("note", ["simulated", "simulated relative astrometry, 5 mas"])
def test_stamp_carries_the_word_simulated(note):
    """Nothing about a drawing reveals that a generator produced it."""
    fig, _ = plt.subplots()
    text = ep.stamp(fig, script="s.py", note=note)
    assert "simulated" in text.get_text()
    plt.close(fig)
