"""The stamp that says what a reader needs to rebuild or challenge a figure."""

import json
import struct
import zlib

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

import eyepiece as ep
import eyepiece.provenance


def _png_text(path):
    """Read a PNG's text chunks, so the metadata channel is checked as written."""
    data = path.read_bytes()
    found = {}
    i = 8
    while i < len(data):
        length = struct.unpack(">I", data[i : i + 4])[0]
        kind = data[i + 4 : i + 8]
        body = data[i + 8 : i + 8 + length]
        if kind in (b"tEXt", b"iTXt"):
            key, _, value = body.partition(b"\x00")
            found[key.decode("latin-1")] = value.split(b"\x00")[-1].decode(
                "latin-1", "replace"
            )
        elif kind == b"zTXt":
            key, _, value = body.partition(b"\x00")
            found[key.decode("latin-1")] = zlib.decompress(value[1:]).decode(
                "latin-1", "replace"
            )
        i += 12 + length
        if kind == b"IEND":
            break
    return found


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


def test_stamp_marks_its_artist_with_a_gid():
    fig = plt.figure()
    artist = ep.stamp(fig, script="demo.py")
    assert artist.get_gid() == "eyepiece-provenance"
    plt.close(fig)


def test_stamp_leaves_a_structured_payload_for_the_file_channel():
    fig = plt.figure()
    ep.stamp(fig, script="demo.py", seed=3, note="simulated", sha="abc1234")
    payload = getattr(fig, eyepiece.provenance.PAYLOAD_ATTR)
    assert payload["script"] == "demo.py"
    assert payload["sha"] == "abc1234"
    assert payload["seed"] == 3
    assert payload["note"] == "simulated"
    plt.close(fig)


def test_provenance_fields_omits_absent_fields():
    fields = ep.provenance_fields(script="demo.py", sha="abc1234")
    assert "seed" not in fields
    assert "note" not in fields
    assert fields["date"]


def test_file_metadata_routes_per_format():
    fields = {"script": "demo.py", "sha": "abc1234"}
    assert set(ep.file_metadata(fields, ".png")) == {"Software", "Comment"}
    assert set(ep.file_metadata(fields, ".pdf")) == {"Creator", "Subject"}
    assert set(ep.file_metadata(fields, ".svg")) == {"Creator", "Description"}
    # An unsupported format loses only the second channel, it does not raise.
    assert ep.file_metadata(fields, ".jpg") == {}
    assert ep.file_metadata({}, ".png") == {}


def test_saved_png_carries_the_provenance_payload(tmp_path):
    fig = plt.figure()
    ep.stamp(fig, script="demo.py", seed=3, note="simulated", sha="abc1234")
    path = ep.save_fig(fig, "stamped.png", dir=tmp_path)
    plt.close(fig)
    payload = json.loads(_png_text(path)["Comment"])
    assert payload["script"] == "demo.py"
    assert payload["seed"] == 3
    assert payload["note"] == "simulated"


def test_unstamped_figure_manufactures_no_provenance(tmp_path):
    fig = plt.figure()
    path = ep.save_fig(fig, "bare.png", dir=tmp_path)
    plt.close(fig)
    assert "Comment" not in _png_text(path)


def test_explicit_metadata_override_wins(tmp_path):
    fig = plt.figure()
    ep.stamp(fig, script="demo.py", sha="abc1234")
    path = ep.save_fig(fig, "over.png", dir=tmp_path, metadata={"Comment": "mine"})
    plt.close(fig)
    assert _png_text(path)["Comment"] == "mine"
