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


def _mixed_fig(scale=1.0):
    fig, ax = plt.subplots(figsize=(3, 2), layout="constrained")
    ax.plot([0, 1, 2], [0, scale, 4])
    ax.set(xlabel="x", ylabel="y")
    ep.stamp(fig, note="simulated", seed=1, date="2026-08-21", sha="abc1234")
    return fig


@pytest.mark.parametrize("suffix", ["png", "svg", "pdf"])
def test_save_fig_is_byte_identical_across_saves(tmp_path, suffix):
    fig = _mixed_fig()
    a = ep.save_fig(fig, f"a.{suffix}", dir=tmp_path)
    b = ep.save_fig(fig, f"b.{suffix}", dir=tmp_path)
    assert a.read_bytes() == b.read_bytes()
    fig2 = _mixed_fig()
    c = ep.save_fig(fig2, f"c.{suffix}", dir=tmp_path)
    assert a.read_bytes() == c.read_bytes()
    plt.close("all")


@pytest.mark.parametrize("suffix", ["png", "svg", "pdf"])
def test_save_fig_changed_figure_changes_file(tmp_path, suffix):
    a = ep.save_fig(_mixed_fig(1.0), f"a.{suffix}", dir=tmp_path)
    b = ep.save_fig(_mixed_fig(3.0), f"b.{suffix}", dir=tmp_path)
    assert a.read_bytes() != b.read_bytes()
    plt.close("all")


def test_save_fig_writes_no_timestamp_but_keeps_provenance(tmp_path):
    fig = _mixed_fig()
    svg = ep.save_fig(fig, "a.svg", dir=tmp_path).read_text()
    assert "<dc:date>" not in svg
    assert "abc1234" in svg
    pdf = ep.save_fig(fig, "a.pdf", dir=tmp_path).read_bytes()
    assert b"CreationDate" not in pdf
    assert b"Producer" not in pdf
    assert b"abc1234" in pdf
    plt.close(fig)


def test_save_fig_explicit_metadata_wins(tmp_path):
    fig = _mixed_fig()
    pdf = ep.save_fig(
        fig, "a.pdf", dir=tmp_path, metadata={"Producer": "mine"}
    ).read_bytes()
    assert b"mine" in pdf
    assert b"abc1234" in pdf  # provenance keys stay
    plt.close(fig)


@pytest.mark.parametrize("layout", ["constrained", "tight", None])
def test_reserved_stamp_does_not_overlap_axes(layout):
    from eyepiece import testing

    fig, ax = plt.subplots(figsize=(4, 3), layout=layout)
    ax.plot([0, 1], [0, 1])
    ax.set(xlabel="x label", ylabel="y")
    result = ep.stamp(fig, note="simulated", reserve=True)
    assert result.artists["text"].get_gid() == ep.provenance.STAMP_GID
    faults = testing.readability_faults(fig, kinds=("overlap",))
    assert faults == []
    if layout is not None:
        fig.canvas.draw()
        stamp_box = result.artists["text"].get_window_extent()
        assert ax.get_tightbbox().y0 >= stamp_box.y1 - 1.0
    plt.close(fig)


def test_unreserved_stamp_overlaps_a_full_height_axes():
    """Positive control: without reserve a bottom-hugging axes collides."""
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes((0.1, 0.0, 0.8, 0.9))
    ax.plot([0, 1], [0, 1])
    text = ep.stamp(fig, note="simulated")
    fig.canvas.draw()
    assert ax.get_window_extent().overlaps(text.get_window_extent())
    plt.close(fig)


def test_stamp_date_defaults_to_commit_date_else_today(tmp_path):
    import datetime
    import subprocess

    script = tmp_path / "make.py"
    script.write_text("")
    today = datetime.date.today().isoformat()
    assert eyepiece.provenance.provenance_text(script=script, sha="x").endswith(today)
    env = {
        "GIT_AUTHOR_DATE": "2020-01-02T03:04:05",
        "GIT_COMMITTER_DATE": "2020-01-02T03:04:05",
        "PATH": __import__("os").environ["PATH"],
        "HOME": str(tmp_path),
    }
    for cmd in (
        ["init", "-q"],
        ["add", "make.py"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "m"],
    ):
        subprocess.run(["git", "-C", str(tmp_path), *cmd], check=True, env=env)
    text = eyepiece.provenance.provenance_text(script=script)
    assert "2020-01-02" in text
    assert eyepiece.provenance.provenance_fields(script=script)["date"] == "2020-01-02"
