"""save_fig: directory resolution chain and mode-aware kwargs."""

import hwostyle
import matplotlib.pyplot as plt

from eyepiece.output import save_fig


def test_dir_resolution_chain(tmp_path, monkeypatch):
    fig, _ = plt.subplots()
    explicit = save_fig(fig, "a", dir=tmp_path / "explicit")
    assert explicit.parent == tmp_path / "explicit" and explicit.exists()

    anchor = tmp_path / "scripts" / "fig_script.py"
    anchor.parent.mkdir()
    anchor.touch()
    anchored = save_fig(fig, "b", anchor=anchor)
    assert anchored.parent == anchor.parent / "output"

    monkeypatch.setenv("EYEPIECE_OUT", str(tmp_path / "env"))
    enved = save_fig(fig, "c")
    assert enved.parent == tmp_path / "env"
    plt.close(fig)


def test_dark_mode_png_is_not_white(tmp_path):
    hwostyle.use("dark")
    try:
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        path = save_fig(fig, "dark", dir=tmp_path)
        corner = plt.imread(path)[0, 0, :3]
        assert corner.max() < 0.5  # black-ish, not white
    finally:
        hwostyle.use("light")
        plt.close("all")


def test_suffix_and_return(tmp_path):
    fig, _ = plt.subplots()
    p = save_fig(fig, "named.pdf", dir=tmp_path)
    assert p.suffix == ".pdf" and p.exists()
    plt.close(fig)


def test_overrides_win(tmp_path):
    fig, _ = plt.subplots()
    p = save_fig(fig, "lowdpi", dir=tmp_path, dpi=50)
    img = plt.imread(p)
    assert img.shape[0] < 400  # 50 dpi image is small
    plt.close(fig)
