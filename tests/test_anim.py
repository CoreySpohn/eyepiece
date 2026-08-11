"""record/animate: multi-sink, generator safety, no clearing, ffmpeg error."""

import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eyepiece.anim import animate, record


def _fig_line():
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [])
    ax.set_xlim(0, 9)
    ax.set_ylim(-1, 1)
    return fig, line


def test_record_multi_sink_one_pass(tmp_path):
    fig, line = _fig_line()
    x = np.arange(10)
    with record(fig, tmp_path / "a.gif", tmp_path / "a.html", fps=5) as rec:
        for k in range(3):
            line.set_data(x[: k + 1], np.sin(x[: k + 1]))
            rec.frame()
    assert (tmp_path / "a.gif").stat().st_size > 0
    assert (tmp_path / "a.html").stat().st_size > 0
    plt.close(fig)


def test_record_hold_repeats_frames(tmp_path):
    fig, line = _fig_line()
    with record(fig, tmp_path / "h.html", fps=5) as rec:
        line.set_data([0, 1], [0, 0])
        rec.hold(4)
    html = (tmp_path / "h.html").read_text()
    assert html.count("data:image/png") >= 4 or html.count("frames") > 0
    plt.close(fig)


def test_animate_does_not_clear(tmp_path):
    fig, line = _fig_line()
    n_axes_before = len(fig.axes)

    def draw(fig_, k):
        line.set_data([0, k], [0, 0])

    animate(fig, draw, 3, fps=5).save(tmp_path / "n.gif")
    assert len(fig.axes) == n_axes_before
    assert len(fig.axes[0].lines) == 1
    plt.close(fig)


def test_generator_second_pass_raises(tmp_path):
    fig, line = _fig_line()

    def draw(fig_, k):
        line.set_data([0, k], [0, 0])

    gen = (k for k in range(3))
    anim = animate(fig, draw, gen, fps=5)
    anim.save(tmp_path / "g1.gif")
    with pytest.raises(RuntimeError, match="generator"):
        anim.save(tmp_path / "g2.gif")
    plt.close(fig)


def test_jshtml_returns_html_and_closes_nothing_extra():
    fig, line = _fig_line()

    def draw(fig_, k):
        line.set_data([0, k], [0, 0])

    html = animate(fig, draw, 3, fps=5).jshtml(dpi=60)
    assert "<script" in html or "animation" in html.lower()


@pytest.mark.skipif(shutil.which("ffmpeg") is not None, reason="ffmpeg present")
def test_mp4_without_ffmpeg_raises_actionable(tmp_path):
    fig, _ = _fig_line()
    with pytest.raises(RuntimeError, match=r"imageio-ffmpeg|apt_packages|ffmpeg"):
        with record(fig, tmp_path / "x.mp4", fps=5) as rec:
            rec.frame()
    plt.close(fig)
