"""The scrubber page: a slider over pre-rendered frames in one offline HTML file."""

import base64
import io
import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest

import eyepiece as ep
from eyepiece import narration as nr

DPI = 80
SIZE = (64, 48)
FRAME = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")
SCRIPT = nr.Script(
    (nr.Line("only", "Watch the {indicate:dot}dot climb to the top.", pad_after_s=0.2),)
)


def tiny_figure():
    """A 64 by 48 pixel figure at 80 dpi and a draw function over knob u."""
    fig, ax = plt.subplots(figsize=(SIZE[0] / DPI, SIZE[1] / DPI), dpi=DPI)
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    (dot,) = ax.plot([0.5], [0.0], "o", ms=4)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, dot


def silent_plan():
    tl = ep.Timeline({"u": 0.0}, tempo=ep.Tempo(factor=1.0))
    tl.hold(0.1, name="open")
    tl.run("rise", 0.3, to={"u": 1.0})
    tl.hold(0.1, name="rest")
    return tl.at_fps(10)


def narrated_plan():
    tl = ep.Timeline({"u": 0.0}, narration=nr.Narration.estimated(SCRIPT), strict=False)
    tl.line("only")
    tl.pulse("dot")
    tl.run("rise", 0.3, to={"u": 1.0}, at=0.5)
    return tl.at_fps(10)


def scrub(plan, samples=(0.0, 0.5, 1.0), **kwargs):
    fig, dot = tiny_figure()

    def draw(k):
        dot.set_ydata([plan.value(k, "u")])

    try:
        return ep.to_scrubber(fig, draw, plan, "u", samples, dpi=DPI, **kwargs)
    finally:
        plt.close(fig)


def frames(page):
    return [base64.b64decode(m) for m in FRAME.findall(page)]


def test_one_range_input_steps_over_the_samples():
    page = scrub(silent_plan())
    inputs = re.findall(r"<input[^>]*>", page)
    assert len(inputs) == 1
    tag = inputs[0]
    assert 'type="range"' in tag
    assert 'min="0"' in tag and 'max="2"' in tag and 'step="1"' in tag


def test_one_embedded_frame_per_sample_at_the_figure_size():
    samples = (0.0, 0.25, 0.5, 1.0)
    found = frames(scrub(silent_plan(), samples))
    assert len(found) == len(samples)
    for png in found:
        image = mpimg.imread(io.BytesIO(png), format="png")
        assert image.shape[:2] == (SIZE[1], SIZE[0])


def test_the_page_needs_no_network():
    page = scrub(narrated_plan())
    for banned in (
        "http://",
        "https://",
        "<link",
        "@import",
        "fetch(",
        "XMLHttpRequest",
    ):
        assert banned not in page
    for value in re.findall(r"(?:src|href)=\"([^\"]*)\"", page):
        assert value.startswith(("data:", "#"))


def test_two_renders_are_byte_identical():
    plan = silent_plan()
    assert scrub(plan) == scrub(plan)


def test_rest_frame_is_the_still_record_writes(tmp_path):
    plan = silent_plan()
    fig, dot = tiny_figure()
    still = tmp_path / "rest.png"
    last = plan.n_frames - 1
    with ep.record(fig, tmp_path / "clip.html", fps=plan.fps, dpi=DPI,
                   stills={still: last / plan.fps}) as rec:  # fmt: skip
        for k in range(plan.n_frames):
            dot.set_ydata([plan.value(k, "u")])
            rec.frame()
    plt.close(fig)
    page = scrub(plan)
    # The visible image before any script runs is the rest state.
    first = re.search(r"<img[^>]*src=\"data:image/png;base64,([^\"]+)\"", page)
    assert base64.b64decode(first.group(1)) == still.read_bytes()


def test_the_path_receives_the_same_page(tmp_path):
    path = tmp_path / "page.html"
    page = scrub(silent_plan(), path=path)
    assert path.read_text() == page


def test_bad_arguments_are_refused():
    plan = silent_plan()
    fig, _ = tiny_figure()
    with pytest.raises(KeyError, match="did you mean"):
        ep.to_scrubber(fig, lambda k: None, plan, "uu", (0.0,))
    plt.close(fig)
    with pytest.raises(ValueError, match="empty"):
        scrub(plan, samples=())
    with pytest.raises(ValueError, match="range"):
        scrub(plan, samples=(0.0, 2.0))
    with pytest.raises(ValueError, match="bytes"):
        scrub(plan, max_bytes=100)


def test_time_is_a_knob():
    plan = silent_plan()
    page = scrub(plan, samples=(0.0, plan.seconds - 1 / plan.fps))
    assert len(frames(page)) == 2


def test_transcript_and_chapters_follow_the_plan():
    narrated = scrub(narrated_plan())
    assert 'class="transcript"' in narrated
    assert "Watch the dot climb to the top." in narrated
    chapters = re.findall(r"<button[^>]*>([^<]*)</button>", narrated)
    assert chapters == ["pulse.dot"]
    silent = scrub(silent_plan())
    assert 'class="transcript"' not in silent
    assert re.findall(r"<button[^>]*>([^<]*)</button>", silent) == [
        "open",
        "rise",
        "rest",
    ]


def test_jpeg_frames_are_allowed():
    page = scrub(silent_plan(), format="jpeg", quality=80)
    assert page.count("data:image/jpeg;base64,") == 3
