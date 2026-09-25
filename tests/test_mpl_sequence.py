"""Prepared sequences through Matplotlib: strips and the shared output schedule."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from eyepiece.style import snapshot_profile


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def drawn_means(monkeypatch):
    """Record the image mean each `MplResult.update` is handed, in call order."""
    import eyepiece.mpl as mpl

    means = []
    original = mpl.MplResult.update

    def spy(self, view):
        means.append(float(np.mean(view.data)))
        return original(self, view)

    monkeypatch.setattr(mpl.MplResult, "update", spy)
    return means


def _decoded_frames(path):
    """Every decoded GIF frame as RGB, with its duration in ms."""
    frames = []
    with Image.open(path) as im:
        for index in range(im.n_frames):
            im.seek(index)
            frames.append((np.asarray(im.convert("RGB")), im.info["duration"]))
    return frames


def test_animate_returns_the_public_animation_type(image_sequence):
    import eyepiece
    import eyepiece.mpl as mpl

    animation = mpl.animate(image_sequence(), run_time=1.0, fps=4)
    assert isinstance(animation, eyepiece.Animation)
    assert animation.n_frames == 4
    assert animation.fps == 4
    assert isinstance(animation.result, mpl.MplResult)
    assert animation.result.fig is animation.fig
    assert "image" in animation.result.parts


@pytest.mark.parametrize(
    ("fps", "expected"),
    [
        (4, [0.0, 10.0, 10.0, 20.0]),
        (10, [0.0] + [10.0] * 8 + [20.0]),
    ],
)
def test_frames_follow_the_schedule_not_the_input_count(
    image_sequence, drawn_means, tmp_path, fps, expected
):
    import eyepiece.mpl as mpl

    sequence = image_sequence()
    animation = mpl.animate(sequence, run_time=1.0, fps=fps)
    animation.save(tmp_path / "clip.gif")
    assert drawn_means == expected
    total_ms = sum(duration for _, duration in _decoded_frames(tmp_path / "clip.gif"))
    assert total_ms == pytest.approx(1000.0)


def test_duration_is_quantized_and_the_final_sample_is_encoded(
    image_sequence, tmp_path
):
    import eyepiece.mpl as mpl

    profile = snapshot_profile()
    fig, ax = plt.subplots(figsize=(3, 2), dpi=50)
    animation = mpl.animate(
        image_sequence(), run_time=0.25, fps=10, ax=ax, profile=profile
    )
    animation.save(tmp_path / "clip.gif", dpi=50)
    frames = _decoded_frames(tmp_path / "clip.gif")
    # ceil(0.25 * 10) = 3 output frames, so the encoded clip lasts 0.3 s.
    assert sum(duration for _, duration in frames) == 300
    fig.canvas.draw()
    px, py = ax.transData.transform((1.5, 1.0))
    last, _ = frames[-1]
    first, _ = frames[0]
    row, col = last.shape[0] - round(py), round(px)
    lut = profile.colormaps["intensity"]
    np.testing.assert_allclose(last[row, col], lut[255][:3], atol=8)
    np.testing.assert_allclose(first[row, col], lut[0][:3], atol=8)


def test_animation_leaves_caller_axes_in_their_slots(image_sequence, tmp_path):
    import eyepiece.mpl as mpl

    fig, (left, right) = plt.subplots(1, 2)
    positions = [a.get_position(original=True).bounds for a in (left, right)]
    animation = mpl.animate(image_sequence(), run_time=0.3, fps=10, ax=right)
    animation.save(tmp_path / "clip.gif")
    after = [a.get_position(original=True).bounds for a in (left, right)]
    assert after == positions
    assert animation.fig is fig
    assert len(left.images) == 0 and len(right.images) == 1
    assert fig.get_layout_engine() is None


def test_selected_strip_keeps_original_bounds_and_labels(image_sequence):
    import eyepiece.mpl as mpl

    profile = snapshot_profile()
    strip = image_sequence().strip([2, 0, 2])
    result = mpl.render(strip, profile=profile)
    fig = result.fig
    lut = profile.colormaps["intensity"]
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    for slot, (lut_index, text) in enumerate(
        ((255, "10 s"), (0, "0 s"), (255, "10 s"))
    ):
        ax = result.axes[f"{slot}/image"]
        image = result.parts[f"{slot}/image"]
        assert tuple(image.get_extent()) == (0, 3, 0, 2)
        norm = result.parts[f"{slot}/image/colorbar"].mappable.norm
        assert (norm.vmin, norm.vmax) == (0, 20)
        label = result.parts[f"{slot}/time"]
        assert label.get_text() == text
        assert label.get_transform() is ax.transAxes
        px, py = ax.transData.transform((1.5, 1.0))
        pixel = buffer[buffer.shape[0] - round(py), round(px)]
        np.testing.assert_array_equal(pixel, lut[lut_index])
