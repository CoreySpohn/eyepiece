"""Shared clip samples preserve the findings from independent passes."""

from contextlib import nullcontext
from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept


def _records(report):
    return (
        str(report),
        [
            [tuple(getattr(fault, f.name) for f in fields(fault)) for fault in group]
            for group in (report.readability, report.meaning, report.motion)
        ],
        report.unread_knobs,
        report.motion_ink,
    )


@pytest.mark.parametrize("ink", [False, True])
def test_shared_samples_match_independent_passes_and_render_once(monkeypatch, ink):
    fig, ax = plt.subplots(figsize=(4, 2), dpi=60, layout="constrained")
    (dot,) = ax.plot([0.2], [0.5], "o", color="black", gid="dot")
    label = ax.text(0.3, 0.5, "small", fontsize=5, gid="label")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    tl = ep.Timeline({"x": 0.2}, tempo=ep.Tempo(factor=1))
    tl.hold(0.3)
    tl.run("slide", 0.6, to={"x": 0.6})
    tl.hold(0.3)
    plan = tl.at_fps(10)

    def draw(k):
        x = plan.value(k, "x")
        dot.set_xdata([x + (0.5 if k >= 7 else 0)])
        label.set_x(x)

    kwargs = {"keys": {"dot": "a", "label": "b"}, "driven": [dot] if ink else None}
    # Freeze layout after settling, just as a clip builder does.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    with monkeypatch.context() as patch:
        patch.setattr(ept, "_share_samples", lambda *a, **k: nullcontext())
        patch.setattr(ept._RasterPool, "intern", lambda self, frame, **k: frame.copy())
        separate = ept.check_clip(fig, draw, plan, **kwargs)
    pixels = np.asarray(fig.canvas.buffer_rgba()).copy()
    calls = []
    original = fig.canvas.draw

    def counted():
        calls.append(1)
        return original()

    monkeypatch.setattr(fig.canvas, "draw", counted)
    renderer = fig.canvas.get_renderer()
    painted = []
    original_text = renderer.draw_text

    def paint_text(*args, **kwargs):
        painted.append(1)
        return original_text(*args, **kwargs)

    monkeypatch.setattr(renderer, "draw_text", paint_text)
    fig.canvas.draw()
    texts_per_frame = len(painted)
    assert texts_per_frame > 0
    painted.clear()
    calls.clear()
    shared = ept.check_clip(fig, draw, plan, **kwargs)
    assert shared.faults
    assert _records(shared) == _records(separate)
    assert len(calls) == plan.n_frames
    assert len(painted) == texts_per_frame * plan.n_frames
    np.testing.assert_array_equal(fig.canvas.buffer_rgba(), pixels)
    assert dot.get_xdata() == [pytest.approx(1.1)]
    plt.close(fig)


def test_raster_identity_does_not_hide_geometry_changes():
    # Subpixel geometry can change without changing any rendered pixels.
    from eyepiece._check_samples import _RasterPool

    pool = _RasterPool()
    a = np.zeros((3, 4, 4), dtype=np.uint8)
    first = pool.intern(a)
    a[1, 1, 0] = 1
    second = pool.intern(a)
    assert first is not second
    assert first[1, 1, 0] == 0
    assert pool.intern(a.copy()) is second
    assert not first.flags.writeable


@pytest.mark.parametrize("threshold", [0, 1, 2, 5, 50, -1, float("nan")])
def test_still_change_threshold_matches_full_buffer(threshold):
    from eyepiece.testing import _pixel_change_reaches

    rng = np.random.default_rng(5)
    reference = rng.integers(0, 256, (7, 11, 3), dtype=np.uint8)
    for change in (0, 12, 13, 255):
        frame = reference.copy()
        frame[::2, ::3] = change
        changed = np.count_nonzero(
            np.any(
                np.abs(frame.astype(np.int16) - reference.astype(np.int16)) > 12, axis=2
            )
        )
        assert _pixel_change_reaches(frame, reference, threshold) == (
            changed >= threshold
        )


def test_raster_pool_is_bounded_and_checks_hash_collisions(monkeypatch):
    import eyepiece._check_samples as samples

    class SameHash:
        def digest(self):
            return b"collision"

    monkeypatch.setattr(samples.hashlib, "sha256", lambda frame: SameHash())
    pool = samples._RasterPool(max_bytes=32)
    a = pool.intern(np.zeros((2, 2, 4), dtype=np.uint8))
    b = pool.intern(np.ones((2, 2, 4), dtype=np.uint8))
    assert a is not b
    assert not np.array_equal(a, b)
    assert pool.nbytes <= 32
    # A frame larger than the byte budget is usable but is not retained.
    pool.intern(np.ones((10, 10, 4), dtype=np.uint8))
    assert pool.nbytes <= 32


def test_identical_pixels_keep_distinct_subpixel_geometry():
    fig = plt.figure(figsize=(2, 1), dpi=60)
    text = fig.text(0.2, 0.5, "same pixels")

    def at(t):
        text.set_x(0.2 + t * 1e-9)

    trace = ept._sample_trace(fig, at, [0, 0.1, 0.2], dpi=None, keep_frames=True)
    assert trace.frames[0] is trace.frames[-1]
    assert trace.box(text, 0) != trace.box(text, 2)
    faults = ept.motion_faults(fig, at, [0, 0.1, 0.2], rest=[text], tol_px=1e-10)
    assert any(f.kind == "drift" for f in faults)
    plt.close(fig)


def test_failed_shared_check_restores_context_and_knob_reads(monkeypatch):
    from eyepiece._check_samples import _ACTIVE

    fig = plt.figure(figsize=(2, 1), dpi=60)
    fig.text(0.2, 0.5, "text")
    tl = ep.Timeline({"x": 0.0})
    tl.hold(0.5)
    plan = tl.at_fps(10)
    plan.value(0, "x")
    before = set(plan._read)
    original = ept._collect_once

    def fail(*args):
        raise RuntimeError("capture failed")

    with monkeypatch.context() as patch:
        patch.setattr(ept, "_collect_once", fail)
        with pytest.raises(RuntimeError, match="capture failed"):
            ept.check_clip(fig, lambda k: None, plan, keys={})
    assert _ACTIVE.get() is None
    assert plan._read == before
    assert ept._collect_once is original
    ept.check_clip(fig, lambda k: plan.value(k, "x"), plan, keys={})
    plt.close(fig)


def test_non_raster_canvas_uses_independent_checks():
    from matplotlib.backend_bases import FigureCanvasBase

    fig = plt.figure(figsize=(2, 1), dpi=60)
    fig.text(0.2, 0.5, "text")
    tl = ep.Timeline()
    tl.hold(0.5)
    plan = tl.at_fps(10)
    canvas = FigureCanvasBase(fig)
    ept.check_clip(fig, lambda k: None, plan)
    assert fig.canvas is canvas
    plt.close(fig)


def _legacy_on_marks(fig, texts, boxes, dpi):
    # Frozen pre-optimization loop, kept here to pin traversal and selection.
    from matplotlib.colors import to_rgba

    cache = {}
    found = []
    for text, box in zip(texts, boxes, strict=True):
        if ept._protected(text):
            continue
        base = to_rgba(text.get_color())
        text_box = text.get_window_extent(ept._renderer(fig))
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            for mark in ept._default_avoid(ax, set()):
                if not mark.get_visible() or not ept._data_colors(mark):
                    continue
                if mark not in cache:
                    cache[mark] = ept._parts(mark, dpi)
                for part in cache[mark]:
                    if part[0][3] < ept._SHOWN or not ept._same_color(part[0], base):
                        continue
                    if ept._hit(part, text_box):
                        detail = (
                            f"{ept._name(text)} is drawn over a "
                            f"{type(mark).__name__} of its color"
                        )
                        found.append(("on-marks", text, mark, ept._tuple(box), detail))
                        break
                else:
                    continue
                break
    return found


def test_on_marks_matches_legacy_traversal_with_transparent_parts(monkeypatch):
    from matplotlib.patches import Rectangle

    fig, (ax, other) = plt.subplots(1, 2, figsize=(4, 2), dpi=60)
    for panel in (ax, other):
        panel.set(xlim=(0, 1), ylim=(0, 1))
        panel.plot([0, 1], [0.5, 0.5], lw=15, color="black")
        panel.plot([0, 1], [0.7, 0.7], lw=15, color=(0, 0, 0, 0))
        panel.add_patch(
            Rectangle((0.1, 0.3), 0.8, 0.1, facecolor="none", edgecolor="black")
        )
        panel.scatter([0.5], [0.3], s=100, color="black")
        panel.text(0.2, 0.5, "crossing", color="black")
        panel.text(0.2, 0.7, "clear", color="black")
        panel.text(0.2, 0.3, "backed", color="black", bbox={"fc": "white"})
    fig.canvas.draw()
    texts = ept._visible_texts(fig)
    boxes = [ept._bbox(t, ept._renderer(fig)) for t in texts]
    expected = _legacy_on_marks(fig, texts, boxes, fig.dpi)
    assert expected
    selected = []
    original = ept._parts

    def counted(mark, dpi):
        selected.append(mark)
        return original(mark, dpi)

    monkeypatch.setattr(ept, "_parts", counted)
    assert ept._on_marks(fig, texts, boxes, fig.dpi) == expected
    assert all(any(c[3] >= ept._SHOWN for c in ept._data_colors(a)) for a in selected)
    plt.close(fig)


def test_still_screen_findings_match_full_buffer_with_reused_source(monkeypatch):
    import eyepiece.narration as nr

    line = nr.Line("speech", "One two three four five six.", pad_after_s=0)
    words = tuple(nr.Word(w, i, i + 0.8) for i, w in enumerate(line.words))
    narration = nr.Narration(
        nr.Script((line,)), {"speech": nr.Spoken("speech", 6, words)}
    )
    buffer = np.zeros((97, 40, 3), dtype=np.uint8)

    def source(t):
        buffer[:] = 0
        # Put differences in the last tile and ignored caption rows, with
        # twelve-level changes that must not count and thirteen that must.
        buffer[-1] = int(t) * 20
        buffer[65:70, :5] = 12 if t < 3 else 13
        return buffer

    def full(frame, reference, threshold):
        return (
            np.count_nonzero(
                np.any(
                    np.abs(frame.astype(np.int16) - reference.astype(np.int16)) > 12,
                    axis=2,
                )
            )
            >= threshold
        )

    with monkeypatch.context() as patch:
        patch.setattr(ept, "_pixel_change_reaches", full)
        expected = ept.timing_faults(narration, source=source, fps=5, ignore_bottom=0.2)
    actual = ept.timing_faults(narration, source=source, fps=5, ignore_bottom=0.2)
    assert actual == expected
    assert any(f.kind == "still-speech" for f in actual)


def test_still_threshold_preserves_shape_mismatch_error():
    for threshold in (0, 1):
        with pytest.raises(ValueError, match="broadcast"):
            ept._pixel_change_reaches(
                np.zeros((5, 8, 3), dtype=np.uint8),
                np.zeros((7, 8, 3), dtype=np.uint8),
                threshold,
            )


def test_callback_enabling_live_layout_matches_independent_passes(monkeypatch):
    def run(shared):
        fig, ax = plt.subplots(figsize=(4, 2), dpi=60)
        text = ax.text(0.2, 0.6, "small", fontsize=5)
        tl = ep.Timeline()
        tl.hold(0.5)
        plan = tl.at_fps(10)
        count = []
        original = fig.canvas.draw

        def counted():
            count.append(1)
            original()

        def draw(k):
            fig.set_layout_engine("constrained")
            text.set_x(0.2 + k * 0.02)

        with monkeypatch.context() as patch:
            patch.setattr(fig.canvas, "draw", counted)
            if not shared:
                patch.setattr(ept, "_share_samples", lambda *a, **k: nullcontext())
            report = ept.check_clip(fig, draw, plan)
        artists = {a: i for i, a in enumerate(fig.findobj())}
        records = [
            [
                tuple(
                    artists.get(v, v) if hasattr(v, "draw") else v
                    for v in (getattr(fault, f.name) for f in fields(fault))
                )
                for fault in group
            ]
            for group in (report.readability, report.meaning, report.motion)
        ]
        plt.close(fig)
        return str(report), records, len(count)

    assert run(shared=True) == run(shared=False)


def test_filtered_artists_keep_independent_raster_passes(monkeypatch):
    # Agg filters replace the low-level renderer while drawing. Matplotlib's
    # disabled-draw context cannot suppress all of those replacement methods.
    fig, ax = plt.subplots(figsize=(2, 1), dpi=60)
    ax.scatter(
        [0.5], [0.5], color="black", s=100, agg_filter=lambda image, dpi: (image, 0, 0)
    )
    tl = ep.Timeline()
    tl.hold(0.5)
    plan = tl.at_fps(10)
    original = fig.canvas.draw
    count = []

    def counted():
        count.append(1)
        original()

    monkeypatch.setattr(fig.canvas, "draw", counted)
    ept.check_clip(fig, lambda k: None, plan)
    assert len(count) == plan.n_frames + len(plan.review_frames())
    plt.close(fig)


@pytest.mark.parametrize("identical", [False, True])
def test_still_threshold_avoids_full_buffer_conversion(identical):
    converted = []

    class CountConversions(np.ndarray):
        def astype(self, *args, **kwargs):
            converted.append(self.shape[0])
            return super().astype(*args, **kwargs)

    reference = np.zeros((128, 8, 3), dtype=np.uint8).view(CountConversions)
    frame = reference.copy()
    if not identical:
        frame[0] = 255
    assert ept._pixel_change_reaches(frame, reference, 1) == (not identical)
    assert sum(converted) < 2 * len(frame)
