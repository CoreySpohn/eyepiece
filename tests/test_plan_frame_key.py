"""Exact plan state, independent key verification, and worker budgeting."""

import io
import struct
from dataclasses import replace
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.artist import allow_rasterization
from matplotlib.figure import Figure
from test_render_helpers import CountingRenderer

import eyepiece as ep
from eyepiece import _render_pool as pool
from eyepiece import _render_variants as variant_driver


def held_plan():
    tl = ep.Timeline({"x": 0.0, "caption": "first"})
    tl.section("one")
    tl.hold(0.3)
    tl.section("two")
    tl.hold(0.3, to={"caption": "second"})
    return tl.at_fps(10)


class PaintRenderer(CountingRenderer):
    """Count every public painting entry point used by Agg and vector backends."""

    def __init__(self, renderer):
        super().__init__(renderer)
        self.other = 0

    def draw_markers(self, *args, **kwargs):
        self.other += 1
        return self.renderer.draw_markers(*args, **kwargs)

    def draw_path_collection(self, *args, **kwargs):
        self.other += 1
        return self.renderer.draw_path_collection(*args, **kwargs)

    def draw_image(self, *args, **kwargs):
        self.other += 1
        return self.renderer.draw_image(*args, **kwargs)

    def draw_gouraud_triangles(self, *args, **kwargs):
        self.other += 1
        return self.renderer.draw_gouraud_triangles(*args, **kwargs)


class PaintFigure(Figure):
    """Count renderer painting without replacing instance draw methods."""

    @allow_rasterization
    def draw(self, renderer):
        counted = PaintRenderer(renderer)
        super().draw(counted)
        if hasattr(self, "paint_log"):
            with Path(self.paint_log).open("a") as stream:
                stream.write(f"{counted.paths + counted.texts + counted.other}\n")


def draw_plan(fig, k, *, plan):
    fig.set_facecolor("white" if plan.value(k, "caption") == "first" else "black")


def build_plan(*, paint_log):
    fig = plt.figure(FigureClass=PaintFigure, figsize=(0.96, 0.54), dpi=100)
    fig.paint_log = paint_log
    return fig, partial(draw_plan, plan=held_plan())


def build_spec_plan(venue, mode, *, paint_log):
    return build_plan(paint_log=paint_log)


def test_plan_key_covers_all_values_bits_types_sections_and_time():
    plan = held_plan()
    key = partial(plan.frame_key, include_time=False)
    assert key(0) == key(2)
    assert key(2) != key(3)
    assert plan.frame_key(0) != plan.frame_key(1)
    assert plan.unused_knobs()[0] == ("caption", "x")
    for value in (
        -0.0,
        float(np.nextafter(0.0, 1.0)),
        0,
        False,
        "0",
        None,
        np.float64(0),
        np.int64(0),
        np.float32(-0.0),
        np.str_("0"),
        1j,
    ):
        plan._knobs["x"][1] = value
        assert key(0) != key(1)
    plan._knobs["x"][1] = 0.0
    plan._labels[1] = "changed"
    assert key(0) != key(1)
    assert hash(key(0)) == hash(key(2))
    plan._labels[1] = plan._labels[0]
    nan = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    other_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0]
    plan._knobs["x"][0:3] = [nan, nan, other_nan]
    assert key(0) != key(1) and key(0) != key(2)
    plan._knobs["x"][0:2] = [(None, (True, b"x", -0.0)), (None, (True, b"x", 0.0))]
    assert key(0) != key(1)
    hash(key(0))


def test_plan_key_rejects_unknown_state_and_outside_frames():
    plan = held_plan()
    for k in (-1, plan.n_frames):
        with pytest.raises(IndexError):
            plan.frame_key(k)
    for value in (object(), [], {}, np.array([1]), np.void(b"a")):
        plan._knobs["x"][0] = value
        with pytest.raises(TypeError, match="frame_key"):
            plan.frame_key(0)


@pytest.mark.parametrize(
    "first,second",
    [(np.int64(7), np.longlong(7)), (7, np.int64(7)), (7.0, np.float64(7))],
)
def test_plan_key_scalar_types_preserve_raster_bytes(first, second):
    tl = ep.Timeline({"x": first})
    tl.hold(0.2)
    tl.hold(0.2, to={"x": second})
    plan = tl.at_fps(10)
    key = partial(plan.frame_key, include_time=False)
    assert type(plan.value(0, "x")) is not type(plan.value(2, "x"))

    def build():
        fig = plt.figure(figsize=(0.96, 0.54))

        def draw(fig, k):
            fig.set_facecolor(
                "white" if type(plan.value(k, "x")) is type(first) else "black"
            )

        return fig, draw

    outputs = []
    for candidate in (None, key):
        with pool._figure(
            build, 4, 100, None, None, None, frame_key=candidate
        ) as state:
            outputs.append([pool._raster(state, k).tobytes() for k in range(4)])
    assert outputs[0][0] != outputs[0][2]
    assert outputs[0] == outputs[1]
    assert key(0) != key(2)


@pytest.mark.parametrize("scalar", [float, np.float32, np.float64, np.longdouble])
def test_plan_key_nan_lookup_preserves_raster_bytes(scalar):
    first, second = scalar("nan"), scalar("nan")
    palette = {first: "white"}
    tl = ep.Timeline({"x": first})
    tl.hold(0.2)
    tl.hold(0.2, to={"x": second})
    plan = tl.at_fps(10)
    key = partial(plan.frame_key, include_time=False)

    def build():
        fig = plt.figure(figsize=(0.96, 0.54))

        def draw(fig, k):
            fig.set_facecolor(palette.get(plan.value(k, "x"), "black"))

        return fig, draw

    outputs = []
    for candidate in (None, key):
        with pool._figure(
            build, 4, 100, None, None, None, frame_key=candidate
        ) as state:
            outputs.append([pool._raster(state, k).tobytes() for k in range(4)])
    assert outputs[0][0] != outputs[0][2]
    assert outputs[0] == outputs[1]
    assert key(0) != key(0)
    assert key(0) != key(2)


def test_verifier_ignores_tight_savefig_bbox():
    fig, _ = plt.subplots(figsize=(2, 2), dpi=100)
    try:
        with matplotlib.rc_context({"savefig.bbox": "tight"}):
            assert (
                ep.testing.check_frame_key(
                    fig, lambda k: None, lambda k: 0, frames=[0, 1]
                )
                == []
            )
            assert matplotlib.rcParams["savefig.bbox"] == "tight"
    finally:
        plt.close(fig)


def test_verifier_reports_every_bad_run_and_gap_without_skipping_paint(tmp_path):
    fig = plt.figure(FigureClass=PaintFigure, figsize=(0.2, 0.2), dpi=100)
    fig.paint_log = tmp_path / "verify.paint"
    original_canvas = fig.canvas
    calls = []

    def draw(k):
        calls.append(k)
        fig.set_facecolor("white" if k % 2 == 0 else "black")

    frames = [0, 1, 2, 4, 5, 7, 8]
    with matplotlib.rc_context({"savefig.dpi": 200}):
        faults = ep.testing.check_frame_key(fig, draw, lambda k: k // 4, frames=frames)
    assert calls == frames
    assert [f["frames"] for f in faults] == [(0, 1, 2), (4, 5)]
    assert [f["key"] for f in faults] == [0, 1]
    assert [f["pixel_count"] for f in faults] == [400, 400]
    assert fig.canvas is original_canvas
    assert len(fig.paint_log.read_text().splitlines()) == len(frames)
    assert all(int(count) > 0 for count in fig.paint_log.read_text().splitlines())
    assert ep.testing.check_frame_key(fig, draw, lambda k: None, frames=frames) == []
    with pytest.raises(ValueError, match="increasing"):
        ep.testing.check_frame_key(fig, draw, lambda k: 0, frames=[1, 0])
    with pytest.raises(TypeError, match="hashable"):
        ep.testing.check_frame_key(fig, draw, lambda k: [], frames=[0, 1])
    for invalid in ([0, 0], [-1, 0], [0, 1.5], [False, 1]):
        with pytest.raises(ValueError, match="increasing"):
            ep.testing.check_frame_key(fig, draw, lambda k: 0, frames=invalid)
    assert ep.testing.check_frame_key(fig, draw, lambda k: 0, frames=[]) == []

    def resize(k):
        fig.set_size_inches(0.2 + 0.1 * k, 0.2)

    with pytest.raises(ValueError, match="dimensions"):
        ep.testing.check_frame_key(fig, resize, lambda k: 0, frames=[0, 1])
    assert fig.canvas is original_canvas
    plt.close(fig)


def test_plan_key_equal_frames_have_identical_raster_and_vector_bytes():
    plan = held_plan()
    key = partial(plan.frame_key, include_time=False)
    fig = plt.figure(figsize=(1, 1), dpi=40)
    ax = fig.add_axes((0.2, 0.2, 0.6, 0.6))
    ax.imshow([[1, 2], [3, 4]], rasterized=True)
    ax.set_axis_off()
    for fmt, metadata in (
        ("rgba", {}),
        ("pdf", {"CreationDate": None, "ModDate": None}),
        ("svg", {"Date": None}),
    ):
        frames = []
        with matplotlib.rc_context({"svg.hashsalt": "frame-key-test"}):
            for k in (0, 2, 3, 5):
                draw_plan(fig, k, plan=plan)
                with io.BytesIO() as buffer:
                    fig.savefig(buffer, format=fmt, metadata=metadata or None)
                    frames.append(buffer.getvalue())
        assert key(0) == key(2) and frames[0] == frames[1]
        assert key(3) == key(5) and frames[2] == frames[3]
        assert key(2) != key(3) and frames[1] != frames[2]
    plt.close(fig)


def test_worker_budget_and_override_keep_encoded_bytes_and_reduce_paint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pool, "_available_cpus", lambda: 16)
    monkeypatch.setattr(pool, "_available_memory", lambda: 30 * 1024**3)
    assert pool._default_workers() == 14
    assert pool._default_workers(10 * 1024**3) == 3
    key = partial(held_plan().frame_key, include_time=False)
    outputs = []
    for name, workers, frame_key in (
        ("full", 1, None),
        ("key", 1, key),
        ("auto", None, key),
    ):
        paint_log = tmp_path / f"{name}.paint"
        path = tmp_path / f"{name}.mp4"
        result = ep.render.render_frames(
            partial(build_plan, paint_log=paint_log),
            6,
            path,
            fps=10,
            dpi=100,
            workers=workers,
            chunk_frames=2,
            frame_key=frame_key,
            worker_peak_bytes=10 * 1024**3,
        )
        assert result.workers == (3 if workers is None else workers)
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1] == outputs[2]
    assert len((tmp_path / "key.paint").read_text().splitlines()) < len(
        (tmp_path / "full.paint").read_text().splitlines()
    )
    with pytest.raises(ValueError, match="worker_peak_bytes"):
        ep.render.render_frames(
            partial(build_plan, paint_log=tmp_path / "bad"),
            6,
            tmp_path / "bad.mp4",
            fps=10,
            dpi=100,
            worker_peak_bytes=0,
        )
    spec = ep.render.ClipSpec(
        "plan",
        partial(build_spec_plan, paint_log=tmp_path / "variant.paint"),
        6,
        (ep.render.Venue("doc", dpi=100, fps=10),),
        frame_key=key,
        worker_peak_bytes=10 * 1024**3,
    )
    observed_workers = []
    original_render = variant_driver.render_frames

    def render_variant(*args, **kwargs):
        result = original_render(*args, chunk_frames=1, **kwargs)
        observed_workers.append(result.workers)
        return result

    monkeypatch.setattr(variant_driver, "render_frames", render_variant)
    selection = [ep.render.Variant("doc", "light", False)]
    manifest = ep.render.Manifest(tmp_path / "manifest.json")
    first = ep.render.render_variants(spec, selection, tmp_path, manifest=manifest)[0]
    assert observed_workers == [3]
    second = ep.render.render_variants(
        replace(spec, worker_peak_bytes=20 * 1024**3),
        selection,
        tmp_path,
        manifest=manifest,
    )[0]
    assert not second.rendered and first.key == second.key
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="worker_peak_bytes"):
            ep.render.render_variants(
                replace(spec, worker_peak_bytes=invalid),
                selection,
                tmp_path,
                manifest=manifest,
            )
