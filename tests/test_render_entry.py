"""Public facade and real subprocess worker entry contracts."""

import os
import pickle
import select
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest
from test_render_helpers import build_tiny_clip

import eyepiece as ep
import eyepiece.render as render
from eyepiece import _render_chunks as chunks
from eyepiece import _render_pool as pool_driver

PUBLIC_NAMES = {
    "ClipSpec",
    "Manifest",
    "RenderResult",
    "Variant",
    "VariantResult",
    "Venue",
    "assert_frames_pure",
    "content_key",
    "frame_faults",
    "render_frames",
    "render_poster",
    "render_still",
    "render_variants",
    "same_picture",
    "variants",
}


def _run_python(code, *arguments, timeout=30, pass_fds=()):
    process = subprocess.Popen(
        [sys.executable, "-c", code, *map(str, arguments)],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=pass_fds,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        assert process.returncode == 0, stdout + stderr
    finally:
        # Cover readiness waits as well as rendering, even when __enter__ hangs.
        # Workers inherit the scenario's process group, so none can escape it.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)


def test_scenario_timeout_stops_descendants(tmp_path):
    reader, writer = os.pipe()
    started = tmp_path / "started"
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_python(
                """
import os
import subprocess
import sys
from pathlib import Path

fd = int(sys.argv[1])
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"], pass_fds=(fd,)
)
os.close(fd)
Path(sys.argv[2]).write_text(str(child.pid))
child.wait()
""",
                writer,
                started,
                timeout=2,
                pass_fds=(writer,),
            )
        assert started.exists(), "scenario did not launch its child"
        os.close(writer)
        writer = None
        assert select.select([reader], [], [], 5)[0], "descendant survived timeout"
        assert os.read(reader, 1) == b""
    finally:
        os.close(reader)
        if writer is not None:
            os.close(writer)


def build_facade_clip():
    """Build in a worker whose test-module import resolves the public facade."""
    assert sys.modules["__main__"].__spec__.name == "eyepiece.render"
    fig, _ = build_tiny_clip()
    return fig, draw_facade_clip


def draw_facade_clip(fig, frame):
    assert type(frame) is render.Variant
    assert frame.name == f"doc-{frame.mode}"
    fig.set_facecolor("white" if frame.mode == "light" else "black")


def _exercise_worker(folder):
    # Use the production job writer and launcher, including its lifetime pipe.
    # The builder is pickled by reference to an importable test module.
    figure = dict(
        build=build_facade_clip,
        frames=[render.Variant("doc", mode) for mode in ("light", "dark") * 2],
        dpi=100,
        frame_at=None,
        setup=None,
        rc=None,
    )
    with chunks._workspace(folder), pool_driver._Workers(folder, 1, figure, 2) as pool:
        jobs = [
            (
                folder / f"chunk-{index}.bin",
                chunks._identity("entry", 4, (0, 0), 100, pool.size, 12, start, stop),
                start,
                stop,
            )
            for index, (start, stop) in enumerate(((0, 2), (2, 4)))
        ]
        pool.start_jobs(jobs, set())
        (process,) = pool.processes
        assert process.args == [
            sys.executable,
            "-m",
            "eyepiece.render",
            str(folder),
            "0",
        ]
        assert process.wait(timeout=30) == 0, (folder / "worker-0.log").read_text()
        pool.check()
        for job in jobs:
            pixels = list(chunks._read_chunk(job, pool.size))
            assert len(pixels) == 2
            assert all(frame.shape == (2, 2, 4) for frame in pixels)
            assert (pixels[0][..., :3] == 255).all()
            assert (pixels[1][..., :3] == 0).all()
        assert not list(folder.glob("*.part"))


def test_worker_module_publishes_chunks(tmp_path):
    _run_python(
        """
import sys
from pathlib import Path
from test_render_entry import _exercise_worker

_exercise_worker(Path(sys.argv[1]))
""",
        tmp_path / "chunks",
    )


@pytest.mark.parametrize(
    "arguments", [[], ["chunks"], ["chunks", "bad"], ["a", "0", "b"]]
)
def test_worker_module_rejects_bad_arguments_with_usage(arguments):
    result = subprocess.run(
        [sys.executable, "-m", "eyepiece.render", *arguments],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "usage:" in result.stderr.lower()


def test_facade_exports_exactly_the_public_names():
    assert set(render.__all__) == PUBLIC_NAMES
    assert len(render.__all__) == len(PUBLIC_NAMES)
    assert {name for name in vars(render) if not name.startswith("_")} == PUBLIC_NAMES
    for name in render.__all__:
        assert getattr(render, name) is getattr(ep, name)


@pytest.mark.parametrize("name", sorted(PUBLIC_NAMES))
def test_public_references_remain_picklable_through_facade(name, tmp_path):
    value = getattr(render, name)
    assert value.__module__ == "eyepiece.render"
    payload = tmp_path / "reference.pkl"
    payload.write_bytes(pickle.dumps(value))
    _run_python(
        """
import pickle
import sys
from pathlib import Path

assert 'eyepiece.render' not in sys.modules
value = pickle.loads(Path(sys.argv[1]).read_bytes())
facade = sys.modules['eyepiece.render']
assert value is getattr(facade, sys.argv[2])
assert value.__module__ == 'eyepiece.render'
""",
        payload,
        name,
    )


def test_public_instances_unpickle_in_fresh_interpreter(tmp_path):
    variant = render.Variant("doc", "dark", True)
    venue = render.Venue("doc", {"font.size": 7}, dpi=80, fps=12)
    instances = {
        "ClipSpec": render.ClipSpec("tiny", build_tiny_clip, [0, 1], (venue,)),
        "Manifest": render.Manifest(tmp_path / "manifest.json"),
        "RenderResult": render.RenderResult(Path("clip.mp4"), 2, 1, 0, 1, 0.5),
        "Variant": variant,
        "VariantResult": render.VariantResult(variant, ["clip.mp4"], "key", False, 2),
        "Venue": venue,
    }
    payload = tmp_path / "instances.pkl"
    payload.write_bytes(pickle.dumps(instances))
    _run_python(
        """
import pickle
import sys
from pathlib import Path

assert 'eyepiece.render' not in sys.modules
instances = pickle.loads(Path(sys.argv[1]).read_bytes())
facade = sys.modules['eyepiece.render']
assert set(instances) == {
    name for name in facade.__all__ if isinstance(getattr(facade, name), type)
}
for name, value in instances.items():
    assert type(value) is getattr(facade, name)
spec = instances['ClipSpec']
from test_render_helpers import build_tiny_clip
assert spec.build is build_tiny_clip
assert (spec.slug, spec.frames, spec.venues) == ('tiny', [0, 1], (instances['Venue'],))
manifest = instances['Manifest']
assert manifest.path == Path(sys.argv[2])
assert manifest.root == manifest.path.parent.resolve()
assert manifest.entry('tiny', 'doc-dark') is None
assert instances['RenderResult'] == (Path('clip.mp4'), 2, 1, 0, 1, 0.5)
assert instances['Variant'].name == 'doc-dark-narrated'
assert instances['VariantResult'] == (
    instances['Variant'], ['clip.mp4'], 'key', False, 2, False
)
assert instances['Venue'] == ('doc', {'font.size': 7}, 80, 12, None)
""",
        payload,
        tmp_path / "manifest.json",
    )
