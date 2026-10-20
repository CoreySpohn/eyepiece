"""Render independent frames in spawned workers into one atomic mp4 output.

Builders and frame sources must be picklable and deterministic. A draw must
replace every artist property it uses; ``frame_faults`` can diagnose some
violations of that contract. Only an explicit caller key enables chunk reuse.
"""

from eyepiece._render_keys import content_key
from eyepiece._render_manifest import Manifest
from eyepiece._render_pool import (
    RenderResult,
    assert_frames_pure,
    frame_faults,
    render_frames,
)
from eyepiece._render_poster import render_poster, render_still, same_picture
from eyepiece._render_variants import (
    ClipSpec,
    Variant,
    VariantResult,
    Venue,
    render_variants,
    variants,
)

__all__ = [
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
]


if __name__ == "__main__":
    import argparse as _argparse
    import sys as _sys
    from pathlib import Path as _Path

    from eyepiece._render_pool import _worker_main

    _parser = _argparse.ArgumentParser(prog="python -m eyepiece.render")
    _parser.add_argument("work_dir", type=_Path)
    _parser.add_argument("worker_index", type=int)
    _arguments = _parser.parse_args()
    _sys.exit(_worker_main(_arguments.work_dir, _arguments.worker_index))
