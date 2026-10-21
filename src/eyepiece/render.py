"""Render independent frames in spawned workers into one atomic mp4 output.

Builders and frame sources must be picklable and deterministic. A draw must
replace every artist property it uses; ``frame_faults`` can diagnose some
violations of that contract. Only an explicit caller key enables chunk reuse.
"""

from typing import TYPE_CHECKING

from eyepiece import _lazy_lookup

if TYPE_CHECKING:
    from eyepiece._ffmpeg import (
        ENCODER_PROFILE,
    )
    from eyepiece._render_keys import (
        content_key,
    )
    from eyepiece._render_manifest import (
        Manifest,
    )
    from eyepiece._render_pool import (
        RenderResult,
        assert_frames_pure,
        frame_faults,
        render_frames,
    )
    from eyepiece._render_poster import (
        render_poster,
        render_still,
        same_picture,
    )
    from eyepiece._render_variants import (
        ClipSpec,
        Variant,
        VariantResult,
        Venue,
        render_variants,
        variants,
    )
    from eyepiece._scrubber import (
        to_scrubber,
    )
    from eyepiece.anim import (
        PRESETS,
        Animation,
        RawSink,
        ScaleDriftError,
        animate,
        record,
    )
    from eyepiece.output import (
        save_fig,
    )
    from eyepiece.provenance import (
        file_metadata,
        provenance_fields,
        provenance_text,
        stamp,
    )

__all__ = [
    "ENCODER_PROFILE",
    "PRESETS",
    "Animation",
    "ClipSpec",
    "Manifest",
    "RawSink",
    "RenderResult",
    "ScaleDriftError",
    "Variant",
    "VariantResult",
    "Venue",
    "animate",
    "assert_frames_pure",
    "content_key",
    "file_metadata",
    "frame_faults",
    "provenance_fields",
    "provenance_text",
    "record",
    "render_frames",
    "render_poster",
    "render_still",
    "render_variants",
    "same_picture",
    "save_fig",
    "stamp",
    "to_scrubber",
    "variants",
]

_EXPORTS = {
    "Animation": ("eyepiece.anim", "Animation"),
    "ClipSpec": ("eyepiece._render_variants", "ClipSpec"),
    "ENCODER_PROFILE": ("eyepiece._ffmpeg", "ENCODER_PROFILE"),
    "Manifest": ("eyepiece._render_manifest", "Manifest"),
    "PRESETS": ("eyepiece.anim", "PRESETS"),
    "RawSink": ("eyepiece.anim", "RawSink"),
    "RenderResult": ("eyepiece._render_pool", "RenderResult"),
    "ScaleDriftError": ("eyepiece.anim", "ScaleDriftError"),
    "Variant": ("eyepiece._render_variants", "Variant"),
    "VariantResult": ("eyepiece._render_variants", "VariantResult"),
    "Venue": ("eyepiece._render_variants", "Venue"),
    "animate": ("eyepiece.anim", "animate"),
    "assert_frames_pure": ("eyepiece._render_pool", "assert_frames_pure"),
    "content_key": ("eyepiece._render_keys", "content_key"),
    "file_metadata": ("eyepiece.provenance", "file_metadata"),
    "frame_faults": ("eyepiece._render_pool", "frame_faults"),
    "provenance_fields": ("eyepiece.provenance", "provenance_fields"),
    "provenance_text": ("eyepiece.provenance", "provenance_text"),
    "record": ("eyepiece.anim", "record"),
    "render_frames": ("eyepiece._render_pool", "render_frames"),
    "render_poster": ("eyepiece._render_poster", "render_poster"),
    "render_still": ("eyepiece._render_poster", "render_still"),
    "render_variants": ("eyepiece._render_variants", "render_variants"),
    "same_picture": ("eyepiece._render_poster", "same_picture"),
    "save_fig": ("eyepiece.output", "save_fig"),
    "stamp": ("eyepiece.provenance", "stamp"),
    "to_scrubber": ("eyepiece._scrubber", "to_scrubber"),
    "variants": ("eyepiece._render_variants", "variants"),
}


def __getattr__(name):
    """Load one output helper on first access."""
    return _lazy_lookup(name, _EXPORTS, globals())


def __dir__():
    """Include lazy output exports in discovery."""
    return sorted(set(globals()) | set(__all__))


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
