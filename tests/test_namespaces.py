"""Task facades preserve object identity and lazy import boundaries."""

import importlib
import subprocess
import sys
import warnings

import pytest

import eyepiece as ep

NAMESPACES = ("plot", "clock", "motion", "style", "render", "testing", "narration")


@pytest.mark.parametrize("namespace", NAMESPACES)
def test_every_namespace_export_keeps_its_original_identity(namespace):
    module = importlib.import_module(f"eyepiece.{namespace}")
    assert module.__all__
    assert len(module.__all__) == len(set(module.__all__))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for name in module.__all__:
            assert name in dir(module)
            value = getattr(module, name)
            if name in OLD_ROOT:
                assert value is getattr(ep, name)
            elif namespace == "narration" and name == "caption_line":
                assert value is ep.testing.caption_line
            else:
                # Existing namespace-only implementations keep their owners.
                owner = ORIGINAL_OWNERS[namespace][name]
                assert value is getattr(importlib.import_module(owner), name)
    assert not [w for w in caught if w.category is DeprecationWarning]


@pytest.mark.parametrize("namespace", NAMESPACES)
def test_namespace_rejects_unknown_names(namespace):
    module = importlib.import_module(f"eyepiece.{namespace}")
    with pytest.raises(AttributeError, match="not_an_export"):
        _ = module.not_an_export


@pytest.mark.parametrize(
    "namespace", ["plot", "clock", "motion", "render", "narration"]
)
def test_fresh_namespace_import_does_not_load_matplotlib(namespace):
    code = f"""
import importlib
import sys
module = importlib.import_module('eyepiece.{namespace}')
assert module.__all__
assert set(module.__all__) <= set(dir(module))
heavy = ('matplotlib', 'kokoro', 'torch', 'manim', 'hwoutils')
assert not [name for name in sys.modules if name.split('.')[0] in heavy]
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_clock_helpers_remain_numerical_without_matplotlib():
    code = """
import sys
from eyepiece import clock
clock.Timeline().hold(0.5)
assert clock.ease(0.0) == 0.0
assert clock.ease(1.0) == 1.0
assert clock.stagger(0.0, 0) == 0.0
assert clock.stagger(1.0, 0) == 1.0
assert clock.pulse_shape(0.0) == 0.0
assert clock.pulse_shape(1.0) == 0.0
assert not any(name.startswith('matplotlib') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_task_assignments_cover_every_former_root_name_once():
    assigned = ["__version__"]
    for namespace in (*NAMESPACES, "motifs"):
        module = importlib.import_module(f"eyepiece.{namespace}")
        for name in module.__all__:
            if name in OLD_ROOT:
                # These old render check paths remain compatible aliases.
                if namespace == "render" and name in {
                    "frame_faults",
                    "assert_frames_pure",
                    "same_picture",
                }:
                    continue
                assigned.append(name)
    assert sorted(assigned) == sorted(OLD_ROOT)


# Frozen 0.16 root surface: compatibility must not follow the new export map.
OLD_ROOT = (
    "ARTIST_KEYS",
    "ENCODER_PROFILE",
    "GLYPHS",
    "PRESETS",
    "TALK_FRACTIONS",
    "VIDEO_RC",
    "Anchor",
    "Animation",
    "ApertureCounts",
    "Arrivals",
    "AttentionFault",
    "ClipSpec",
    "Event",
    "Fault",
    "Frame",
    "Manifest",
    "MosaicResult",
    "Motif",
    "MotifSet",
    "MotionContract",
    "PageCamera",
    "Plan",
    "PlotResult",
    "RawSink",
    "RenderResult",
    "ScaleDriftError",
    "SourceStyles",
    "Span",
    "Stage",
    "Tempo",
    "Time",
    "Timeline",
    "Variant",
    "VariantResult",
    "Venue",
    "__version__",
    "angle_arc",
    "animate",
    "aperture_counts",
    "arrivals",
    "arrow",
    "assert_frames_pure",
    "audit",
    "backing",
    "blend",
    "bracket",
    "brush",
    "build_axes",
    "calibration",
    "callout",
    "capture",
    "carried_tag",
    "circumscribe",
    "click",
    "collapse",
    "compare_grid",
    "compare_row",
    "compass",
    "content_key",
    "convergence",
    "corner",
    "corner_overlay",
    "cov_ellipse",
    "curve_insets",
    "display_limits",
    "draw_on",
    "ease",
    "equation",
    "exposure_text",
    "extent_arcsec",
    "extent_au",
    "extent_lod",
    "extent_lod_from_pixels",
    "fade",
    "fading_track",
    "file_metadata",
    "flight",
    "follow_zoom",
    "frame_count",
    "frame_faults",
    "head_draw",
    "hist_fill",
    "hist_vs_pdf",
    "hop",
    "imshow_diverging",
    "imshow_log",
    "indicate",
    "interval_points",
    "kymograph",
    "label_arcsec",
    "label_au",
    "label_lod",
    "leader",
    "link",
    "log_zero_floor",
    "morph",
    "neutral",
    "overlay_circle",
    "overlay_line",
    "overview_box",
    "phase_ring",
    "phasor",
    "photons",
    "pixel_quads",
    "plot_contrast_curve",
    "plot_radial",
    "plot_radial_image",
    "provenance_fields",
    "provenance_text",
    "pulse_shape",
    "quad_image",
    "quiet_spines",
    "rail",
    "rail_panels",
    "reading_words",
    "readout",
    "record",
    "ref_line",
    "render_frames",
    "render_poster",
    "render_still",
    "render_variants",
    "role",
    "ruler",
    "ruler_readout",
    "same_picture",
    "save_fig",
    "scatter_values",
    "schematic",
    "shade_intervals",
    "show_field",
    "signed_trace",
    "sky_fan",
    "spotlight",
    "spread_labels",
    "stagger",
    "stamp",
    "step_list",
    "strip",
    "talk_rc",
    "term_row",
    "to_scrubber",
    "trail",
    "transform",
    "triptych",
    "underline",
    "unsnapped",
    "use_video",
    "variants",
    "video_roles",
    "view_limits",
    "warp",
    "warp_grid",
    "weighted_draws",
    "words",
    "write",
    "zoom_path",
)

# Original implementation owners for names that were already namespace-only.
ORIGINAL_OWNERS = {
    "narration": {
        "Cue": "eyepiece.narration._script",
        "CueEvents": "eyepiece.narration._cues",
        "CueTime": "eyepiece.narration._timeline",
        "KOKORO_MODEL": "eyepiece.narration._script",
        "KOKORO_VERSION": "eyepiece.narration._script",
        "Line": "eyepiece.narration._script",
        "MissingAudioError": "eyepiece.narration._timeline",
        "MissingSynthesisError": "eyepiece.narration._timeline",
        "Narration": "eyepiece.narration._timeline",
        "NarrationBeat": "eyepiece.narration._timeline",
        "NumberSyncFault": "eyepiece.narration._speech",
        "Pause": "eyepiece.narration._script",
        "Script": "eyepiece.narration._script",
        "ScriptFault": "eyepiece.narration._speech",
        "SilenceFault": "eyepiece.narration._speech",
        "SpeechFault": "eyepiece.narration._speech",
        "SpeechTextError": "eyepiece.narration._speech",
        "Spoken": "eyepiece.narration._timeline",
        "Word": "eyepiece.narration._timeline",
        "audio_track": "eyepiece.narration._mux",
        "caption_at": "eyepiece.narration._mux",
        "captions": "eyepiece.narration._mux",
        "check_number_sync": "eyepiece.narration._speech",
        "check_script_speech": "eyepiece.narration._speech",
        "check_silence": "eyepiece.narration._speech",
        "click_sting": "eyepiece.narration._cues",
        "count_tone": "eyepiece.narration._cues",
        "cue_track": "eyepiece.narration._cues",
        "estimate": "eyepiece.narration._timeline",
        "line_hash": "eyepiece.narration._script",
        "lint_script": "eyepiece.narration._speech",
        "mux": "eyepiece.narration._mux",
        "parse_cues": "eyepiece.narration._script",
        "pixel_ticks": "eyepiece.narration._cues",
        "say": "eyepiece.narration._speech",
        "scientific": "eyepiece.narration._speech",
        "sentences": "eyepiece.narration._mux",
        "speech_faults": "eyepiece.narration._speech",
        "write_captions": "eyepiece.narration._mux",
        "write_chapters": "eyepiece.narration._mux",
        "write_srt": "eyepiece.narration._mux",
        "write_vtt": "eyepiece.narration._mux",
    },
    "render": {
        "ClipSpec": "eyepiece._render_variants",
        "Manifest": "eyepiece._render_manifest",
        "RenderResult": "eyepiece._render_pool",
        "Variant": "eyepiece._render_variants",
        "VariantResult": "eyepiece._render_variants",
        "Venue": "eyepiece._render_variants",
        "assert_frames_pure": "eyepiece._render_pool",
        "content_key": "eyepiece._render_keys",
        "frame_faults": "eyepiece._render_pool",
        "render_frames": "eyepiece._render_pool",
        "render_poster": "eyepiece._render_poster",
        "render_still": "eyepiece._render_poster",
        "render_variants": "eyepiece._render_variants",
        "same_picture": "eyepiece._render_poster",
        "variants": "eyepiece._render_variants",
    },
    "style": {
        "RenderProfile": "eyepiece.style",
        "SourceCast": "eyepiece.style",
        "backing": "eyepiece.style",
        "neutral": "eyepiece.style",
        "role": "eyepiece.style",
        "snapshot_profile": "eyepiece.style",
    },
    "testing": {
        "ClipError": "eyepiece.testing",
        "ClipReport": "eyepiece.testing",
        "GROUNDING_KINDS": "eyepiece.testing",
        "GroundingError": "eyepiece.testing",
        "GroundingFault": "eyepiece.testing",
        "HaloError": "eyepiece.testing",
        "HaloFault": "eyepiece.testing",
        "KINDS": "eyepiece.testing",
        "MEANING_KINDS": "eyepiece._meaning",
        "MOTIF_KINDS": "eyepiece._motif_checks",
        "MOTION_KINDS": "eyepiece.testing",
        "MeaningError": "eyepiece._meaning",
        "MeaningFault": "eyepiece._meaning",
        "MotifError": "eyepiece._motif_checks",
        "MotifFault": "eyepiece._motif_checks",
        "MotionError": "eyepiece.testing",
        "MotionFault": "eyepiece.testing",
        "ReadabilityError": "eyepiece.testing",
        "ReadabilityFault": "eyepiece.testing",
        "SpeechRate": "eyepiece.testing",
        "SpeechRateError": "eyepiece.testing",
        "TIMING_KINDS": "eyepiece.testing",
        "TimingError": "eyepiece.testing",
        "TimingFault": "eyepiece.testing",
        "VENUES": "eyepiece.testing",
        "VENUE_FPS": "eyepiece.testing",
        "Window": "eyepiece.testing",
        "assert_clip": "eyepiece.testing",
        "assert_frames_pure": "eyepiece.testing",
        "assert_grounded": "eyepiece.testing",
        "assert_meaning": "eyepiece._meaning",
        "assert_motifs": "eyepiece._motif_checks",
        "assert_motion": "eyepiece.testing",
        "assert_no_halos": "eyepiece.testing",
        "assert_readable": "eyepiece.testing",
        "assert_speech_rate": "eyepiece.testing",
        "assert_timing": "eyepiece.testing",
        "caption_line": "eyepiece.testing",
        "check_clip": "eyepiece.testing",
        "frame_faults": "eyepiece.testing",
        "grounding_faults": "eyepiece.testing",
        "halo_faults": "eyepiece.testing",
        "meaning_faults": "eyepiece._meaning",
        "meaning_faults_chain": "eyepiece._meaning",
        "meaning_report": "eyepiece._meaning",
        "motif_faults": "eyepiece._motif_checks",
        "motif_margins": "eyepiece._motif_checks",
        "motion_faults": "eyepiece.testing",
        "motion_ink": "eyepiece.testing",
        "readability_faults": "eyepiece.testing",
        "review_sheet": "eyepiece.testing",
        "same_picture": "eyepiece.testing",
        "speech_rate": "eyepiece.testing",
        "timing_faults": "eyepiece.testing",
    },
}


SHORTCUTS = {
    "Timeline",
    "Plan",
    "Tempo",
    "PlotResult",
    "MosaicResult",
    "MotionContract",
    "animate",
    "record",
    "save_fig",
    "__version__",
}
TASK_MODULES = {
    "plot",
    "clock",
    "motion",
    "style",
    "render",
    "testing",
    "narration",
    "motifs",
    "prepared",
    "mpl",
    "manim",
}


def test_root_discovery_stays_small_after_legacy_access():
    assert set(ep.__all__) == SHORTCUTS
    assert len(ep.__all__) == len(SHORTCUTS)
    assert set(dir(ep)) == SHORTCUTS | TASK_MODULES
    for name in OLD_ROOT:
        getattr(ep, name)
    assert set(dir(ep)) == SHORTCUTS | TASK_MODULES


def test_star_import_exposes_only_stable_shortcuts():
    namespace = {}
    exec("from eyepiece import *", namespace)
    assert set(namespace) - {"__builtins__"} == SHORTCUTS


@pytest.mark.parametrize("name", OLD_ROOT)
def test_every_former_root_name_still_resolves_silently(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        value = getattr(ep, name)
        imported = {}
        exec(f"from eyepiece import {name}", imported)
        assert imported[name] is value
    assert not [w for w in caught if w.category is DeprecationWarning]


def test_fresh_root_discovery_loads_no_renderers():
    code = f"""
import sys
import eyepiece as ep
assert set(ep.__all__) == {SHORTCUTS!r}
assert set(dir(ep)) == {SHORTCUTS | TASK_MODULES!r}
assert 'matplotlib' not in sys.modules
assert 'eyepiece.plot' not in sys.modules
assert 'eyepiece.motion' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_version_has_only_its_root_home():
    assert isinstance(ep.__version__, str)
    assert "__version__" not in ep.render.__all__
    assert "__version__" not in dir(ep.render)
    with pytest.raises(AttributeError, match="__version__"):
        _ = ep.render.__version__
