"""Key-driven clip variants and narrated outputs."""

import filecmp
import hashlib
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from numbers import Integral, Real
from pathlib import Path
from typing import NamedTuple

import matplotlib

from eyepiece._motion import frame_count
from eyepiece._render_keys import content_key
from eyepiece._render_pool import _frame_source, _validate_on_rescale, render_frames
from eyepiece._render_poster import render_poster


class Venue(NamedTuple):
    """Rendering settings for an output destination.

    The frame budget keeps silent clips short. Narrated clips are exempt
    because speech determines their length.

    Attributes:
        name: Destination name used in variant filenames.
        rc: Matplotlib overrides; None means an empty mapping.
        dpi: Raster resolution, defaulting to 100 dots per inch.
        fps: Video frame rate, defaulting to 30 frames per second.
        frame_budget: Maximum silent frame source length, or None.
    """

    name: str
    rc: dict | None = None
    dpi: int = 100
    fps: float = 30
    frame_budget: int | None = None


class Variant(NamedTuple):
    """A destination, style mode, and optional narration track.

    Attributes:
        venue: Destination name matching a Venue.
        mode: Style mode passed to hwostyle.use.
        narrated: Whether to mux the spec's audio track.
    """

    venue: str
    mode: str
    narrated: bool = False

    @property
    def name(self) -> str:
        """Return the filename and manifest identifier for this variant."""
        suffix = "-narrated" if self.narrated else ""
        return f"{self.venue}-{self.mode}{suffix}"


def variants(venues, modes=("light", "dark"), narrated=()) -> list[Variant]:
    """List silent variants in venue/mode order, then narrated variants.

    Args:
        venues: Ordered destination names.
        modes: Ordered style modes, defaulting to light and dark.
        narrated: Destination names also receiving narrated variants.

    Returns:
        The ordered list of variants.

    Raises:
        ValueError: If a narrated destination is absent from venues.
    """
    venues, modes, narrated = tuple(venues), tuple(modes), tuple(narrated)
    for venue in narrated:
        if venue not in venues:
            raise ValueError(f"Unknown narrated venue: {venue}")
    return [Variant(venue, mode) for venue in venues for mode in modes] + [
        Variant(venue, mode, True)
        for venue in venues
        if venue in narrated
        for mode in modes
    ]


@dataclass(frozen=True)
class ClipSpec:
    """Explicit inputs for key-driven rendering of a clip's variants.

    Venue budgets apply only to silent variants; speech determines narrated
    length. Each render starts from fresh default style state, then applies
    the variant mode and venue rc. Parent toggles such as ep.use_video are not
    inherited. For render_frames, pass style changes through setup and rc so
    both the in-process and spawned paths receive them explicitly.

    Attributes:
        slug: Clip identifier and output filename prefix.
        build: Picklable callable taking (venue, mode), returning (fig, draw).
        frames: Frame item sequence or integer count; may be None when
            frames_for supplies every requested variant's frames.
        venues: Tuple of destination settings.
        frame_at: Optional picklable index-to-item callable. Its behavior must
            be represented by clock or the named source inputs for keying.
        hold_s: Extra first/last frame holds in seconds.
        audio: Optional callable taking a Variant and returning an audio path.
        after: Optional callable taking (variant, out_dir), returning extra
            output paths. Runs for rendered, adopted, and kept variants in the
            real output directory; writers should be idempotent.
        script: Script text or a Path to source bytes for the key.
        modules: Module source inputs for the key.
        data: Tuple of data inputs, or a callable returning one per Variant.
        clock: Canonical clock inputs for the key.
        encoder: Optional encoder profile overrides.
        extra_ffmpeg_args: Optional additional encoding arguments.
        extra: Additional content key inputs, including stamp text if needed.
        poster: Optional frame index for each silent variant's PNG poster.
        on_rescale: "warn", "raise", or "ignore" for data-scale drift, as in
            record. Exporters that require fixed data scales set "raise".
        frames_for: Optional parent-only callable taking a Variant and
            overriding frames with that variant's sequence or integer count.
        frame_at_for: Optional parent-only callable taking a Variant and
            overriding frame_at with a picklable index-to-item callable or None.
            Its behavior must be represented by clock or named source inputs.
        adopt_build: Optional picklable builder with the same signature as
            build, reproducing videos made with an older figure or stamp.
    """

    slug: str
    build: object
    frames: object
    venues: tuple[Venue, ...]
    frame_at: object = None
    hold_s: tuple[float, float] = (0.0, 0.0)
    audio: object = None
    after: object = None
    script: str | Path | None = None
    modules: tuple = ()
    data: object = ()
    clock: object = None
    encoder: dict | None = None
    extra_ffmpeg_args: object = None
    extra: tuple = ()
    poster: int | None = None
    on_rescale: str = "warn"
    frames_for: object = None
    frame_at_for: object = None
    adopt_build: object = None


class VariantResult(NamedTuple):
    """One rendered or kept variant and its portable output identities.

    Attributes:
        variant: Destination, mode, and narration selection.
        outputs: POSIX output paths relative to out_dir.
        key: Content key computed before rendering.
        rendered: True when the video was rendered during this call.
        frames: Encoded frame count, including holds.
        adopted: True when an older render matched the existing video bytes.
    """

    variant: Variant
    outputs: list[str]
    key: str
    rendered: bool
    frames: int
    adopted: bool = False


def _variant_video(
    spec, variant, venue, out_dir, key, hold, audio, workers, *, target=None
):
    """Render once, atomically replacing a narrated target after mux succeeds."""
    import hwostyle

    if target is None:
        target = out_dir / f"{spec.slug}-{variant.name}.mp4"
    silent = out_dir / f".{target.stem}.silent.mp4" if variant.narrated else target
    muxed = out_dir / f".{target.stem}.mux.mp4"
    try:
        result = render_frames(
            partial(spec.build, venue, variant.mode),
            spec.frames,
            silent,
            fps=venue.fps,
            dpi=venue.dpi,
            workers=workers,
            frame_at=spec.frame_at,
            hold=hold,
            encoder=spec.encoder,
            extra_ffmpeg_args=spec.extra_ffmpeg_args,
            setup=partial(hwostyle.use, variant.mode),
            rc=dict(venue.rc or {}),
            work_dir=out_dir / f".chunks-{target.stem}",
            key=key,
            on_rescale=spec.on_rescale,
        )
        if variant.narrated:
            from eyepiece import narration

            narration.mux(silent, audio, muxed)
            os.replace(muxed, target)
        return target, result.frames
    finally:
        if variant.narrated:
            silent.unlink(missing_ok=True)
            muxed.unlink(missing_ok=True)


def _adopt_video(spec, variant, venue, out_dir, target, hold, audio, workers):
    """Compare an isolated old render, preserving the target and resume chunks."""
    scratch = target.with_name(f".{target.stem}.adopt-{uuid.uuid4().hex}.mp4")
    try:
        _variant_video(
            replace(spec, build=spec.adopt_build),
            variant,
            venue,
            out_dir,
            None,
            hold,
            audio,
            workers,
            target=scratch,
        )
        return filecmp.cmp(scratch, target, shallow=False)
    finally:
        scratch.unlink(missing_ok=True)
        shutil.rmtree(out_dir / f".chunks-{scratch.stem}", ignore_errors=True)


def render_variants(
    spec, variants, out_dir, *, manifest, workers=None, force=False, adopt=False
) -> list[VariantResult]:
    """Render each requested variant once, skipping current manifest entries.

    All destinations, silent budgets, poster indices, audio requirements, and
    keys are checked before any figure is built. Output paths and implicit
    stamps never enter keys. Duplicate requests are coalesced in their first
    occurrence's order. Silent variants also save the requested PNG poster.
    Completed entries are saved even if a later variant fails.
    A replaced variant's old entry is discarded before rendering, so failures
    cannot leave it marked current. Narrated variants are exempt from venue
    frame budgets because speech determines their length.
    Per-variant frame callbacks run only in the parent before keying or building.
    With adopt=True, an old builder may reproduce an existing non-current video
    exactly; matching videos keep their bytes and modification times and are
    recorded under the new key. Posters use the current builder. Extra outputs
    are generated in out_dir as for a normal render; after should be idempotent.

    Args:
        spec: Frozen ClipSpec describing explicit inputs and figure builders.
        variants: Iterable of Variant selections.
        out_dir: Folder for videos, posters, extra outputs, and per-variant chunks.
        manifest: Manifest whose root contains all outputs.
        workers: Worker count forwarded to render_frames.
        force: Render even when the manifest entry is current.
        adopt: Try spec.adopt_build for existing non-current videos unless forced.

    Returns:
        Results in first-request order with paths relative to out_dir.

    Raises:
        ValueError: For duplicate or unknown venues, invalid modes or venue
            settings, exceeded silent frame budgets, missing narrated audio,
            invalid frames, poster indices, or on_rescale, outputs outside out_dir,
            or adopt=True without adopt_build.
        TypeError: If an explicit key input is unsupported.
        RuntimeError: If rendering or audio muxing fails.
        ScaleDriftError: If a data scale drifts with spec.on_rescale="raise".
    """
    import hwostyle

    _validate_on_rescale(spec.on_rescale)
    if adopt and spec.adopt_build is None:
        raise ValueError("adopt=True requires adopt_build")
    requested = list(dict.fromkeys(variants))
    venues = {venue.name: venue for venue in spec.venues}
    if len(venues) != len(spec.venues):
        raise ValueError("Venue names must be unique")
    for venue in venues.values():
        for name in ("dpi", "fps"):
            value = getattr(venue, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"Venue {venue.name}: {name} must be positive and finite"
                )
        if venue.rc is not None and not isinstance(venue.rc, Mapping):
            raise ValueError(f"Venue {venue.name}: rc must be a mapping or None")
        try:
            matplotlib.RcParams(venue.rc or {})
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Venue {venue.name}: invalid rc settings: {exc}") from exc
    resolved = []
    for variant in requested:
        if (
            not isinstance(variant.mode, str)
            or not variant.mode
            or variant.mode not in hwostyle.core.MODE_RC
        ):
            raise ValueError(f"Invalid style mode: {variant.mode}")
        if variant.venue not in venues:
            raise ValueError(f"Unknown venue: {variant.venue}")
        variant_spec = replace(
            spec,
            frames=spec.frames_for(variant)
            if spec.frames_for is not None
            else spec.frames,
            frame_at=(
                spec.frame_at_for(variant)
                if spec.frame_at_for is not None
                else spec.frame_at
            ),
        )
        if variant_spec.frames is None:
            raise ValueError(f"Variant {variant.name} requires frames or frames_for")
        count = _frame_source(variant_spec.frames, variant_spec.frame_at)
        if (
            not variant.narrated
            and spec.poster is not None
            and (not isinstance(spec.poster, Integral) or not 0 <= spec.poster < count)
        ):
            raise ValueError(
                f"Variant {variant.name}: poster must be a frame index in [0, {count})"
            )
        venue = venues[variant.venue]
        if (
            not variant.narrated
            and venue.frame_budget is not None
            and count > venue.frame_budget
        ):
            raise ValueError(
                f"{spec.slug} venue {venue.name}: {count} frames exceeds "
                f"budget {venue.frame_budget}"
            )
        if variant.narrated and spec.audio is None:
            raise ValueError(f"Narrated variant {variant.name} requires audio")
        resolved.append((variant, variant_spec, count))
    jobs = []
    for variant, variant_spec, count in resolved:
        venue = venues[variant.venue]
        hold = tuple(frame_count(seconds, venue.fps) for seconds in spec.hold_s)
        audio = Path(spec.audio(variant)) if variant.narrated else None
        extra = spec.extra
        if audio is not None:
            with audio.open("rb") as stream:
                extra = (*extra, hashlib.file_digest(stream, "sha256").hexdigest())
        key = content_key(
            script=spec.script,
            clock=spec.clock,
            modules=spec.modules,
            data=spec.data(variant) if callable(spec.data) else spec.data,
            settings={
                "variant": variant._asdict(),
                "fps": venue.fps,
                "dpi": venue.dpi,
                "rc": dict(venue.rc or {}),
                "hold": hold,
                "frames": variant_spec.frames,
                "frame_at": variant_spec.frame_at is not None,
                "encoder": spec.encoder,
                "extra_ffmpeg_args": spec.extra_ffmpeg_args,
                **({"poster": spec.poster} if spec.poster is not None else {}),
            },
            extra=extra,
        )
        jobs.append((variant, variant_spec, count, venue, hold, audio, key))
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for variant, variant_spec, count, venue, hold, audio, key in jobs:
            entry = manifest.entry(spec.slug, variant.name)
            target = out_dir / f"{spec.slug}-{variant.name}.mp4"
            expected = [target]
            poster = out_dir / f"{spec.slug}-{variant.name}-poster.png"
            if spec.poster is not None and not variant.narrated:
                expected.append(poster)
            recorded = (
                []
                if entry is None
                else [manifest.root / output["path"] for output in entry["outputs"]]
            )
            current = manifest.current(spec.slug, variant.name, key)
            rendered = (
                force
                or recorded[: len(expected)] != expected
                or any(not path.is_relative_to(out_dir) for path in recorded)
                or not current
            )
            adopted = False
            if rendered:
                manifest.discard(spec.slug, variant.name)
                if adopt and not force and not current and target.is_file():
                    adopted = _adopt_video(
                        variant_spec,
                        variant,
                        venue,
                        out_dir,
                        target,
                        hold,
                        audio,
                        workers,
                    )
                if adopted:
                    rendered = False
                    frames = count + sum(hold)
                else:
                    target, frames = _variant_video(
                        variant_spec, variant, venue, out_dir, key, hold, audio, workers
                    )
                outputs = [target]
                if spec.poster is not None and not variant.narrated:
                    render_poster(
                        partial(spec.build, venue, variant.mode),
                        variant_spec.frames,
                        spec.poster,
                        poster,
                        dpi=venue.dpi,
                        frame_at=variant_spec.frame_at,
                        setup=partial(hwostyle.use, variant.mode),
                        rc=dict(venue.rc or {}),
                    )
                    outputs.append(poster)
            else:
                frames = entry["frames"]
                outputs = [
                    manifest.root / output["path"] for output in entry["outputs"]
                ]
            if spec.after is not None:
                outputs.extend(
                    path if path.is_absolute() else out_dir / path
                    for path in map(Path, spec.after(variant, out_dir))
                )
            outputs = list(dict.fromkeys(path.resolve() for path in outputs))
            relative = [path.relative_to(out_dir).as_posix() for path in outputs]
            manifest.record(
                spec.slug,
                variant.name,
                key=key,
                outputs=outputs,
                frames=frames,
                fps=venue.fps,
                dpi=venue.dpi,
            )
            results.append(
                VariantResult(variant, relative, key, rendered, frames, adopted)
            )
    finally:
        manifest.save()
    return results


# Keep public pickle references stable through the facade.
Venue.__module__ = "eyepiece.render"
Variant.__module__ = "eyepiece.render"
variants.__module__ = "eyepiece.render"
ClipSpec.__module__ = "eyepiece.render"
VariantResult.__module__ = "eyepiece.render"
render_variants.__module__ = "eyepiece.render"
