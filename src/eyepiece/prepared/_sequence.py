"""Replayable sequence states and one pure physical-time evaluator.

A `Sequence` pairs a fixed-topology template view with physical timestamps
and a set of `channels` that each declare how one element of that template
varies from frame to frame: `ArrayChannel` for a leading-N array (an
`ImageView`'s `data`/`valid`, or a `Points` mark's `xy`), `PathWindow` for a
static `Path`'s revealed sub-range, and `Clock` for a `Label` that reports
acquisition time. Static geometry (full paths, unreferenced marks, axes,
scales) is stored once on the template and shared, unchanged, by every
frame `replace_elements` builds -- only the branch a channel actually
touches is rebuilt.

Seeking is a pure function of physical time, not accumulated state: left
sample-and-hold selects the sample acquired at or before the requested
time (the final sample holds through and including the final timestamp),
and evaluating the same physical time twice, in any order, returns the
same sample. This is deliberate -- "seeking backwards reconstructs the
same visible trail without copying every prefix" -- so a caller can scrub
forward, backward, or straight to an arbitrary time without the sequence
needing to remember where it has already been.

This module imports only the standard library and NumPy.
"""

import dataclasses
import math
from numbers import Integral

import numpy as np

from eyepiece.prepared._views import (
    ImageView,
    Label,
    PanelGroup,
    Path,
    Points,
    View,
    _combine_valid,
    _mask_to_valid,
    find_element,
    replace_elements,
)

_ARRAY_CHANNEL_FIELDS = ("data", "valid", "xy")

# Panel-space placement (figure-fraction, top-left) for the label `strip`
# adds to every slot so each strip panel is labelled even when the
# template carries no Clock-driven label of its own.
_STRIP_LABEL_XY = (0.02, 0.95)

# Relative tolerance for treating `run_time * fps` as a whole frame count.
_FRAME_COUNT_RTOL = 1e-9


def _clock_text(acquisition_time, time_unit):
    """Format an acquisition time as a compact "<time> <unit>" clock string.

    This describes the ACQUISITION time of the sampled frame -- the same
    value a `Sample.acquisition_time` reports -- never a continuously
    interpolated playhead value.
    """
    return f"{float(acquisition_time):g} {time_unit}"


@dataclasses.dataclass(frozen=True, eq=False)
class ArrayChannel:
    """One time-varying leaf array, driving one field of one element.

    Attributes:
        element_id: The view element this channel updates each frame.
        field: Which field varies: "data" or "valid" for an `ImageView`,
            "xy" for a `Points` mark.
        values: Array shaped `(N, ...)`, one leading slot per sequence
            sample. Borrowed and indexed by frame (`values[index]` is a
            view, never a copy). Given a `numpy.ma.MaskedArray` (only
            meaningful for `field="data"`), its mask is captured once here
            -- see `mask_valid` -- exactly as a single `ImageView.data`
            does; a plain array carries no implied mask.
        mask_valid: Boolean array shaped like `values`, True where a
            sample is valid, or `None` when `values` carried no mask. Not
            a constructor argument -- derived once from `values`, so no
            whole-cube mask is ever allocated when there is none.
    """

    element_id: str
    field: str
    values: np.ndarray
    mask_valid: np.ndarray | None = dataclasses.field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if self.field not in _ARRAY_CHANNEL_FIELDS:
            raise ValueError(
                f"{self.element_id}: field must be one of {_ARRAY_CHANNEL_FIELDS}, "
                f"got {self.field!r}"
            )
        data, mask_valid = _mask_to_valid(self.values)
        object.__setattr__(self, "values", data)
        object.__setattr__(self, "mask_valid", mask_valid)

    def __len__(self):
        return len(self.values)


@dataclasses.dataclass(frozen=True, eq=False)
class PathWindow:
    """Declares that `Path` `element_id`'s visible range tracks the frame index.

    The full path is stored once on the template; a `PathWindow` never
    carries geometry, only the reveal policy. `history=None` (the default)
    reveals an ever-growing prefix from the start: `[0, i+1)` at frame
    `i`. A positive `history` instead reveals a sliding trailing window of
    at most `history` vertices: `[max(0, i+1-history), i+1)`.

    Attributes:
        element_id: The `Path` this window controls.
        history: `None` for a growing prefix, or a positive integer
            vertex count for a trailing window.
    """

    element_id: str
    history: int | None = None

    def __post_init__(self):
        if self.history is not None:
            if not isinstance(self.history, Integral) or self.history <= 0:
                raise ValueError(
                    f"{self.element_id}: history must be a positive integer or "
                    f"None, got {self.history!r}"
                )


@dataclasses.dataclass(frozen=True, eq=False)
class Clock:
    """Declares that `Label` `element_id` reports this sequence's acquisition time.

    The label text is `f"{t:g} {time_unit}"`, where `t` is the exact
    ACQUISITION time of the sampled frame -- the same value
    `Sample.acquisition_time` reports for that frame. It never shows a
    continuously interpolated measurement.

    Attributes:
        element_id: The `Label` this clock drives.
    """

    element_id: str


@dataclasses.dataclass(frozen=True, eq=False)
class Sample:
    """One evaluated physical-time sample.

    Attributes:
        index: The held sample's index into the sequence's `times`.
        physical_time: The raw time that was requested (unclamped, even
            when outside the sampled range).
        acquisition_time: `times[index]` -- the actual timestamp of the
            held sample.
        view: The complete rendered state at that sample.
    """

    index: int
    physical_time: float
    acquisition_time: float
    view: View


def _check_channel_target(template, element_id, expected_type, channel_label):
    target = find_element(template, element_id)
    if not isinstance(target, expected_type):
        raise ValueError(
            f"{element_id}: {channel_label} requires {expected_type.__name__}, "
            f"got {type(target).__name__}"
        )
    return target


@dataclasses.dataclass(frozen=True, eq=False)
class Sequence:
    """A fixed-topology, replayable sequence over physical time.

    Preparing a `Sequence` computes nothing scientific; it only validates
    the declared channels against `template` and `times` once so that
    `.frame`, `.at`, `.strip`, and `.schedule` can trust their inputs
    without simulation callbacks entering the sequence at all.

    Attributes:
        template: The root `View`, fully built at some reference frame
            (frame 0's values, by convention -- see the module-level
            `make_image_sequence`-style helpers used in tests). Every
            other frame replaces only the elements named by `channels`.
        times: Physical timestamps, one per sample. Must be finite and
            strictly increasing (never sorted or de-duplicated for the
            caller); at least one sample is required.
        time_unit: Unit string used verbatim in `Clock` label text.
        channels: `ArrayChannel`, `PathWindow`, and `Clock` instances
            declaring how elements of `template` vary across `times`.
        epoch: Optional finite reference time, carried as metadata; it
            does not otherwise affect sampling.
        sample_kind: Must be `"instantaneous"`; any other value raises
            (exposure-integrated and other semantics are a later
            contract extension).
    """

    template: View
    times: np.ndarray
    time_unit: str
    channels: tuple = ()
    epoch: float | None = None
    sample_kind: str = "instantaneous"

    def __post_init__(self):
        if self.sample_kind != "instantaneous":
            raise ValueError(f"{self.sample_kind} sample_kind is not supported")

        times = np.asarray(self.times, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError(
                f"times must be a nonempty 1D array, got shape {times.shape}"
            )
        if not np.all(np.isfinite(times)):
            raise ValueError("times must be finite")
        if times.size > 1 and not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing")
        object.__setattr__(self, "times", times)

        if self.epoch is not None:
            epoch = float(self.epoch)
            if not math.isfinite(epoch):
                raise ValueError(f"epoch must be finite, got {self.epoch!r}")
            object.__setattr__(self, "epoch", epoch)

        channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)

        array_by_element = {}
        path_windows = {}
        clocks = []
        for channel in channels:
            if isinstance(channel, ArrayChannel):
                if len(channel.values) != times.size:
                    raise ValueError(
                        f"{channel.element_id}: channel length "
                        f"{len(channel.values)} does not match sequence length "
                        f"{times.size}"
                    )
                if channel.field in ("data", "valid"):
                    _check_channel_target(
                        self.template, channel.element_id, ImageView, "data/valid"
                    )
                else:
                    _check_channel_target(
                        self.template, channel.element_id, Points, "xy"
                    )
                array_by_element.setdefault(channel.element_id, {})[channel.field] = (
                    channel
                )
            elif isinstance(channel, PathWindow):
                target = _check_channel_target(
                    self.template, channel.element_id, Path, "PathWindow"
                )
                if target.xy.shape[0] != times.size:
                    raise ValueError(
                        f"{channel.element_id}: path has {target.xy.shape[0]} "
                        f"vertices, expected {times.size} to match the "
                        "sequence length"
                    )
                path_windows[channel.element_id] = channel.history
            elif isinstance(channel, Clock):
                _check_channel_target(self.template, channel.element_id, Label, "Clock")
                clocks.append(channel.element_id)
            else:
                raise TypeError(f"unsupported channel type {type(channel).__name__}")

        object.__setattr__(self, "_array_by_element", array_by_element)
        object.__setattr__(self, "_path_windows", path_windows)
        object.__setattr__(self, "_clocks", tuple(clocks))

    def frame(self, index):
        """Return the complete `View` at sample `index`.

        Args:
            index: An integer sample index.

        Returns:
            A `View` sharing every array and unchanged branch with
            `template`; only elements named by `channels` are rebuilt.

        Raises:
            TypeError: If `index` is not an integer.
            IndexError: If `index` is outside `[0, len(times))`.
        """
        if not isinstance(index, Integral):
            raise TypeError(
                f"frame index must be an integer, got {type(index).__name__}"
            )
        n = self.times.size
        if index < 0 or index >= n:
            raise IndexError(
                f"frame index {index} out of range for sequence of length {n}"
            )
        return self._build_frame(int(index))

    def _build_frame(self, index):
        changes = {}
        for element_id, fields in self._array_by_element.items():
            current = find_element(self.template, element_id)
            if isinstance(current, ImageView):
                data_channel = fields.get("data")
                data = (
                    data_channel.values[index]
                    if data_channel is not None
                    else current.data
                )
                mask_valid = (
                    data_channel.mask_valid[index]
                    if data_channel is not None and data_channel.mask_valid is not None
                    else None
                )
                valid_channel = fields.get("valid")
                explicit_valid = (
                    valid_channel.values[index] if valid_channel is not None else None
                )
                valid = _combine_valid(mask_valid, explicit_valid)
                changes[element_id] = dataclasses.replace(
                    current, data=data, valid=valid
                )
            else:
                xy_channel = fields["xy"]
                changes[element_id] = dataclasses.replace(
                    current, xy=xy_channel.values[index]
                )

        for element_id, history in self._path_windows.items():
            current = find_element(self.template, element_id)
            start = max(0, index + 1 - history) if history is not None else 0
            stop = index + 1
            changes[element_id] = dataclasses.replace(current, visible=(start, stop))

        for element_id in self._clocks:
            current = find_element(self.template, element_id)
            text = _clock_text(self.times[index], self.time_unit)
            changes[element_id] = dataclasses.replace(current, text=text)

        return replace_elements(self.template, changes)

    def at(self, physical_time):
        """Evaluate the sequence at an arbitrary physical time.

        Left sample-and-hold: selects the sample acquired at or before
        `physical_time`, holding the final sample for any time at or past
        the final timestamp. A request outside `[times[0], times[-1]]`
        clamps to the nearest end sample; `physical_time` itself is
        reported unclamped.

        Args:
            physical_time: The requested time, in `time_unit`.

        Returns:
            A `Sample`.

        Raises:
            ValueError: If `physical_time` is NaN or infinite.
        """
        physical_time = float(physical_time)
        if not math.isfinite(physical_time):
            raise ValueError(f"physical_time must be finite, got {physical_time!r}")
        index = int(
            np.clip(
                np.searchsorted(self.times, physical_time, side="right") - 1,
                0,
                len(self.times) - 1,
            )
        )
        return Sample(
            index=index,
            physical_time=physical_time,
            acquisition_time=float(self.times[index]),
            view=self._build_frame(index),
        )

    def strip(self, indices):
        """Return a `PanelGroup` of the selected epochs, one slot per index.

        Each slot's element IDs are prefixed `f"{slot}/"` (`slot` is the
        0-based position within `indices`, not the sample index itself),
        so repeated or out-of-order indices still produce a tree with
        globally unique IDs. `source_id`, scales, and arrays are shared,
        unchanged, with `.frame(index)`; only `id` fields differ.

        Every slot also gets an added panel-space `Label` with ID
        `f"{slot}/time"` giving that slot's acquisition time -- present
        even when the template already carries a `Clock`-driven label
        (which is itself already updated to that slot's acquisition time
        by `.frame`), so every strip panel is labelled the same way. The
        added label is placed on the slot's top-level view when that view
        is a single view, or on the first marked view reached by
        descending through `views[0]` when the slot is itself a
        `PanelGroup` -- recursively, so a nested `PanelGroup` (a grid of
        panels used as one strip slot) still gets exactly one added label,
        on its innermost first child (documented here since neither
        convention is forced by the record shapes alone).

        Args:
            indices: Sample indices to select, one per output slot.

        Returns:
            A `PanelGroup` with id `"strip"`.
        """
        slots = []
        for slot, index in enumerate(indices):
            prefix = f"{slot}/"
            prefixed = _prefix_ids(self.frame(index), prefix)
            label = Label(
                f"{slot}/time",
                _clock_text(self.times[index], self.time_unit),
                _STRIP_LABEL_XY,
                space="panel",
            )
            slots.append(_add_strip_label(prefixed, label))
        return PanelGroup("strip", views=tuple(slots))

    def schedule(self, run_time, fps):
        """Return the shared output schedule for encoded playback.

        For duration `run_time` and frame rate `fps`, the schedule has
        `N = max(2 if len(times) > 1 else 1, ceil(run_time * fps))`
        physical times, evenly spaced from the first sample through the
        last (both endpoints included), so the encoded duration is `N/fps`
        and the final sample is visible for at least one output frame. A
        product `run_time * fps` within a relative 1e-9 of a whole number
        counts as that number, so `run_time = n / fps` gives `n` frames.

        Args:
            run_time: Presentation duration, finite and positive.
            fps: Output frame rate, finite and positive.

        Returns:
            An ndarray of physical times.

        Raises:
            ValueError: If `run_time` or `fps` is not finite and positive.
        """
        run_time = float(run_time)
        fps = float(fps)
        if not (math.isfinite(run_time) and run_time > 0):
            raise ValueError(f"run_time must be finite and positive, got {run_time!r}")
        if not (math.isfinite(fps) and fps > 0):
            raise ValueError(f"fps must be finite and positive, got {fps!r}")
        frames = run_time * fps
        # A duration of a whole number of frames given as run_time = n / fps
        # can multiply back to n plus float noise (0.28 * 25 is
        # 7.000000000000001); that is n frames, not n + 1.
        nearest = round(frames)
        if abs(frames - nearest) <= _FRAME_COUNT_RTOL * max(1.0, frames):
            frames = nearest
        count = max(2 if len(self.times) > 1 else 1, math.ceil(frames))
        return np.linspace(self.times[0], self.times[-1], count)


def _prefix_ids(element, prefix):
    """Return `element`'s tree with every ID prefixed, arrays shared."""
    new_id = f"{prefix}{element.id}"
    if isinstance(element, PanelGroup):
        new_views = tuple(_prefix_ids(view, prefix) for view in element.views)
        return dataclasses.replace(element, id=new_id, views=new_views)
    if hasattr(element, "marks"):
        new_marks = tuple(_prefix_ids(mark, prefix) for mark in element.marks)
        return dataclasses.replace(element, id=new_id, marks=new_marks)
    return dataclasses.replace(element, id=new_id)


def _add_strip_label(view, label):
    """Attach `label` to the first marked view in `view`'s tree.

    `view` is either a single marked view (`ImageView`, `CurveView`, or
    `TrackView`), which gets `label` directly, or a `PanelGroup`, in which
    case this descends into `views[0]` -- recursively, since a slot's
    template may itself be a nested `PanelGroup` -- until it reaches the
    first marked view at any depth, and attaches `label` there. Every
    ancestor `PanelGroup` on that path is rebuilt to carry the updated
    child; every sibling and unrelated branch is untouched.
    """
    if isinstance(view, PanelGroup):
        first, *rest = view.views
        updated_first = _add_strip_label(first, label)
        return dataclasses.replace(view, views=(updated_first, *rest))
    return dataclasses.replace(view, marks=(*view.marks, label))
