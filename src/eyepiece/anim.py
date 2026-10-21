"""Multi-sink animation recording: one draw pass feeds every output file.

Both entry points here are built on matplotlib's movie writers plus explicit
`grab_frame()` calls rather than on `FuncAnimation.save`, because a single
rendered frame must be handed to every requested sink at once. An mp4, a
gif, and an embedded-HTML player are all MovieWriters, each re-rastering the
figure at its own dpi, so `FuncAnimation.save` (one writer per call) would
have to replay the whole animation once per format. That is impossible when
the frame source is a one-shot generator, and wasteful when each frame costs
a simulation step. `record` opens all the sinks together and `frame()` grabs
into all of them, so `record(fig, "a.mp4", "a.gif")` costs exactly one pass.

Two consequences of that design are contracts, not accidents:

- **Nothing here ever clears the figure.** The documented default is update
  mode: build the figure once, then mutate artists between frames (a
  primitive's `.update`, or `set_data`). Redraw mode is nothing more than
  the caller putting `ax.clear()` in their own `draw` function, which works
  because this module does not touch axes at all. Blitting is meaningless
  under grab-frame rendering, so it is not offered.
- **Frame state belongs to the caller's iterable.** No frame data is
  materialized or cached here. Multi-rate animation is an `if` on the
  caller's index; a cumulative animation is an accumulator in the caller's
  generator.

The mechanics this module does own, each closing a specific trap:

- `savefig.bbox` is forced to None for the duration of a recording. A tight
  bbox varies the frame size from frame to frame, which breaks a writer's
  fixed-size pipe. It is applied through `matplotlib.rc_context`, so it is
  scoped to the recording and never leaks into the caller's rcParams.
  `path.snap` is turned off the same way unless `record(snap=True)`: Agg
  snaps markers and axis-aligned edges to whole pixels, which crisps a still
  and makes a slow move stall, then jump a pixel. `VIDEO_RC` (butt caps,
  round joins) holds the same way for lines made during the recording.
- A constrained-layout figure has its layout engine frozen on the first
  `frame()` call, before that frame is grabbed, and the freeze is scoped to
  the recording exactly like the `savefig.bbox` override: `record` restores
  whatever engine the figure had when the caller's own context exits.
  Constrained layout re-solves on every draw, which can shift axes between
  frames or resize the canvas outright -- the same trap `savefig.bbox`
  closes, from the other direction. Locking the engine at the first
  `frame()` call, instead of at `record`'s entry, means the freeze captures
  a layout solved against real content rather than an empty figure. The
  freeze has to land BEFORE the first grab: a grab is itself a draw, and
  two solves of the same figure need not agree (a layout that had not
  converged keeps moving, and one solved at a different dpi lands
  elsewhere), so grabbing first would give frame 0 a layout no other frame
  has and a looping video would jump at its seam. The layout is solved at
  the first sink's dpi, the dpi its frames are rasterized at, and re-solved
  until the axes stop moving, so the frozen layout is the settled one.
- Every grab passes `facecolor=fig.get_facecolor()`, so a dark-mode figure
  is not written onto a white background by savefig's own default.
- mp4 sinks get `-vf crop=trunc(iw/2)*2:trunc(ih/2)*2`. h264 rejects odd
  pixel dimensions, which a figure size in inches times a dpi hits often.
- Frames are rasterized through an Agg canvas, swapped in for the duration
  and swapped back out after. An interactive backend snaps the figure size
  to whole device pixels, which desynchronizes the frame size a writer
  declared to ffmpeg from the size it actually rasterizes, and a sheared
  mp4 is the only symptom. See `_agg_canvas`.
- The dpi for HTML output is passed explicitly to the writer. Setting
  `fig.dpi` does NOT work: the style library pins `savefig.dpi` for print
  figures, and savefig's dpi wins over the figure's, so an embedded player
  would carry 300-dpi frames. The writer dpi is the only effective control.

`PRESETS` holds fps/dpi pairs for the usual destinations, e.g.
`record(fig, "talk.mp4", **PRESETS["talk"])`. The ``talk`` preset runs at
60 fps: the default ease moves fastest at mid-move, and 30 fps still judders
there on turns and camera moves. ``talk_60`` is the same pair under its old
name. The documentation presets run at 30 fps, which keeps a slow move smooth
and costs little in bytes for flat graphics. `talk_rc` sizes type and strokes
for a talk frame from its height; documentation figures keep their own sizes.
``doc_hd`` rasterizes at 200 dpi, the resolution of a documentation still,
so a clip beside its still is not visibly softer on a 2x screen.

mp4 sinks all share one encoder profile, `eyepiece.ENCODER_PROFILE`: BT.709
tagged, crf 18, the animation tune, a keyframe every 2 s and faststart.
"""

import os
import subprocess
import tempfile
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, HTMLWriter, PillowWriter
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg

from eyepiece._ffmpeg import (
    convert_filter,
    encoder_args,
    resolve_ffmpeg,
    with_caller_pixels,
)
from eyepiece._motion import frame_count

PRESETS = {
    "talk": {"fps": 60, "dpi": 120},
    "talk_60": {"fps": 60, "dpi": 120},
    "gif": {"fps": 8, "dpi": 85},
    "jshtml": {"fps": 14, "dpi": 100},
    "docs": {"fps": 30, "dpi": 100},
    "doc_hd": {"fps": 30, "dpi": 200},
}

# Line ends and corners for video: butt caps, so a line is exactly as long as
# it is measured (a projecting cap adds half the width at each end), and round
# joins. Matplotlib reads these rcParams when an artist is created, never when
# it is drawn: `record` applies them, but only to artists made during the
# recording. Build a clip's figure under ``matplotlib.rc_context(VIDEO_RC)``
# (or `talk_rc`, which includes it); setting them after the figure exists, or
# only around the recording, leaves its lines with the caps they were made
# with.
VIDEO_RC = {
    "lines.solid_capstyle": "butt",
    "lines.dash_capstyle": "butt",
    "lines.solid_joinstyle": "round",
    "lines.dash_joinstyle": "round",
}

# Type and stroke of a talk frame as fractions of the frame height: a frame
# is read from across a room, so it is sized from the frame, not from a page.
TALK_FRACTIONS = {
    "title": 0.058,
    "body": 0.054,
    "axis_label": 0.039,
    "tick": 0.034,
    "data_stroke": 0.005,
    "scenery_of_data": 0.6,
    "margin": 0.04,
}


def talk_rc(height_px=1080, dpi=None):
    """The rcParams of a talk frame, with type and strokes sized from its height.

    An em of body text is 5.4 percent of the frame height, titles 5.8, axis
    labels 3.9 and tick labels 3.4; a data stroke is 0.5 percent and scenery
    (axes, grids, patch edges) 0.6 of that. At 1080 px and 120 dpi body text
    is 35 pt and a data stroke 3.2 pt. Tick marks are dropped, and the
    constrained-layout pads keep a 4 percent margin. `VIDEO_RC` is included.

    Matplotlib reads these rcParams when an artist is created, not when it
    is drawn, so build the whole figure under them; entering the context
    after the figure exists, or only around `record`, changes nothing already
    made. The documentation sizes are left alone::

        with matplotlib.rc_context(ep.talk_rc()):
            fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

    Args:
        height_px: Frame height in pixels.
        dpi: Dots per inch of the frame; None uses the ``talk`` preset's.

    Returns:
        A dict of rcParams.
    """
    dpi = PRESETS["talk"]["dpi"] if dpi is None else dpi
    frame_in = height_px / dpi
    frame_pt = frame_in * 72.0
    f = TALK_FRACTIONS
    data = f["data_stroke"] * frame_pt
    scenery = f["scenery_of_data"] * data
    pad = f["margin"] * frame_in
    return {
        **VIDEO_RC,
        "font.size": f["body"] * frame_pt,
        "legend.fontsize": f["body"] * frame_pt,
        "axes.titlesize": f["title"] * frame_pt,
        "figure.titlesize": f["title"] * frame_pt,
        "axes.labelsize": f["axis_label"] * frame_pt,
        "xtick.labelsize": f["tick"] * frame_pt,
        "ytick.labelsize": f["tick"] * frame_pt,
        "lines.linewidth": data,
        "lines.markersize": 3.0 * data,
        "axes.linewidth": scenery,
        "grid.linewidth": scenery,
        "patch.linewidth": scenery,
        "xtick.major.size": 0.0,
        "xtick.minor.size": 0.0,
        "ytick.major.size": 0.0,
        "ytick.minor.size": 0.0,
        "figure.constrained_layout.w_pad": pad,
        "figure.constrained_layout.h_pad": pad,
    }


_SUFFIX_PRESET = {".mp4": "talk", ".gif": "gif", ".html": "jshtml"}


def _resolve_ffmpeg():
    """Locate an ffmpeg binary for matplotlib's FFMpegWriter.

    Tries an explicit `rcParams["animation.ffmpeg_path"]` first, then the
    search of `eyepiece._ffmpeg.resolve_ffmpeg`: ``ffmpeg`` on PATH, then the
    binary shipped with `imageio_ffmpeg`.

    Returns:
        The path or bare name to run ffmpeg with.

    Raises:
        RuntimeError: If no ffmpeg can be found, naming both fixes.
    """
    return resolve_ffmpeg(matplotlib.rcParams.get("animation.ffmpeg_path", "ffmpeg"))


def _make_writer(path, fps, extra_ffmpeg_args, encoder=None):
    """Build the MovieWriter that owns a single output path.

    Args:
        path: Output path; its suffix picks the writer.
        fps: Frames per second baked into the output.
        extra_ffmpeg_args: Extra ffmpeg command-line arguments, appended
            after the even-dimension crop filter. Ignored by the non-mp4
            writers, which take no such arguments. Passing a `-vf` of your
            own REPLACES the crop rather than adding to it, because ffmpeg
            honors only the last `-vf`, which brings the odd-dimension h264
            failure back; write the crop into your own filter chain if you
            need one.
        encoder: Keys of `eyepiece.ENCODER_PROFILE` to replace for an mp4. An
            ``-pix_fmt`` in the extra arguments turns the color conversion
            and tags off, since the caller then owns the pixel format.

    Returns:
        An unopened `MovieWriter`.

    Raises:
        ValueError: If the suffix is not one of .mp4, .gif, or .html.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".mp4":
        encoder = with_caller_pixels(encoder, extra_ffmpeg_args)
        chain = ["crop=trunc(iw/2)*2:trunc(ih/2)*2", convert_filter(encoder)]
        movie = ["-vf", ",".join(f for f in chain if f), *encoder_args(fps, encoder)]
        return FFMpegWriter(fps=fps, extra_args=movie + list(extra_ffmpeg_args or []))
    if suffix == ".gif":
        return PillowWriter(fps=fps)
    if suffix == ".html":
        return HTMLWriter(fps=fps, embed_frames=True)
    raise ValueError(
        f"Unsupported animation suffix {suffix!r}: use .mp4, .gif, or .html"
    )


def _sink_dpi(path, dpi):
    """Rasterization dpi for one sink: explicit `dpi`, else its suffix default."""
    if dpi is not None:
        return dpi
    return PRESETS[_SUFFIX_PRESET[Path(path).suffix.lower()]]["dpi"]


def _rc_overrides(paths, snap=True):
    """The rcParams overrides that must hold for the duration of a recording.

    `savefig.bbox` is disabled because a tight bbox lets the frame size
    change between frames, which a writer's fixed-size pipe cannot take.
    The ffmpeg path is resolved here, before any writer is opened, so a
    missing ffmpeg fails with an actionable error instead of a bare
    FileNotFoundError from a subprocess launch. `VIDEO_RC` sets the caps
    and joins of lines made during the recording. Without `snap`,
    `path.snap` is off, which unsnaps every artist, whatever its own setting.
    """
    overrides = {**VIDEO_RC, "savefig.bbox": None}
    if not snap:
        overrides["path.snap"] = False
    if any(Path(p).suffix.lower() == ".mp4" for p in paths):
        overrides["animation.ffmpeg_path"] = _resolve_ffmpeg()
    return overrides


@contextmanager
def _agg_canvas(fig):
    """Rasterize through an Agg canvas for the duration of a recording.

    An interactive backend's canvas snaps `set_size_inches` to whole
    device pixels, and matplotlib's writers need that call to land
    exactly. `MovieWriter` shrinks the figure to the even pixel
    dimensions h264 demands, remembers the size it asked for, and
    restores that size before every grab -- but the frame size it
    declared to ffmpeg was read back off the figure afterwards. On a
    snapping backend the two disagree by a pixel, so ffmpeg reads each
    rasterized row at the wrong stride and the movie comes out sheared,
    with the frame wrapping further sideways down every row. Nothing
    raises: the mp4 is silently wrong while the gif beside it, whose
    writer never resizes anything, is correct. Agg does not snap.
    """
    original = fig.canvas
    FigureCanvasAgg(fig)
    try:
        yield
    finally:
        fig.canvas = original


def _finish_quietly(writer, path, *, warn):
    """Close `writer` without letting a failure of its own escape.

    Used while another error is already on its way out, where an escaping
    teardown error would replace the reason the caller actually needs. A
    writer that received no frame cannot finalize at all and that is
    expected here, so `warn` is False for it; one that did receive frames
    and still failed to close has left a truncated file behind, which is
    worth saying out loud even though the original error still wins.
    """
    try:
        writer.finish()
    except Exception as exc:
        if warn:
            warnings.warn(
                f"{path} may be truncated: closing it raised {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )


@contextmanager
def _sink(writer, fig, path, dpi, counter):
    """Open one writer, and close it without masking the body's failure.

    `AbstractMovieWriter.saving` finalizes in a bare `finally`, so a writer
    that never received a frame raises out of its own teardown -- Pillow's
    indexes `self._frames[0]` -- and that error REPLACES whatever actually
    went wrong inside the block. A draw that raised, or a glyph the backend
    refused to render, would reach the caller as `IndexError: list index
    out of range` from inside matplotlib, pointing at the wrong library
    entirely. The writer is therefore driven directly; `saving`'s only
    other duty is an inner `savefig.bbox` override that `_rc_overrides`
    already holds open for the whole recording.
    """
    writer.setup(fig, path, dpi)
    try:
        yield
    except BaseException:
        _finish_quietly(writer, path, warn=bool(counter["frames"]))
        raise
    writer.finish()


def _scale_signature(fig):
    """Every axes' data scales: the numbers a reader converts into a size.

    Axis limits and image/collection norms, keyed by axes index. This is
    deliberately not the geometry (position, size); geometry stability is
    already guaranteed by the `savefig.bbox` override and the layout
    freeze. This is the other half: what the axes claim the numbers MEAN.
    """
    signature = {}
    for index, ax in enumerate(fig.axes):
        entry = {"x": tuple(ax.get_xlim()), "y": tuple(ax.get_ylim())}
        if hasattr(ax, "get_zlim"):
            entry["z"] = tuple(ax.get_zlim())
        norms = []
        for artist in [*ax.images, *ax.collections]:
            get_clim = getattr(artist, "get_clim", None)
            if get_clim is None:
                continue
            try:
                norms.append(tuple(float(v) for v in get_clim()))
            except (TypeError, ValueError):
                continue
        entry["color"] = tuple(norms)
        signature[index] = entry
    return signature


def _span_ratio(before, after):
    """How many times bigger `after`'s span is than `before`'s, or None."""
    old = abs(before[1] - before[0])
    new = abs(after[1] - after[0])
    if old == 0.0 or new == 0.0:
        return None
    ratio = new / old
    return ratio if abs(ratio - 1.0) > 0.01 else None


def _drift_warnings(first, current, free=frozenset()):
    """Human-readable descriptions of every scale that moved.

    Axes whose index is in `free` may move their limits (a camera); their
    color norms are still compared.
    """
    messages = []
    for index, before in first.items():
        after = current.get(index)
        if after is None:
            continue
        for key in ("x", "y", "z"):
            if index in free:
                continue
            if key not in before or key not in after:
                continue
            ratio = _span_ratio(before[key], after[key])
            if ratio is not None:
                messages.append(
                    f"axes[{index}] {key}-limits {before[key]} -> {after[key]} "
                    f"({ratio:.2f}x the original span)"
                )
        pairs = zip(before["color"], after["color"], strict=False)
        for slot, (old, new) in enumerate(pairs):
            ratio = _span_ratio(old, new)
            if ratio is not None:
                messages.append(
                    f"axes[{index}] color norm #{slot} {old} -> {new} "
                    f"({ratio:.2f}x the original span)"
                )
    return messages


_SETTLE_PASSES = 5
_SETTLE_TOL_PX = 0.1


def _axes_pixels(fig):
    """Every axes' position as (x0, y0, x1, y1) in display pixels."""
    return [ax.get_position().transformed(fig.transFigure).extents for ax in fig.axes]


def _max_shift(before, after):
    """Largest coordinate change between two `_axes_pixels` snapshots."""
    if len(before) != len(after):
        return float("inf")
    shifts = [
        abs(a - b)
        for old, new in zip(before, after, strict=True)
        for a, b in zip(old, new, strict=True)
    ]
    return max(shifts, default=0.0)


class ScaleDriftError(RuntimeError):
    """A data scale changed mid-recording and `on_rescale="raise"` was set.

    The message names every axes and the ratio its span changed by. `record`
    deletes the files it had started before this propagates.
    """


_ON_RESCALE = ("warn", "raise", "ignore")


class _Recorder:
    """Frame grabber yielded by `record`; fans one figure out to every sink."""

    def __init__(
        self,
        fig,
        writers,
        counter,
        layout_dpi,
        on_rescale="warn",
        free=frozenset(),
        stills=None,
        stills_dpi=None,
    ):
        self.fig = fig
        self.writers = writers
        self._counter = counter
        self._layout_dpi = layout_dpi
        self._layout_frozen = False
        self._on_rescale = on_rescale
        self._free = free
        self._first_scales = None
        self._warned = False
        self._stills = stills or {}
        self._stills_dpi = stills_dpi or layout_dpi
        self.stills_written = []

    def frame(self):
        """Grab the figure exactly as it stands now, into every open sink.

        The figure's own facecolor is passed to each grab so a dark figure
        is not rasterized onto savefig's white default.

        The first call also freezes the figure's layout engine, against
        this frame's content and before it is grabbed, so frame 0 carries
        the same layout as every later frame: see `record` for why the
        freeze waits here instead of happening at `record`'s entry.
        """
        if not self._layout_frozen:
            self._freeze_layout()
        facecolor = self.fig.get_facecolor()
        for writer in self.writers:
            writer.grab_frame(facecolor=facecolor)
        for path in self._stills.get(self._counter["frames"], ()):
            self._write_still(path, facecolor)
        self._counter["frames"] += 1
        self._check_scales()

    def _write_still(self, path, facecolor):
        """Save the figure as it stands as a PNG, the state the sinks just grabbed.

        The metadata is cleared so the same frame always gives the same file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(
            path,
            format="png",
            dpi=self._stills_dpi,
            facecolor=facecolor,
            metadata={"Software": None},
        )
        self.stills_written.append(path)

    def _freeze_layout(self):
        """Settle the layout at the sink dpi, then lock it in place.

        The solve runs at the dpi the frames are rasterized at rather than
        the figure's own, so the frozen layout is the one a grab at that dpi
        would have solved. It is repeated until no axes moves by more than
        `_SETTLE_TOL_PX` sink pixels, at most `_SETTLE_PASSES` times: a
        constrained layout whose inputs changed after its last solve (a
        footer rect set after a settling draw, say) needs several solves to
        come to rest, and freezing after one would lock in a layout tens of
        pixels from where it is heading. Every grab after this, frame 0's
        included, draws that single settled layout.
        """
        figure_dpi = self.fig.dpi
        self.fig.dpi = self._layout_dpi
        try:
            before = _axes_pixels(self.fig)
            for _ in range(_SETTLE_PASSES):
                self.fig.draw_without_rendering()
                after = _axes_pixels(self.fig)
                moved = _max_shift(before, after)
                before = after
                if moved < _SETTLE_TOL_PX:
                    break
        finally:
            self.fig.dpi = figure_dpi
        self.fig.set_layout_engine("none")
        self._layout_frozen = True

    def _check_scales(self):
        """Report once if any axes changed what its numbers mean mid-recording.

        A reader cannot separate a change in the data from a change in the
        drawing, so a scale that follows the data redraws every frame at
        full height and a quantity that fell by an order of magnitude reads
        as one that never moved. That failure is silent -- the animation
        renders perfectly -- so it is caught here rather than in review.
        """
        if self._on_rescale == "ignore" or self._warned:
            return
        current = _scale_signature(self.fig)
        if self._first_scales is None:
            self._first_scales = current
            return
        messages = _drift_warnings(self._first_scales, current, self._free)
        if not messages:
            return
        self._warned = True
        text = (
            "data scales changed during recording, so a quantity is drawn at "
            "a different size in different frames and the change will read as "
            "a change in the data: "
            + "; ".join(messages)
            + ". Pin the limits from the union over all frames before the "
            'frame loop, or pass on_rescale="ignore" if the rescaling is the '
            "point."
        )
        if self._on_rescale == "raise":
            raise ScaleDriftError(text)
        warnings.warn(text, RuntimeWarning, stacklevel=3)

    def hold(self, n):
        """Repeat the current figure for `n` frames, to dwell on a moment.

        Args:
            n: Number of frames to emit. The figure is not redrawn between
                them, so the cost is `n` rasterizations of the same state.
        """
        for _ in range(n):
            self.frame()


@contextmanager
def record(
    fig,
    *paths,
    fps=10,
    dpi=None,
    extra_ffmpeg_args=None,
    encoder=None,
    on_rescale="warn",
    free_limits=(),
    stills=None,
    plan=None,
    stills_dpi=None,
    snap=False,
    allow_rescale=None,
):
    """Open every output file at once and record `fig` frame by frame.

    Use this when the frames come from a loop the caller already owns (a
    simulation stepping forward, say) rather than from a frame index. The
    figure is never cleared: mutate its artists between `frame()` calls, or
    call `ax.clear()` yourself for redraw-style animation.

    If `fig` has a constrained-layout engine, its layout is settled at the
    first sink's dpi (re-solved until the axes stop moving, at most five
    times) and frozen when the first `frame()` call starts,
    before that frame is grabbed, so the first frame and every later one
    share a single layout. The engine is restored to
    whatever it was before this call once the `with` block exits -- on
    the exception path too, the same as the writer cleanup this context
    manager already guarantees. A figure with no layout engine, or one
    the caller already froze, records normally: freezing is a no-op on
    top of "none" already, and a figure that had no engine is left with
    no engine -- restored deliberately, with the layout rcParams held off
    so that `None` means "no engine" rather than "consult the rcParams",
    which would otherwise install an engine the figure never had.
    The engine is captured before the `savefig.bbox` override and the
    writers are set up, so it unwinds last, after every writer has
    finished and the rcParams overrides are back -- the freeze outlives
    everything it was protecting.

    Example::

        with record(fig, "run.mp4", "run.gif", fps=10) as rec:
            for state in simulation:
                result.update(state.image)
                rec.frame()
            rec.hold(8)

    Args:
        fig: The Figure to rasterize each frame from.
        *paths: One output path per sink. The suffix picks the writer:
            .mp4 (ffmpeg), .gif (Pillow), .html (a self-contained player
            with the frames embedded). A path's parent directories are
            created if they do not exist, as `save_fig` does.
        fps: Frames per second recorded into every sink.
        dpi: Rasterization dpi applied to every sink. None gives each sink
            the default for its suffix (.gif 85, .html 100, .mp4 120),
            which is usually what mixed output wants.
        extra_ffmpeg_args: Extra ffmpeg arguments for mp4 sinks, appended
            after the even-dimension crop filter. A `-vf` of your own
            replaces that crop instead of adding to it (ffmpeg honors the
            last `-vf` only), which reopens the odd-dimension h264 failure;
            include the crop in your own filter chain if you need one.
        encoder: Keys of `eyepiece.ENCODER_PROFILE` to replace for the mp4
            sinks, e.g. ``{"crf": 23}``; the rest of the profile stays.
        on_rescale: What to do when a data scale drifts: "warn" (a
            `RuntimeWarning`, once), "raise" (a `ScaleDriftError`, after
            which every file this call had started, stills included, is
            deleted so no partial output survives), or "ignore" (no check).
            Every axes' limits and color norms are fingerprinted on the
            first frame and compared on each later one, because an axis that
            follows the data redraws every frame at full height and hides the
            very change the animation exists to show. Use "ignore" when the
            rescaling is deliberate.
        free_limits: An axes, or axes, whose limits are allowed to change,
            such as a camera zooming through `view_limits`. Only their limits are
            exempt: their color norms, and every other axes, are still
            checked, which `on_rescale="ignore"` would not do.
        stills: PNG files to write from chosen frames during the same
            encode, as a mapping from path to when. A number is a time in
            seconds, mapped to the frame at ``frame_count(t, fps)``; a span
            name (needs `plan`) is that span's last frame, its settled state; a
            ``(span, where)`` pair picks ``"start"``, ``"mid"`` or ``"last"``.
            A still is the figure as the sinks grabbed it at that frame, with
            no second pass over the frames. Parent directories are created.
        plan: The `Plan` the frames were laid out by, to name stills by span.
            Its fps must equal `fps`. With a plan, the recording also warns
            once, when the block exits, with a `RuntimeWarning` listing every
            knob the draw function never read (through `Plan.value`,
            `Plan.frames` or `Plan.progress`) and every knob that never
            changes over the clip: a clip whose concept is motion needs a knob
            that moves. The ``indicate.*`` and ``circumscribe.*`` knobs are
            exempt, as `Plan.unused_knobs` describes.
        stills_dpi: Dpi of the stills; None uses the first sink's dpi, which
            makes a still pixel-identical to that sink's frame.
        snap: Whether Agg snaps markers, axis-aligned lines and rectilinear
            patch edges to whole pixels. The default False draws every frame
            at sub-pixel positions, so a slow move glides instead of stalling
            and jumping a pixel; a frame then matches a snapped still of the
            same state only to within half a pixel. True keeps the snapping
            of a still.
        allow_rescale: Deprecated; `True` means ``on_rescale="ignore"``. It
            warns with a `DeprecationWarning` and will be removed in a later
            release.

    Yields:
        A recorder with `.frame()` and `.hold(n)`.

    Raises:
        ValueError: If a path has an unsupported suffix, an axes in
            `free_limits` is not on `fig`, `on_rescale` is not one of the
            three values, or a still names a span without a `plan`, an
            unknown span, or a negative time.
        ScaleDriftError: If a scale drifted and `on_rescale` is "raise".
        RuntimeError: If an mp4 sink is requested and no ffmpeg is found,
            or if the block exits without ever grabbing a frame, or before
            the frame a still asked for.
    """
    on_rescale = _resolve_on_rescale(on_rescale, allow_rescale)
    still_frames = _resolve_stills(stills, plan, fps)
    if plan is not None:
        plan._read.clear()
    free = set()
    if isinstance(free_limits, Axes):
        free_limits = [free_limits]
    for ax in free_limits:
        if ax not in fig.axes:
            raise ValueError("free_limits holds an axes that is not on this figure")
        free.add(fig.axes.index(ax))
    free = frozenset(free)
    saved_engine = fig.get_layout_engine()

    def restore_engine():
        # set_layout_engine(None) does not mean "no engine": matplotlib
        # reads figure.autolayout / figure.constrained_layout.use and
        # installs whichever they ask for, so replaying a saved None would
        # give a figure an engine it never had.
        if saved_engine is not None:
            fig.set_layout_engine(saved_engine)
            return
        # Restoring "no engine" still has to be done, not skipped. During
        # the recording matplotlib's own writer bookkeeping calls
        # set_layout_engine(None) itself, and under either layout rcParam
        # that BUILDS an engine on a figure that had none; the freeze then
        # demotes it to a placeholder, which is what the caller was left
        # holding. Turning both rcParams off for the length of the call is
        # what makes None mean what it says, using public API rather than
        # assigning the private engine attribute.
        with matplotlib.rc_context(
            {"figure.autolayout": False, "figure.constrained_layout.use": False}
        ):
            fig.set_layout_engine(None)

    counter = {"frames": 0}
    recorder = None
    try:
        with ExitStack() as stack:
            stack.callback(restore_engine)
            stack.enter_context(matplotlib.rc_context(_rc_overrides(paths, snap)))
            stack.enter_context(_agg_canvas(fig))
            writers = [_make_writer(p, fps, extra_ffmpeg_args, encoder) for p in paths]
            for writer, path in zip(writers, paths, strict=True):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                stack.enter_context(
                    _sink(writer, fig, str(path), _sink_dpi(path, dpi), counter)
                )
            recorder = _Recorder(
                fig,
                writers,
                counter,
                _sink_dpi(paths[0], dpi) if paths else fig.dpi,
                on_rescale=on_rescale,
                free=free,
                stills=still_frames,
                stills_dpi=stills_dpi,
            )
            yield recorder
            _check_frames(counter, paths)
            _check_stills_reached(still_frames, counter["frames"])
            _warn_unused_knobs(plan)
    except ScaleDriftError:
        # The sinks have closed by now; a half-written movie must not pass
        # for a finished one.
        for path in [*paths, *(recorder.stills_written if recorder else ())]:
            Path(path).unlink(missing_ok=True)
        raise


def _warn_unused_knobs(plan):
    """One warning listing the knobs the draw function never read or that never move."""
    if plan is None:
        return
    unread, constant = plan.unused_knobs()
    parts = []
    if unread:
        parts.append(f"never read by the draw function: {list(unread)}")
    if constant:
        parts.append(f"constant over the whole clip: {list(constant)}")
    if parts:
        warnings.warn(
            "the plan has knobs that "
            + "; ".join(parts)
            + " (a knob that never changes cannot show motion)",
            RuntimeWarning,
            stacklevel=3,
        )


def _resolve_on_rescale(on_rescale, allow_rescale):
    """The effective `on_rescale`, folding in the deprecated `allow_rescale`."""
    if on_rescale not in _ON_RESCALE:
        raise ValueError(f"on_rescale must be one of {_ON_RESCALE}, got {on_rescale!r}")
    if allow_rescale is None:
        return on_rescale
    warnings.warn(
        "record(allow_rescale=) is deprecated; pass on_rescale='ignore' instead of "
        "allow_rescale=True (the default on_rescale='warn' is allow_rescale=False)",
        DeprecationWarning,
        stacklevel=4,
    )
    if allow_rescale:
        if on_rescale == "raise":
            raise ValueError("allow_rescale=True contradicts on_rescale='raise'")
        return "ignore"
    return on_rescale


def _resolve_stills(stills, plan, fps):
    """Map each still's frame index to its paths, failing before anything opens."""
    if not stills:
        return {}
    if plan is not None and abs(plan.fps - fps) > 1e-9:
        raise ValueError(f"plan was resolved at {plan.fps} fps but record has {fps}")
    frames = {}
    for path, when in dict(stills).items():
        frames.setdefault(_still_frame(when, plan, fps), []).append(path)
    return frames


def _still_frame(when, plan, fps):
    """Frame index for one `stills` entry: seconds, a span name, or (span, where)."""
    if isinstance(when, str) or isinstance(when, tuple):
        span, where = (when, "last") if isinstance(when, str) else when
        if plan is None:
            raise ValueError(f"a still at span {span!r} needs plan=")
        if where not in ("start", "mid", "last"):
            raise ValueError(f"a still's where is start, mid or last, got {where!r}")
        try:
            return plan.frame_of(span, where)
        except KeyError:
            raise ValueError(f"the plan has no span named {span!r}") from None
    if when < 0:
        raise ValueError(f"a still's time must be at least 0 s, got {when}")
    return frame_count(when, fps)


def _check_stills_reached(still_frames, grabbed):
    """Fail if the recording ended before a requested still's frame."""
    late = sorted(k for k in still_frames if k >= grabbed)
    if late:
        raise RuntimeError(
            f"the recording grabbed {grabbed} frames, so the stills at frame "
            f"{late[0]} and later were never written; a time that rounds to the "
            "clip's frame count is past the last frame, and the settled state of "
            "a span is the still (span, 'last'), not its end time"
        )


def _check_frames(counter, paths):
    """Raise if no frame was grabbed."""
    if not counter["frames"]:
        # Raised here rather than per sink so the message can name them
        # all. It travels out through the sinks, which take their
        # exception path and close without adding noise of their own.
        joined = ", ".join(str(p) for p in paths)
        raise RuntimeError(
            f"no frames were grabbed, so the outputs are empty or "
            f"missing ({joined}); call rec.frame() at least once "
            "inside the record block"
        )


_BITEXACT = ("-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1")


class RawSink:
    """Write raw frames of one size to an H.264 mp4 through ffmpeg.

    `record` grabs frames from a figure; a `RawSink` takes frames from
    anywhere, such as `PageCamera.render`, which crops a figure in ways a
    movie writer cannot. Frames go to ffmpeg as raw RGBA and encode as H.264
    in the profile's pixel format (yuv420p by default) at constant quality,
    with no metadata and bit-exact encoder flags, so the same frames always
    give the same file. The movie is
    written beside `path` and moved into place only when it finishes, so a
    failed run never replaces a good file with a broken one.

    Example::

        with RawSink("tour.mp4", (1920, 1080), fps=30) as sink:
            for view in views:
                sink.write(camera.render(view))

    Args:
        path: Output path; it must end in ``.mp4``. Parent directories are
            created as needed.
        size_px: ``(width, height)`` in pixels, both even (H.264 in yuv420p
            needs even dimensions).
        fps: Frames per second.
        extra_ffmpeg_args: Extra ffmpeg output arguments, appended last, so
            they override the defaults (``-crf 18`` among them).
        encoder: Keys of `eyepiece.ENCODER_PROFILE` to replace, e.g.
            ``{"crf": 23}``; the rest of the profile stays.

    Raises:
        ValueError: If the path is not an mp4 or a dimension is not a
            positive even integer.
        RuntimeError: If no ffmpeg can be found.
    """

    def __init__(self, path, size_px, fps, *, extra_ffmpeg_args=None, encoder=None):
        """Validate the size and start ffmpeg on a file beside `path`."""
        path = Path(path)
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"RawSink writes .mp4 files, got {path.name!r}")
        width, height = (int(v) for v in size_px)
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(
                f"size_px must be positive and even for H.264, got {width}x{height}"
            )
        self.path = path
        self.size_px = (width, height)
        self.frames = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._partial = path.with_name(f".{path.stem}.partial.mp4")
        encoder = with_caller_pixels(encoder, extra_ffmpeg_args)
        convert = convert_filter(encoder)
        filters = ["-vf", convert] if convert else []
        cmd = [
            _resolve_ffmpeg(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", "-an",
            "-vcodec", "libx264",
            *filters, *encoder_args(fps, encoder), *_BITEXACT,
            *(extra_ffmpeg_args or ()), str(self._partial),
        ]  # fmt: skip
        # stderr goes to a file, not a pipe: a pipe nobody reads fills after
        # about 64 kB and then blocks ffmpeg, and with it every write.
        self._log = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)

    def write(self, frame):
        """Append one frame.

        Args:
            frame: ``(height, width, 4)`` or ``(height, width, 3)`` uint8
                pixels, top row first, exactly `size_px`.

        Raises:
            ValueError: If the frame's size, channel count, or dtype is wrong.
            RuntimeError: If ffmpeg has stopped accepting frames.
        """
        frame = np.asarray(frame)
        width, height = self.size_px
        if frame.ndim != 3 or frame.shape[:2] != (height, width):
            raise ValueError(
                f"frame is {frame.shape}, expected {width}x{height} pixels "
                f"as ({height}, {width}, 4)"
            )
        if frame.dtype != np.uint8 or frame.shape[2] not in (3, 4):
            raise ValueError("frames must be uint8 RGB or RGBA")
        if frame.shape[2] == 3:
            alpha = np.full((height, width, 1), 255, np.uint8)
            frame = np.concatenate([frame, alpha], axis=2)
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise RuntimeError(
                f"ffmpeg stopped accepting frames: {self._stderr()}"
            ) from None
        self.frames += 1

    def _stderr(self):
        try:
            self._log.seek(0)
            return self._log.read().decode(errors="replace").strip()[-2000:]
        except (OSError, ValueError):
            return ""

    def _cleanup(self):
        self._log.close()
        self._partial.unlink(missing_ok=True)

    def close(self):
        """Finish the file and move it into place.

        Raises:
            RuntimeError: If no frame was written or ffmpeg failed; the
                output path is then left as it was.
        """
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        code = self._proc.wait()
        if not self.frames:
            self._cleanup()
            raise RuntimeError(f"no frames were written to {self.path}")
        if code != 0:
            message = self._stderr()
            self._cleanup()
            raise RuntimeError(f"ffmpeg failed writing {self.path}: {message}")
        os.replace(self._partial, self.path)
        self._log.close()

    def __enter__(self):
        """Return the sink."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Finish the file, or stop ffmpeg and discard it if the body raised."""
        if exc_type is None:
            self.close()
            return
        # The body's error wins; stop ffmpeg without masking it, and leave
        # whatever file was at the path untouched.
        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        self._proc.kill()
        self._proc.wait()
        self._cleanup()


def _is_one_shot(frames):
    """Whether `frames` is an iterator, which can only be walked once."""
    try:
        return iter(frames) is frames
    except TypeError:
        return False


class Animation:
    """A figure, a draw function, and a frame source, bound but not yet run.

    Build one with `animate`; nothing is rendered until `.save`, `.jshtml`,
    or `.video` is called. Every one of those renders by walking `frames`
    from the start, calling `draw(fig, ctx)` once per item and grabbing a
    frame right after -- they all route through `record`, so a recording's
    scoped mechanics (rcParams overrides, layout freeze) apply the same way
    whether you call `.save` directly or one of the other two.

    Whether a second render is possible depends on what `frames` was: a
    sequence or an int can be walked again for free, a bare generator or
    iterator raises RuntimeError on the second render, and a zero-argument
    callable is called again to produce a fresh iterator. See `animate` for
    the full breakdown.

    Attributes:
        fig: The bound Figure, rasterized by every render.
        draw: The per-frame draw callable, `draw(fig, ctx)`.
        fps: Default playback rate for `.save`, `.jshtml`, and `.video`,
            each of which takes an `fps` of its own to override it.
        frames: The frame source, coerced from an int to a `range`.
        n_frames: The frame count, if it is known: measured from `frames`
            when that has a length, else the `n_frames` passed to
            `animate`, else None.
    """

    def __init__(self, fig, draw, frames, fps, n_frames):
        """Bind the figure, draw function, and frame source; see `animate`.

        Not meant to be called directly -- use `animate`, which fills in
        `n_frames` from `frames` when it is not given explicitly.
        """
        self.fig = fig
        self.draw = draw
        self.fps = fps
        self.frames = range(frames) if isinstance(frames, int) else frames
        self._one_shot = not callable(self.frames) and _is_one_shot(self.frames)
        self._consumed = False
        if n_frames is None:
            try:
                n_frames = len(self.frames)
            except TypeError:
                n_frames = None
        self.n_frames = n_frames

    def _frame_iter(self):
        """A fresh iterator over the frame source, or fail loudly.

        A callable frame source is called again for every render, which is
        the supported way to render the same animation more than once from
        a generator. A bare one-shot iterator cannot be replayed, so a
        second render raises here -- BEFORE any sink is opened, so a failed
        second `save` leaves no truncated file behind -- rather than
        silently writing an empty animation.
        """
        if callable(self.frames):
            return iter(self.frames())
        if self._one_shot:
            if self._consumed:
                raise RuntimeError(
                    "the frame source is a one-shot generator and was "
                    "already consumed; pass a sequence, an int, or a "
                    "zero-argument callable returning a fresh iterator to "
                    "render more than once"
                )
            self._consumed = True
        return iter(self.frames)

    def save(
        self,
        *paths,
        dpi=None,
        fps=None,
        stills=None,
        plan=None,
        on_rescale="warn",
        stills_dpi=None,
    ):
        """Render every frame once, into every path.

        Args:
            *paths: Output paths, one per sink; see `record`.
            dpi: Rasterization dpi for every sink. None gives each sink the
                default for its suffix.
            fps: Playback rate override; None keeps the animation's own.
            stills: PNGs to write from chosen frames during this render; see
                `record`.
            plan: The `Plan` the frames follow, to name stills by span.
            on_rescale: "warn", "raise" or "ignore"; see `record`.
            stills_dpi: Dpi of the stills; None uses the first sink's.

        Returns:
            The list of `pathlib.Path` written, in the order given.

        Raises:
            RuntimeError: If the frame source is an exhausted one-shot
                generator, or if it yields no frames at all.
        """
        frames = self._frame_iter()
        rate = self.fps if fps is None else fps
        with record(
            self.fig,
            *paths,
            fps=rate,
            dpi=dpi,
            on_rescale=on_rescale,
            stills=stills,
            plan=plan,
            stills_dpi=stills_dpi,
        ) as rec:
            for ctx in frames:
                self.draw(self.fig, ctx)
                rec.frame()
        return [Path(p) for p in paths]

    def jshtml(
        self,
        dpi=100,
        fps=None,
        *,
        stills=None,
        plan=None,
        on_rescale="warn",
        stills_dpi=None,
    ):
        """Render a self-contained HTML player and return it as a string.

        The frames are embedded as base64 PNGs, so the result needs no
        ffmpeg, no server, and no sidecar files. Display it in a notebook
        with `IPython.display.HTML(...)`.

        `dpi` is handed to the writer explicitly. Reducing `fig.dpi`
        instead does not shrink the embedded frames, because savefig's own
        dpi (pinned high for print figures by the style library) wins over
        the figure's.

        The figure is closed once the HTML exists, before returning: a
        notebook's inline backend would otherwise also render the figure
        itself, leaving a static copy of the last frame parked under the
        player. Closing any earlier would leave nothing to rasterize.

        Args:
            dpi: Rasterization dpi for the embedded frames. Frames cost
                roughly 30 kB each at the default.
            fps: Playback rate override; None keeps the animation's own.
            stills: PNGs to write from chosen frames during this render; see
                `record`.
            plan: The `Plan` the frames follow, to name stills by span.
            on_rescale: "warn", "raise" or "ignore"; see `record`.
            stills_dpi: Dpi of the stills; None uses the player's dpi.

        Returns:
            The HTML document as a string.

        Raises:
            RuntimeError: If the frame source is an exhausted one-shot
                generator, or if it yields no frames at all.
        """
        frames = self._frame_iter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "animation.html"
            rate = self.fps if fps is None else fps
            with record(
                self.fig,
                path,
                fps=rate,
                dpi=dpi,
                on_rescale=on_rescale,
                stills=stills,
                plan=plan,
                stills_dpi=stills_dpi,
            ) as rec:
                for ctx in frames:
                    self.draw(self.fig, ctx)
                    rec.frame()
            html = path.read_text()
        plt.close(self.fig)
        return html

    def video(self, path, dpi=100, fps=None):
        """Render to a video file and return something a notebook can show.

        Args:
            path: Output path; the suffix picks the writer, as in `record`.
            dpi: Rasterization dpi.
            fps: Playback rate override; None keeps the animation's own.

        Returns:
            An `IPython.display.Video` carrying the embedded file when
            IPython is installed, else the `pathlib.Path` written.

        Raises:
            RuntimeError: If the frame source is an exhausted one-shot
                generator, if it yields no frames at all, or if an mp4 is
                requested without ffmpeg.
        """
        self.save(path, dpi=dpi, fps=fps)
        try:
            from IPython.display import Video
        except ImportError:
            return Path(path)
        return Video(str(path), embed=True, html_attributes="controls loop")


def animate(fig, draw, frames, *, fps=10, n_frames=None):
    """Bind a figure, a per-frame draw function, and a frame source.

    Nothing is rendered until `.save`, `.jshtml`, or `.video` is called, so
    the same animation can go to several destinations at different dpi (as
    long as the frame source can be walked again; see `frames` below).

    `draw(fig, ctx)` is called once per frame with the figure and the
    current item from `frames`. It must mutate existing artists rather than
    build new ones -- this module never clears anything, so re-plotting each
    frame would stack artists up. To animate in redraw style instead, call
    `ax.clear()` inside `draw`.

    Example::

        result = imshow_log(frames_data[0])

        def draw(fig, k):
            result.update(frames_data[k])

        animate(result.fig, draw, len(frames_data), fps=10).save("out.gif")

    Args:
        fig: The Figure to rasterize.
        draw: Callable `draw(fig, ctx)` invoked once per frame.
        frames: The frame source, in one of four forms. An int `n` means
            `range(n)`. Any re-iterable (list, range, array) is walked
            afresh for every render. A one-shot generator or iterator is
            walked once, and a second render raises RuntimeError instead
            of writing an empty file. A zero-argument callable is called
            per render to produce a fresh iterator, which is how a
            generator-backed animation is rendered more than once.
        fps: Playback rate, overridable per call for `.jshtml`.
        n_frames: Declared frame count, exposed as `.n_frames` for callers
            that need it up front. Only useful when `frames` has no length
            of its own (a generator or callable); otherwise it is measured.

    Returns:
        An `Animation` with `.save(*paths, dpi=None)`,
        `.jshtml(dpi=100, fps=None)`, and `.video(path, dpi=100)`.
    """
    return Animation(fig, draw, frames, fps=fps, n_frames=n_frames)
