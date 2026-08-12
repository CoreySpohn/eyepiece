"""Multi-sink animation recording: one draw pass feeds every output file.

Both entry points here are built on `MovieWriter.saving()` plus explicit
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
- A constrained-layout figure has its layout engine frozen right after the
  first frame is drawn, and the freeze is scoped to the recording exactly
  like the `savefig.bbox` override: `record` restores whatever engine the
  figure had when the caller's own context exits. Constrained layout
  re-solves on every draw, which can shift axes between frames or resize
  the canvas outright -- the same trap `savefig.bbox` closes, from the
  other direction. Locking the engine after frame one, instead of at
  `record`'s entry, means the freeze captures a layout solved against real
  content rather than an empty figure.
- Every grab passes `facecolor=fig.get_facecolor()`, so a dark-mode figure
  is not written onto a white background by savefig's own default.
- mp4 sinks get `-vf crop=trunc(iw/2)*2:trunc(ih/2)*2`. h264 rejects odd
  pixel dimensions, which a figure size in inches times a dpi hits often.
- The dpi for HTML output is passed explicitly to the writer. Setting
  `fig.dpi` does NOT work: the style library pins `savefig.dpi` for print
  figures, and savefig's dpi wins over the figure's, so an embedded player
  would carry 300-dpi frames. The writer dpi is the only effective control.

`PRESETS` holds measured fps/dpi pairs for the usual destinations, e.g.
`record(fig, "talk.mp4", **PRESETS["talk"])`.
"""

import shutil
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, HTMLWriter, PillowWriter

PRESETS = {
    "talk": {"fps": 10, "dpi": 120},
    "gif": {"fps": 8, "dpi": 85},
    "jshtml": {"fps": 14, "dpi": 100},
    "docs": {"fps": 10, "dpi": 100},
}

_SUFFIX_PRESET = {".mp4": "talk", ".gif": "gif", ".html": "jshtml"}


def _resolve_ffmpeg():
    """Locate an ffmpeg binary for matplotlib's FFMpegWriter.

    Resolution order: an explicit `rcParams["animation.ffmpeg_path"]`, then
    `ffmpeg` on PATH, then the binary shipped with `imageio_ffmpeg` (an
    optional dependency, imported here and only here so that plain
    `import eyepiece` never requires it).

    Every candidate is checked with `shutil.which` before it is returned,
    the configured one included: a stale rcParams path would otherwise be
    handed straight to `subprocess`, turning an actionable error into a bare
    FileNotFoundError from a failed launch.

    Returns:
        The path or bare name to run ffmpeg with.

    Raises:
        RuntimeError: If no ffmpeg can be found, naming both fixes.
    """
    configured = matplotlib.rcParams.get("animation.ffmpeg_path", "ffmpeg")
    if shutil.which(configured):
        return configured
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError(
            "mp4 export needs ffmpeg, and none was found: install the "
            "imageio-ffmpeg package (pip install imageio-ffmpeg), which "
            "ships a binary, or put ffmpeg on PATH -- a documentation "
            "builder can install one with the build apt_packages option "
            "(build.apt_packages: [ffmpeg])."
        ) from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _make_writer(path, fps, extra_ffmpeg_args):
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

    Returns:
        An unopened `MovieWriter`.

    Raises:
        ValueError: If the suffix is not one of .mp4, .gif, or .html.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".mp4":
        crop = ["-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2"]
        return FFMpegWriter(fps=fps, extra_args=crop + list(extra_ffmpeg_args or []))
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


def _rc_overrides(paths):
    """The rcParams overrides that must hold for the duration of a recording.

    `savefig.bbox` is disabled because a tight bbox lets the frame size
    change between frames, which a writer's fixed-size pipe cannot take.
    The ffmpeg path is resolved here, before any writer is opened, so a
    missing ffmpeg fails with an actionable error instead of a bare
    FileNotFoundError from a subprocess launch.
    """
    overrides = {"savefig.bbox": None}
    if any(Path(p).suffix.lower() == ".mp4" for p in paths):
        overrides["animation.ffmpeg_path"] = _resolve_ffmpeg()
    return overrides


class _Recorder:
    """Frame grabber yielded by `record`; fans one figure out to every sink."""

    def __init__(self, fig, writers):
        self.fig = fig
        self.writers = writers
        self._layout_frozen = False

    def frame(self):
        """Grab the figure exactly as it stands now, into every open sink.

        The figure's own facecolor is passed to each grab so a dark figure
        is not rasterized onto savefig's white default.

        The first call also freezes the figure's layout engine, once this
        frame's content has actually been drawn: see `record` for why the
        freeze waits here instead of happening at `record`'s entry.
        """
        facecolor = self.fig.get_facecolor()
        for writer in self.writers:
            writer.grab_frame(facecolor=facecolor)
        if not self._layout_frozen:
            self.fig.canvas.draw()
            self.fig.set_layout_engine("none")
            self._layout_frozen = True

    def hold(self, n):
        """Repeat the current figure for `n` frames, to dwell on a moment.

        Args:
            n: Number of frames to emit. The figure is not redrawn between
                them, so the cost is `n` rasterizations of the same state.
        """
        for _ in range(n):
            self.frame()


@contextmanager
def record(fig, *paths, fps=10, dpi=None, extra_ffmpeg_args=None):
    """Open every output file at once and record `fig` frame by frame.

    Use this when the frames come from a loop the caller already owns (a
    simulation stepping forward, say) rather than from a frame index. The
    figure is never cleared: mutate its artists between `frame()` calls, or
    call `ax.clear()` yourself for redraw-style animation.

    If `fig` has a constrained-layout engine, its layout is frozen the
    moment the first `frame()` call finishes drawing, and restored to
    whatever it was before this call once the `with` block exits -- on
    the exception path too, the same as the writer cleanup this context
    manager already guarantees. A figure with no layout engine, or one
    the caller already froze, records normally: freezing is a no-op on
    top of "none" already, and a figure that had no engine is left with
    no engine (the restore is skipped rather than replaying `None`
    through `set_layout_engine`, which would consult the autolayout
    rcParams and install an engine the figure never had).
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

    Yields:
        A recorder with `.frame()` and `.hold(n)`.

    Raises:
        ValueError: If a path has an unsupported suffix.
        RuntimeError: If an mp4 sink is requested and no ffmpeg is found.
    """
    saved_engine = fig.get_layout_engine()

    def restore_engine():
        # set_layout_engine(None) does not mean "no engine": matplotlib
        # reads figure.autolayout / figure.constrained_layout.use and
        # installs whichever they ask for, so replaying a saved None would
        # give a figure that had no engine one it never had.
        if saved_engine is not None:
            fig.set_layout_engine(saved_engine)

    with ExitStack() as stack:
        stack.callback(restore_engine)
        stack.enter_context(matplotlib.rc_context(_rc_overrides(paths)))
        writers = [_make_writer(p, fps, extra_ffmpeg_args) for p in paths]
        for writer, path in zip(writers, paths, strict=True):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(writer.saving(fig, str(path), _sink_dpi(path, dpi)))
        yield _Recorder(fig, writers)


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

    def save(self, *paths, dpi=None, fps=None):
        """Render every frame once, into every path.

        Args:
            *paths: Output paths, one per sink; see `record`.
            dpi: Rasterization dpi for every sink. None gives each sink the
                default for its suffix.
            fps: Playback rate override; None keeps the animation's own.

        Returns:
            The list of `pathlib.Path` written, in the order given.

        Raises:
            RuntimeError: If the frame source is an exhausted one-shot
                generator.
        """
        frames = self._frame_iter()
        rate = self.fps if fps is None else fps
        with record(self.fig, *paths, fps=rate, dpi=dpi) as rec:
            for ctx in frames:
                self.draw(self.fig, ctx)
                rec.frame()
        return [Path(p) for p in paths]

    def jshtml(self, dpi=100, fps=None):
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

        Returns:
            The HTML document as a string.

        Raises:
            RuntimeError: If the frame source is an exhausted one-shot
                generator.
        """
        frames = self._frame_iter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "animation.html"
            rate = self.fps if fps is None else fps
            with record(self.fig, path, fps=rate, dpi=dpi) as rec:
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
                generator, or if an mp4 is requested without ffmpeg.
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
