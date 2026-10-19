"""Exchangeable draws played one after another, ending on their tally.

A posterior, an ensemble or a bootstrap is a set of draws that are all
equally likely, and no one of them is the answer. `hop` shows that by
playing the draws at a steady rate: the current draw is drawn in full, the
draws before it stay behind as ghosts that dim with age, and a tally of a
quantity over the draws so far converges, with its interval, on the value
the whole set gives. The arrays stay in NumPy: this is the I/O edge of a
clip, not a model.
"""

import dataclasses
import itertools
from collections.abc import Callable
from statistics import NormalDist

import numpy as np

from eyepiece._result import MotionContract, PlotResult

# Fastest draw rate the eye follows as separate draws: past about eight a
# second, successive draws average into a band and the run reads as a blur.
MAX_RATE = 8.0

# Tolerance on the knob at a window boundary, so a knob sampled at exactly
# ``i / n`` (up to rounding) shows draw ``i`` rather than the next one.
_WINDOW_TOL = 1e-9

_KINDS = ("fraction", "mean")


@dataclasses.dataclass(frozen=True, eq=False)
class _HopResult(PlotResult):
    """A `PlotResult` that also reports the draw shown and its tally."""

    draw: Callable | None = None
    steps: Callable | None = None
    tally: Callable | None = None
    duration: float = 0.0
    n: int = 0
    tally_artists: dict = dataclasses.field(default_factory=dict)


def _interval(kind, values, k, z):
    """The estimate and interval of the tally over the first `k` draws."""
    head = values[:k]
    if kind == "fraction":
        # Wilson score interval: it stays inside [0, 1] and is honest at a
        # small count or a fraction near 0 or 1, where the normal one is not.
        p = float(head.mean())
        denom = 1.0 + z * z / k
        center = (p + z * z / (2.0 * k)) / denom
        half = z * np.sqrt(p * (1.0 - p) / k + z * z / (4.0 * k * k)) / denom
        return p, center - half, center + half
    mean = float(head.mean())
    if k < 2:
        return mean, np.nan, np.nan
    half = z * float(head.std(ddof=1)) / np.sqrt(k)
    return mean, mean - half, mean + half


def hop(
    draws,
    *,
    ax=None,
    rate=4.0,
    ghosts=6,
    ghost_alpha=0.45,
    color=None,
    ghost_color=None,
    tally=None,
    tally_ax=None,
    kind="fraction",
    level=0.95,
    ref=None,
    ref_label=None,
    tally_fmt=None,
    tally_color=None,
    tally_kw=None,
    text_kw=None,
    counter_at=None,
    line_kw=None,
    zorder=None,
) -> PlotResult:
    """Play exchangeable draws at a steady rate, leaving ghosts and a tally.

    The mark of uncertainty shown as draws (synthesis family six): each draw
    of a set (an orbit from a posterior, a curve from an ensemble) is shown
    in turn as one line, the current draw in full color and the draws before
    it as ghosts that dim with age, so the eye sees the spread the set
    covers rather than a single answer. The draw index is on screen. A tally
    of one quantity over the draws so far (a fraction of draws that meet a
    condition, or a mean) runs beside the draws with its interval at the
    current count, narrowing as draws accumulate, so the clip ends on what
    the whole set says and how well it says it.

    Motion contract (exchangeable draws): nothing moves continuously. At
    each draw the lead line takes the new draw, every ghost ages by one
    draw, the counter steps by one and the tally takes one more draw; the
    picture is otherwise still. The draw index is the knob: ``update(u)``
    gives each draw an equal window of `u`, draw ``i`` (from 1) owning
    ``((i - 1) / n, i / n]`` and draw 1 also ``u = 0``, so a clip that plays
    the run over `duration` seconds shows `rate` draws a second (drive the
    knob linearly in time, as a `Timeline` run does: an eased knob plays the
    draws at an uneven rate). The cuts
    are by design: declare the frames ``steps`` returns to
    `eyepiece.testing.check_clip` (or their times to ``motion_faults``). The
    rest state, ``u = 1``, shows the last draw, its ghosts and the final
    tally, and is the still of the run.

    Ghosts are the `ghosts` draws before the current one, never more, so the
    picture does not fill in as the run goes on; the draw just before the
    current one is drawn at `ghost_alpha`, and each older one dimmer, down
    to ``ghost_alpha / ghosts``.

    Args:
        draws: The draws, an array shaped ``(n, m, 2)``: `n` draws of `m`
            ``(x, y)`` points in data coordinates of `ax`, drawn as a line
            (or ``(n, 2)``, one point per draw, drawn as a marker). The draws
            must be exchangeable (any order is equally likely), so the order
            given is the order played.
        ax: Axes for the draws, or None for a new figure. Its limits are set
            once from every draw, so they never change during the run.
        rate: Draws per second, in (0, 8]. Three to six reads as separate
            draws; past eight, draws average into a band.
        ghosts: How many earlier draws stay on screen, a non-negative
            integer.
        ghost_alpha: Opacity of the newest ghost, in (0, 1].
        color: Color of the current draw; None uses the first palette color.
        ghost_color: Color of the ghosts; None uses `color`.
        tally: One value per draw for the tally, or None for no tally. For
            ``kind="fraction"`` each value is True or False (or 1 or 0): does
            this draw meet the condition.
        tally_ax: Axes for the tally (a `eyepiece.stats.convergence` panel
            against the draw count). Required with `tally`.
        kind: ``"fraction"`` for the fraction of draws that meet the
            condition, with a Wilson score interval, or ``"mean"`` for the
            mean of the values, with a normal interval (none at one draw).
        level: Confidence level of the interval, in (0, 1).
        ref: A reference value for the tally panel, or None.
        ref_label: Label of the reference.
        tally_fmt: A callable ``tally_fmt(estimate, lo, hi, k)`` returning
            the text written just above the tally panel, at its left edge, at
            each count, or None for no text.
        tally_color: Color of the tally line, interval and text; None uses
            `color`.
        tally_kw: Extra kwargs for the `eyepiece.stats.convergence` panel,
            applied last (``label_side="below"`` puts the reference label
            under its line).
        text_kw: Extra kwargs for the tally text, applied last.
        counter_at: Where the draw counter (``draw 7 of 40``) is, an
            `eyepiece.Anchor` (its baseline's left end); None puts it at the
            upper left of `ax`.
        line_kw: Extra kwargs for the lead and the ghost lines, applied
            last.
        zorder: Drawing order of the lead line; ghosts draw just below it,
            older ones lower.

    Returns:
        A `PlotResult` on `ax` with ``artists["line"]`` (the current draw),
        ``artists["lines"]`` (the ghosts, newest first) and ``artists["text"]``
        (the counter's parts). With a tally, its ``tally_artists`` holds the
        tally panel's changing artists in the same vocabulary: ``"line"``
        (the running estimate), ``"lines"`` (the interval, a list of one),
        ``"scatter"`` (the estimate at the current count) and ``"text"``
        (when `tally_fmt` is given). ``contract.moves`` declares all of them,
        ``contract.may_overlap`` the pairs that cross by design (any two
        draws; the interval, its point and the running line), and
        ``update(u, level=1.0)`` poses the run, where ``level`` dims it all.
        Its ``draw(u=None)`` returns the index (from 1) of the draw shown
        (at the last `u` set when None); ``steps(knob)``, given the knob's
        value `u` at every frame of a clip, returns the indices of the
        frames that show a new draw; ``tally(k=None)`` returns ``(estimate,
        lo, hi)`` over the first `k` draws (the draws shown when None);
        ``duration`` is the run's length in seconds at `rate`; and ``n`` is
        the number of draws.

    Raises:
        ValueError: If `draws` is not shaped ``(n, m, 2)`` or ``(n, 2)`` or
            holds a non-finite value, `rate`, `ghosts`, `ghost_alpha` or
            `level` is out of range, `kind` is unknown, `tally` does not
            have one value per draw (or is not 0 or 1 for a fraction), or
            `tally` is given without `tally_ax`.

    Example::

        mark = ep.hop(orbits, ax=sky, rate=4.0, tally=within, tally_ax=panel)
        plan = ep.FramePlan.from_duration(mark.duration + 1.0)
        for t in plan.times:
            mark.update(min(t / mark.duration, 1.0))
            rec.frame()
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    from eyepiece import _style
    from eyepiece.attention import Anchor, readout
    from eyepiece.stats import convergence, interval_points

    data = np.asarray(draws, dtype=float)
    if data.ndim == 2 and data.shape[-1] == 2:
        data = data[:, None, :]
    if data.ndim != 3 or data.shape[-1] != 2 or not data.shape[0] or not data.shape[1]:
        raise ValueError(
            f"hop draws must be shaped (n, m, 2) or (n, 2), got {np.shape(draws)}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("hop draws must be finite")
    n, m = data.shape[:2]
    rate = float(rate)
    if not 0.0 < rate <= MAX_RATE:
        raise ValueError(
            f"hop rate must be in (0, {MAX_RATE:g}] draws per second, got {rate:g}; "
            "past eight a second the draws average into a band"
        )
    if int(ghosts) != ghosts or ghosts < 0:
        raise ValueError(f"hop ghosts must be a non-negative integer, got {ghosts}")
    ghosts = int(ghosts)
    ghost_alpha = float(ghost_alpha)
    if not 0.0 < ghost_alpha <= 1.0:
        raise ValueError(f"hop ghost_alpha must be in (0, 1], got {ghost_alpha}")
    if kind not in _KINDS:
        raise ValueError(f"unknown hop kind: {kind!r}; known: {list(_KINDS)}")
    level_ci = float(level)
    if not 0.0 < level_ci < 1.0:
        raise ValueError(f"hop level must be in (0, 1), got {level}")
    values = None
    if tally is not None:
        if tally_ax is None:
            raise ValueError("hop tally= needs tally_ax= for its panel")
        values = np.asarray(tally, dtype=float).ravel()
        if values.size != n:
            raise ValueError(f"hop tally has {values.size} values for {n} draws")
        if not np.all(np.isfinite(values)):
            raise ValueError("hop tally values must be finite")
        if kind == "fraction" and not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError("hop tally for a fraction must be True or False")
    z = NormalDist().inv_cdf(0.5 + 0.5 * level_ci)

    if ax is None:
        _, ax = plt.subplots()
    lead_color = _style.color(0, color)
    shade = lead_color if ghost_color is None else ghost_color
    point = m == 1
    style = ({"ls": "none", "marker": "o"} if point else {"marker": "none"}) | dict(
        line_kw or {}
    )
    (lead,) = ax.plot(*data[-1].T, color=lead_color, **style)
    if zorder is not None:
        lead.set_zorder(zorder)
    base_z = lead.get_zorder()
    pool = []
    for age in range(1, ghosts + 1):
        (ghost,) = ax.plot(*data[-1].T, color=shade, **style)
        ghost.set_zorder(base_z - 0.01 * age)
        pool.append(ghost)
    flat = data.reshape(-1, 2)
    ax.update_datalim(flat)
    ax.autoscale_view()
    ax.set_xlim(*ax.get_xlim())
    ax.set_ylim(*ax.get_ylim())

    if counter_at is None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        counter_at = Anchor.from_data(
            ax, (x0 + 0.04 * (x1 - x0), y1 - 0.08 * (y1 - y0))
        )
    counter = readout(
        counter_at,
        n,
        fmt="d",
        prefix="draw ",
        unit=f"of {n}",
        digits=len(str(n)),
        signed=False,
    )
    texts = list(counter.artists["text"])

    tally_parts = {}
    if values is not None:
        tone = lead_color if tally_color is None else tally_color
        board = convergence(
            values,
            ax=tally_ax,
            refs=ref,
            ref_labels=None if ref_label is None else [ref_label],
            show_samples=kind == "mean",
            running="mean",
            color=tone,
            **(tally_kw or {}),
        )
        spans = np.array([_interval(kind, values, k, z) for k in range(1, n + 1)])
        est, lo, hi = spans[-1]
        bar = interval_points(
            [n],
            [est],
            [lo],
            [hi],
            ax=tally_ax,
            color=tone,
            hollow_zero=False,
            interval_kw={"lw": 1.5 * float(plt.rcParams["lines.linewidth"])},
        )
        # The panel's limits clear the point at the first and the last draw,
        # and the ends of every interval, by the point's radius plus a margin,
        # whatever the panel's size: the pads are solved in points.
        reach = 0.5 * float(np.sqrt(bar.artists["scatter"].get_sizes().max())) + 2.0
        box = tally_ax.get_position()
        width = box.width * tally_ax.figure.get_figwidth() * 72.0
        height = box.height * tally_ax.figure.get_figheight() * 72.0
        x0, x1 = tally_ax.get_xlim()
        dx = max(1.0, reach * max(n - 1, 1) / max(width - 2.0 * reach, 1.0))
        tally_ax.set_xlim(min(x0, 0.0, 1.0 - dx), max(x1, n + 1.0, n + dx))
        lo_all, hi_all = np.nanmin(spans[:, 1]), np.nanmax(spans[:, 2])
        y0, y1 = tally_ax.get_ylim()
        if np.isfinite(lo_all) and np.isfinite(hi_all):
            low, high = min(y0, lo_all), max(y1, hi_all)
            share = max(0.04, reach / max(height - 2.0 * reach, 1.0))
            pad = share * (high - low)
            tally_ax.set_ylim(min(y0, lo_all - pad), max(y1, hi_all + pad))
        tally_parts = {
            "board": board,
            "scatter": bar.artists["scatter"],
            "bar": bar.artists["lines"][0],
            "spans": spans,
        }
        if tally_fmt is not None:
            words = tally_ax.text(
                0.0,
                1.0,
                tally_fmt(est, lo, hi, n),
                transform=tally_ax.transAxes,
                ha="left",
                va="bottom",
                color=tone,
                zorder=8,
                **(text_kw or {}),
            )
            # Above the panel, a line's height clear of its top edge, so the
            # text never covers an early interval, which can span the panel.
            words.set_transform(
                tally_ax.transAxes
                + ScaledTranslation(
                    0.0,
                    0.4 * words.get_fontsize() / 72.0,
                    tally_ax.figure.dpi_scale_trans,
                )
            )
            tally_parts["text"] = words

    state = {"u": 1.0}

    def where(u):
        x = u * n - _WINDOW_TOL
        return min(max(int(np.ceil(x)), 1), n)

    def draw(u=None):
        u = state["u"] if u is None else min(max(float(u), 0.0), 1.0)
        return where(u)

    def tally_at(k=None):
        if values is None:
            raise ValueError("hop has no tally")
        k = draw() if k is None else min(max(int(k), 1), n)
        est, lo, hi = tally_parts["spans"][k - 1]
        return float(est), float(lo), float(hi)

    def update(u, level=1.0):
        u = min(max(float(u), 0.0), 1.0)
        level = min(max(float(level), 0.0), 1.0)
        state["u"] = u
        k = where(u)
        lead.set_data(*data[k - 1].T)
        lead.set_alpha(None if level == 1.0 else level)
        for age, ghost in enumerate(pool, start=1):
            i = k - age
            if i < 1:
                ghost.set_visible(False)
                continue
            ghost.set_data(*data[i - 1].T)
            ghost.set_alpha(level * ghost_alpha * (1.0 - (age - 1) / ghosts))
            ghost.set_visible(True)
        counter.show(k)
        counter.update(1.0, level=level)
        if values is not None:
            tally_parts["board"].update(k)
            est, lo, hi = tally_parts["spans"][k - 1]
            tally_parts["scatter"].set_offsets([[k, est]])
            bar = tally_parts["bar"]
            bar.set_data([k, k], [lo, hi])
            bar.set_visible(bool(np.isfinite(lo)))
            running = tally_parts["board"].artists["line"]
            for artist in (tally_parts["scatter"], bar, running):
                artist.set_alpha(None if level == 1.0 else level)
            if "text" in tally_parts:
                tally_parts["text"].set_text(tally_fmt(est, lo, hi, k))
                tally_parts["text"].set_alpha(None if level == 1.0 else level)

    def steps(knob):
        shown = [draw(u) for u in knob]
        return [i for i in range(1, len(shown)) if shown[i] != shown[i - 1]]

    artists = {"line": lead, "lines": pool, "text": texts}
    board_artists = {}
    if values is not None:
        board_artists = {
            "line": tally_parts["board"].artists["line"],
            "lines": [tally_parts["bar"]],
            "scatter": tally_parts["scatter"],
        }
        if "text" in tally_parts:
            board_artists["text"] = tally_parts["text"]
    moving = [lead, *pool, *texts]
    for value in board_artists.values():
        moving += value if isinstance(value, list) else [value]
    # Draws are exchangeable, so they cross one another by nature, and the
    # interval runs through its estimate and its running line.
    overlap = list(itertools.combinations([lead, *pool], 2))
    if board_artists:
        overlap += itertools.combinations(
            [board_artists["line"], *board_artists["lines"], board_artists["scatter"]],
            2,
        )
    update(1.0)
    return _HopResult(
        ax=ax,
        artists=artists,
        update=update,
        contract=MotionContract(moves=tuple(moving), may_overlap=tuple(overlap)),
        draw=draw,
        steps=steps,
        tally=tally_at,
        duration=n / rate,
        n=n,
        tally_artists=board_artists,
    )
