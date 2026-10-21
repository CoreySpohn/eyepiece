"""A scrubber page: a clip's frames under a slider, in one offline HTML file.

`to_scrubber` renders chosen frames of a plan through the same path `record`
writes its stills by, embeds them as data URIs, and writes a page whose
slider is one knob of the plan. The page loads nothing from anywhere, so it
works where video does not play and where the network is closed, and two
renders of the same inputs give the same bytes.
"""

import base64
import difflib
import html
import io
import itertools
import json
import math
import tempfile
from pathlib import Path

import matplotlib.image as mpimg

from eyepiece._motion import frame_count
from eyepiece.anim import record

DEFAULT_MAX_BYTES = 8_000_000
_FORMATS = ("png", "jpeg")

_STYLE = """\
body{margin:0;padding:16px;background:#fff;color:#111;font:15px/1.4 sans-serif}
@media (prefers-color-scheme:dark){body{background:#111;color:#eee}}
main{max-width:100%}
img{display:block;max-width:100%;height:auto}
.knob{display:flex;gap:12px;align-items:center;margin:12px 0}
.knob input{flex:1;min-width:0}
.chapters{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.transcript li{opacity:.55}
.transcript li.active{opacity:1;font-weight:bold}"""

_SCRIPT = """\
(function(){
var d=JSON.parse(document.getElementById("scrubber-data").textContent);
var img=document.getElementById("frame");
var knob=document.getElementById("knob");
var out=document.getElementById("value");
var lines=document.querySelectorAll(".transcript li");
d.strip[d.start]=img.src;
function show(i){
img.src=d.strip[i];
out.textContent=d.labels[i];
for(var j=0;j<lines.length;j++){
lines[j].className=(j===d.lines[i])?"active":"";
}
}
knob.addEventListener("input",function(){show(Number(knob.value));});
var buttons=document.querySelectorAll(".chapters button");
for(var b=0;b<buttons.length;b++){
buttons[b].addEventListener("click",function(e){
knob.value=e.currentTarget.getAttribute("data-sample");
show(Number(knob.value));
knob.focus();
});
}
show(d.start);
})();"""


def to_scrubber(
    fig,
    draw,
    plan,
    knob,
    samples,
    path=None,
    *,
    dpi=100,
    format="png",
    quality=90,
    max_bytes=DEFAULT_MAX_BYTES,
    unit="",
    title=None,
):
    """Write a self-contained HTML page whose slider scrubs one knob of a clip.

    Each sample is a value of `knob`. The frame shown for a sample is the
    plan's frame whose knob value is nearest it; when several frames tie, the
    latest wins, so the knob's final value shows the clip's settled end. Each
    such frame is drawn by calling ``draw(k)`` with its frame index, as the
    loop that feeds `record` does, and is rasterized by `record`'s still
    writer at `dpi`. A frame of the page is therefore the same PNG that
    ``record(..., stills=...)`` writes for that frame at the same dpi, and
    the slider and the clip cannot disagree. `draw` must set every artist
    from the frame index alone, since only the sampled frames are drawn.

    The page needs no network and no other file: the frames are base64 data
    URIs in one script block, one image is swapped by a range input whose
    steps are the samples (by index, ``0`` to ``len(samples) - 1``), and the
    knob's name, `unit` and the sample's value are shown beside the slider.
    The left and right arrow keys step one sample while the slider has
    focus. The image visible before any script runs is the sample nearest
    the clip's last frame (its rest state), so a reader with scripts blocked
    still sees the finished figure; include the knob's final value among the
    samples to make that image the rest state itself.

    Chapters: each span a narration cue of its own line targets (the span
    named by the cue's target, or its ``indicate.`` or ``circumscribe.``
    span) becomes a button that jumps to the sample nearest the cue's frame;
    when no span has a cue, every span of the plan becomes a button that
    jumps to the sample nearest the span's first frame. With a narration,
    the spoken text of each line is listed under the figure and the line
    running at the slider's frame is highlighted; a plan with no narration
    has no transcript.

    Example::

        page = ep.to_scrubber(fig, draw, plan, "u", [0.0, 0.25, 0.5, 0.75, 1.0],
                              "sweep.html", dpi=80)

    Args:
        fig: The Figure that `draw` updates.
        draw: The per-frame function, ``draw(k)`` for a frame index ``k``.
        plan: A `Plan` from `Timeline.at_fps`.
        knob: The name of one of ``plan.knobs``, or ``"t"`` for the time of
            the frame in seconds.
        samples: Knob values, strictly increasing and inside the range the
            knob takes over the clip.
        path: Also write the page here, as UTF-8, when given.
        dpi: Rasterization dpi of every frame.
        format: ``"png"`` (lossless, the default) or ``"jpeg"``.
        quality: JPEG quality, 1 to 95; ignored for PNG.
        max_bytes: The largest page allowed, in bytes.
        unit: The knob's unit, shown after its value.
        title: The page title; None uses the knob's name.

    Returns:
        The page as a string.

    Raises:
        KeyError: If `knob` is not a knob of the plan or ``"t"``, naming near
            matches.
        ValueError: If `samples` is empty, not strictly increasing, not
            finite, or outside the knob's range; if the knob is not numeric;
            if `format` or `quality` is not allowed; or if the page would be
            larger than `max_bytes`, naming its size.
    """
    if format not in _FORMATS:
        raise ValueError(f"format is one of {_FORMATS}, got {format!r}")
    if format == "jpeg" and not (isinstance(quality, int) and 1 <= quality <= 95):
        raise ValueError(f"quality is an int from 1 to 95, got {quality!r}")
    values = _knob_values(plan, knob)
    samples = _check_samples(samples, values, knob)
    frames = [_nearest_frame(values, s) for s in samples]
    last = plan.n_frames - 1
    start = min(range(len(frames)), key=lambda i: (abs(frames[i] - last), -i))
    images = _render(fig, draw, sorted(set(frames)), dpi)
    if format == "jpeg":
        images = {k: _to_jpeg(png, quality) for k, png in images.items()}
    uris = [_data_uri(images[k], format) for k in frames]
    chapters = _chapters(plan, frames)
    transcript, active = _transcript(plan, frames)
    labels = [_label(knob, s, unit) for s in samples]
    page = _page(
        title if title is not None else knob,
        f"{knob} ({unit})" if unit else knob,
        uris,
        start,
        labels,
        chapters,
        transcript,
        active,
    )
    size = len(page.encode("utf-8"))
    if size > max_bytes:
        raise ValueError(
            f"the page would be {size} bytes, over max_bytes={max_bytes}; use "
            "fewer samples, a lower dpi, or format='jpeg'"
        )
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page, encoding="utf-8")
    return page


def _knob_values(plan, knob):
    """The knob's value on every frame of the plan."""
    if knob == "t":
        return [k / plan.fps for k in range(plan.n_frames)]
    if knob not in plan.knobs:
        names = [*plan.knobs, "t"]
        near = difflib.get_close_matches(knob, names, n=3)
        hint = f"; did you mean {near}" if near else ""
        raise KeyError(f"no knob named {knob!r}{hint}; knobs are {sorted(names)}")
    values = [plan.value(k, knob) for k in range(plan.n_frames)]
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        raise ValueError(f"knob {knob!r} is not numeric, so it cannot be a slider")
    return [float(v) for v in values]


def _check_samples(samples, values, knob):
    """The samples as floats, refused when empty, unordered or out of range."""
    samples = [float(s) for s in samples]
    if not samples:
        raise ValueError("samples is empty; give at least one knob value")
    if not all(math.isfinite(s) for s in samples):
        raise ValueError(f"samples must be finite, got {samples}")
    if any(b <= a for a, b in itertools.pairwise(samples)):
        raise ValueError(f"samples must be strictly increasing, got {samples}")
    lo, hi = min(values), max(values)
    tol = 1e-9 * max(1.0, abs(lo), abs(hi))
    outside = [s for s in samples if s < lo - tol or s > hi + tol]
    if outside:
        raise ValueError(
            f"samples {outside} are outside the range of {knob!r} over the clip, "
            f"{lo:g} to {hi:g}"
        )
    return samples


def _nearest_frame(values, sample):
    """The frame whose value is nearest `sample`, the latest on a tie."""
    return min(range(len(values)), key=lambda k: (abs(values[k] - sample), -k))


def _render(fig, draw, frames, dpi):
    """PNG bytes of each frame, written by `record`'s still writer."""
    with tempfile.TemporaryDirectory() as tmp:
        stills = {Path(tmp) / f"{i}.png": float(i) for i in range(len(frames))}
        saved = fig.get_dpi()
        fig.set_dpi(dpi)
        try:
            with record(fig, fps=1, stills=stills, stills_dpi=dpi) as rec:
                for k in frames:
                    draw(k)
                    rec.frame()
        finally:
            fig.set_dpi(saved)
        return {k: (Path(tmp) / f"{i}.png").read_bytes() for i, k in enumerate(frames)}


def _to_jpeg(png, quality):
    """Re-encode one PNG frame as a JPEG of the given quality."""
    rgba = mpimg.imread(io.BytesIO(png), format="png")
    buf = io.BytesIO()
    mpimg.imsave(buf, rgba[..., :3], format="jpeg", pil_kwargs={"quality": quality})
    return buf.getvalue()


def _data_uri(data, format):
    return f"data:image/{format};base64,{base64.b64encode(data).decode('ascii')}"


def _nearest_sample(frames, k):
    """Index of the sample whose frame is nearest frame `k`, the first on a tie."""
    return min(range(len(frames)), key=lambda i: (abs(frames[i] - k), i))


def _chapters(plan, frames):
    """(label, sample index) per chapter button, in the plan's span order."""
    cues = plan.narration.cues() if plan.narration is not None else []
    out = []
    for name, s in plan.spans.items():
        hits = [
            frame_count(c.at_s, plan.fps)
            for c in cues
            if c.line == s.section and name.split(".", 1)[-1] == c.target
        ]
        if hits:
            out.append((name, _nearest_sample(frames, min(hits[0], plan.n_frames - 1))))
    if out:
        return out
    return [(name, _nearest_sample(frames, s.k0)) for name, s in plan.spans.items()]


def _transcript(plan, frames):
    """The narration's spoken lines, and the line running at each sample."""
    if plan.narration is None:
        return [], [-1] * len(frames)
    lines, spans = [], []
    for line in plan.narration.script.lines:
        text = line.display or line.spoken
        if not text:
            continue
        beat = plan.narration.beat(line.key)
        spans.append(
            (frame_count(beat.start_s, plan.fps), frame_count(beat.end_s, plan.fps))
        )
        lines.append(text)
    active = []
    for k in frames:
        hit = [i for i, (a, b) in enumerate(spans) if a <= k < b]
        active.append(hit[0] if hit else -1)
    return lines, active


def _label(knob, sample, unit):
    value = f"{sample:.4g}"
    return f"{knob} = {value} {unit}".rstrip()


def _page(title, slider, uris, start, labels, chapters, transcript, active):
    """The HTML text of the page."""
    esc = html.escape
    strip = [("" if i == start else u) for i, u in enumerate(uris)]
    data = json.dumps(
        {"strip": strip, "start": start, "labels": labels, "lines": active},
        separators=(",", ":"),
    ).replace("</", "<\\/")
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{esc(title)}</title>",
        f"<style>\n{_STYLE}\n</style>",
        "</head>",
        "<body>",
        "<main>",
        f'<img id="frame" alt="{esc(labels[start])}" src="{uris[start]}">',
        '<div class="knob">',
        f'<input id="knob" type="range" min="0" max="{len(uris) - 1}" step="1" '
        f'value="{start}" aria-label="{esc(slider)}">',
        f'<output id="value">{esc(labels[start])}</output>',
        "</div>",
    ]
    if chapters:
        parts.append('<nav class="chapters">')
        parts += [
            f'<button type="button" data-sample="{i}">{esc(name)}</button>'
            for name, i in chapters
        ]
        parts.append("</nav>")
    if transcript:
        parts.append('<ol class="transcript">')
        mark = ' class="active"'
        parts += [
            f"<li{mark if j == active[start] else ''}>{esc(text)}</li>"
            for j, text in enumerate(transcript)
        ]
        parts.append("</ol>")
    parts += [
        f'<script type="application/json" id="scrubber-data">{data}</script>',
        f"<script>\n{_SCRIPT}\n</script>",
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts) + "\n"
