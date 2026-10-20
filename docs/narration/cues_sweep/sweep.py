"""A 20 s wavelength sweep example with its count heard through cue sounds.

Pitch is proportional to the count; one tick per pixel read; one two-note
sting at the click.

The clip: an imager reads its 4 pixels; the wavelength sweeps from 500 nm to
1000 nm while the number of lenslets across the field, which goes as the
square of the wavelength, counts from 4 to 16; an integral field
spectrograph then reads its 96 pixels in one second; a click lands the
count. The wavelength, the count and the reads are knobs of one plan, and
the cue events are read from the same frames the picture draws, so the
picture and the sound cannot disagree.

No voice: there is no cached speech for this sweep and nothing is
synthesized here, so the track carries one silent placeholder line, the cues
and the room-tone bed. With no voice to normalize, `mux` runs with
``loudness=None`` and the cues keep the levels they are synthesized at.

Run from the repository root (needs ffmpeg)::

    python docs/narration/cues_sweep/sweep.py /tmp/cues-sweep

It writes ``sweep-cues.mp4`` (cues on) and ``sweep-silent-cues.mp4`` (cues
off, room tone only), the same picture and length, and prints the measured
loudness of the cued file.
"""

import argparse
import json
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import hwostyle
import hwostyle.core
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep
from eyepiece import narration as nr
from eyepiece.narration._mux import _write_wav

FPS = 30
LAMBDA_NM = (500.0, 1000.0)
COUNT = (4.0, 16.0)
PIXELS = {"imager": 4, "ifs": 96}
GLOW_S = 0.15
FIGSIZE = (6.4, 3.6)
# Seconds of a 20 s clip. The holds and the sweep scale with the duration; the
# reads (1 s each, so 96 ticks stay apart) and the click do not.
SEGMENTS = (
    ("open", 1.5),
    ("read.imager", 1.0),
    ("look", 1.5),
    ("sweep", 8.0),
    ("pause", 1.5),
    ("read.ifs", 1.0),
)
CLICK_S = 0.6 + 1.5  # Timeline.click: the beat and its hold


@contextmanager
def _dark_style():
    """Keep the sweep's style local, including an initially inactive style."""
    names = (
        "_current_mode", "_current_family", "_activated", "palette", "cmaps", "roles"
    )  # fmt: skip
    saved = {name: getattr(hwostyle.core, name) for name in names}
    # hwostyle.dark() does not restore custom rcParams or the inactive state.
    with plt.rc_context():
        try:
            hwostyle.use("dark")
            yield
        finally:
            for name, value in saved.items():
                setattr(hwostyle.core, name, value)


def count_at(wavelength_nm):
    """Lenslets across the field: 4 at 500 nm, as the square of the wavelength."""
    return COUNT[0] * (wavelength_nm / LAMBDA_NM[0]) ** 2


def make_plan(duration_s=20.0):
    """The sweep's plan, `duration_s` long (at least 5 s)."""
    if duration_s < 5.0:
        raise ValueError(f"duration_s must be at least 5 s, got {duration_s}")
    fixed = sum(s for n, s in SEGMENTS if n.startswith("read.")) + CLICK_S
    scaled = sum(s for n, s in SEGMENTS if not n.startswith("read."))
    scale = (duration_s - fixed) / (20.0 - fixed)
    tl = ep.Timeline(
        {"wavelength": LAMBDA_NM[0], "read.imager": 0.0, "read.ifs": 0.0},
        tempo=ep.Tempo(factor=1.0),
    )
    for name, seconds in SEGMENTS:
        if name.startswith("read."):
            tl.run(name, seconds, to=float(PIXELS[name[5:]]))
        elif name == "sweep":
            tl.move("wavelength", seconds * scale, to=LAMBDA_NM[1])
        else:
            tl.hold(seconds * scale, name=name)
    tl.click("count")
    tl.hold(duration_s - fixed - scaled * scale, name="rest")
    return tl.at_fps(FPS)


def _crossings(plan, knob):
    """Times at which a read knob passes 1, 2, ... (a pixel lights at each)."""
    values = np.array([plan.value(k, knob) for k in range(plan.n_frames)])
    out = []
    for i in range(1, round(values.max()) + 1):
        k = int(np.flatnonzero(values >= i)[0])
        v0, v1 = values[k - 1], values[k]
        out.append((k - 1 + (i - v0) / (v1 - v0)) / plan.fps)
    return out


def cue_events(plan):
    """The cue events, read from the plan's own numbers."""
    span = plan.spans["wavelength"]
    frames = range(span.k0, span.k1 + 1)
    return nr.CueEvents(
        count_times=tuple(k / plan.fps for k in frames),
        count_values=tuple(count_at(plan.value(k, "wavelength")) for k in frames),
        reference=COUNT[0],
        tick_times=tuple(
            _crossings(plan, "read.imager") + _crossings(plan, "read.ifs")
        ),
        sting_s=plan.spans["click.count"].k1 / plan.fps,
        bell_at=2.0 * COUNT[0],
    )


class Scene:
    """Count against wavelength above, the two pixel rows below."""

    @_dark_style()
    def __init__(self, plan, dpi):
        """Build the finished figure at `dpi` and freeze its layout."""
        self.plan = plan
        self.fig = plt.figure(figsize=FIGSIZE, dpi=dpi)
        grid = self.fig.add_gridspec(3, 1, height_ratios=(3.0, 0.3, 0.3), hspace=1.5)
        ax = self.fig.add_subplot(grid[0])
        lam = np.linspace(*LAMBDA_NM, 200)
        ax.plot(lam, count_at(lam), color=ep.neutral(0.35), lw=1.0)
        (self.trace,) = ax.plot([], [], color="C0", lw=2.0)
        ax.set(
            xlim=LAMBDA_NM, ylim=(0, 18), xlabel="wavelength [nm]", ylabel="lenslets"
        )
        self.ax = ax
        self.rows = {}
        for row, (name, n) in zip((1, 2), PIXELS.items(), strict=True):
            pax = self.fig.add_subplot(grid[row])
            image = pax.imshow(
                np.zeros((1, PIXELS["ifs"])), cmap="gray", vmin=0.0, vmax=1.0,
                aspect="auto", interpolation="nearest",
                extent=(0, PIXELS["ifs"], 0, 1),
            )  # fmt: skip
            pax.set(xlim=(0, PIXELS["ifs"]), yticks=[], xticks=[])
            pax.set_title(f"{name}: {n} pixels", fontsize="small", loc="left")
            self.rows[name] = (image, n, _crossings(plan, f"read.{name}"))
        self.fig.canvas.draw()
        self.fig.set_layout_engine("none")
        self.lam = ep.readout(
            ep.Anchor.from_data(ax, (560.0, 15.5)), LAMBDA_NM[0],
            fmt=".0f", prefix="wavelength = ", unit="nm", digits=4,
        )  # fmt: skip
        self.count = ep.readout(
            ep.Anchor.from_data(ax, (560.0, 12.5)), COUNT[0],
            fmt=".1f", prefix="lenslets = ", digits=2,
        )  # fmt: skip
        self.click = ep.click(
            self.trace,
            dot=ep.Anchor.from_data(ax, (LAMBDA_NM[1], COUNT[1])),
            readout=self.count,
        )

    def draw(self, k):
        """Set every artist from frame `k` of the plan."""
        plan, t = self.plan, k / self.plan.fps
        lam = plan.value(k, "wavelength")
        path = np.linspace(LAMBDA_NM[0], lam, 100)
        self.trace.set_data(path, count_at(path))
        self.lam.set(lam)
        self.lam.update(1.0)
        self.count.set(count_at(lam))
        self.count.update(1.0)
        for name, (image, n, times) in self.rows.items():
            row = np.zeros(PIXELS["ifs"])
            width = PIXELS["ifs"] // n
            lit = int(plan.value(k, f"read.{name}") + 1e-9)
            for i, ti in enumerate(times[:lit]):
                if ti <= t + 1e-9:
                    glow = max(0.0, 1.0 - (t - ti) / GLOW_S)
                    row[i * width : (i + 1) * width] = 0.55 + 0.45 * glow
            image.set_data(row[None, :])
        self.click.update(plan.value(k, "click.count"))


@_dark_style()
def _render(plan, path, dpi):
    scene = Scene(plan, dpi)
    with ep.record(scene.fig, path, fps=plan.fps, plan=plan, dpi=dpi) as rec:
        for k in range(plan.n_frames):
            scene.draw(k)
            rec.frame()
    plt.close(scene.fig)
    return path


def _audio(plan, path, cues):
    """The track: one silent placeholder line, plus the cues when given."""
    script = nr.Script((nr.Line("sweep", "A count heard as pitch."),))
    narration = nr.Narration.estimated(script)
    tail = plan.n_frames / plan.fps - narration.total_s
    beat = narration.beat("sweep")
    n = round((beat.speech_end_s - beat.speech_start_s) * 48000)
    lines = path.parent / "lines"
    _write_wav(
        lines / f"{script.line_key(script.line('sweep'))}.wav",
        np.zeros(n, dtype="<i2"),
        48000,
    )
    return nr.audio_track(
        narration, lines, path, tail_s=tail, rate=48000,
        cues=cue_events(plan) if cues else None,
    )  # fmt: skip


def build(path, *, duration_s=20.0, cues=True, dpi=150):
    """Write the sweep MP4 to `path`: cues on, or room tone only."""
    path = Path(path)
    plan = make_plan(duration_s)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        silent = _render(plan, tmp / "silent.mp4", dpi)
        track = _audio(plan, tmp / "track.wav", cues)
        nr.mux(silent, track, path, loudness=None)
    return path


def loudness(path):
    """Integrated loudness (LUFS) and true peak (dBTP) of a file's audio."""
    done = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    stats = json.loads(re.findall(r"\{[^{}]*\}", done.stderr)[-1])
    return float(stats["input_i"]), float(stats["input_tp"])


def main():
    """Write both MP4s and print the cued file's loudness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cued = build(args.output_dir / "sweep-cues.mp4")
    build(args.output_dir / "sweep-silent-cues.mp4", cues=False)
    lufs, tp = loudness(cued)
    events = cue_events(make_plan())
    print(f"{cued.name}: {cued.stat().st_size} bytes")
    print(f"integrated {lufs:.1f} LUFS, true peak {tp:.1f} dBTP")
    print(f"ticks {len(events.tick_times)}, count {events.count_values[0]:.1f} to "
          f"{events.count_values[-1]:.1f}, pitch "
          f"{220.0 * events.count_values[0] / events.reference:.1f} to "
          f"{220.0 * events.count_values[-1] / events.reference:.1f} Hz")  # fmt: skip
    print(f"sting ends at {events.sting_s:.3f} s")


if __name__ == "__main__":
    main()
