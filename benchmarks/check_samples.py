"""Benchmark clip checks on 2100 artists in 300 mostly absent Stage groups.

Run with the checkout on PYTHONPATH, for example::

    python benchmarks/check_samples.py --out .codex-bench/synthetic

No external data or network is used. The report contains every finding,
including artist identities, times, geometry and messages.
"""

import argparse
import cProfile
import json
import time
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg

import eyepiece as ep


def make_clip():
    """Build 300 groups of six labels and one mark, with four groups present."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=60)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    groups = {}
    for i in range(300):
        x, y = (i % 20 + 0.5) / 20, (i // 20 + 0.5) / 15
        (dot,) = ax.plot([x], [y], "o", ms=3, color="black")
        labels = [
            ax.text(x, y + j * 0.008, f"item {i}.{j}", fontsize=5, color="black")
            for j in range(6)
        ]
        groups[str(i)] = [dot, *labels]
    stage = ep.motion.Stage(groups)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    FigureCanvasAgg(fig)
    tl = ep.Timeline({"phase": 0.0}, tempo=ep.Tempo(factor=1.0))
    tl.hold(0.5)
    tl.run("change", 2.0, to={"phase": 1.0})
    tl.hold(0.5)
    plan = tl.at_fps(10)

    def draw(k):
        # Repeated states exercise exact raster reuse while retaining all
        # transparent artists and their original visibility flags.
        offset = int(plan.value(k, "phase") * 5) * 4
        stage.weigh({name: float(offset <= int(name) < offset + 4) for name in groups})

    return fig, draw, plan


def report_text(fig, report):
    """Serialize complete findings with stable artist identifiers."""
    artists = {a: f"{type(a).__name__}:{i}" for i, a in enumerate(fig.findobj())}

    def value(item):
        if isinstance(item, Artist):
            return artists[item]
        if isinstance(item, tuple):
            return [value(v) for v in item]
        return item

    result = {"report": str(report), "unread_knobs": report.unread_knobs}
    for group in ("readability", "meaning", "motion", "plan_faults"):
        result[group] = [
            {f.name: value(getattr(fault, f.name)) for f in fields(fault)}
            for fault in getattr(report, group)
        ]
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def main():
    """Time one clip check and retain its report and profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig, draw, plan = make_clip()
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    report = ep.testing.check_clip(fig, draw, plan)
    profile.disable()
    elapsed = time.perf_counter() - start
    profile.dump_stats(args.out / "profile.pstats")
    (args.out / "report.txt").write_text(report_text(fig, report))
    (args.out / "timing.json").write_text(json.dumps({"wall_s": elapsed}) + "\n")
    print(f"{elapsed:.3f} s, {len(report.faults)} findings, {plan.n_frames} frames")
    plt.close(fig)


if __name__ == "__main__":
    main()
