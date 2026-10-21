"""Time a busy clip based on gallery/motion-attention and test_motion_faults.

Run with PYTHONPATH pointing to the revision under test. Only motion_faults
is timed; --profile writes cProfile data and --faults writes comparable JSON.
"""

import argparse
import cProfile
import json
import time
from dataclasses import fields

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep
import eyepiece.testing as ept


def busy_clip():
    """Return a four-second orbit scene with labels, captions and emphasis."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.5), dpi=60)
    fig.subplots_adjust(bottom=0.22, hspace=0.6, wspace=0.4)
    phase = np.linspace(0, 2 * np.pi, 200)
    markers = []
    labels = []
    for i, ax in enumerate(axes.flat):
        ax.plot(0.6 * np.cos(phase), 0.6 * np.sin(phase), color="0.6")
        (marker,) = ax.plot([0.6], [0], "o", ms=7)
        markers.append(marker)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="x", ylabel="y")
        ax.set_title(f"Orbit {i + 1}")
        ax.set_axis_off()
        for j in range(12):
            labels.append(
                ax.text(-0.95, 0.85 - j * 0.15, f"sample {j}: {i + j}.25", fontsize=5)
            )
    fig.canvas.draw()
    fig.set_layout_engine("none")
    ring = ep.circumscribe(markers[0], shape="ellipse", pad_pt=5)
    caption = ep.words(
        ep.Anchor.from_data(axes[1, 0], (-0.9, -2.0)),
        "The moving world carries the measured velocity around its orbit",
        fontsize=12,
    )
    # A deliberate late departure supplies a nonempty fault comparison.
    departure = fig.text(0.5, 0.06, "velocity", fontsize=12)

    def at(t):
        u = min(1.0, t / 4.0)
        for i, marker in enumerate(markers):
            angle = 2 * np.pi * u + i / 3
            marker.set_data([0.6 * np.cos(angle)], [0.6 * np.sin(angle)])
        for label in labels:
            label.set_x(-0.95 + u)
        # Like test_hop_groups, one mark is an exchangeable draw. Its move
        # carries the panel labels as captions while they travel into place.
        markers[0].set_xdata([0.2 if int(t * 3) % 2 else 0.6])
        caption.reveal(min(1.0, t / 1.5), by="word")
        ring.update(u)
        departure.set_x(0.5 + 0.6 * u)

    moves = [
        {
            "artists": [*markers, *labels, *caption.artists["text"]],
            "hops": [{"artists": [markers[0]], "rate": 3.0}],
        }
    ]
    return fig, at, moves


def fault_records(found, fig):
    """Serialize every fault field, using stable names for artist identities."""
    artists = {
        artist: f"{type(artist).__name__}:{i}" for i, artist in enumerate(fig.findobj())
    }

    def value(v):
        if isinstance(v, matplotlib.artist.Artist):
            return artists[v]
        if isinstance(v, tuple):
            return [value(item) for item in v]
        return v

    return [
        {f.name: value(getattr(fault, f.name)) for f in fields(fault)}
        for fault in found
    ]


def main():
    """Print elapsed seconds per clip second, optionally retaining evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile")
    parser.add_argument("--faults")
    args = parser.parse_args()
    fig, at, moves = busy_clip()
    profile = cProfile.Profile() if args.profile else None
    start = time.perf_counter()
    if profile:
        profile.enable()
    found = ept.motion_faults(fig, at, ept.Window(0, 4), moves=moves)
    if profile:
        profile.disable()
        profile.dump_stats(args.profile)
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.3f} s; {elapsed / 4:.3f} s/clip-second; {len(found)} faults")
    if args.faults:
        with open(args.faults, "w") as stream:
            json.dump(fault_records(found, fig), stream, indent=2)
    plt.close(fig)


if __name__ == "__main__":
    main()
