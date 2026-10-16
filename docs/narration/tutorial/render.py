"""Render tutorial 3 with regenerated speech audio and WebVTT captions.

Run from the repository root::

    python docs/narration/tutorial/render.py /tmp/narrated-tutorial

Regenerate the speech cache audio with the narration tool before rendering.
The output directory receives the silent MP4, WAV track, narrated MP4,
WebVTT, and SRT. Rendering requires ffmpeg and soundfile.
"""

import argparse
import shutil
from pathlib import Path

from clip import Scene, caption_steps, make_timeline
from script import SCRIPT

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece import narration as nr


def main():
    """Record the shared scene, assemble speech, and mux the final clip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    cache_dir = Path(__file__).resolve().parent / "cache"
    if shutil.which("ffmpeg") is None:
        raise SystemExit("render.py needs ffmpeg on PATH")
    if not list(cache_dir.glob("*.opus")) and not list(cache_dir.glob("*.flac")):
        raise SystemExit("no speech audio in the cache; run the narration tool first")
    narration = nr.Narration.load(SCRIPT, cache_dir)
    tl = make_timeline(narration)
    plan = tl.at_fps(10)
    cues = nr.captions(tl.narration)
    scene = Scene()

    def draw(k):
        scene.draw(k, plan, cues)

    print(
        ept.assert_clip(
            scene.fig, draw, plan, moves=scene.moves, steps=caption_steps(plan, cues)
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "narrated"
    nr.write_captions(tl.narration, stem, line=90)
    silent = args.output_dir / "silent.mp4"
    with ep.record(scene.fig, silent, plan=plan, dpi=80) as rec:
        for k in range(plan.n_frames):
            draw(k)
            rec.frame()
    track = nr.audio_track(tl.narration, cache_dir, args.output_dir / "track.wav")
    nr.mux(silent, track, stem.with_suffix(".mp4"))


if __name__ == "__main__":
    main()
