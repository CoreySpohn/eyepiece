"""Render tutorial 3 with regenerated speech audio and WebVTT captions.

Run from the repository root::

    python docs/narration/tutorial/render.py /tmp/narrated-tutorial

For parallel rendering, keep the builder in an importable module, for example
``worker_clip.py`` beside this script::

    # worker_clip.py
    from pathlib import Path
    from clip import Scene, make_timeline
    from script import SCRIPT
    import eyepiece as ep

    def build():
        scene = Scene()
        cache = Path(__file__).resolve().parent / "cache"
        narration = ep.narration.Narration.load(SCRIPT, cache)
        timeline = make_timeline(narration)
        plan = timeline.at_fps(10)
        cues = ep.narration.captions(timeline.narration)
        return scene.fig, lambda fig, k: scene.draw(k, plan, cues)

Then the rendering script uses ``from worker_clip import build`` and calls
``ep.render.render_frames(build, plan.n_frames, silent, fps=10, dpi=80)`` inside its
``if __name__ == "__main__":`` guard. Workers import ``build`` by module name;
defining it in this script's ``__main__`` cannot work with spawned workers.
Only the builder crosses the process boundary; the draw closure it returns is
created inside each worker. See docs/how-to/render-a-clip-in-parallel.md for a
complete runnable example. For drafts, ``render_variants`` and ``record`` accept
``time_range=(start_s, stop_s)`` while retaining absolute source frame indices.

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
    narration = ep.narration.Narration.load(SCRIPT, cache_dir)
    tl = make_timeline(narration)
    plan = tl.at_fps(10)
    cues = ep.narration.captions(tl.narration)
    scene = Scene()

    def draw(k):
        scene.draw(k, plan, cues)

    print(
        ep.testing.assert_clip(
            scene.fig, draw, plan, moves=scene.moves, steps=caption_steps(plan, cues)
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "narrated"
    ep.narration.write_captions(tl.narration, stem, line=90)
    silent = args.output_dir / "silent.mp4"
    with ep.render.record(scene.fig, silent, plan=plan, dpi=80) as rec:
        for k in range(plan.n_frames):
            draw(k)
            rec.frame()
    track = ep.narration.audio_track(
        tl.narration, cache_dir, args.output_dir / "track.wav"
    )
    ep.narration.mux(silent, track, stem.with_suffix(".mp4"))


if __name__ == "__main__":
    main()
