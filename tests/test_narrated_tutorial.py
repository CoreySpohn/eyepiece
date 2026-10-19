"""Exercise the committed tutorial narration without synthesis or audio."""

import math
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from eyepiece import narration as nr

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "docs" / "narration" / "tutorial"
SCRIPT = runpy.run_path(TUTORIAL / "script.py")["SCRIPT"]
MAKE_TIMELINE = runpy.run_path(TUTORIAL / "clip.py")["make_timeline"]
TIMELINE = MAKE_TIMELINE(nr.Narration.load(SCRIPT, TUTORIAL / "cache"))
PLAN = TIMELINE.at_fps(10)
CUES = nr.captions(TIMELINE.narration)


@pytest.mark.parametrize("index", range(len(CUES)))
def test_tutorial_caption_edges(index):
    start, end, text = CUES[index]
    previous = CUES[index - 1] if index else None
    following = CUES[index + 1] if index + 1 < len(CUES) else None
    before = previous[2] if previous and previous[1] == start else ""
    after = following[2] if following and following[0] == end else ""

    assert nr.caption_at(CUES, math.nextafter(start, -math.inf)) == before
    assert nr.caption_at(CUES, start) == text
    assert nr.caption_at(CUES, math.nextafter(end, -math.inf)) == text
    assert nr.caption_at(CUES, end) == after


def test_docs_track_no_generated_audio_video_or_captions():
    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    checkout = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if checkout.returncode or checkout.stdout.strip() != "true":
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "docs/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    names = [name for name in tracked.stdout.split("\0") if name]
    excluded = {".opus", ".flac", ".wav", ".mp4", ".vtt", ".srt"}

    def committed_speech(name):
        # The spoken audio of a line is committed beside its timing JSON in a
        # narration cache, so the docs build never synthesizes speech.
        path = Path(name)
        return (
            path.suffix.lower() == ".opus"
            and path.parent.name == "cache"
            and str(path.with_suffix(".json")) in names
        )

    media = [
        name
        for name in names
        if Path(name).suffix.lower() in excluded and not committed_speech(name)
    ]
    assert media == []


def test_tutorial_script_passes_speech_text_check():
    nr.check_script_speech(SCRIPT)


def test_caption_steps_are_the_frames_where_the_caption_changes():
    caption_steps = runpy.run_path(TUTORIAL / "clip.py")["caption_steps"]
    shown = [nr.caption_at(CUES, k / PLAN.fps) for k in range(PLAN.n_frames)]
    changes = [k for k in range(1, PLAN.n_frames) if shown[k] != shown[k - 1]]
    steps = caption_steps(PLAN, CUES)
    assert set(changes) <= set(steps)
    assert len(CUES) == 7
