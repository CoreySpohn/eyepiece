r"""Synthesize a narration script with Kokoro, and cache it beside the clip.

This is the only module of the package that needs the ``narration`` extra
(``pip install eyepiece[narration]``: Kokoro and soundfile), and it imports
them only when a line has to be spoken. Kokoro pulls in torch, so run this
module in an environment of its own, with eyepiece importable and the
Kokoro version the script names (`KOKORO_VERSION`)::

    HF_HUB_DISABLE_XET=1 PYTHONPATH=<eyepiece>/src uv run --no-project \
        --python 3.12 --with kokoro==0.9.4 --with soundfile --with numpy \
        python -m eyepiece.narration.synth script.py cache_dir

``script.py`` defines ``SCRIPT``, a `Script`. For each line it writes two
files into ``cache_dir`` named by the line's cache key: ``<key>.opus`` (Ogg
Opus, or ``<key>.flac`` when the installed libsndfile cannot write Opus) and
``<key>.json`` (word times, length, sample count, voice and versions). Both
belong in the repository: the audio is the audio the cue times were
measured on, and a machine that has the pair never speaks the line again.

A line is skipped when its JSON and audio are both present. A JSON without
its audio is an error, since speaking it again could move the cue times;
``--respeak`` says that is intended. ``--force`` speaks every line again,
``--prune`` deletes cache files that no line of the script uses, and
``--list`` prints what the cache holds.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from ._align import align
from ._timeline import MissingAudioError, Spoken, Word

SAMPLE_RATE = 24000
_CACHE_FILE = re.compile(r"^(?P<key>[0-9a-f]{16})\.(json|opus|flac|wav)$")
_INSTALL = (
    "Speech synthesis needs the narration extra (pip install "
    "'eyepiece[narration]': kokoro and soundfile), which pulls in torch. Run "
    "this module in an environment of its own, for example with `uv run "
    "--no-project --with kokoro==%s --with soundfile --with numpy` and "
    "eyepiece on PYTHONPATH, or through a wrapper that does so."
)


def _kokoro(script):
    """Build the Kokoro ``speak`` function for ``script``."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from kokoro import KPipeline
    except ImportError:
        raise ImportError(_INSTALL % script.engine_version) from None
    installed = importlib.metadata.version("kokoro")
    if installed != script.engine_version:
        raise RuntimeError(
            f"the script is keyed to kokoro {script.engine_version} but "
            f"{installed} is installed; install the keyed version, or set "
            "engine_version on the Script and speak the lines again"
        )
    pipeline = KPipeline(lang_code=script.lang, repo_id="hexgrad/Kokoro-82M")

    def speak(line):
        chunks, tokens, offset = [], [], 0.0
        for result in pipeline(line.spoken, voice=script.voice, speed=script.speed):
            audio = np.asarray(result.audio, dtype=np.float32)
            for t in result.tokens:
                start = None if t.start_ts is None else offset + float(t.start_ts)
                end = None if t.end_ts is None else offset + float(t.end_ts)
                tokens.append((t.text, start, end))
            chunks.append(audio)
            offset += len(audio) / SAMPLE_RATE
        audio = np.concatenate(chunks)
        spans = align(line.words, tokens)
        words = tuple(
            Word(w, a, b) for w, (a, b) in zip(line.words, spans, strict=True)
        )
        return audio, SAMPLE_RATE, words

    return speak


# Engine name to a function that takes a Script and returns
# ``speak(line) -> (audio, rate, words)``: float audio in [-1, 1], its sample
# rate, and one `Word` per written word, timed from the audio start.
ENGINES = {"kokoro": _kokoro}


def _replace(path, writer):
    """Write through a temporary file so a stop never leaves a partial file."""
    tmp = path.with_name(f".{path.name}.part")
    writer(tmp)
    os.replace(tmp, path)


def _write_compressed(directory, key, audio, rate):
    """Write ``audio`` as Ogg Opus, or FLAC if Opus is unavailable; return the name."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(_INSTALL % "the pinned version") from None
    if "OPUS" in sf.available_subtypes("OGG"):
        name, options = f"{key}.opus", {"format": "OGG", "subtype": "OPUS"}
    else:
        name, options = f"{key}.flac", {"format": "FLAC", "subtype": "PCM_16"}
    path = Path(directory) / name
    _replace(path, lambda tmp: sf.write(tmp, audio, rate, **options))
    frames = sf.info(path).frames
    if abs(frames - len(audio)) > 0.02 * rate:
        path.unlink()
        raise RuntimeError(
            f"{name} holds {frames} samples but {len(audio)} were written"
        )
    return name


def _write_json(path, record):
    _replace(path, lambda tmp: tmp.write_text(json.dumps(record, indent=1) + "\n"))


def synthesize(
    script,
    cache_dir,
    *,
    speak=None,
    write_audio=None,
    force=False,
    respeak=False,
    prune=False,
):
    """Speak every line of ``script`` that is not cached.

    Args:
        script: The `Script`.
        cache_dir: Where the timing JSON and the audio go.
        speak: ``speak(line) -> (audio, rate, words)``; None takes the engine
            named by ``script.engine`` from `ENGINES`, built when the first
            line needs speaking.
        write_audio: ``write_audio(directory, key, audio, rate) -> name``;
            None writes Ogg Opus or FLAC.
        force: Speak every line again.
        respeak: Speak again a line whose JSON exists but whose audio does
            not. Without it that is an error.
        prune: Delete the cache files that no line of the script uses.

    Returns:
        The keys of the lines that were spoken.

    Raises:
        MissingAudioError: If a line has JSON but no audio and ``respeak`` is
            not set.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_audio = _write_compressed if write_audio is None else write_audio
    spoken_keys = []
    for line in script.lines:
        key = script.line_key(line)
        meta = cache_dir / f"{key}.json"
        if meta.exists() and not force:
            audio_name = json.loads(meta.read_text()).get("audio")
            if audio_name and (cache_dir / audio_name).exists():
                continue
            if not respeak:
                raise MissingAudioError(
                    f"line {line.key!r} has {meta.name} but no audio beside it; "
                    "speaking it again can move its cue times. Restore the "
                    "audio from git, or pass respeak=True (--respeak) to "
                    "speak it again and review the cues"
                )
        if speak is None:
            speak = ENGINES[script.engine](script)
        audio, rate, words = speak(line)
        name = write_audio(cache_dir, key, audio, rate)
        spoken = Spoken(
            key, len(audio) / rate, tuple(words), line.spoken, name, rate, len(audio)
        )
        record = spoken.to_json()
        record.update(
            voice=script.voice,
            speed=script.speed,
            lang=script.lang,
            engine=script.engine,
            engine_version=script.engine_version,
            model=script.model,
        )
        _write_json(meta, record)
        spoken_keys.append(line.key)
    if prune:
        prune_cache(script, cache_dir)
    return spoken_keys


def prune_cache(script, cache_dir):
    """Delete the cache files that no line of ``script`` uses.

    Returns:
        The paths deleted.
    """
    keep = set(script.cache_keys())
    removed = []
    for path in sorted(Path(cache_dir).iterdir()):
        match = _CACHE_FILE.match(path.name)
        if match and match["key"] not in keep:
            path.unlink()
            removed.append(path)
    return removed


def list_cache(script, cache_dir):
    """What the cache holds: one ``(key, line, text, has_audio)`` per JSON.

    ``line`` is the key of the script line that uses the entry, or None for
    an entry no line uses.
    """
    by_key = dict(zip(script.cache_keys(), (x.key for x in script.lines), strict=True))
    rows = []
    for path in sorted(Path(cache_dir).glob("*.json")):
        record = json.loads(path.read_text())
        audio = record.get("audio")
        rows.append(
            (
                path.stem,
                by_key.get(path.stem),
                record.get("text", ""),
                bool(audio) and (Path(cache_dir) / audio).exists(),
            )
        )
    return rows


def _load_script(path):
    spec = importlib.util.spec_from_file_location("narration_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["narration_script"] = module
    spec.loader.exec_module(module)
    return module.SCRIPT


def main(argv=None):
    """Command line; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", help="a .py file that defines SCRIPT")
    parser.add_argument("cache_dir")
    parser.add_argument("--force", action="store_true", help="speak every line again")
    parser.add_argument(
        "--respeak",
        action="store_true",
        help="speak again a line whose JSON has lost its audio",
    )
    parser.add_argument("--prune", action="store_true", help="delete unused files")
    parser.add_argument("--list", action="store_true", help="print the cache and stop")
    args = parser.parse_args(argv)
    script = _load_script(args.script)
    if args.list:
        for key, line, text, has_audio in list_cache(script, args.cache_dir):
            audio = "audio" if has_audio else "NO AUDIO"
            print(f"{key}  {line or '(unused)':<12} {audio}  {text}")
        return
    spoken = synthesize(
        script,
        args.cache_dir,
        force=args.force,
        respeak=args.respeak,
        prune=args.prune,
    )
    print(f"spoke {len(spoken)} lines: {', '.join(spoken) or 'none, all cached'}")


if __name__ == "__main__":
    main()
