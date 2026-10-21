# Speak a script

Export a script in the project environment, then speak the JSON in an
isolated environment. This page shows commands without executing them.
Run the examples from the repository root.

The tutorial script already defines its cue marks and line keys. Exporting
JSON preserves those settings without requiring the speech environment to
import the script module's dependencies.

```python
import sys
from pathlib import Path

import eyepiece as ep

tutorial_dir = Path("docs/narration/tutorial")
sys.path.insert(0, str(tutorial_dir.resolve()))
from script import SCRIPT

ep.narration.check_script_speech(SCRIPT)
SCRIPT.to_json("/tmp/tutorial-script.json")
```

Run the narration tool with Kokoro pinned to the version used in cache keys.
The tutorial already has timing JSON but no audio, so `--respeak` explicitly
regenerates that pair. For a new cache with no existing files, omit the flag.

```bash
PYTHONPATH="$PWD/src" uv run --no-project --python 3.12 \
    --with kokoro==0.9.4 --with soundfile --with numpy \
    python -m eyepiece.narration.synth /tmp/tutorial-script.json \
    docs/narration/tutorial/cache --respeak
```

For each line, the tool writes timing JSON and compressed audio, using Opus
when available and FLAC otherwise. Commit the script and cache JSON, keeping
the generated audio out of version control. If speech is regenerated, review
and commit its new JSON together with any script change so that later loads
use the matching word times. Ordinary documentation builds load only JSON.

With ffmpeg available and the audio regenerated, render the narrated movie
and caption files into an output directory outside the source tree.

```bash
python docs/narration/tutorial/render.py /tmp/tutorial-3
```

See {doc}`../tutorials/narrated` for word-timed reveals and
{doc}`estimate-length-before-speaking` for a draft duration without synthesis.
