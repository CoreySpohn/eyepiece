---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Estimate length before speaking

Estimate speech duration while drafting a script, before installing or running
the speech engine. `nr.estimate` counts the spoken forms of numbers and
allows short pauses for punctuation. The estimate excludes line padding and
animation holds, and cached word times replace it for a finished render.

```{code-cell} python
from eyepiece import narration as nr

script = nr.Script((
    nr.Line("open", "First the axes appear."),
    nr.Line("profile", "The row profile reaches a relative intensity of 0.943."),
))
nr.check_script_speech(script)
total_s = 0.0
for line in script.lines:
    spoken = nr.estimate(line)
    total_s += spoken.duration_s
    print(f"{line.key}: {spoken.duration_s:.2f} s")
print(f"Estimated speech: {total_s:.2f} s")
```

For a script exported to JSON, the equivalent command is
`python -m eyepiece.narration.synth script.json --estimate`. That mode checks
the speech text and prints per-line durations without importing Kokoro or
requiring a cache directory. See {doc}`speak-a-script` for JSON export and
speech synthesis.
