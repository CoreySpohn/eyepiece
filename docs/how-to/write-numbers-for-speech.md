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

# Write numbers for speech

Keep the figure's numeric format when forming a spoken value, then expand
its operators and units with `nr.say`. Written notation can have several
readings, so use words that state the intended operation or unit.

| Written form | Spoken wording |
| --- | --- |
| `1e-15` | one times ten to the minus fifteen |
| `3 x 10^-8` | three times ten to the minus eight |
| `+/-` | plus or minus |
| `~` | approximately |
| `a/b` | a divided by b |
| `mas` | milliarcseconds |

`nr.say` retains digits for the voice to read while expanding scientific
notation, signs, percentages, and supported units. Use the same source value
and format specification as the figure.

```{code-cell} python
from eyepiece import narration as nr

print(nr.say(1e-15, ".0e"))
print(nr.say(3e-8, ".0e"))
print(nr.say(12, "g", unit="mas"))
print(nr.say(-0.126, ".1%"))
```

Check draft text with `speech_faults` to find notation that needs wording.
The checker recognizes operators such as exponents and slashes, but it does
not identify every ambiguous abbreviation. Expand units such as `mas` with
`nr.say(..., unit="mas")` even when the checker reports no fault.

```{code-cell} python
draft = "The level is 1e-15, or 3 x 10^-8 +/- a/b, at ~ 12 mas."
for fault in nr.speech_faults(draft):
    print(f"{fault.token}: {fault.kind}")

line = nr.Line(
    "level",
    f"The level is {nr.say(1e-15, '.0e')} at {nr.say(12, 'g', unit='mas')}.",
    display="The level is 1e-15 at 12 mas.",
)
script = nr.Script((line,))
nr.check_script_speech(script)
assert nr.speech_faults(line.spoken) == []
print(line.spoken)
```

`display=` keeps compact notation in captions while `spoken` supplies the
voice. Keep the same sentence count in both versions to preserve caption
timing within the line. See {doc}`../explanation/narration-design` for supported units.

## Pair scientific notation with speech

`scientific` returns mathtext and spoken text from one value and exponential
format. It uses the same rounded mantissa in both. Bind the first word of
the spoken quantity to the screen artist in the number gate. Both gates
read the exponent as part of the value, and `lint_script` counts it as one
quantity. Scientific screen comparisons and spoken synchronization use a relative
tolerance cap, so small
values cannot match merely because both are below the plain absolute floor.

```python
import matplotlib.pyplot as plt
from eyepiece import narration as nr
import eyepiece.testing as ept

screen, spoken = nr.scientific(1.3e-4, ".1e")
fig, ax = plt.subplots()
label = ax.text(0.5, 0.5, screen)
line = nr.Line("value", "The value is " + spoken + ".")
narration = nr.Narration.estimated(nr.Script((line,)))
faults = ept.timing_faults(
    narration, at=lambda t: None, readouts={("value", 3): label},
    ceiling=200, short_ceiling=200,
)
assert faults == []
```
