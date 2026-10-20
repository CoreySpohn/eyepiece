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

# Ground a symbol in the mark it names

When a symbol appears, the viewer must know which mark it means. Binding a
term to its mark gives the term the mark's color, lets one cue light both,
and lets a check find a symbol with no referent, in the wrong color, or
before the thing it names.

## Bind a term

```{code-cell} python
import hwostyle
import matplotlib.pyplot as plt
from IPython.display import display

import eyepiece as ep
import eyepiece.testing as ept

hwostyle.use("dark")
motifs = ep.MotifSet.default()
line_kw = motifs["planet"].line_kw()
line_kw["ls"] = "-"
fig, ax = plt.subplots(figsize=(4.6, 2.8), layout="constrained")
(line,) = ax.plot([0, 0.25, 0.5, 0.75, 1], [0.2, 0.35, 0.5, 0.35, 0.2],
                  **line_kw)
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Orbital phase", ylabel="Planet flux")
eq = ep.equation(ep.Anchor.from_data(ax, (0.5, 0.82)),
                 r"$F = {{flux:F_p}} / F_s$", fontsize=24)
eq.bind("flux", line, motif="planet")
```

Here `F_p` names the planet flux curve and `F_s` is the stellar flux, so `F`
is their ratio. For term syntax and transforms, see the
{doc}`equation gallery <../gallery/motion-equations>`.

## Cue both at once

```{code-cell} python
pulse = ep.indicate(eq.linked("flux"))
pulse.update(0.5)
display(fig)
pulse.update(1.0)
```

The pulse lights the term and its mark together and restores both at `u = 1`.

## Check the grounding

Sample the resting states of two beats, first with the equation before the
curve, then with the curve before the equation.

```{code-cell} python
def symbol_first(t):
    eq.update(1.0)
    line.set_alpha(0.0 if t < 1.0 else 1.0)


faults = ept.grounding_faults(eq, at=symbol_first, times=[0.0, 1.0])
print([(fault.kind, fault.term, fault.time_s) for fault in faults])


def mark_first(t):
    line.set_alpha(1.0)
    eq.update(0.0 if t < 1.0 else 1.0)


print(ept.grounding_faults(eq, at=mark_first, times=[0.0, 1.0]))
plt.close(fig)
```

The first order reports `symbol-first` and `no-referent` for `flux` at zero
seconds, and the corrected order prints `[]`.
An unbound colored term also reports `no-referent`, wrong term or mark ink
reports `color-mismatch`, and `ept.assert_grounded` raises `GroundingError`
for findings in a test.
