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

# Add cue sounds

```{warning}
The cue sounds are a prototype. They wait on one keep-or-drop listening test
of the sweep in `docs/narration/cues_sweep/sweep.py`, and the names below may
be removed if they fail it.
```

Cue sounds are a second encoding of numbers the picture already shows. A clip
that uses them states its mapping in one sentence:

> Pitch is proportional to the count; one tick per pixel read; one two-note
> sting at the click.

## When a count is better heard

A sound helps when the quantity is a ratio that is easy to hear and hard to
read. A count that goes from 4 to 16 is two octaves, and an ear hears "it
doubled, then doubled again" while the eye is busy following a curve. A run of
pixel reads is a rate: 4 ticks in one second and 96 ticks in one second sound
different at once, while 96 flashes in one second look like one smear. A
number the viewer needs exactly (0.943, say) is better read than heard; put it
in a readout and leave it silent.

## The three voices

`nr.CueEvents` holds what the voices play, in seconds on the clip's clock, and
`nr.cue_track` turns it into samples. Read the times and the counts from the
same plan frames the picture draws, so picture and sound cannot disagree.

```{code-cell} python
import numpy as np

from eyepiece import narration as nr

events = nr.CueEvents(
    count_times=(1.0, 3.0),
    count_values=(4.0, 16.0),
    reference=4.0,
    tick_times=tuple(3.5 + np.arange(96) / 96),
    sting_s=4.8,
    bell_at=8.0,
)
track = nr.cue_track(events, 5.0, rate=48000, seed=0)
print(f"{len(track)} samples, peak {20 * np.log10(np.abs(track).max()):.1f} dBFS")
```

- `nr.count_tone`: a sine at 220 Hz x count / reference, gliding without a
  phase jump, with a 40 ms attack and a 250 ms decay that ends at the last
  knot. An optional octave bell (440 Hz, 60 ms) rings once where the count
  first reaches `bell_at`.
- `nr.pixel_ticks`: one 4 ms raised-cosine click at 2.4 kHz per time, each
  moved by at most 1 ms by a seeded generator, so a fast run does not buzz.
- `nr.click_sting`: 90 ms of 880 Hz and 1318.5 Hz (a fifth) with a soft
  attack, ending exactly at its time, which is the end of the click's swell.

Pass the events to `nr.audio_track(..., cues=events)`. The cues are mixed
before the loudness pass, and the voice drops out under them through the same
fades that duck it elsewhere. With `cues=None` the track is unchanged, byte
for byte.

## Levels

| Sound | Level |
| --- | --- |
| Count tone | -30 dBFS RMS |
| Octave bell | -26 dBFS peak |
| Pixel ticks | -26 dBFS peak |
| Click sting | -26 dBFS peak |
| Room tone (from `mux`) | -60 dBFS |

Every cue stays at least 12 dB below the voice. That holds when the speech files are near
full scale, as the cached speech is. Because `mux` normalizes the whole mix,
the cues move with the speech gain; a clip with no voice should pass
`loudness=None` to `mux`, or the normalization lifts the cues to the speech
target. There is no music bed.

## Signs of a gimmick

Drop a cue sound if either of these is true:

- The mapping cannot be stated in one sentence.
- A sound plays that matches no quantity on screen.

## The prototype sweep

`docs/narration/cues_sweep/sweep.py` builds a 20 s sweep, a count from 4 to 16
as the square of a wavelength, with an imager that reads 4 pixels and a
spectrograph that reads 96, and writes it twice: with cues and with room tone
only. Run it from the repository root (it needs ffmpeg):

```bash
python docs/narration/cues_sweep/sweep.py /tmp/cues-sweep
```

The test is one listen to `sweep-cues.mp4`: keep the cues if the mapping is
obviously right within 10 s, otherwise drop them.
