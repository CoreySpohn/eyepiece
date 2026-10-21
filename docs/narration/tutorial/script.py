"""Narration for tutorial 3: the tutorial 2 clip, spoken."""

import eyepiece as ep

SCRIPT = ep.narration.Script(
    (
        ep.narration.Line(
            "panel",
            "A coronagraph image is easier to read when it arrives in stages. "
            "First the {show:panel}axes and their labels appear.",
        ),
        ep.narration.Line(
            "image",
            "Next comes the {show:image}field, a synthetic coronagraph image "
            "with a dark hole at its center.",
        ),
        ep.narration.Line(
            "profile",
            "Then the {show:profile}row profile is drawn across the image, "
            "from left to right.",
        ),
        ep.narration.Line(
            "marker",
            "The {indicate:marker}marker picks out one separation along that profile.",
            pad_after_s=0.4,
        ),
        ep.narration.Line(
            "close",
            "The finished figure is one picture, and each part of it was "
            "introduced in turn.",
        ),
    ),
    speed=0.85,
)
