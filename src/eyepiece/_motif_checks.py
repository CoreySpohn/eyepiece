"""Checks on a `MotifSet`: can its motifs be told apart, in every mode and vision?

`motif_faults` reads a set, not a picture. It resolves each motif's color in a
mode and reports, as `MotifFault` records:

- ``"signature"``: two motifs that resolve to one color and share every
  non-color channel, so nothing separates them in grayscale.
- ``"sharing"``: a pair declared never to co-occur (a hue shared by design, or
  two image roles that share a map family) present together in one set.
- ``"budget"``: more hues than the budget. A fifth or sixth hue needs a second
  channel (marker, hatch, arrow or font style), and a seventh is never allowed.
- ``"cvd"``: two hue motifs whose colors fall close in CAM02-UCS under normal
  vision or a color-vision deficiency. A second channel softens the finding.
- ``"dimmed"``: advisory, hues that fall close once the set is dimmed together.
- ``"cue"``: the cue is not a form in the text color, or a hue in the set
  matches the cue's color.

The checks apply to the subset a figure or clip draws with
(`MotifSet.only`), not to the whole default table. `motif_margins` returns the
numbers behind the findings.
"""

import itertools
from dataclasses import dataclass

import hwostyle
from matplotlib.colors import to_hex, to_rgb

from eyepiece import _style

MOTIF_KINDS = ("signature", "sharing", "budget", "cvd", "dimmed", "cue")
_VISIONS = ("deuteranomaly", "protanomaly", "tritanomaly")
_CUE_SAME = 2.0
_HARD_FLOOR = 5.0


class MotifError(AssertionError):
    """A motif set has findings that `assert_motifs` refuses."""


@dataclass(frozen=True)
class MotifFault:
    """One finding about a motif set.

    Attributes:
        kind: One of `MOTIF_KINDS`.
        severity: ``"fault"``, ``"warning"`` or ``"advisory"``.
        keys: The motif keys involved.
        detail: A sentence saying what is wrong.
        de: The CAM02-UCS distance behind a ``cvd`` or ``dimmed`` finding,
            else None.
    """

    kind: str
    severity: str
    keys: tuple
    detail: str
    de: float | None = None


def _vision():
    try:
        import colorspacious
    except ImportError as exc:
        msg = (
            "eyepiece.testing.motif_faults needs colorspacious; "
            "install the test extra (pip install eyepiece[test])"
        )
        raise ImportError(msg) from exc
    return colorspacious


def _lab(hex_color, cvd):
    cs = _vision()
    rgb = list(to_rgb(hex_color))
    if cvd is not None:
        rgb = cs.cspace_convert(
            rgb, {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}, "sRGB1"
        ).clip(0, 1)
    return cs.cspace_convert(rgb, "sRGB1", "CAM02-UCS")


def _de(a, b, cvd=None):
    return float(((_lab(a, cvd) - _lab(b, cvd)) ** 2).sum() ** 0.5)


def _worst(a, b):
    """Smallest distance over normal vision and the three deficiencies."""
    return min(_de(a, b, v) for v in (None, *_VISIONS))


def _mode_name(mode):
    return mode or hwostyle.current_mode() or "light"


def _dim_toward_background(color, level, mode):
    rc = hwostyle.styles.MODE_RC[_mode_name(mode)]
    back = to_rgb(rc["axes.facecolor"])
    rgb = to_rgb(color)
    return "#" + "".join(
        f"{round(255 * (b + level * (c - b))):02x}"
        for b, c in zip(back, rgb, strict=True)
    )


def _hue_motifs(motifs, mode):
    """Hue motifs of a set as ``(key, hex)``, one per key."""
    return [(k, m.resolve(mode)) for k, m in motifs.items() if m.is_hue]


def _distinct_hues(pairs):
    """The first motif of each distinct hue, in set order."""
    seen = {}
    for key, color in pairs:
        seen.setdefault(color.lower(), (key, color))
    return list(seen.values())


def motif_margins(motifs, *, mode=None, dim=None):
    """The separation numbers behind `motif_faults`.

    Args:
        motifs: A `MotifSet`.
        mode: A hwostyle mode name, or None for the active mode.
        dim: Contrast a dimmed motif keeps; None uses the mode's level.

    Returns:
        A dict with ``hues`` (distinct hue count), ``min_de`` and
        ``min_pair`` (smallest CAM02-UCS distance among hue motifs under
        normal vision and the three deficiencies, sharing pairs excluded),
        ``dim`` (the level used), and ``dim_min_de`` and ``dim_min_pair``
        (the same minimum with every hue dimmed together). Distances are None
        when fewer than two hues are present.
    """
    hues = _distinct_hues(_hue_motifs(motifs, mode))
    level = _style.dim_level(_mode_name(mode)) if dim is None else float(dim)
    out = {
        "hues": len(hues),
        "min_de": None,
        "min_pair": None,
        "dim": level,
        "dim_min_de": None,
        "dim_min_pair": None,
    }
    for (ka, ca), (kb, cb) in itertools.combinations(hues, 2):
        if motifs.never_together(ka, kb):
            continue
        de = _worst(ca, cb)
        if out["min_de"] is None or de < out["min_de"]:
            out["min_de"], out["min_pair"] = de, (ka, kb)
        dimmed = _worst(
            _dim_toward_background(ca, level, mode),
            _dim_toward_background(cb, level, mode),
        )
        if out["dim_min_de"] is None or dimmed < out["dim_min_de"]:
            out["dim_min_de"], out["dim_min_pair"] = dimmed, (ka, kb)
    return out


def _signature_faults(motifs, mode):
    by_color = {}
    for key, m in motifs.items():
        by_color.setdefault(m.resolve(mode).lower(), []).append(key)
    out = []
    for color, keys in by_color.items():
        for a, b in itertools.combinations(keys, 2):
            if motifs[a].signature() == motifs[b].signature():
                out.append(
                    MotifFault(
                        "signature",
                        "fault",
                        (a, b),
                        f"{a} and {b} share {color} and every non-color channel, "
                        "so a grayscale print cannot tell them apart",
                    )
                )
    return out


def _sharing_faults(motifs):
    present = set(motifs)
    out = []
    for pair in (*motifs.sharing, *motifs.exclusions):
        a, b = pair
        if a in present and b in present:
            out.append(
                MotifFault(
                    "sharing",
                    "fault",
                    (a, b),
                    f"{a} and {b} are declared never to appear together "
                    "(they share a color or a map family), and both are in the set",
                )
            )
    return out


def _budget_faults(motifs, mode):
    hues = _distinct_hues(_hue_motifs(motifs, mode))
    budget = motifs.hue_budget
    out = []
    for i, (key, _) in enumerate(hues):
        if i < budget:
            continue
        if i >= 6:
            out.append(
                MotifFault(
                    "budget",
                    "fault",
                    (key,),
                    f"{key} would be hue {i + 1}; never use more than six hues",
                )
            )
        elif not motifs[key].has_second_channel:
            out.append(
                MotifFault(
                    "budget",
                    "fault",
                    (key,),
                    f"{key} is hue {i + 1} of a budget of {budget} and has no "
                    "marker, hatch, arrow or font style as a second channel",
                )
            )
    return out


def _cvd_faults(motifs, mode, fault_de, warn_de):
    hues = _distinct_hues(_hue_motifs(motifs, mode))
    out = []
    for (ka, ca), (kb, cb) in itertools.combinations(hues, 2):
        if motifs.never_together(ka, kb):
            continue
        de = _worst(ca, cb)
        second = motifs[ka].signature() != motifs[kb].signature()
        if de < _HARD_FLOOR:
            severity = "fault"
        elif de < fault_de:
            severity = "warning" if second else "fault"
        elif de < warn_de and not second:
            severity = "warning"
        else:
            continue
        out.append(
            MotifFault(
                "cvd",
                severity,
                (ka, kb),
                f"{ka} and {kb} are {de:.1f} apart in CAM02-UCS under the "
                "worst vision type"
                + ("" if not second else ", softened by a second channel"),
                de,
            )
        )
    return out


def _dimmed_faults(motifs, mode, fault_de, dim):
    margins = motif_margins(motifs, mode=mode, dim=dim)
    de = margins["dim_min_de"]
    if de is None or de >= fault_de:
        return []
    a, b = margins["dim_min_pair"]
    return [
        MotifFault(
            "dimmed",
            "advisory",
            (a, b),
            f"{a} and {b} are {de:.1f} apart once dimmed to {margins['dim']:.2f}; "
            "rely on position, glyph and label for finished steps",
            de,
        )
    ]


def _cue_faults(motifs, mode):
    if "cue" not in motifs:
        return []
    cue = motifs["cue"]
    out = []
    text = hwostyle.styles.MODE_RC[_mode_name(mode)]["text.color"]
    cue_color = cue.resolve(mode)
    if _de(cue_color, to_hex(text)) > _CUE_SAME:
        out.append(
            MotifFault(
                "cue",
                "fault",
                ("cue",),
                f"the cue resolves to {cue_color}, not the text color; the cue is "
                "a form that spends no hue",
            )
        )
    if cue.marker or cue.arrowstyle or cue.fontstyle:
        out.append(
            MotifFault(
                "cue",
                "fault",
                ("cue",),
                "the cue is a ring, band or underlay, never a marker, arrow or text",
            )
        )
    for key, m in motifs.items():
        if key != "cue" and _de(m.resolve(mode), cue_color) <= _CUE_SAME:
            out.append(
                MotifFault(
                    "cue",
                    "fault",
                    ("cue", key),
                    f"{key} resolves to the cue's color, so a cue on it would not show",
                )
            )
    return out


def motif_faults(motifs, *, mode=None, fault_de=10.0, warn_de=20.0, dim=None):
    """Find the ways a motif set cannot be read.

    Args:
        motifs: A `MotifSet`, normally the subset one figure or clip uses.
        mode: A hwostyle mode name, or None for the active mode.
        fault_de: CAM02-UCS distance under which two hues are a fault
            (a warning when a second channel separates them).
        warn_de: Distance under which two hues without a second channel are
            a warning.
        dim: Contrast a dimmed motif keeps; None uses the mode's level.

    Returns:
        A list of `MotifFault`, faults first.
    """
    found = [
        *_signature_faults(motifs, mode),
        *_sharing_faults(motifs),
        *_budget_faults(motifs, mode),
        *_cvd_faults(motifs, mode, fault_de, warn_de),
        *_dimmed_faults(motifs, mode, fault_de, dim),
        *_cue_faults(motifs, mode),
    ]
    rank = {"fault": 0, "warning": 1, "advisory": 2}
    return sorted(found, key=lambda f: rank[f.severity])


def assert_motifs(motifs, *, strict=False, **kwargs):
    """Raise `MotifError` when a motif set has a fault.

    Args:
        motifs: A `MotifSet`.
        strict: Also refuse warnings.
        **kwargs: Passed to `motif_faults`.

    Raises:
        MotifError: Naming each finding.
    """
    refused = ("fault", "warning") if strict else ("fault",)
    bad = [f for f in motif_faults(motifs, **kwargs) if f.severity in refused]
    if bad:
        lines = [f"[{f.kind}] {f.severity}: {f.detail}" for f in bad]
        raise MotifError("motif set has findings:\n" + "\n".join(lines))


__all__ = [
    "MOTIF_KINDS",
    "MotifError",
    "MotifFault",
    "assert_motifs",
    "motif_faults",
    "motif_margins",
]
