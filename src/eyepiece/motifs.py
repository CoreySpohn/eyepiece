"""Motifs: one keyed vocabulary for the objects that recur across figures.

A star, a planet, a detector or a reference line should look the same in
every figure and every clip. A `Motif` is one such object's drawing style:
the color it takes by role or neutral rank, and the channels besides color
(marker, line style, hatch, arrow style, font style) that identify it when
color is shared, dimmed or lost to color vision. A `MotifSet` holds motifs by
key, never by position, so the key ``"planet"`` is the same planet whatever
else a set lists.

A motif never stores a hex value. Its color resolves at call time through
`eyepiece.role`, so a mode switch is seen, and a saved figure state keeps the
motif key and not a color that would go stale. `MotifSet.default()` reads the
default table from `hwostyle.motifs`; an older hwostyle without it gets a
smaller stand-in table, as `eyepiece.role` does for a missing role.

A set describes the vocabulary a figure or clip draws with, so check the
subset it uses (`MotifSet.only`) with `eyepiece.testing.motif_faults`, not the
whole default table, which holds hue pairs that are never meant to appear
together.

Example::

    import eyepiece as ep

    motifs = ep.MotifSet.default()
    ax.plot(x, y, **motifs["model"].line_kw())
    ax.plot(x, y0, **motifs["truth"].line_kw())
"""

import dataclasses
from collections.abc import Mapping

import hwostyle
from matplotlib.colors import to_hex, to_rgba

from eyepiece import _style

# The hue roles: every color role except the neutral gray ``truth``. A motif
# whose role is in here spends a hue; any other motif is a gray.
_NEUTRAL_ROLES = ("truth",)
_FIELDS = (
    "key",
    "noun",
    "role",
    "marker",
    "linestyle",
    "hatch",
    "arrowstyle",
    "fontstyle",
    "linewidth",
    "fill_alpha",
    "fill_alpha_dark",
    "on_image",
    "label_color",
)

# Stand-ins when hwostyle lacks `motifs`: a row-for-row copy of its table.
# The hwostyle table is the source of truth; a parity test keeps this copy
# equal to it, and it exists only so an older hwostyle still works.
_FALLBACK_RANKS = {
    "background": 0.0,
    "furniture": 0.2,
    "scenery": 0.45,
    "threshold": 0.6,
    "data_flow": 0.6,
    "hardware": 0.7,
    "integration": 0.8,
    "text": 1.0,
    "truth": None,
}
_FALLBACK_SHARING = (
    ("disk", "prior"),
    ("local_zodi", "reference"),
    ("local_zodi", "alert"),
    ("local_zodi", "posterior"),
)
_FALLBACK_EXCLUSIONS = (("pupil", "probability"), ("pupil", "statistic"))
_FALLBACK_ROWS = (
    {
        "key": "star",
        "noun": "star",
        "role": "star",
        "marker": "*",
        "linestyle": "none",
        "on_image": ("+", "text"),
    },
    {
        "key": "starlight",
        "noun": "starlight",
        "role": "star",
        "marker": ".",
        "linestyle": "none",
        "hatch": "..",
    },
    {
        "key": "planet",
        "noun": "planet",
        "role": "planet",
        "marker": "o",
        "linestyle": "none",
        "on_image": ("o", "text"),
    },
    {
        "key": "disk",
        "noun": "exozodiacal dust",
        "role": "disk",
        "hatch": "///",
        "linewidth": 0.8,
        "fill_alpha": 0.12,
        "fill_alpha_dark": 0.22,
        "label_color": "disk",
    },
    {
        "key": "local_zodi",
        "noun": "local zodiacal dust",
        "role": "zodi",
        "hatch": "..",
        "linewidth": 0.8,
        "fill_alpha": 0.10,
        "fill_alpha_dark": 0.18,
        "label_color": "zodi",
    },
    {
        "key": "aperture",
        "noun": "aperture",
        "role": "hardware",
        "marker": "|",
        "linewidth": 3.0,
    },
    {
        "key": "optics",
        "noun": "instrument optics",
        "role": "hardware",
        "marker": "d",
        "fill_alpha": 0.15,
    },
    {
        "key": "detector",
        "noun": "detector",
        "role": "hardware",
        "marker": "s",
        "hatch": "xx",
        "fill_alpha": 0.10,
    },
    {
        "key": "pixel",
        "noun": "pixel",
        "role": "planet",
        "marker": "s",
        "fill_alpha": 0.30,
        "on_image": ("s", "text"),
    },
    {"key": "ray", "noun": "light ray", "role": "star", "arrowstyle": "-|>"},
    {
        "key": "integration",
        "noun": "integration direction",
        "role": "integration",
        "linestyle": "--",
        "arrowstyle": "]->",
    },
    {
        "key": "data_flow",
        "noun": "data flow",
        "role": "data_flow",
        "arrowstyle": "simple",
        "linewidth": 0.8,
        "fill_alpha": 0.18,
    },
    {
        "key": "reference_plane",
        "noun": "reference plane",
        "role": "scenery",
        "linestyle": "-.",
        "fontstyle": "italic",
        "linewidth": 0.8,
        "fill_alpha": 0.10,
        "fill_alpha_dark": 0.25,
    },
    {
        "key": "speckle",
        "noun": "speckle",
        "role": "star",
        "marker": "x",
        "linestyle": "none",
        "on_image": ("x", "text"),
    },
    {
        "key": "data",
        "noun": "measurement",
        "role": "data",
        "marker": "|",
        "linestyle": "none",
        "on_image": ("|", "text"),
    },
    {"key": "model", "noun": "model", "role": "model"},
    {"key": "truth", "noun": "true value", "role": "truth", "linestyle": "--"},
    {"key": "prior", "noun": "prior", "role": "prior", "label_color": "prior"},
    {"key": "posterior", "noun": "posterior", "role": "posterior", "linewidth": 1.6},
    {
        "key": "threshold",
        "noun": "threshold",
        "role": "threshold",
        "linestyle": ":",
        "fontstyle": "italic",
    },
    {
        "key": "alert",
        "noun": "alert",
        "role": "alert",
        "marker": "X",
        "linestyle": "none",
        "on_image": ("X", "text"),
    },
    {
        "key": "positive",
        "noun": "certified claim",
        "role": "positive",
        "marker": "o",
        "linestyle": "none",
    },
    {
        "key": "negative",
        "noun": "rejected claim",
        "role": "negative",
        "marker": "s",
        "linestyle": "none",
    },
    {"key": "cue", "noun": "cue", "role": "text", "linewidth": 2.0, "fill_alpha": 0.25},
)


@dataclasses.dataclass(frozen=True)
class AcceptedPair:
    """Two co-occurring hues left under 10 dE on purpose.

    Attributes:
        a: Role name of the first hue.
        b: Role name of the second hue.
        modes: Modes in which the pair is accepted.
        channels: Motif fields (``marker``, ``linestyle``, ``hatch``,
            ``arrowstyle``, ``fontstyle``) that must differ between the two
            motifs for the pair to be accepted.
        de: Measured CAM02-UCS distance of the pair.
    """

    a: str
    b: str
    modes: tuple
    channels: tuple
    de: float


# Row-for-row copy of hwostyle's ``ACCEPTED_PAIRS``, kept equal by the parity test.
_FALLBACK_ACCEPTED = (
    AcceptedPair("reference", "alert", ("dark",), ("linestyle", "marker"), 7.3),
    AcceptedPair("model", "negative", ("dark",), ("linestyle", "marker"), 9.6),
    AcceptedPair("reference", "negative", ("paper",), ("marker",), 6.6),
    AcceptedPair("disk", "positive", ("paper",), ("marker",), 7.7),
)


def _hwostyle_table():
    """The `hwostyle.motifs` module, or None when hwostyle predates it."""
    try:
        import hwostyle.motifs as table
    except ImportError:
        return None
    return table


def _ranks():
    table = _hwostyle_table()
    return dict(table.NEUTRAL_RANKS if table is not None else _FALLBACK_RANKS)


def _role_in(name, mode):
    """Hex color of a role, in a named mode or in the active one."""
    if mode is None:
        return _style.role(name)
    from hwostyle.palettes import Palette, Roles

    from eyepiece._video import VIDEO_MODE, base_mode, video_roles

    if mode == VIDEO_MODE and name in video_roles():
        return video_roles()[name]
    mode = base_mode(mode)
    try:
        return to_hex(Roles(mode)[name])
    except (KeyError, TypeError):
        slot = _style._ROLE_FALLBACKS[name]
        return to_hex(slot) if isinstance(slot, str) else to_hex(Palette(mode)[slot])


def _neutral_in(level, mode):
    """Hex gray `level` of the way from the background to the text color."""
    if mode is None:
        return to_hex(_style.neutral(level))
    from eyepiece._video import base_mode

    rc = hwostyle.styles.MODE_RC[base_mode(mode)]
    back = to_rgba(rc["axes.facecolor"])[:3]
    front = to_rgba(rc["text.color"])[:3]
    return to_hex(tuple(b + (f - b) * level for b, f in zip(back, front, strict=True)))


def _label_in(role, mark, mode):
    """Label ink of a role in a mode, or `mark` when there is none."""
    if role is None:
        return mark
    from eyepiece._video import VIDEO_MODE, video_label_inks

    if (mode or _style.mode()) == VIDEO_MODE:
        return video_label_inks().get(role, mark)
    if mode is None:
        roles = getattr(hwostyle, "roles", None) if hwostyle.current_mode() else None
    else:
        from hwostyle.palettes import Roles

        roles = Roles(mode)
    ink = getattr(roles, "label_ink", None)
    if ink is None:
        return mark
    try:
        return to_hex(ink(role))
    except KeyError:
        return mark


def resolve_role(role, mode=None):
    """Hex color for a motif role or neutral rank.

    Args:
        role: A hwostyle color role (``"star"``, ``"model"``) or a neutral rank
            (``"hardware"``, ``"text"``).
        mode: A hwostyle mode name, or None for the active mode.

    Returns:
        A hex string.

    Raises:
        KeyError: If the name is neither a role nor a rank.
    """
    ranks = _ranks()
    if role in ranks:
        level = ranks[role]
        if level is None:
            return _role_in(role, mode)
        return _neutral_in(level, mode)
    return _role_in(role, mode)


def is_hue_role(role):
    """Whether a role or rank spends a hue (not a gray, not a rank)."""
    return role not in _ranks() and role not in _NEUTRAL_ROLES


@dataclasses.dataclass(frozen=True)
class Motif:
    """The drawing style of one recurring object.

    The color is a role name or neutral rank and resolves when asked for, so
    the same motif is the right color in dark, light and paper modes and a
    stored state needs no hex.

    Attributes:
        key: The motif key, for example ``"star"``.
        noun: A plain display name for labels.
        role: A hwostyle color role or a neutral rank. Never a hex string.
        marker: Marker when the motif is drawn as a point, else None.
        linestyle: Line style of a line or outline; ``"none"`` for a point.
        hatch: Hatch pattern when drawn as a region.
        arrowstyle: Arrow style when drawn as an arrow.
        fontstyle: Font style of its label.
        linewidth: Line width in points, as the consumer scales it.
        fill_alpha: Face opacity of a region on a white page.
        fill_alpha_dark: Face opacity of a region on a dark page, or None
            to use `fill_alpha`.
        on_image: ``(glyph, ink)`` drawn over an image in place of the hue,
            where the hue would collide with the map: a marker and a rank.
        label_color: A role whose label ink colors this motif's text, for a
            hue whose mark is too faint to read as words; None labels in the
            mark color.
    """

    key: str
    noun: str
    role: str
    marker: str | None = None
    linestyle: str = "-"
    hatch: str | None = None
    arrowstyle: str | None = None
    fontstyle: str | None = None
    linewidth: float = 1.0
    fill_alpha: float = 0.0
    fill_alpha_dark: float | None = None
    on_image: tuple | None = None
    label_color: str | None = None

    def __post_init__(self):
        """Refuse a hex role: a motif stores a role name, never a color."""
        if str(self.role).startswith("#"):
            raise ValueError(
                f"{self.key}: role {self.role!r} is a hex color; use a role name"
            )

    @property
    def color(self):
        """The color in the active mode, resolved now."""
        return resolve_role(self.role)

    def resolve(self, mode=None):
        """The color in a named mode, or in the active mode when None."""
        return resolve_role(self.role, mode)

    def resolve_label(self, mode=None):
        """Text color for this motif's label in a mode (the active one when None).

        Returns the label ink of ``label_color`` when hwostyle has one for this
        mode and palette family, else the mark color.
        """
        return _label_in(self.label_color, self.resolve(mode), mode)

    @property
    def label(self):
        """The label text color in the active mode, resolved now."""
        return self.resolve_label()

    @property
    def is_hue(self):
        """Whether this motif spends a hue rather than a gray."""
        return is_hue_role(self.role)

    @property
    def has_second_channel(self):
        """Whether a channel beyond color and a plain line identifies it.

        A marker, hatch, arrow style, font style or a line style other than
        solid (or ``"none"``, which a marker implies) counts.
        """
        return bool(
            self.marker
            or self.hatch
            or self.arrowstyle
            or self.fontstyle
            or self.linestyle not in ("-", "none")
        )

    def signature(self):
        """The channels other than color that identify this motif.

        Returns:
            ``(marker, linestyle, hatch, arrowstyle, fontstyle)``. No two
            motifs that share a color share a signature, so a grayscale
            print still separates them.
        """
        return (
            self.marker,
            self.linestyle,
            self.hatch,
            self.arrowstyle,
            self.fontstyle,
        )

    def fill(self, mode=None):
        """Face opacity of a region in a mode (the active one when None)."""
        from eyepiece._video import base_mode

        name = base_mode(mode or _style.mode())
        if name == "dark" and self.fill_alpha_dark is not None:
            return self.fill_alpha_dark
        return self.fill_alpha

    def line_kw(self, *, gid=None):
        """Keyword arguments for ``ax.plot`` of this motif as a line.

        Args:
            gid: Artist identifier; None uses the motif key.

        Returns:
            A dict of line properties, including ``gid``.
        """
        return {
            "color": self.color,
            "ls": self.linestyle,
            "lw": self.linewidth,
            "gid": self.key if gid is None else gid,
        }

    def label_kw(self, *, gid=None):
        """Keyword arguments for ``ax.text`` of this motif's label.

        Args:
            gid: Artist identifier; None uses the motif key.

        Returns:
            A dict of label properties, including ``gid``.
        """
        kw = {"color": self.label, "gid": self.key if gid is None else gid}
        if self.fontstyle:
            kw["fontstyle"] = self.fontstyle
        return kw

    def marker_kw(self, size=None, *, gid=None):
        """Keyword arguments for ``ax.plot`` of this motif as a point.

        Args:
            size: Marker size in points; None keeps the rcParams default.
            gid: Artist identifier; None uses the motif key.

        Returns:
            A dict of marker properties, including ``gid``.
        """
        kw = {
            "color": self.color,
            "marker": self.marker,
            "ls": "none",
            "gid": self.key if gid is None else gid,
        }
        if size is not None:
            kw["ms"] = size
        return kw

    def patch_kw(self, *, gid=None):
        """Keyword arguments for a matplotlib patch of this motif as a region.

        Args:
            gid: Artist identifier; None uses the motif key.

        Returns:
            A dict of patch properties, including ``gid``.
        """
        color = self.color
        kw = {
            "facecolor": to_rgba(color, self.fill()),
            "edgecolor": color,
            "lw": self.linewidth,
            "ls": self.linestyle if self.linestyle != "none" else "-",
            "gid": self.key if gid is None else gid,
        }
        if self.hatch:
            kw["hatch"] = self.hatch
            kw["hatchcolor"] = color
        return kw

    def derive(self, **changes):
        """A copy with some fields changed, for a chapter's own vocabulary.

        Args:
            **changes: Any field of the motif, for example ``noun="issued
                report"``.

        Returns:
            A new `Motif`.
        """
        return dataclasses.replace(self, **changes)


def _motif_from_row(row):
    if isinstance(row, Motif):
        return row
    if isinstance(row, Mapping):
        fields = row
    else:
        fields = {
            name: getattr(row, _row_name(name))
            for name in _FIELDS
            if hasattr(row, _row_name(name))
        }
    fields = dict(fields)
    fields.setdefault("noun", fields["key"].replace("_", " "))
    return Motif(**{k: v for k, v in fields.items() if k in _FIELDS})


def _row_name(field):
    """The hwostyle table calls the role field ``color``."""
    return "color" if field == "role" else field


class MotifSet(Mapping):
    """The motifs of a figure or clip, keyed by name.

    Index by motif key. A set is immutable: `only`, `derive` and `add`
    return new sets. Colors resolve when read, so one set serves every mode.

    Args:
        motifs: `Motif` records, or a mapping of key to `Motif`.
        sharing: Pairs of keys or role names that share a hue and never
            appear in the same clip.
        exclusions: Pairs that share a map family and never share a figure.
        hue_budget: Most hue motifs one figure spends.
        accepted: `AcceptedPair` records: co-occurring hues left under 10 dE
            that a checker accepts when their motifs differ in the listed
            channels.

    Raises:
        ValueError: If two motifs have the same key.
    """

    def __init__(self, motifs, *, sharing=(), exclusions=(), hue_budget=4, accepted=()):
        """Hold the motifs in the order given."""
        items = motifs.values() if isinstance(motifs, Mapping) else motifs
        table = {}
        for motif in items:
            if motif.key in table:
                raise ValueError(f"{motif.key}: duplicate key in MotifSet")
            table[motif.key] = motif
        self._motifs = table
        self.sharing = tuple(tuple(pair) for pair in sharing)
        self.exclusions = tuple(tuple(pair) for pair in exclusions)
        self.hue_budget = int(hue_budget)
        self.accepted = tuple(accepted)

    @classmethod
    def default(cls):
        """The default table from `hwostyle.motifs`.

        Falls back to a row-for-row copy of the table when the installed hwostyle
        predates it. The table is the whole vocabulary, not one figure's: it
        holds hue pairs that never share a clip, so check `only(...)` the
        keys a clip uses.
        """
        table = _hwostyle_table()
        if table is None:
            rows = [Motif(**r) for r in _FALLBACK_ROWS]
            return cls(
                rows,
                sharing=_FALLBACK_SHARING,
                exclusions=_FALLBACK_EXCLUSIONS,
                accepted=_FALLBACK_ACCEPTED,
            )
        return cls(
            [_motif_from_row(row) for row in table.MOTIFS.values()],
            sharing=table.SHARING_SETS,
            exclusions=table.EXCLUSION_PAIRS,
            hue_budget=table.HUE_BUDGET,
            accepted=[
                AcceptedPair(r.a, r.b, tuple(r.modes), tuple(r.channels), r.de)
                for r in getattr(table, "ACCEPTED_PAIRS", ())
            ],
        )

    def __getitem__(self, key):
        """Motif for `key`."""
        try:
            return self._motifs[key]
        except KeyError:
            raise KeyError(
                f"No motif {key!r}. Available: {list(self._motifs)}"
            ) from None

    def __iter__(self):
        """Iterate over motif keys in declaration order."""
        return iter(self._motifs)

    def __len__(self):
        """Number of motifs."""
        return len(self._motifs)

    def meaning_keys(self) -> dict:
        """Map motif gid prefixes to meaning keys for meaning checks.

        Returns:
            A dict mapping each motif key to itself, for the ``keys`` argument
            of `eyepiece.testing.meaning_faults`.

        Example::

            motifs = MotifSet.default().only("model", "truth")
            faults = ept.meaning_faults(fig, keys=motifs.meaning_keys())
        """
        return {key: key for key in self}

    def _like(self, motifs):
        return MotifSet(
            motifs,
            sharing=self.sharing,
            exclusions=self.exclusions,
            hue_budget=self.hue_budget,
            accepted=self.accepted,
        )

    def only(self, *keys):
        """The subset a figure or clip uses, in the order given.

        Raises:
            KeyError: If a key is not in the set.
        """
        return self._like([self[k] for k in keys])

    def derive(self, key, **changes):
        """A set with the motif at `key` changed.

        Args:
            key: Key of a motif in the set.
            **changes: Fields of the motif to change.

        Raises:
            KeyError: If `key` is not in the set. Use `add` with
                ``motif.derive(key=...)`` to add a variant under a new key.
        """
        self[key]
        return self._like(
            [m.derive(**changes) if m.key == key else m for m in self.values()]
        )

    def add(self, *motifs):
        """A set with more motifs, for a chapter's own vocabulary."""
        return self._like([*self.values(), *motifs])

    def never_together(self, a, b):
        """Whether two keys or role names are declared never to co-occur."""
        pair = {a, b}
        return any(pair == set(p) for p in (*self.sharing, *self.exclusions))

    def accepts(self, a, b, mode):
        """Whether the motifs at keys `a` and `b` form an accepted pair in `mode`.

        A pair is accepted when its roles are listed for the mode and the two
        motifs differ in every channel the record requires.
        """
        from eyepiece._video import base_mode

        ma, mb = self[a], self[b]
        mode = base_mode(mode)
        for rec in self.accepted:
            if mode not in rec.modes:
                continue
            if (ma.role, mb.role) == (rec.a, rec.b):
                pair = (ma, mb)
            elif (ma.role, mb.role) == (rec.b, rec.a):
                pair = (mb, ma)
            else:
                continue
            if all(getattr(pair[0], f) != getattr(pair[1], f) for f in rec.channels):
                return True
        return False

    @property
    def dim(self):
        """Contrast a dimmed motif keeps in the active mode."""
        return _style.dim_level()


__all__ = ["AcceptedPair", "Motif", "MotifSet", "is_hue_role", "resolve_role"]
