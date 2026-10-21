"""Keep the capability catalog complete, and every name it lists real."""

import importlib
import inspect
import re
import warnings
from pathlib import Path

import pytest

import eyepiece

CATALOG = Path(__file__).resolve().parents[1] / "docs" / "reference" / "catalog.md"
NAMESPACES = (
    "eyepiece",
    "eyepiece.plot",
    "eyepiece.clock",
    "eyepiece.motion",
    "eyepiece.narration",
    "eyepiece.testing",
    "eyepiece.render",
    "eyepiece.prepared",
    "eyepiece.motifs",
    "eyepiece.mpl",
    "eyepiece.manim",
    "eyepiece.style",
)

# Public names the catalog may leave out, each with the reason. Empty while
# the catalog lists every public name; add an entry here, not a silent skip.
NOT_CATALOGED = {}

# Deprecated aliases, as the catalog writes them, and their replacements. The
# task tables must use the replacement; the "Deprecated names" table lists both.
DEPRECATED = {
    "testing.Span": "testing.Window",
    "spans": "plot.shade_intervals",
    "radial_profile_plot": "plot.plot_radial_image",
    "motifs.resolve_role": "style.role",
    "tl.pulse": "tl.indicate",
    "tl.ring": "tl.circumscribe",
}

# Prefixes the catalog uses, as its introduction states them.
PREFIXES = {
    "ep": "eyepiece",
    **{ns.removeprefix("eyepiece."): ns for ns in NAMESPACES[1:]},
    "synth": "eyepiece.narration.synth",
}
# Existing aliases remain callable, but the catalog has one path per task.
CANONICAL_ALIASES = {
    "clock.Fault": "testing.Fault",
    "render.frame_faults": "testing.frame_faults",
    "render.assert_frames_pure": "testing.assert_frames_pure",
    "render.same_picture": "testing.same_picture",
    "testing.caption_line": "narration.caption_line",
}
INSTANCES = {"tl": eyepiece.Timeline, "plan": eyepiece.Plan, "stage": eyepiece.Stage}
NAME = re.compile(r"[A-Za-z_][\w.]*")


def _module(name):
    """Import a namespace, skipping the optional Manim one when it is absent."""
    if name == "eyepiece.manim":
        pytest.importorskip("manim")
    return importlib.import_module(name)


def _catalog_paths():
    """Qualified paths in task tables, excluding deprecated aliases."""
    paths = set()
    for span in _use_cells():
        match = NAME.match(span)
        if not match:
            continue
        parts = match.group(0).rstrip(".").split(".")
        paths.add(".".join(parts))
        paths.add(".".join(parts[:2]) if parts[0] in PREFIXES else parts[0])
    return paths


def _canonical_path(namespace, name):
    if namespace == "eyepiece":
        return name
    path = f"{namespace.removeprefix('eyepiece.')}.{name}"
    return CANONICAL_ALIASES.get(path, path)


def _use_cells():
    """Backticked spans of the 'Use' column of every catalog table."""
    spans = []
    for line in CATALOG.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) != 4:
            continue
        if cells[1] == "Use" or set(cells[1]) <= {"-"}:
            continue
        spans += re.findall(r"`([^`]+)`", cells[1])
    return spans


def _documented_attribute(owner, name):
    """An instance attribute a class declares as a field or in its docstring."""
    if name in getattr(owner, "__dataclass_fields__", {}):
        return True
    doc = inspect.getdoc(owner) or ""
    return re.search(rf"^\s+{re.escape(name)}:", doc, re.M) is not None


def _resolve(dotted):
    """The object a catalog name refers to, or None for a documented field."""
    head, *rest = dotted.split(".")
    if head in INSTANCES:
        obj = INSTANCES[head]
    elif head in PREFIXES:
        obj = _module(PREFIXES[head])
    else:
        assert head in eyepiece.__all__ or dotted in DEPRECATED, (
            f"{head!r} needs its canonical task namespace"
        )
        obj = getattr(eyepiece, head)
    for part in rest:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif inspect.isclass(obj) and _documented_attribute(obj, part):
            return None
        else:
            raise AssertionError(f"{dotted!r}: no attribute {part!r}")
    return obj


@pytest.mark.parametrize("namespace", NAMESPACES)
def test_catalog_lists_every_public_name(namespace):
    public = getattr(_module(namespace), "__all__", ())
    paths = _catalog_paths()
    missing = sorted(
        _canonical_path(namespace, name)
        for name in public
        if _canonical_path(namespace, name) not in paths
        and not (name in eyepiece.__all__ and name in paths)
        and name not in NOT_CATALOGED
    )
    assert not missing, f"{namespace} names missing from catalog.md: {missing}"


def test_not_cataloged_names_are_public():
    public = set()
    for namespace in NAMESPACES:
        try:
            public.update(getattr(_module(namespace), "__all__", ()))
        except pytest.skip.Exception:
            continue
    assert set(NOT_CATALOGED) <= public


@pytest.mark.parametrize("span", _use_cells())
def test_catalog_names_resolve(span):
    match = NAME.match(span)
    assert match, f"Use column entry is not a name: {span!r}"
    obj = _resolve(match.group(0).rstrip("."))
    call = span[match.end() :]
    if obj is None or not call.startswith("(") or not callable(obj):
        return
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return
    for keyword in re.findall(r"(\w+)=", call.split(")")[0]):
        assert keyword in params, f"{span!r}: no parameter {keyword!r}"


@pytest.mark.parametrize("span", _use_cells())
def test_catalog_uses_no_deprecated_alias(span):
    match = NAME.match(span)
    if match:
        name = match.group(0).rstrip(".")
        assert name not in DEPRECATED, f"{span!r}: use {DEPRECATED.get(name)}"


@pytest.mark.parametrize("span", _use_cells())
def test_task_rows_use_canonical_paths(span):
    match = NAME.match(span)
    assert match, span
    name = match.group(0).rstrip(".")
    head, *rest = name.split(".")
    if head == "ep":
        assert rest, span
        assert rest[0] in eyepiece.__all__ or rest[0] in PREFIXES, span
        name = name.removeprefix("ep.")
        head, *rest = name.split(".")
    assert head in eyepiece.__all__ or head in INSTANCES or head in PREFIXES, span
    assert name not in CANONICAL_ALIASES, f"{span!r}: use {CANONICAL_ALIASES.get(name)}"
    if head in PREFIXES and head not in {"ep", "synth"}:
        assert rest, span
        assert rest[0] in _module(PREFIXES[head]).__all__, span


@pytest.mark.parametrize(("old", "new"), DEPRECATED.items())
def test_deprecated_aliases_listed_and_still_resolve(old, new):
    text = CATALOG.read_text()
    section = text[text.index("## Deprecated names") :]
    row = next(line for line in section.splitlines() if f"`{old}(" in line)
    assert f"`{new}(" in row, f"{old} row does not name {new}"
    # Some aliases warn on access and some on call; tests/test_deprecated_names.py
    # pins the warnings. Here the alias must still resolve beside its replacement.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert _resolve(old) is not None
    assert _resolve(new) is not None


def test_package_docstring_names_the_public_submodules():
    for namespace in NAMESPACES[1:]:
        name = namespace.removeprefix("eyepiece.")
        assert f"``{name}``" in eyepiece.__doc__
