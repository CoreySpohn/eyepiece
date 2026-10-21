"""Keep the capability catalog complete, and every name it lists real."""

import importlib
import inspect
import re
from pathlib import Path

import pytest

import eyepiece

CATALOG = Path(__file__).resolve().parents[1] / "docs" / "reference" / "catalog.md"
NAMESPACES = (
    "eyepiece",
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

# Prefixes the catalog uses, as its introduction states them.
PREFIXES = {
    "ep": "eyepiece",
    "nr": "eyepiece.narration",
    "synth": "eyepiece.narration.synth",
    "ept": "eyepiece.testing",
    "render": "eyepiece.render",
    "motifs": "eyepiece.motifs",
    "style": "eyepiece.style",
    "mpl": "eyepiece.mpl",
    "manim": "eyepiece.manim",
    "prepared": "eyepiece.prepared",
}
INSTANCES = {"tl": eyepiece.Timeline, "plan": eyepiece.Plan, "stage": eyepiece.Stage}
NAME = re.compile(r"[A-Za-z_][\w.]*")


def _module(name):
    """Import a namespace, skipping the optional Manim one when it is absent."""
    if name == "eyepiece.manim":
        pytest.importorskip("manim")
    return importlib.import_module(name)


def _catalog_words():
    words = set()
    for span in re.findall(r"`([^`]+)`", CATALOG.read_text()):
        words.update(re.findall(r"[A-Za-z_]\w*", span))
    return words


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
        owners = [ns for ns in NAMESPACES if ns != "eyepiece.manim"]
        found = [
            getattr(_module(ns), head) for ns in owners if hasattr(_module(ns), head)
        ]
        assert found, f"{head!r} is not public in any eyepiece namespace"
        obj = found[0]
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
    missing = sorted(
        name
        for name in public
        if name not in _catalog_words() and name not in NOT_CATALOGED
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
