"""Keep documentation pages linked, executable, and plain ASCII."""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
EXPLANATION_NAMES = (
    "contract",
    "clock",
    "evidence",
    "narration-design",
    "prepared-views",
    "eyepiece-and-manim",
    "rendering-in-manim",
)
PAGES = [
    *sorted((DOCS / "tutorials").glob("*.md")),
    *sorted((DOCS / "how-to").glob("*.md")),
    *(DOCS / "explanation" / f"{name}.md" for name in EXPLANATION_NAMES),
]
CONTRACTION = re.compile(
    r"\b\w+n't\b|\b(?:it|that|there|here|what|who)'s\b|\b\w+'(?:re|ve|ll|d|m)\b",
    re.IGNORECASE,
)


def split_fences(text):
    """Separate prose and fenced blocks, preserving directive headers."""
    prose, blocks, body = [], [], []
    fence = None
    for line in text.splitlines():
        if fence is None:
            opening = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
            if opening:
                fence, info = opening.groups()
                body = []
            else:
                prose.append(line)
        elif re.fullmatch(rf"\s*{re.escape(fence[0])}{{{len(fence)},}}\s*", line):
            blocks.append((info.strip(), "\n".join(body)))
            fence = None
        else:
            body.append(line)
    assert fence is None, "Unclosed code fence"
    return prose, blocks


def check_page(text):
    """Validate prose rules and return executable Python cells."""
    assert text.isascii(), "Non-ASCII text"
    prose, blocks = split_fences(text)
    for line in prose:
        assert "\u2014" not in line, "Em dash in prose"
        assert not CONTRACTION.search(line), f"Contraction in prose: {line}"
    return [body for info, body in blocks if info == "{code-cell} python"]


def toctree_pages():
    """Resolve page entries relative to the document containing each tree."""
    pages = set()
    for source in DOCS.rglob("*.md"):
        _, blocks = split_fences(source.read_text())
        for info, body in blocks:
            if info != "{toctree}":
                continue
            for line in body.splitlines():
                entry = line.strip()
                if not entry or entry.startswith(":"):
                    continue
                entry = re.sub(r"^.*<([^>]+)>$", r"\1", entry)
                base = DOCS if entry.startswith("/") else source.parent
                pages.add((base / entry.lstrip("/")).with_suffix(".md").resolve())
    return pages


@pytest.mark.parametrize("page", PAGES, ids=lambda path: str(path.relative_to(DOCS)))
def test_docs_page(page):
    cells = check_page(page.read_text())
    assert page.resolve() in toctree_pages(), f"Page missing from toctrees: {page}"
    for cell in cells:
        compile(cell, str(page), "exec")


@pytest.mark.parametrize(
    "text",
    [
        "A sentence\u2014with a dash.",
        "It isn't ready.",
        "It's ready.",
        "THAT'S ready.",
        "There's a marker.",
        "Here's a marker.",
        "What's next?",
        "Who's next?",
        "We're ready.",
        "They've arrived.",
        "It'll move.",
        "I'd move it.",
        "I'm ready.",
        "A caf\u00e9.",
        "```{code-cell} python\nlabel = '\u03b1'\n```",
    ],
)
def test_reject_invalid_page(text):
    with pytest.raises(AssertionError):
        check_page(text)


def test_code_fences_allow_ascii_contractions():
    text = 'A marker.\n```{code-cell} python\nlabel = "it\'s ready"\n```\n'
    assert check_page(text) == ['label = "it\'s ready"']


@pytest.mark.parametrize(
    "page", [*sorted(DOCS.rglob("*.md")), DOCS.parent / "README.md"]
)
def test_doc_links_resolve(page):
    for link in re.findall(r"\{doc\}`([^`]+)`", page.read_text()):
        target = re.sub(r"^.*<([^>]+)>$", r"\1", link)
        base = DOCS if target.startswith("/") else page.parent
        resolved = (base / target.lstrip("/")).with_suffix(".md")
        assert resolved.is_file(), f"Broken doc link in {page}: {link}"


def test_all_docs_pages_in_toctrees():
    pages = {page.resolve() for page in DOCS.rglob("*.md")}
    assert pages - {DOCS / "index.md"} <= toctree_pages()


def test_explanation_toctree_order():
    _, blocks = split_fences((DOCS / "index.md").read_text())
    tree = next(
        body
        for info, body in blocks
        if info == "{toctree}" and ":caption: Explanation" in body
    )
    entries = [line for line in tree.splitlines() if line and not line.startswith(":")]
    assert entries == [f"explanation/{name}" for name in EXPLANATION_NAMES] + [
        "viz-convention"
    ]


def test_manim_parity_cell_lists_all_rows():
    page = DOCS / "explanation" / "eyepiece-and-manim.md"
    cells = check_page(page.read_text())
    assert cells, "The parity table needs an executable cell"
    code = "\n".join(cells)
    for name in (
        "FadeIn",
        "Create",
        "Indicate",
        "Circumscribe",
        "Write",
        "Transform",
        "TransformMatchingTex",
        "rate functions",
    ):
        assert re.search(rf'^\s+"{re.escape(name)}[", ]', code, re.M), name
