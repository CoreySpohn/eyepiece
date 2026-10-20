"""Check the equation gallery at the first and last frames of each move."""

from types import SimpleNamespace

import hwostyle
import matplotlib.pyplot as plt
from test_docs_pages import DOCS, check_page

import eyepiece as ep
import eyepiece.testing as ept


def gallery_faults(text, monkeypatch):
    equations, faults = [], []
    equation = ep.equation

    def record_equation(*args, **kwargs):
        result = equation(*args, **kwargs)
        equations.append(result)
        return result

    def animate(fig, draw, n, **kwargs):
        for frame in (0, n - 1):
            faults.extend(
                ept.grounding_faults(
                    equations, at=lambda t, frame=frame: draw(fig, frame), times=[0]
                )
            )
        equations.clear()
        return SimpleNamespace(jshtml=lambda dpi: "")

    monkeypatch.setattr(ep, "equation", record_equation)
    monkeypatch.setattr(ep, "animate", animate)
    namespace = {}
    try:
        for index, cell in enumerate(check_page(text)):
            if index == 0 or "ep.equation(" in cell:
                exec(compile(cell, "motion-equations.md", "exec"), namespace)
        assert not equations, "Equations were built without checking their frames"
    finally:
        plt.close("all")
    return faults


def test_gallery_equations_are_grounded(monkeypatch):
    page = DOCS / "gallery" / "motion-equations.md"
    assert gallery_faults(page.read_text(), monkeypatch) == []


def test_old_bayes_colors_report_missing_referents():
    hwostyle.use("dark")
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    anchor = ep.Anchor.from_data(ax, (0.06, 0.5))
    try:
        first = ep.equation(
            anchor,
            r"${{lhs:\mathrm{Pr}(H|E)=}}\frac{{num:\mathrm{Pr}(E|H)\,\mathrm{Pr}(H)}}"
            r"{{den:\mathrm{Pr}(E)}}$",
            fontsize=34,
            colors={"lhs": "C0", "num": "C1", "den": "C3", "res": "C2"},
            ha="left",
        )
        last = ep.equation(
            anchor,
            r"${{lhs:\mathrm{Pr}(H|E)=}}\frac{{num:0.24}}"
            r"{{den:0.24+0.07}}{{res:=0.77}}$",
            fontsize=34,
            colors={"lhs": "C0", "num": "C1", "den": "C3", "res": "C2"},
            ha="left",
        )
        move = ep.transform(first, last, stagger=0.55)
        faults = ept.grounding_faults([first, last], at=move.update, times=[0, 1])
        assert any(fault.kind == "no-referent" for fault in faults)
    finally:
        plt.close(fig)


def test_bound_colored_term_is_grounded():
    hwostyle.use("dark")
    fig, ax = plt.subplots()
    try:
        (line,) = ax.plot([0, 1], [0.2, 0.4], color="C1")
        eq = ep.equation(ax, r"${{flux:F_p}}$", fontsize=24, colors={"flux": "C1"})
        eq.bind("flux", line)
        assert ept.grounding_faults(eq) == []
    finally:
        plt.close(fig)
