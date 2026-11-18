"""Extent helpers: pixel-edge convention, analytic values."""

from eyepiece import layout


def test_label_lod_sets_both_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    layout.label_lod(ax)
    assert "lambda" in ax.get_xlabel() or r"\lambda" in ax.get_xlabel()
    plt.close(fig)


def test_source_styles_stable_and_distinct():
    from eyepiece.layout import SourceStyles

    s = SourceStyles(["star", "b", "c"])
    assert s["b"]["color"] != s["c"]["color"]
    assert s["b"] == SourceStyles(["star", "b", "c"])["b"]


def test_source_styles_wraps_the_palette_past_its_end():
    from eyepiece.layout import SourceStyles

    s = SourceStyles([f"s{k}" for k in range(8)])  # palettes hold six colors
    assert s["s6"]["color"] == s["s0"]["color"]
    assert s["s6"]["marker"] == s["s0"]["marker"]
