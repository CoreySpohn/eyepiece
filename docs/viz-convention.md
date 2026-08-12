# The viz module convention

eyepiece plots arrays. It does not know what a pupil, an orbit, or a
detector frame is in your code, and it never will, because the moment it did
it would be coupled to the package that defines them. A simulation library
that wants its own types to plot themselves therefore ships a thin plotting
module of its own, built on these primitives. This page is the convention
those modules follow.

The convention is written for a library maintainer adopting it. Following it
gets you type-aware plotting that costs your base users nothing, looks like
every other figure built on eyepiece, and does not turn into a private
plotting framework two releases later.

## The division of labor

eyepiece owns generic mechanisms over arrays: norms and colormaps, panel
grids, colorbar geometry, extents and labels, animation, styled saving, and
the {doc}`return contract <contract>` that ties them together. Your library
owns knowledge of its own types: which attribute holds the array, what units
it is in, how a chromatic field is laid out, what a sensible default view of
it is.

The test that decides where a function belongs is one question. Would this
function have to change if some other library changed its API? If the answer
is yes, the function belongs in that library's plotting module, not in
eyepiece and not in yours. A function that renders your types is yours. A
function that renders arrays is a candidate for eyepiece. A function that
renders someone else's types is theirs.

## Placement and packaging

**One module, inside the library.** A `viz.py` next to the rest of the
package, promoted to a `viz/` package once it passes roughly four functions.
Never a separate repository: plotting for your types versions with your
types.

**Lazy import behind an optional extra.** The plotting module imports
eyepiece, which brings matplotlib and a style library with it, and a
simulation library should not force that on a user who only wants numbers.
Declare the dependency as an extra, and import it inside the functions. A
small helper keeps the error message actionable:

```python
# mylib/_require.py
def eyepiece():
    """Import eyepiece, or raise with the install hint."""
    try:
        import eyepiece
    except ImportError:
        raise ImportError(
            "mylib.viz requires eyepiece: pip install 'mylib[viz]'"
        ) from None
    return eyepiece
```

A `viz/` package re-exports its functions lazily through the module-level
`__getattr__` of PEP 562, with a matching `__dir__`, so importing the
package does not import every submodule and therefore does not import
eyepiece either.

The property that matters here is that the base install imports clean, and
it is worth an explicit test rather than an assumption. Block the import in
a subprocess and import your library:

```python
# tests/test_base_install.py
import subprocess
import sys

CODE = "import sys; sys.modules['eyepiece'] = None; import mylib"


def test_library_imports_without_eyepiece():
    subprocess.run([sys.executable, "-c", CODE], check=True)
```

**No top-level re-exports.** Do not surface `mylib.plot_thing` from the
library's `__init__`. A top-level re-export executes the plotting import
chain for every user, which defeats the extra entirely and is the single
easiest way to lose the property the previous test protects. Users write
`from mylib import viz`, or `import mylib.viz as viz`.

**Zero import side effects.** Importing the plotting module must not
activate a style mode, write to `matplotlib.rcParams`, or change any global.
In particular, never bind a style library's `cmaps` or `palette` globals at
import time. Activating a mode rebinds those names, so a captured reference
silently freezes whichever mode happened to be active when your module
loaded. Resolve them as module attributes inside the call, exactly as
eyepiece does.

**Version the extra honestly.** The extra pins the eyepiece release that
actually carries the primitives you call, and a release of your library that
advertises the extra ships only after that eyepiece version is on PyPI.

## The function contract

A plotting function follows the same contract every eyepiece primitive
follows, which is documented in full on the {doc}`contract <contract>` page
and summarized here as it applies to you.

The shape is stateless and data-first:

```python
def plot_thing(
    thing_or_array,
    *,
    ax=None,
    pixscale_lod=0.25,
    imshow_kw=None,
    cbar_kw=None,
): ...
```

Named parameters for what the function owns semantically, per-target keyword
dicts for everything routed onward, no flat `**kwargs`, and an
`eyepiece.PlotResult` or `eyepiece.MosaicResult` on the way out. `ax=None`
creates a figure, an axes passed in is drawn into and nothing else is
touched.

**Every function that accepts one of your types also accepts the bare
arrays.** Real consumers frequently hold arrays loaded from a file rather
than a live object, and a function that demands the object forces them to
reconstruct one just to draw a picture.

**Type-awareness is extract, delegate, decorate.** Pull the arrays and the
extent out of your types, delegate the actual rendering to an eyepiece
primitive, then apply the domain decoration the primitive cannot know about:

```python
# mylib/viz.py
import numpy as np

from mylib._require import eyepiece


def plot_psf(psf, *, ax=None, pixscale_lod=0.25, imshow_kw=None, cbar_kw=None):
    """Draw a PSF, log-scaled, on a lambda/D grid."""
    ep = eyepiece()
    arr = np.asarray(getattr(psf, "intensity", psf), dtype=float)
    result = ep.imshow_log(
        arr,
        ax=ax,
        extent=ep.extent_lod_from_pixels(arr.shape[-1], pixscale_lod),
        imshow_kw=imshow_kw,
        cbar_kw=cbar_kw,
    )
    ep.label_lod(result.ax)
    return result
```

Note where the units live. eyepiece stays unit-agnostic and takes an extent
tuple; knowing that this array is sampled at a given number of lambda over D
per pixel is your library's job, expressed as a plain float in a
unit-suffixed argument name.

**Declare the axes shape you accept.** A function drawing a regular grid
takes `axes=` of exactly one documented shape, and a mismatch raises
`ValueError` naming both the expected shape and the received one. When the
shape depends on the data, document the formula rather than a number. A
ragged layout, such as a panel spanning several cells above a row of them,
accepts a `SubFigure` only, owns its internal gridspec, and returns `.axes`
as a flat object array in a documented order.

Never call `sharex` or `sharey` on axes the caller handed you; matplotlib
raises on already-shared axes, and the caller may have shared them
deliberately. Set identical limits instead.

**Geometry stays inside the handed slots.** The rule is inherited whole from
eyepiece. Colorbars in a handed slot are insets via `ax.inset_axes`, never
`fig.colorbar(ax=...)`, which is banned beside an equal-aspect panel even in
a figure your function created. Figure-level artists, cross-panel arrows and
`ConnectionPatch` among them, are legal only when your function owns the
`Figure` or `SubFigure` they are drawn on.

**Provide an updater when you transform.** A function that derives its
display data, taking a log, unwrapping a phase, collapsing a wavelength
axis, passes an `update` through the result so an animation loop mutates
artists without re-deriving the transform. Animations build on
`eyepiece.record` and `eyepiece.animate`, and an animated figure must fit
the figure size it declares, because the frame-grab path offers no
tight-bounding-box rescue.

## The firewall

Your plotting module renders your types and arrays. Nothing else. A figure
that puts one library's output beside another's is composed in the user's
script, on shared axes, by calling both libraries' functions. It is never
written inside either library, because that would make each a dependency of
the other for the sake of one figure.

When a device turns out to be genuinely generic, the answer is promotion,
not duplication. A drawing shape that has been hand-rolled independently in
two or more places, and that takes only arrays and floats, is a candidate
for eyepiece under the accretion rule on the
{doc}`contract <contract>` page. Copying it into a second library's plotting
module instead is how two implementations drift.

## Visual language

Recurring entities in your domain, the roles that show up in figure after
figure, are data, not code. Map them onto the style library's brand roles or
palette slots and never introduce new hex values, because a color chosen
inside a plotting function cannot respond to a mode switch and will not
match anything else in the document.

Declare the cast once per document, in the preamble, and thread it through
every figure:

```python
import eyepiece as ep

styles = ep.SourceStyles(["star", "planet b", "planet c"])
```

`SourceStyles` assigns each name a color from the active palette in
declaration order, plus a marker, so the same name draws identically in
every panel and every later figure of that document. Build it once and pass
it down. Rebuilding a registry per figure is how panel three ends up
disagreeing with panel one.

The grammar rule that follows from this is worth stating on its own: cause
and effect share color, within a figure and across every subsequent figure
of the same document. If the aberration is drawn in one color on the pupil
panel, the speckle it produces is drawn in that color on the image panel.

## Method: census first

Before writing a plotting module, count. Sweep the library's own example
pages and the analysis scripts its users actually write, list the recurring
plot shapes, and count how many independent implementations each one has.
Build the top of that list first. The point is not thoroughness for its own
sake, it is that a function nobody hand-rolled twice is a function nobody
needed.

Scope by the same accretion rule eyepiece uses. A new function wants two or
more cited hand-rolled instances, and it carries a sunset clause: fewer than
two consumers after two release cycles means deprecation.

Existing figure-owning plot functions in a library that is already past 1.0
are frozen rather than converted. Changing a return type is a major version
bump, so add the new contract-following function beside the old one, mark
the old one deprecated with a warning that names its replacement and the
release that removes it, and leave its behavior alone.

Before calling a plotting module done, dogfood it. Convert one example page
and one real consumer script, and check that each came out shorter and
clearer than what it replaced. A conversion that came out longer is evidence
against the API, and the API is what changes.

## Testing checklist

Run everything on the `Agg` backend.

- **Contract smoke tests.** Call each function, assert the result type is
  right, assert the artist keys are drawn from eyepiece's documented
  vocabulary, and assert that reuse works: an `ax`, an `axes` array, and a
  `SubFigure` where the function accepts one.
- **The declared shape error.** Hand a wrong-shaped `axes` and assert the
  `ValueError` message names the expected and received shapes.
- **The base install.** The subprocess test above, which is the only thing
  standing between the optional extra and a hard dependency.
- **Geometry.** Draw into `axes[0]` of a one-by-two grid and assert the two
  slots still have equal width via `get_position(original=True)`.
  Post-draw positions falsely convict an equal-aspect panel, which shrinks
  to its data aspect regardless of what the function did.
- **Your own regression class.** Whatever shape of data your library gets
  wrong when nobody is looking, a wavelength-first cube for instance, gets an
  explicit test.
- **No image baselines.** Pixel comparisons are expensive to maintain and
  fail for reasons that have nothing to do with the code. The executed
  example gallery is the visual net instead, which works only when those
  pages are genuinely executed at build time.

## Documentation adoption

Add eyepiece to the library's docs extra, and add `imageio-ffmpeg` as well
if any page writes an mp4.

Start every example page with the same preamble: activate the style mode,
import the plotting module, and declare the document's cast. Consistency
across pages is most of the benefit.

Author new pages as MyST markdown rather than notebooks. With `.md` mapped
to `myst-nb`, a page with `{code-cell}` blocks executes at build exactly as
a notebook does, and it diffs and merges as text. Existing notebooks stay
tracked without outputs, which a clean filter enforces locally and a test
that loads each notebook and asserts it has no stored outputs enforces in
continuous integration. A page too data-heavy to execute on the builder is
the exception: it keeps its committed outputs deliberately, is flagged as
such so the filter leaves it alone, and is re-executed by hand.

Keep animations small. Roughly one per page, no more than about thirty
frames, at a dpi around 100, which lands near a few megabytes of embedded
HTML. Documentation builders generally have no ffmpeg, so embed
`Animation.jshtml` rather than a video file, and pass its dpi explicitly:
setting `fig.dpi` does not control the writer, and a style library that pins
`savefig.dpi` for print figures will otherwise hand you 300-dpi frames.
