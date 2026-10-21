# Working on eyepiece

Eyepiece is a public plotting and animation library built on Matplotlib and NumPy.
Keep library code independent of simulation packages and private project state.
Read README.md, docs/reference/catalog.md and docs/explanation/contract.md first.
Follow neighboring implementations, tests and documentation.

## Public namespace map

Use `import eyepiece as ep` in documentation and examples, then call through
one task namespace. Implementation files stay in place; facades re-export
the same objects lazily through the package lookup mechanism.

| Namespace | Home for |
|---|---|
| `ep.plot` | Stills, updatable data displays, marks, layouts, and plot results |
| `ep.clock` | Scheduling, easing, progress shapes, and timing records |
| `ep.motion` | Moves, attention, camera geometry, and motion contracts |
| `ep.style` | Appearance, source styles, and render profiles |
| `ep.render` | Animation output, recording, files, workers, and provenance |
| `ep.testing` | Clip, timing, readability, connector, and frame checks |
| `ep.narration` | Speech, captions including `caption_line`, chapters, and cue sounds |
| `ep.motifs` | Recurring styles |
| `ep.prepared` | Scene records and replayable sequences |
| `ep.mpl`, `ep.manim` | Prepared renderers |

The root exports only `Timeline`, `Plan`, `Tempo`, `PlotResult`, `MosaicResult`,
`MotionContract`, `animate`, `record`, `save_fig`, and `__version__` through
`__all__`. Root discovery also lists the task namespaces. Every former root
path still works silently in 0.17 through `_LEGACY_ROOT`; do not warn on these
paths yet. The 0.16 deprecated aliases keep their existing warnings.

Keep namespace imports lazy. In particular, importing `ep.clock` or
`ep.narration` must not import Matplotlib. Do not add a module whose name
shadows a callable export, including `equation`, `phasor`, or `stagger`.

## Environment and checks

Use the existing development environment and its installed dependencies.
Do not create an environment or refresh dependencies unless the task needs it.
Run focused tests against this checkout, for example:

```sh
PYTHONPATH=src python -m pytest tests/test_catalog.py -q
PYTHONPATH=src python -m pytest tests/test_head_draw.py -q
ruff check path/to/changed.py
ruff format path/to/changed.py
git diff --check
```

Run only test files relevant to the change, including any test files you edit.
Run the full suite only when explicitly requested.
Add tests for observable behavior, including endpoints and failure conditions.
Use deterministic seeds for stochastic examples and tests.

## Public API and compatibility

docs/reference/catalog.md maps every public name to a task and a worked example.
Add new public names to the task namespace __all__ and its catalog task table.
Use one canonical namespaced path per name in the catalog; stable root
shortcuts may appear bare.
Update the catalog's "Shown in" link when adding or moving an example.
tests/test_catalog.py must pass after API or catalog edits.
Keep a deprecated alias for a renamed or removed public name for one minor release.
Aliases warn once at the caller with DeprecationWarning and name the replacement.
Keep deprecated aliases out of __all__ and list them in the deprecated catalog table.
Follow tests/test_deprecated_names.py for warning location and compatibility checks.

## Motion contract

Read docs/reference/motion-contract.md and the signature moves in the contract.
State what may move, what stays still and what the rest state must be.
Declare only promises the move keeps in its result's MotionContract.
Keep updates absolute: drawing a frame must not depend on earlier frames.
Keep data scales fixed and reuse artists during updates.
For a new move, test intermediate geometry, protected artists and both endpoints.
Compare the settled frame to the plain still, including a nonsequential update path.
Run focused contract tests and ep.testing.motion_faults with the move result in moves=.
Use ep.testing.check_clip for readability and timing when a plan drives the example.
Use ep.testing.frame_faults when frame order independence needs checking.
Declare intentional draw cuts with steps rather than hiding motion faults.

## Code, docs and commits

Write ASCII-only Python code; use plain, direct prose without em dashes or contractions.
Preserve ax-first arguments, result records and call-time style resolution.
Put tutorials in docs/tutorials, task guides in docs/how-to and examples in docs/gallery.
Put explanations in docs/explanation and API/task reference material in docs/reference.
Use MyST code cells for runnable docs and link new pages from docs/index.md.
Execute every new or changed page's cells; use short jshtml players, not MP4 embeds.
Use at most one animation per page, about 30 frames, and dpi at most 100.
Commit one item at a time as type(scope): message, using feat, fix, docs, test or chore.
Use a one-line subject with no refactor type, exclamation mark or co-author trailer.
Do not include private notes or external workspace paths; do not push or merge unasked.
