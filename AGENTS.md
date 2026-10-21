# Working on eyepiece

Eyepiece is a public plotting and animation library built on Matplotlib and NumPy.
Keep library code independent of simulation packages and private project state.
Read README.md, docs/reference/catalog.md and docs/explanation/contract.md first.
Follow neighboring implementations, tests and documentation.

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
Do not run the full suite: tests/test_bezier.py currently needs about 20 GB.
Add tests for observable behavior, including endpoints and failure conditions.
Use deterministic seeds for stochastic examples and tests.

## Public API and compatibility

docs/reference/catalog.md maps every public name to a task and a worked example.
Add new public names to the appropriate __all__ and catalog task table.
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
Run focused contract tests and ept.motion_faults with the move result in moves=.
Use ept.check_clip for readability and timing when a plan drives the example.
Use render.frame_faults when frame order independence needs checking.
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
