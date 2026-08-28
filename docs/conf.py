"""Sphinx configuration file."""

from importlib.metadata import version as get_version

project = "eyepiece"
copyright = "2026, Corey Spohn"
author = "Corey Spohn"
release = get_version("eyepiece")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/*version.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # document top-level re-exports so ``{class}`~eyepiece.X``` roles
    # resolve to the flat public namespace, not the internal submodule.
    "imported-members",
]
autodoc_typehints = "description"

# Render Google-style ``Attributes:`` sections as inline ``:ivar:`` fields, so
# they do not collide with the ``py:attribute`` directives autoapi generates
# from the same class fields (avoids duplicate-object warnings).
napoleon_use_ivar = True

# Silence the harmless generated _version import note.
suppress_warnings = ["autoapi.python_import_resolution"]

myst_enable_extensions = ["amsmath", "dollarmath"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
master_doc = "index"
html_title = "eyepiece"

html_theme_options = {
    "repository_url": "https://github.com/CoreySpohn/eyepiece",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_context = {
    "default_mode": "dark",
}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
nb_execution_mode = "auto"
nb_execution_timeout = 300
nb_execution_raise_on_error = True
# ... and print the failing cell's traceback: raise_on_error alone names only
# the page, which is not enough to fix it from a CI log.
nb_execution_show_tb = True
# Drop benign import-time stderr (e.g. tqdm's IProgress warning) from the
# rendered output; genuine execution errors still raise via the flag above.
nb_output_stderr = "remove"
