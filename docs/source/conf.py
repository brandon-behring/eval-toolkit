"""Sphinx configuration for eval-toolkit docs (v0.31.0).

Shipped in v0.31.0 (Phase 3 of the mkdocs-material → Sphinx +
pydata-sphinx-theme migration). Locked decisions per the Q1-Q13
walkthrough (see ~/.claude/plans/what-do-we-need-synchronous-turtle.md
"Decision log"):

- pydata-sphinx-theme (Q3) — matches numpy / scipy / sklearn convention
- myst-nb for Markdown + executable cells (Q9)
- nb_execution_mode = "cache" (Q11) — re-execute only on source change
- per-symbol autosummary pages (Q7) — `:toctree: generated/<mod>/`
- minimal branding: favicon only, no header logo (Q10, Q13)
- single-version, GitHub Pages (Q5)
- intersphinx to numpy, scipy, sklearn, pandas, matplotlib (Q1 pain #3)
"""

from __future__ import annotations

import importlib.metadata
from datetime import datetime

# -- Project information --------------------------------------------------

project = "eval-toolkit"
author = "Brandon Behring"
copyright_year = datetime.now().strftime("%Y")
copyright = f"{copyright_year}, Brandon Behring"

# Pull __version__ from installed package metadata so docs build doesn't
# need to import eval_toolkit at conf-time (avoids optional-dep surprises).
try:
    release = importlib.metadata.version("eval-toolkit")
except importlib.metadata.PackageNotFoundError:
    release = "0.31.0-dev"
version = ".".join(release.split(".")[:2])  # 0.31 from 0.31.0.devN

# -- General configuration ------------------------------------------------

extensions = [
    "myst_nb",  # MyST Markdown + Jupyter-style execution
    # (supersedes myst-parser; provides both)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # NumPy-style docstring parser
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

# autodoc / autosummary
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Type-hint rendering — `sphinx-autodoc-typehints` integrates with napoleon
typehints_fully_qualified = False
always_document_param_types = True

# MyST-NB / MyST-Parser configuration
nb_execution_mode = "cache"  # Q11 — execute on source change
nb_execution_timeout = 90  # seconds per cell
nb_execution_show_tb = True  # show full Python traceback on failure
# v0.48 Decision R7-A (Round 7 audit): fail sphinx-build on notebook
# execution errors instead of leaving them as advisory warnings. Without
# this, docs CI runs `sphinx-build -b html` without `-W` (intentional
# because ~56 pre-existing xref/duplicate-label warnings predate v0.47
# polish per .github/workflows/docs.yml:51-58), so notebook execution
# failures would silently pass the build. This narrower setting fails
# specifically on notebook-execution failures while leaving the
# unrelated xref warnings as advisory — Codex R7-F1's recommended fix.
nb_execution_raise_on_error = True
nb_merge_streams = True  # merge stdout/stderr cleanly
myst_enable_extensions = [
    "amsmath",  # \begin{align} ... \end{align}
    "dollarmath",  # $x^2$ and $$x^2$$
    "deflist",  # definition lists
    "fieldlist",  # field lists (:param: etc.)
    "colon_fence",  # ::: admonition syntax (close to mkdocs)
    "linkify",  # auto-detect bare URLs
    "substitution",  # variable substitution
    "tasklist",  # GitHub-style task lists
    "attrs_block",  # `## Heading {#anchor}` block-level attributes
    "attrs_inline",  # `{class="..."}` inline attributes
]
myst_heading_anchors = 3  # auto-generate anchors for H1-H3

# Intersphinx — live links to numpy / scipy / sklearn / pandas / matplotlib
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Source files
source_suffix = {
    ".md": "myst-nb",
    ".rst": "restructuredtext",
}
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Research dossier is archival — exclude from main nav so it doesn't
    # flood the toctree. Will revisit in Phase 2 if we want a research/
    # dedicated section.
    "research/**",
    "archive/**",
    "internals/**",
]

# -- HTML output --------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = project
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "github_url": "https://github.com/brandon-behring/eval-toolkit",
    "use_edit_page_button": True,
    "show_prev_next": True,
    "navigation_with_keys": True,
    "footer_start": ["copyright", "last-updated"],
    "footer_end": ["sphinx-version", "theme-version"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/eval-toolkit/",
            "icon": "fa-brands fa-python",
        },
    ],
}

html_context = {
    "github_user": "brandon-behring",
    "github_repo": "eval-toolkit",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Cross-link `:class:`numpy.ndarray`` etc. — controlled via intersphinx above.
# Disable warnings on missing references for now; Phase 2 will fix and
# eventually flip nitpicky_mode on.
nitpicky = False
