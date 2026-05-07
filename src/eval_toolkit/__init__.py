"""eval-toolkit — reusable evaluation toolkit for binary classification.

Public API: import directly from ``eval_toolkit`` or from the relevant submodule:

    from eval_toolkit import pr_auc, bootstrap_ci, BootstrapCI
    from eval_toolkit.metrics import pr_auc

See ``STYLE.md`` for coding standards and ``CHANGELOG.md`` for version history.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-exports populated as modules land in subsequent A2-A4 phases.
__all__: list[str] = ["__version__"]
