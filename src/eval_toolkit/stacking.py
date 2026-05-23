"""Detector stacking — combine multiple binary scorers into a calibrated ensemble.

Implements the :class:`MetaLearner` Protocol and one reference impl,
:class:`LogisticStacker`, for stacking the outputs of multiple base detectors
into a single P(positive) estimate. The classic use case for prompt-injection
detection: you have a fine-tuned classifier, an activation probe, and an
LLM-judge — each emits a per-sample score in ``[0, 1]``. A stacker learns the
best (regularized) linear combination of those scores.

The :class:`MetaLearner` Protocol is intentionally minimal: ``fit`` takes a
``(n_samples, n_detectors)`` score matrix plus binary labels; ``predict_proba``
returns the sklearn-standard ``(n_samples, 2)`` probability matrix. The shape
mirrors :class:`~eval_toolkit.probes.Probe` so consumers can drop a stacker into
any sklearn-shaped evaluation harness.

Stacking sits AFTER per-detector calibration in a typical pipeline:

1. Train each base detector on training data.
2. Calibrate each detector individually (e.g. :func:`fit_platt_binary`).
3. On a held-out **stacking** set (disjoint from each detector's training set
   to avoid optimistic stacking), collect the calibrated scores into a matrix.
4. Fit a :class:`LogisticStacker` on the matrix + labels.
5. Optionally calibrate the stacker's output via another
   :func:`fit_platt_binary` / :func:`fit_isotonic_binary` pass.

The Protocol carries an attribute contract (``coef_``, ``classes_``,
``intercept_``) and a method contract (``fit``, ``predict``, ``predict_proba``)
so :class:`MetaLearner` instances are interchangeable inside the harness.

References
----------
.. [1] Wolpert, D. H. 1992. "Stacked generalization." Neural Networks
       5(2), 241–259. doi:10.1016/S0893-6080(05)80023-1.
.. [2] Breiman, L. 1996. "Stacked regressions." Machine Learning
       24(1), 49–64.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from sklearn.linear_model import LogisticRegression

from eval_toolkit._rng import RNGLike, SeedLike

_logger = logging.getLogger(__name__)

__all__ = [
    "LogisticStacker",
    "MetaLearner",
]


@runtime_checkable
class MetaLearner(Protocol):
    """Combines per-sample scores from multiple base detectors into P(positive).

    The Protocol takes a ``(n_samples, n_detectors)`` score matrix — one row per
    sample, one column per base detector — plus binary labels for fitting.
    Output is the sklearn-standard ``(n_samples, 2)`` probability matrix.

    The contract is sklearn-shaped so stackers compose with the existing
    probe-and-harness machinery. Concrete implementations are expected to be
    deterministic given a fixed ``rng`` and to expose ``coef_`` /
    ``intercept_`` for interpretability + ``RunManifest`` logging.

    Attributes
    ----------
    coef_ : numpy.ndarray
        Fitted coefficient vector, shape ``(n_detectors,)`` for binary
        classification. Available after :meth:`fit`. Reading before ``fit``
        raises ``AttributeError`` or returns ``None`` (impl-defined; the
        reference :class:`LogisticStacker` raises).
    classes_ : numpy.ndarray
        Class labels, shape ``(2,)``. For binary stacking, always
        ``array([0, 1])`` after :meth:`fit`.
    intercept_ : numpy.ndarray
        Fitted intercept, shape ``(1,)``. Available after :meth:`fit`.

    Notes
    -----
    Tier-2 Protocol (frozen at v1.0 per ADR 0003). Additive subprotocols are
    permitted in minor releases; method-signature changes require v2.0.

    When passed to a parallel-capable harness (``n_jobs > 1``), implementations
    MUST be picklable — joblib's loky backend serializes the entire delayed
    call before worker dispatch. See ``docs/source/methodology/parallelism.md``
    for the picklability contract.

    Distinct from :class:`~eval_toolkit.protocols.Scorer` (which consumes raw
    feature data and returns 1-D ``P(positive)``). A stacker can be wrapped to
    satisfy ``Scorer`` via ``lambda X: stacker.predict_proba(X)[:, 1]`` once
    callers have collected base-detector scores into ``X``.

    See Also
    --------
    LogisticStacker : reference implementation wrapping sklearn LogisticRegression.
    """

    coef_: np.ndarray
    classes_: np.ndarray
    intercept_: np.ndarray

    def fit(self, score_matrix: np.ndarray, y: np.ndarray) -> MetaLearner:  # pragma: no cover
        """Fit the stacker on a ``(n_samples, n_detectors)`` score matrix.

        Parameters
        ----------
        score_matrix : numpy.ndarray
            Per-detector calibrated scores, shape ``(n_samples, n_detectors)``.
            Values are typically in ``[0, 1]`` but the contract does not
            require it.
        y : numpy.ndarray
            Binary labels in ``{0, 1}``, shape ``(n_samples,)``.

        Returns
        -------
        MetaLearner
            ``self`` (sklearn convention) — the fitted estimator.
        """
        ...

    def predict(self, score_matrix: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Return binary predictions for ``score_matrix``, shape ``(n_samples,)``."""
        ...

    def predict_proba(self, score_matrix: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Return ``(n_samples, 2)`` probability matrix.

        Column order matches :attr:`classes_`. Column 1 is ``P(positive)``.
        """
        ...


@dataclass
class LogisticStacker:
    """Reference :class:`MetaLearner` using :class:`sklearn.linear_model.LogisticRegression`.

    Wraps sklearn's logistic regression with a stacker-shaped public API.
    Configuration goes in the constructor; fitted state is populated on
    :meth:`fit` and exposed via the standard sklearn ``coef_`` / ``classes_`` /
    ``intercept_`` attributes.

    Parameters
    ----------
    C : float, optional
        Inverse regularization strength. Smaller ``C`` → stronger L2/L1
        regularization → more shrinkage toward zero on detector weights.
        Default ``1.0`` (sklearn default). For very-few-detector stacking
        (≤3 base scorers), stronger regularization (``C=0.1``) helps prevent
        the stacker from over-fitting to a single dominant detector on small
        stacking sets.
    fit_intercept : bool, optional
        Whether to fit an intercept term. Default ``True``.
    class_weight : str or dict or None, optional
        Per-class sample weighting. Default ``"balanced"`` — automatically
        weights inversely proportional to class frequencies, useful for
        imbalanced injection / non-injection sets.
    penalty : {"l1", "l2", "elasticnet", None}, optional
        Regularization norm. Default ``"l2"``. ``"l1"`` zeros out non-useful
        detector columns (sparsity); ``"l2"`` shrinks them uniformly.
    solver : str, optional
        Optimizer. Default ``"lbfgs"`` (L2-only). Use ``"liblinear"`` for L1
        penalty on small stacking sets; ``"saga"`` for elasticnet.
    max_iter : int, optional
        Maximum iterations. Default ``1000`` (generous; stacking problems are
        small and usually converge in <100).
    rng : RNGLike | SeedLike | None, optional
        Seed for the underlying ``LogisticRegression``. Default ``None``.
        Set for deterministic fitting when the solver involves randomness
        (e.g. ``"saga"``).

    Attributes
    ----------
    coef_ : numpy.ndarray
        Fitted detector weights, shape ``(n_detectors,)``. Set on
        :meth:`fit`.
    classes_ : numpy.ndarray
        Class labels, shape ``(2,)``. Always ``array([0, 1])`` after a binary
        :meth:`fit`.
    intercept_ : numpy.ndarray
        Fitted intercept, shape ``(1,)``. Set on :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> # Three synthetic detectors with varying noise + signal alignment.
    >>> y = rng.binomial(1, 0.3, size=n)
    >>> scores = np.column_stack([
    ...     np.clip(y * 0.7 + rng.normal(0, 0.2, n), 0, 1),
    ...     np.clip(y * 0.5 + rng.normal(0, 0.3, n), 0, 1),
    ...     np.clip(y * 0.4 + rng.normal(0, 0.4, n), 0, 1),
    ... ])
    >>> stacker = LogisticStacker(C=1.0).fit(scores, y)
    >>> stacker.coef_.shape
    (3,)
    >>> stacker.classes_.tolist()
    [0, 1]
    >>> proba = stacker.predict_proba(scores)
    >>> proba.shape
    (500, 2)
    >>> bool(np.allclose(proba.sum(axis=1), 1.0))
    True

    Raises
    ------
    ValueError
        On shape mismatch between ``score_matrix`` and ``y``, on empty inputs,
        on non-finite values in ``score_matrix``, or when ``y`` contains only
        one class (logistic regression is undefined).
    RuntimeError
        Propagated from the underlying sklearn solver if it fails to converge.

    Notes
    -----
    **No new dependencies.** ``scikit-learn`` is already a core eval-toolkit
    dependency since v0.27.

    **Calibration chaining.** A logistic stacker is not automatically
    well-calibrated on the global P(positive) scale — `LogisticRegression`'s
    sigmoid output is well-calibrated on the training data's class prior but
    can drift on held-out distributions. For downstream calibration metrics
    (ECE, Brier), chain through :func:`fit_platt_binary` or
    :func:`fit_isotonic_binary` on a separate calibration set:

    >>> from eval_toolkit import fit_platt_binary
    >>> stacker_proba = stacker.predict_proba(scores)[:, 1]
    >>> (_, _), calibrate = fit_platt_binary(y, stacker_proba)
    >>> calibrated = calibrate(stacker.predict_proba(scores)[:, 1])
    >>> calibrated.shape == (500,)
    True

    See Also
    --------
    eval_toolkit.protocols.Scorer : 1-D ``P(positive)`` contract for raw
        feature inputs.
    eval_toolkit.probes.ActivationDeltaProbe : another sklearn-shaped probe
        producing detector scores.
    eval_toolkit.fit_platt_binary : calibrate stacker output to global prior.
    """

    C: float = 1.0
    fit_intercept: bool = True
    class_weight: str | dict[Any, float] | None = "balanced"
    penalty: Literal["l1", "l2", "elasticnet"] | None = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    rng: RNGLike | SeedLike | None = None

    # Fitted state — populated on fit(); excluded from constructor + repr.
    _model: LogisticRegression | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    @property
    def coef_(self) -> np.ndarray:
        """Fitted detector weights, shape ``(n_detectors,)``. Raises if unfit."""
        self._assert_fitted()
        assert self._model is not None  # narrowed by _assert_fitted; tell mypy
        # sklearn returns (1, n_features) for binary; flatten to (n_features,)
        # np.asarray() wraps sklearn's Any-typed attribute into a known ndarray.
        return np.asarray(self._model.coef_).ravel()

    @property
    def classes_(self) -> np.ndarray:
        """Class labels, shape ``(2,)``. Raises if unfit."""
        self._assert_fitted()
        assert self._model is not None
        return np.asarray(self._model.classes_)

    @property
    def intercept_(self) -> np.ndarray:
        """Fitted intercept, shape ``(1,)``. Raises if unfit."""
        self._assert_fitted()
        assert self._model is not None
        return np.asarray(self._model.intercept_)

    def fit(self, score_matrix: np.ndarray, y: np.ndarray) -> LogisticStacker:
        """Fit the stacker on a ``(n_samples, n_detectors)`` score matrix.

        Parameters
        ----------
        score_matrix : numpy.ndarray
            Per-detector scores. Shape ``(n_samples, n_detectors)``. Must be
            finite. Single-detector stacking (``n_detectors == 1``) is
            permitted but trivial — equivalent to recalibrating that detector.
        y : numpy.ndarray
            Binary labels in ``{0, 1}``. Shape ``(n_samples,)``.

        Returns
        -------
        LogisticStacker
            ``self``, with fitted state populated.

        Raises
        ------
        ValueError
            On shape mismatch, empty inputs, non-finite ``score_matrix``, or
            single-class ``y``.
        """
        sm = np.asarray(score_matrix, dtype=float)
        yarr = np.asarray(y).ravel()
        _validate_fit_inputs(sm, yarr)

        # Derive int seed for sklearn — sklearn LogisticRegression's
        # random_state accepts int | None | RandomState across versions <1.4;
        # safer to derive an int at the boundary than pin a higher sklearn
        # minimum.
        sklearn_seed: int | None
        if self.rng is None:
            sklearn_seed = None
        else:
            sklearn_seed = int(np.random.default_rng(self.rng).integers(0, 2**31 - 1))

        model = LogisticRegression(
            C=self.C,
            fit_intercept=self.fit_intercept,
            class_weight=self.class_weight,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=sklearn_seed,
        )
        model.fit(sm, yarr)
        self._model = model
        self._fitted = True
        return self

    def predict(self, score_matrix: np.ndarray) -> np.ndarray:
        """Return binary predictions, shape ``(n_samples,)``.

        Threshold is sklearn's default 0.5 on column-1 (``P(positive)``).
        For other operating points, use :func:`metrics_at_threshold` against
        :meth:`predict_proba` output directly.

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called yet, or on shape / finiteness
            issues in ``score_matrix``.
        """
        self._assert_fitted()
        assert self._model is not None  # narrowed by _assert_fitted; tell mypy
        sm = np.asarray(score_matrix, dtype=float)
        _validate_predict_inputs(sm, expected_n_features=self.coef_.shape[0])
        return np.asarray(self._model.predict(sm))

    def predict_proba(self, score_matrix: np.ndarray) -> np.ndarray:
        """Return ``(n_samples, 2)`` probability matrix.

        Column order matches :attr:`classes_` (``[0, 1]``); column 1 is
        ``P(positive)``.

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called yet, or on shape / finiteness
            issues in ``score_matrix``.
        """
        self._assert_fitted()
        assert self._model is not None
        sm = np.asarray(score_matrix, dtype=float)
        _validate_predict_inputs(sm, expected_n_features=self.coef_.shape[0])
        return np.asarray(self._model.predict_proba(sm))

    def _assert_fitted(self) -> None:
        """Raise if :meth:`fit` has not been called."""
        if not self._fitted or self._model is None:
            raise ValueError(
                "LogisticStacker has not been fit yet. Call `.fit(score_matrix, y)` first."
            )


def _validate_fit_inputs(score_matrix: np.ndarray, y: np.ndarray) -> None:
    """Shared input validation for :meth:`LogisticStacker.fit`.

    Raises ``ValueError`` with a context-rich message on every failure mode.
    """
    if score_matrix.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2-D (n_samples, n_detectors); got ndim={score_matrix.ndim}"
        )
    if score_matrix.size == 0:
        raise ValueError("score_matrix is empty; provide at least one sample")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D (n_samples,); got ndim={y.ndim}")
    if score_matrix.shape[0] != y.shape[0]:
        raise ValueError(
            "score_matrix and y must have matching n_samples; "
            f"got score_matrix.shape[0]={score_matrix.shape[0]}, y.shape[0]={y.shape[0]}"
        )
    if not np.all(np.isfinite(score_matrix)):
        raise ValueError("score_matrix contains non-finite values (NaN or inf)")
    unique = np.unique(y)
    if unique.size < 2:
        raise ValueError(
            "y is single-class; LogisticStacker requires both classes "
            f"in the training set (got y.unique() = {unique.tolist()})"
        )


def _validate_predict_inputs(score_matrix: np.ndarray, *, expected_n_features: int) -> None:
    """Shared input validation for predict / predict_proba.

    Verifies the score matrix is 2-D, non-empty, finite, and has the expected
    number of detector columns. Raises ``ValueError`` with context on failure.
    """
    if score_matrix.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2-D (n_samples, n_detectors); got ndim={score_matrix.ndim}"
        )
    if score_matrix.size == 0:
        raise ValueError("score_matrix is empty; provide at least one sample")
    if not np.all(np.isfinite(score_matrix)):
        raise ValueError("score_matrix contains non-finite values (NaN or inf)")
    if score_matrix.shape[1] != expected_n_features:
        raise ValueError(
            "score_matrix has wrong number of detectors; "
            f"expected {expected_n_features}, got {score_matrix.shape[1]}"
        )
