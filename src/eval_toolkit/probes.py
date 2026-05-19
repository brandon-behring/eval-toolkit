"""Activation probes — linear classifiers over transformer hidden states.

Implements :class:`ActivationDeltaProbe`, a port of the TaskTracker
methodology ([1]_) for prompt-injection detection via linear probing of
activation deltas at a chosen transformer layer.

The probe:

1. Loads a backbone HuggingFace ``transformers.AutoModel`` (encoder OR decoder).
2. Computes a reference activation at the chosen ``layer_index`` for a
   ``clean_baseline_text`` (cached once per backbone).
3. For each input, computes the activation at the same layer, aggregates
   over the sequence dimension (``mean`` / ``max`` / ``cls``), subtracts
   the clean baseline → a fixed-length **delta vector**.
4. Fits a :class:`sklearn.linear_model.LogisticRegression` over the
   stacked deltas with ``class_weight="balanced"`` for imbalanced
   injection sets.

The result is a tiny linear classifier with ``coef_`` available for
interpretability and ``predict_proba`` returning a standard sklearn
``(n_samples, 2)`` probability matrix.

This module is **base-install safe**: ``torch`` and ``transformers`` are
soft-imported inside the methods that need them. ``pip install
eval-toolkit[probes]`` installs both. The lazy-import pattern matches
the :class:`HFDatasetsLoader` (``datasets``) and the embeddings extra
(``sentence-transformers``) precedents.

References
----------
.. [1] Abdelnabi, S., et al. 2024. "Are You Still on Track? Catching LLM
       Task Drift with Activations." arXiv:2406.00799.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)

__all__ = [
    "ActivationDeltaProbe",
    "ActivationExtractor",
    "Probe",
]


# Aggregation modes for collapsing (batch, seq_len, hidden) → (batch, hidden)
AggregateMode = Literal["mean", "max", "cls"]


@runtime_checkable
class Probe(Protocol):
    """Minimal probe surface: fit + predict on raw text inputs.

    Probes wrap a feature extractor + linear classifier and expose a
    scikit-learn-shaped API so downstream evaluators can drop them in
    alongside any other ``predict_proba``-capable model.

    Notes
    -----
    Unlike :class:`~eval_toolkit.protocols.Scorer` (which returns a
    1-D ``P(positive)`` array), probes return the full sklearn
    ``(n_samples, 2)`` probability matrix from ``predict_proba``.
    Wrap with ``lambda probe: probe.predict_proba(X)[:, 1]`` to adapt
    a probe to the Scorer contract.
    """

    classes_: np.ndarray
    coef_: np.ndarray

    def fit(
        self, clean_texts: Sequence[str], injected_texts: Sequence[str]
    ) -> Probe:  # pragma: no cover
        """Fit the probe on two labeled text lists."""
        ...

    def predict(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        """Return ``(n,)`` binary predictions in ``classes_``."""
        ...

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        """Return ``(n, 2)`` probability matrix matching ``classes_`` column order."""
        ...


@runtime_checkable
class ActivationExtractor(Protocol):
    """Extract a hidden-state vector for each input text.

    Decoupled from :class:`ActivationDeltaProbe` so tests can inject a
    deterministic mock extractor without loading a real HF model. The
    default implementation is :class:`_HFActivationExtractor` (built
    lazily inside :meth:`ActivationDeltaProbe.fit`).

    Implementations must be deterministic: same backbone + same text →
    same activation vector across calls.

    Notes
    -----
    ``hidden_size`` and ``backbone_id`` are declared as ``@property`` so
    implementations may compute them lazily (the default HF extractor
    needs to load the model before it can report ``hidden_size``).
    Plain dataclass-field implementations also satisfy the structural
    check.
    """

    @property
    def hidden_size(self) -> int:  # pragma: no cover
        """Dimension of the activation vector returned by :meth:`extract`."""
        ...

    @property
    def backbone_id(self) -> str:  # pragma: no cover
        """Stable identifier for the backbone (used as cache-key namespace)."""
        ...

    def extract(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        """Return ``(len(texts), hidden_size)`` float32 activation matrix."""
        ...


def _sha256_text(text: str) -> str:
    """Stable sha256 hex of UTF-8-encoded text — used as cache key suffix."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_cache_dir() -> Path:
    """Default activation-cache root: ``$XDG_CACHE_HOME/eval-toolkit/probes`` or ``~/.cache/...``."""
    base_str = os.environ.get("XDG_CACHE_HOME", "")
    base = Path(base_str) if base_str else Path.home() / ".cache"
    return base / "eval-toolkit" / "probes"


@dataclass
class ActivationDeltaProbe:
    """TaskTracker-style activation-delta linear probe.

    For each input text, computes the hidden-state activation at the
    chosen layer, subtracts the cached clean-baseline activation, and
    feeds the resulting delta vector to a logistic classifier. Works
    with any HuggingFace encoder or decoder backbone exposing
    ``output_hidden_states=True``.

    Parameters
    ----------
    backbone : str
        HF model id (e.g. ``"answerdotai/ModernBERT-base"``,
        ``"distilbert-base-uncased"``, ``"meta-llama/Llama-3.2-1B"``).
        Loaded lazily on first :meth:`fit` call via
        ``transformers.AutoModel.from_pretrained``.
    layer_index : int, optional
        Index into ``outputs.hidden_states``. ``-1`` (default) is the
        last layer. Negative indexing supported (``-4`` → 4th-from-last).
    aggregate : {"mean", "max", "cls"}, optional
        How to collapse ``(seq_len, hidden)`` to ``(hidden,)``:

        - ``"mean"`` (default) — mean-pool over the sequence dim.
        - ``"max"`` — max-pool over the sequence dim.
        - ``"cls"`` — take position 0 (the ``[CLS]`` token in BERT-family
          encoders or the BOS token in decoders).

    clean_baseline_text : str, optional
        Reference input whose activation is subtracted from every other
        input's activation. Default ``"\\n"`` (newline). The delta — not
        the absolute activation — is the feature the linear classifier
        learns on.
    device : str, optional
        Torch device for the backbone forward pass. Default ``"cpu"``.
        Pass ``"cuda"`` when a GPU is available.
    cache_dir : str or pathlib.Path or None, optional
        Directory for cached per-text activations. Default
        ``$XDG_CACHE_HOME/eval-toolkit/probes`` or ``~/.cache/...``.
        Cache key is ``(backbone, layer_index, aggregate, sha256(text))``.
    classifier : sklearn.base.BaseEstimator or None, optional
        Override the default ``LogisticRegression(class_weight="balanced")``
        classifier. Must expose sklearn ``fit`` / ``predict`` /
        ``predict_proba`` / ``coef_`` / ``classes_``.
    extractor : ActivationExtractor or None, optional
        Inject a pre-built extractor (e.g. a mock for unit tests). When
        provided, the ``backbone`` / ``layer_index`` / ``aggregate`` /
        ``device`` params are ignored — the extractor encapsulates that
        configuration. Default ``None`` (build a fresh HF-backed
        extractor on first ``fit``).

    Attributes
    ----------
    coef_ : numpy.ndarray
        Logistic regression coefficient vector (after :meth:`fit`).
    classes_ : numpy.ndarray
        ``array([0, 1])`` — class label order matching
        ``predict_proba`` columns.

    Examples
    --------
    >>> # Real usage requires the [probes] extra + a HF backbone.
    >>> # See docs/source/examples/activation_delta_probe.md for a
    >>> # mocked end-to-end illustration.

    Raises
    ------
    ImportError
        Raised on first :meth:`fit` / :meth:`predict` if the ``[probes]``
        extra is not installed (and no ``extractor`` was injected).
        Message includes the install hint.
    """

    backbone: str
    layer_index: int = -1
    aggregate: AggregateMode = "mean"
    clean_baseline_text: str = "\n"
    device: str = "cpu"
    cache_dir: str | Path | None = None
    classifier: Any = None  # noqa: ANN401 — sklearn estimator (no shared base type)
    extractor: ActivationExtractor | None = None
    _baseline_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def _resolve_cache_dir(self) -> Path:
        """Return the cache root, creating it if absent."""
        d = Path(self.cache_dir).expanduser() if self.cache_dir else _default_cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _build_default_extractor(self) -> ActivationExtractor:
        """Build the default HF-backed extractor (lazy import)."""
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "ActivationDeltaProbe requires torch + transformers. "
                "Install with: pip install eval-toolkit[probes]"
            ) from exc

        return _HFActivationExtractor(
            backbone=self.backbone,
            layer_index=self.layer_index,
            aggregate=self.aggregate,
            device=self.device,
            torch_module=torch,
            auto_model_cls=AutoModel,
            auto_tokenizer_cls=AutoTokenizer,
        )

    def _get_extractor(self) -> ActivationExtractor:
        """Lazy-build the extractor on first use."""
        if self.extractor is None:
            object.__setattr__(self, "extractor", self._build_default_extractor())
        # `extractor` is non-None after the assignment above.
        assert self.extractor is not None
        return self.extractor

    def _extract_with_cache(self, texts: Sequence[str]) -> np.ndarray:
        """Extract activations for ``texts``, caching per-text results to disk."""
        cache_root = self._resolve_cache_dir()
        extractor = self._get_extractor()
        backbone_key = hashlib.sha256(extractor.backbone_id.encode("utf-8")).hexdigest()[:16]
        per_backbone = cache_root / backbone_key
        per_backbone.mkdir(parents=True, exist_ok=True)

        config_suffix = f"L{self.layer_index}_{self.aggregate}"
        outputs: list[np.ndarray | None] = [None] * len(texts)
        misses: list[int] = []
        miss_texts: list[str] = []
        for i, text in enumerate(texts):
            cache_path = per_backbone / f"{_sha256_text(text)}_{config_suffix}.npy"
            if cache_path.exists():
                outputs[i] = np.load(cache_path)
            else:
                misses.append(i)
                miss_texts.append(text)

        if miss_texts:
            new_acts = extractor.extract(miss_texts)
            for j, miss_idx in enumerate(misses):
                arr = new_acts[j].astype(np.float32)
                outputs[miss_idx] = arr
                cache_path = per_backbone / f"{_sha256_text(miss_texts[j])}_{config_suffix}.npy"
                np.save(cache_path, arr)

        # All slots populated by construction.
        out = np.stack([o for o in outputs if o is not None]).astype(np.float32)
        if out.shape[0] != len(texts):  # pragma: no cover — defensive
            raise RuntimeError(
                f"ActivationDeltaProbe: cache assembly missed {len(texts) - out.shape[0]} rows; "
                f"this is an internal bug"
            )
        return out

    def _compute_deltas(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n, hidden)`` matrix of activation deltas vs the cached baseline."""
        if self._baseline_cache is None:
            baseline = self._extract_with_cache([self.clean_baseline_text])[0]
            object.__setattr__(self, "_baseline_cache", baseline)
        baseline = self._baseline_cache
        assert baseline is not None
        activations = self._extract_with_cache(list(texts))
        deltas: np.ndarray = activations - baseline[None, :]
        return deltas

    def fit(
        self,
        clean_texts: Sequence[str],
        injected_texts: Sequence[str],
    ) -> ActivationDeltaProbe:
        """Fit the linear classifier on activation deltas.

        Parameters
        ----------
        clean_texts : sequence of str
            Control inputs (label = 0).
        injected_texts : sequence of str
            Adversarial inputs (label = 1).

        Returns
        -------
        ActivationDeltaProbe
            ``self``, for sklearn-style chaining.

        Raises
        ------
        ValueError
            If either input list is empty.
        ImportError
            If ``[probes]`` extra is missing and no extractor was injected.
        """
        if not clean_texts:
            raise ValueError("ActivationDeltaProbe.fit: clean_texts is empty")
        if not injected_texts:
            raise ValueError("ActivationDeltaProbe.fit: injected_texts is empty")

        clean_deltas = self._compute_deltas(list(clean_texts))
        injected_deltas = self._compute_deltas(list(injected_texts))
        X = np.vstack([clean_deltas, injected_deltas])
        y = np.concatenate(
            [np.zeros(len(clean_texts), dtype=int), np.ones(len(injected_texts), dtype=int)]
        )

        if self.classifier is None:
            from sklearn.linear_model import LogisticRegression

            object.__setattr__(
                self,
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            )
        self.classifier.fit(X, y)
        object.__setattr__(self, "_fitted", True)
        return self

    @property
    def coef_(self) -> np.ndarray:
        """Logistic regression coefficient vector (after :meth:`fit`)."""
        self._require_fitted()
        coef: np.ndarray = np.asarray(self.classifier.coef_)
        return coef

    @property
    def classes_(self) -> np.ndarray:
        """Class label array ``array([0, 1])`` from the fitted classifier."""
        self._require_fitted()
        classes: np.ndarray = np.asarray(self.classifier.classes_)
        return classes

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n,)`` binary predictions.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._require_fitted()
        deltas = self._compute_deltas(list(texts))
        preds: np.ndarray = np.asarray(self.classifier.predict(deltas))
        return preds

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n, 2)`` probability matrix.

        Column order matches :attr:`classes_` — i.e. ``[:, 0]`` is
        P(class 0) and ``[:, 1]`` is P(class 1).
        """
        self._require_fitted()
        deltas = self._compute_deltas(list(texts))
        proba: np.ndarray = np.asarray(self.classifier.predict_proba(deltas))
        return proba

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "ActivationDeltaProbe: call .fit(clean_texts, injected_texts) before "
                ".predict / .predict_proba / .coef_ / .classes_"
            )


@dataclass(frozen=True, slots=True)
class _HFActivationExtractor:
    """Default HF-backed :class:`ActivationExtractor` (lazy-imported torch/transformers).

    Not exposed in ``__all__``; consumers build via
    :class:`ActivationDeltaProbe` constructor. Tests inject mock
    extractors directly via the ``extractor=`` constructor kwarg.
    """

    backbone: str
    layer_index: int
    aggregate: AggregateMode
    device: str
    torch_module: Any
    auto_model_cls: Any
    auto_tokenizer_cls: Any

    @property
    def backbone_id(self) -> str:
        return self.backbone

    @property
    def hidden_size(self) -> int:
        model = self._load_model_tokenizer()[0]
        return int(model.config.hidden_size)

    def _load_model_tokenizer(self) -> tuple[Any, Any]:
        tokenizer = self.auto_tokenizer_cls.from_pretrained(self.backbone)
        model = self.auto_model_cls.from_pretrained(self.backbone, output_hidden_states=True)
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def extract(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch_module
        model, tokenizer = self._load_model_tokenizer()
        inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states[self.layer_index]  # (batch, seq, hidden)
        attention_mask = inputs.get("attention_mask")

        if self.aggregate == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.aggregate == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
        elif self.aggregate == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).bool()
                hidden_states = hidden_states.masked_fill(~mask, float("-inf"))
            pooled = hidden_states.max(dim=1).values
        else:  # pragma: no cover — Literal type narrows this away
            raise ValueError(f"_HFActivationExtractor: unknown aggregate {self.aggregate!r}")

        arr: np.ndarray = pooled.cpu().numpy().astype(np.float32)
        return arr
