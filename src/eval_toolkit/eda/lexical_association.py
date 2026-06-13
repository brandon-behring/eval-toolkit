"""Lexical-association + competency-baseline shortcut diagnostics.

This module is **Job-2** of the EDA-first research program (the analytic layer
above :mod:`eval_toolkit.eda.data_audit`'s Job-1 integrity gate). Where Job-1
asks *"is the data clean enough to model?"*, Job-2 asks the sharper question a
generic audit misses: *"is the label recoverable from a surface shortcut that
will not transfer out-of-distribution?"* — the mechanism behind the predecessor
prototype's post-hoc OOD wall.

Two analyses, both torch-free (NumPy + scikit-learn only):

- **C1 — weighted log-odds (Monroe, Colaresi & Quinn 2008, informative
  Dirichlet prior) + smoothed PMI.** Surfaces the tokens most associated with
  each class/attack-type, regularized so rare tokens do not dominate. The z-score
  ranking is what feeds the portfolio's V5 log-odds scatter.
- **C2 — partial-input / competency baselines** (length-only, char-n-gram, BoW).
  Each is a deliberately *weak* model: how well it recovers the label is the
  **true shortcut floor**. A baseline scoring far above the positive-prevalence
  floor means the label leaks through a non-semantic channel (annotation
  artifact). Caveat (Feng, Wallace & Boyd-Graber, ACL 2019): a high partial-input
  score bounds *dataset exploitability*, not any particular model's behaviour.

The portfolio consumes C1+C2 per held-out attack-type to build a pre-modeling
shortcut-exposure ranking; see ``portfolio:docs/planning/eda-design.md`` §C and
``portfolio:experiments/eda/OOD_WALL_PREDICTION/criteria.md``.

Stability tier
--------------
``eval_toolkit.eda.*`` — **Tier-2** per ADR 0003 (intentionally evolvable; not in
the v2.0-frozen package-root surface). Import explicitly::

    from eval_toolkit.eda import class_lexical_association, competency_baselines
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from eval_toolkit._rng import RNGLike, SeedLike

__all__ = [
    "DEFAULT_CHAR_NGRAM_RANGE",
    "DEFAULT_MIN_COUNT",
    "DEFAULT_PRIOR_SCALE",
    "BaselineScore",
    "CompetencyResult",
    "LexicalAssociationResult",
    "StrTokenizer",
    "class_lexical_association",
    "competency_baselines",
    "default_tokenizer",
    "weighted_log_odds",
]

# A string tokenizer maps one text to its word-token sequence. Callers may pass a
# domain-specific tokenizer; the default is a lowercased word/number regex. (This
# is distinct from ``data_audit.Tokenizer``, which returns token *ids* for
# length quantiles — here we need the surface word strings for log-odds.)
StrTokenizer = Callable[[str], Sequence[str]]


# Informative-Dirichlet prior strength. The per-token pseudocount is
# ``DEFAULT_PRIOR_SCALE * background_count`` (background = the pooled corpus), so
# the prior is *proportional to overall token frequency* — Monroe et al.'s
# informative prior, not a flat Laplace pseudocount. Small values barely
# regularize; the value below shrinks z-scores for low-count tokens enough that
# the min-count filter (below) is the primary noise control.
DEFAULT_PRIOR_SCALE: Final[float] = 0.01

# Tokens whose *pooled* count (corpus A + corpus B) is below this are dropped
# from the log-odds result — rare-token z-scores are noise (the V5 pitfall).
DEFAULT_MIN_COUNT: Final[int] = 5

# Character n-gram span for the C2 char-n-gram competency baseline. (2, 5) is
# the standard range for catching sub-word shortcuts (suffixes, delimiters,
# punctuation runs) without exploding the feature space.
DEFAULT_CHAR_NGRAM_RANGE: Final[tuple[int, int]] = (2, 5)


# Lowercased word/number tokens. Apostrophes are kept inside tokens (don't,
# it's) but not as boundaries; everything else splits.
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9']+")


def default_tokenizer(text: str) -> list[str]:
    """Lowercase and split ``text`` into word/number tokens.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    list of str
        Lowercased tokens matching ``[a-z0-9']+``. Empty input → ``[]``.

    Examples
    --------
    >>> default_tokenizer("Ignore the PREVIOUS instructions!")
    ['ignore', 'the', 'previous', 'instructions']
    >>> default_tokenizer("don't fail")
    ["don't", 'fail']
    >>> default_tokenizer("")
    []
    """
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True, slots=True)
class LexicalAssociationResult:
    """Per-token class-association statistics (C1).

    Built by :func:`weighted_log_odds` / :func:`class_lexical_association`. Every
    array field is aligned by index with ``tokens`` and contains only tokens whose
    pooled count met the ``min_count`` floor. An empty/over-filtered input yields
    all-empty tuples (no fabricated tokens, no division-by-zero).

    A positive ``z_scores[i]`` / ``pmi[i]`` means token ``i`` leans toward corpus A
    (the positive class, for :func:`class_lexical_association`).

    Parameters
    ----------
    tokens : tuple of str
        The surviving vocabulary, ordered by descending log-odds z-score.
    z_scores : tuple of float
        Weighted log-odds-ratio z-scores (Monroe et al. 2008), corpus A vs B.
    deltas : tuple of float
        The un-standardized log-odds-ratio deltas (A vs B).
    pmi : tuple of float
        Smoothed pointwise mutual information (bits) of each token with corpus A.
    counts_a, counts_b : tuple of int
        Raw token counts in corpus A and corpus B.
    n_a, n_b : int
        Total token counts of corpus A and corpus B.
    min_count : int
        The pooled-count floor applied.
    """

    tokens: tuple[str, ...]
    z_scores: tuple[float, ...]
    deltas: tuple[float, ...]
    pmi: tuple[float, ...]
    counts_a: tuple[int, ...]
    counts_b: tuple[int, ...]
    n_a: int
    n_b: int
    min_count: int

    def top_a(self, k: int = 20) -> list[tuple[str, float]]:
        """Return the ``k`` tokens most associated with corpus A (highest z).

        Parameters
        ----------
        k : int
            Number of tokens to return (clamped to the vocabulary size).

        Returns
        -------
        list of (str, float)
            ``(token, z_score)`` pairs, descending by z-score.
        """
        pairs = list(zip(self.tokens, self.z_scores, strict=True))
        return pairs[: max(k, 0)]

    def top_b(self, k: int = 20) -> list[tuple[str, float]]:
        """Return the ``k`` tokens most associated with corpus B (lowest z).

        Parameters
        ----------
        k : int
            Number of tokens to return (clamped to the vocabulary size).

        Returns
        -------
        list of (str, float)
            ``(token, z_score)`` pairs, ascending by z-score (most B-leaning first).
        """
        pairs = list(zip(self.tokens, self.z_scores, strict=True))
        return pairs[len(pairs) - max(k, 0) :][::-1] if k > 0 else []

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the result.

        Returns
        -------
        dict
            All fields with concrete types preserved.
        """
        return {
            "tokens": list(self.tokens),
            "z_scores": list(self.z_scores),
            "deltas": list(self.deltas),
            "pmi": list(self.pmi),
            "counts_a": list(self.counts_a),
            "counts_b": list(self.counts_b),
            "n_a": self.n_a,
            "n_b": self.n_b,
            "min_count": self.min_count,
        }


def _count_tokens(corpus: Sequence[str], tokenizer: StrTokenizer) -> Counter[str]:
    """Pool ``corpus`` into a single token :class:`collections.Counter`."""
    counts: Counter[str] = Counter()
    for text in corpus:
        counts.update(tokenizer(text))
    return counts


def weighted_log_odds(
    corpus_a: Sequence[str],
    corpus_b: Sequence[str],
    *,
    prior_scale: float = DEFAULT_PRIOR_SCALE,
    min_count: int = DEFAULT_MIN_COUNT,
    tokenizer: StrTokenizer | None = None,
) -> LexicalAssociationResult:
    """Weighted log-odds-ratio with an informative Dirichlet prior (Monroe 2008).

    For each token ``w`` surviving the ``min_count`` floor, computes the
    log-odds-ratio of ``w`` between corpus A and corpus B, regularized by a prior
    pseudocount proportional to ``w``'s pooled frequency, and standardized by its
    analytic variance into a z-score (Monroe, Colaresi & Quinn 2008,
    *Fightin' Words*). Smoothed PMI of each token with corpus A is reported
    alongside. Tokens are returned ordered by descending z-score.

    Parameters
    ----------
    corpus_a, corpus_b : sequence of str
        The two text corpora to contrast.
    prior_scale : float, optional
        Informative-prior strength; per-token pseudocount is
        ``prior_scale * pooled_count``. Default :data:`DEFAULT_PRIOR_SCALE`.
    min_count : int, optional
        Drop tokens whose pooled (A+B) count is below this. Default
        :data:`DEFAULT_MIN_COUNT`.
    tokenizer : callable, optional
        ``str -> sequence of str``. Default :func:`default_tokenizer`.

    Returns
    -------
    LexicalAssociationResult
        Empty (all-zero) when either corpus is empty or no token meets
        ``min_count``.

    Raises
    ------
    ValueError
        If ``prior_scale <= 0`` or ``min_count < 1``.

    Examples
    --------
    >>> a = ["ignore previous instructions", "ignore the system prompt"]
    >>> b = ["the weather is nice", "the weather is cold today"]
    >>> r = weighted_log_odds(a, b, min_count=2)
    >>> r.tokens[0]  # most A-leaning token
    'ignore'
    >>> r.z_scores[0] > 0  # A-leaning => positive z
    True
    >>> "weather" in r.tokens and r.z_scores[r.tokens.index("weather")] < 0
    True
    """
    if prior_scale <= 0:
        raise ValueError(f"prior_scale must be > 0, got {prior_scale}")
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")

    tok = tokenizer or default_tokenizer
    counts_a = _count_tokens(corpus_a, tok)
    counts_b = _count_tokens(corpus_b, tok)
    n_a = int(sum(counts_a.values()))
    n_b = int(sum(counts_b.values()))

    if n_a == 0 or n_b == 0:
        return LexicalAssociationResult(
            tokens=(),
            z_scores=(),
            deltas=(),
            pmi=(),
            counts_a=(),
            counts_b=(),
            n_a=n_a,
            n_b=n_b,
            min_count=min_count,
        )

    # Informative Dirichlet prior: pseudocount ∝ pooled frequency; α0 = total
    # prior mass = prior_scale * pooled token total.
    pooled_total = n_a + n_b
    alpha_0 = prior_scale * pooled_total

    vocab = [
        w
        for w in (counts_a.keys() | counts_b.keys())
        if counts_a.get(w, 0) + counts_b.get(w, 0) >= min_count
    ]
    if not vocab:
        return LexicalAssociationResult(
            tokens=(),
            z_scores=(),
            deltas=(),
            pmi=(),
            counts_a=(),
            counts_b=(),
            n_a=n_a,
            n_b=n_b,
            min_count=min_count,
        )

    rows: list[tuple[str, float, float, float, int, int]] = []
    for w in vocab:
        c_a = counts_a.get(w, 0)
        c_b = counts_b.get(w, 0)
        prior_w = prior_scale * (c_a + c_b)
        # log-odds within each corpus (with the prior in both numerator and the
        # complementary denominator); the denominators are > 0 whenever the
        # corpus has any other token, which holds after the n_a/n_b == 0 guard.
        num_a = c_a + prior_w
        den_a = n_a + alpha_0 - c_a - prior_w
        num_b = c_b + prior_w
        den_b = n_b + alpha_0 - c_b - prior_w
        # Degenerate guard: a denominator is non-positive only when the token
        # *is* essentially the entire corpus (e.g. a single repeated token), so
        # no contrast is defined — skip it rather than crash on log(x/0).
        if den_a <= 0 or den_b <= 0:
            continue
        delta = math.log(num_a / den_a) - math.log(num_b / den_b)
        variance = 1.0 / (c_a + prior_w) + 1.0 / (c_b + prior_w)
        z = delta / math.sqrt(variance)
        # Smoothed PMI(token; corpus A) in bits: P(w|A) over P(w), both
        # prior-smoothed so a token absent from A is finite, not -inf.
        p_w_given_a = num_a / (n_a + alpha_0)
        p_w = (c_a + c_b + 2.0 * prior_w) / (pooled_total + 2.0 * alpha_0)
        pmi = math.log2(p_w_given_a / p_w)
        rows.append((w, z, delta, pmi, c_a, c_b))

    rows.sort(key=lambda r: r[1], reverse=True)
    return LexicalAssociationResult(
        tokens=tuple(r[0] for r in rows),
        z_scores=tuple(r[1] for r in rows),
        deltas=tuple(r[2] for r in rows),
        pmi=tuple(r[3] for r in rows),
        counts_a=tuple(r[4] for r in rows),
        counts_b=tuple(r[5] for r in rows),
        n_a=n_a,
        n_b=n_b,
        min_count=min_count,
    )


def class_lexical_association(
    texts: Sequence[str],
    labels: Sequence[object],
    *,
    positive_label: object = 1,
    prior_scale: float = DEFAULT_PRIOR_SCALE,
    min_count: int = DEFAULT_MIN_COUNT,
    tokenizer: StrTokenizer | None = None,
) -> LexicalAssociationResult:
    """Log-odds + PMI of the positive class vs the negative class (C1).

    Convenience wrapper over :func:`weighted_log_odds` that splits ``texts`` by
    ``labels`` into corpus A = positive class, corpus B = everything else, so a
    positive z-score / PMI marks a positive-class shortcut token.

    Parameters
    ----------
    texts : sequence of str
        All texts.
    labels : sequence
        Per-text labels, aligned with ``texts``.
    positive_label : object, optional
        The value in ``labels`` denoting the positive class. Default ``1``.
    prior_scale, min_count, tokenizer
        Forwarded to :func:`weighted_log_odds`.

    Returns
    -------
    LexicalAssociationResult
        Corpus A = positive class, corpus B = negative class.

    Raises
    ------
    ValueError
        If ``texts`` and ``labels`` differ in length, if ``positive_label``
        matches no label (e.g. a type mismatch: ``positive_label=1`` against
        string labels ``"1"``), or if every label matches ``positive_label``
        (the negative corpus would be empty).

    Examples
    --------
    >>> texts = ["ignore previous instructions", "the weather is nice",
    ...          "ignore the system prompt", "the weather is cold"]
    >>> labels = [1, 0, 1, 0]
    >>> r = class_lexical_association(texts, labels, min_count=2)
    >>> r.tokens[0]
    'ignore'
    >>> r.z_scores[0] > 0
    True
    """
    if len(texts) != len(labels):
        raise ValueError(f"texts ({len(texts)}) and labels ({len(labels)}) length mismatch")
    pos = [t for t, y in zip(texts, labels, strict=True) if y == positive_label]
    neg = [t for t, y in zip(texts, labels, strict=True) if y != positive_label]
    # An unmatched positive_label (typically a type mismatch — 1 vs "1") would
    # otherwise produce an empty corpus and a documented all-empty result that
    # silently reads "no shortcut signal". repr() exposes the type difference.
    if not pos:
        observed = sorted({repr(y) for y in labels})
        raise ValueError(
            f"positive_label {positive_label!r} matches no label; "
            f"observed label values: {observed}"
        )
    if not neg:
        raise ValueError(
            f"every label matches positive_label {positive_label!r}; "
            "the negative corpus would be empty"
        )
    return weighted_log_odds(
        pos, neg, prior_scale=prior_scale, min_count=min_count, tokenizer=tokenizer
    )


@dataclass(frozen=True, slots=True)
class BaselineScore:
    """One competency baseline's held-out score (C2).

    Parameters
    ----------
    name : str
        Baseline identifier (``"length"`` / ``"char_ngram"`` / ``"bow"``).
    average_precision : float
        Held-out average precision (area under the PR curve).
    roc_auc : float
        Held-out ROC AUC.
    positive_prevalence : float
        Positive-class prevalence in the test set — the AP floor (a baseline at
        the floor has learned nothing exploitable).
    """

    name: str
    average_precision: float
    roc_auc: float
    positive_prevalence: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the score."""
        return {
            "name": self.name,
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
            "positive_prevalence": self.positive_prevalence,
        }


@dataclass(frozen=True, slots=True)
class CompetencyResult:
    """Competency-baseline shortcut floor over one train→test split (C2).

    Parameters
    ----------
    baselines : tuple of BaselineScore
        One entry per baseline, in fit order (length, char_ngram, bow).
    n_train, n_test : int
        Train / test row counts.
    test_positive_prevalence : float
        Positive prevalence of the test set (the shared AP floor).
    """

    baselines: tuple[BaselineScore, ...]
    n_train: int
    n_test: int
    test_positive_prevalence: float

    def best(self) -> BaselineScore:
        """Return the baseline with the highest average precision.

        Returns
        -------
        BaselineScore

        Raises
        ------
        ValueError
            If there are no baselines.
        """
        if not self.baselines:
            raise ValueError("no baselines to compare")
        return max(self.baselines, key=lambda b: b.average_precision)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the result."""
        return {
            "baselines": [b.to_dict() for b in self.baselines],
            "n_train": self.n_train,
            "n_test": self.n_test,
            "test_positive_prevalence": self.test_positive_prevalence,
        }


def _binarize(labels: Sequence[object], positive_label: object) -> np.ndarray:
    """Map labels to a 0/1 int array (1 == ``positive_label``)."""
    return np.fromiter((1 if y == positive_label else 0 for y in labels), dtype=int)


def _score_baseline(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    sklearn_seed: int,
) -> BaselineScore:
    """Fit a logistic regression on ``x_train`` and score it on the test set."""
    clf = LogisticRegression(max_iter=1000, random_state=sklearn_seed)
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    return BaselineScore(
        name=name,
        average_precision=float(average_precision_score(y_test, scores)),
        roc_auc=float(roc_auc_score(y_test, scores)),
        positive_prevalence=float(y_test.mean()),
    )


def competency_baselines(
    train_texts: Sequence[str],
    train_labels: Sequence[object],
    test_texts: Sequence[str],
    test_labels: Sequence[object],
    *,
    positive_label: object = 1,
    char_ngram_range: tuple[int, int] = DEFAULT_CHAR_NGRAM_RANGE,
    rng: RNGLike | SeedLike | None = 0,
) -> CompetencyResult:
    """Fit partial-input / competency baselines and score the shortcut floor (C2).

    Three deliberately weak models are trained on the training split and scored on
    the test split: **length-only** (a single character-length feature),
    **char-n-gram** (sub-word :class:`~sklearn.feature_extraction.text.CountVectorizer`
    + logistic regression), and **BoW** (word unigram counts + logistic
    regression). Each baseline's held-out average precision relative to the
    positive-prevalence floor measures how much label signal leaks through a
    non-semantic channel — the *shortcut floor*. When evaluated per held-out
    attack-type, a high floor that fails to transfer predicts OOD collapse.

    Parameters
    ----------
    train_texts, train_labels : sequence
        Training texts and aligned labels.
    test_texts, test_labels : sequence
        Test texts and aligned labels.
    positive_label : object, optional
        Value in the label sequences denoting the positive class. Default ``1``.
    char_ngram_range : (int, int), optional
        Character n-gram span for the char-n-gram baseline. Default
        :data:`DEFAULT_CHAR_NGRAM_RANGE`.
    rng : RNGLike | SeedLike | None, optional
        RNG argument per `Scientific Python SPEC 7 <https://scientific-python.org/specs/spec-0007/>`_;
        seeds the logistic-regression solvers. **Inert under the default
        ``lbfgs`` solver** (lbfgs ignores ``random_state``); it only changes
        output for stochastic solvers such as ``saga`` / ``liblinear``. Int
        seed (default ``0``), ``Generator``, or ``None`` (entropy).

    Returns
    -------
    CompetencyResult

    Raises
    ------
    ValueError
        If any text/label sequence length mismatches, if a split is empty, or if
        either the train or test labels are single-class (AP / ROC AUC undefined).

    Examples
    --------
    >>> tr = ["ignore the instructions", "the weather is nice",
    ...       "ignore previous text", "the cat sat down"]
    >>> tr_y = [1, 0, 1, 0]
    >>> te = ["ignore all of this", "the dog ran fast"]
    >>> te_y = [1, 0]
    >>> res = competency_baselines(tr, tr_y, te, te_y)
    >>> [b.name for b in res.baselines]
    ['length', 'char_ngram', 'bow']
    >>> res.n_train, res.n_test
    (4, 2)
    """
    if len(train_texts) != len(train_labels):
        raise ValueError(
            f"train texts ({len(train_texts)}) and labels ({len(train_labels)}) length mismatch"
        )
    if len(test_texts) != len(test_labels):
        raise ValueError(
            f"test texts ({len(test_texts)}) and labels ({len(test_labels)}) length mismatch"
        )
    if len(train_texts) == 0 or len(test_texts) == 0:
        raise ValueError("train and test splits must both be non-empty")

    y_train = _binarize(train_labels, positive_label)
    y_test = _binarize(test_labels, positive_label)
    if len(set(y_train.tolist())) < 2:
        raise ValueError("train labels are single-class; logistic baselines cannot fit")
    if len(set(y_test.tolist())) < 2:
        raise ValueError("test labels are single-class; average precision / ROC AUC undefined")

    # length-only: one feature = character length.
    len_train = np.array([[len(t)] for t in train_texts], dtype=float)
    len_test = np.array([[len(t)] for t in test_texts], dtype=float)

    # char-n-gram and BoW vectorizers are fit on TRAIN only (no test leakage).
    char_vec = CountVectorizer(analyzer="char", ngram_range=char_ngram_range)
    char_train = char_vec.fit_transform(train_texts)
    char_test = char_vec.transform(test_texts)

    bow_vec = CountVectorizer(analyzer="word")
    bow_train = bow_vec.fit_transform(train_texts)
    bow_test = bow_vec.transform(test_texts)

    # Derive an int seed for sklearn at the boundary (sklearn's random_state
    # does not accept Generator across supported versions) — the
    # cross_validate_metric pattern.
    gen = np.random.default_rng(rng)
    sklearn_seed = int(gen.integers(0, 2**31 - 1))

    baselines = (
        _score_baseline("length", len_train, y_train, len_test, y_test, sklearn_seed=sklearn_seed),
        _score_baseline(
            "char_ngram", char_train, y_train, char_test, y_test, sklearn_seed=sklearn_seed
        ),
        _score_baseline("bow", bow_train, y_train, bow_test, y_test, sklearn_seed=sklearn_seed),
    )
    return CompetencyResult(
        baselines=baselines,
        n_train=len(train_texts),
        n_test=len(test_texts),
        test_positive_prevalence=float(y_test.mean()),
    )
