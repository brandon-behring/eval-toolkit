"""Adversarial robustness: 12-technique character-injection bypass suite.

Implements the character-injection bypass techniques from Microsoft Research
2024 ([1]_) for testing prompt-injection-detection scorers under adversarial
input perturbation. Each technique is deterministic given a ``seed`` and
preserves the surface meaning of the text from a human reader's perspective
while shifting the tokenizer / scorer's representation.

Core techniques (shipped in v0.43.0):

- :class:`ZeroWidthSpaceInjection` — insert U+200B zero-width spaces
- :class:`HomoglyphSubstitution` — Latin → Cyrillic/Greek lookalikes
- :class:`DiacriticInjection` — combining-mark insertion (NFC bypass)
- :class:`WhitespaceInjection` — variable whitespace padding (regular + NBSP)
- :class:`CaseRandomization` — random case-flipping per character
- :class:`PunctuationInjection` — non-semantic punctuation insertion

Advanced techniques (shipped in v0.47 per Decision Q11.3):

- :class:`BidiRTLInjection` — U+202E…U+202C override block
- :class:`TagStrippingInjection` — ``<…>`` tag removal (idempotent)
- :class:`SynonymSubstitution` — whitelisted-word swap, seed-deterministic
- :class:`TokenSplitting` — mid-word single-space insertion
- :class:`UnicodeNormalization` — NFC / NFD / NFKC / NFKD form switch
- :class:`InvisibleCharsInjection` — 5 invisible code points

The convenience tuples :data:`CORE_TECHNIQUES` (6-tuple),
:data:`ADVANCED_TECHNIQUES` (6-tuple), and :data:`ALL_TECHNIQUES`
(12-tuple = core + advanced) enumerate the suite for sweep callers.

Use the v0.47 top-level :func:`eval_toolkit.sweep` to apply any set of
:class:`~eval_toolkit.TextTransform` strategies against a corpus (and
optionally a :class:`~eval_toolkit.protocols.Scorer`); the v0.43–v0.46
module-level ``sweep()`` function and the ``character_injection``
``SimpleNamespace`` were removed at v0.47 (Decisions D + K + N).

References
----------
.. [1] Microsoft Research, 2024. "Adversarial Suffixes Hidden in Plain
       Sight." arXiv:2404.13208.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

__all__ = [
    "ADVANCED_TECHNIQUES",
    "ALL_TECHNIQUES",
    "BidiRTLInjection",
    "CORE_TECHNIQUES",
    "CaseRandomization",
    "DiacriticInjection",
    "HomoglyphSubstitution",
    "InvisibleCharsInjection",
    "PunctuationInjection",
    "SynonymSubstitution",
    "TagStrippingInjection",
    "TokenSplitting",
    "UnicodeNormalization",
    "WhitespaceInjection",
    "ZeroWidthSpaceInjection",
]


# Character constants
_ZWSP = "​"  # zero-width space
_NBSP = " "  # non-breaking space
_TAB = "\t"
_PUNCT_CHARS = ".,;:!?-_"

# Latin → Cyrillic homoglyph substitution table (visually identical, distinct
# code points). Subset chosen to maximize Unicode-confusable coverage without
# breaking text legibility in most fonts.
_HOMOGLYPHS: dict[str, str] = {
    "a": "а",  # CYRILLIC SMALL LETTER A
    "c": "с",  # CYRILLIC SMALL LETTER ES
    "e": "е",  # CYRILLIC SMALL LETTER IE
    "o": "о",  # CYRILLIC SMALL LETTER O
    "p": "р",  # CYRILLIC SMALL LETTER ER
    "x": "х",  # CYRILLIC SMALL LETTER HA
    "y": "у",  # CYRILLIC SMALL LETTER U
    "A": "А",  # CYRILLIC CAPITAL LETTER A
    "B": "В",  # CYRILLIC CAPITAL LETTER VE
    "E": "Е",  # CYRILLIC CAPITAL LETTER IE
    "K": "К",  # CYRILLIC CAPITAL LETTER KA
    "O": "О",  # CYRILLIC CAPITAL LETTER O
    "P": "Р",  # CYRILLIC CAPITAL LETTER ER
    "T": "Т",  # CYRILLIC CAPITAL LETTER TE
}

# Combining marks (NFD-style). Applied after a base char they render as
# diacritics; the visual is unchanged in most fonts but the byte stream
# carries extra code points.
_COMBINING_MARKS: tuple[str, ...] = (
    "̀",  # combining grave accent
    "́",  # combining acute accent
    "̂",  # combining circumflex accent
    "̃",  # combining tilde
    "̄",  # combining macron
    "̆",  # combining breve
    "̇",  # combining dot above
    "̈",  # combining diaeresis
)

# Whitespace variants (regular " " excluded — substituting space-for-space is a
# no-op and prevents the test from observing the substitution actually happened).
_WHITESPACE_VARIANTS: tuple[str, ...] = (_NBSP, _TAB)


# The v0.43–v0.46 ``CharacterInjectionStrategy`` per-module Protocol was
# removed at v0.47 (Decision K + plan §4B). Use the top-level
# :class:`eval_toolkit.TextTransform` Protocol instead — it has the same
# shape (``name: str`` + ``transform(text) -> str``) and covers both
# adversarial and preprocessing strategies. Every concrete adversarial
# class continues to satisfy ``TextTransform`` structurally.


@dataclass(frozen=True, slots=True)
class ZeroWidthSpaceInjection:
    """Insert U+200B zero-width spaces between characters at the given ratio.

    Visually identical to the original text; tokenizers split on the inserted
    code point, often shifting sub-word boundaries enough to evade pattern-
    matching detectors.

    Parameters
    ----------
    ratio : float, optional
        Probability of inserting a ZWSP after each character. ``0.5`` (default)
        injects between roughly every other char. Must be in ``[0, 1]``.
    seed : int, optional
        Random seed for deterministic insertion. Default ``42``.
    name : str, optional
        Override the default technique name (``"zero_width_space"``); rarely
        needed.
    """

    ratio: float = 0.5
    seed: int = 42
    name: str = "zero_width_space"

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"ZeroWidthSpaceInjection: ratio must be in [0, 1]; got {self.ratio}")

    def transform(self, text: str) -> str:
        rng = random.Random(self.seed)
        out: list[str] = []
        for ch in text:
            out.append(ch)
            if rng.random() < self.ratio:
                out.append(_ZWSP)
        return "".join(out)


@dataclass(frozen=True, slots=True)
class HomoglyphSubstitution:
    """Substitute Latin characters with Cyrillic/Greek homoglyph lookalikes.

    Each substitution is deterministic given the seed: a per-character RNG
    decides whether to swap based on ``ratio``. Characters not in the
    homoglyph table pass through unchanged.

    Parameters
    ----------
    ratio : float, optional
        Probability of substituting each homoglyph-eligible character.
        Default ``0.3``.
    seed : int, optional
        Random seed for determinism. Default ``42``.
    name : str, optional
        Override technique name. Default ``"homoglyph"``.
    """

    ratio: float = 0.3
    seed: int = 42
    name: str = "homoglyph"

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"HomoglyphSubstitution: ratio must be in [0, 1]; got {self.ratio}")

    def transform(self, text: str) -> str:
        rng = random.Random(self.seed)
        out: list[str] = []
        for ch in text:
            if ch in _HOMOGLYPHS and rng.random() < self.ratio:
                out.append(_HOMOGLYPHS[ch])
            else:
                out.append(ch)
        return "".join(out)


@dataclass(frozen=True, slots=True)
class DiacriticInjection:
    """Insert combining diacritic marks after random characters.

    Combining marks (U+0300..U+0308) attach to the preceding character
    visually but produce a distinct code-point stream that tokenizers
    process as different sub-words. Most fonts render the result with
    subtle visual artifacts (zalgo-style for high ratios).

    Parameters
    ----------
    ratio : float, optional
        Probability of inserting a combining mark after each base
        character. Default ``0.3``.
    seed : int, optional
        Random seed. Default ``42``.
    name : str, optional
        Override technique name. Default ``"diacritic"``.
    """

    ratio: float = 0.3
    seed: int = 42
    name: str = "diacritic"

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"DiacriticInjection: ratio must be in [0, 1]; got {self.ratio}")

    def transform(self, text: str) -> str:
        rng = random.Random(self.seed)
        out: list[str] = []
        for ch in text:
            out.append(ch)
            # Combining marks attach to base characters (alphanumeric); skip on
            # whitespace / punctuation where they render poorly.
            if ch.isalnum() and rng.random() < self.ratio:
                out.append(rng.choice(_COMBINING_MARKS))
        return "".join(out)


@dataclass(frozen=True, slots=True)
class WhitespaceInjection:
    """Insert / substitute whitespace variants (regular, non-breaking, tab).

    Replaces existing spaces with random whitespace variants and optionally
    pads non-space positions with extra spaces. Detectors that tokenize on
    whitespace can shift token boundaries; some lose detection signal
    entirely when NBSPs replace regular spaces.

    Parameters
    ----------
    pad_ratio : float, optional
        Probability of inserting an extra space after each character (regardless
        of whether the position was a space). Default ``0.1``.
    substitute_ratio : float, optional
        Probability of replacing each existing space with a non-breaking-space
        or tab variant. Default ``0.5``.
    seed : int, optional
        Random seed. Default ``42``.
    name : str, optional
        Override technique name. Default ``"whitespace"``.
    """

    pad_ratio: float = 0.1
    substitute_ratio: float = 0.5
    seed: int = 42
    name: str = "whitespace"

    def __post_init__(self) -> None:
        for field_name, val in (
            ("pad_ratio", self.pad_ratio),
            ("substitute_ratio", self.substitute_ratio),
        ):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"WhitespaceInjection: {field_name} must be in [0, 1]; got {val}")

    def transform(self, text: str) -> str:
        rng = random.Random(self.seed)
        out: list[str] = []
        for ch in text:
            if ch == " " and rng.random() < self.substitute_ratio:
                out.append(rng.choice(_WHITESPACE_VARIANTS))
            else:
                out.append(ch)
            if rng.random() < self.pad_ratio:
                out.append(" ")
        return "".join(out)


@dataclass(frozen=True, slots=True)
class CaseRandomization:
    """Randomly flip the case of alphabetic characters.

    Deterministic given the seed. Numeric / punctuation / whitespace pass
    through unchanged. Useful against detectors that don't lower-case their
    input.

    Parameters
    ----------
    ratio : float, optional
        Probability of flipping each alphabetic character's case.
        Default ``0.5``.
    seed : int, optional
        Random seed. Default ``42``.
    name : str, optional
        Override technique name. Default ``"case_random"``.
    """

    ratio: float = 0.5
    seed: int = 42
    name: str = "case_random"

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"CaseRandomization: ratio must be in [0, 1]; got {self.ratio}")

    def transform(self, text: str) -> str:
        rng = random.Random(self.seed)
        out: list[str] = []
        for ch in text:
            if ch.isalpha() and rng.random() < self.ratio:
                out.append(ch.lower() if ch.isupper() else ch.upper())
            else:
                out.append(ch)
        return "".join(out)


@dataclass(frozen=True, slots=True)
class PunctuationInjection:
    """Insert non-semantic punctuation between characters.

    Inserts characters from ``.,;:!?-_`` at random positions; renders as
    obvious noise to a human but can fragment regex / pattern-matching
    detectors. Use sparingly (low ``ratio``).

    Parameters
    ----------
    ratio : float, optional
        Probability of inserting punctuation after each character.
        Default ``0.1``.
    seed : int, optional
        Random seed. Default ``42``.
    name : str, optional
        Override technique name. Default ``"punctuation"``.
    """

    ratio: float = 0.1
    seed: int = 42
    name: str = "punctuation"

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"PunctuationInjection: ratio must be in [0, 1]; got {self.ratio}")

    def transform(self, text: str) -> str:
        rng = random.Random(self.seed)
        out: list[str] = []
        for ch in text:
            out.append(ch)
            # Only inject after alphanumeric chars; avoids double-punctuation
            # noise where it would be obvious to a human reader.
            if ch.isalnum() and rng.random() < self.ratio:
                out.append(rng.choice(_PUNCT_CHARS))
        return "".join(out)


# ----------------------------------------------------------------------------
# Advanced 6 — added at v0.47.0 (Decision Q11→11.3; closes the forward-look
# from the v0.43.0 CHANGELOG). All satisfy the top-level
# :class:`eval_toolkit.TextTransform` Protocol structurally.
# ----------------------------------------------------------------------------


# Unicode bidi controls used by BidiRTLInjection
_RLO = "‮"  # RIGHT-TO-LEFT OVERRIDE
_LRO = "‭"  # LEFT-TO-RIGHT OVERRIDE
_PDF = "‬"  # POP DIRECTIONAL FORMATTING

# Invisible / ignorable Unicode code points used by InvisibleCharsInjection.
# These render to zero advance width in most environments and are skipped by
# many tokenizers / search engines, but contribute distinct bytes that perturb
# subword segmentation.
_INVISIBLE_CHARS: tuple[str, ...] = (
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "⁠",  # WORD JOINER
    "﻿",  # ZERO WIDTH NO-BREAK SPACE (BOM)
)

# Light synonym table — restricted to function words + a handful of common
# verbs so the substitution preserves semantics with high probability.
# Intentionally tiny; larger tables introduce semantic drift that defeats
# the "looks-like-the-original" invariant the technique relies on.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "ignore": ("disregard", "overlook"),
    "instructions": ("directions", "guidance"),
    "system": ("framework", "platform"),
    "secret": ("private", "confidential"),
    "send": ("transmit", "forward"),
    "all": ("every", "all of"),
}


@dataclass(frozen=True, slots=True)
class BidiRTLInjection:
    """Wrap the input in a Unicode bidi-RTL override block.

    Prepends ``U+202E`` (RIGHT-TO-LEFT OVERRIDE) and appends ``U+202C``
    (POP DIRECTIONAL FORMATTING). Visually the text often renders in
    reverse; many naive tokenizers ignore the controls entirely, so the
    byte stream differs from the original while the semantic intent
    survives.

    Parameters
    ----------
    name : str, optional
        Override technique name. Default ``"bidi_rtl"``.
    """

    name: str = "bidi_rtl"

    def transform(self, text: str) -> str:
        if not text:
            return text
        return f"{_RLO}{text}{_PDF}"


@dataclass(frozen=True, slots=True)
class TagStrippingInjection:
    """Strip HTML/XML-like tags from the input.

    Removes anything matching ``<...>`` (greedy, single-line). Useful for
    studying how detectors trained on tagged markup behave when callers
    pre-strip tags. Idempotent — running twice produces the same output
    as running once.

    Parameters
    ----------
    name : str, optional
        Override technique name. Default ``"tag_strip"``.
    """

    name: str = "tag_strip"

    def transform(self, text: str) -> str:
        # Non-greedy match so `<a><b>` strips both tags rather than
        # collapsing the whole interval. ``re.DOTALL`` so `<…\n…>` also
        # gets stripped — pathological but represented in real prompt-
        # injection payloads.
        import re as _re

        return _re.sub(r"<[^>]*>", "", text, flags=_re.DOTALL)


@dataclass(frozen=True, slots=True)
class SynonymSubstitution:
    """Replace whitelisted words with deterministic synonyms.

    Only the keys in the internal whitelist (``_SYNONYMS``) are eligible.
    For each eligible word the choice is deterministic given ``seed``.

    Parameters
    ----------
    ratio : float, optional
        Per-word substitution probability among the eligible set.
        Default ``1.0`` (always substitute).
    seed : int, optional
        Random seed for determinism. Default ``42``.
    name : str, optional
        Override technique name. Default ``"synonym"``.

    Notes
    -----
    The eligible-word set is the module-level ``_SYNONYMS`` dict, a fixed
    6-entry whitelist hand-curated to preserve semantics:

    - ``ignore`` → ``disregard``, ``overlook``
    - ``instructions`` → ``directions``, ``guidance``
    - ``system`` → ``framework``, ``platform``
    - ``secret`` → ``private``, ``confidential``
    - ``send`` → ``transmit``, ``forward``
    - ``all`` → ``every``, ``all of``

    Inputs containing none of those whitelist words are returned unchanged
    — the transform is a no-op on such inputs. This is intentional: the
    technique's invariant is "looks like the original," so the substitution
    deliberately stays small. The trade-off is easy to be surprised by
    when running ``SynonymSubstitution`` on a corpus that doesn't share
    the prompt-injection vocabulary the whitelist was built from. If you
    need broader substitution, the whitelist isn't extension-friendly
    today — fork the dict at the module level, or treat
    ``SynonymSubstitution`` as a reference implementation for your own
    text-transform with a richer table.
    """

    ratio: float = 1.0
    seed: int = 42
    name: str = "synonym"

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"SynonymSubstitution: ratio must be in [0, 1]; got {self.ratio}")

    def transform(self, text: str) -> str:
        rng = random.Random(self.seed)
        import re as _re

        def _maybe_swap(match: _re.Match[str]) -> str:
            word = match.group(0)
            key = word.lower()
            if key not in _SYNONYMS:
                return word
            if rng.random() >= self.ratio:
                return word
            choice = rng.choice(_SYNONYMS[key])
            # Best-effort case preservation for capitalized words
            if word.istitle():
                return choice.capitalize()
            if word.isupper():
                return choice.upper()
            return choice

        return _re.sub(r"\b\w+\b", _maybe_swap, text)


@dataclass(frozen=True, slots=True)
class TokenSplitting:
    """Insert a single space inside each long enough word.

    Forces subword tokenizers to break a single token into two, often
    enough to slip past a detector's keyword-style heuristics while
    leaving the text human-readable.

    Parameters
    ----------
    min_word_length : int, optional
        Only words with at least this many characters are split. Default
        ``5``.
    ratio : float, optional
        Probability of splitting each eligible word. Default ``1.0``.
    seed : int, optional
        Random seed for determinism. Default ``42``.
    name : str, optional
        Override technique name. Default ``"token_split"``.
    """

    min_word_length: int = 5
    ratio: float = 1.0
    seed: int = 42
    name: str = "token_split"

    def __post_init__(self) -> None:
        if self.min_word_length < 2:
            raise ValueError(
                f"TokenSplitting: min_word_length must be >= 2; got {self.min_word_length}"
            )
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"TokenSplitting: ratio must be in [0, 1]; got {self.ratio}")

    def transform(self, text: str) -> str:
        rng = random.Random(self.seed)
        import re as _re

        def _maybe_split(match: _re.Match[str]) -> str:
            word = match.group(0)
            if len(word) < self.min_word_length:
                return word
            if rng.random() >= self.ratio:
                return word
            # Split near the middle (deterministic given seed: position
            # offset depends on `rng.randint` not on the word content).
            offset = rng.randint(1, max(1, len(word) - 1))
            return word[:offset] + " " + word[offset:]

        return _re.sub(r"\b\w+\b", _maybe_split, text)


@dataclass(frozen=True, slots=True)
class UnicodeNormalization:
    """Apply a Unicode normalization form to the input.

    Defaults to NFKC which folds compatibility characters (e.g., ``ＡＢＣ``
    fullwidth → ``ABC`` ASCII). The byte stream changes substantially; many
    detectors trained on natural text don't see the variant code points,
    while NFKC produces a canonical ASCII-ish equivalent.

    Parameters
    ----------
    form : {"NFC", "NFD", "NFKC", "NFKD"}, optional
        Normalization form. Default ``"NFKC"``.
    name : str, optional
        Override technique name. Default ``"unicode_normalize"``.
    """

    form: str = "NFKC"
    name: str = "unicode_normalize"

    def __post_init__(self) -> None:
        if self.form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError(
                f"UnicodeNormalization: form must be NFC / NFD / NFKC / NFKD; got {self.form!r}"
            )

    def transform(self, text: str) -> str:
        import unicodedata as _ud

        return _ud.normalize(self.form, text)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class InvisibleCharsInjection:
    """Insert invisible / zero-width Unicode characters between characters.

    Distinct from :class:`ZeroWidthSpaceInjection` in that this technique
    samples from a tuple of 5 invisible code points (ZWSP, ZWNJ, ZWJ,
    word joiner, BOM) instead of always inserting U+200B.

    Parameters
    ----------
    ratio : float, optional
        Per-position insertion probability. Default ``0.5``.
    seed : int, optional
        Random seed for determinism. Default ``42``.
    name : str, optional
        Override technique name. Default ``"invisible_chars"``.
    """

    ratio: float = 0.5
    seed: int = 42
    name: str = "invisible_chars"

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"InvisibleCharsInjection: ratio must be in [0, 1]; got {self.ratio}")

    def transform(self, text: str) -> str:
        rng = random.Random(self.seed)
        out: list[str] = []
        for ch in text:
            out.append(ch)
            if rng.random() < self.ratio:
                out.append(rng.choice(_INVISIBLE_CHARS))
        return "".join(out)


# Core 6 — registered by default in :func:`sweep` when ``techniques="all"``
# at v0.43.0. Advanced 6 added at v0.47.0 (Decision Q11→11.3) live in
# :data:`ADVANCED_TECHNIQUES`. Together they form the v0.47-final
# 12-technique suite; ``ALL_TECHNIQUES`` is the union for callers wanting
# the complete set.
#
# Annotated as ``tuple[type[Any], ...]`` because Protocol classes can't be
# used as the parameter of ``type[...]`` in concrete tuple literals without
# triggering a strict-mypy assignability error (Protocol is structural;
# ``type[Protocol]`` rejects concrete-class types). The runtime check in
# :func:`sweep` re-asserts each instance satisfies the Protocol.
CORE_TECHNIQUES: tuple[type[Any], ...] = (
    ZeroWidthSpaceInjection,
    HomoglyphSubstitution,
    DiacriticInjection,
    WhitespaceInjection,
    CaseRandomization,
    PunctuationInjection,
)
ADVANCED_TECHNIQUES: tuple[type[Any], ...] = (
    BidiRTLInjection,
    TagStrippingInjection,
    SynonymSubstitution,
    TokenSplitting,
    UnicodeNormalization,
    InvisibleCharsInjection,
)
ALL_TECHNIQUES: tuple[type[Any], ...] = CORE_TECHNIQUES + ADVANCED_TECHNIQUES


# ----------------------------------------------------------------------------
# Module-level function namespace (matches issue spec API)
# ----------------------------------------------------------------------------


def _zero_width_space(text: str, ratio: float = 0.5, seed: int = 42) -> str:
    """Functional alias for :class:`ZeroWidthSpaceInjection`."""
    return ZeroWidthSpaceInjection(ratio=ratio, seed=seed).transform(text)


def _homoglyph(text: str, ratio: float = 0.3, seed: int = 42) -> str:
    """Functional alias for :class:`HomoglyphSubstitution`."""
    return HomoglyphSubstitution(ratio=ratio, seed=seed).transform(text)


def _diacritic(text: str, ratio: float = 0.3, seed: int = 42) -> str:
    """Functional alias for :class:`DiacriticInjection`."""
    return DiacriticInjection(ratio=ratio, seed=seed).transform(text)


def _whitespace(
    text: str, pad_ratio: float = 0.1, substitute_ratio: float = 0.5, seed: int = 42
) -> str:
    """Functional alias for :class:`WhitespaceInjection`."""
    return WhitespaceInjection(
        pad_ratio=pad_ratio, substitute_ratio=substitute_ratio, seed=seed
    ).transform(text)


def _case_random(text: str, ratio: float = 0.5, seed: int = 42) -> str:
    """Functional alias for :class:`CaseRandomization`."""
    return CaseRandomization(ratio=ratio, seed=seed).transform(text)


def _punctuation(text: str, ratio: float = 0.1, seed: int = 42) -> str:
    """Functional alias for :class:`PunctuationInjection`."""
    return PunctuationInjection(ratio=ratio, seed=seed).transform(text)


# Module-level ``character_injection`` SimpleNamespace + module-level
# ``sweep`` were removed at v0.47 (Decisions D + K + N; plan §§4C/4E).
# The top-level :func:`eval_toolkit.sweep` consumes any
# :class:`TextTransform`; the 6 internal functional aliases above
# (``_zero_width_space`` etc.) remain as implementation conveniences
# used by the dataclasses' ``transform`` methods.
