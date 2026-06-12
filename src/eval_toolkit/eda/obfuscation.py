"""Obfuscation / encoding / invisible-Unicode prevalence detectors.

This module is the **Job-1 integrity prerequisite** companion to
:mod:`eval_toolkit.eda.data_audit` : it measures *seen-text ≠ scored-text*
hazards in a dataset's text columns before any downstream length / n-gram /
embedding statistic is computed. Invisible characters silently corrupt
every character-based stat; encoded blobs (base64 / hex) destroy lexical
signal; homoglyphs and compatibility-decomposable characters
(mathematical-bold Latin, fullwidth, ligatures) defeat regex-based markers.

Scope is intentionally narrow and torch-free: pure-stdlib detectors and a
single aggregating :class:`ObfuscationProfile`. Public functions are
composable so notebooks can use any one detector standalone; the
:func:`analyze_obfuscation` orchestrator returns the corpus-level
prevalence dataclass that
:func:`eval_toolkit.eda.data_audit.audit_dataset` records per split.

Detectors
---------
- **Invisible Unicode** — Zero-Width Space / Joiner / Non-Joiner, Word
  Joiner, BOM / ZWNBSP, Variation Selectors (basic + supplement), Tags
  block (U+E0000 – U+E007F).
- **NFKC normalization delta** — counts texts whose NFKC normalization
  differs from the raw text, plus the absolute char-count delta. Catches
  mathematical alphanumeric Latin, fullwidth Latin, ligatures, super/
  subscript digits — common PI obfuscations against regex detectors.
- **High-entropy alnum runs** — base64-alphabet runs (≥20 chars,
  Shannon entropy ≥ 4.5 bits/char) and hex runs (≥16 chars, entropy
  ≥ 3.5 bits/char). Conservative thresholds — natural English doesn't
  reach them.
- **ROT13 PI markers** — presence of ROT13-encoded forms of common
  prompt-injection vocabulary (e.g. ``vtaber`` = ``ignore``).
- **Leetspeak tokens** — alphanumeric tokens with ≥2 leet substitutions
  (e.g. ``1gn0r3``). Pure version-numbery suffixes (``v1``, ``py3``) are
  excluded.

Why these specific detectors are the *integrity* slice — and ROT13 +
leetspeak count here even though they're attack-content signals — is
documented in
``portfolio:docs/planning/eda-design.md`` §B2 and §7. The §B2 full list
is implemented here; downstream Job-2 modules will revisit ROT13 /
leetspeak as attack-type technique features.
"""

from __future__ import annotations

import codecs
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "BASE64_ALNUM_MIN_RUN",
    "BASE64_ENTROPY_THRESHOLD",
    "HEX_ALNUM_MIN_RUN",
    "HEX_ENTROPY_THRESHOLD",
    "INVISIBLE_RANGES",
    "LEET_SUBSTITUTIONS",
    "ObfuscationProfile",
    "ROT13_PI_MARKERS",
    "analyze_obfuscation",
    "count_invisible_chars",
    "has_high_entropy_alnum_run",
    "has_rot13_marker",
    "is_leeted_token",
    "leetspeak_counts",
    "nfkc_char_delta",
    "nfkc_changed",
    "shannon_entropy",
]


# Codepoint ranges (inclusive) that render as zero-width / invisible. Used by
# count_invisible_chars; not every glyph in these ranges is malicious (a
# legitimate emoji ZWJ sequence uses U+200D) but in a *text classification*
# corpus their presence is a corruption signal worth flagging.
INVISIBLE_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x200B, 0x200D),  # ZWSP, ZWNJ, ZWJ
    (0x2060, 0x2060),  # WORD JOINER
    (0xFEFF, 0xFEFF),  # ZWNBSP / BOM
    (0xFE00, 0xFE0F),  # Variation Selectors
    (0xE0000, 0xE007F),  # Tags block (Unicode Tag injection vector)
    (0xE0100, 0xE01EF),  # Variation Selectors Supplement
)


# Conservative thresholds: a 20-char base64 run with entropy ≥ 4.5 bits/char
# is well-separated from any natural English substring of the same length.
# Hex max entropy is log2(16) = 4.0, so we use a lower threshold.
BASE64_ALNUM_MIN_RUN: Final[int] = 20
BASE64_ENTROPY_THRESHOLD: Final[float] = 4.5
HEX_ALNUM_MIN_RUN: Final[int] = 16
HEX_ENTROPY_THRESHOLD: Final[float] = 3.5


# Common prompt-injection vocabulary, lowercased. The ROT13-encoded forms are
# what we search for in text; the originals are kept here so a reader can
# audit the wordlist directly.
ROT13_PI_MARKERS: Final[tuple[str, ...]] = (
    "ignore",
    "previous",
    "instructions",
    "instruction",
    "system",
    "prompt",
    "assistant",
    "command",
    "commands",
    "override",
    "disregard",
    "forget",
    "rules",
    "policy",
)


# Common leet substitutions. We deliberately omit ambiguous singletons like
# "1" → "l" / "i" (overlap with version numbers) and rely on the ≥2-sub
# threshold in is_leeted_token() to filter false positives like "h0w".
LEET_SUBSTITUTIONS: Final[dict[str, str]] = {
    "4": "a",
    "@": "a",
    "3": "e",
    "1": "i",
    "!": "i",
    "0": "o",
    "5": "s",
    "$": "s",
    "7": "t",
    "+": "t",
    "9": "g",
    "6": "g",
    "8": "b",
    "2": "z",
}


# Pre-computed ROT13 forms of ROT13_PI_MARKERS; we search for these substrings
# in lowercased text. codecs.encode with "rot_13" is the stdlib-canonical way.
_ROT13_ENCODED_MARKERS: Final[tuple[str, ...]] = tuple(
    codecs.encode(m, "rot_13") for m in ROT13_PI_MARKERS
)


# Regex backbones: BASE64 alphabet (RFC 4648) and lower/uppercase hex. Both
# include digits so a pure-hex match is also a base64 match — we test the
# hex threshold first (looser) so a 16-char hex string is flagged even when
# the base64 entropy bar (4.5) would reject it.
_BASE64_RUN_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9+/=]{" + str(BASE64_ALNUM_MIN_RUN) + r",}"
)
_HEX_RUN_RE: Final[re.Pattern[str]] = re.compile(r"[0-9A-Fa-f]{" + str(HEX_ALNUM_MIN_RUN) + r",}")


# Token boundary for the leetspeak scan: alphanumerics plus the two leet
# symbols that aren't already digits (@ and $) and the two ambiguous
# punctuation-style ones (! and +). Anything else (whitespace / hyphens /
# typical sentence punctuation) terminates the token.
_LEET_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9@$!+]+")


def count_invisible_chars(text: str) -> int:
    """Count invisible (zero-width / variation-selector / tag) chars.

    Parameters
    ----------
    text : str
        Input string. Empty string returns ``0``.

    Returns
    -------
    int
        Number of characters whose codepoint falls in
        :data:`INVISIBLE_RANGES` .

    Examples
    --------
    >>> count_invisible_chars("hello")
    0
    >>> count_invisible_chars("he​llo")  # ZWSP
    1
    >>> count_invisible_chars("a\U000e0041\U000e0042b")  # tag-block 'A' + 'B'
    2
    >>> count_invisible_chars("emoji‍joiner")  # ZWJ counts even in emoji
    1
    """
    n = 0
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in INVISIBLE_RANGES):
            n += 1
    return n


def nfkc_changed(text: str) -> bool:
    """Return ``True`` iff NFKC normalization changes ``text`` in any way.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    bool

    Examples
    --------
    >>> nfkc_changed("hello")
    False
    >>> nfkc_changed("ﬁle")  # 'fi' ligature U+FB01
    True
    >>> nfkc_changed("\U0001d408gnore")  # mathematical bold capital I
    True
    """
    return unicodedata.normalize("NFKC", text) != text


def nfkc_char_delta(text: str) -> int:
    """Absolute char-count delta between ``text`` and its NFKC normalization.

    A non-zero delta signals decomposition (e.g. the ﬁ ligature becoming
    two characters); a zero delta does **not** prove the text is canonical
    — same-length codepoint substitutions (mathematical-bold Latin → ASCII
    Latin) also change the text. Use :func:`nfkc_changed` for that signal.

    Parameters
    ----------
    text : str

    Returns
    -------
    int
        ``abs(len(text) - len(NFKC(text)))``.

    Examples
    --------
    >>> nfkc_char_delta("hello")
    0
    >>> nfkc_char_delta("ﬁle")  # ﬁ -> fi adds one character
    1
    >>> nfkc_char_delta("\U0001d408gnore")  # mathematical bold I -> I, same length
    0
    """
    return abs(len(text) - len(unicodedata.normalize("NFKC", text)))


def shannon_entropy(s: str) -> float:
    """Shannon entropy of ``s`` measured in bits/character.

    Parameters
    ----------
    s : str
        Input string. Empty string returns ``0.0``.

    Returns
    -------
    float
        Entropy in bits, ``0.0`` for ``s == ""``.

    Examples
    --------
    >>> shannon_entropy("")
    0.0
    >>> round(shannon_entropy("aaaa"), 4)
    0.0
    >>> round(shannon_entropy("abcd"), 4)
    2.0
    """
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    n = len(s)
    # H = -Σ p_i log2 p_i ; p_i = counts[i] / n
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def has_high_entropy_alnum_run(text: str) -> bool:
    """Detect a base64-shaped or hex-shaped high-entropy run in ``text``.

    Hex runs are tested first against the lower
    :data:`HEX_ENTROPY_THRESHOLD` so a 16-char hex string is flagged even
    when the broader :data:`BASE64_ENTROPY_THRESHOLD` bar would reject it.

    Parameters
    ----------
    text : str

    Returns
    -------
    bool

    Examples
    --------
    >>> has_high_entropy_alnum_run("just plain english text here")
    False
    >>> # 32 distinct base64-alphabet chars (entropy = log2(32) = 5.0):
    >>> has_high_entropy_alnum_run("payload AbCdEfGh12IjKlMn34OpQrSt56UvWxYz end")
    True
    >>> # 16 distinct hex chars (entropy = log2(16) = 4.0 > 3.5 threshold):
    >>> has_high_entropy_alnum_run("hash 0123456789abcdef tail")
    True
    >>> # hand-crafted "memorable" hex has low entropy and is correctly skipped:
    >>> has_high_entropy_alnum_run("debug deadbeefcafef00d here")
    False
    """
    if any(
        shannon_entropy(match.group()) >= HEX_ENTROPY_THRESHOLD
        for match in _HEX_RUN_RE.finditer(text)
    ):
        return True
    return any(
        shannon_entropy(match.group()) >= BASE64_ENTROPY_THRESHOLD
        for match in _BASE64_RUN_RE.finditer(text)
    )


def has_rot13_marker(text: str) -> bool:
    """Return ``True`` if any ROT13-encoded PI marker appears in ``text``.

    Searches for the ROT13-encoded forms of :data:`ROT13_PI_MARKERS` as
    substrings in ``text.lower()``. The wordlist is intentionally short —
    this is a screening heuristic for *encoded payload presence*, not a
    full ROT13 corpus classifier.

    Parameters
    ----------
    text : str

    Returns
    -------
    bool

    Examples
    --------
    >>> has_rot13_marker("normal english text")
    False
    >>> # rot13("ignore the previous instructions"):
    >>> has_rot13_marker("vtaber gur cerivbhf vafgehpgvbaf")
    True
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _ROT13_ENCODED_MARKERS)


def is_leeted_token(token: str) -> bool:
    """Heuristic: does ``token`` look like leetspeak?

    Requires length ∈ ``[3, 12]``, ≥ 1 alphabetic character, and ≥ 2
    leet-substitute characters (digits or ``@$!+``). The ≥ 2 floor filters
    incidental noise like ``v1``, ``py3``, ``h0w``, and isolated
    phone-number digits; the length-cap filters long hex-shaped identifiers
    (16-char hashes, ``deadbeef`` -style debug constants, SHA-1/-256
    digests) that otherwise satisfy the letter / leet thresholds.

    Parameters
    ----------
    token : str

    Returns
    -------
    bool

    Examples
    --------
    >>> is_leeted_token("hello")
    False
    >>> is_leeted_token("h3ll0")
    True
    >>> is_leeted_token("1gn0r3")  # 'ignore' leeted, 3 subs
    True
    >>> is_leeted_token("v1")
    False
    >>> is_leeted_token("h0w")  # one sub only -> rejected
    False
    >>> is_leeted_token("0123456789abcdef")  # 16-char hex too long
    False
    """
    if len(token) < 3 or len(token) > 12:
        return False
    n_letters = 0
    n_leet = 0
    for c in token:
        if c.isalpha():
            n_letters += 1
        if c in LEET_SUBSTITUTIONS:
            n_leet += 1
    return n_letters >= 1 and n_leet >= 2


def leetspeak_counts(text: str) -> tuple[int, int]:
    """Count leetspeak tokens and total alnum-shaped tokens in ``text``.

    Parameters
    ----------
    text : str

    Returns
    -------
    tuple of (int, int)
        ``(n_leeted_tokens, n_total_alnum_tokens)``. Both ``0`` when the
        text has no matching tokens.

    Examples
    --------
    >>> leetspeak_counts("normal text")
    (0, 2)
    >>> leetspeak_counts("1gn0r3 pr3v10us h3ll0")
    (3, 3)
    >>> leetspeak_counts("v1 plus py3 are plain version tags")
    (0, 7)
    """
    tokens = _LEET_TOKEN_RE.findall(text)
    if not tokens:
        return 0, 0
    n_leet = sum(1 for t in tokens if is_leeted_token(t))
    return n_leet, len(tokens)


@dataclass(frozen=True, slots=True)
class ObfuscationProfile:
    """Corpus-level obfuscation prevalence statistics.

    Built by :func:`analyze_obfuscation` over a sequence of texts. All
    count fields are non-negative; all rate fields are in ``[0.0, 1.0]``.
    An empty input yields all zeros (no division-by-zero, no fabricated
    rates).

    Parameters
    ----------
    n_texts : int
        Number of input texts.
    n_with_invisible_chars : int
        Texts containing ≥1 invisible/zero-width/tag-block character.
    total_invisible_chars : int
        Sum of invisible-char counts across all texts.
    invisible_char_rate : float
        ``n_with_invisible_chars / n_texts``; ``0.0`` for empty corpus.
    n_with_nfkc_change : int
        Texts where NFKC normalization changes the string in any way
        (codepoint substitution counts even when length is preserved).
    n_with_nfkc_length_delta : int
        Texts where NFKC normalization changes the *character count*
        (ligature decomposition, fullwidth → halfwidth, etc.).
    total_nfkc_char_delta : int
        Sum of absolute length deltas across all texts.
    nfkc_change_rate : float
        ``n_with_nfkc_change / n_texts``.
    n_with_high_entropy_run : int
        Texts containing a base64- or hex-shaped high-entropy run per
        :func:`has_high_entropy_alnum_run`.
    high_entropy_run_rate : float
        ``n_with_high_entropy_run / n_texts``.
    n_with_rot13_marker : int
        Texts where a ROT13-encoded PI marker appears.
    rot13_marker_rate : float
        ``n_with_rot13_marker / n_texts``.
    n_leeted_tokens : int
        Sum of leet-shaped tokens across the whole corpus.
    n_total_alnum_tokens : int
        Sum of alnum-shaped tokens across the whole corpus (denominator
        for ``leetspeak_token_rate``).
    leetspeak_token_rate : float
        ``n_leeted_tokens / n_total_alnum_tokens``; ``0.0`` when there are
        no alnum tokens.
    """

    n_texts: int
    n_with_invisible_chars: int
    total_invisible_chars: int
    invisible_char_rate: float
    n_with_nfkc_change: int
    n_with_nfkc_length_delta: int
    total_nfkc_char_delta: int
    nfkc_change_rate: float
    n_with_high_entropy_run: int
    high_entropy_run_rate: float
    n_with_rot13_marker: int
    rot13_marker_rate: float
    n_leeted_tokens: int
    n_total_alnum_tokens: int
    leetspeak_token_rate: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the profile.

        Returns
        -------
        dict
            All fields with their concrete types preserved (ints stay ints,
            floats stay floats).
        """
        return {
            "n_texts": self.n_texts,
            "n_with_invisible_chars": self.n_with_invisible_chars,
            "total_invisible_chars": self.total_invisible_chars,
            "invisible_char_rate": self.invisible_char_rate,
            "n_with_nfkc_change": self.n_with_nfkc_change,
            "n_with_nfkc_length_delta": self.n_with_nfkc_length_delta,
            "total_nfkc_char_delta": self.total_nfkc_char_delta,
            "nfkc_change_rate": self.nfkc_change_rate,
            "n_with_high_entropy_run": self.n_with_high_entropy_run,
            "high_entropy_run_rate": self.high_entropy_run_rate,
            "n_with_rot13_marker": self.n_with_rot13_marker,
            "rot13_marker_rate": self.rot13_marker_rate,
            "n_leeted_tokens": self.n_leeted_tokens,
            "n_total_alnum_tokens": self.n_total_alnum_tokens,
            "leetspeak_token_rate": self.leetspeak_token_rate,
        }


def analyze_obfuscation(texts: Sequence[str]) -> ObfuscationProfile:
    """Compute corpus-level obfuscation prevalence across ``texts``.

    Walks the input once per detector family — five passes total — and
    returns aggregated counts + rates. Pure (no IO, no logging beyond what
    callers configure).

    Parameters
    ----------
    texts : sequence of str
        The text samples to scan. An empty sequence yields an all-zero
        :class:`ObfuscationProfile`.

    Returns
    -------
    ObfuscationProfile

    Examples
    --------
    >>> p = analyze_obfuscation([])
    >>> p.n_texts, p.invisible_char_rate, p.leetspeak_token_rate
    (0, 0.0, 0.0)
    >>> p = analyze_obfuscation([
    ...     "normal text",
    ...     "he​llo",            # 1 invisible char
    ...     "ﬁle",                # NFKC delta (ligature)
    ...     "1gn0r3 pr3v10us all",    # leetspeak
    ... ])
    >>> p.n_texts
    4
    >>> p.n_with_invisible_chars
    1
    >>> p.n_with_nfkc_change
    1
    >>> p.n_leeted_tokens
    2
    """
    n_texts = len(texts)
    if n_texts == 0:
        return ObfuscationProfile(
            n_texts=0,
            n_with_invisible_chars=0,
            total_invisible_chars=0,
            invisible_char_rate=0.0,
            n_with_nfkc_change=0,
            n_with_nfkc_length_delta=0,
            total_nfkc_char_delta=0,
            nfkc_change_rate=0.0,
            n_with_high_entropy_run=0,
            high_entropy_run_rate=0.0,
            n_with_rot13_marker=0,
            rot13_marker_rate=0.0,
            n_leeted_tokens=0,
            n_total_alnum_tokens=0,
            leetspeak_token_rate=0.0,
        )

    n_with_invisible = 0
    total_invisible = 0
    n_with_nfkc_change = 0
    n_with_nfkc_length_delta = 0
    total_nfkc_delta = 0
    n_with_entropy_run = 0
    n_with_rot13 = 0
    n_leeted = 0
    n_total_alnum = 0

    for text in texts:
        n_invis = count_invisible_chars(text)
        if n_invis:
            n_with_invisible += 1
            total_invisible += n_invis
        if nfkc_changed(text):
            n_with_nfkc_change += 1
        char_delta = nfkc_char_delta(text)
        if char_delta:
            n_with_nfkc_length_delta += 1
            total_nfkc_delta += char_delta
        if has_high_entropy_alnum_run(text):
            n_with_entropy_run += 1
        if has_rot13_marker(text):
            n_with_rot13 += 1
        n_leet, n_tokens = leetspeak_counts(text)
        n_leeted += n_leet
        n_total_alnum += n_tokens

    leet_rate = (n_leeted / n_total_alnum) if n_total_alnum else 0.0
    return ObfuscationProfile(
        n_texts=n_texts,
        n_with_invisible_chars=n_with_invisible,
        total_invisible_chars=total_invisible,
        invisible_char_rate=n_with_invisible / n_texts,
        n_with_nfkc_change=n_with_nfkc_change,
        n_with_nfkc_length_delta=n_with_nfkc_length_delta,
        total_nfkc_char_delta=total_nfkc_delta,
        nfkc_change_rate=n_with_nfkc_change / n_texts,
        n_with_high_entropy_run=n_with_entropy_run,
        high_entropy_run_rate=n_with_entropy_run / n_texts,
        n_with_rot13_marker=n_with_rot13,
        rot13_marker_rate=n_with_rot13 / n_texts,
        n_leeted_tokens=n_leeted,
        n_total_alnum_tokens=n_total_alnum,
        leetspeak_token_rate=leet_rate,
    )
