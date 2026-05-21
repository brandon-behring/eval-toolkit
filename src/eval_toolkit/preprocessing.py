"""Structural-defense preprocessing — Spotlighting variants for prompt injection.

Implements the 3 Spotlighting transforms from Hines et al. 2024 ([1]_) for
defending LLMs against indirect prompt injection by *structurally marking*
untrusted input so the model can distinguish it from system instructions.

The three variants:

- :func:`delimit` — wrap text in unusual delimiters (default ``<<...>>``)
- :func:`datamark` — prepend a marker character before each whitespace token
  (default ``^``)
- :func:`encode` — encode the text (default ``base64``); the LLM is told to
  decode but treat the result as data, not instructions

The 3 ``Variant`` dataclasses (:class:`DelimitVariant`,
:class:`DatamarkVariant`, :class:`EncodeVariant`) added at v0.47 wrap the
functions in the :class:`~eval_toolkit.TextTransform` Protocol shape so
they compose with adversarial-side strategies via the top-level
:func:`eval_toolkit.sweep` entry point.

The v0.44–v0.46 module-level ``sweep()`` function and ``spotlighting``
``SimpleNamespace`` were removed at v0.47 (Decisions D + K + N; plan
§§4C/4E). The 12-technique top-level sweep covers the same surface
without API duplication.

All three transforms are deterministic, side-effect-free, and base-install
safe — only stdlib used.

References
----------
.. [1] Hines, K., et al. 2024. "Defending Against Indirect Prompt Injection
       Attacks With Spotlighting." arXiv:2403.14720.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DatamarkVariant",
    "DelimitVariant",
    "EncodeVariant",
    "datamark",
    "delimit",
    "encode",
]


# Default constants per the Hines et al. paper § 3
_DEFAULT_DELIMITER = "<<"
_DEFAULT_DELIMITER_END = ">>"
_DEFAULT_MARKER = "^"
_DEFAULT_ENCODING: Literal["base64"] = "base64"


def delimit(text: str, *, delimiter: str = _DEFAULT_DELIMITER, end: str | None = None) -> str:
    """Wrap ``text`` in unusual delimiters so the LLM can spot the boundary.

    Recoverable via simple slicing (caller knows the delimiter pair).
    Deterministic: same input + delimiter → same output.

    Parameters
    ----------
    text : str
        Input to wrap.
    delimiter : str, optional
        Opening delimiter. Default ``"<<"``. Choose something unlikely to
        appear in user-generated content.
    end : str or None, optional
        Closing delimiter. If ``None`` (default), the opening delimiter
        is reversed character-by-character (``"<<"`` → ``">>"``,
        ``"[["`` → ``"]]"``, ``"BEGIN"`` → ``"NIGEB"``). Pass an explicit
        ``end`` for asymmetric pairs.

    Returns
    -------
    str
        ``delimiter + text + end``.

    Examples
    --------
    >>> delimit("hello")
    '<<hello>>'
    >>> delimit("hello", delimiter="[", end="]")
    '[hello]'
    """
    if end is None:
        end = _mirror_delimiter(delimiter)
    return f"{delimiter}{text}{end}"


def _mirror_delimiter(d: str) -> str:
    """Return the mirrored closing form of an opening delimiter.

    Examples: ``"<<"`` → ``">>"``, ``"[["`` → ``"]]"``, ``"BEGIN"`` → ``"NIGEB"``.
    """
    mirror = {"<": ">", "[": "]", "(": ")", "{": "}"}
    return "".join(mirror.get(ch, ch) for ch in reversed(d))


def datamark(text: str, *, marker: str = _DEFAULT_MARKER) -> str:
    """Prepend ``marker`` before each non-leading whitespace run.

    The LLM sees a textual signal that every word boundary belongs to
    untrusted data. Recoverable by stripping the marker before each
    whitespace run.

    Parameters
    ----------
    text : str
        Input to datamark.
    marker : str, optional
        Character (or short string) to inject. Default ``"^"``.

    Returns
    -------
    str
        Text with ``marker`` inserted before every whitespace run.

    Examples
    --------
    >>> datamark("hello world")
    'hello^ world'
    >>> datamark("a b c", marker="*")
    'a* b* c'
    """
    if not text:
        return text
    # Insert marker before any internal whitespace run (one or more spaces /
    # tabs / newlines). Leading whitespace is preserved as-is.
    return re.sub(r"(\S)(\s+)", lambda m: m.group(1) + marker + m.group(2), text)


def encode(text: str, *, encoding: Literal["base64"] = _DEFAULT_ENCODING) -> str:
    """Encode ``text`` so the LLM treats the result as data, not instructions.

    Only ``base64`` is supported in v0.44.0 — the paper's default + the
    most LLM-friendly encoding (most foundation models can decode base64
    on demand).

    Recoverable via ``base64.b64decode``.

    Parameters
    ----------
    text : str
        Input to encode.
    encoding : {"base64"}, optional
        Encoding scheme. Default ``"base64"``.

    Returns
    -------
    str
        Encoded text as an ASCII string.

    Raises
    ------
    ValueError
        On unknown ``encoding``.

    Examples
    --------
    >>> encode("hello")
    'aGVsbG8='
    """
    if encoding == "base64":
        return base64.b64encode(text.encode("utf-8")).decode("ascii")
    raise ValueError(f"encode: unsupported encoding {encoding!r}; supported: 'base64'")


# ─────────────────────────────────────────────────────────────────────────────
# Spotlighting variants as ``TextTransform``-shaped dataclasses (v0.47)
#
# Decision K + Audit R5-F3: the functional API (``delimit`` / ``datamark``
# / ``encode``) remains the implementation; these frozen dataclasses are
# thin ``TextTransform`` wrappers so the top-level :func:`sweep` and
# downstream code can treat them uniformly with adversarial-side
# strategies (``ZeroWidthSpaceInjection`` etc.) via structural subtyping.
#
# All three satisfy the top-level :class:`eval_toolkit.TextTransform`
# Protocol: ``name: str`` attribute + ``transform(text: str) -> str``
# method. Frozen + ``slots=True`` per house style.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DelimitVariant:
    """:class:`TextTransform` wrapper around :func:`delimit`.

    See the underlying function for semantics; this dataclass exists so
    callers can pass a uniform-shape strategy object into
    :func:`eval_toolkit.sweep` alongside adversarial-side strategies.
    """

    name: str = "delimit"
    delimiter: str = _DEFAULT_DELIMITER
    end: str | None = None

    def transform(self, text: str) -> str:
        """Delegate to :func:`delimit` with the dataclass's configured kwargs."""
        return delimit(text, delimiter=self.delimiter, end=self.end)


@dataclass(frozen=True, slots=True)
class DatamarkVariant:
    """:class:`TextTransform` wrapper around :func:`datamark`."""

    name: str = "datamark"
    marker: str = _DEFAULT_MARKER

    def transform(self, text: str) -> str:
        """Delegate to :func:`datamark` with the dataclass's configured kwargs."""
        return datamark(text, marker=self.marker)


@dataclass(frozen=True, slots=True)
class EncodeVariant:
    """:class:`TextTransform` wrapper around :func:`encode`."""

    name: str = "encode"
    encoding: Literal["base64"] = _DEFAULT_ENCODING

    def transform(self, text: str) -> str:
        """Delegate to :func:`encode` with the dataclass's configured kwargs."""
        return encode(text, encoding=self.encoding)


# Module-level ``sweep`` removed at v0.47 — consolidated into the top-level
# :func:`eval_toolkit.sweep` which accepts any :class:`TextTransform`
# (Decisions K + D + plan §4C). Module-level ``spotlighting``
# :class:`SimpleNamespace` removed at v0.47 (Decision N + plan §4E) — the
# 3 ``Variant`` dataclasses + the underlying functional ``delimit`` /
# ``datamark`` / ``encode`` functions are now the only public API.
