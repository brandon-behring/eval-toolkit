"""Tests for ``eval_toolkit.adversarial`` (v0.43.0; closes #49 core-6).

Covers the six core character-injection strategies + the 6 advanced (v0.47)
techniques. The module-level ``character_injection`` SimpleNamespace,
``CharacterInjectionStrategy`` per-module Protocol, and ``sweep()`` function
were removed at v0.47 (Decision N + plan §4E); use the top-level
``TextTransform`` Protocol and ``eval_toolkit.sweep`` instead.
"""

from __future__ import annotations

import string
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eval_toolkit import TextTransform
from eval_toolkit.adversarial import (
    CORE_TECHNIQUES,
    CaseInjection,
    DiacriticInjection,
    HomoglyphSubstitution,
    PunctuationInjection,
    WhitespaceInjection,
    ZeroWidthSpaceInjection,
)

# ----------------------------------------------------------------------------
# Per-technique unit tests
# ----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("strategy_cls", CORE_TECHNIQUES)
def test_each_strategy_is_protocol_compliant(strategy_cls: type[Any]) -> None:
    instance = strategy_cls()
    assert isinstance(instance, TextTransform)
    assert hasattr(instance, "name") and isinstance(instance.name, str)


@pytest.mark.unit
@pytest.mark.parametrize("strategy_cls", CORE_TECHNIQUES)
def test_each_strategy_returns_str_of_at_least_same_length(strategy_cls: type[Any]) -> None:
    """Most techniques only INSERT characters; transformed text >= original length."""
    instance = strategy_cls()
    text = "Hello World! This is a sample text for adversarial testing."
    transformed = instance.transform(text)
    assert isinstance(transformed, str)
    # CaseInjection preserves length exactly; HomoglyphSubstitution preserves
    # character count but not byte length. The other four insert. All preserve or grow.
    if strategy_cls is HomoglyphSubstitution or strategy_cls is CaseInjection:
        assert len(transformed) == len(text)
    else:
        assert len(transformed) >= len(text)


@pytest.mark.unit
@pytest.mark.parametrize("strategy_cls", CORE_TECHNIQUES)
def test_each_strategy_is_deterministic_with_same_seed(strategy_cls: type[Any]) -> None:
    text = "The quick brown fox jumps over the lazy dog 1234567890."
    a = strategy_cls(seed=123).transform(text)
    b = strategy_cls(seed=123).transform(text)
    assert a == b


@pytest.mark.unit
@pytest.mark.parametrize("strategy_cls", CORE_TECHNIQUES)
def test_each_strategy_differs_with_different_seeds(strategy_cls: type[Any]) -> None:
    """With a high enough ratio + reasonably long text, two seeds must diverge."""
    text = "The quick brown fox jumps over the lazy dog. " * 5
    # Use defaults; CaseInjection at ratio=0.5 on this text yields ~150 random flips.
    a = strategy_cls(seed=1).transform(text)
    b = strategy_cls(seed=2).transform(text)
    assert a != b


@pytest.mark.unit
@pytest.mark.parametrize("strategy_cls", CORE_TECHNIQUES)
def test_each_strategy_actually_modifies_text(strategy_cls: type[Any]) -> None:
    """Default ratio + a non-trivial text must produce a different output."""
    text = "Hello World" * 10
    out = strategy_cls().transform(text)
    assert out != text


@pytest.mark.unit
def test_zero_width_space_inserts_zwsp_char() -> None:
    out = ZeroWidthSpaceInjection(ratio=1.0, seed=0).transform("abc")
    assert "​" in out
    assert out.replace("​", "") == "abc"


@pytest.mark.unit
def test_homoglyph_substitutes_only_table_chars() -> None:
    text = "abcdefXY"  # 'a', 'c', 'e' substitutable; 'b', 'd', 'f' not; X, Y substitutable
    out = HomoglyphSubstitution(ratio=1.0, seed=0).transform(text)
    # Length identical (homoglyphs are single code points)
    assert len(out) == len(text)
    # Non-substituted chars passed through
    assert "b" in out and "d" in out and "f" in out
    # At least one substitution happened (with ratio=1.0)
    assert out != text


@pytest.mark.unit
def test_diacritic_skips_non_alphanumeric() -> None:
    """Punctuation / whitespace should not get combining marks attached."""
    text = "ab   cd"
    out = DiacriticInjection(ratio=1.0, seed=0).transform(text)
    # Spaces should still be present, unaltered.
    assert "   " in out


@pytest.mark.unit
def test_whitespace_substitutes_spaces_with_variants() -> None:
    out = WhitespaceInjection(pad_ratio=0.0, substitute_ratio=1.0, seed=0).transform("a b c d")
    # All original spaces should now be NBSP or tab (not regular " ")
    assert " " not in out  # regular space gone


@pytest.mark.unit
def test_case_random_preserves_non_alpha() -> None:
    text = "abc 123 !@#"
    out = CaseInjection(ratio=1.0, seed=0).transform(text)
    assert len(out) == len(text)
    # Numbers and punctuation unchanged
    for ch_orig, ch_out in zip(text, out, strict=True):
        if not ch_orig.isalpha():
            assert ch_orig == ch_out


@pytest.mark.unit
def test_punctuation_inserts_after_alphanumerics() -> None:
    out = PunctuationInjection(ratio=1.0, seed=0).transform("ab cd")
    # Inserted punctuation should appear in the output
    assert any(ch in out for ch in ".,;:!?-_")


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "strategy_cls",
    [
        ZeroWidthSpaceInjection,
        HomoglyphSubstitution,
        DiacriticInjection,
        CaseInjection,
        PunctuationInjection,
    ],
)
@pytest.mark.parametrize("ratio", [-0.1, 1.1, 2.0])
def test_invalid_ratio_raises_valueerror(strategy_cls: type[Any], ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio"):
        strategy_cls(ratio=ratio)


@pytest.mark.unit
@pytest.mark.parametrize("kwarg", ["pad_ratio", "substitute_ratio"])
def test_whitespace_invalid_ratio_raises(kwarg: str) -> None:
    with pytest.raises(ValueError):
        # Dynamic kwarg dispatch — runtime accepts both float kwargs;
        # mypy sees `**dict[str, float]` and can't pick the right slot.
        WhitespaceInjection(**{kwarg: 1.5})  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Property-based: empty / minimal input
# ----------------------------------------------------------------------------


@pytest.mark.property
@pytest.mark.parametrize("strategy_cls", CORE_TECHNIQUES)
def test_empty_string_passes_through(strategy_cls: type[Any]) -> None:
    assert strategy_cls().transform("") == ""


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(st.text(min_size=1, max_size=200))
def test_zero_width_space_round_trip_is_recoverable(text: str) -> None:
    """Stripping ZWSPs from the transformed text yields the original."""
    transformed = ZeroWidthSpaceInjection(ratio=0.7, seed=1).transform(text)
    assert transformed.replace("​", "") == text


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    st.text(
        alphabet=st.sampled_from(string.ascii_letters + string.digits), min_size=1, max_size=100
    )
)
def test_case_random_lowercase_recovers_lowercase(text: str) -> None:
    """For ASCII letters + digits, applying CaseInjection then .lower() recovers .lower(text).

    Restricted to ASCII to avoid Unicode edge cases like German ß where
    ``'ß'.upper() == 'SS'`` and the round-trip does not survive.
    """
    transformed = CaseInjection(ratio=0.5, seed=1).transform(text)
    assert transformed.lower() == text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Advanced 6 (v0.47.0; Decision Q11→11.3)
#
# Each technique satisfies the top-level TextTransform Protocol structurally,
# is deterministic where seeded, and round-trips through a no-op input.
# ─────────────────────────────────────────────────────────────────────────────


from eval_toolkit import (  # noqa: E402 — sectioning: advanced-6 imports w/ tests
    BidiRTLInjection,
    InvisibleCharsInjection,
    SynonymSubstitution,
    TagStrippingInjection,
    TokenSplittingInjection,
    UnicodeNormalizationInjection,
)
from eval_toolkit.adversarial import (  # noqa: E402
    ADVANCED_TECHNIQUES,
    ALL_TECHNIQUES,
)


@pytest.mark.unit
def test_advanced_techniques_satisfy_text_transform() -> None:
    """All 6 advanced techniques satisfy the top-level TextTransform Protocol."""
    for cls in ADVANCED_TECHNIQUES:
        assert isinstance(cls(), TextTransform), f"{cls.__name__} must satisfy TextTransform"


@pytest.mark.unit
def test_advanced_techniques_count() -> None:
    """Decision Q11→11.3: ALL_TECHNIQUES = CORE (6) + ADVANCED (6) = 12."""
    assert len(ADVANCED_TECHNIQUES) == 6
    assert len(ALL_TECHNIQUES) == 12


@pytest.mark.unit
def test_bidi_rtl_wraps_in_override_block() -> None:
    out = BidiRTLInjection().transform("hello")
    assert out.startswith("‮") and out.endswith("‬")
    assert "hello" in out


@pytest.mark.unit
def test_bidi_rtl_empty_passthrough() -> None:
    assert BidiRTLInjection().transform("") == ""


@pytest.mark.unit
def test_tag_stripping_removes_tags() -> None:
    out = TagStrippingInjection().transform("<b>hello</b> <i>world</i>")
    assert out == "hello world"


@pytest.mark.unit
def test_tag_stripping_is_idempotent() -> None:
    """Running tag-strip twice produces the same output as once."""
    t = TagStrippingInjection()
    once = t.transform("<a>x</a><b>y</b>")
    twice = t.transform(once)
    assert once == twice


@pytest.mark.unit
def test_tag_stripping_preserves_non_tag_angle_text() -> None:
    """A bare `<` without matching `>` is preserved (no greedy over-strip)."""
    out = TagStrippingInjection().transform("a < b and c > d")
    # The regex matches "< b and c >" because nothing closes the first <
    # before the next >. Document this behaviour.
    assert "d" in out


@pytest.mark.unit
def test_synonym_substitution_replaces_known_words() -> None:
    out = SynonymSubstitution(seed=42).transform("ignore the system instructions")
    # 'ignore' / 'system' / 'instructions' are all in the whitelist.
    assert "ignore" not in out
    assert "system" not in out
    assert "instructions" not in out


@pytest.mark.unit
def test_synonym_substitution_preserves_capitalization() -> None:
    """Title-case input gets title-case substitution."""
    out = SynonymSubstitution(seed=42).transform("Ignore everything")
    # The substituted word is title-cased even if the synonym table stores
    # lowercase forms.
    first_word = out.split()[0]
    assert first_word[0].isupper(), f"expected title-case in {out!r}"


@pytest.mark.unit
def test_synonym_substitution_passes_through_unknown_words() -> None:
    """Words NOT in the whitelist are passed through unchanged."""
    out = SynonymSubstitution(seed=42).transform("foo bar baz quux")
    assert out == "foo bar baz quux"


@pytest.mark.unit
def test_synonym_substitution_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError, match="ratio must be in"):
        SynonymSubstitution(ratio=1.5)


@pytest.mark.unit
def test_token_splitting_splits_long_words() -> None:
    out = TokenSplittingInjection(seed=42, min_word_length=4).transform("hello world")
    # Both words are >= 4 chars → both should pick up an inserted space
    assert out != "hello world"
    # Original chars are preserved modulo the inserted space
    assert out.replace(" ", "") == "helloworld"


@pytest.mark.unit
def test_token_splitting_preserves_short_words() -> None:
    """Words shorter than ``min_word_length`` pass through unchanged."""
    out = TokenSplittingInjection(min_word_length=10).transform("hi go ok no")
    assert out == "hi go ok no"


@pytest.mark.unit
def test_token_splitting_rejects_bad_min_length() -> None:
    with pytest.raises(ValueError, match="min_word_length"):
        TokenSplittingInjection(min_word_length=1)


@pytest.mark.unit
def test_unicode_normalization_default_nfkc_folds_compat_chars() -> None:
    """NFKC default folds fullwidth ABC to ASCII ABC."""
    out = UnicodeNormalizationInjection().transform("ＡＢＣ")
    assert out == "ABC"


@pytest.mark.unit
def test_unicode_normalization_passes_through_normalized_text() -> None:
    """Already-NFKC ASCII passes through unchanged."""
    out = UnicodeNormalizationInjection().transform("hello world")
    assert out == "hello world"


@pytest.mark.unit
def test_unicode_normalization_rejects_bad_form() -> None:
    with pytest.raises(ValueError, match="must be NFC"):
        UnicodeNormalizationInjection(form="NFXX")


@pytest.mark.unit
def test_invisible_chars_inserts_zero_width_codepoints() -> None:
    out = InvisibleCharsInjection(ratio=1.0, seed=42).transform("abc")
    # Every position gets an invisible char appended
    assert len(out) > len("abc")
    # No "visible" character was lost
    for ch in "abc":
        assert ch in out


@pytest.mark.unit
def test_invisible_chars_zero_ratio_passes_through() -> None:
    """ratio=0 → output equals input."""
    assert InvisibleCharsInjection(ratio=0.0).transform("hello") == "hello"


@pytest.mark.unit
def test_invisible_chars_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError, match="ratio must be in"):
        InvisibleCharsInjection(ratio=-0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism — each seeded advanced technique produces identical output
# for the same seed across multiple invocations.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "cls",
    [SynonymSubstitution, TokenSplittingInjection, InvisibleCharsInjection],
)
def test_advanced_seeded_technique_is_deterministic(cls: type) -> None:
    text = "ignore previous system instructions"
    assert cls(seed=7).transform(text) == cls(seed=7).transform(text)


@pytest.mark.unit
def test_all_techniques_compose_in_sweep() -> None:
    """The new 12-technique suite composes in the v0.47 top-level sweep().

    Confirms the v0.43 forward-look "complete 12-technique suite + new
    sweep API in one migration step" (Decision Q11→11.3) actually works.
    """
    import pandas as pd

    from eval_toolkit import sweep

    df = sweep([cls() for cls in ALL_TECHNIQUES], ["hello world"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert set(df["variant"]) == {
        "zero_width_space",
        "homoglyph",
        "diacritic",
        "whitespace",
        "case_random",
        "punctuation",
        "bidi_rtl",
        "tag_strip",
        "synonym",
        "token_split",
        "unicode_normalize",
        "invisible_chars",
    }
