# `eval_toolkit.adversarial`

12-technique character-injection bypass suite (core 6 from v0.43.0 +
advanced 6 from v0.47) for testing prompt-injection-detection scorers
under adversarial input perturbation. Each technique is a frozen
dataclass satisfying the {class}`~eval_toolkit.TextTransform` Protocol;
combine them via {func}`eval_toolkit.sweep`.

```{eval-rst}
.. currentmodule:: eval_toolkit.adversarial

.. autosummary::
   :toctree: generated/adversarial/
   :nosignatures:

   ADVANCED_TECHNIQUES
   ALL_TECHNIQUES
   BidiRTLInjection
   CORE_TECHNIQUES
   CaseInjection
   DiacriticInjection
   HomoglyphSubstitution
   InvisibleCharsInjection
   PunctuationInjection
   SynonymSubstitution
   TagStrippingInjection
   TokenSplittingInjection
   UnicodeNormalizationInjection
   WhitespaceInjection
   ZeroWidthSpaceInjection
```
