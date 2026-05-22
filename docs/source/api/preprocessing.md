# `eval_toolkit.preprocessing`

Structural-defense preprocessing — the 3 Spotlighting transforms from
Hines et al. 2024 ([arXiv 2403.14720](https://arxiv.org/abs/2403.14720))
exposed as both a functional API (`delimit` / `datamark` / `encode`) and
v0.47 dataclass wrappers (`DelimitVariant`, `DatamarkVariant`,
`EncodeVariant`) that satisfy the {class}`~eval_toolkit.TextTransform`
Protocol for {func}`eval_toolkit.sweep`.

```{eval-rst}
.. currentmodule:: eval_toolkit.preprocessing

.. autosummary::
   :toctree: generated/preprocessing/
   :nosignatures:

   DatamarkVariant
   DelimitVariant
   EncodeVariant
   datamark
   delimit
   encode
```
