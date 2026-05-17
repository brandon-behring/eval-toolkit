# `eval_toolkit.embeddings`

Factories for the canonical embedder recipes used with
{class}`~eval_toolkit.text_dedup.EmbeddingCosineStrategy`. The toolkit owns
cosine + k-NN; this module wraps the popular embedder backends so callers
don't reinvent the same model-loading + numpy-coercion glue each time.

Install the optional dependency: `pip install eval-toolkit[embeddings]`.
This pulls `sentence-transformers` (which transitively brings `torch` +
`tokenizers`). It is intentionally **not** in `[all]` to keep contributor
setup small.

```{eval-rst}
.. currentmodule:: eval_toolkit.embeddings

.. autosummary::
   :toctree: generated/embeddings/
   :nosignatures:

   make_minilm_embedder
```
