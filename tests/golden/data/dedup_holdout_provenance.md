# Dedup-holdout fixture provenance

## Source

This 50-pair adversarial dedup-holdout fixture was migrated from
`brandon-behring/prompt-injection-detection-submission` at git SHA
**`4c1407d4300e770138c51b2c642f9b25e2304578`** (commit dated
2026-05-17 13:06:22 -0400). See
[`scripts/refresh_dedup_holdout.py`](../../../scripts/refresh_dedup_holdout.py)
for the migration helper that captures + records the consumer SHA on
each refresh.

## File structure

JSONL with **1 metadata header** + **50 pair records**:

- **Line 1** (`_metadata: true`): generation provenance — UTC timestamp,
  seed, encoder model + revision, ADR reference, LLM pre-label model +
  token counts.
- **Lines 2–51** (pair records): each carries
  - `source_a`, `source_b` — corpus origin (one of the 4 sources below)
  - `text_a`, `text_b` — raw prompts
  - `cosine` — sentence-transformer cosine similarity at generation time
  - `band` — stratified cosine band (e.g., `[0.55, 0.65]`)
  - `pair_id` — stable identifier
  - `true_duplicate` — boolean ground-truth label
  - `llm_judge_label`, `llm_judge_reasoning`, `llm_judge_model` — LLM
    pre-label (gpt-4o-2024-08-06)
  - `human_label` — optional hand-review override (mostly null per ADR-042)

## Source datasets

All 4 source corpora are permissively-licensed prompt-injection eval datasets:

| Source key | Upstream | License | URL |
|---|---|---|---|
| `deepset_prompt_injections` | `deepset/prompt-injections` (HF) | CC-BY 4.0 | <https://huggingface.co/datasets/deepset/prompt-injections> |
| `lakera_gandalf_ignore_instructions` | `Lakera/gandalf_ignore_instructions` (HF) | MIT | <https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions> |
| `lakera_mosscap_prompt_injection` | `Lakera/mosscap_prompt_injection` (HF) | MIT | <https://huggingface.co/datasets/Lakera/mosscap_prompt_injection> |
| `hackaprompt` | `hackaprompt/hackaprompt-dataset` (HF) | MIT | <https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset> |

Re-verify licenses at refresh time: each refresh should confirm the
upstream datasets still carry the same license; bump this table if not.

## Stratification

The 50 pairs are **5 cosine bands × 10 pairs**:

- `[0.65, 0.70)` (10 pairs)
- `[0.70, 0.75)` (10 pairs)
- `[0.75, 0.80)` (10 pairs)
- `[0.80, 0.85)` (10 pairs)
- `[0.85, 0.95]` (10 pairs)

This design (per consumer ADR-041 Q5 + ADR-042) ensures the eval covers
the boundary region where dedup strategies disagree, not just trivially
identical or trivially different pairs.

## Labeling

Pre-labeled by `gpt-4o-2024-08-06` via the consumer's LLM-judge prompt
(captured in `llm_judge_reasoning` per record); a hand-review override
in `human_label` is intentionally rare per ADR-042's pre-labeling-as-truth
methodology.

## Refresh flow

Run `python scripts/refresh_dedup_holdout.py` from the eval-toolkit repo
root. The script:
1. Reads the consumer repo path (defaults to
   `../prompt-injection-detection-submission/`).
2. Copies `data/dedup_holdout.jsonl` into `tests/golden/data/`
   (overwrites; bytes-identical check guard).
3. Captures the consumer git SHA + commit date into this file.
4. Regenerates `tests/golden/data/dedup_holdout_expected.json` (the
   3-deterministic-strategy snapshot) by running the test suite with
   `REGEN_DEDUP_HOLDOUT_GOLDEN=1`.

## Test usage

See `tests/golden/test_dedup_holdout_calibration.py`. Asserts FPR + FNR
at thresholds `{0.75, 0.80, 0.85}` for the 3 deterministic
`SimilarityStrategy` variants (`TfidfCosineStrategy`,
`ExactNormalizedHashStrategy`, `JaccardNgramStrategy`). The
`EmbeddingCosineStrategy` variant (via v0.33.1's `make_minilm_embedder`)
is gated with `pytest.importorskip("sentence_transformers")` +
`@pytest.mark.slow`; its assertion is a soft bound (`FPR < 0.5`,
`FNR < 0.5`) at threshold 0.80 since sentence-transformers model weights
can shift across upstream releases (snapshotting them would be brittle).
