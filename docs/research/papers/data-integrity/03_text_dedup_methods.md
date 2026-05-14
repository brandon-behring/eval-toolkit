# Text Deduplication Methods — Synthesis

This file synthesizes C1 (algorithms for detecting near-duplicate text). Companion raw-table dossier: `_dossier/03_text_dedup_methods.md`. The downstream effects of pre-training corpus dedup live in `04_pretrain_dedup_effects.md`. The eval-toolkit's `text_dedup` module uses TF-IDF cosine (default), exact-hash, embedding-cosine, Jaccard-ngram, and MinHash-LSH variants.

---

## C1. Near-duplicate detection algorithms

- **On the resemblance and containment of documents** — Broder (Compression and Complexity of Sequences 1997).
  - **Source:** https://ieeexplore.ieee.org/document/666900
  - **Code:** —
  - **Mechanism:** Defines document resemblance via Jaccard set similarity on shingles (k-grams); samples Jaccard using minimum-wise hashing.
  - **Result:** Foundational paper for MinHash — a constant-size fingerprint for sets that estimates Jaccard similarity with O(1) per-pair compute. Basis for the MinHash-LSH dedup variant in the eval-toolkit.
  - **Status:** Verified (no widely-known repo).

- **Approximate nearest neighbors: towards removing the curse of dimensionality** — Indyk & Motwani (STOC 1998).
  - **Source:** https://dl.acm.org/doi/10.1145/276698.276876
  - **Code:** —
  - **Mechanism:** Locality-sensitive hashing (LSH) — hash families where similar inputs collide with high probability — enabling sublinear-time approximate-nearest-neighbor search.
  - **Result:** Foundational paper for LSH; combined with MinHash gives MinHash-LSH, the standard scalable algorithm for finding near-duplicates in massive corpora. Underlies `text_dedup.minhash_lsh` in the eval-toolkit.
  - **Status:** Verified (no widely-known repo).

- **SemDeDup: Data-efficient learning at web-scale through semantic deduplication** — Abbas et al. (arXiv 2023).
  - **Source:** https://arxiv.org/abs/2303.09540
  - **Code:** https://github.com/facebookresearch/SemDeDup
  - **Mechanism:** Clusters pretrained-embedding-similar examples; removes one from each near-duplicate pair based on cosine similarity in embedding space.
  - **Result:** Demonstrates that on LAION-style web data, removing ~50% of examples by semantic duplication preserves downstream performance with minimal loss, effectively halving training time. Primary reference for semantic / embedding-cosine dedup in the eval-toolkit.
  - **Status:** Verified.
