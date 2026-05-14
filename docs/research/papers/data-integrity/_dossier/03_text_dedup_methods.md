# Text Deduplication Methods

This file covers C1 (algorithms for detecting near-duplicate text — MinHash, LSH, semantic dedup). The downstream effects of pre-training corpus dedup are in `04_pretrain_dedup_effects.md`. The eval-toolkit's `text_dedup` module uses TF-IDF cosine (default), exact-hash, embedding-cosine, Jaccard-ngram, and MinHash-LSH variants.

---

## C1. Near-duplicate detection algorithms

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| On the resemblance and containment of documents | Broder (1997) | Compression and Complexity of Sequences (IEEE) | DOI:10.1109/SEQUEN.1997.666900 | — | Defines document resemblance via Jaccard set similarity on shingles; samples Jaccard using minimum-wise hashing (MinHash) | Foundational paper for MinHash — a constant-size fingerprint for sets that estimates Jaccard similarity with O(1) per-pair compute. Basis for the MinHash-LSH dedup variant in the eval-toolkit |
| Approximate nearest neighbors: towards removing the curse of dimensionality | Indyk & Motwani (1998) | STOC 1998 | DOI:10.1145/276698.276876 | — | Locality-sensitive hashing (LSH) for sublinear-time approximate nearest-neighbor search in high dimensions | Foundational paper for LSH — combined with MinHash gives MinHash-LSH, the standard scalable algorithm for finding near-duplicates in massive corpora |
| SemDeDup: Data-efficient learning at web-scale through semantic deduplication | Abbas et al. (2023) | arXiv preprint | arXiv:2303.09540 | facebookresearch/SemDeDup | Clusters pretrained-embedding similar examples and removes one from each near-duplicate pair | Demonstrates that on LAION-style web data, removing ~50% of examples by semantic duplication preserves downstream performance with minimal loss, effectively halving training time |
