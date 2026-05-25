# Migration guides

Per-version migration guides for breaking + behavior-changing
upgrades. Use these alongside [`CHANGELOG.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/CHANGELOG.md) — the
changelog records *what* changed; these docs walk through *how* to
update consumer code.

## Index

| From → To | Type | Guide |
|---|---|---|
| v0.6.x → v0.7.x | BREAKING | [`migration/v0.7.md`](migration/v0.7.md) — `select_threshold` API + new Protocol surfaces |
| v0.7.x → v0.8.0 | BREAKING (small) | [`migration/v0.8.md`](migration/v0.8.md) — ECE input validation enforced; `__version__` mismatch closed; `[parquet]` extra |
| v0.8.x → v0.9.0 | ADDITIVE | [`migration/v0.9.md`](migration/v0.9.md) — evidence core: `claims`, `artifacts`, `operating_points`, `evidence`, `analysis`, `protocols`; six new optional `RunResult` fields; `[validation]` extra |
| v0.45.x → v0.46.0 | BREAKING | [`migration/v0.46.md`](migration/v0.46.md) — scorecard as primary metric surface (ADR 0002) |
| v0.46.x → v0.47.0 | BREAKING | [`migration/v0.47.md`](migration/v0.47.md) — Sweep unification + TextTransform + advanced-6 adversarial |
| v0.47.x → v0.48.0 | BREAKING | [`migration/v0.48.md`](migration/v0.48.md) — `BootstrapCI.to_dict()` schema rewrite; sweep `strategy_id`; scorer-output shape validation |
| v0.48.x → v0.49.0 | BREAKING | [`migration/v0.49.md`](migration/v0.49.md) — global naming-standards sweep (5 Tier-1 renames + scorecards module promotion); ADR 0004 |
| v0.49.x → v0.50.0 | BREAKING | [`migration/v0.50.md`](migration/v0.50.md) — SPEC 7 `rng` parameter adoption (22 Tier-1 signatures: `seed=` → `rng=`) |
| v0.50.x → v0.51.0 | BREAKING | [`migration/v0.51.md`](migration/v0.51.md) — Round 8 audit rectification (R8-C3 recall_at_fpr sentinel; R8-C4a/b RNG parallel stability; R8-C2 SourceDisjoint k-cap) + R8-C1 reseed_splitter param + R8-C6/F1/F2/F3 validation rigor |

## Conventions

- Each guide is **self-contained** — copy-pastable before/after blocks
  + a `TypeError` / `ValueError` decoder when the new version raises
  loudly on legacy patterns.
- `TypeError` / `ValueError` messages from the toolkit cross-link
  back to these guides where applicable.
- For consumer projects pinning a version range (e.g.
  `eval-toolkit>=0.7,<0.8`), the relevant guide is the one matching
  your *target* version.

If you're migrating across multiple versions, read each guide in
order — the upgrades are not always commutative.
