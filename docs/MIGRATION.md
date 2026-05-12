# Migration guides

Per-version migration guides for breaking + behavior-changing
upgrades. Use these alongside [`CHANGELOG.md`](../CHANGELOG.md) — the
changelog records *what* changed; these docs walk through *how* to
update consumer code.

## Index

| From → To | Type | Guide |
|---|---|---|
| v0.6.x → v0.7.x | BREAKING | [`migration/v0.7.md`](migration/v0.7.md) — `select_threshold` API + new Protocol surfaces |
| v0.7.x → v0.8.0 | BREAKING (small) | [`migration/v0.8.md`](migration/v0.8.md) — ECE input validation enforced; `__version__` mismatch closed; `[parquet]` extra |
| v0.8.x → v0.9.0 | ADDITIVE | [`migration/v0.9.md`](migration/v0.9.md) — evidence core: `claims`, `artifacts`, `operating_points`, `evidence`, `analysis`, `protocols`; six new optional `RunResult` fields; `[validation]` extra |

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
