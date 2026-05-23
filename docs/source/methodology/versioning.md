# Versioning Tier-2 implementations

> **Background** *(skip if you've internalized this)*. Cross-version
> metric comparisons are silently misleading. If a `Scorer`'s scoring
> recipe changed between two runs (different model checkpoint, bumped
> LoRA rank, new prompt template), reporting "PR-AUC went up by 0.04"
> is meaningless because the metric isn't measuring the same thing.
> `lm-evaluation-harness` solved this by stamping every task with a
> `VERSION` field, surfaced in every output. eval-toolkit adopts the
> same pattern via the
> [`Versioned`](../api/leakage.md) opt-in Protocol.

This chapter covers when to expose a `version` attribute on your
Tier-2 implementations (Scorer, LeakageCheck, Splitter,
ThresholdSelector, DatasetLoader) and how it threads into the
`RunManifest`.

## Setup

```python
from eval_toolkit import Versioned, make_manifest, write_manifest
import tempfile
```

(versioned-protocol)=
## The Versioned Protocol
The toolkit ships
[`Versioned`](../api/leakage.md) as a one-attribute
runtime-checkable Protocol:

```python
# from eval_toolkit.leakage import Versioned
# @runtime_checkable
# class Versioned(Protocol):
#     @property
#     def version(self) -> str: ...
```

**Opt-in**: implementations are not required to expose `version`. If
present, `make_manifest(versioned=...)` auto-collects it into
`RunManifest.versioned_objects`; if absent, the object is silently
skipped (per `_collect_versioned` in
[`manifest.py`](../api/manifest.md)).

(manifest-thread)=
## Threading through to the manifest
Pass any iterable or mapping of Tier-2 objects via the `versioned`
parameter:

```python
class _Scorer:
    """A toy Scorer with a version attribute."""
    version = "lr-tfidf-v1.2.0"
    def predict_proba(self, X):
        import numpy as np
        return np.full(len(X), 0.5)

class _ScorerNoVersion:
    """No version attribute → silently skipped."""
    def predict_proba(self, X):
        import numpy as np
        return np.full(len(X), 0.5)

m = make_manifest(
    run_id="versioning-demo",
    config={"model": "lr-tfidf"},
    versioned={"my_scorer": _Scorer(), "no_version": _ScorerNoVersion()},
)
print(m.versioned_objects)  # {"my_scorer": "lr-tfidf-v1.2.0"}
```

The mapping form gives you stable keys (recommended); the sequence
form keys by `type(obj).__name__`. Per
[manifest.py § _collect_versioned](../api/manifest.md),
both work.

(version-string)=
## Choosing a version string
A good version string is **a fingerprint that changes whenever the
metric's interpretation changes**. Conventions that work:

- **Semver**: `"v1.2.0"` — bump on any change to scoring logic. Good
  for stable models.
- **Date + hash**: `"2026-05-08-abc1234"` — bump on every retraining.
  Good for frequently-retrained models.
- **Composite**: `"deberta-v3-base-lora-rank8-2026-q1"` — embeds the
  recipe in the string. Self-documenting; verbose but unambiguous.
- **Checkpoint SHA**: `f"deberta-lora-{checkpoint_sha[:8]}"` —
  appends the model artifact's hash. Bit-identical replay possible.

**What NOT to do**: use a static `"v1"` for the lifetime of the
project. The whole point of the field is to *change* when the recipe
changes; a frozen string defeats it.

(versioning-when)=
## When to expose `version`
Expose it on every Tier-2 implementation whose output your reports
depend on. In practice that's:

- **Every `Scorer`**. If you have a regex baseline + an LR + a LoRA
  transformer + an LLM-judge, all four expose distinct version
  strings.
- **Custom `LeakageCheck` impls** if you maintain them — bumping the
  check's logic should invalidate prior leakage reports.
- **Custom `Splitter` / `DatasetLoader` impls** if their output is
  non-deterministic across versions (e.g., a fold-assignment bug fix
  would bump the version).

The toolkit's *built-in* reference impls intentionally don't expose
`version` — they're versioned indirectly via
`code_versions["eval_toolkit"]` in the manifest. Bumping the toolkit
version invalidates everything; you don't need per-class versions on
top.

(examples)=
## Convention examples (consumer-side)
From `prompt-injection-clean/docs/eval_toolkit_gaps.md` Gap 4 — the
canonical mapping for that project's 5 scorers:

```python
class LRBaselineScorer:
    """sklearn LogisticRegression on TF-IDF features."""
    version = "lr-tfidf-v1"

class FrozenProbeScorer:
    """Frozen DeBERTa-v3-base + sklearn LR head."""
    version = "frozen-deberta-v3-base-v1"

class ProtectAIScorer:
    """HuggingFace `protectai/deberta-v3-base-prompt-injection`."""
    def __init__(self, variant: str, revision: str) -> None:
        self.variant = variant
        self.revision = revision
        # Embeds variant + revision SHA prefix → bumps when either changes.
        self.version = f"protectai-{variant}-{revision[:8]}"

class LoRAScorer:
    """In-house DeBERTa + LoRA fine-tune."""
    def __init__(self, checkpoint_sha: str) -> None:
        self.checkpoint_sha = checkpoint_sha
        self.version = f"deberta-lora-v1-{checkpoint_sha[:8]}"
```

Then at eval time:

```python
class _LRBaselineScorer:
    version = "lr-tfidf-v1"
    def predict_proba(self, X):
        import numpy as np
        return np.full(len(X), 0.5)

class _LoRAScorer:
    def __init__(self, sha):
        self.version = f"deberta-lora-v1-{sha[:8]}"
    def predict_proba(self, X):
        import numpy as np
        return np.full(len(X), 0.5)

scorers = {"lr": _LRBaselineScorer(), "lora": _LoRAScorer("abc123def456")}
m = make_manifest(run_id="r", config={}, versioned=scorers)
print(m.versioned_objects)
```

(versioning-pitfalls)=
## Pitfalls / Common mistakes
- **Static version strings.** A scorer with `version = "v1"` that
  never changes is worse than no version at all — it gives a false
  sense of fingerprinting.
- **Version strings that include timestamps that aren't fingerprints.**
  `f"lr-{datetime.now().isoformat()}"` changes every run, making the
  manifest's `versioned_objects` field uselessly noisy. Use a stable
  recipe-fingerprint (semver, hash, frozen-date), not wall-clock.
- **Forgetting to bump.** If you change a scorer's hyperparameter
  (LoRA rank, regex pattern set, LLM model name) without bumping
  the version, the manifest looks the same as before — but the
  metric isn't comparable. Discipline matters.
- **Not passing `versioned=` to `make_manifest`.** The manifest
  builder doesn't auto-discover Versioned objects; you must pass them
  explicitly. Cross-link from the harness script you wrote: typically
  `versioned=scorers` (the dict you also pass to `evaluate(...)`).
- **Versioning eval-toolkit primitives separately.** Don't expose
  `version` on a vanilla `MaxF1Selector()` — it's already
  fingerprinted by `code_versions["eval_toolkit"]`. The Versioned
  Protocol is for *your* code, not the toolkit's reference impls.

## Putting it all together

```python
from eval_toolkit import (
    EvalSlice, evaluate, make_manifest, write_manifest,
    set_global_seeds,
)
import pandas as pd
import numpy as np

set_global_seeds(42)

class _RegexScorer:
    version = "regex-v1.0"
    def predict_proba(self, X):
        return np.full(len(X), 0.5)

class _LRScorer:
    version = "lr-tfidf-v1.2.0"
    def predict_proba(self, X):
        return np.full(len(X), 0.5)

scorers = {"regex": _RegexScorer(), "lr": _LRScorer()}
df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
result = evaluate(scorers, [EvalSlice(name="test", df=df)], run_id="r")

m = make_manifest(
    run_id="r",
    config={"scorers": list(scorers.keys())},
    versioned=scorers,  # ← auto-collects {regex: "regex-v1.0", lr: "lr-tfidf-v1.2.0"}
)
import tempfile, json
with tempfile.TemporaryDirectory() as d:
    path = write_manifest(m, d)
    print(json.loads(path.read_text())["versioned_objects"])
```

(schema-evolution)=
## Schema evolution policy
The toolkit's JSON Schemas live in
[`src/eval_toolkit/schemas/`](https://github.com/brandon-behring/eval-toolkit/tree/main/src/eval_toolkit/schemas) and
follow a `.vN.json` filename convention (`results.v1.json`,
`results_full.v1.json`, `manifest.v1.json`). This section documents
when the filename bumps and when it stays.

### Filename stays the same (additive changes)

A `.vN.json` filename is **stable** so long as every change is
additive and optional. Concretely, the filename does NOT bump when:

- A new top-level optional field is added.
- A new optional sub-field is added inside an existing object.
- A field's `enum` gains a value (consumers using older enums still
  validate; their value is just one of several allowed).
- Documentation strings (`description`) change.

The forward-compatibility contract is `additionalProperties: true`
on every object node: v0.8 consumers reading a v0.9 result silently
tolerate the new fields. v0.9 consumers reading a hypothetical v0.10
result tolerate that version's additions in the same way, as long as
v0.10 stays additive.

**v0.9 case study**: `results.v1.json` gained six new top-level
optional fields (`claim_report`, `prediction_artifacts`,
`evidence_axes`, `pairing_metadata`, `aggregate_evidence`,
`threshold_policy`) without a filename bump. v0.8 consumers continue
to read v0.9 outputs cleanly.

### Filename bumps (`.vN.json` → `.v(N+1).json`)

A new schema file ships under a bumped filename when any of these
apply:

- A field is **removed** from the schema.
- A field is **renamed**.
- A field's **semantic meaning** changes (same name, different
  interpretation downstream).
- A field's **type** changes (e.g., `string` → `integer`).
- A field is **moved** between `properties` and `required`.
- An object node's `additionalProperties` policy tightens from
  `true` to `false`.

Filename bumps are announced in the per-version migration guide
under `docs/migration/` and called out in `CHANGELOG.md`. When `v2`
ships, `v1` stays on disk for one more release cycle to give
consumers time to migrate; consumers can pin to the `.v1.json`
filename until they upgrade.

### The `schema_version` field inside each schema

Each schema also declares a top-level
`"version": "<N>"` and (for result schemas) a `schema_version` const
mirroring the filename:

```json
{
  "$id": "https://eval-toolkit/schemas/results.v1.json",
  "version": "1",
  "properties": {
    "schema_version": {"const": "v1"}
  }
}
```

The two fields are redundant on purpose: `$id` and `version` are for
schema-aware tools; `schema_version` is for human consumers who'd
rather grep a result file than its schema. Both are bumped together
on a filename change.

### Why not semantic versioning (`v1.1.json`)?

The `.vN.json` convention is intentionally coarser than semver. Any
breaking change → new file. There's no notion of a backward-
compatible-but-non-additive bump (`v1.1` adding a new required
field would still break v1 consumers; calling it `v1.1` would
mislead). Additive changes — the only ones that don't break — get
no filename change at all. This keeps consumer code simple: pin to
`.v1.json` and you're guaranteed forward-compat within the v1 line.

### See also

- [`migration/v0.9.md`](../migration/v0.9.md) §5 for the v0.9
  schema additions.
- [`CHANGELOG.md`](https://github.com/brandon-behring/eval-toolkit/blob/main/CHANGELOG.md) for the per-version record
  of which schemas grew.

## Further reading

- EleutherAI lm-evaluation-harness, [`task_guide`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
  — the canonical `VERSION`-field pattern this Protocol mirrors.
- Stanford HELM,
  [`schema_classic.yaml`](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/static/schema_classic.yaml)
  — versioned-as-artifact pattern for benchmark schemas.
- Mitchell, M. et al. *Model Cards for Model Reporting.* FAccT 2019.
  — the canonical "what should be in a model artifact's metadata"
  reference; `version` is one of the required fields.

See also: [reproducibility.md](reproducibility.md) (the manifest the
versioned objects land in), [extending.md
§"Implementing a Scorer"](../extending.md#scorer) (where to add the
`version` attribute on a custom Scorer).
