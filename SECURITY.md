# Security Policy

## Supported Versions

eval-toolkit follows [Semantic Versioning](https://semver.org/). Security
fixes are applied to the latest minor release line.

| Version | Supported |
| ------- | --------- |
| 0.28.x  | ✅ Current release line |
| 0.27.x  | ⚠️ Critical security fixes only |
| ≤ 0.26  | ❌ Unsupported |

Pre-1.0 minor bumps may include breaking changes per
[`CONTRIBUTING.md`](CONTRIBUTING.md). Security fixes preserve API
compatibility within a minor line whenever feasible.

## Reporting a Vulnerability

Please report security vulnerabilities **privately** — do not open a
public GitHub issue.

**Preferred:** [GitHub Security Advisory](https://github.com/brandon-behring/eval-toolkit/security/advisories/new)
(private vulnerability reporting).

**Alternate:** email `brandon.m.behring@gmail.com` with subject
`[eval-toolkit security]`.

### What to include

- Affected version(s)
- Description of the vulnerability + impact assessment
- Steps to reproduce (minimal example preferred)
- Any suggested mitigation or fix (optional)

### Response timeline

- **Acknowledgement:** within 72 hours
- **Initial assessment + triage:** within 7 days
- **Fix + disclosure:** depends on severity and complexity. We coordinate
  on a disclosure timeline that balances user safety with responsible
  upstream patching.

### Disclosure + credit

Once a fix is shipped, reporters are credited in the `CHANGELOG.md`
entry for the fix release unless they request anonymity. Contributors
who provide a working patch are co-authored on the fix commit.

## Scope

In-scope:

- Vulnerabilities in `src/eval_toolkit/` source code that affect
  consumers (e.g., a deserializer that can be tricked into RCE, a
  metric that silently returns wrong values on adversarial input).
- Vulnerabilities in build / publish workflows
  (`.github/workflows/publish.yml`) that could be exploited to ship a
  malicious release.
- JSON schema validation bypasses that could let malformed manifests
  slip through `validate_manifest` / `validate_results`.

Out-of-scope:

- Vulnerabilities in upstream dependencies (`numpy`, `scipy`,
  `scikit-learn`, etc.) — report those to the upstream project.
- Denial-of-service via deliberately-malformed input where the toolkit
  raises a clear `ValueError` (working as designed).
- Vulnerabilities specific to consumer code that calls eval-toolkit
  (not the toolkit itself).

## See also

- [CONTRIBUTING.md](CONTRIBUTING.md) — development workflow and
  responsible-disclosure expectations for contributors.
- [LICENSE](LICENSE) — MIT.
