<!--
Pull request template for eval-toolkit. Fill out each section; delete
the HTML comments before submitting.

GitHub Forms doesn't support PR templates yet, so this stays markdown.
-->

## Summary

<!-- 2-3 sentences: what does this change, and why? -->

## Testing

<!-- Check off as you go. Required for non-trivial changes. -->

- [ ] `make ci` passes locally (lint + type + tests + coverage gate)
- [ ] New tests added (or rationale for not needing them — e.g., docstring-only change)
- [ ] Manual verification done if user-visible behavior changes (paste the command + observed output)
- [ ] v4 sibling-smoke advisory check passes on the PR (or red is diagnosed in a comment)

## CHANGELOG

<!-- If this lands in a release, add an entry under `## [Unreleased]` in CHANGELOG.md
following Keep-a-Changelog format. Internal-only / no-user-impact changes can be omitted. -->

- [ ] CHANGELOG.md updated
- [ ] N/A — internal-only, no user-visible change

## Linked issue

<!-- "Closes #N" to auto-close on merge; "Refs #N" for related-but-not-resolved. -->

## Risk

<!-- One-line risk assessment: low / medium / high + brief why. Examples:
  - low — docstring-only change
  - medium — touches harness.evaluate(), but covered by existing reproducibility tests
  - high — changes a public dataclass shape; consumers will need to update
-->
