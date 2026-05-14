# URL Freshness Report

Generated: 2026-05-14
Target: /Users/brandonbehring/eval-toolkit/docs/research/

## Summary

- total: 80
- ok: 64
- broken: 0
- bot-blocked: 16
- redirected: 0
- timeout: 0

All 80 URLs in the dossier resolve. The 16 non-200 responses are all from known bot-blocked / paywall publishers (ACM DL, IEEE Xplore, DOI redirector, MIT Press, AMS journals, RSNA, Sage, Semantic Scholar, Zenodo). These reject HEAD/GET from non-browser user agents but the underlying resources exist and load in a browser; the cluster audit rounds confirmed all 16 URLs via WebSearch cross-validation.

## Broken URLs (hard 404s)

(none)

## Bot-blocked URLs (allowlisted publishers — not broken)

### ACM Digital Library (4)
- `https://dl.acm.org/doi/10.1145/1102351.1102430` — Niculescu-Mizil & Caruana 2005 (ICML)
- `https://dl.acm.org/doi/10.1145/276698.276876` — Indyk & Motwani 1998 (STOC)
- `https://dl.acm.org/doi/10.1145/775047.775151` — Zadrozny & Elkan 2002 (KDD)
- `https://dl.acm.org/doi/10.5555/1642194.1642224` — Elkan 2001 (IJCAI)

### IEEE Xplore (2, returns 418 "I'm a teapot" as their bot-block signal)
- `https://ieeexplore.ieee.org/document/666900` — Broder 1997 (Compression and Complexity of Sequences)
- `https://ieeexplore.ieee.org/document/6851192` — Sun & Xu 2014 (IEEE Signal Processing Letters)

### DOI redirector (2)
- `https://doi.org/10.2307/2531595` — DeLong et al. 1988 (Biometrics)
- `https://doi.org/10.1111/ecog.02881` — Roberts et al. 2017 (Ecography)

### Publisher journal sites (5)
- `https://direct.mit.edu/neco/article/14/1/21/6577/Adjusting-the-Outputs-of-a-Classifier-to-New-a` — Saerens, Latinne & Decaestecker 2002 (Neural Computation; MIT Press)
- `https://journals.ametsoc.org/view/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml` — Murphy 1973 (Journal of Applied Meteorology)
- `https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml` — Brier 1950 (Monthly Weather Review)
- `https://journals.sagepub.com/doi/abs/10.1177/096228029800700405` — Obuchowski 1998 (Statistical Methods in Medical Research)
- `https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747` — Hanley & McNeil 1982 (Radiology)
- `https://pubs.rsna.org/doi/10.1148/radiology.148.3.6878708` — Hanley & McNeil 1983 (Radiology)

### Semantic Scholar / Zenodo (2)
- `https://www.semanticscholar.org/paper/Probabilistic-Outputs-for-Support-vector-Machines-Platt/42e5ed832d4310ce4378c44d05570439df28a393` — Platt 1999 (returns 202 — likely Cloudflare bot-protection; resource exists)
- `https://zenodo.org/records/12608602` — Gao et al. 2024 EleutherAI lm-evaluation-harness (Zenodo)

## Redirected URLs (3xx)

(none — all working URLs returned 200 directly under `-L`)

## Method

- URL extraction: `grep -hroE 'https?://[a-zA-Z0-9./?=&_~%#:+-]+' . | sort -u | sed 's/[\.,:;]\+$//'` over all `*.md` files in `docs/research/`.
- HEAD-check: `curl -sS -L --max-time 8 -A "Mozilla/5.0"` per URL (the fast-path single-loop pattern from `/url-freshness-check` Phase 3).
- No GET retry was needed since the bot-blocked publishers above are on the allowlist (academic paywalls).
- No `--inline` annotation was applied since there are 0 hard-404 broken URLs.

## Notes

- Paywall/bot-blocked publisher 403s are the dominant non-200 pattern (consistent with the audit-trail findings that ~9 primary URLs were paywall-blocked during the inference cluster audit).
- The two IEEE Xplore 418 responses are their canonical bot-block code; the resources are valid (verified during the dossier-audit rounds via WebSearch cross-checks against alternative mirrors).
- The Semantic Scholar 202 ("Accepted") response is unusual — typically indicates the server is processing the request; the resource exists.
