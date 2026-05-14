# Prompt-Injection Attacks — Taxonomy and Methods

This file covers A1 (attack taxonomy and foundational attack papers) and A2 (specific automated attack methods — GCG, PAIR). Defenses live in `02_defenses.md`. Evaluation benchmarks live in `03_benchmarks_and_standards.md`.

---

## A1. Attack taxonomy and foundational attacks

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Ignore Previous Prompt: Attack Techniques For Language Models | Perez & Ribeiro (2022) | NeurIPS 2022 ML Safety Workshop (Best Paper) | arXiv:2211.09527 | agencyenterprise/PromptInject | First systematic study of prompt-injection attacks; introduces PromptInject framework for goal hijacking and prompt leaking | Foundational paper that named and operationalized the prompt-injection attack class; established goal hijacking vs prompt leaking as the two foundational attack types |
| Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection | Greshake et al. (2023) | ACM AISec 2023 | arXiv:2302.12173 | — | Introduces indirect prompt injection — embedding adversarial instructions in data retrieved by LLM-integrated applications | Foundational paper for the indirect (data-channel) attack vector; comprehensive taxonomy of impacts: data theft, worming, information ecosystem contamination |
| Jailbroken: How Does LLM Safety Training Fail? | Wei, Haghtalab & Steinhardt (2023) | NeurIPS 2023 | arXiv:2307.02483 | — | Analyzes the failure modes of safety-trained LLMs through two principles: competing objectives and mismatched generalization | Foundational analytical framework for why aligned LLMs jailbreak; widely cited taxonomy of jailbreak categories grounded in training objectives |

## A2. Specific automated attack methods

| Title | Authors (year) | Venue | arXiv/DOI | GitHub | One-line description | Key contribution |
|-------|----------------|-------|-----------|--------|----------------------|------------------|
| Universal and Transferable Adversarial Attacks on Aligned Language Models | Zou et al. (2023) | arXiv preprint | arXiv:2307.15043 | llm-attacks/llm-attacks | GCG — Greedy Coordinate Gradient attack that optimizes adversarial suffixes against multiple open-source LLMs and transfers them to closed-source models | Demonstrated transferable jailbreak suffixes that work across ChatGPT, Bard, Claude, LLaMA-2-Chat. Reference attack for white-box-driven black-box transfer |
| Jailbreaking Black Box Large Language Models in Twenty Queries | Chao et al. (2023) | NeurIPS 2023 R0-FoMo Workshop | arXiv:2310.08419 | patrickrchao/JailbreakingLLMs | PAIR — Prompt Automatic Iterative Refinement; an attacker LLM iteratively refines jailbreaks against a target LLM | Semantic black-box jailbreaks in fewer than 20 queries; orders of magnitude more query-efficient than GCG. Reference attack for black-box red-teaming |
