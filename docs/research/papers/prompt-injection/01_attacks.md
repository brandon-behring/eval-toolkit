# Prompt-Injection Attacks — Synthesis

This file synthesizes A1 (attack taxonomy + foundational attack papers) and A2 (specific automated attack methods). Companion raw-table dossier: `_dossier/01_attacks.md`.

---

## A1. Attack taxonomy and foundational attacks

- **Ignore Previous Prompt: Attack Techniques For Language Models** — Perez & Ribeiro (NeurIPS 2022 ML Safety Workshop, Best Paper).
  - **Source:** https://arxiv.org/abs/2211.09527
  - **Code:** https://github.com/agencyenterprise/PromptInject
  - **Mechanism:** Introduces PromptInject — a modular framework for assembling adversarial prompts that probe GPT-3's robustness; demonstrates two attack types: goal hijacking and prompt leaking.
  - **Result:** Foundational paper that named and operationalized the prompt-injection attack class; established goal hijacking and prompt leaking as the canonical attack-type distinction still used by OWASP LLM01:2025.
  - **Status:** Verified.

- **Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection** — Greshake et al. (ACM AISec 2023).
  - **Source:** https://arxiv.org/abs/2302.12173
  - **Code:** —
  - **Mechanism:** Demonstrates injection attacks that embed adversarial instructions in data sources (web pages, emails, files) that an LLM-integrated application retrieves rather than user prompts.
  - **Result:** Foundational paper for the indirect (data-channel) attack vector; gives a security-flavored taxonomy of impacts: data theft, worming, information ecosystem contamination. Demonstrates against Bing GPT-4 Chat. Underpins OWASP LLM01:2025's split between direct and indirect injection.
  - **Status:** Verified (no widely-known repo).

- **Jailbroken: How Does LLM Safety Training Fail?** — Wei, Haghtalab & Steinhardt (NeurIPS 2023).
  - **Source:** https://arxiv.org/abs/2307.02483
  - **Code:** —
  - **Mechanism:** Analyzes failure modes of safety-trained LLMs through two principles: (1) competing objectives — safety training conflicts with capability/helpfulness; (2) mismatched generalization — safety training doesn't generalize to all input distributions.
  - **Result:** Foundational analytical framework explaining why aligned LLMs jailbreak; widely cited two-principle taxonomy grounded in training-objective analysis rather than ad-hoc attack-pattern enumeration.
  - **Status:** Verified (no widely-known repo).

## A2. Specific automated attack methods

- **Universal and Transferable Adversarial Attacks on Aligned Language Models** — Zou et al. (arXiv 2023).
  - **Source:** https://arxiv.org/abs/2307.15043
  - **Code:** https://github.com/llm-attacks/llm-attacks
  - **Mechanism:** GCG (Greedy Coordinate Gradient) — optimizes adversarial suffixes via coordinate-wise gradients against multiple smaller open-source LLMs to maximize affirmative-response probability.
  - **Result:** Demonstrated transferable jailbreak suffixes that work across ChatGPT, Bard, Claude, LLaMA-2-Chat, Pythia, Falcon. Reference attack for white-box-driven black-box transfer; AdvBench (the attack's benchmark dataset) became a widely-used eval set.
  - **Status:** Verified.

- **Jailbreaking Black Box Large Language Models in Twenty Queries** — Chao et al. (NeurIPS 2023 R0-FoMo Workshop).
  - **Source:** https://arxiv.org/abs/2310.08419
  - **Code:** https://github.com/patrickrchao/JailbreakingLLMs
  - **Mechanism:** PAIR (Prompt Automatic Iterative Refinement) — an attacker LLM iteratively queries the target LLM and refines its jailbreak candidates, inspired by social-engineering attacks.
  - **Result:** Semantic black-box jailbreaks in fewer than 20 queries; orders of magnitude more query-efficient than GCG. Reference attack for query-efficient black-box red-teaming; works against GPT-3.5/4, Vicuna, and Gemini.
  - **Status:** Verified.
