# NeurIPS 2026 Official Review

**Paper:** The Information Bottleneck of Chain-of-Thought: Quantifying Reasoning Redundancy in Large Language Models

**Recommendation: Borderline Reject (Score: 4/10)**

**Confidence: 4/5 (High)**

---

## 1. Summary

The paper proposes an information-theoretic framework to measure per-token reasoning redundancy in LLM chain-of-thought (CoT) generation. The central metric, Reasoning Information Gain (RIG), measures the reduction in answer entropy at each reasoning step. The authors derive a lower bound on minimum effective reasoning length (Theorem 1), characterize a three-phase information structure (Proposition 4), and propose an early stopping method. Experiments on GSM8K, MATH, ARC, and HumanEval with two 7B models are reported.

---

## 2. Strengths

**S1. Timely and important problem.** The inefficiency of reasoning models is a critical bottleneck for deployment. Providing a theoretical foundation for "how much reasoning is enough" is a genuinely valuable research direction that the community needs.

**S2. Clean formulation.** The RIG definition (Definition 1), CRI telescoping identity (Eq. 4), and the efficiency metric eta(t) are cleanly formulated. The conceptual pipeline from information-theoretic quantity to practical early stopping is logical and easy to follow.

**S3. Comprehensive theoretical apparatus.** The paper presents a substantial set of formal results: 2 theorems, 2 propositions, 1 lemma, 1 corollary, 2 definitions, 2 remarks. The framework overview figure (Figure 1) is excellent and clearly communicates the full pipeline.

**S4. Well-structured related work.** The four-paragraph organization (CoT, test-time compute, efficient reasoning, IB) is thorough, positions the work cleanly, and correctly identifies the gap.

---

## 3. Weaknesses

**W1. [Critical] The core theoretical results are shallow.**

- **Theorem 1** (lower bound T*(alpha) >= alpha * I_total / h_r) follows in two lines from the data processing inequality and the definition of entropy rate. This is a direct, elementary application of standard information theory with no technical novelty. The bound applies identically to *any* sequential process.

- **Theorem 2** (truncation loss via Fano's inequality) is simply a restatement of Fano's inequality (1961). The "proof" consists of one sentence.

- **Proposition 4** (three-phase characterization) assumes a multiplicative model RIG(t) = gamma(t) * U(t) with unimodal gamma and derives that the product has a single peak then decays. This is a tautology: the proposition assumes its own conclusion.

**W2. [Critical] The RIG estimator (Definition 5) is not the RIG (Definition 1), and the gap is not adequately addressed.**

- Part (1) of Proposition 3 establishes an upper bound, meaning the estimator could be arbitrarily larger than the true RIG.
- Part (3) uses vague qualifiers ("up to an additive correction") without specifying the correction term.
- The proof of Part (1) contains a logical error: it writes two inequalities pointing in opposite directions that do not chain.
- The estimator spikes at syntactically surprising tokens that carry zero answer-relevant information.

**W3. [Critical] All experimental results use synthetic/placeholder data.** No evidence that reported numbers come from actual model runs.

**W4. [Major] Only 7B models, 4-bit quantized, greedy decoding.** Narrow evaluation scope. Quantization artifacts likely affect information-theoretic quantities.

**W5. [Major] The early stopping method is simplistic and under-evaluated.** Missing comparisons to Certaindex, Token-Budget-Aware, OThink-R1, answer convergence. No ablation study.

**W6. [Moderate] Lemma 1 Part (2) is incorrect.** Equal entropy does not imply equal distributions.

**W7. [Moderate] The "Information Bottleneck" framing is misleading.** The paper does not solve an IB optimization problem.

**W8. [Minor] Missing important references.** Self-consistency (Wang et al. 2023), CoT theoretical expressiveness (Feng et al. 2024, Merrill & Sabharwal 2023).

---

## 4. Questions for Authors

Q1. Can you provide a concrete example where Theorem 1 gives a non-trivial prediction? For typical values (h_r ~ 3 nats, I_total ~ 5 nats), the bound gives T* >= 1.6 tokens.

Q2. How do you distinguish information-rich tokens from merely syntactically surprising ones in the estimator?

Q3. How sensitive are early stopping results to the prompt suffix choice?

Q4. Proposition 4 assumes unimodal gamma. How does the framework handle multi-modal gamma (repeated eureka moments)?

---

## 5. Minor Issues

- Eq. 4 second equality holds only when all RIG(i) >= 0.
- Rate-distortion remark is suggestive but not formalized.
- deepseekr1 BibTeX entry has incorrect title.
- ARC-Challenge and HumanEval lack BibTeX citations.

---

## 6. Recommendations for Revision

1. Replace Theorem 1 with a tighter, reasoning-specific bound exploiting CoT structure.
2. Formally bound the estimator gap |RIG_hat(t) - RIG(t)|.
3. Run all experiments with real model outputs.
4. Expand baselines to include Certaindex, Token-Budget-Aware, answer convergence.
5. Test on larger models (14B+) and fp16 precision.
6. Fix Lemma 1(2), Proposition 3 proof, BibTeX errors.
7. Rename or reframe the "Information Bottleneck" connection.
