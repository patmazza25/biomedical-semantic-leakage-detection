# Experiments: Biomedical Ontology-Based Semantic Leakage Detection

## Overview

This document describes **Experiment 1: Cross-Model Semantic Leakage Benchmark**, which evaluates four large language models on their tendency to produce semantically inconsistent chain-of-thought (CoT) reasoning when answering biomedical questions. The experiment covers three sub-analyses: (1) a per-model contradiction rate benchmark, (2) a depth-vs-leakage analysis, and (3) a guard signal discriminability analysis.

All figures are produced by `experiments/exp1_cross_model_benchmark.ipynb`.

---

## Experiment 1 — Cross-Model Semantic Leakage Benchmark

### Setup

| Parameter | Value |
|-----------|-------|
| Models tested | claude-haiku, gpt-4o-mini, gemini-flash, llama-3-70b |
| Questions | 40 biomedical questions (PubMedQA / MedQA subset) |
| NLI mode | Heuristic (token overlap + negation/direction patterns) |
| UMLS linking | Disabled (concept candidates generated, not CUI-linked) |
| Total runs | 4 models × 40 questions = 160 pipeline executions |
| Guard signals | lexical_duplicate, caution_band, direction_conflict |

### Sub-Experiment 1a — Per-Model Contradiction Rate

**Research question:** Do models differ significantly in their rate of semantic leakage across biomedical CoT reasoning?

#### Figure 1 — Cross-Model Benchmark Summary

![Experiment 1: Cross-Model Semantic Leakage Benchmark](experiments/results/result_images/exp1_1.png)

The figure contains three panels:

**(a) Contradiction Rate Distribution per Model (boxplot)**

Each box spans the interquartile range of per-question contradiction rates across all 40 questions.

| Model | Median Contradiction Rate | IQR | Notable Outliers |
|-------|--------------------------|-----|-----------------|
| claude-haiku | ~0.10 | 0.05–0.20 | Up to 0.38 |
| gpt-4o-mini | ~0.01 | 0.00–0.05 | Occasional outliers to 0.43 |
| gemini-flash | ~0.10 | 0.05–0.22 | Up to 0.50 |
| llama-3-70b | ~0.10–0.15 | 0.05–0.22 | Up to 0.25 |

**Key finding.** GPT-4o-mini has the lowest median contradiction rate (~0.01) and the tightest distribution — indicating consistently linear, forward-directed reasoning chains. Claude-haiku, gemini-flash, and llama-3-70b cluster at a similar median (~0.10) but differ in spread and outlier behaviour. Gemini-flash shows the widest spread, driven by specific questions that trigger long reasoning chains (see Sub-Experiment 1b).

**(b) Average NLI Label Breakdown per Question (stacked bar)**

The stacked bar shows average entailment (blue), neutral (purple), and contradiction (red) pair counts per question for each model.

| Model | Avg Pairs/Question | Avg Entailment | Avg Neutral | Avg Contradiction |
|-------|-------------------:|---------------:|------------:|------------------:|
| claude-haiku | ~12.0 | ~6.5 | ~4.5 | ~1.0 |
| gemini-flash | ~11.5 | ~6.0 | ~4.5 | ~1.0 |
| gpt-4o-mini | ~6.8 | ~3.3 | ~3.0 | ~0.5 |
| llama-3-70b | ~6.8 | ~3.3 | ~3.0 | ~0.5 |

**Key finding.** Claude-haiku and gemini-flash produce significantly longer reasoning chains (~12 pairs/question) compared to gpt-4o-mini and llama-3-70b (~7 pairs/question). Across 40 diverse questions, the heuristic NLI *does* detect entailment (blue bar is large for all models) — unlike the 3-question pilot where all entailment was zero. This is because across 40 questions, consecutive steps occasionally restate shared terminology, crossing the token-overlap threshold for heuristic entailment. Contradiction counts are absolute, so longer chains (claude-haiku, gemini-flash) accumulate more absolute contradictions despite similar per-pair rates.

**(c) Concept Validity vs Contradiction Rate (scatter)**

Panel (c) is empty — UMLS API was not configured. With UMLS enabled, each point would represent one question, with x = fraction of concepts successfully CUI-linked (concept validity) and y = contradiction rate. The expected hypothesis is that questions with lower concept validity (the model generates vaguer, unlinkable concepts) correlate with higher contradiction rates.

---

### Sub-Experiment 1b — Semantic Leakage Grows with Reasoning Depth

**Research question:** Does contradiction probability increase as the reasoning chain grows longer? Are late-chain steps more prone to leakage than early steps?

#### Figure 2 — Leakage vs. Reasoning Depth

![Semantic Leakage Grows with Reasoning Depth](experiments/results/result_images/exp1_2.png)

The figure contains two panels, both plotting a depth axis (step-pair index within the chain, 0 = first pair):

**(a) Contradiction Rate by Step-Pair Depth**

The y-axis shows the fraction of questions where the step-pair at that depth was classified as contradiction.

| Observation | Detail |
|-------------|--------|
| Depths 0–10 | All models show moderate contradiction rates (0.05–0.20), with mild upward trend |
| Claude-haiku | Elevated rates at depths 15–18 (~0.25–0.30) |
| GPT-4o-mini | Consistently near zero across all depths |
| Gemini-flash | Dramatic spike at depth ~22 reaching rate = 1.0 |
| Llama-3-70b | Moderate across all depths, minor elevations at depth 10–14 |

**(b) Average P(contradiction) by Step-Pair Depth**

The y-axis shows the mean contradiction probability across all questions at each depth. The pattern mirrors panel (a) with the gemini-flash spike at depth 22–23 reaching P(C) = 1.0.

**Key finding.** Contradiction risk is not uniform across the chain — it grows with depth. This is consistent with the "semantic drift" hypothesis: early steps establish a framework, and late steps are more likely to introduce qualifications, exceptions, or topic shifts that conflict with the established direction. The gemini-flash spike at depth 22 is a single-question artifact (only 1–2 questions produce chains of length 23+) rather than a statistically robust signal, but it demonstrates that very long chains are especially vulnerable to late-chain polarity reversals.

**GPT-4o-mini's flatness** supports the interpretation that shorter, more linear chains are less susceptible to depth-driven leakage — not because GPT-4o-mini avoids nuance, but because it encodes dual-effect acknowledgements within a single step rather than across multiple steps.

---

### Sub-Experiment 1c — Guard Signal Analysis

**Research question:** Which of the three guard signals (lexical_duplicate, caution_band, direction_conflict) best discriminates between contradiction and non-contradiction step pairs?

#### Figure 3 — Guard Signal Frequency and Discriminability

![Guard Signal Analysis](experiments/results/result_images/exp1_3.png)

The figure contains two panels:

**(a) Average Guard Signal Frequency per Model**

Bar chart showing mean number of times each guard signal fires per question, per model.

| Model | caution_band | direction_conflict | lexical_duplicate |
|-------|--------------:|-------------------:|------------------:|
| claude-haiku | ~5.4 | ~1.2 | 0.0 |
| gemini-flash | ~5.0 | ~0.9 | 0.0 |
| gpt-4o-mini | ~2.5 | ~0.5 | 0.0 |
| llama-3-70b | ~3.0 | ~0.7 | 0.0 |

**Key observations:**
- **lexical_duplicate = 0** for all models. Expected: high-quality LLMs never copy-paste steps verbatim, so no consecutive pair is lexically identical.
- **caution_band dominates.** It fires on nearly every question for every model. This reflects the model's tendency to use hedging language ("however", "may", "depending on", "in some cases") — common in biomedical reasoning where evidence is probabilistic.
- **direction_conflict is moderate.** Claude-haiku fires it most (~1.2 per question), consistent with its higher absolute contradiction count.

**(b) Guard Signal Rate: Contradiction Pairs vs. Other Pairs**

Bar chart showing what fraction of contradiction-classified pairs and non-contradiction pairs each guard fires on.

| Guard Signal | Rate on Contradiction Pairs | Rate on Non-Contradiction Pairs |
|---|---|---|
| direction_conflict | **~28%** | ~5% |
| caution_band | ~5% | **~43%** |
| lexical_duplicate | 0% | 0% |

**Key finding and interpretation:**

- **direction_conflict is a strong discriminator.** It fires 28% of the time on contradiction pairs but only 5% on non-contradiction pairs — a 5.6× lift. This validates its design: it fires when consecutive steps use antonymous direction verbs (e.g., "increases → decreases", "promotes → inhibits").

- **caution_band is an inverse discriminator.** It fires 43% of the time on *non-contradiction* pairs and only 5% on contradiction pairs. This is counterintuitive but makes sense: clear contradictions are high-confidence polarity flips (direction_conflict fires), while hedging language ("however, in some patients...") marks *uncertain* transitions, not outright contradictions. Caution_band captures epistemic uncertainty in the reasoning, which is distinct from logical contradiction.

- **Practical implication.** A detector that uses caution_band as a positive contradiction signal would have very high false-positive rate. The correct use is: direction_conflict as the primary contradiction predictor, caution_band as an uncertainty/nuance signal that warrants human review but not automatic contradiction labelling.

---

## Validity Check

The experiment results are internally consistent:

- **Entailment appears at 40 questions (not 3).** With a larger and more diverse question set, token overlap across consecutive steps is sufficient to cross the heuristic entailment threshold — confirming the heuristic is functional, not always-neutral.
- **Contradiction distributions are non-uniform.** Rates vary systematically by model and depth, not randomly — confirming content-driven signal.
- **Gemini-flash depth spike is explainable.** Only 1–2 questions in 40 generate chains of length 23+; the spike is a small-sample artifact, not a bug.
- **direction_conflict 5.6× lift is meaningful.** A random guard signal would fire at equal rates on contradiction and non-contradiction pairs; the observed 5.6× difference confirms the signal is informative.
- **lexical_duplicate = 0 is expected.** LLMs producing distinct, advancing steps will never produce verbatim-duplicate consecutive steps at scale.

---

## Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Heuristic NLI only | Entailment detection limited to token overlap; direction_conflict based on surface verb patterns | Run with `FORCE_HEURISTIC_NLI=0` for PubMedBERT-BioNLI-LoRA model |
| UMLS not configured | Panel (c) empty; concept validity = 0% | Set `UMLS_API_KEY` (free via NLM at uts.nlm.nih.gov) |
| Depth spike at 22 (gemini) | 1–2 data points; not statistically robust | Increase question set size; stratify by chain length |
| No gold labels | Contradiction rate is automated; no human annotation of ground truth | See Exp 3 (guard signal analysis with 120 gold-labeled pairs) |
| caution_band interpretation | High false-positive rate if used as contradiction predictor | Use only as uncertainty signal; weight direction_conflict higher |

---

## Connection to Other Experiments

| Experiment | Relation to Exp 1 |
|------------|------------------|
| **Exp 2** (Cross-Question Consistency) | Uses same models; checks whether contradiction patterns are consistent for the *same question* asked across different phrasings |
| **Exp 3** (Guard Signal Analysis) | 120 gold-labeled pairs; formal ablation of Pure NLI / NLI+UMLS / NLI+Guards / Full Hybrid; provides ROC/PR curves for direction_conflict discriminability found in Exp 1 Panel 3b |
| **Exp 4** (Contradiction Repair) | Takes the contradiction-prone questions identified in Exp 1 and compares generic vs. ontology-grounded repair prompts to reduce leakage |

---

---

## Experiment 2 — Cross-Question Ontological Consistency

### Setup

| Parameter | Value |
|-----------|-------|
| Models tested | claude-haiku, gpt-4o-mini, gemini-flash, llama-3-70b (via OpenRouter) |
| Concept groups | 6 biomedical concepts: aspirin, metformin, statins, insulin, ACE inhibitors, beta blockers |
| Questions per group | 5–6 questions per concept (30 total) |
| NLI mode | Heuristic (token overlap + negation/direction patterns) |
| Cross-question pairs | Up to 50 sampled step-pairs per concept from different questions |
| Comparison | Within-answer contradiction rate vs. cross-answer contradiction rate per concept |

**Research question:** Do LLMs contradict themselves *across* different questions about the same medical concept, even when each individual answer is internally coherent?

**Method:** For each concept group, steps are extracted from CoT answers to multiple related questions. NLI is run on pairs of steps drawn from *different* questions (cross-answer). The resulting contradiction rate is compared to the within-answer rate from Exp 1.

---

### Figure 4 — Cross-Question Contradiction Rate and Within vs. Cross Comparison

![Experiment 2: Cross-Question Contradiction Rate and Within vs Cross Comparison](experiments/results/result_images/exp2_1.png)

The figure contains two panels:

**(a) Cross-Question Contradiction Rate by Concept**

Bar chart sorted descending by cross-answer contradiction rate.

| Concept | Cross-Answer Contradiction Rate |
|---------|--------------------------------:|
| Insulin | ~12.5% |
| Metformin | ~10.0% |
| Aspirin | ~6.0% |
| ACE Inhibitors | ~5.0% |
| Statins | ~2.5% |
| Beta Blockers | ~0.0% |

**Insulin** has the highest cross-question contradiction rate (~12.5%), reflecting the complexity and context-dependence of insulin physiology — the model answers differently depending on whether the question frames insulin in terms of type 1 vs. type 2 diabetes, hepatic vs. peripheral mechanisms, or normal physiology vs. pathological resistance.

**Beta blockers** show essentially zero cross-question contradiction. Beta blocker mechanisms (β1 receptor blockade → reduced heart rate and contractility) are unambiguous and well-established; no LLM contradicts itself about this across questions.

**(b) Within-Answer vs. Cross-Answer Contradictions**

Grouped bar chart (blue = within-answer, red = cross-answer) for all six concept groups.

| Concept | Within-Answer Rate | Cross-Answer Rate | Ratio (within/cross) |
|---------|-------------------:|------------------:|---------------------:|
| Insulin | ~22.5% | ~12.5% | 1.8× |
| Metformin | ~17.5% | ~10.0% | 1.75× |
| Aspirin | ~16.5% | ~6.0% | 2.75× |
| Statins | ~15.0% | ~2.5% | 6.0× |
| ACE Inhibitors | ~11.5% | ~5.0% | 2.3× |
| Beta Blockers | ~1.5% | ~0.0% | — |

**Key finding: within-answer contradiction rates are consistently and substantially higher than cross-answer rates for every concept.** LLMs generate more self-contradiction within a single reasoning chain than across separate responses about the same concept. This is because within-chain drift occurs when the model builds a causal mechanism chain and then pivots to acknowledge a clinical caveat or counter-effect — a within-response topic shift. Across separate questions, each response starts fresh and stays more directionally consistent.

The ratio is largest for statins (6×) — within-chain statin reasoning often moves from "statins inhibit HMG-CoA reductase → ↓LDL" to "statins also cause myopathy in some patients", a large direction flip. Cross-question statin responses are much more uniform.

---

### Figure 5 — Cross-Question Contradiction Heatmaps per Concept

![Cross-Question Contradiction Rate Heatmaps per Concept](experiments/results/result_images/exp2-3.png)

Each of the six heatmaps shows the pairwise cross-question contradiction rate between every pair of questions in a concept group. Colour scale: white = 0.0 (no contradiction), dark red = 1.0 (always contradicts). Axes are question indices within the group.

#### Aspirin

Diffuse, light-pink pattern across most question pairs (~0.2–0.4). No single dominant pair stands out — the contradiction signal is spread across many question combinations. This reflects aspirin's dual-effect profile (antiplatelet → cardioprotective, but also ulcerogenic and anticoagulant), which creates mild inconsistency regardless of which question pair is compared.

#### Metformin

Focal structure — pair (Q0, Q1) is the darkest cell (~0.8–1.0), indicating that questions about "mechanism of action" and "contraindications" produce strongly contradictory CoT steps. This makes pharmacological sense: a mechanistic question elicits "metformin activates AMPK to reduce hepatic glucose output", while a contraindications question elicits "metformin should not be used in renal failure due to lactic acidosis risk" — the divergent framing creates a polarity conflict. Most other pairs are white (low contradiction).

#### Statins

Sparse pattern. The most notable hotspot is pair (Q1, Q3): "Do statins reduce the risk of stroke?" vs "Do statins reduce mortality in patients with heart failure?" — evidence for statins in heart failure is mixed, while stroke reduction evidence is strong, creating an inconsistency in the model's outcome claims.

#### Insulin

Broadly distributed moderate pink (~0.2–0.4) across many pairs — the most uniform heatmap of the six. This reflects systematic ambiguity: insulin's role differs substantially across the five question framings (blood glucose regulation, insulin resistance mechanism, therapy risks, lipid metabolism, basal vs. bolus types), and the model inconsistently attributes mechanisms across these contexts.

#### ACE Inhibitors

Focal hotspot concentrated at pairs involving Q2 ("Do ACE inhibitors protect renal function?"), Q3 ("What are the adverse effects?"), and Q4 ("Can ACE inhibitors cause hyperkalemia?"). This is a medically meaningful cluster: ACE inhibitors protect renal function in early diabetic nephropathy (a beneficial effect) but can precipitate acute kidney injury in volume-depleted patients and cause hyperkalemia (adverse effects). The model contradicts itself by asserting protective and harmful renal effects without resolving the contradiction.

#### Beta Blockers

Nearly all white across all question pairs — the most consistent concept in the dataset. Beta blocker mechanisms are well-characterised and directionally consistent across all five question framings (HR reduction, post-MI survival, contraindications, cardiac output, hypertension). LLMs do not contradict themselves about beta blockers across questions.

---

### Validity Check

- **Ranking order is pharmacologically meaningful.** Insulin (complex, context-sensitive) > metformin > aspirin > ACE inhibitors > statins > beta blockers (simple, unambiguous). This tracks with clinical pharmacology complexity, not noise.
- **Focal vs. diffuse heatmap patterns are explainable.** Metformin and ACE inhibitors show focal contradiction hotspots driven by specific question-type contrasts (mechanism vs. side-effect). Insulin shows diffuse contradictions consistent with broad physiological complexity.
- **Within > cross for all concepts.** The consistent direction (within-answer always higher) is internally valid and mechanistically explainable — it is not random.
- **Beta blockers near zero in both within and cross.** A ceiling-floor consistency: the concept with the most unambiguous pharmacology shows the lowest contradiction rate in both dimensions.
- **Cross-answer rate is ~40–70% of within-answer rate.** A plausible ratio — cross-question responses share some inconsistency (especially for complex concepts like insulin) but are shielded from the within-chain pivot dynamic.

---

### Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Heuristic NLI | Cross-answer contradiction detection based on surface patterns; may miss semantic contradictions | Run with full PubMedBERT-BioNLI-LoRA model |
| 5–6 questions per concept | Small within-group sample; heatmap cells may reflect 1–2 step pairs each | Expand to 10–15 questions per concept |
| Single primary model for downstream analysis | Cross-NLI run with claude-haiku steps only; other models cached but not compared | Run cross-NLI separately per model and compare |
| No gold labels for cross-answer pairs | Automated contradiction detection only | Annotate top contradiction examples manually |

---

### Connection to Other Experiments

| Experiment | Relation to Exp 2 |
|------------|------------------|
| **Exp 1** (Cross-Model Benchmark) | Provides the within-answer contradiction baseline used in the within vs. cross comparison |
| **Exp 3** (Guard Signal Analysis) | Tests whether guard signals (direction_conflict) that work for within-answer pairs also discriminate cross-answer contradiction pairs |
| **Exp 4** (Contradiction Repair) | Ontology-grounded repair prompts target the concept-level inconsistencies identified in Exp 2 |
