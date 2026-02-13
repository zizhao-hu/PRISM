# Research Plan: Classifier-free Intent-based System Prompt Routing through Self-Learning

**Title:** Classifier-free Intent-based System Prompt Routing through Self-Learning

## 1. Problem Statement & Motivation

### The System Prompt Dilemma
A one-size-fits-all system prompt is suboptimal for all user tasks:
- A **helpful assistant** persona may give unsafe answers to sensitive queries.
- A **safe assistant** persona may be overly cautious and less helpful on benign queries.
- Existing solutions either use **routing classifiers** (brittle, require labeled data) or **always-on safety** (degrades utility).

### Our Goal
Instead of routing queries to different prompts at inference time, we **route the model to the desired behavior** by distilling prompt-conditioned behaviors directly into the model weights. The model learns to implicitly activate the appropriate behavioral mode based on the query intent — without any explicit classifier or prompt switching.

### Core Objectives
1. **Prevent system prompt drift**: A safety prompt should not shift the model's global persona. Safety behavior should activate *only* for relevant queries, not degrade helpfulness everywhere.
2. **Improve utility on general queries**: For queries that don't require safety constraints, the model should perform identically (or better) compared to the base model without any system prompt.

## 2. Methodology

### Overview
We use **synthetic self-learning** to teach the model intent-conditioned behavior routing:

1. **Contrastive Data Generation:**
   - **Positive Set ($D_{+}$):** Queries where the safety context is relevant → generate responses *conditioned on* the safety system prompt.
   - **Negative Set ($D_{-}$):** Queries where the safety context is irrelevant → generate responses *without* the safety prompt (preserve original behavior).

2. **Self-Distillation:**
   - The model serves as its own teacher: the prompted model provides target behavior for $D_{+}$, and the unprompted model provides targets for $D_{-}$.
   - Training uses **KL-divergence distillation** against the teacher's output distribution, not just SFT on text outputs.
   - This preserves the full distribution rather than memorizing specific response patterns.

3. **Training Modes Studied:**
   - **SFT (Finetune):** Standard supervised fine-tuning on (query, response) pairs.
   - **Distillation:** KL-divergence matching against teacher logits.
   - **First-Token-Only:** Train only on the first steering token (e.g., "Sorry" vs "Sure") — tests whether directional signals alone suffice.

### Key Innovation: Classifier-Free Routing
Unlike methods that require an external classifier to decide when to apply safety, our approach:
- Learns to **implicitly recognize** when safety behavior is needed.
- **Routes internally** via learned weight modulations (LoRA adapters).
- Requires **no classifier at inference time** — the model itself determines the appropriate behavior.

## 3. Experiment Settings

### Base Model
- Primary: **Qwen2.5-1.5B-Instruct** (efficient for ablation studies)
- Extended: Qwen2.5-3B, Llama-3.2-3B, Llama-3.1-8B, Gemma-2-2B, Mistral-7B

### Safety Context
- **Context 1 (General Safety):** Comprehensive "harmless AI assistant" system prompt
- Extended: Target Safety, Claude Safety, Claude System prompts

### Evaluation Metrics
- **Safety:** Refusal Rate (RR ↑) on HarmBench adversarial queries
- **Utility:** Pairwise Win Rate vs. Base Model on benign AlpacaEval queries (higher = better preserved utility)
- **Drift:** KL Divergence between distilled model and base model on benign prompts (lower = less global persona shift)

### Hyperparameter Grid
- **Data sizes:** N = {50, 100, 200, 500}
- **Training epochs:** {2, 4, 6, 8, 10}  
- **Training steps:** epochs × N / batch_size (batch_size = 4)
- **Methods:** SFT (Finetune), Distillation, First-Token-Only variants

### Baselines
- **Base Model (No Context):** Unmodified model — high utility, lower safety
- **Base + In-Context Safety Prompt:** Oracle upper bound for safety at inference cost
- **Prompt Tuning:** Learned soft prompt prefix [Lester et al., 2021]
- **Context Compression (ICAE):** Compress long context into memory tokens
- **Standard Context Distillation:** KL-based distillation with generic data

## 4. Results & Findings

### Main Results
[Table: Safety and Utility across models and methods — see LaTeX paper Table 1]

### Ablation Study
Progressive ablation on Qwen2.5-1.5B:
1. Standard CD (External data)
2. Standard CD (Synthetic data)  
3. +Associative queries ($D_{+}$)
4. +Negative data ($D_{+} + D_{-}$)
5. +Rejection Sampling
6. +Trigger Token (Full DREAM)

---

### Finding 1: Distillation Achieves Comparable Safety with Dramatically Less Model Drift
**Observation:** Across all data sizes, distillation achieves ~67–70% refusal rate (comparable to finetune's 54–72%) while maintaining KL divergence 100–170× lower (0.001 vs 0.1–0.5).

**Interpretation:** Distillation preserves the model's global distribution while still learning safety-relevant behavior. This suggests the safety signal can be encoded in subtle probability shifts rather than large weight changes — consistent with the hypothesis that safety routing is more about *steering* than *rewriting*.

### Finding 2: Safety Performance is Primarily a Function of Data Quality, Not Quantity
**Observation:** For finetune, only N=200 shows consistent safety improvement (57→72% over epochs). N=500 does not outperform N=200, and N=50/100 plateau at ~60%. For distillation, N=50–100 already achieve ~68–70% with minimal training.

**Interpretation:** Beyond a moderate dataset size, adding more data provides diminishing returns. The model quickly learns the intent boundary from a small number of high-quality contrastive examples. Excessive data (N=500) may introduce noise or conflicting signals.

### Finding 3: Utility Drift Scales Linearly with Training Steps — But Only for Finetune
**Observation:** Finetune KL divergence grows linearly with training steps regardless of N. Distillation KL stays near-zero even at 1250 steps.

**Interpretation:** SFT causes cumulative parameter drift proportional to gradient updates. Distillation constrains the student to match the teacher's full output distribution, providing an implicit regularization that prevents drift even under prolonged training.

### Finding 4: [Hypothesized] First-Token Steering May Achieve Safety with Minimal Information
**Observation:** [Pending results from first-token experiments]

**Hypothesis:** If training only on the first word ("Sorry", "Sure", "I") achieves comparable safety, it would demonstrate that the model needs only a directional nudge at generation onset — the safety knowledge is already present in the pretrained weights, and our method simply learns when to activate it.

### Finding 5: [Hypothesized] The Classifier-Free Routing Emerges From Contrastive Examples
**Observation:** [Pending analysis of per-query routing behavior]

**Hypothesis:** Without any explicit intent classifier, the model learns to differentiate harmful from benign queries through the contrastive training signal alone. The positive/negative data acts as implicit supervision for query-intent recognition, with the LoRA weights serving as a learned routing function.

### Finding 6: [Hypothesized] Cross-Prompt Generalization
**Observation:** [Pending multi-context experiments]

**Hypothesis:** A model distilled on one safety context (e.g., general harmlessness) may partially generalize to related but unseen safety contexts (e.g., specific policy guidelines), suggesting the model learns a general "safety intent detector" rather than memorizing a specific prompt's outputs.

## 5. Implementation Status

### Completed
- [x] Data generation pipeline (`0_data_gen.py`)
- [x] Training pipeline with SFT, Distill, Hybrid, Grad-Proj modes (`1_train.py`)
- [x] Evaluation pipeline with safety, utility, and drift metrics (`2_eval.py`)
- [x] Hyperparameter search: Finetune (complete, 30-query utility)
- [x] Hyperparameter search: Distillation (complete, 100-query utility)
- [x] First-token-only training mode implementation
- [ ] Hyperparameter search: First-token experiments (submitted, job 6290992)
- [ ] Re-run finetune eval with 100-query utility for fair comparison
- [ ] Baseline methods (Prompt Tuning, Context Compression, Standard CD)
- [ ] Multi-model experiments
- [ ] Multi-context experiments

### Key Scripts
- `scripts/0_data_gen.py` — Synthetic contrastive data generation
- `scripts/1_train.py` — Training with SFT/Distill/Hybrid/GradProj + `--first_token_only`
- `scripts/2_eval.py` — Safety (HarmBench), Utility (G-Eval, Win Rate), Drift (KL)
- `run_hyperparam_search.sh` — Finetune hyperparameter sweep
- `run_hyperparam_search_distill.sh` — Distillation hyperparameter sweep
- `run_hyperparam_search_first_token.sh` — First-token experiments (both FT + distill)
