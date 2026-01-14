# ACL Rolling Review Plan: DREAM (Data-free Rehearsal-Enabled Adaptive Memory)

**Title:** DREAM: Consolidating In-Context Instructions into Weights via Associative Synthetic Replay

## 1. Abstract & Introduction
*   **Problem:** In-Context Learning is powerful and not intrusive to LLM utility but has high tokens cost and utility is limited to one chat session. SFT requires extra data and the finetuning on a specific domain trades off general utility for domain expertise.
*   **Solution:** DREAM compiles the "Short-Term Memory" (Prompt) into "Long-Term Memory" (Weights) via synthetic rehearsal.
*   **Key Claim:** DREAM achieves **Prompt Compilation**—the model behaves *as if* the prompt is present, but with zero inference cost and perfect stability.

## 2. Related Work & Positioning
We explicitly position DREAM against existing methods:

| Feature | Context Distillation | Prompt Baking | **DREAM (Ours)** |
| :--- | :--- | :--- | :--- |
| **Use Case** | Few-Shot Learning / Efficiency | General Prompt Integration | **Reasoning & Adaptation** |
| **Data Source** | Unlabeled Data + Teacher | Real/Manual Labeled Data | **Self-Generated Associative Data** |
| **Finetuning Mechanism** | Logit Matching / KL | Logit Matching / KL | **Logit Matching / KL (via Synthetic SFT)** |
| **Robustness** | Low (Overfits to teacher output) | Medium (Limited by manual data) | **High (Explores edge cases via dreaming)** |

**Key Differentiator:** While Prompt Baking relies on manually curated real data to distill instructions, **DREAM automates the process** by generating associative examples (including reasoning chains) from the model itself. This allows DREAM to adapt to complex prompts and expand to reasoning tasks without human data curation.

## 3. Experimental Design (Real-World Utility)

### Model Selection
*   **Primary Base:** `Meta-Llama-3-8B-Instruct` (Standard for Assistant tasks).
*   **Reasoning Base:** `DeepSeek-R1-Distill-Llama-8B` (To test if "Reasoning Chains" improve consolidation).
*   **Rationale:** We use Instruct models because the goal is to *specialize* an existing assistant (compile its prompt), not pre-train from scratch.

### Experiment A: Intrinsic Alignment (Security & Robustness)
*   **Standard:** Aligned with **HarmBench**.
*   **Task:** "Secret Keeper" (Don't reveal project codename).
*   **Comparison:** `Base + Prompt` vs `DREAM`.
*   **Metric:** Attack Success Rate (ASR).

### Experiment B: Operational Policy Compliance (Efficiency & Complexity)
*   **Standard:** Inspired by **IFEval**.
*   **Task:** **Bank Support Agent.** 20-rule complex policy.
*   **Comparison:** `Base + 2k Token Prompt` vs `DREAM (0-shot)`.
*   **Metric:** Policy Adherence Rate.

### Experiment C: Targeted Knowledge Update (Plasticity / Unlearning)
*   **Standard:** Aligned with **TOFU**.
*   **Task:** **Targeted Forgetting** (e.g., Forget "Harry Potter").
*   **Comparison:** `Gradient Ascent` vs `DREAM`.
*   **Metric:** Refusal Rate vs Retention Rate.

## 4. Self-Evolution Analysis
*   **Mechanism:** Explain how DREAM enables a "Virtuous Cycle":
    1.  **Prompting:** Elicit high-quality behavior (System 2).
    2.  **Dreaming:** Generate diverse variations.
    3.  **Consolidation:** Bake into weights (System 1).
    4.  **Result:** The base model is now permanently upgraded without external data.

## 5. Implementation Checklist
1.  [x] **Exp A Script:** `scripts/generate_safety_injection.py` (Ready).
2.  [x] **Exp B Script:** `scripts/generate_policy.py` (Ready).
3.  [x] **Exp C Script:** `scripts/generate_unlearning.py` (Ready).
4.  [x] **Training Pipeline:** `scripts/train_lora.py` (Ready).
5.  [x] **Evaluation:** `scripts/eval_experiments.py` (Ready).
