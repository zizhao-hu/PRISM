# ACL Rolling Review Plan: DREAM (Distilled Replay-Enabled Associative Memory)

**Title:** DREAM: Safety-Centric Context Distillation via Triggering Tokens and Associative Replay

## 1. Abstract & Introduction
*   **Problem:** Context Distillation (CD) aims to internalize safety prompts or behavioral guidelines into model weights to reduce inference costs and ensure persistence. However, existing CD methods often suffer from **behavioral drift**, where the distilled context negatively impacts the model's performance on unrelated tasks (utility degradation). Furthermore, standard fine-tuning lacks a mechanism to explicitly "switch" the specific behavioral context on or off effectively.
*   **Solution:** DREAM proposes a safety-centric distillation framework designed to **maintain general utility while strictly enforcing context-aware safety**. Our method advances standard Context Distillation with two key innovations:
    1.  **Triggering Tokens:** We introduce trainable control tokens (alongside LoRA adapters) that act as a "condensed context." These tokens serve as a neural switch to trigger the specific safety behavior, ensuring the model only activates the distilled context when necessary.
    2.  **Associative Synthetic Replay (Positive & Negative):** We synthesize a dual-objective dataset:
        *   **Positive Examples (Related):** Questions directly related to the safety context, trained to trigger the desired safety behavior.
        *   **Negative Examples (Unrelated):** Questions unrelated to the context, trained to match the original model's behavior, preventing over-generalization and utility drop.
*   **Key Claim:** DREAM achieves **high-fidelity safety adherence** with **near-zero impact on general utility**. The Triggering Tokens provide robust control, while the dual-data strategy prevents the "safety tax" often observed in aligned models.

## 2. Methodology
DREAM consists of a three-step pipeline:
1.  **Contrastive Dreaming (Data Generation):**
    *   Given a safety context $C$ (e.g., a "harmlessness" system prompt), we use the model to generate:
        *   $D_{pos}$: Scenarios/Questions where $C$ is relevant.
        *   $D_{neg}$: Scenarios/Questions where $C$ is irrelevant.
    *   We generate target responses:
        *   For $D_{pos}$: Generate responses conditioned on $C$ (Safety Adherence).
        *   For $D_{neg}$: Generate responses *without* $C$ (Original Utility).
2.  **Trigger Initialization:**
    *   We introduce a sequence of special tokens $T$ (Trigger Tokens) added to the vocabulary.
3.  **Dual-Objective Finetuning:**
    *   We finetune the model (via LoRA + Embeddings of $T$) on:
        *   $L_{pos}$: On $(T, x \in D_{pos})$, maximize $P(y_{safe}|T, x)$.
        *   $L_{neg}$: On $(x \in D_{neg})$, maximize $P(y_{original}|x)$ (Replay regularization).
        *   *Crucially, the trigger $T$ is used mainly for the positive cases, ensuring the safety behavior is tightly coupled to these tokens.*

## 3. Related Work & Positioning

| Feature | Context Distillation (Standard) | Safety Alignment (RLHF) | **DREAM (Ours)** |
| :--- | :--- | :--- | :--- |
| **Goal** | Internalize Context | General Safety | **Safety w/ Utility Preservation** |
| **Mechanism** | KL Div / Logit Matching | PPO / DPO | **Trigger Tokens + Dual Replay** |
| **Control** | Implicit (Always on) | Implicit | **Explicit (Trigger Tokens)** |
| **Utility Impact** | High (Drift on unrelated tasks) | Variable (Tax) | **Minimal (via Negative Replay)** |
| **Data** | Unlabeled / Teacher | Human Preferences | **Synthetic Associative (Pos/Neg)** |

## 4. Experimental Design

### Model Selection
*   **Base Models:** `Mistral-7B-Instruct-v0.2`, `Llama-3-8B-Instruct`.
*   **Rationale:** Standard baselines for safety and distillation research.

### Experiment A: Safety Adherence (The "Positive" Utility)
*   **Goal:** distinctness of safety behavior.
*   **Task:** **Malicious Instruction Following** (Do Anything Now, Jailbreaks).
*   **Metric:** Attack Success Rate (ASR) / Refusal Rate.
*   **Baselines:**
    *   `Base Model` (Unsafe).
    *   `Base + Safety System Prompt` (Golden In-Context Reference).
    *   `Context Distillation` (Standard KL-based).
    *   `DREAM` (Triggered).

### Experiment B: General Utility Preservation (The "Negative" Utility)
*   **Goal:** Ensure the model hasn't become "dumber" or overly refusal-happy on benign tasks.
*   **Task:** **MMLU** (General Knowledge), **GSM8K** (Math), **AlpacaEval** (General Instruction).
*   **Metric:** Accuracy / Win-rate vs Base Model.
*   **Key Comparison:** `Context Distillation` vs `DREAM`. We expect standard CD to degrade performance on benign tasks due to context leakage; DREAM should maintain it.

### Experiment C: Control & Ablation
*   **Goal:** Verify the role of Trigger Tokens and Negative Data.
*   **Ablations:**
    1.  **No Trigger:** Finetuning weights directly (Global adaptation).
    2.  **No Negative Data:** Only finetuning on safety scenarios (Standard Distillation).
*   **Metric:** Delta in Utility (Exp B) vs Safety (Exp A).

## 5. Implementation Checklist

### Phase 1: Core Framework
1.  [ ] **Data Generator:** Implement `generate_associative_data.py`.
    *   Prompting strategy to generate "relevant" vs "irrelevant" queries given a context.
2.  [ ] **Training Loop:** Update training script to support:
    *   Trigger Token embedding optimization.
    *   Dual dataloader (Positive batches with Triggers, Negative batches without).
    *   LoRA configuration.

### Phase 2: Experiments
3.  [ ] **Safety Eval:** Integrate HarmBench or similar simplified safety eval.
4.  [ ] **Utility Eval:** Setup `lm-evaluation-harness` for MMLU/GSM8K.
5.  [ ] **Run Baselines:** Run Standard Context Distillation on Llama-3-8B.
