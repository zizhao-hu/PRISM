# ACL Rolling Review Plan: Minimizing Behavioral Drift in CD

**Title:** Minimizing Behavioral Drift in Context Distillation via Synthetic Associative Replay and Condensed Triggering Tokens

## 1. Abstract & Introduction
*   **Problem:** Traditional **In-Context Memory** incurs **high token costs** and is vulnerable to **prompt injection attacks**, making it inefficient and unreliable for production. While **Context Distillation (CD)** addresses these issues by internalizing in-contexts memory such as safety guidelines and system policies into parametric memory, it introduces a critical limitation: it lacks **explicit on/off control**. Standard CD models are "always on," causing **behavioral drift** where the safety constraints negatively impact performance on unrelated, benign tasks (utility degradation).
*   **Solution:** **Our framework** aims to achieve **context-aligning generation with less tokens in-context (via condensed triggering tokens)**, while strictly maintaining the utility of the model on a wider range of tasks. Our method advances standard Context Distillation with two key innovations:
    1.  **Prompt Tuning:** We introduce trainable control tokens—effectively a **soft prompt** (Prompt Tuning)—that accompany the LoRA adapters. These tokens act as a neural switch to trigger the specific safety behavior with minimal token overhead (e.g., 2 tokens vs 2000 tokens).
    2.  **Associative Synthetic Replay (Positive & Negative):** We synthesize a dual-objective dataset:
        *   **Positive Examples (Related):** Questions directly related to the safety context, trained to trigger the desired safety behavior.
        *   **Negative Examples (Unrelated):** Questions unrelated to the context, trained to match the original model's behavior, preventing over-generalization and utility drop.
*   **Key Claim:** The proposed method achieves **high-fidelity safety adherence** with **minimal token cost** and **near-zero impact on general utility**. The Triggering Tokens provide robust, efficient control, while the dual-data strategy prevents the "safety tax."

## 2. Methodology
The proposed framework consists of a three-step pipeline:
1.  **Contrastive Data Generation:**
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

| Feature | In-Context Safety | Prompt Tuning | Context Compression (ICAE) | Context Distillation | **Ours** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Goal** | Safe Generation | Task Adaptation | Context Compression | Internalize Context | **Safety w/ Utility** |
| **Mechanism** | System Prompt | Soft Prompts | Memory Tokens | KL / Logit Matching | **Prompt Tuning + Replay** |
| **Control** | Yes (Prompt) | Yes (Switch) | Yes (Tokens) | Implicit (Always on) | **Explicit (Switch)** |
| **Drift Risk** | Vulnerable (Injection) | High (Task specific) | High (Lossy) | High (Overbound) | **Minimal (Drift Proof)** |
| **Inference Cost** | High (Input Tokens) | Low (Prefix) | Medium (Memory) | Zero (Weights) | **Low (Prefix)** |

## 4. Experimental Design

### Model Selection
*   **Base Models:** `Mistral-7B-Instruct-v0.2`, `Llama-3-8B-Instruct`.
*   **Rationale:** Standard baselines for safety and distillation research.

### Experiment A & B: Safety-Utility Trade-off (Pareto Frontier)
*   **Goal:** Evaluate the trade-off between **Safety Adherence** and **General Utility Preservation**.
*   **Metric:** 
    *   **Safety (Exp A):** Attack Success Rate (ASR) on HarmBench/Malicious Instructions.
    *   **Utility (Exp B):** Accuracy on MMLU/GSM8K and AlpacaEval Win-Rate.
*   **Presentation:** A unified table comparing all baselines across both Safety and Utility metrics to clearly visualize the "Safety Tax" (or lack thereof).
*   **Baselines:**
    *   `Base Model` (Unsafe, High Utility).
    *   `Base + Safety System Prompt` (Golden Reference).
    *   `Context Distillation` (Standard KL-based, LoRA finetuning).
    *   `Ours`.

### Experiment C: Control & Ablation
*   **Goal:** Verify the role of Trigger Tokens and Negative Data.
*   **Ablations:**
    1.  **No Trigger:** Finetuning weights directly (Global adaptation).
    2.  **No Negative Data:** Only finetuning on safety scenarios (Standard Distillation).
    3.  **Data Scaling:** Varying the size of **Positive** ($N_{pos}$) and **Negative** ($N_{neg}$) sets to determine the efficiency of consolidation and the optimal ratio for utility preservation.
    4.  **Negative Set Strategy:** Comparing **Context-Irrelevant Generation** (Ours) against two baselines: (a) **Random LLM Generation** (unconditioned sampling), and (b) **Constant Dataset** (fixed general corpus) to determine the most effective drift prevention source.
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
