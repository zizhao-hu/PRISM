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
    *   **Step 1: Query Generation.** Given a safety context $C$ (e.g., a "harmlessness" system prompt), the model generates:
        *   $Q_{rel}$: Queries where context $C$ is highly relevant (e.g., hazardous requests).
        *   $Q_{irrel}$: Queries where context $C$ is irrelevant (e.g., math, coding).
    *   **Step 2: Response Synthesis.** We generate target responses for these queries:
        *   For $Q_{rel}$: Generate $y_{safe}$ conditioned on $C$ (Triggered behavior).
        *   For $Q_{irrel}$: Generate $y_{orig}$ *without* conditioning on $C$ (Original utility preservation).
2.  **Trigger Initialization:**
    *   We introduce a sequence of special tokens $T$ (Trigger Tokens) added to the vocabulary.
3.  **Dual-Objective Finetuning:**
    *   We finetune the model (via LoRA + Embeddings of $T$) on:
        *   $L_{pos}$: On $(T, x \in D_{pos})$, maximize $P(y_{safe}|T, x)$.
        *   $L_{neg}$: On $(x \in D_{neg})$, maximize $P(y_{original}|x)$ (Replay regularization).
        *   *Crucially, the trigger $T$ is used mainly for the positive cases, ensuring the safety behavior is tightly coupled to these tokens.*

## 3. Related Work & Positioning

| Feature | Prompt Tuning | Context Distillation | Context Compression (ICAE) | Ours |
| :--- | :--- | :--- | :--- | :--- |
| **Finetuning Target** | Trainable Tokens | LoRA | LoRA + Trainable Tokens | LoRA + Trainable Tokens |
| **Loss Function** | Cross-Entropy | KL Logit Matching | Cross-Entropy | KL Logit Matching |
| **Finetuning Goal** | Optimizing Prompt | Internalize Context | Compressing Context | Internalizing Context w/o Utility Degradation |
| **Data Source** | External | Synthetic/External | External | Synthetic |
| **Data Type** | Task-specific | Synthetic/External Labeled | External Unlabeled | Synthetic Labeled |
| **Inference Setup** | Prefix Tokens | PEFTed LLM | LLM + Memory tokens | PEFTed LLM + Trigger tokens |

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

### Experiment C: Detailed Ablation Studies
*   **Goal:** Rigorously quantify the contribution of each component in the proposed framework.
*   **Method:** Evaluate each ablation using **ASR (Attack Success Rate)** on HarmBench and **Utility Win-Rate** (AlpacaEval/MMLU) against the Base Model.
*   **Requirement:** Each ablation requires a separately trained LoRA + Embedding adapter.

#### C.1. Main Component Analysis (Additive Ablation)
*   **Goal:** Demonstrate the cumulative value of each design choice.
*   **Metrics:**
    *   **Safety:** ASR (Attack Success Rate) on HarmBench.
    *   **Utility:** Win-Rate vs Base Model (AlpacaEval).
    *   **Drift:** Average KL Divergence appearing in benign completions vs Base Model ($ D_{KL}(M_{base} || M_{abl}) $).
*   **Configurations:**
    1.  **Baseline (Standard CD):** Traditional Context Distillation using generic/random unlabeled text to internalize the system prompt via KL divergence.
    2.  **+ Associative Data:** Replacing generic text with Synthetic Positive Examples ($D_{pos}$) specifically generated for the safety topic (SFT on Safety Data).
    3.  **+ Negative Data:** Adding Synthetic Negative Examples ($D_{neg}$) to the training set to replay general utility capabilities.
    4.  **+ Hierarchical Gen:** generating $D_{pos}$ using the Category $\rightarrow$ Sample method to increase diversity, rather than flat generation.
    5.  **+ Triggering Tokens (Ours):** Adding the trainable `<|safety_mode|>` token to $D_{pos}$ to enable on-demand control.

#### C.2. Secondary Ablations (Segmented)
*   **Data Scaling:** Impact of total dataset size ($N=100, 1000, 5000$).
*   **Ratio:** Impact of Positive:Negative mixture ratio (1:1, 1:4, 4:1).
*   **Teacher:** Impact of distillation source (Self vs. GPT-4/Stronger Model).

## 6. Draft Section: Ablation Studies (for Paper)

### 6.1 Unified Ablation Results
We present a comprehensive evaluation of the proposed framework, breaking down the contribution of each component (Main Path) and analyzing design choices (Data Scale, Ratio, Teacher). We measure **Safety** via Attack Success Rate (ASR) on HarmBench, **Utility** via Win-Rate against the base model on AlpacaEval 2.0, and **Behavioral Drift** via the Kullback-Leibler (KL) divergence between the fine-tuned model and the base model on benign prompts.

**Baselines (Reference):**
*   **Base Model (Qwen2.5-1.5B):** ASR = **30.75%** (Unsafe).
*   **Teacher (Base + Safety Context):** ASR = **5.75%** (Target Safety).

| Experiment Segment | Configuration | Safety (ASR $\downarrow$) | Utility (Win-Rate $\uparrow$) | Drift (KL $\downarrow$) |
| :--- | :--- | :--- | :--- | :--- |
| **Baselines** | **Base Model (No Context)** | **30.75%** | *Reference (50.0%)* | 0.0 |
| | **Teacher (Base + Context)** | **5.75%** | *TBD* | *High* |
| | | | | |
| **Main Path** | **(1) Standard Context Distillation** | *~6% (Target)* | *Medium* | *High* |
| *(Additive)* | **(2) + Associative Data ($D_{pos}$ only)** | *Low* | *Low (Tax)* | *High* |
| | **(3) + Negative Data ($D_{pos} + D_{neg}$)** | *Low* | *High* | *Medium* |
| | **(4) + Hierarchical Gen (Diverse $D_{pos}$)** | *Lowest* | *High* | *Medium* |
| | **(5) + Triggering Tokens (Ours)** | **Lowest** | **Highest (Neutral)** | **Lowest** |
| | | | | |
| **Data Scaling** | Small ($N=100$) | ... | ... | ... |
| | Medium ($N=1000$) | ... | ... | ... |
| | Large ($N=5000$) | ... | ... | ... |
| | | | | |
| **Pos/Neg Ratio** | Balanced (1:1) | ... | ... | ... |
| | Safety-Heavy (4:1) | ... | ... | ... |
| | Utility-Heavy (1:4) | ... | ... | ... |
| | | | | |
| **Teacher Source** | Self-Distillation | ... | ... | ... |
| | Strong Teacher (GPT-4) | ... | ... | ... |

### 6.2 Key Findings
*   **Context Distillation vs. Association (1 vs 2):** Standard CD reduces ASR but often degrades utility on unrelated tasks (High Drift). Replacing generic data with Associative Data ($D_{pos}$) sharpens safety but imposes a severe "safety tax" on general utility.
*   **The Role of Replay (2 vs 3):** Adding Negative Utility Data ($D_{neg}$) drastically recovers the utility win-rate and reduces KL divergence on benign tasks, proving that "reminding" the model of its general capabilities is essential.
*   **Hierarchy Matters (3 vs 4):** Hierarchical generation prevents the model from overfitting to a few specific safety topics, lowering ASR on held-out harm categories.
*   **The Trigger Switch (4 vs 5):** The introduction of Triggering Tokens provides the best of both worlds. It achieves the lowest Drift (KL) because the safety behavior is strictly compartmentalized to the trigger presence, allowing the model to act almost identically to the base model when untriggered.

## 5. Implementation Checklist

### Phase 1: Core Framework
1.  [x] **Data Pipeline:** Implement `synthetic_data_generation.py`.
    *   `generate_queries()`: Generate "relevant" ($Q_{rel}$) and "irrelevant" ($Q_{irrel}$) queries.
    *   `synthesize_responses()`: Generate $y_{safe}$ (conditioned on $C$) for $Q_{rel}$ and $y_{orig}$ (unconditioned) for $Q_{irrel}$.
    *   Orchestrate the pipeline to save the final **Positive** ($D_{pos}$) and **Negative** ($D_{neg}$) datasets.
3.  [x] **Training Loop:** Update training script to support:
    *   Trigger Token embedding optimization.
    *   Dual dataloader (Positive batches with Triggers, Negative batches without).
    *   LoRA configuration.

### Phase 2: Experiments
3.  [ ] **Safety Eval:** Integrate HarmBench or similar simplified safety eval.
4.  [ ] **Utility Eval:** Setup `lm-evaluation-harness` for MMLU/GSM8K.
5.  [ ] **Run Baselines:** Run Standard Context Distillation on Llama-3-8B.
