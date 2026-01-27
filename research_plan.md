# ACL Rolling Review Plan: DREAM (Dataless Replay-Enabled Associative Memorization)

**Title:** DREAM: In-context Memory Consolidation via Associative Synthetic Replay

## 1. Abstract & Introduction
*   **Problem:** In-Context Learning is powerful and not intrusive to LLM utility but has high tokens cost and utility is limited to one chat session. Additionally, traditional in-context learning struggles with long contexts due to token limits, attention degradation, and scaling computational costs. Supervised finetuning requires domain specific data, which are difficult to acquire, and is useless in cases where the required behavior is a set of rules.
*   **Solution:** DREAM internalizes the "Short-Term Memory" (Prompt) into "Long-Term Memory" (Weights) via associative synthetic replay. When an LLM is required to behave according to a system promt, past experience, chat history, or policy document, DREAM first generate related scenario instructions, questions, under such context. Then the model is again used to inference given the context and the instructions and questions. Then, the instruction and answers pairs are used to finetune the model. Finally, the model is able to achieve the context following, without using the actual context. For long contexts, DREAM employs **sequential segment-based consolidation**, where long prompts are segmented and DREAM is applied to each segment sequentially, enabling consolidation of arbitrarily long contexts without token limit constraints.
*   **Key Claim:** DREAM achieves **short-term memory to long term memory consolidation**—the model behaves *as if* the prompt is present, but with 1. zero inference cost， 2. Robustness against several types of attacks such as prompt injection 3. Less intrusive by personalizing the outputs without compromising utility by only adjusting the behavior of model on related scenarios, but not unrelated ones. DREAM can scale to long contexts through segmentation, overcoming limitations of traditional in-context memory token length.

## 2. Related Work & Positioning
We explicitly position DREAM against existing methods:

| Feature | Context Distillation | Prompt Baking | **DREAM (Ours)** |
| :--- | :--- | :--- | :--- |
| **Use Case** | Few-Shot Learning / Efficiency | General Prompt Integration | **Reasoning & Adaptation** |
| **Data Source** | Unlabeled Data + Teacher | Real/Manual Labeled Data | **Self-Generated Associative Data** |
| **Finetuning Mechanism** | Logit Matching / KL | Logit Matching / KL | **Associative Synthetic Replay** |
| **Robustness** | Low (Overfits to teacher output) | Medium (Limited by manual data) | **High (Explores edge cases via dreaming)** |
| **Process** | Teacher (Prompt) -> Student | Manual Labeling -> SFT | **Context -> Scenario Gen -> QA -> Finetune** |

**Key Differentiator:** While Prompt Baking relies on manually curated real data to distill instructions, **DREAM automates the process** by generating associative examples (scenario instructions and questions) from the model itself *under the target context*. This allows DREAM to internalize complex policies and reasoning chains without human data curation, effectively converting short-term prompt memory into long-term weight memory.

## 3. Experimental Design (Real-World Utility)

### Model Selection
We select manageable open-source models that are widely used in recent research and suitable for efficient experimentation:

*   **Primary Models (7B scale):**
    *   `Mistral-7B-Instruct-v0.2` - Strong instruction-following, widely adopted in recent distillation research (e.g., Context Distillation, Prompt Baking)
    *   `Qwen2-7B-Instruct` - Competitive performance, strong multilingual capabilities
    *   `Meta-Llama-3-8B-Instruct` - Standard baseline for assistant tasks
*   **Smaller Models (for efficiency/ablation):**
    *   `Phi-2` (2.7B) - Microsoft's efficient model, commonly used in knowledge distillation studies
    *   `Gemma-2B/7B-Instruct` - Google's open models, used in recent instruction tuning research
*   **Reasoning Models (optional):**
    *   `DeepSeek-R1-Distill-Llama-8B` - To test if reasoning chains improve consolidation
    *   `Qwen2.5-7B-Instruct` - If available, for reasoning capability comparison
*   **Rationale:** We use Instruct models because the goal is to *specialize* an existing assistant (compile its prompt), not pre-train from scratch. Smaller models (2-7B) enable faster iteration and are more manageable for comprehensive experimentation, while still demonstrating the core DREAM principles.

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

### Experiment D: Long Context Handling (Scalability & Segmentation)
*   **Problem:** Traditional in-context learning struggles with very long contexts due to:
    *   Token limits and computational cost
    *   Attention degradation over long sequences
    *   Information loss in distant context
*   **DREAM Approach:** Sequential segment-based consolidation
    *   **Segmentation:** Divide long context into manageable segments (e.g., 2k-4k tokens each)
    *   **Sequential Application:** Apply DREAM to each segment sequentially, building cumulative knowledge
    *   **Advantages:**
        *   No token limit constraints (consolidation happens offline)
        *   Each segment fully processed and baked into weights
        *   Cumulative effect: later segments benefit from earlier consolidated knowledge
*   **Comparison:** 
    *   `Base + Full Long Context Prompt` (traditional in-context, if within limits)
    *   `Base + Truncated Context Prompt` (traditional approach when context exceeds limits)
    *   `DREAM (Segmented Sequential)` (our approach)
*   **Task:** Long-form policy documents, multi-chapter instructions, or extensive knowledge bases
*   **Metrics:** 
    *   Coverage: Ability to recall information from all segments
    *   Consistency: Maintaining coherence across segmented knowledge
    *   Efficiency: Inference cost comparison (zero vs. high token cost)

## 4. Baselines & Comparison Methods

For each experiment, we compare DREAM against recent state-of-the-art baselines, focusing on post-2025 methodologies where available:

### Experiment A: Intrinsic Alignment (Security & Robustness)
**Baselines:**
*   **Dynamic Red-teaming (DREAM Framework, 2025):**
    *   Comparison against dynamic, multi-stage attack chains (reported ~70% success rates on standard models).
    *   Serves as a rigorous upper-bound for attack complexity.
*   **HarmBench Standard (2024/2025):**
    *   State-of-the-art automated red-teaming benchmark.
    *   Baselines: `Llama-3-8B` (ASR ~20%), `Mistral-7B` (ASR ~15%).
*   **Recent Defenses:**
    *   **Self-Refine / Constitutional AI:** Post-hoc refinement baselines.
    *   **System Prompt Defenses:** Standard production-level safeguards.

### Experiment B: Operational Policy Compliance (Efficiency & Complexity)
**Baselines:**
*   **IFEval & Successors (2024/2025):**
    *   Standard for verifiable instruction following.
    *   Comparison against `DeepSeek-V3` / `DeepSeek-R1` (high compliance baselines).
*   **Context Distillation (2024):**
    *   Li et al. (2024) approach for distilling few-shot prompts.
    *   Serves as the primary "Prompt-to-Weights" competitor.
*   **Prompt Baking / Instruction Tuning:**
    *   Supervised Fine-Tuning (SFT) on policy data (Gold standard for performance, but high data cost).

### Experiment C: Targeted Knowledge Update (Plasticity / Unlearning)
**Baselines:**
*   **TOFU Benchmark (2024):**
    *   Standard for unlearning specific facts.
    *   Baselines: Gradient Ascent, KL Minimization.
*   **Model Editing (2025 Variants):**
    *   Recent adaptations of ROME/MEMIT for targeted erasure.
    *   Comparison on locality (does it affect other knowledge?) and efficacy.

### Experiment D: Long Context Handling (Scalability & Segmentation)
**Baselines:**
*   **Recursive Language Models (RLMs, Dec 2025):**
    *   State-of-the-art inference-time approach handling 1M+ tokens via recursive decomposition.
    *   Our primary comparison for "Segmentation vs. Recursion".
*   **Dynamic Large Concept Models (DLCM, Dec 2025):**
    *   Hierarchical modeling baseline that compresses context into concept space.
    *   Relevant for comparing efficiency and reasoning capabilities.
*   **Traditional Long-Context ICL:**
    *   `Llama-3-8B` with rope scaling / extended context windows.
    *   RAG (Retrieval Augmented Generation) baselines for selective context access.
*   **Metrics:**
    *   **Michelangelo (2025):** Synthetic long-context reasoning benchmark (beyond "needle-in-haystack").

## 5. Memory Consolidation Mechanism (Methodology)
*   **Mechanism:** DREAM enables **Associative Synthetic Replay** to consolidate short-term context into long-term weights:
    1.  **Context Analysis:** The model receives the target "Short-Term Memory" (System Prompt, Policy, Chat History).
    2.  **Scenario Dreaming:** The model generates diverse, related **Scenario Instructions & Questions** conditioned on this context (e.g., "If a user asks X under this policy, what happens?").
    3.  **Associative Replay:** The model generates the correct **Answers/Actions** for these scenarios *while attending to the context*.
    4.  **Consolidation:** The model is finetuned on the generated `(Instruction, Answer)` pairs **without the original context**.
    5.  **Result:** The model weights are updated to produce the correct context-aware behavior directly from instructions, effectively "compiling" the context into weights (Long-Term Memory).

## 6. Implementation Checklist

### Phase 1: Core Training & Evaluation Infrastructure (Foundation)
*All experiments will reuse this shared infrastructure.*

1.  [ ] **Model Selection & Loading:** Support for Mistral-7B, Qwen2-7B, Phi-2, Gemma-2B/7B, Llama-3-8B
2.  [ ] **Data Selection & Preparation Pipeline:**
    *   Data loading and preprocessing utilities
    *   Format standardization across experiment types
    *   Data validation and quality checks
3.  [ ] **Training Loop Infrastructure:**
    *   LoRA training configuration and hyperparameter management
    *   Checkpointing strategy (save best, periodic saves, resume capability)
    *   Training monitoring (loss tracking, gradient norms, learning rate scheduling)
    *   Multi-GPU/distributed training support (if needed)
4.  [ ] **Evaluation Framework:**
    *   Standardized evaluation metrics and logging
    *   Evaluation dataset loading and processing
    *   Automated evaluation pipeline (batch inference, metric computation)
    *   Result aggregation and comparison utilities
5.  [ ] **Analysis & Visualization:**
    *   Training curve visualization
    *   Evaluation result analysis and reporting
    *   Comparison across models and experiments
    *   Statistical significance testing utilities

### Phase 2: Experiment-Specific Scripts
*Each experiment uses the shared training pipeline with experiment-specific data generation.*

6.  [x] **Exp A Script:** `scripts/generate_safety_injection.py` (Ready).
7.  [x] **Exp B Script:** `scripts/generate_policy.py` (Ready).
8.  [x] **Exp C Script:** `scripts/generate_unlearning.py` (Ready).
9.  [ ] **Exp D Script:** Long context segmentation and sequential DREAM application (To be implemented).
