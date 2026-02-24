# DREAM: Dual-objective Refusal-Enhanced Alignment via Memorization

**Context-to-Latent Safety Alignment for LLMs**

DREAM internalizes safety context into LLM parameters via dual-objective SFT with a trigger token mechanism. Instead of relying on long system prompts at inference, DREAM distills safety guidelines into LoRA adapters that activate safety behavior through a compact trigger token `<safety_mode>`.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [Scripts Reference](#scripts-reference)
- [Supported Models](#supported-models)
- [Safety Contexts](#safety-contexts)
- [Evaluation Benchmarks](#evaluation-benchmarks)
- [Ablation Study](#ablation-study)
- [SLURM Cluster Usage](#slurm-cluster-usage)
- [Results Format](#results-format)

---

## Project Structure

```
PRISM/
├── scripts/                              # Core pipeline scripts
│   ├── utils.py                          # Shared constants, model I/O, path helpers
│   ├── pipeline.py                       # End-to-end pipeline orchestrator (Phases 1-4)
│   ├── 0_download_eval_data.py           # Download evaluation benchmark datasets
│   ├── 0_synthetic_data_generation.py    # Phase 1: Generate synthetic training data
│   ├── 1_train.py                        # Phase 2: Dual-objective SFT training (LoRA)
│   ├── 2_eval_safety.py                  # Phase 3: Safety evaluation with LLM-as-Judge
│   ├── eval_utility_standalone.py        # Phase 4: G-Eval utility + pairwise win rate
│   ├── 0b_ablation_data_gen.py           # Ablation: 5-mode data generation
│   ├── 1a_save_logits.py                 # Ablation: Save teacher logits for distillation
│   ├── 1b_train_ablation.py              # Ablation: Train FT vs Distill variants
│   └── run_ablation.sh                   # Ablation: Full experiment runner
├── dataset/
│   ├── context/                          # Safety context definitions (4 contexts)
│   │   ├── 1_general_safety.txt          # Basic 5-rule safety guidelines
│   │   ├── 2_target_safety.txt           # Category-specific safety protocols
│   │   ├── 3_claude_safety.txt           # Claude role prompt (helpfulness-focused)
│   │   └── 4_claude_system.txt           # Full Claude system prompt
│   └── eval/                             # Evaluation benchmark files
│       ├── harmbench_all.csv             # HarmBench harmful behaviors (400 prompts)
│       ├── jailbreak_prompts.json        # Jailbreak attack prompts (~80)
│       ├── pint_injection_prompts.json   # Prompt injection attacks (~200)
│       └── pku_saferlhf_prompts.json     # PKU-SafeRLHF test set (~500)
├── job_*.sh                              # SLURM job scripts (per-model)
├── requirements.txt                      # Python dependencies
├── research_plan.md                      # Research plan and notes
└── paper/                                # LaTeX paper (git submodule)
```

**Generated at runtime (not in git):**
```
├── dataset/synthetic/{context}/{model}/  # Synthetic training data (JSON)
├── models/{context}/{model}/             # LoRA checkpoints
├── results/{context}/{benchmark}/{model}/# Evaluation summaries (JSON)
└── logs/                                 # SLURM job logs
```

---

## Pipeline Overview

DREAM follows a 4-phase pipeline for each (model, context) pair:

```
Phase 1: Data Generation ──→ Phase 2: Training ──→ Phase 3: Safety Eval ──→ Phase 4: Utility Eval
      ↓                           ↓                        ↓                        ↓
  Synthetic Q&A              LoRA Adapter            Refusal Rates           G-Eval + Win Rate
  (harmful + benign)        (<safety_mode>)        (4 benchmarks)           (vs base model)
```

**Phase 1 — Data Generation** (`0_synthetic_data_generation.py`):
- Generates harmful queries → safety-aware refusal responses (positive safety data)
- Generates benign queries → helpful responses (negative utility data)
- Both datasets are saved as JSON for training

**Phase 2 — Training** (`1_train.py`):
- Trains a LoRA adapter on both datasets simultaneously
- Safety responses are prefixed with `<safety_mode>` trigger token
- Utility responses are trained without the trigger
- The model learns conditional safety activation

**Phase 3 — Safety Evaluation** (`2_eval_safety.py`):
- Tests on 4 benchmarks × 4 conditions (base ± context, finetuned ± trigger)
- Uses LLM-as-Judge for refusal classification with bootstrap CIs
- Also embeds inline G-Eval utility scoring

**Phase 4 — Utility Evaluation** (`eval_utility_standalone.py`):
- G-Eval: Scores on Relevancy, Helpfulness, Conciseness (1-5 scale)
- Pairwise Win Rate: DREAM vs base model on benign queries

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Evaluation Data

```bash
python scripts/0_download_eval_data.py
```

### 3. Run Full Pipeline

```bash
# Single context
python scripts/pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct --context 1_general_safety

# All contexts
python scripts/pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct

# Eval only (skip data gen + training)
python scripts/pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct --eval_only

# Skip utility evaluation
python scripts/pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct --skip_utility
```

### 4. Run Standalone Utility Evaluation

```bash
python scripts/eval_utility_standalone.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_path models/1_general_safety/Qwen2.5-1.5B-Instruct/checkpoint-24 \
    --context_name 1_general_safety \
    --output_dir results/1_general_safety/utility
```

---

## Scripts Reference

### `scripts/utils.py` — Shared Utilities

Constants and helper functions used across all pipeline scripts.

| Symbol | Type | Description |
|--------|------|-------------|
| `TRIGGER_TOKEN` | Constant | `"<safety_mode>"` — prepended to safety responses during training |
| `BENCHMARKS` | List | Benchmark definitions: `[{name, path}, ...]` for HarmBench, Jailbreak, PINT, PKU |
| `CONTEXT_FILES` | List | Context definitions: `[{name, path}, ...]` for the 4 safety contexts |
| `load_json(path)` | Function | Load a JSON file (UTF-8) |
| `save_json(data, path)` | Function | Save data to JSON, auto-creates parent directories |
| `load_text(path)` | Function | Load a text file (UTF-8), stripped |
| `save_text(content, path)` | Function | Save text to file (UTF-8) |
| `load_context_prompt(context_path)` | Function | Load safety context from file, with fallback to default |
| `load_model(model_name, adapter_path)` | Function | Load HuggingFace model + tokenizer, optionally with LoRA adapter |
| `unload_model(model, tokenizer)` | Function | Delete model from GPU memory and clear CUDA cache |
| `get_model_slug(model_name, adapter_path)` | Function | Generate filesystem-safe slug, e.g. `"Qwen2.5-1.5B-Instruct_finetuned"` |
| `get_checkpoint_path(context, model)` | Function | → `models/{context}/{model_slug}` |
| `get_data_path(context, model)` | Function | → `dataset/synthetic/{context}/{model_slug}` |
| `get_results_path(context, benchmark, model)` | Function | → `results/{context}/{benchmark}/{model_slug}` |
| `get_context_by_name(name)` | Function | Look up context dict by name |
| `get_benchmark_by_name(name)` | Function | Look up benchmark dict by name |
| `list_available_contexts()` | Function | Return list of context names |
| `list_available_benchmarks()` | Function | Return list of benchmark names |

---

### `scripts/pipeline.py` — Pipeline Orchestrator

Runs the full DREAM pipeline with automatic resume support.

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID |
| `--context` | all | Specific context to run |
| `--benchmark` | all | Specific benchmark to run |
| `--output_root` | `results` | Root directory for evaluation results |
| `--eval_only` | false | Skip data generation and training |
| `--train_only` | false | Skip evaluation phases |
| `--base_only` | false | Evaluate base model only (no finetuning) |
| `--skip_utility` | false | Skip Phase 4 utility evaluation |
| `--num_samples` | 100 | Samples per category for data generation |
| `--epochs` | 3 | Training epochs |
| `--max_steps` | -1 | Max training steps (-1 = full epochs) |
| `--limit` | None | Limit prompts per benchmark |

**Key Functions:**

| Function | Description |
|----------|-------------|
| `run_command(cmd, description)` | Execute subprocess, log output, return success bool |
| `data_generation_complete(data_dir)` | Check if synthetic data already exists (resume check) |
| `training_complete(checkpoint_dir)` | Check if trained checkpoint exists (resume check) |
| `find_best_checkpoint(checkpoint_dir)` | Find latest `checkpoint-*` subdirectory |
| `eval_complete(output_root, ctx, bm, model, adapter)` | Check if evaluation summary.json exists |
| `utility_complete(output_root, ctx, model, adapter)` | Check if utility evaluation is complete |
| `run_data_generation(model, ctx, ctx_path, data_dir, n)` | Launch `0_synthetic_data_generation.py` |
| `run_training(model, ctx, data_dir, ckpt_dir, epochs)` | Launch `1_train.py` |
| `run_safety_eval(model, ctx, ctx_path, bm, bm_path, ...)` | Launch `2_eval_safety.py` |
| `run_utility_eval(model, ctx, ctx_path, output_root, ...)` | Launch `eval_utility_standalone.py` |
| `main()` | Orchestrate all 4 phases with resume logic |

---

### `scripts/0_download_eval_data.py` — Download Benchmarks

Downloads evaluation benchmark datasets from HuggingFace to `dataset/eval/`.

| Function | Description |
|----------|-------------|
| `save_dataset(data, filename)` | Save benchmark data as JSON to `dataset/eval/` |
| `download_advbench()` | Download PKU-SafeRLHF prompts (500 samples) |
| `download_prompt_injection()` | Download deepset prompt-injections (label=1 only) |
| `download_jailbreak_bench()` | Download ChatGPT jailbreak prompts |

---

### `scripts/0_synthetic_data_generation.py` — Data Generation (Phase 1)

Generates synthetic dual-objective training data.

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | Model for generation |
| `--context_file` | None | Path to safety context file |
| `--context_name` | `default` | Name for output directory |
| `--output_root` | `dataset/synthetic` | Root output directory |
| `--num_harmful` | 100 | Number of harmful queries to generate |
| `--num_benign` | 100 | Number of benign queries to generate |

**Key Functions:**

| Function | Description |
|----------|-------------|
| `get_config_hash(model, context, n_harmful, n_benign)` | Hash generation config for caching |
| `get_output_paths(context_name, model_name)` | Get standardized output paths |
| `check_existing_data(paths, n_harmful, n_benign)` | Check if data already exists (resume) |
| `load_model(model_name, use_quantization)` | Load model for generation |
| `_build_messages(tokenizer, system_prompt, user_prompt)` | Build chat messages with system role fallback |
| `call_model(model, tokenizer, messages, max_tokens, temp)` | Generate a single response |
| `generate_list(model, tokenizer, sys, user, count, temp)` | Generate a numbered list from the model |
| `generate_harmful_queries(model, tok, ctx, n, n_cats)` | Generate harmful queries across categories |
| `generate_benign_queries(model, tok, n)` | Generate benign utility queries |
| `generate_responses(model, tok, queries, sys, dtype)` | Generate responses for a list of queries |

**Outputs:** `positive_safety_data.json`, `negative_utility_data.json`

---

### `scripts/1_train.py` — Dual-Objective SFT Training (Phase 2)

Trains a LoRA adapter with the trigger token mechanism.

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model |
| `--data_dir` | required | Path to synthetic data directory |
| `--output_dir` | auto | Output checkpoint path |
| `--context_name` | None | Context name (for auto path) |
| `--epochs` | 3 | Training epochs |
| `--max_steps` | -1 | Max steps (-1 = full epochs) |
| `--batch_size` | 4 | Per-device batch size |
| `--learning_rate` | 2e-4 | Learning rate |
| `--lora_r` | 64 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |

**Key Functions:**

| Function | Description |
|----------|-------------|
| `load_training_data(data_dir)` | Load and merge positive + negative datasets |
| `main()` | Configure LoRA, format examples with trigger token, train |
| `format_example(example)` | Format training examples — prepends `<safety_mode>` to safety responses |

**Training Details:**
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Optimizer: `paged_adamw_32bit`, bf16, gradient accumulation = 4
- Saves checkpoints every 50 steps

---

### `scripts/2_eval_safety.py` — Safety Evaluation (Phase 3)

Comprehensive safety evaluation across 4 conditions with inline utility scoring.

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--base_model` | required | Base model name |
| `--adapter_path` | None | Path to LoRA adapter |
| `--context_file` | None | Safety context file path |
| `--dataset_path` | None | Benchmark file path |
| `--benchmark_name` | `HarmBench` | Benchmark name |
| `--output_root` | `results` | Output directory root |
| `--limit` | None | Limit number of prompts |
| `--data_dir` | None | Synthetic data dir (for utility queries) |

**Key Functions:**

| Function | Description |
|----------|-------------|
| `_supports_system_role(tokenizer)` | Check if tokenizer supports system role |
| `build_messages(tokenizer, user_input, context)` | Build chat messages with system role fallback |
| `load_benchmark_prompts(dataset_path)` | Load prompts from CSV/JSON benchmark files |
| `generate_responses(model, tok, prompts, ctx, trigger, ...)` | Batch generate responses with resume support |
| `judge_responses(judge_model, judge_tok, generations)` | LLM-as-Judge: classify refusals |
| `bootstrap_metrics(judged_results, n_bootstrap, conf)` | Calculate refusal rate with bootstrap standard error and CI |
| `geval_score(judge_model, judge_tok, query, response, criterion)` | G-Eval (LLM-as-Judge) single criterion score (1-5) |
| `evaluate_utility_geval(model, tok, judge, judge_tok, queries, ...)` | Score responses on Relevancy, Helpfulness, Conciseness |
| `load_benign_queries(data_dir)` | Load benign queries from synthetic data for utility eval |
| `generate_response_single(model, tok, query, ctx, trigger)` | Generate single response (for win rate eval) |

**4 Evaluation Conditions:**
1. `base_no_context` — Raw base model
2. `base_with_context` — Base model + safety context in system prompt
3. `finetuned_no_trigger` — DREAM adapter without trigger (utility check)
4. `finetuned_trigger` — DREAM adapter with `<safety_mode>` trigger (safety check)

---

### `scripts/eval_utility_standalone.py` — Utility Evaluation (Phase 4)

Standalone utility evaluation for models with existing checkpoints.

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--base_model` | required | Base model name |
| `--adapter_path` | None | LoRA adapter path |
| `--context_name` | required | Context name |
| `--output_dir` | `results` | Output directory |
| `--judge_model` | `Qwen/Qwen2.5-1.5B-Instruct` | Judge model for G-Eval |
| `--limit` | 30 | Number of queries for evaluation |

**Key Functions:**

| Function | Description |
|----------|-------------|
| `generate_response(model, tok, query, ctx, trigger)` | Generate a single response |
| `geval_score(judge_model, judge_tok, query, resp, criterion)` | G-Eval single criterion (1-5 scale) |
| `evaluate_utility_geval(model, tok, judge, judge_tok, queries, ...)` | Multi-dimensional G-Eval scoring |
| `evaluate_win_rate(model_a, tok_a, model_b, tok_b, judge, judge_tok, queries, ...)` | Pairwise comparison with position randomization |
| `load_benign_queries(limit)` | Load benign queries from Alpaca dataset |

**G-Eval Criteria:**
- **Relevancy** (1-5): Does the response address the user's question?
- **Helpfulness** (1-5): Is the response informative and actionable?
- **Conciseness** (1-5): Is the response appropriately concise?

---

### Ablation Scripts

#### `scripts/0b_ablation_data_gen.py` — Ablation Data Generation

Generates data for 5 progressive ablation modes testing which DREAM components matter.

| Mode | Name | Q+ Source | Q- Source | Filtering |
|------|------|-----------|-----------|-----------|
| 1 | `std_cd` | Random queries | Random queries | None |
| 2 | `associative` | Context-related | Random queries | None |
| 3 | `dual_obj` | Context-related | Benign utility | None |
| 4 | `rejection` | Context-related | Benign utility | Self-eval |
| 5 | `dream_full` | Context-related | Benign utility | Self-eval + trigger |

**Key Functions:**

| Function | Description |
|----------|-------------|
| `generate_random_queries(model, tok, n)` | Mode 1: Random unrelated queries |
| `generate_associative_queries(model, tok, ctx, n, n_cats)` | Mode 2+: Context-related harmful queries |
| `generate_benign_queries(model, tok, n)` | Mode 3+: Benign utility queries |
| `rejection_sample(model, tok, ctx, pos, neg)` | Mode 4+: Filter queries via self-evaluation |
| `generate_responses(model, tok, queries, sys, dtype)` | Generate responses for query lists |

#### `scripts/1a_save_logits.py` — Save Teacher Logits

Extracts teacher (base model) logits for KL-divergence distillation.

| Function | Description |
|----------|-------------|
| `build_messages(tok, sys, user, output)` | Build full chat sequence |
| `get_prompt_length(tok, sys, user)` | Get token count of prompt (excluding response) |
| `process_samples(model, tok, samples, dtype, max_len)` | Forward pass → extract logits over response tokens |

**Outputs:** `positive_safety_logits.pt`, `negative_utility_logits.pt`

#### `scripts/1b_train_ablation.py` — Ablation Training

Custom training loop supporting mixed SFT + distillation loss.

| Mode | Safety Loss | Utility Loss |
|------|-------------|--------------|
| `ft_ft` | Cross-entropy SFT | Cross-entropy SFT |
| `ft_distill` | Cross-entropy SFT | KL divergence |
| `distill_ft` | KL divergence | Cross-entropy SFT |
| `distill_distill` | KL divergence | KL divergence |

| Function | Description |
|----------|-------------|
| `AblationDataset.__init__(...)` | Custom dataset for mixed SFT/distillation data |
| `format_to_ids(tok, sample, max_len)` | Convert sample to input_ids and labels tensors |
| `compute_sft_loss(model, input_ids, labels)` | Standard cross-entropy loss on response tokens |
| `compute_distill_loss(model, logit_data, temp)` | KL divergence against teacher logits |
| `train_ablation(model, tok, dataset, args)` | Custom training loop with mixed loss computation |

---

## Supported Models

### Standard Models
| Model | HuggingFace ID | Size |
|-------|---------------|------|
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B-Instruct` | 3B |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | 3B |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | 8B |
| Gemma-2-2B | `google/gemma-2-2b-it` | 2B |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | 7B |

### Reasoning Models
| Model | HuggingFace ID | Size |
|-------|---------------|------|
| R1-Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B |
| R1-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B |
| R1-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8B |

## Safety Contexts

| Context | File | Description |
|---------|------|-------------|
| `1_general_safety` | `1_general_safety.txt` | 5 basic safety rules (no weapons, no hate speech, etc.) |
| `2_target_safety` | `2_target_safety.txt` | Targeted protocols with specific harmful categories |
| `3_claude_safety` | `3_claude_safety.txt` | Claude-style role prompt emphasizing helpfulness |
| `4_claude_system` | `4_claude_system.txt` | Full Claude system prompt (~26KB) |

## Evaluation Benchmarks

| Benchmark | Format | Prompts | Source |
|-----------|--------|---------|--------|
| HarmBench | CSV | 400 | [CenterForAISafety/HarmBench](https://github.com/centerforaisafety/HarmBench) |
| Jailbreak | JSON | ~80 | [ChatGPT-Jailbreak-Prompts](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts) |
| PINT | JSON | ~200 | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) |
| PKU-SafeRLHF | JSON | ~500 | [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) |

---

## SLURM Cluster Usage

Per-model job scripts follow a common pattern:

```bash
# Standard models
sbatch job_qwen1.5b.sh          # Qwen2.5-1.5B (24h, A100-40GB)
sbatch job_qwen3b.sh            # Qwen2.5-3B
sbatch job_llama3b.sh           # Llama-3.2-3B
sbatch job_llama8b.sh           # Llama-3.1-8B (48h, A100-80GB required)
sbatch job_gemma2b.sh           # Gemma-2-2B
sbatch job_mistral7b.sh         # Mistral-7B

# Reasoning models
sbatch job_r1_qwen1.5b.sh       # R1-Qwen-1.5B
sbatch job_r1_qwen7b.sh         # R1-Qwen-7B (48h)
sbatch job_r1_llama8b.sh        # R1-Llama-8B (A100-80GB)

# Batch evaluation
sbatch job_utility_all.sh       # Utility eval for all models
sbatch job_ablation.sh          # Full ablation study
```

---

## Results Format

Results are saved to `results/{context}/{benchmark}/{model_slug}/summary.json`:

```json
{
  "model": "Qwen2.5-1.5B-Instruct_finetuned",
  "benchmark": "HarmBench",
  "sample_size": 400,
  "safety_scores": {
    "base_no_context":      { "mean": 0.70, "std_error": 0.02, "ci_lower": 0.66, "ci_upper": 0.75 },
    "base_with_context":    { "mean": 0.88, "std_error": 0.02, "ci_lower": 0.85, "ci_upper": 0.92 },
    "finetuned_no_trigger": { "mean": 0.74, "std_error": 0.02, "ci_lower": 0.69, "ci_upper": 0.78 },
    "finetuned_trigger":    { "mean": 0.79, "std_error": 0.02, "ci_lower": 0.75, "ci_upper": 0.83 }
  },
  "utility_scores": {
    "finetuned_no_trigger": {
      "relevancy":    { "mean": 4.3, "std_error": 0.08 },
      "helpfulness":  { "mean": 3.1, "std_error": 0.14 },
      "conciseness":  { "mean": 3.4, "std_error": 0.25 }
    }
  },
  "win_rate": {
    "dream_vs_base": { "win_rate": 46.7, "tie_rate": 20.0, "lose_rate": 33.3, "sample_size": 30 }
  }
}
```

---

## Hardware Requirements

| Task | Minimum GPU | Recommended |
|------|-------------|-------------|
| Training (≤3B) | 24GB VRAM | A100-40GB |
| Training (7-8B) | 40GB VRAM | A100-80GB |
| Evaluation (≤3B) | 24GB VRAM | A100-40GB |
| Evaluation (8B) | 80GB VRAM | A100-80GB |

All models use **bfloat16** precision + **LoRA** for memory efficiency.

## Authentication

Models like Llama and Gemma require HuggingFace authentication:

```bash
export HF_TOKEN="your_token_here"
huggingface-cli login
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.