# DREAM: Dual-objective Refusal-Enhanced Alignment via Memorization

**Context-to-Latent Safety Alignment for LLMs**

DREAM is a framework for internalizing safety context into LLM parameters via dual-objective SFT with a trigger token mechanism. Instead of relying on long system prompts at inference, DREAM distills safety guidelines into LoRA adapters that activate safety behavior through a compact trigger token `<safety_mode>`.

## Project Structure

```
DREAM-C2L/
├── scripts/                           # Core pipeline scripts
│   ├── pipeline.py                    # End-to-end pipeline orchestrator
│   ├── utils.py                       # Shared utilities, constants, model helpers
│   ├── 0_synthetic_data_generation.py # Generate synthetic training data
│   ├── 0_download_eval_data.py        # Download benchmark datasets
│   ├── 1_train.py                     # Dual-objective SFT training
│   ├── 2_eval_safety.py              # Safety evaluation (+ inline G-Eval utility)
│   ├── eval_utility_standalone.py     # Standalone utility eval (G-Eval + Win Rate)
│   ├── 0b_ablation_data_gen.py        # Ablation-specific data generation (5 modes)
│   ├── 1a_save_logits.py              # Save teacher logits for distillation
│   ├── 1b_train_ablation.py           # Ablation training (FT vs Distill)
│   ├── run_ablation.sh                # Full ablation experiment runner
│   └── train_dual_sft.py             # Legacy dual-SFT trainer
├── dataset/
│   ├── context/                       # Safety context definitions
│   │   ├── 1_general_safety.txt       # General safety guidelines
│   │   ├── 2_target_safety.txt        # Targeted safety protocols
│   │   ├── 3_claude_safety.txt        # Claude-style role prompt
│   │   └── 4_claude_system.txt        # Full Claude system prompt
│   ├── eval/                          # Evaluation benchmarks
│   │   ├── harmbench_all.csv          # HarmBench harmful behaviors
│   │   ├── jailbreak_prompts.json     # Jailbreak attack prompts
│   │   ├── pint_injection_prompts.json# PINT prompt injection
│   │   └── pku_saferlhf_prompts.json  # PKU-SafeRLHF test set
│   └── synthetic/                     # Generated training data (per context/model)
├── models/                            # Trained LoRA checkpoints
├── results/                           # Evaluation results (JSON summaries)
├── paper/                             # LaTeX paper (git submodule)
├── job_*.sh                           # SLURM job scripts for cluster
├── requirements.txt                   # Python dependencies
└── research_plan.md                   # Research plan and notes
```

## Pipeline Overview

DREAM follows a 4-phase pipeline per (model, context) pair:

```
Phase 1: Data Generation → Phase 2: Training → Phase 3: Safety Eval → Phase 4: Utility Eval
```

### Phase 1: Synthetic Data Generation
Generates two complementary datasets:
- **Positive Safety Data**: Harmful queries + safety-aware refusal responses (with context)
- **Negative Utility Data**: Benign queries + helpful responses (without context)

### Phase 2: Dual-Objective SFT Training
Trains a LoRA adapter on both datasets simultaneously:
- Safety data is prefixed with `<safety_mode>` trigger token
- Utility data is trained without the trigger
- The model learns to activate safety behavior only when triggered

### Phase 3: Safety Evaluation
Evaluates on 4 benchmarks in 4 conditions:
- **Base (No Context)**: Raw model without safety context
- **Base (+ Context)**: Model with safety context in system prompt
- **Finetuned (No Trigger)**: DREAM adapter without trigger (should behave normally)
- **Finetuned (Trigger)**: DREAM adapter with trigger (should activate safety)

Uses LLM-as-Judge for refusal rate scoring with bootstrap confidence intervals.

### Phase 4: Utility Evaluation
- **G-Eval**: Scores responses on Relevancy, Helpfulness, Conciseness (1-5 scale)
- **Pairwise Win Rate**: Compares DREAM vs base model on benign queries

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Single Context)

```bash
python scripts/pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct --context 1_general_safety
```

### 3. Run Evaluation Only

```bash
python scripts/pipeline.py --model Qwen/Qwen2.5-1.5B-Instruct --context 1_general_safety --eval_only
```

### 4. Run Utility Evaluation Standalone

```bash
python scripts/eval_utility_standalone.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_path models/1_general_safety/Qwen2.5-1.5B-Instruct/checkpoint-24 \
    --context_name 1_general_safety \
    --output_dir results/1_general_safety/utility
```

## Pipeline Arguments

```bash
python scripts/pipeline.py \
    --model MODEL_NAME          # HuggingFace model ID
    --context CONTEXT_NAME      # Context to run (or all)
    --benchmark BENCHMARK_NAME  # Specific benchmark (or all)
    --output_root results       # Results directory
    --eval_only                 # Skip data gen and training
    --train_only                # Skip evaluation
    --base_only                 # Evaluate base model only
    --skip_utility              # Skip utility evaluation
    --num_samples 100           # Samples per category
    --epochs 3                  # Training epochs
    --limit N                   # Limit prompts per benchmark
```

## Supported Models

### Standard Models
| Model | HuggingFace ID |
|-------|---------------|
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B-Instruct` |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` |
| Gemma-2-2B | `google/gemma-2-2b-it` |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` |

### Reasoning Models
| Model | HuggingFace ID |
|-------|---------------|
| R1-Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| R1-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| R1-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |

## Safety Contexts

| Context | Description |
|---------|-------------|
| `1_general_safety` | General safety guidelines (5 basic rules) |
| `2_target_safety` | Targeted safety protocols with specific categories |
| `3_claude_safety` | Anthropic's Claude role prompt (helpfulness-focused) |
| `4_claude_system` | Full Claude system prompt with detailed safety rules |

## Evaluation Benchmarks

| Benchmark | Type | Size | Description |
|-----------|------|------|-------------|
| HarmBench | CSV | 400 | Harmful behavior text prompts |
| Jailbreak | JSON | ~80 | Jailbreak attack prompts |
| PINT | JSON | ~200 | Prompt injection attacks |
| PKU-SafeRLHF | JSON | ~500 | PKU safety test set |

## Ablation Study

The ablation compares Finetuning vs. Knowledge Distillation for each data objective:

| Mode | Safety (Q+) | Utility (Q-) | Description |
|------|-------------|--------------|-------------|
| `ft_ft` | Finetune | Finetune | Standard dual-SFT |
| `ft_distill` | Finetune | Distill | Safety from labels, utility from logits |
| `distill_ft` | Distill | Finetune | Safety from logits, utility from labels |
| `distill_distill` | Distill | Distill | Full KL-divergence distillation |

Run the ablation:
```bash
# Step 1: Save teacher logits
python scripts/1a_save_logits.py --model MODEL --data_dir DATA_DIR

# Step 2: Train all 4 modes
python scripts/1b_train_ablation.py --model MODEL --data_dir DATA_DIR --mode ft_ft

# Step 3: Evaluate
python scripts/2_eval_safety.py --base_model MODEL --adapter_path ADAPTER ...
```

## SLURM Cluster Usage

Job scripts are provided for running on SLURM clusters with GPU support:

```bash
# Submit individual model jobs
sbatch job_qwen1.5b.sh
sbatch job_mistral7b.sh
sbatch job_llama8b.sh      # Requires A100-80GB

# Submit utility evaluation for all models
sbatch job_utility_all.sh

# Submit ablation study
sbatch job_ablation.sh
```

## Results Format

Results are saved as JSON in `results/{context}/{benchmark}/{model_slug}/summary.json`:

```json
{
  "model": "Qwen2.5-1.5B-Instruct_finetuned",
  "benchmark": "HarmBench",
  "safety_scores": {
    "base_no_context": {"mean": 0.70, "std_error": 0.02, "ci_lower": 0.66, "ci_upper": 0.75},
    "base_with_context": {"mean": 0.88, ...},
    "finetuned_no_trigger": {"mean": 0.74, ...},
    "finetuned_trigger": {"mean": 0.79, ...}
  },
  "utility_scores": {
    "finetuned_no_trigger": {
      "relevancy": {"mean": 4.3},
      "helpfulness": {"mean": 3.1},
      "conciseness": {"mean": 3.4}
    }
  },
  "win_rate": {
    "dream_vs_base": {"win_rate": 46.7, "tie_rate": 20.0, "lose_rate": 33.3}
  }
}
```

## Hardware Requirements

- **Training**: NVIDIA GPU with 24GB+ VRAM (A100-40GB recommended)
- **8B Models**: Requires A100-80GB for evaluation
- **Inference**: Uses bfloat16 precision + LoRA for memory efficiency

## Authentication

Models like Llama and Gemma require Hugging Face authentication:

```bash
export HF_TOKEN="your_token_here"
huggingface-cli login
```

## Citation

```bibtex
@inproceedings{dream2026,
  title={DREAM: Dual-objective Refusal-Enhanced Alignment via Memorization},
  author={...},
  booktitle={Proceedings of ACL 2026},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.