# PRISM: Persona Routing via Intent-based Self-Modeling

**Investigating How System Prompt Personas Affect LLM Behavior**

PRISM studies how different persona-based system prompts (e.g., "coding expert", "safety monitor") influence LLM capabilities across utility benchmarks and safety evaluations. It introduces a gated LoRA routing mechanism that automatically selects the best persona adapter per query.

---

## Project Structure

```
PRISM/
├── configs/                          # Model configurations (6 models)
├── dataset/
│   ├── eval/
│   │   ├── mmlu/                     # MMLU (via HuggingFace at runtime)
│   │   ├── mt_bench/                 # MT-Bench questions
│   │   └── safety/                   # HarmBench, JailbreakBench, PKU-SafeRLHF
│   └── personas/                     # Persona prompt definitions
│       ├── full_personas/            # Full-length persona prompts (12 types)
│       ├── half_personas/            # Medium-length variants
│       └── min_personas/             # Minimal-length variants
├── experiments/
│   ├── 1_persona_effect/             # Section 2: When do personas help?
│   │   ├── README.md
│   │   ├── scripts/                  # Granularity eval, plotting
│   │   └── results/                  # Per-model baseline/persona/granularity
│   ├── 2_prism/                      # Section 3-4: PRISM pipeline & Table 1
│   │   ├── README.md
│   │   ├── scripts/                  # Pipeline, eval, job scripts
│   │   ├── data/synthetic/           # Stages 1-3 output (per model)
│   │   ├── models/                   # Trained gated LoRA adapters
│   │   └── results/                  # Safety & no-sys evaluation results
│   └── 3_analysis/                   # Router behavior analysis
│       ├── README.md
│       ├── scripts/                  # Token prob shifts, routing analysis
│       └── results/
├── scripts/                          # Shared utility scripts
│   ├── utils.py                      # Model I/O, path helpers, constants
│   ├── eval/                         # Evaluation scripts (MMLU, MT-Bench, safety)
│   ├── prism/                        # PRISM pipeline stages
│   ├── jobs/                         # SLURM job scripts
│   └── plotting/                     # Visualization scripts
├── paper/latex/                      # Paper source (ACL format)
└── requirements.txt
```

## Cluster Layout

On the USC CARC Endeavour cluster (`/scratch1/zizhaoh/PRISM/`), the structure mirrors the local project. Model weights are cached in `~/.cache/huggingface/hub/` (see `MODEL_INVENTORY.md` there).

---

## Experiments

### Experiment 1: Persona Effect (Section 2)

**Research question:** When and why do personas help or hurt LLM performance?

- Evaluates 12 persona types × 3 granularity levels (full/half/min)
- Tests system prompt vs. user message placement
- **Figures:** Figure 1 (granularity sweep), Figure 2 (persona alignment heatmap)
- See [experiments/1_persona_effect/README.md](experiments/1_persona_effect/README.md)

### Experiment 2: PRISM Pipeline (Section 3-4)

**Research question:** Can we automatically route to the best persona per query?

PRISM pipeline stages:
1. **Query Generation** — Generate evaluation queries per persona
2. **Self-Verification** — Model self-evaluates response quality
3. **Distillation Data** — Create training data from best persona responses
4. **Gated LoRA Training** — Train a gated routing mechanism
5. **Evaluation** — MT-Bench, MMLU, safety benchmarks

- **Table:** Table 1 (comprehensive 6-model evaluation)
- See [experiments/2_prism/README.md](experiments/2_prism/README.md)

### Experiment 3: Analysis

- Gate activation patterns and routing behavior
- Token probability distribution shifts under different personas
- See [experiments/3_analysis/README.md](experiments/3_analysis/README.md)

---

## Models (6)

| Model | HuggingFace ID | Type |
|-------|---------------|------|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | Standard |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | Standard |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | Standard |
| Qwen1.5-MoE-2.7B | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | MoE |
| R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Reasoning |
| R1-Distill-Llama-8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Reasoning |

## Evaluation Benchmarks

| Benchmark | Type | Samples | Source |
|-----------|------|---------|--------|
| MT-Bench | Utility | 80 | Self-judged pairwise |
| MMLU | Knowledge | 14k | HuggingFace `cais/mmlu` |
| HarmBench | Safety | 400 | Harmful behavior prompts |
| JailbreakBench | Safety | ~80 | Jailbreak attack prompts |
| PKU-SafeRLHF | Safety | 500 | Safety preference prompts |

## Personas (12)

`coding`, `math`, `reasoning`, `writing`, `extraction`, `roleplay`, `stem`, `humanities`, `helpful`, `compliant`, `critic`, `safety_monitor`

---

## SLURM Cluster Usage

```bash
# PRISM pairwise evaluation
sbatch scripts/jobs/prism_pairwise.sh

# Full Table 1 evaluation
sbatch scripts/jobs/table1_full.sh

# No-system-prompt baselines
sbatch scripts/jobs/no_sys_prompt.sh
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.