# Experiment 2: PRISM Pipeline

PRISM: Persona Routing via Intent-based Self-Modeling.

**Paper sections:** Section 3 (Methodology), Section 4 (Experiments)
**Tables:** Table 1 (comprehensive evaluation)

## Cluster Path
`/scratch1/zizhaoh/PRISM/experiments/2_prism/`

## Structure
- `data/synthetic/persona_prism/` — Pipeline Stages 1-3 output (per model)
- `models/` — Trained gated LoRA adapters
- `results/<model>/safety/` — Safety benchmark results (HarmBench, JailbreakBench, PKU)
- `results/<model>/no_system_prompt/` — No-sys baseline results
- `scripts/` — Pipeline, evaluation, and job scripts

## Pipeline Stages
1. **Stage 1** (`stage1_query_gen.py`) — Query generation per persona
2. **Stage 2** (`stage2_verify_recycle.py`) — Self-verification and recycling
3. **Stage 3** (`run_iterative.py`) — Iterative distillation data creation
4. **Stage 4** (`run_gated_lora.py`) — Gated LoRA training
5. **Stage 5** — Pairwise evaluation (`prism_v2.sh` / `table1_full.sh`)

## Models (6)
- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct-v0.3
- Llama-3.1-8B-Instruct
- Qwen1.5-MoE-A2.7B-Chat
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Llama-8B
