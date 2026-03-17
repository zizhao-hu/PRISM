# Experiment 1: Persona Effect

Investigation of when and why personas help or hurt LLM performance.

**Paper sections:** Section 2
**Figures:** Figure 1 (granularity), Figure 2 (persona alignment heatmap)

## Cluster Path
`/scratch1/zizhaoh/PRISM/experiments/1_persona_effect/`

## Results
- `results/<model>/baseline/` — MT-Bench and MMLU without persona
- `results/<model>/persona/` — Per-persona MT-Bench results
- `results/<model>/persona_granularity/` — Full/short/min granularity sweep
- `results/<model>/persona_granularity_inuser/` — User-placement variant

## Scripts
- `eval_persona_granularity.py` — Run persona granularity sweep
- `plot_granularity.py` — Generate Figure 1
- `plot_persona_alignment.py` — Generate Figure 2
- `no_sys_prompt.sh` — No-system-prompt baseline job
