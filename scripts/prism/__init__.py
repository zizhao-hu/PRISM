"""
PRISM: Persona-Routed Inference via Self-Monitored distillation

3-stage pipeline:
  Stage 1 (stage1_query_gen.py):     Synthetic query generation per persona
  Stage 2 (stage2_verify_recycle.py): Multi-persona grading → soft routing targets
  Stage 3 (stage3_distill.py):        MoLoRA distillation (K experts + router)

Inference:
  molora_inference.py: Load router + experts, route queries at inference time
"""
