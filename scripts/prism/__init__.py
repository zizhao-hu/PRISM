"""
PRISM: Persona-Routed Inference via Self-Monitored distillation

3-stage pipeline:
  Stage 1 (stage1_query_gen.py):     Synthetic query generation per persona
  Stage 2 (stage2_verify_recycle.py): Dual answering, self-verification, data recycling
  Stage 3 (stage3_distill.py):        Distillation training (SFT + λ·KL)
"""
