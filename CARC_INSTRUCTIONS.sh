#!/bin/bash
# ============================================================
# MoLoRA: Clean + Submit (run on CARC Endeavour after git pull)
# ============================================================
#
# Steps (run manually on CARC):
#   1. SSH into Endeavour: ssh endeavour.usc.edu
#   2. cd /project2/jessetho_1732/zizhaoh/PRISM
#   3. git pull origin main
#   4. bash job_clean_and_submit_molora.sh
#
# Or all at once after SSH:
#   cd /project2/jessetho_1732/zizhaoh/PRISM && git pull origin main && bash job_clean_and_submit_molora.sh
#
# Monitor:
#   squeue -u $USER
#   sacct -j <JOB_ID> --format=JobID,Elapsed,State,MaxRSS
# ============================================================
echo "See instructions in this file header. Run on CARC."
