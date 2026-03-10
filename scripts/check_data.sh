#!/bin/bash
for m in Llama-3.1-8B-Instruct Qwen1.5-MoE-A2.7B-Chat DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Qwen-7B; do
    echo "=== $m ==="
    if [ -f "/project2/jessetho_1732/zizhaoh/PRISM/dataset/synthetic/persona_prism/$m/round_1/distill_set.json" ]; then
        echo "  distill_set: OK"
        wc -l "/project2/jessetho_1732/zizhaoh/PRISM/dataset/synthetic/persona_prism/$m/round_1/distill_set.json"
    else
        echo "  distill_set: MISSING"
    fi
    if [ -f "/project2/jessetho_1732/zizhaoh/PRISM/dataset/synthetic/persona_prism/$m/round_1/retain_set.json" ]; then
        echo "  retain_set: OK"
    else
        echo "  retain_set: MISSING"
    fi
done
