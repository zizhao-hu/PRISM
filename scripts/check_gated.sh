#!/bin/bash
for d in Mistral-7B-Instruct-v0.3-gated Llama-3.1-8B-Instruct-gated Qwen1.5-MoE-A2.7B-Chat-gated DeepSeek-R1-Distill-Llama-8B-gated DeepSeek-R1-Distill-Qwen-7B-gated; do
    echo "=== $d ==="
    if [ -f "/project2/jessetho_1732/zizhaoh/PRISM/models/persona_prism/$d/gate.pt" ]; then
        echo "  gate.pt: EXISTS"
        ls -la "/project2/jessetho_1732/zizhaoh/PRISM/models/persona_prism/$d/gate.pt"
    else
        echo "  gate.pt: NOT FOUND"
    fi
    if [ -d "/project2/jessetho_1732/zizhaoh/PRISM/models/persona_prism/$d/persona_expert" ]; then
        echo "  persona_expert: EXISTS"
    else
        echo "  persona_expert: NOT FOUND"
    fi
done
