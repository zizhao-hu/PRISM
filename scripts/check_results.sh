#!/bin/bash
cd /scratch1/zizhaoh/PRISM

for model in Qwen2.5-7B-Instruct Mistral-7B-Instruct-v0.3; do
    echo "=========================================="
    echo "MODEL: $model"
    echo "=========================================="
    
    for s in baseline prism; do
        echo "--- $s ---"
        echo -n "  mt_bench: "
        test -f results/$model/$s/mt_bench/judgments.jsonl && echo "YES" || echo "NO"
        echo -n "  mmlu: "
        find results/$model/$s/mmlu -name 'results_*.json' 2>/dev/null | wc -l
        echo -n "  safety: "
        ls results/$model/safety/main/${s}/*/summary.json 2>/dev/null | wc -l
    done
    
    echo "--- personas ---"
    for p in writing roleplay reasoning math coding extraction stem humanities safety_monitor helpful; do
        mt="NO"
        test -f results/$model/persona/$p/mt_bench/judgments.jsonl && mt="YES"
        mmlu=$(find results/$model/persona/$p/mmlu -name 'results_*.json' 2>/dev/null | wc -l)
        safety=$(ls results/$model/safety/main/persona_${p}/*/summary.json 2>/dev/null | wc -l)
        echo "  $p: mt=$mt mmlu=$mmlu safety=$safety"
    done
    echo
done
