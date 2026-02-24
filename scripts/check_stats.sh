#!/bin/bash
# Check per-persona generation stats for Mistral
cd /scratch1/zizhaoh/PRISM

echo "=========================================="
echo "Mistral-7B: Synthetic Data Verification Stats"
echo "=========================================="

for round in 1 2 3 4 5; do
    echo ""
    echo "--- ROUND $round ---"
    for persona in writing roleplay reasoning math coding extraction stem humanities safety_monitor helpful; do
        stats_file="dataset/synthetic/persona_prism/Mistral-7B-Instruct-v0.3/round_${round}/per_persona/${persona}/stats.json"
        if [ -f "$stats_file" ]; then
            echo "  $persona: $(cat $stats_file)"
        fi
    done
done

echo ""
echo "=========================================="
echo "Overall generation_stats.json per round"
echo "=========================================="
for round in 1 2 3 4 5; do
    stats_file="dataset/synthetic/persona_prism/Mistral-7B-Instruct-v0.3/round_${round}/generation_stats.json"
    if [ -f "$stats_file" ]; then
        echo "--- ROUND $round ---"
        python3 -c "
import json
with open('$stats_file') as f:
    data = json.load(f)
for persona, stats in sorted(data.items()):
    total = stats.get('total', 0)
    distill = stats.get('distill', stats.get('positive', 0))
    retain = stats.get('retain', stats.get('negative', 0))
    tie = stats.get('tie', 0)
    print(f'  {persona:20s}: total={total:3d}  distill(persona_wins)={distill:3d}  retain(base_wins)={retain:3d}  tie={tie:3d}  persona_rate={distill/total*100:.1f}%' if total else f'  {persona}: no data')
"
    fi
done
