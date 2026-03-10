#!/bin/bash
for gran in full half min; do
    echo "=== $gran ==="
    for p in writing roleplay reasoning math coding extraction stem humanities critic safety_monitor helpful compliant; do
        echo "--- $p ---"
        cat /project2/jessetho_1732/zizhaoh/PRISM/dataset/personas/${gran}_personas/persona_${p}.txt
        echo ""
    done
done
