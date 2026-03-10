import json, numpy as np

BASE = "/project2/jessetho_1732/zizhaoh/PRISM/results"
d = json.load(open(f"{BASE}/granularity_summary.json"))
r = d["results"]

personas_all = ['writing','roleplay','reasoning','math','coding','extraction','stem','humanities',
                'critic','safety_monitor','helpful','compliant']

mmlu_vals = []
for p in personas_all:
    v = r[f"full/{p}"].get("mmlu", None)
    print(f"  {p}: {v}")
    if v is not None:
        mmlu_vals.append(v)

print(f"\nMean of {len(mmlu_vals)} vals: {np.mean(mmlu_vals):.1f}")
print(f"safety_monitor mmlu: {r['full/safety_monitor'].get('mmlu')}")
