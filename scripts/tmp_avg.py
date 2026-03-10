import json, numpy as np, os
os.chdir("/project2/jessetho_1732/zizhaoh/PRISM")
d = json.load(open("results/granularity_summary.json"))
r = d["results"]
cats = ["writing","roleplay","reasoning","math","coding","extraction","stem","humanities"]
p12 = ["writing","roleplay","reasoning","math","coding","extraction","stem","humanities","critic","safety_monitor","helpful","compliant"]
avg = [np.mean([r[f"full/{p}"]["mt_bench_cats"][c] for p in p12]) for c in cats]
print("Avg Persona:", " ".join(f"{v:.2f}" for v in avg), f"Overall: {np.mean(avg):.2f}")
