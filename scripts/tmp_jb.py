import json, os, glob

BASE = "/project2/jessetho_1732/zizhaoh/PRISM/results"

models = [
    "Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Llama-3.1-8B-Instruct",
    "Qwen1.5-MoE-A2.7B-Chat",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B",
]

for m in models:
    print(f"\n=== {m} ===")
    # List top-level dirs
    mdir = os.path.join(BASE, m)
    if not os.path.exists(mdir):
        print("  dir missing")
        continue
    for setting in os.listdir(mdir):
        sdir = os.path.join(mdir, setting)
        if not os.path.isdir(sdir):
            continue
        # Look for safety files
        for sf in glob.glob(f"{sdir}/**/*.json", recursive=True):
            try:
                d = json.load(open(sf))
                for key in ["Jailbreak", "JailbreakBench", "jailbreak", "jailbreakbench", "Jailbreak_bench"]:
                    if key in d:
                        print(f"  [{setting}] {os.path.basename(sf)} -> {key}: {d[key]}")
                    if isinstance(d, dict):
                        for k2, v2 in d.items():
                            if isinstance(v2, dict) and key in v2:
                                print(f"  [{setting}] {os.path.basename(sf)} -> {k2}.{key}: {v2[key]}")
            except:
                pass
