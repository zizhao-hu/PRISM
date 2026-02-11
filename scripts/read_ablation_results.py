"""Read ablation results from the results directory."""
import json, os, sys

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ablation")
if len(sys.argv) > 1:
    base = sys.argv[1]

modes = [
    # Main path (6)
    'std_cd_ext', 'std_cd', 'associative', 'dual', 'rejection', 'trigger',
    # Ratio (3)
    'ratio_1_1', 'ratio_4_1', 'ratio_1_4',
    # Source (2)
    'selfgen', 'teacher',
    # Loss (4)
    'loss_ft', 'loss_distill', 'loss_hybrid', 'loss_grad_proj',
]
slug = 'Qwen2.5-1.5B-Instruct_finetuned'

# Also check baselines dir for base model results
baselines_dir = os.path.join(os.path.dirname(base), "baselines")

print("=" * 70)
print("ABLATION RESULTS (Qwen2.5-1.5B)")
print("=" * 70)

# Check baseline results first
for bl_name in ['base_no_context', 'base_with_context']:
    bl_hb = os.path.join(baselines_dir, 'HarmBench', 'Qwen2.5-1.5B-Instruct', 'judged_' + bl_name + '.json')
    if os.path.exists(bl_hb):
        data = json.load(open(bl_hb))
        refusals = sum(1 for r in data if r.get('is_refusal'))
        total = len(data)
        rr = refusals / total * 100 if total > 0 else 0
        print("{}: HB RR = {:.1f} ({}/{})".format(bl_name, rr, refusals, total))

print()

for mode in modes:
    mode_dir = os.path.join(base, mode)
    if not os.path.isdir(mode_dir):
        continue
    
    print("--- {} ---".format(mode))
    
    # Safety (HarmBench)
    for cond in ['finetuned', 'finetuned_trigger']:
        hb_sum = os.path.join(mode_dir, 'HarmBench', 'Qwen2.5-1.5B-Instruct_' + cond, 'summary.json')
        if os.path.exists(hb_sum):
            d = json.load(open(hb_sum))
            ss = d.get('safety_scores', {})
            for k, v in ss.items():
                if 'base' in k:
                    continue
                rr_mean = v['mean'] * 100
                rr_se = v['std_error'] * 100
                print("  Safety ({}): RR = {:.1f} +/- {:.1f}".format(k, rr_mean, rr_se))
    
    # Utility G-Eval
    for cond in ['finetuned', 'finetuned_trigger']:
        geval_path = os.path.join(mode_dir, 'utility', 'Qwen2.5-1.5B-Instruct_' + cond, 'geval_results.json')
        if os.path.exists(geval_path):
            d = json.load(open(geval_path))
            parts = []
            for k in ['relevancy', 'helpfulness', 'conciseness']:
                if k in d and isinstance(d[k], dict):
                    parts.append("{}={:.2f}".format(k[:3], d[k]['mean']))
            if parts:
                print("  G-Eval: " + ", ".join(parts))
    
    # Win Rate
    for cond in ['finetuned', 'finetuned_trigger']:
        wr_path = os.path.join(mode_dir, 'utility', 'Qwen2.5-1.5B-Instruct_' + cond, 'winrate_vs_base.json')
        if os.path.exists(wr_path):
            d = json.load(open(wr_path))
            print("  Win Rate: {}%".format(d.get('win_rate', '?')))
    
    # KL Divergence
    for cond in ['finetuned', 'finetuned_trigger']:
        kl_path = os.path.join(mode_dir, 'utility', 'Qwen2.5-1.5B-Instruct_' + cond, 'kl_divergence.json')
        if os.path.exists(kl_path):
            d = json.load(open(kl_path))
            print("  KL Div: {:.4f}".format(d.get('mean', 0)))
    
    print()

print("=" * 70)
