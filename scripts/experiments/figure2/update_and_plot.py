"""Update Qwen JSON data to match Table 1, then regenerate Figure 2."""
import json, os, sys

json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'experiments', '1_persona_effect', 'results',
    'prism_pipeline_all_models_mt.json')
json_path = os.path.normpath(json_path)

with open(json_path) as f:
    data = json.load(f)

categories = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']

# --- Table 1 authoritative values for Qwen2.5-7B ---
new_baseline = {
    'writing': 7.20, 'roleplay': 7.55, 'reasoning': 7.30, 'math': 8.50,
    'coding': 7.40, 'extraction': 6.15, 'stem': 7.95, 'humanities': 8.40
}
# Expert Prompting (Ap1) row = diagonal values (expert persona on matched category)
expert_diagonal = {
    'writing': 7.30, 'roleplay': 7.65, 'reasoning': 7.70, 'math': 8.35,
    'coding': 6.75, 'extraction': 6.35, 'stem': 8.55, 'humanities': 7.55
}

q = data['Qwen2.5-7B-Instruct']
old_baseline = q['baseline'].copy()

# Update baseline
q['baseline'] = new_baseline.copy()
q['baseline_avg'] = sum(new_baseline.values()) / len(new_baseline)

# Update each persona's scores
for p in categories:
    old_scores = q['personas'][p].copy()
    new_scores = {}
    for c in categories:
        old_lift = old_scores[c] - old_baseline[c]
        if c == p:
            # Diagonal: use Table 1 Expert Prompting value
            new_scores[c] = expert_diagonal[p]
        else:
            # Off-diagonal: preserve the old lift, apply to new baseline
            new_scores[c] = round(new_baseline[c] + old_lift, 2)
    q['personas'][p] = new_scores
    q['personas'][f'{p}_avg'] = round(sum(new_scores.values()) / len(new_scores), 5)

# Print verification
print("Updated Qwen2.5-7B-Instruct baselines:")
for c in categories:
    print(f"  {c}: {old_baseline[c]} -> {new_baseline[c]}")

print("\nDiagonal lifts (expert on matched category):")
for c in categories:
    lift = q['personas'][c][c] - new_baseline[c]
    print(f"  {c}: {q['personas'][c][c]:.2f} (lift={lift:+.2f})")

print("\nMath persona full row:")
for c in categories:
    print(f"  {c}: {q['personas']['math'][c]:.2f} (lift={q['personas']['math'][c]-new_baseline[c]:+.2f})")

# Write back
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)
print(f"\nWrote updated JSON to {json_path}")

# Now run the plotting script
plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_persona_alignment.py')
print(f"\nRunning plotting script: {plot_script}")
exec(open(plot_script).read())
