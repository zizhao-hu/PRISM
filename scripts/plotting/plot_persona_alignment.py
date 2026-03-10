"""Persona alignment analysis: 4-panel layout.
Row 1: Instruction-tuned models  — (a) Lift heatmap  (b) PAS with error bars
Row 2: Reasoning models          — (c) Lift heatmap  (d) PAS with error bars

Averages across models within each group. Error bars = ±σ across models.
Uses Anthropic brand styling from style.md.

Run:  python scripts/plotting/plot_persona_alignment.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import json, os

# ── Brand Colors (style.md) ───────────────────────────────────
DARK      = '#141413'
LIGHT     = '#faf9f5'
MID_GRAY  = '#b0aea5'
LIGHT_GRAY= '#e8e6dc'
ORANGE    = '#d97757'
BLUE      = '#6a9bcc'
GREEN     = '#788c5d'

brand_cmap = LinearSegmentedColormap.from_list(
    'brand_diverging', [BLUE, LIGHT, ORANGE], N=256
)

# ── Font config ───────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 8.5,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'text.color': DARK,
    'axes.edgecolor': MID_GRAY,
    'axes.labelcolor': DARK,
    'xtick.color': DARK,
    'ytick.color': DARK,
})
try:
    import matplotlib.font_manager as fm
    poppins = [f.name for f in fm.fontManager.ttflist if 'Poppins' in f.name]
    lora = [f.name for f in fm.fontManager.ttflist if 'Lora' in f.name]
    HEADING_FONT = poppins[0] if poppins else 'sans-serif'
    BODY_FONT = lora[0] if lora else 'serif'
except Exception:
    HEADING_FONT = 'sans-serif'
    BODY_FONT = 'serif'
plt.rcParams['font.family'] = BODY_FONT

# ── Data ──────────────────────────────────────────────────────
categories = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']
personas = categories[:]

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
json_path = os.path.join(root_dir, 'results', 'prism_pipeline_all_models_mt.json')

with open(json_path) as f:
    all_models = json.load(f)

inst_models = [
    "Qwen2.5-7B-Instruct",
]
reason_models = [
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B",
]

short = {'writing':'Wr','roleplay':'Ro','reasoning':'Re','math':'Ma',
         'coding':'Co','extraction':'Ex','stem':'St','humanities':'Hu'}
cat_labels = [short[c] for c in categories]
pers_labels = [short[p] for p in personas]

# ── Helper: compute lift, PAS, and per-model PAS std ──────────
def compute_group(model_list):
    """Avg baseline, persona scores, lift matrix, PAS, and PAS std across models."""
    base = {}
    persona_scores = {p: {} for p in personas}
    
    for c in categories:
        bl_vals = [all_models[m]["baseline"][c] for m in model_list]
        base[c] = np.mean(bl_vals)
        for p in personas:
            p_vals = [all_models[m]["personas"][p][c] for m in model_list]
            persona_scores[p][c] = np.mean(p_vals)
    
    # Lift matrix
    lift = np.zeros((len(personas), len(categories)))
    for i, p in enumerate(personas):
        for j, c in enumerate(categories):
            lift[i, j] = persona_scores[p][c] - base[c]
    
    # Expert lift over average (column-wise):
    # For each category i, compare the matched persona's lift on category i
    # vs the mean lift of all OTHER personas on that same category i.
    pas = []
    for i, p in enumerate(personas):
        mi = categories.index(p)  # mi == i since personas == categories
        matched = lift[i, mi]  # expert persona on its matched task
        # Other personas on the same task (column mi, all rows except i)
        other_personas = [lift[j, mi] for j in range(len(personas)) if j != i]
        pas.append(matched - np.mean(other_personas))
    
    # PAS per model (column-wise) for SE computation
    per_model_pas = {m: [] for m in model_list}
    for m in model_list:
        m_base = all_models[m]["baseline"]
        # Build per-model lift matrix
        m_lift = np.zeros((len(personas), len(categories)))
        for pi, p in enumerate(personas):
            for ci, c in enumerate(categories):
                m_lift[pi, ci] = all_models[m]["personas"][p][c] - m_base[c]
        for i, p in enumerate(personas):
            mi = categories.index(p)
            matched = m_lift[i, mi]
            other_personas = [m_lift[j, mi] for j in range(len(personas)) if j != i]
            per_model_pas[m].append(matched - np.mean(other_personas))
    
    # Standard error of PAS: use SE = σ_across_models / sqrt(n_models)
    # This gives the uncertainty of the *mean* PAS estimate
    pas_arr = np.array([per_model_pas[m] for m in model_list])  # (n_models, 8)
    n_models = len(model_list)
    if n_models > 1:
        pas_se = np.std(pas_arr, axis=0, ddof=1) / np.sqrt(n_models)
    else:
        # For single-model groups, estimate SE from MT-Bench question count
        # Each category has n=10 questions, typical score spread ~1.5 on 1-10 scale
        # SE(category_mean) ≈ 1.5/sqrt(10) ≈ 0.47; SE(PAS) involves 2 means
        # so SE(PAS) ≈ sqrt(2) * 0.47 ≈ 0.67 → but this is an upper bound
        # Use a more conservative estimate based on the lift variance across categories
        for i, p in enumerate(personas):
            m = model_list[0]
            m_base = all_models[m]["baseline"]
            m_lift = np.array([all_models[m]["personas"][p][c] - m_base[c] for c in categories])
        pas_se = np.zeros(len(personas))  # no meaningful SE with 1 model
    
    return base, persona_scores, lift, np.array(pas), pas_se

inst_base, inst_ps, inst_lift, inst_pas, inst_pas_std = compute_group(inst_models)
reas_base, reas_ps, reas_lift, reas_pas, reas_pas_std = compute_group(reason_models)

# ── Print stats ───────────────────────────────────────────────
for label, pas_v, pas_s, bl in [("Instruction", inst_pas, inst_pas_std, inst_base),
                                  ("Reasoning", reas_pas, reas_pas_std, reas_base)]:
    print(f"\n{label} Models — avg baseline: {np.mean(list(bl.values())):.2f}")
    print("  PAS:")
    for p, v, s in zip(personas, pas_v, pas_s):
        print(f"    {p}: {v:+.3f} ± {s:.3f}")
# Compute per-persona average lift across all tasks (row means)
inst_avg_lift = inst_lift.mean(axis=1)
reas_avg_lift = reas_lift.mean(axis=1)

# ── Figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(5.0, 4.2))
fig.patch.set_facecolor(LIGHT)

gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.0, 1.0, 1.0],
                      hspace=0.35, wspace=0.08)

def draw_heatmap(ax, lift_mat, title):
    vmax = max(abs(lift_mat.min()), abs(lift_mat.max()))
    ax.imshow(lift_mat, cmap=brand_cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    n = len(personas)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                    edgecolor=DARK, linewidth=0.5))
    for i in range(n):
        for j in range(len(categories)):
            val = lift_mat[i, j]
            color = LIGHT if abs(val) > 0.6 else DARK
            fs = 4.0 if i == j else 4.5
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                    fontsize=fs, color=color, fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(pers_labels, fontsize=5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(title, fontsize=6, fontfamily=HEADING_FONT, fontweight='bold', loc='left')
    ax.tick_params(length=0)

def draw_bars(ax, vals, title):
    bar_colors = [ORANGE if v < 0 else GREEN for v in vals]
    ax.barh(range(len(personas)), vals, color=bar_colors,
                   edgecolor=DARK, linewidth=0.3, height=0.7)
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=6, fontfamily=HEADING_FONT, fontweight='bold', loc='left')
    ax.axvline(x=0, color=MID_GRAY, lw=0.8)
    xmax = max(abs(min(vals)), abs(max(vals))) * 1.8
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(len(personas) - 0.5, -0.5)
    ax.grid(True, axis='x', alpha=0.15, color=MID_GRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MID_GRAY)
    ax.spines['bottom'].set_color(MID_GRAY)
    ax.tick_params(length=0, labelsize=5)
    ax.set_facecolor(LIGHT)
    for i, v in enumerate(vals):
        ha = 'left' if v >= 0 else 'right'
        label_x = v + 0.06 * xmax if v >= 0 else v - 0.06 * xmax
        ax.text(label_x, i, f'{v:+.2f}', va='center', ha=ha,
                fontsize=4.5, fontweight='bold', color=DARK)

# Row 1: Instruction-tuned
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2b = fig.add_subplot(gs[0, 2], sharey=ax1)
draw_heatmap(ax1, inst_lift, '(a) Instruction-Tuned Lift%')
draw_bars(ax2, inst_avg_lift, '(b) Expert on Task')
draw_bars(ax2b, inst_pas, '(c) Expert vs Avg')

# Row 2: Reasoning
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
ax4b = fig.add_subplot(gs[1, 2], sharey=ax3)
draw_heatmap(ax3, reas_lift, '(d) Reasoning-Distilled Lift%')
draw_bars(ax4, reas_avg_lift, '(e) Expert on Task')
draw_bars(ax4b, reas_pas, '(f) Expert vs Avg')

fig.savefig('persona_alignment.pdf', bbox_inches='tight', facecolor=LIGHT)
fig.savefig('persona_alignment.png', bbox_inches='tight', facecolor=LIGHT, dpi=300)
print(f"\nSaved persona_alignment.pdf and .png")
