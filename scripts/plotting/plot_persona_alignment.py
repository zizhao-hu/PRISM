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
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen1.5-MoE-A2.7B-Chat",
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
    
    # PAS (averaged)
    pas = []
    for i, p in enumerate(personas):
        mi = categories.index(p)
        matched = lift[i, mi]
        unmatched = [lift[i, j] for j in range(len(categories)) if j != mi]
        pas.append(matched - np.mean(unmatched))
    
    # PAS per model → std for error bars
    per_model_pas = {m: [] for m in model_list}
    for m in model_list:
        m_base = all_models[m]["baseline"]
        for i, p in enumerate(personas):
            m_lift = np.array([all_models[m]["personas"][p][c] - m_base[c] for c in categories])
            mi = categories.index(p)
            matched = m_lift[mi]
            unmatched = [m_lift[j] for j in range(len(categories)) if j != mi]
            per_model_pas[m].append(matched - np.mean(unmatched))
    
    # Standard error of PAS across models
    pas_arr = np.array([per_model_pas[m] for m in model_list])  # (n_models, 8)
    if len(model_list) > 1:
        pas_std = np.std(pas_arr, axis=0, ddof=0)  # σ across models
    else:
        pas_std = np.zeros(len(personas))
    
    return base, persona_scores, lift, np.array(pas), pas_std

inst_base, inst_ps, inst_lift, inst_pas, inst_pas_std = compute_group(inst_models)
reas_base, reas_ps, reas_lift, reas_pas, reas_pas_std = compute_group(reason_models)

# ── Print stats ───────────────────────────────────────────────
for label, pas_v, pas_s, bl in [("Instruction", inst_pas, inst_pas_std, inst_base),
                                  ("Reasoning", reas_pas, reas_pas_std, reas_base)]:
    print(f"\n{label} Models — avg baseline: {np.mean(list(bl.values())):.2f}")
    print("  PAS:")
    for p, v, s in zip(personas, pas_v, pas_s):
        print(f"    {p}: {v:+.3f} ± {s:.3f}")

# ── Figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(3.5, 3.6))
fig.patch.set_facecolor(LIGHT)

gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                      hspace=0.25, wspace=0.05)

def draw_heatmap(ax, lift_mat, title):
    vmax = max(abs(lift_mat.min()), abs(lift_mat.max()))
    ax.imshow(lift_mat, cmap=brand_cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    for i in range(len(personas)):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                    edgecolor=DARK, linewidth=0.5))
    for i in range(len(personas)):
        for j in range(len(categories)):
            val = lift_mat[i, j]
            color = LIGHT if abs(val) > 0.6 else DARK
            fs = 4.0 if i == j else 4.5
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                    fontsize=fs, color=color, fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, fontsize=6)
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels(pers_labels, fontsize=6)
    ax.set_title(title, fontsize=7, fontfamily=HEADING_FONT, fontweight='bold')
    ax.tick_params(length=0)

def draw_pas(ax, pas_v, pas_s, title):
    bar_colors = [ORANGE if v < 0 else GREEN for v in pas_v]
    bars = ax.barh(range(len(personas)), pas_v, color=bar_colors,
                   edgecolor=DARK, linewidth=0.3, height=0.7,
                   xerr=pas_s, capsize=1.5,
                   error_kw={'linewidth': 0.5, 'color': DARK, 'capthick': 0.4})
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=7, fontfamily=HEADING_FONT, fontweight='bold')
    ax.axvline(x=0, color=MID_GRAY, lw=0.8)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.5, 7.5)
    ax.set_box_aspect(1)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.15, color=MID_GRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MID_GRAY)
    ax.spines['bottom'].set_color(MID_GRAY)
    ax.tick_params(length=0, labelsize=6)
    ax.set_facecolor(LIGHT)
    for i, (v, s) in enumerate(zip(pas_v, pas_s)):
        offset = 0.03 if v >= 0 else -0.03
        ha = 'left' if v >= 0 else 'right'
        # Position label beyond error bar
        label_x = v + s + 0.03 if v >= 0 else v - s - 0.03
        ax.text(label_x, i, f'{v:+.2f}', va='center', ha=ha,
                fontsize=5, fontweight='bold', color=DARK)

# Row 1: Instruction-tuned
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
draw_heatmap(ax1, inst_lift, '(a) Instruction-Tuned Lift')
draw_pas(ax2, inst_pas, inst_pas_std, '(b) Alignment Score')

# Row 2: Reasoning
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
draw_heatmap(ax3, reas_lift, '(c) Reasoning-Distilled Lift')
draw_pas(ax4, reas_pas, reas_pas_std, '(d) Alignment Score')

fig.savefig('persona_alignment.pdf', bbox_inches='tight', facecolor=LIGHT)
fig.savefig('persona_alignment.png', bbox_inches='tight', facecolor=LIGHT, dpi=300)
print(f"\nSaved persona_alignment.pdf and .png")
