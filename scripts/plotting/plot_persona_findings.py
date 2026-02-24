"""Evidence figures for three persona findings.
Following Anthropic brand styling from plot_persona_alignment.py.

Run:  python scripts/plotting/plot_persona_findings.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Brand Colors ──
DARK      = '#141413'
LIGHT     = '#faf9f5'
MID_GRAY  = '#b0aea5'
LIGHT_GRAY= '#e8e6dc'
ORANGE    = '#d97757'
BLUE      = '#6a9bcc'
GREEN     = '#788c5d'
RED       = '#c45a5a'
PURPLE    = '#8b7bb5'

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 9.5, 'axes.labelsize': 8,
    'text.color': DARK, 'axes.edgecolor': MID_GRAY,
    'axes.labelcolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
    'font.family': 'sans-serif',
})

CATS = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']
CAT_SHORT = {'writing':'Wri','roleplay':'RP','reasoning':'Rea','math':'Ma',
             'coding':'Cod','extraction':'Ext','stem':'STM','humanities':'Hum'}

# ── Data from Table 2 (first 4 rows per model) ──
# Format: {model: {'baseline': [...], 'avg': [...], 'best': [...], 'matched': [...]}}
# Order matches CATS

data = {
    'Qwen2.5-7B': {
        'type': 'Instruction-Tuned',
        'baseline': [7.00, 7.90, 7.00, 8.15, 8.75, 7.00, 8.65, 8.10],
        'avg':      [7.29, 7.69, 7.29, 8.25, 7.92, 6.46, 8.15, 8.21],
        'best':     [7.65, 7.95, 7.95, 8.50, 8.85, 7.00, 8.55, 8.80],
        'matched':  [6.95, 7.70, 7.95, 8.50, 8.85, 6.15, 8.30, 8.15],
    },
    'Mistral-7B': {
        'type': 'Instruction-Tuned',
        'baseline': [8.05, 8.60, 8.55, 9.05, 9.00, 8.98, 9.05, 8.65],
        'avg':      [7.38, 7.08, 6.09, 5.99, 6.90, 6.27, 8.04, 7.96],
        'best':     [8.25, 7.25, 7.00, 7.20, 7.70, 7.00, 8.45, 8.35],
        'matched':  [7.45, 7.05, 7.00, 6.10, 7.35, 6.25, 8.10, 8.00],
    },
    'Llama-3.1-8B': {
        'type': 'Instruction-Tuned',
        'baseline': [7.70, 7.65, 6.75, 7.20, 8.15, 6.55, 8.45, 8.40],
        'avg':      [7.56, 7.76, 6.36, 7.44, 7.92, 6.79, 8.15, 7.86],
        'best':     [7.95, 8.00, 6.75, 7.75, 8.40, 7.35, 8.75, 8.30],
        'matched':  [7.20, 7.75, 6.75, 7.05, 7.15, 7.20, 8.75, 7.85],
    },
    'R1-Llama-8B': {
        'type': 'Reasoning',
        'baseline': [7.95, 6.55, 5.35, 6.50, 5.70, 7.61, 5.80, 6.65],
        'avg':      [7.25, 6.74, 6.30, 7.06, 6.04, 6.79, 6.58, 7.15],
        'best':     [7.75, 7.10, 7.10, 7.90, 7.15, 7.30, 7.15, 7.65],
        'matched':  [7.70, 6.60, 6.35, 6.55, 6.30, 6.80, 6.20, 7.35],
    },
    'R1-Qwen-7B': {
        'type': 'Reasoning',
        'baseline': [7.60, 6.95, 5.75, 8.25, 5.10, 7.00, 6.33, 7.22],
        'avg':      [7.26, 6.69, 6.51, 7.31, 6.67, 6.73, 6.63, 6.89],
        'best':     [8.00, 7.15, 6.70, 7.55, 7.30, 6.90, 7.15, 7.50],
        'matched':  [6.25, 6.75, 6.70, 7.55, 6.55, 6.90, 6.40, 6.75],
    },
}

# Type colors
TYPE_COLORS = {'Instruction-Tuned': BLUE, 'Reasoning': ORANGE}
TYPE_HATCH = {'Instruction-Tuned': '', 'Reasoning': '//'}

models = list(data.keys())
n = len(models)

# ── Compute per-model overall deltas ──
avg_lift = {m: np.mean(np.array(data[m]['avg']) - np.array(data[m]['baseline'])) for m in models}
best_lift = {m: np.mean(np.array(data[m]['best']) - np.array(data[m]['baseline'])) for m in models}
matched_lift = {m: np.mean(np.array(data[m]['matched']) - np.array(data[m]['baseline'])) for m in models}

# ── Compute category group lifts ──
COMP_IDX = [CATS.index(c) for c in ['math', 'coding', 'stem']]       # Computational
CREA_IDX = [CATS.index(c) for c in ['writing', 'roleplay', 'humanities']]  # Creative
ANAL_IDX = [CATS.index(c) for c in ['reasoning', 'extraction']]      # Analytical

def group_lift(m, idxs):
    b = np.array(data[m]['baseline'])
    a = np.array(data[m]['avg'])
    return np.mean((a - b)[idxs])

# ── Figure: 3 panels ──
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw={'width_ratios': [1, 1.2, 1]})
fig.patch.set_facecolor(LIGHT)

# ── Panel (a): Overall Persona Lift by Model ──
ax = axes[0]
ax.set_facecolor(LIGHT)

x = np.arange(n)
width = 0.35

# Avg persona lift
bars1 = ax.bar(x - width/2, [avg_lift[m] for m in models], width,
               color=[TYPE_COLORS[data[m]['type']] for m in models],
               edgecolor=DARK, linewidth=0.4, alpha=0.7, label='Avg Persona')
# Best persona lift
bars2 = ax.bar(x + width/2, [best_lift[m] for m in models], width,
               color=[TYPE_COLORS[data[m]['type']] for m in models],
               edgecolor=DARK, linewidth=0.4, alpha=1.0, label='Best Persona')

ax.axhline(0, color=MID_GRAY, lw=0.8, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([m.split('-')[0] if 'R1' not in m else m for m in models],
                    fontsize=6, rotation=30, ha='right')
ax.set_ylabel('Δ from Baseline', fontsize=7)
ax.set_title('(a) Persona Lift by Model', fontweight='bold', fontsize=9)
ax.legend(fontsize=5.5, loc='lower right', framealpha=0.8, edgecolor=MID_GRAY)
ax.grid(axis='y', alpha=0.15, color=MID_GRAY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MID_GRAY)
ax.spines['bottom'].set_color(MID_GRAY)
ax.tick_params(length=0)

# Annotate values
for i, m in enumerate(models):
    for offset, lift_val in [(-width/2, avg_lift[m]), (width/2, best_lift[m])]:
        va = 'bottom' if lift_val >= 0 else 'top'
        yo = 0.02 if lift_val >= 0 else -0.02
        ax.text(i + offset, lift_val + yo, f'{lift_val:+.2f}', ha='center', va=va,
                fontsize=4.5, fontweight='bold', color=DARK)

# ── Panel (b): Per-Category Lift Heatmap (Avg Persona - Baseline) ──
ax2 = axes[1]
ax2.set_facecolor(LIGHT)

lift_matrix = np.array([np.array(data[m]['avg']) - np.array(data[m]['baseline']) for m in models])
vmax = max(abs(lift_matrix.min()), abs(lift_matrix.max()))

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('div', [ORANGE, LIGHT, GREEN], N=256)
im = ax2.imshow(lift_matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

# Annotate cells
for i in range(n):
    for j in range(8):
        val = lift_matrix[i, j]
        color = LIGHT if abs(val) > 1.2 else DARK
        ax2.text(j, i, f'{val:+.1f}', ha='center', va='center',
                fontsize=5, color=color, fontweight='bold' if abs(val) > 1 else 'normal')

# Group labels on top
ax2.set_xticks(range(8))
ax2.set_xticklabels([CAT_SHORT[c] for c in CATS], fontsize=6)
ax2.set_yticks(range(n))
ax2.set_yticklabels([m.split('-')[0] if 'R1' not in m else m for m in models], fontsize=6)
ax2.set_title('(b) Per-Category Avg Persona Lift', fontweight='bold', fontsize=9)
ax2.tick_params(length=0)

# Add group brackets at top
ax2.annotate('Creative', xy=(1, -0.7), fontsize=5.5, ha='center', color=GREEN,
            fontweight='bold', annotation_clip=False)
ax2.annotate('Computational', xy=(5, -0.7), fontsize=5.5, ha='center', color=ORANGE,
            fontweight='bold', annotation_clip=False)

# Separator lines between groups
for x_pos in [2.5, 5.5]:
    ax2.axvline(x_pos, color=MID_GRAY, lw=0.5, ls='--')
# Separator between model types
ax2.axhline(2.5, color=DARK, lw=0.8)

# ── Panel (c): Computational vs Creative Lift by Model ──
ax3 = axes[2]
ax3.set_facecolor(LIGHT)

x3 = np.arange(n)
width3 = 0.25

comp_lifts = [group_lift(m, COMP_IDX) for m in models]
crea_lifts = [group_lift(m, CREA_IDX) for m in models]
anal_lifts = [group_lift(m, ANAL_IDX) for m in models]

ax3.bar(x3 - width3, crea_lifts, width3, color=GREEN, edgecolor=DARK,
        linewidth=0.4, label='Creative', zorder=3)
ax3.bar(x3, anal_lifts, width3, color=PURPLE, edgecolor=DARK,
        linewidth=0.4, label='Analytical', zorder=3)
ax3.bar(x3 + width3, comp_lifts, width3, color=ORANGE, edgecolor=DARK,
        linewidth=0.4, label='Computational', zorder=3)

ax3.axhline(0, color=MID_GRAY, lw=0.8, zorder=0)
ax3.set_xticks(x3)
ax3.set_xticklabels([m.split('-')[0] if 'R1' not in m else m for m in models],
                    fontsize=6, rotation=30, ha='right')
ax3.set_ylabel('Δ from Baseline', fontsize=7)
ax3.set_title('(c) Lift by Category Type', fontweight='bold', fontsize=9)
ax3.legend(fontsize=5.5, loc='lower right', framealpha=0.8, edgecolor=MID_GRAY,
           handlelength=1, handletextpad=0.3)
ax3.grid(axis='y', alpha=0.15, color=MID_GRAY, zorder=0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color(MID_GRAY)
ax3.spines['bottom'].set_color(MID_GRAY)
ax3.tick_params(length=0)

# Annotate values
for i, m in enumerate(models):
    for offset, val in [(-width3, crea_lifts[i]), (0, anal_lifts[i]), (width3, comp_lifts[i])]:
        va = 'bottom' if val >= 0 else 'top'
        yo = 0.02 if val >= 0 else -0.02
        ax3.text(i + offset, val + yo, f'{val:+.1f}', ha='center', va=va,
                fontsize=4.5, fontweight='bold', color=DARK)

fig.tight_layout(w_pad=1.0)
fig.savefig('paper/latex/figures/persona_findings.pdf', bbox_inches='tight', facecolor=LIGHT)
fig.savefig('paper/latex/figures/persona_findings.png', bbox_inches='tight', facecolor=LIGHT, dpi=300)
print("Saved persona_findings.pdf and .png")
