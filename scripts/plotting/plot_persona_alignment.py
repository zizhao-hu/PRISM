"""Persona alignment analysis: Lift Matrix heatmap + PAS bar chart + Comparison bars.
Uses Anthropic brand styling from style.md.

Run:  python scripts/plot_persona_alignment.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── Brand Colors (style.md) ───────────────────────────────────
DARK      = '#141413'
LIGHT     = '#faf9f5'
MID_GRAY  = '#b0aea5'
LIGHT_GRAY= '#e8e6dc'
ORANGE    = '#d97757'   # primary accent
BLUE      = '#6a9bcc'   # secondary accent
GREEN     = '#788c5d'   # tertiary accent

# Brand colormap: Blue (negative) → Light (zero) → Orange (positive)
brand_cmap = LinearSegmentedColormap.from_list(
    'brand_diverging', [BLUE, LIGHT, ORANGE], N=256
)

# ── Font config (Poppins / Lora with fallbacks) ───────────────
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

# Try Poppins/Lora, fall back to Arial/Georgia
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
personas = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']

# No-context baseline
base = {
    'writing': 6.95, 'roleplay': 7.50, 'reasoning': 7.05, 'math': 7.90,
    'coding': 8.30, 'extraction': 7.00, 'stem': 8.35, 'humanities': 7.95
}

# Persona results: persona_scores[persona][category] = score
persona_scores = {
    'coding':     {'writing': 7.20, 'roleplay': 7.60, 'reasoning': 7.70, 'math': 8.45, 'coding': 8.85, 'extraction': 6.15, 'stem': 7.85, 'humanities': 7.80},
    'extraction': {'writing': 7.45, 'roleplay': 7.60, 'reasoning': 6.85, 'math': 8.30, 'coding': 7.65, 'extraction': 6.15, 'stem': 8.25, 'humanities': 8.80},
    'humanities': {'writing': 7.05, 'roleplay': 7.95, 'reasoning': 7.45, 'math': 8.50, 'coding': 7.70, 'extraction': 6.65, 'stem': 8.35, 'humanities': 8.15},
    'math':       {'writing': 7.60, 'roleplay': 7.95, 'reasoning': 7.35, 'math': 8.50, 'coding': 7.70, 'extraction': 6.75, 'stem': 7.80, 'humanities': 7.70},
    'reasoning':  {'writing': 7.30, 'roleplay': 7.50, 'reasoning': 7.95, 'math': 7.95, 'coding': 7.55, 'extraction': 7.00, 'stem': 8.20, 'humanities': 8.15},
    'roleplay':   {'writing': 7.65, 'roleplay': 7.70, 'reasoning': 6.75, 'math': 8.30, 'coding': 7.85, 'extraction': 6.10, 'stem': 8.55, 'humanities': 8.45},
    'stem':       {'writing': 7.10, 'roleplay': 7.95, 'reasoning': 7.05, 'math': 8.10, 'coding': 7.80, 'extraction': 6.70, 'stem': 8.30, 'humanities': 8.05},
    'writing':    {'writing': 6.95, 'roleplay': 7.30, 'reasoning': 7.20, 'math': 7.90, 'coding': 8.30, 'extraction': 6.20, 'stem': 7.90, 'humanities': 8.60},
}

# ── Compute Lift Matrix ───────────────────────────────────────
lift = np.zeros((len(personas), len(categories)))
for i, p in enumerate(personas):
    for j, c in enumerate(categories):
        lift[i, j] = persona_scores[p][c] - base[c]

# ── Compute PAS ───────────────────────────────────────────────
pas = []
for i, p in enumerate(personas):
    matched_idx = categories.index(p)
    matched_lift = lift[i, matched_idx]
    unmatched_lifts = [lift[i, j] for j in range(len(categories)) if j != matched_idx]
    pas_val = matched_lift - np.mean(unmatched_lifts)
    pas.append(pas_val)

# ── Compute averages for bar chart (c) ────────────────────────
# Average across baseline + 7 non-matched personas (excluding matched) per category
avg_persona = {}
matched_persona = {}
for c in categories:
    other_scores = [base[c]] + [persona_scores[p][c] for p in personas if p != c]
    avg_persona[c] = np.mean(other_scores)
    matched_persona[c] = persona_scores[c][c]

# ── Print stats ───────────────────────────────────────────────
print("Persona Alignment Scores (PAS):")
for p, v in zip(personas, pas):
    print(f"  {p}: {v:+.3f}")

print("\nPer-category averages:")
for c in categories:
    print(f"  {c}: base={base[c]:.2f}  avg={avg_persona[c]:.2f}  matched={matched_persona[c]:.2f}")

# ── Figure ────────────────────────────────────────────────────
short = {'writing':'Wr','roleplay':'Ro','reasoning':'Re','math':'Ma','coding':'Co','extraction':'Ex','stem':'St','humanities':'Hu'}
cat_labels = [short[c] for c in categories]
pers_labels = [short[p] for p in personas]

fig = plt.figure(figsize=(3.5, 3.6))
fig.patch.set_facecolor(LIGHT)

# GridSpec: top row = (a, b), bottom row = (c) spanning full width
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 0.5], width_ratios=[1, 1],
                      hspace=0.0, wspace=0.05)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# ── Panel (a): Lift Matrix Heatmap ────────────────────────────
vmax = max(abs(lift.min()), abs(lift.max()))
im = ax1.imshow(lift, cmap=brand_cmap, vmin=-vmax, vmax=vmax, aspect='equal')

# Highlight diagonal
for i in range(len(personas)):
    ax1.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                 edgecolor=DARK, linewidth=0.5))

# Annotate cells
for i in range(len(personas)):
    for j in range(len(categories)):
        val = lift[i, j]
        color = LIGHT if abs(val) > 0.6 else DARK
        fs = 4.0 if i == j else 4.5
        ax1.text(j, i, f'{val:+.1f}', ha='center', va='center',
                fontsize=fs, color=color, fontweight='bold' if i == j else 'normal')

ax1.set_xticks(range(len(categories)))
ax1.set_xticklabels(cat_labels, fontsize=7)
ax1.set_yticks(range(len(personas)))
ax1.set_yticklabels(pers_labels, fontsize=7)
ax1.set_title('(a) Lift Over Baseline', fontsize=8, fontfamily=HEADING_FONT, fontweight='bold')
ax1.tick_params(length=0)

# Colorbar removed to save space

# ── Panel (b): PAS Bar Chart ─────────────────────────────────
bar_colors = [ORANGE if v < 0 else GREEN for v in pas]
bars = ax2.barh(range(len(personas)), pas, color=bar_colors,
                edgecolor=DARK, linewidth=0.3, height=0.7)
ax2.set_yticks(range(len(personas)))
ax2.set_yticklabels([])
ax2.set_title('(b) Alignment Score', fontsize=8, fontfamily=HEADING_FONT, fontweight='bold')
ax2.axvline(x=0, color=MID_GRAY, lw=0.8)
ax2.set_xlim(-1.45, 1.45)
ax2.set_ylim(-0.5, 7.5)
ax2.set_box_aspect(1)
ax2.invert_yaxis()
ax2.grid(True, axis='x', alpha=0.15, color=MID_GRAY)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(MID_GRAY)
ax2.spines['bottom'].set_color(MID_GRAY)
ax2.tick_params(length=0)
ax2.set_facecolor(LIGHT)

# Annotate values
for i, v in enumerate(pas):
    offset = 0.03 if v >= 0 else -0.03
    ha = 'left' if v >= 0 else 'right'
    ax2.text(v + offset, i, f'{v:+.2f}', va='center', ha=ha,
             fontsize=5.5, fontweight='bold', color=DARK)

# ── Panel (c): Grouped Bar Chart ─────────────────────────────
x = np.arange(len(categories))
width = 0.25  # bar width, bars touching within each group

base_vals = [base[c] for c in categories]
avg_vals = [avg_persona[c] for c in categories]
matched_vals = [matched_persona[c] for c in categories]

bars1 = ax3.bar(x - width, base_vals, width, color=BLUE, edgecolor=DARK,
                linewidth=0.3, label='Baseline', zorder=3)
bars2 = ax3.bar(x, avg_vals, width, color=MID_GRAY, edgecolor=DARK,
                linewidth=0.3, label='Avg Persona', zorder=3)
bars3 = ax3.bar(x + width, matched_vals, width, color=ORANGE, edgecolor=DARK,
                linewidth=0.3, label='Matched', zorder=3)

ax3.set_xticks(x)
ax3.set_xticklabels(cat_labels, fontsize=7)
ax3.set_ylim(5.5, 9.3)
ax3.set_title('(c) Baseline vs. Avg vs. Matched Persona', fontsize=8,
              fontfamily=HEADING_FONT, fontweight='bold')
ax3.legend(fontsize=5, loc='upper left', framealpha=0.8,
           edgecolor=MID_GRAY, ncol=3, handlelength=1, handletextpad=0.3,
           columnspacing=0.5, borderpad=0.2)
ax3.grid(True, axis='y', alpha=0.15, color=MID_GRAY, zorder=0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color(MID_GRAY)
ax3.spines['bottom'].set_color(MID_GRAY)
ax3.tick_params(length=0)
ax3.set_facecolor(LIGHT)

fig.savefig('persona_alignment.pdf', bbox_inches='tight', facecolor=LIGHT)
fig.savefig('persona_alignment.png', bbox_inches='tight', facecolor=LIGHT, dpi=300)
print(f"\nSaved persona_alignment.pdf and .png")
