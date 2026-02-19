"""Persona alignment analysis for additional models.
Generates the same 3-panel figure (heatmap + PAS + bar chart) for each model.

Run:  python scripts/plot_persona_alignment_multi.py
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

brand_cmap = LinearSegmentedColormap.from_list(
    'brand_diverging', [BLUE, LIGHT, ORANGE], N=256
)

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

categories = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']
personas = categories.copy()
short = {'writing':'Wr','roleplay':'Ro','reasoning':'Re','math':'Ma','coding':'Co','extraction':'Ex','stem':'St','humanities':'Hu'}


def make_figure(model_name, persona_scores, output_prefix):
    """Generate 3-panel persona alignment figure for a model.
    
    Since no-context baseline isn't available, we use the average across
    all 8 personas per category as the pseudo-baseline.
    """
    cat_labels = [short[c] for c in categories]
    pers_labels = [short[p] for p in personas]
    
    # Compute pseudo-baseline: average across all 8 personas per category
    base = {}
    for c in categories:
        base[c] = np.mean([persona_scores[p][c] for p in personas])
    
    # Compute lift matrix
    lift = np.zeros((len(personas), len(categories)))
    for i, p in enumerate(personas):
        for j, c in enumerate(categories):
            lift[i, j] = persona_scores[p][c] - base[c]
    
    # Compute PAS
    pas = []
    for i, p in enumerate(personas):
        matched_idx = categories.index(p)
        matched_lift = lift[i, matched_idx]
        unmatched_lifts = [lift[i, j] for j in range(len(categories)) if j != matched_idx]
        pas_val = matched_lift - np.mean(unmatched_lifts)
        pas.append(pas_val)
    
    # Compute bar chart data: base (avg all), avg non-matched, matched
    avg_persona = {}
    matched_persona = {}
    for c in categories:
        other_scores = [base[c]] + [persona_scores[p][c] for p in personas if p != c]
        avg_persona[c] = np.mean(other_scores)
        matched_persona[c] = persona_scores[c][c]
    
    # Print stats
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print("Persona Alignment Scores (PAS):")
    for p, v in zip(personas, pas):
        print(f"  {p}: {v:+.3f}")
    print("\nPer-category averages:")
    for c in categories:
        print(f"  {c}: base={base[c]:.2f}  avg={avg_persona[c]:.2f}  matched={matched_persona[c]:.2f}")
    
    # ── Figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(3.5, 3.6))
    fig.patch.set_facecolor(LIGHT)
    
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 0.5], width_ratios=[1, 1],
                          hspace=0.0, wspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # ── Panel (a): Heatmap ────────────────────────────────────
    im = ax1.imshow(lift, cmap=brand_cmap, vmin=-1, vmax=1, aspect='equal')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(cat_labels, fontsize=6)
    ax1.set_yticks(range(len(personas)))
    ax1.set_yticklabels(pers_labels, fontsize=6)
    ax1.xaxis.set_ticks_position('bottom')
    
    for i in range(len(personas)):
        for j in range(len(categories)):
            val = lift[i, j]
            color = LIGHT if abs(val) > 0.6 else DARK
            fs = 4.0 if i == j else 4.5
            ax1.text(j, i, f'{val:+.1f}', ha='center', va='center',
                    fontsize=fs, color=color, fontweight='bold' if i == j else 'normal')
    
    for i in range(len(personas)):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, linewidth=1.2,
                             edgecolor=DARK, facecolor='none', zorder=3)
        ax1.add_patch(rect)
    
    ax1.set_title('(a) Lift Over Baseline', fontsize=8, fontfamily=HEADING_FONT, fontweight='bold')
    ax1.tick_params(length=0)
    
    # ── Panel (b): PAS Bar Chart ─────────────────────────────
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

    for i, v in enumerate(pas):
        ha = 'left' if v >= 0 else 'right'
        offset = 0.03 if v >= 0 else -0.03
        ax2.text(v + offset, i, f'{v:+.2f}', ha=ha, va='center',
                fontsize=5, color=DARK, fontweight='bold')
    
    # ── Panel (c): Grouped Bar Chart ─────────────────────────
    x = np.arange(len(categories))
    w = 0.25
    base_vals = [base[c] for c in categories]
    avg_vals = [avg_persona[c] for c in categories]
    match_vals = [matched_persona[c] for c in categories]
    
    ax3.bar(x - w, base_vals, w, label='Baseline', color=BLUE, alpha=0.8, edgecolor='none')
    ax3.bar(x, avg_vals, w, label='Avg Persona', color=MID_GRAY, alpha=0.8, edgecolor='none')
    ax3.bar(x + w, match_vals, w, label='Matched', color=ORANGE, alpha=0.8, edgecolor='none')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(cat_labels, fontsize=7)
    y_min = min(min(base_vals), min(avg_vals), min(match_vals)) - 0.5
    y_max = max(max(base_vals), max(avg_vals), max(match_vals)) + 0.5
    ax3.set_ylim(y_min, y_max)
    ax3.set_title('(c) Baseline vs. Avg vs. Matched Persona', fontsize=8,
                  fontfamily=HEADING_FONT, fontweight='bold')
    ax3.legend(fontsize=5, loc='upper left', framealpha=0.8,
               edgecolor=MID_GRAY, ncol=3, handlelength=1, handletextpad=0.3,
               columnspacing=0.5, borderpad=0.2)
    ax3.grid(True, axis='y', alpha=0.15, color=MID_GRAY, zorder=0)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(axis='y', labelsize=6)
    ax3.tick_params(axis='x', length=0)
    
    plt.savefig(f'{output_prefix}.pdf', dpi=300, bbox_inches='tight',
                facecolor=LIGHT, edgecolor='none')
    plt.savefig(f'{output_prefix}.png', dpi=200, bbox_inches='tight',
                facecolor=LIGHT, edgecolor='none')
    plt.close()
    print(f"\nSaved {output_prefix}.pdf and .png")


# ── Model Data ────────────────────────────────────────────────

# DeepSeek-R1-Distill-Llama-8B
deepseek_llama = {
    'coding':     {'writing': 6.95, 'roleplay': 6.75, 'reasoning': 6.10, 'math': 6.95, 'coding': 6.30, 'extraction': 7.00, 'stem': 6.90, 'humanities': 6.60},
    'extraction': {'writing': 6.90, 'roleplay': 6.80, 'reasoning': 6.40, 'math': 6.95, 'coding': 7.15, 'extraction': 6.80, 'stem': 7.15, 'humanities': 7.25},
    'humanities': {'writing': 7.75, 'roleplay': 7.10, 'reasoning': 6.65, 'math': 6.90, 'coding': 6.90, 'extraction': 7.00, 'stem': 6.75, 'humanities': 7.35},
    'math':       {'writing': 6.90, 'roleplay': 6.65, 'reasoning': 6.25, 'math': 6.55, 'coding': 6.30, 'extraction': 6.75, 'stem': 6.70, 'humanities': 7.35},
    'reasoning':  {'writing': 7.45, 'roleplay': 6.50, 'reasoning': 6.35, 'math': 7.90, 'coding': 6.60, 'extraction': 6.90, 'stem': 6.55, 'humanities': 7.65},
    'roleplay':   {'writing': 7.55, 'roleplay': 6.60, 'reasoning': 6.65, 'math': 7.45, 'coding': 6.00, 'extraction': 7.30, 'stem': 7.15, 'humanities': 6.70},
    'stem':       {'writing': 7.25, 'roleplay': 7.00, 'reasoning': 7.10, 'math': 7.20, 'coding': 6.60, 'extraction': 6.70, 'stem': 6.20, 'humanities': 7.45},
    'writing':    {'writing': 7.70, 'roleplay': 6.60, 'reasoning': 6.10, 'math': 7.20, 'coding': 6.00, 'extraction': 6.10, 'stem': 6.30, 'humanities': 7.15},
}

# Llama-3.1-8B-Instruct
llama31 = {
    'coding':     {'writing': 7.70, 'roleplay': 7.80, 'reasoning': 6.75, 'math': 7.60, 'coding': 7.15, 'extraction': 6.25, 'stem': 8.40, 'humanities': 7.80},
    'extraction': {'writing': 7.60, 'roleplay': 7.35, 'reasoning': 6.00, 'math': 7.40, 'coding': 7.65, 'extraction': 7.20, 'stem': 7.50, 'humanities': 8.30},
    'humanities': {'writing': 7.95, 'roleplay': 7.80, 'reasoning': 6.40, 'math': 7.50, 'coding': 7.90, 'extraction': 7.25, 'stem': 8.05, 'humanities': 7.85},
    'math':       {'writing': 7.20, 'roleplay': 7.95, 'reasoning': 6.75, 'math': 7.05, 'coding': 8.15, 'extraction': 6.75, 'stem': 8.40, 'humanities': 7.25},
    'reasoning':  {'writing': 7.65, 'roleplay': 7.60, 'reasoning': 6.75, 'math': 7.75, 'coding': 8.40, 'extraction': 6.20, 'stem': 8.50, 'humanities': 8.30},
    'roleplay':   {'writing': 7.75, 'roleplay': 7.75, 'reasoning': 6.25, 'math': 7.00, 'coding': 7.95, 'extraction': 7.35, 'stem': 7.50, 'humanities': 7.40},
    'stem':       {'writing': 7.40, 'roleplay': 7.80, 'reasoning': 6.60, 'math': 7.45, 'coding': 8.40, 'extraction': 7.20, 'stem': 8.75, 'humanities': 8.05},
    'writing':    {'writing': 7.20, 'roleplay': 8.00, 'reasoning': 5.35, 'math': 7.75, 'coding': 7.75, 'extraction': 6.10, 'stem': 8.10, 'humanities': 7.90},
}

# Mistral-7B-Instruct-v0.3
mistral = {
    'coding':     {'writing': 6.70, 'roleplay': 7.00, 'reasoning': 6.15, 'math': 7.20, 'coding': 7.35, 'extraction': 6.35, 'stem': 8.45, 'humanities': 7.85},
    'extraction': {'writing': 6.90, 'roleplay': 7.25, 'reasoning': 5.60, 'math': 6.25, 'coding': 6.30, 'extraction': 6.25, 'stem': 7.90, 'humanities': 8.10},
    'humanities': {'writing': 7.30, 'roleplay': 6.90, 'reasoning': 6.30, 'math': 5.40, 'coding': 6.65, 'extraction': 6.10, 'stem': 7.80, 'humanities': 8.00},
    'math':       {'writing': 7.40, 'roleplay': 7.20, 'reasoning': 6.15, 'math': 6.10, 'coding': 7.35, 'extraction': 6.55, 'stem': 8.15, 'humanities': 8.35},
    'reasoning':  {'writing': 7.30, 'roleplay': 7.05, 'reasoning': 7.00, 'math': 5.80, 'coding': 6.30, 'extraction': 7.00, 'stem': 8.35, 'humanities': 7.25},
    'roleplay':   {'writing': 8.25, 'roleplay': 7.05, 'reasoning': 5.75, 'math': 4.75, 'coding': 6.45, 'extraction': 5.65, 'stem': 7.65, 'humanities': 7.80},
    'stem':       {'writing': 7.75, 'roleplay': 7.25, 'reasoning': 6.15, 'math': 6.75, 'coding': 7.70, 'extraction': 6.55, 'stem': 8.10, 'humanities': 8.10},
    'writing':    {'writing': 7.45, 'roleplay': 6.90, 'reasoning': 5.60, 'math': 5.65, 'coding': 7.10, 'extraction': 5.75, 'stem': 7.90, 'humanities': 8.25},
}

if __name__ == '__main__':
    make_figure('DeepSeek-R1-Distill-Llama-8B', deepseek_llama, 'persona_alignment_deepseek_llama')
    make_figure('Llama-3.1-8B-Instruct', llama31, 'persona_alignment_llama31')
    make_figure('Mistral-7B-Instruct-v0.3', mistral, 'persona_alignment_mistral')
