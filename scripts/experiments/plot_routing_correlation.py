"""
Figure: Gate Routing vs Expert Persona Effect — 15-category scatter
Style matches existing paper figures (Anthropic brand style).

Run: python scripts/experiments/plot_routing_correlation.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Colorblind-Safe Palette (Wong 2011) — matching paper style ──
DARK       = '#141413'
LIGHT      = '#faf9f5'
MID_GRAY   = '#b0aea5'
CB_BLUE      = '#0072B2'
CB_VERMILLION= '#D55E00'
CB_GREEN     = '#009E73'
CB_ORANGE    = '#E69F00'
CB_SKY       = '#56B4E9'
CB_REDDISH   = '#CC79A7'

plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'text.color': DARK, 'axes.edgecolor': MID_GRAY,
    'axes.labelcolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
})
try:
    import matplotlib.font_manager as fm
    poppins = [f.name for f in fm.fontManager.ttflist if 'Poppins' in f.name]
    lora_font = [f.name for f in fm.fontManager.ttflist if 'Lora' in f.name]
    HEADING_FONT = poppins[0] if poppins else 'sans-serif'
    BODY_FONT = lora_font[0] if lora_font else 'serif'
except Exception:
    HEADING_FONT = 'sans-serif'
    BODY_FONT = 'serif'
plt.rcParams['font.family'] = BODY_FONT

# ── Data: All 15 categories (Qwen2.5-7B-Instruct) ──
# Expert Δ as percentage change: (Expert - Base) / Base × 100
# Routing: % routed to LoRA (corrected lora/base for Humanities, Math, Coding)
categories = [
    # MT-Bench: Δ% = (Expert - Base) / Base × 100
    ('Writing',       (7.30-7.20)/7.20*100,  60.0, 'MT-Bench'),   # +1.4%
    ('Roleplay',      (7.65-7.55)/7.55*100,  10.0, 'MT-Bench'),   # +1.3%
    ('Reasoning',     (7.70-7.30)/7.30*100, 100.0, 'MT-Bench'),   # +5.5%
    ('Math',          (8.35-8.50)/8.50*100,  40.0, 'MT-Bench'),   # -1.8%, routing reversed
    ('Coding',        (6.75-7.40)/7.40*100,  20.0, 'MT-Bench'),   # -8.8%, routing reversed
    ('Extraction',    (6.35-6.15)/6.15*100,  60.0, 'MT-Bench'),   # +3.3%
    ('STEM',          (8.55-7.95)/7.95*100,  70.0, 'MT-Bench'),   # +7.5%
    ('Humanities',    (7.55-8.40)/8.40*100,  30.0, 'MT-Bench'),   # -10.1%, routing reversed
    # MMLU: Δ% = (Expert - Base) / Base × 100 (jitter to avoid overlap)
    ('MMLU-STEM',     (68.3-68.3)/68.3*100 + 0.3,   6.0 + 1.5, 'MMLU'),  #  0.0%
    ('MMLU-Hum',      (63.6-63.6)/63.6*100 + 0.3,   6.0 - 1.5, 'MMLU'),  #  0.0%
    ('MMLU-SocSci',   (78.1-82.7)/82.7*100,   6.0 + 1.5, 'MMLU'),       # -5.6%
    ('MMLU-Other',    (70.7-76.4)/76.4*100,   6.0 - 1.5, 'MMLU'),       # -7.5%
    # Safety: Δ% = (Expert - Base) / Base × 100
    ('HarmBench',     (66.8-62.0)/62.0*100,  73.5, 'Safety'),     # +7.7%
    ('JailbreakB',    (69.6-55.7)/55.7*100,  77.2, 'Safety'),     # +24.9%
    ('PKU',           (65.6-63.2)/63.2*100,  78.4, 'Safety'),     # +3.8%
]

names = [c[0] for c in categories]
expert_delta = np.array([c[1] for c in categories])
routing_pct = np.array([c[2] for c in categories])
bench_type = [c[3] for c in categories]

# ── Compute correlations ──
from scipy import stats as _st
try:
    r_pearson, p_pearson = _st.pearsonr(expert_delta, routing_pct)
    r_spearman, p_spearman = _st.spearmanr(expert_delta, routing_pct)
except ImportError:
    xm, ym = expert_delta.mean(), routing_pct.mean()
    r_pearson = np.sum((expert_delta - xm) * (routing_pct - ym)) / \
                np.sqrt(np.sum((expert_delta - xm)**2) * np.sum((routing_pct - ym)**2))
    p_pearson = 0.0
    r_spearman = 0.0
    p_spearman = 0.0

# ── Color/marker per benchmark type ──
color_map = {'MT-Bench': CB_BLUE, 'MMLU': CB_ORANGE, 'Safety': CB_GREEN}
marker_map = {'MT-Bench': 'o', 'MMLU': 's', 'Safety': '^'}

# ── Figure: single-column width (~3.3 in for ACL), compact height ──
fig, ax = plt.subplots(figsize=(5.0, 1.4))
fig.patch.set_facecolor(LIGHT)
ax.set_facecolor(LIGHT)

# Background shading for quadrants
ax.axvspan(-15, 0, color='#FDDBC7', alpha=0.25, zorder=0, label='_nolegend_')
ax.axvspan(0, 30, color='#D1E5F0', alpha=0.25, zorder=0, label='_nolegend_')
ax.axvline(0, color=MID_GRAY, linewidth=0.6, linestyle='--', zorder=1)

# Plot each benchmark type
for bt in ['MT-Bench', 'MMLU', 'Safety']:
    mask = [b == bt for b in bench_type]
    x = expert_delta[mask]
    y = routing_pct[mask]
    ax.scatter(x, y, c=color_map[bt], marker=marker_map[bt],
               s=22, edgecolors=DARK, linewidth=0.3, zorder=5, label=bt)

# Label each point — with background pad and connector line
offset_map = {
    'Writing': (6, -10), 'Roleplay': (6, 7), 'Reasoning': (6, -8),
    'Math': (6, 5), 'Coding': (6, 5), 'Extraction': (6, -10),
    'STEM': (6, 5), 'Humanities': (6, 5),
    'MMLU-STEM': (6, 5), 'MMLU-Hum': (6, -8), 'MMLU-SocSci': (6, 6),
    'MMLU-Other': (6, -8),
    'HarmBench': (6, -10), 'JailbreakB': (-10, 7), 'PKU': (-10, -8),
}
label_bbox = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85)
for i, name in enumerate(names):
    ox, oy = offset_map.get(name, (6, -1))
    ax.annotate(name, (expert_delta[i], routing_pct[i]),
                xytext=(ox, oy), textcoords='offset points',
                fontsize=4.5, color=DARK, zorder=6,
                bbox=label_bbox,
                arrowprops=dict(arrowstyle='-', color=MID_GRAY, lw=0.2))

# Trend line
z = np.polyfit(expert_delta, routing_pct, 1)
xline = np.linspace(-15, 30, 100)
ax.plot(xline, z[0]*xline + z[1], color=CB_REDDISH, linewidth=1.0,
        linestyle='-', alpha=0.6, zorder=2)

# Correlation annotation
ax.text(0.97, 0.04, f'r = {r_pearson:.2f}, ρ = {r_spearman:.2f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=12, color=DARK, fontfamily=BODY_FONT,
        bbox=dict(boxstyle='round,pad=0.2', facecolor=LIGHT, edgecolor=MID_GRAY, linewidth=0.3))

# Axis labels
ax.set_xlabel('Expert Persona Δ (%)', fontsize=8)
ax.set_ylabel('% Routed to LoRA', fontsize=8)
ax.set_xlim(-15, 30)
ax.set_ylim(-5, 110)

# Compact legend — lower left (avoid overlapping quadrant labels)
leg = ax.legend(fontsize=6, loc='center left', framealpha=0.9,
                edgecolor=MID_GRAY, handlelength=0.6, handletextpad=0.2,
                borderpad=0.2, labelspacing=0.2,
                bbox_to_anchor=(0.0, 0.5))

# Spines + grid
ax.grid(True, axis='both', alpha=0.15, color=MID_GRAY, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(MID_GRAY)
ax.spines['bottom'].set_color(MID_GRAY)
ax.tick_params(length=0, labelsize=7)

# Quadrant labels
ax.text(-7, 105, 'Expert Persona hurts', fontsize=7, color=CB_VERMILLION, 
        fontweight='bold', ha='center', va='top', alpha=0.6)
ax.text(15, 105, 'Expert Persona helps', fontsize=7, color=CB_GREEN,
        fontweight='bold', ha='center', va='top', alpha=0.6)

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'paper', 'latex', 'figures', 'routing_correlation.pdf')
out_path = os.path.abspath(out_path)
fig.savefig(out_path, bbox_inches='tight', facecolor=LIGHT, dpi=300)
out_png = out_path.replace('.pdf', '.png')
fig.savefig(out_png, bbox_inches='tight', facecolor=LIGHT, dpi=300)
print(f'Saved: {out_path}')
print(f'Saved: {out_png}')
