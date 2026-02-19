"""Lambda sweep: single compact figure with dual y-axis.
Uses Anthropic brand styling from style.md.

  Left Y:  Safety (RR%) and Utility (Win Rate%)
  Right Y: KL Divergence
  Legend at bottom.

Run:  python scripts/plot_lambda_sweep.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Brand Colors (style.md) ───────────────────────────────────
DARK      = '#141413'
LIGHT     = '#faf9f5'
MID_GRAY  = '#b0aea5'
LIGHT_GRAY= '#e8e6dc'
ORANGE    = '#d97757'   # primary accent
BLUE      = '#6a9bcc'   # secondary accent
GREEN     = '#788c5d'   # tertiary accent

# ── Font config (Poppins / Lora with fallbacks) ───────────────
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
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
lambdas = [0, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0]
safety_rr = [72.5, 71.8, 71.2, 70.8, 70.2, 69.5, 69.2, 68.2]
win_rate  = [42.0, 44.0, 46.0, 48.0, 50.0, 50.5, 50.8, 51.0]
kl_div    = [0.0018, 0.0016, 0.0015, 0.0014, 0.0012, 0.0010, 0.0008, 0.0006]

# ── Figure ────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(3.8, 2.6))
fig.patch.set_facecolor(LIGHT)
ax1.set_facecolor(LIGHT)

# Left axis: Safety & Utility
l1, = ax1.plot(lambdas, safety_rr, 's-', color=ORANGE, lw=1.8, ms=5, markeredgecolor=DARK, markeredgewidth=0.3)
l2, = ax1.plot(lambdas, win_rate, 'D-', color=GREEN, lw=1.8, ms=5, markeredgecolor=DARK, markeredgewidth=0.3)

ax1.axhline(y=70.0, color=ORANGE, ls=':', lw=0.7, alpha=0.4)
ax1.axhline(y=50.0, color=GREEN, ls=':', lw=0.7, alpha=0.4)

ax1.set_xlabel('λ (Recycled Sample Weight)', fontfamily=BODY_FONT)
ax1.set_ylabel('Score (%)', fontfamily=BODY_FONT)
ax1.set_xlim(-0.15, 4.5)
ax1.set_ylim(35, 80)
ax1.grid(True, alpha=0.15, color=MID_GRAY)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_color(MID_GRAY)
ax1.spines['left'].set_color(MID_GRAY)
ax1.spines['bottom'].set_color(MID_GRAY)

# Right axis: KL
ax2 = ax1.twinx()
ax2.set_facecolor(LIGHT)
l3, = ax2.plot(lambdas, [k * 1000 for k in kl_div], 'o-', color=BLUE,
               lw=1.8, ms=5, markeredgecolor=DARK, markeredgewidth=0.3)
ax2.set_ylabel('KL Divergence (×10⁻³)', color=BLUE, fontfamily=BODY_FONT)
ax2.tick_params(axis='y', labelcolor=BLUE)
ax2.set_ylim(0, 2.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color(BLUE)
ax2.spines['left'].set_color(MID_GRAY)

# Legend at bottom
fig.legend([l1, l2, l3],
           ['Safety (Refusal Rate %)', 'Utility (Win Rate %)', 'KL Divergence (×10⁻³)'],
           loc='lower center', ncol=3, frameon=True, edgecolor=LIGHT_GRAY,
           facecolor=LIGHT, bbox_to_anchor=(0.5, -0.08), fontsize=7)

fig.tight_layout()

fig.savefig('lambda_sweep.pdf', bbox_inches='tight', facecolor=LIGHT)
print(f"Saved lambda_sweep.pdf")
