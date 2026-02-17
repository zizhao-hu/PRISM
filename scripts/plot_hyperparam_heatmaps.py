#!/usr/bin/env python3
"""Finetune vs Distill overlaid — same color per N, solid=FT, dashed=Distill."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BATCH_SIZE = 4
data_sizes = [50, 100, 200, 500]
epochs = [2, 4, 6, 8, 10]
colors = {50: '#e74c3c', 100: '#f39c12', 200: '#2ecc71', 500: '#3498db'}
markers_ft = {50: 'o', 100: 's', 200: 'D', 500: '^'}
markers_dl = {50: 'o', 100: 's', 200: 'D', 500: '^'}

# ── Finetune ──
ft_safety = {
    (50,2):64.0,(50,4):59.0,(50,6):59.5,(50,8):61.8,(50,10):61.5,
    (100,2):54.2,(100,4):56.0,(100,6):55.5,(100,8):59.2,(100,10):58.8,
    (200,2):57.2,(200,4):63.0,(200,6):69.0,(200,8):70.2,(200,10):72.2,
    (500,2):54.2,(500,4):62.3,(500,6):60.5,(500,8):61.0,(500,10):61.5,
}
ft_wr = {
    (50,2):40.0,(50,4):36.7,(50,6):26.7,(50,8):40.0,(50,10):36.7,
    (100,2):40.0,(100,4):36.7,(100,6):43.3,(100,8):43.3,(100,10):40.0,
    (200,2):36.7,(200,4):26.7,(200,6):50.0,(200,8):50.0,(200,10):43.3,
    (500,2):53.3,(500,4):33.3,(500,6):30.0,(500,8):36.7,(500,10):50.0,
}
ft_kl = {
    (50,2):0.013,(50,4):0.058,(50,6):0.089,(50,8):0.104,(50,10):0.111,
    (100,2):0.063,(100,4):0.125,(100,6):0.165,(100,8):0.193,(100,10):0.203,
    (200,2):0.117,(200,4):0.174,(200,6):0.243,(200,8):0.306,(200,10):0.345,
    (500,2):0.135,(500,4):0.191,(500,6):0.312,(500,8):0.448,(500,10):0.523,
}

# ── Distill ──
dl_safety = {
    (50,2):67.5,(50,4):67.8,(50,6):66.8,(50,8):68.2,(50,10):67.2,
    (100,2):68.2,(100,4):69.8,(100,6):70.2,(100,8):69.0,(100,10):69.2,
    (200,2):68.2,(200,4):61.5,(200,6):47.2,(200,8):61.8,(200,10):69.0,
    (500,2):68.8,(500,4):65.2,(500,6):60.2,(500,8):55.0,(500,10):58.2,
}
dl_wr = {
    (50,2):50.0,(50,4):45.0,(50,6):52.0,(50,8):40.0,(50,10):52.0,
    (100,2):50.0,(100,4):44.0,(100,6):40.0,(100,8):46.0,(100,10):41.0,
    (200,2):48.0,(200,4):36.0,(200,6):50.0,(200,8):46.0,(200,10):49.0,
    (500,2):46.0,(500,4):53.0,(500,6):39.0,(500,8):43.0,(500,10):51.0,
}
dl_kl = {
    (50,2):0.0007,(50,4):0.0008,(50,6):0.0010,(50,8):0.0017,(50,10):0.0012,
    (100,2):0.0007,(100,4):0.0010,(100,6):0.0018,(100,8):0.0016,(100,10):0.0011,
    (200,2):0.0010,(200,4):0.0083,(200,6):0.0090,(200,8):0.0056,(200,10):0.0036,
    (500,2):0.0014,(500,4):0.0069,(500,6):0.0103,(500,8):0.0075,(500,10):0.0046,
}

def get_steps(n, ep):
    return ep * n // BATCH_SIZE

# ═══════════════════════════════════════
# Plot 1: 3 metrics, all N overlaid
# ═══════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

metrics = [
    (ft_safety, dl_safety, 'Refusal Rate %', 'Safety (Refusal Rate %) ↑'),
    (ft_wr, dl_wr, 'Win Rate %', 'Utility (Win Rate vs Base %) ↑'),
    (ft_kl, dl_kl, 'KL Divergence', 'Model Drift (KL Divergence) ↓'),
]

for ax, (ft_data, dl_data, ylabel, title) in zip(axes, metrics):
    for n in data_sizes:
        steps = [get_steps(n, ep) for ep in epochs]
        ft_vals = [ft_data[(n, ep)] for ep in epochs]
        dl_vals = [dl_data[(n, ep)] for ep in epochs]
        ax.plot(steps, ft_vals, color=colors[n], marker=markers_ft[n], markersize=6,
                linewidth=2, linestyle='-', zorder=3, alpha=0.85)
        ax.plot(steps, dl_vals, color=colors[n], marker=markers_dl[n], markersize=6,
                linewidth=2, linestyle='--', zorder=3, alpha=0.85)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

# Custom legend: color = N, line style = method
legend_elements = []
for n in data_sizes:
    legend_elements.append(mlines.Line2D([], [], color=colors[n], marker=markers_ft[n],
                                          markersize=6, linewidth=2, label=f'N={n}'))
legend_elements.append(mlines.Line2D([], [], color='gray', linewidth=2, linestyle='-', label='Finetune'))
legend_elements.append(mlines.Line2D([], [], color='gray', linewidth=2, linestyle='--', label='Distill'))

fig.legend(handles=legend_elements, loc='upper center', ncol=6, frameon=True,
           fancybox=True, edgecolor='#ccc', fontsize=10, bbox_to_anchor=(0.5, 1.02))
fig.suptitle('Hyperparameter Search: Finetune (solid) vs Distill (dashed)', fontsize=14,
             fontweight='bold', y=1.08)
plt.tight_layout()
plt.savefig('hyperparam_ft_vs_distill_overlay.png', dpi=150, bbox_inches='tight')
plt.savefig('hyperparam_ft_vs_distill_overlay.pdf', bbox_inches='tight')
print('Saved overlay plot')

# ═══════════════════════════════════════
# Plot 2: Per-N subplots (4 rows × 3 cols)
# ═══════════════════════════════════════
fig2, axes2 = plt.subplots(4, 3, figsize=(17, 14))

for i, n in enumerate(data_sizes):
    for j, (ft_data, dl_data, ylabel, title) in enumerate(metrics):
        ax = axes2[i, j]
        steps = [get_steps(n, ep) for ep in epochs]
        ft_vals = [ft_data[(n, ep)] for ep in epochs]
        dl_vals = [dl_data[(n, ep)] for ep in epochs]
        
        ax.plot(steps, ft_vals, color='#e74c3c', marker='o', markersize=7,
                linewidth=2.5, linestyle='-', label='Finetune', zorder=3)
        ax.plot(steps, dl_vals, color='#3498db', marker='s', markersize=7,
                linewidth=2.5, linestyle='--', label='Distill', zorder=3)
        
        ax.set_xlabel('Training Steps')
        if i == 0:
            ax.set_title(title, fontweight='bold', fontsize=11)
        if j == 0:
            ax.set_ylabel(f'N={n}\n{ylabel}', fontweight='bold')
        else:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle='--')
        if i == 0 and j == 0:
            ax.legend(frameon=True, fancybox=True, edgecolor='#ccc', fontsize=9)

fig2.suptitle('Finetune vs Distill — Per Data Size', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('hyperparam_ft_vs_distill_per_n.png', dpi=150, bbox_inches='tight')
plt.savefig('hyperparam_ft_vs_distill_per_n.pdf', bbox_inches='tight')
print('Saved per-N plot')
