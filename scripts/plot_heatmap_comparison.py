"""Generate compact 2×4 heatmap grid comparing all training modes.
Rows: Refusal Rate, Win Rate
Cols: Full Finetune, Full Distill, First-Token Finetune, First-Token Distill
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Data ──────────────────────────────────────────────────────
data_sizes = [50, 100, 200, 500]
epochs = [2, 4, 6, 8, 10]

# Full Finetune (hyperparam_v2)  – utility_limit=30
full_ft = {
    'rr': {(50,2):64.0,(50,4):59.0,(50,6):59.5,(50,8):61.8,(50,10):61.5,
           (100,2):54.2,(100,4):56.0,(100,6):55.5,(100,8):59.2,(100,10):58.8,
           (200,2):57.2,(200,4):63.0,(200,6):69.0,(200,8):70.2,(200,10):72.2,
           (500,2):54.2,(500,4):62.3,(500,6):60.5,(500,8):61.0,(500,10):61.5},
    'wr': {(50,2):40.0,(50,4):36.7,(50,6):26.7,(50,8):40.0,(50,10):36.7,
           (100,2):40.0,(100,4):36.7,(100,6):43.3,(100,8):43.3,(100,10):40.0,
           (200,2):36.7,(200,4):26.7,(200,6):50.0,(200,8):50.0,(200,10):43.3,
           (500,2):53.3,(500,4):33.3,(500,6):30.0,(500,8):36.7,(500,10):50.0},
}

# Full Distill (hyperparam_distill) – utility_limit=100
full_dl = {
    'rr': {(50,2):67.5,(50,4):67.8,(50,6):66.8,(50,8):68.2,(50,10):67.2,
           (100,2):68.2,(100,4):69.8,(100,6):70.2,(100,8):69.0,(100,10):69.2,
           (200,2):68.2,(200,4):61.5,(200,6):47.2,(200,8):61.8,(200,10):69.0,
           (500,2):68.8,(500,4):65.2,(500,6):60.2,(500,8):55.0,(500,10):58.2},
    'wr': {(50,2):50.0,(50,4):45.0,(50,6):52.0,(50,8):40.0,(50,10):52.0,
           (100,2):50.0,(100,4):44.0,(100,6):40.0,(100,8):46.0,(100,10):41.0,
           (200,2):48.0,(200,4):36.0,(200,6):50.0,(200,8):46.0,(200,10):49.0,
           (500,2):46.0,(500,4):53.0,(500,6):39.0,(500,8):43.0,(500,10):51.0},
}

# First-Token Finetune – utility_limit=100
ft1t = {
    'rr': {(50,2):68.0,(50,4):67.2,(50,6):80.5,(50,8):80.0,(50,10):77.5,
           (100,2):68.2,(100,4):73.0,(100,6):62.5,(100,8):58.5,(100,10):57.2,
           (200,2):67.0,(200,4):41.0,(200,6):38.8,(200,8):39.2,(200,10):37.0,
           (500,2):37.5,(500,4):26.0,(500,6):26.0,(500,8):23.5,(500,10):17.2},
    'wr': {(50,2):45.0,(50,4):47.0,(50,6):28.0,(50,8):30.0,(50,10):29.0,
           (100,2):42.0,(100,4):19.0,(100,6):17.0,(100,8):9.0,(100,10):11.0,
           (200,2):23.0,(200,4):11.0,(200,6):10.0,(200,8):11.0,(200,10):7.0,
           (500,2):19.0,(500,4):14.0,(500,6):9.0,(500,8):10.0,(500,10):9.0},
}

# First-Token Distill – utility_limit=100
dl1t = {
    'rr': {(50,2):69.8,(50,4):71.5,(50,6):70.2,(50,8):67.2,(50,10):68.8,
           (100,2):69.8,(100,4):69.2,(100,6):68.2,(100,8):64.8,(100,10):67.8,
           (200,2):69.8,(200,4):60.5,(200,6):66.8,(200,8):66.2,(200,10):66.5,
           (500,2):59.0,(500,4):62.3,(500,6):43.8,(500,8):64.5,(500,10):65.0},
    'wr': {(50,2):46.0,(50,4):35.0,(50,6):49.0,(50,8):42.0,(50,10):55.0,
           (100,2):44.0,(100,4):44.0,(100,6):46.0,(100,8):41.0,(100,10):51.0,
           (200,2):42.0,(200,4):57.0,(200,6):42.0,(200,8):42.0,(200,10):53.0,
           (500,2):47.0,(500,4):44.0,(500,6):49.0,(500,8):44.0,(500,10):42.0},
}

# ── Build matrices ────────────────────────────────────────────
def to_matrix(d, metric):
    mat = np.zeros((len(data_sizes), len(epochs)))
    for i, n in enumerate(data_sizes):
        for j, ep in enumerate(epochs):
            mat[i, j] = d[metric].get((n, ep), 0)
    return mat

methods = [
    ("SFT (Full)", full_ft),
    ("Distill (Full)", full_dl),
    ("SFT (1st Token)", ft1t),
    ("Distill (1st Token)", dl1t),
]

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 6), constrained_layout=True)

row_labels = ["Safety (RR% ↑)", "Utility (Win% ↑)"]
metrics = ['rr', 'wr']
# Colour map ranges
rr_vmin, rr_vmax = 15, 85
wr_vmin, wr_vmax = 5, 60

for col_idx, (title, data) in enumerate(methods):
    for row_idx, (metric, ylabel) in enumerate(zip(metrics, row_labels)):
        ax = axes[row_idx, col_idx]
        mat = to_matrix(data, metric)

        if metric == 'rr':
            vmin, vmax, cmap = rr_vmin, rr_vmax, 'RdYlGn'
        else:
            vmin, vmax, cmap = wr_vmin, wr_vmax, 'RdYlGn'

        sns.heatmap(mat, ax=ax, annot=True, fmt='.0f', cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    xticklabels=epochs, yticklabels=data_sizes if col_idx == 0 else False,
                    cbar=col_idx == 3,  # only rightmost gets colorbar
                    cbar_kws={'shrink': 0.8, 'label': ylabel if col_idx == 3 else ''},
                    annot_kws={'fontsize': 9, 'fontweight': 'bold'},
                    linewidths=0.5, linecolor='white')

        if row_idx == 0:
            ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        if col_idx == 0:
            ax.set_ylabel(f'{ylabel}\nData Size (N)', fontsize=9)
        else:
            ax.set_ylabel('')
        if row_idx == 1:
            ax.set_xlabel('Epochs', fontsize=9)
        else:
            ax.set_xlabel('')

fig.suptitle('PRISM Hyperparameter Search: Full Response vs First-Token Training',
             fontsize=14, fontweight='bold', y=1.02)

out_path = 'hyperparam_heatmap_comparison.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved to {out_path}")
