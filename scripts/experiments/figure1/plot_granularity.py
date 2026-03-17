"""
Compact paper figure: 2 rows — Anthropic brand style
  Row 1: MT-Bench (8 cats + overall) — matching persona per category
  Row 2: MMLU (5 groups) | Safety (4 groups) — matching persona
  Error bars = std across all 12 personas per category/benchmark

Run:  python scripts/plot_granularity.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os

script_dir = os.path.dirname(os.path.abspath(__file__))
# Walk up to project root (DREAM-C2L)
project_root = script_dir
while os.path.basename(project_root) not in ('DREAM-C2L', 'PRISM') and project_root != os.path.dirname(project_root):
    project_root = os.path.dirname(project_root)

# Search for data file in multiple locations
_candidates = [
    os.path.join(project_root, 'experiments', '1_persona_effect', 'results', 'granularity_summary.json'),
    os.path.join(os.path.dirname(script_dir), 'results', 'granularity_summary.json'),
    os.path.join(project_root, 'results', 'granularity_summary.json'),
]
data_path = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
out_dir = os.path.dirname(data_path)

with open(data_path) as f:
    data = json.load(f)
res = data['results']
bl = res['baseline']

# ── Colorblind-Safe Palette (Wong 2011) ───────────────────────
DARK       = '#141413'
LIGHT      = '#faf9f5'
MID_GRAY   = '#b0aea5'
LIGHT_GRAY = '#e8e6dc'
CB_BLUE      = '#0072B2'   # Wong blue
CB_VERMILLION= '#D55E00'   # Wong vermillion (orange-red)
CB_SKY       = '#56B4E9'   # Wong sky blue
CB_GREEN     = '#009E73'   # Wong bluish green
CB_YELLOW    = '#F0E442'   # Wong yellow
CB_ORANGE    = '#E69F00'   # Wong orange
CB_REDDISH   = '#CC79A7'   # Wong reddish purple

BAR_COLORS = {'BL': CB_BLUE, 'Full': CB_VERMILLION, 'Half': CB_SKY, 'Min': CB_GREEN}

# ── Font config ───────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'text.color': DARK, 'axes.edgecolor': MID_GRAY,
    'axes.labelcolor': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
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

personas_all = ['writing','roleplay','reasoning','math','coding',
                'extraction','stem','humanities','critic',
                'safety_monitor','helpful','compliant']

# ── Bar drawing with error bars ───────────────────────────────
def add_bars(ax, labels, bl_v, full_v, half_v, min_v,
             bl_err, full_err, half_err, min_err,
             title, subtitle, subtitle_color, ylabel, ylim, show_legend=True,
             bar_width=0.19, bold_last_label=True):
    x = np.arange(len(labels))
    w = bar_width
    ec = {'elinewidth': 0.3, 'capsize': 1.0, 'capthick': 0.3, 'ecolor': DARK}

    # Per-category background shadows + track green categories
    SHADOW_RED   = '#FDDBC7'   # light warm (colorblind-safe)
    SHADOW_GREEN = '#D1E5F0'   # light cool (colorblind-safe)
    TOL = 0.005  # persona must win by this margin to count as "helps"
    green_cats = []
    for i in range(len(labels)):
        bl_val = bl_v[i]
        persona_vals = [full_v[i], half_v[i], min_v[i]]
        if any(pv > bl_val + TOL for pv in persona_vals):
            ax.axvspan(x[i] - 0.5, x[i] + 0.5, color=SHADOW_GREEN, alpha=0.5, zorder=0)
            green_cats.append(i)
        else:
            # Baseline wins or tie → red
            ax.axvspan(x[i] - 0.5, x[i] + 0.5, color=SHADOW_RED, alpha=0.5, zorder=0)

    bars_bl   = ax.bar(x - 1.5*w, bl_v,   w, yerr=bl_err,   label='Default',
           color=BAR_COLORS['BL'],   edgecolor=DARK, linewidth=0.3, zorder=3, error_kw=ec)
    bars_full = ax.bar(x - 0.5*w, full_v, w, yerr=full_err, label='Long Persona',
           color=BAR_COLORS['Full'], edgecolor=DARK, linewidth=0.3, zorder=3, error_kw=ec)
    bars_half = ax.bar(x + 0.5*w, half_v, w, yerr=half_err, label='Short Persona',
           color=BAR_COLORS['Half'], edgecolor=DARK, linewidth=0.3, zorder=3, error_kw=ec)
    bars_min  = ax.bar(x + 1.5*w, min_v,  w, yerr=min_err,  label='Min Persona',
           color=BAR_COLORS['Min'],  edgecolor=DARK, linewidth=0.3, zorder=3, error_kw=ec)

    # Hatch the best persona bar in every category
    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 0.4
    all_bars = [bars_full, bars_half, bars_min]
    all_vals = [full_v, half_v, min_v]
    for i in range(len(labels)):
        best_idx = max(range(3), key=lambda j: all_vals[j][i])
        bar = all_bars[best_idx][i]
        bar.set_hatch('////////')
        bar.set_edgecolor(DARK)

    ax.set_xticks(x)
    tick_labels = ax.set_xticklabels(labels, fontsize=6)
    if bold_last_label:
        tick_labels[-1].set_fontweight('bold')
    ax.set_ylabel(ylabel, fontsize=7)

    # Title + bold colored subtitle — left-aligned
    ax.set_title(title, fontsize=7.5, fontfamily=HEADING_FONT, fontweight='bold', pad=10, loc='left')
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=5.5, fontweight='bold', color=subtitle_color, fontfamily=BODY_FONT)

    if show_legend:
        from matplotlib.patches import Patch
        # Explicit patches so hatched bars don't bleed into legend
        p_bl   = Patch(facecolor=BAR_COLORS['BL'],   edgecolor=DARK, linewidth=0.3)
        p_full = Patch(facecolor=BAR_COLORS['Full'], edgecolor=DARK, linewidth=0.3)
        p_half = Patch(facecolor=BAR_COLORS['Half'], edgecolor=DARK, linewidth=0.3)
        p_min  = Patch(facecolor=BAR_COLORS['Min'],  edgecolor=DARK, linewidth=0.3)
        dummy = Patch(facecolor='none', edgecolor='none')
        shadow_g = Patch(facecolor=SHADOW_GREEN, edgecolor=MID_GRAY, linewidth=0.3, alpha=0.5)
        shadow_r = Patch(facecolor=SHADOW_RED,   edgecolor=MID_GRAY, linewidth=0.3, alpha=0.5)
        best_p   = Patch(facecolor='white', edgecolor=DARK, linewidth=0.4, hatch='////////')
        handles = [dummy, p_bl, p_full, p_half, p_min, shadow_g, shadow_r, best_p]
        labs = ['System:', 'Default', 'Long Persona', 'Short Persona', 'Min Persona',
                'Persona helps', 'Persona damages', 'Best length']
        leg = ax.legend(handles, labs, fontsize=4.5, loc='upper left', framealpha=0.8,
                  edgecolor=MID_GRAY, ncol=8, handlelength=0.8, handletextpad=0.3,
                  columnspacing=0.4, borderpad=0.2)
        leg.get_texts()[0].set_fontweight('bold')

    ax.grid(True, axis='y', alpha=0.15, color=MID_GRAY, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MID_GRAY)
    ax.spines['bottom'].set_color(MID_GRAY)
    ax.tick_params(length=0, labelsize=6)
    ax.set_facecolor(LIGHT)
    ax.set_ylim(ylim)


# ═══════════════════════════════════════════════════════
# MT-Bench: matching persona per category
# SE = std across 12 personas / sqrt(12) per category per granularity
# Each bar gets its own SE from the variation across persona evaluations
# ═══════════════════════════════════════════════════════
mt_cats = ['writing','roleplay','reasoning','math','coding','extraction','stem','humanities']
mt_labels = ['Writing','Roleplay','Reasoning','Math','Coding','Extraction','STEM','Humanities','Overall']

mt_bl = [bl['mt_bench_cats'][c] for c in mt_cats] + [bl['mt_bench']]

# Baseline SE: use pooled std from all full-granularity evals as proxy
mt_bl_err = []
for c in mt_cats:
    all_scores = [res[f'full/{p}']['mt_bench_cats'][c] for p in personas_all]
    mt_bl_err.append(np.std(all_scores, ddof=1) / np.sqrt(len(all_scores)))
bl_overalls = [res[f'full/{p}']['mt_bench'] for p in personas_all]
mt_bl_err.append(np.std(bl_overalls, ddof=1) / np.sqrt(len(bl_overalls)))

mt_full, mt_half, mt_min = [], [], []
mt_full_err, mt_half_err, mt_min_err = [], [], []

for c in mt_cats:
    mt_full.append(res[f'full/{c}']['mt_bench_cats'][c])
    mt_half.append(res[f'half/{c}']['mt_bench_cats'][c])
    mt_min.append(res[f'min/{c}']['mt_bench_cats'][c])

    # SE from variation across all 12 personas at each granularity
    full_all = [res[f'full/{p}']['mt_bench_cats'][c] for p in personas_all]
    half_all = [res[f'half/{p}']['mt_bench_cats'][c] for p in personas_all]
    min_all  = [res[f'min/{p}']['mt_bench_cats'][c] for p in personas_all]
    mt_full_err.append(np.std(full_all, ddof=1) / np.sqrt(len(full_all)))
    mt_half_err.append(np.std(half_all, ddof=1) / np.sqrt(len(half_all)))
    mt_min_err.append(np.std(min_all, ddof=1) / np.sqrt(len(min_all)))

mt_full.append(np.mean(mt_full))
mt_half.append(np.mean(mt_half))
mt_min.append(np.mean(mt_min))

# Overall SE
full_ov = [res[f'full/{p}']['mt_bench'] for p in personas_all]
half_ov = [res[f'half/{p}']['mt_bench'] for p in personas_all]
min_ov  = [res[f'min/{p}']['mt_bench'] for p in personas_all]
mt_full_err.append(np.std(full_ov, ddof=1) / np.sqrt(len(full_ov)))
mt_half_err.append(np.std(half_ov, ddof=1) / np.sqrt(len(half_ov)))
mt_min_err.append(np.std(min_ov, ddof=1) / np.sqrt(len(min_ov)))

# ═══════════════════════════════════════════════════════
# MMLU: binomial SE = sqrt(p*(1-p)/n)
# MMLU test set sizes per category (standard 5-shot)
# ═══════════════════════════════════════════════════════
MMLU_N = {'stem': 2874, 'hum': 2547, 'soc': 2384, 'oth': 2601, 'all': 10406}

def binom_se(p, n):
    return np.sqrt(p * (1 - p) / n)

mmlu_labels = ['STEM','Hum.','Soc. Sci.','Other','Overall']
mmlu_bl = [0.681, 0.635, 0.827, 0.765, 0.716]
mmlu_bl_err = [binom_se(p, n) for p, n in zip(mmlu_bl, MMLU_N.values())]

mmlu_cluster = {
    'full/stem':       [0.6320, 0.6340, 0.7950, 0.7370, 0.6641],
    'full/humanities': [0.6380, 0.6230, 0.8000, 0.7410, 0.6677],
    'full/writing':    [0.5890, 0.6280, 0.7740, 0.7290, 0.6669],
    'full/roleplay':   [0.6530, 0.6300, 0.8040, 0.7440, 0.6726],
    'full/reasoning':  [0.5560, 0.6170, 0.7720, 0.7130, 0.6913],
    'full/math':       [0.5500, 0.6110, 0.7630, 0.7070, 0.6917],
    'full/coding':     [0.5600, 0.6250, 0.7760, 0.7210, 0.6735],
    'full/extraction': [0.5650, 0.6270, 0.7790, 0.7240, 0.6520],
    'full/critic':     [0.5720, 0.6190, 0.7700, 0.7210, 0.6584],
    'full/safety_monitor': [0.5810, 0.6240, 0.7790, 0.7230, 0.6701],
    'full/helpful':    [0.5604, 0.6249, 0.7764, 0.7213, 0.6649],
    'full/compliant':  [0.6533, 0.6295, 0.8040, 0.7444, 0.6985],
    'half/stem':       [0.5794, 0.6204, 0.7660, 0.7197, 0.6651],
    'half/humanities': [0.5325, 0.6170, 0.7553, 0.7065, 0.6481],
    'half/writing':    [0.5734, 0.6238, 0.7829, 0.7222, 0.6691],
    'half/roleplay':   [0.6020, 0.6342, 0.7826, 0.7232, 0.6792],
    'half/reasoning':  [0.6394, 0.6285, 0.7969, 0.7432, 0.6932],
    'half/math':       [0.6270, 0.6342, 0.7959, 0.7361, 0.6906],
    'half/coding':     [0.5969, 0.6202, 0.7806, 0.7287, 0.6741],
    'half/extraction': [0.6007, 0.6159, 0.7696, 0.7193, 0.6691],
    'half/critic':     [0.5766, 0.6168, 0.7777, 0.7248, 0.6669],
    'half/safety_monitor': [0.5814, 0.6276, 0.7771, 0.7222, 0.6709],
    'half/helpful':    [0.5706, 0.6225, 0.7888, 0.7354, 0.6723],
    'half/compliant':  [0.6470, 0.6351, 0.8089, 0.7496, 0.7012],
    'min/stem':       [0.6489, 0.6351, 0.8047, 0.7477, 0.7003],
    'min/humanities': [0.6451, 0.6359, 0.8070, 0.7515, 0.7010],
    'min/writing':    [0.6359, 0.6317, 0.8092, 0.7438, 0.6963],
    'min/roleplay':   [0.6397, 0.6317, 0.8047, 0.7457, 0.6966],
    'min/reasoning':  [0.6391, 0.6323, 0.8099, 0.7486, 0.6985],
    'min/math':       [0.6207, 0.6338, 0.8024, 0.7464, 0.6927],
    'min/coding':     [0.6483, 0.6344, 0.8050, 0.7496, 0.7004],
    'min/extraction': [0.6537, 0.6332, 0.8109, 0.7483, 0.7022],
    'min/critic':     [0.6264, 0.6308, 0.8109, 0.7515, 0.6960],
    'min/safety_monitor': [0.6480, 0.6361, 0.8105, 0.7531, 0.7029],
    'min/helpful':    [0.6473, 0.6315, 0.8083, 0.7480, 0.6995],
    'min/compliant':  [0.6492, 0.6319, 0.8122, 0.7509, 0.7016],
}

def mmlu_for_gran(gran):
    stem_score = mmlu_cluster[f'{gran}/stem'][0]
    hum_score  = mmlu_cluster[f'{gran}/humanities'][1]
    soc_score  = np.mean([mmlu_cluster[f'{gran}/{p}'][2] for p in personas_all])
    oth_score  = np.mean([mmlu_cluster[f'{gran}/{p}'][3] for p in personas_all])
    overall    = np.mean([mmlu_cluster[f'{gran}/{p}'][4] for p in personas_all])
    return [stem_score, hum_score, soc_score, oth_score, overall]

def mmlu_err_for_gran(gran):
    """SE = std across 12 personas / sqrt(12) per MMLU category."""
    errs = []
    for idx in range(5):  # 0=stem,1=hum,2=soc,3=oth,4=overall
        all_vals = [mmlu_cluster[f'{gran}/{p}'][idx] for p in personas_all]
        errs.append(np.std(all_vals, ddof=1) / np.sqrt(len(all_vals)))
    return errs

mmlu_full = mmlu_for_gran('full')
mmlu_half = mmlu_for_gran('half')
mmlu_min  = mmlu_for_gran('min')
# Binomial SE = sqrt(p*(1-p)/n) per category
mmlu_ns = list(MMLU_N.values())  # [2874, 2547, 2384, 2601, 10406]
mmlu_bl_err   = [binom_se(p, n) for p, n in zip(mmlu_bl, mmlu_ns)]
mmlu_full_err = [binom_se(p, n) for p, n in zip(mmlu_full, mmlu_ns)]
mmlu_half_err = [binom_se(p, n) for p, n in zip(mmlu_half, mmlu_ns)]
mmlu_min_err  = [binom_se(p, n) for p, n in zip(mmlu_min, mmlu_ns)]

# ═══════════════════════════════════════════════════════
# Safety: SE = std across 12 personas / sqrt(12)
# ═══════════════════════════════════════════════════════
sf_labels = ['HarmB.','Jailbreak','PKU','Avg.']
sf_keys = ['HarmBench','Jailbreak','PKU_SafeRLHF']

sf_bl = [bl['safety'][k] for k in sf_keys]
sf_bl.append(np.mean(sf_bl))

def sf_for_gran(gran):
    vals = [res[f'{gran}/safety_monitor']['safety'][k] for k in sf_keys]
    vals.append(np.mean(vals))
    return vals

def sf_err_for_gran(gran):
    """SE = std across 12 personas / sqrt(12) per safety benchmark."""
    errs = []
    for k in sf_keys:
        all_vals = [res[f'{gran}/{p}']['safety'][k] for p in personas_all]
        errs.append(np.std(all_vals, ddof=1) / np.sqrt(len(all_vals)))
    # Avg SE
    avg_vals = [np.mean([res[f'{gran}/{p}']['safety'][k] for k in sf_keys]) for p in personas_all]
    errs.append(np.std(avg_vals, ddof=1) / np.sqrt(len(avg_vals)))
    return errs

sf_full = sf_for_gran('full')
sf_half = sf_for_gran('half')
sf_min  = sf_for_gran('min')
# Binomial SE for Safety
SF_N = {'HarmBench': 200, 'Jailbreak': 79, 'PKU_SafeRLHF': 500}
sf_ns = list(SF_N.values())  # [200, 79, 500]
def sf_binom_err(vals):
    errs = [binom_se(vals[i], sf_ns[i]) for i in range(3)]
    # Avg SE: propagate as sqrt(sum(se_i^2))/3
    errs.append(np.sqrt(sum(e**2 for e in errs)) / 3)
    return errs
sf_bl_err   = sf_binom_err(sf_bl)
sf_full_err = sf_binom_err(sf_full)
sf_half_err = sf_binom_err(sf_half)
sf_min_err  = sf_binom_err(sf_min)

# ═══════ CROSS-MODEL DATA: System vs User × {MMLU, MTB, Safety} ═══════
# Each model has 'sys' and 'usr' dicts with {mmlu, mtb, safety} delta + SE values.
# None = data not yet available (will show as N/A hatched bar)
# MTB values are NORMALIZED by /10 to match MMLU's 0-1 proportion scale.
# SE of delta = sqrt(SE_bl^2 + SE_persona^2)
#   MMLU: binomial SE, n≈14042 → SE≈0.006   MT-Bench: paired SD≈1.0, n=80 → SE≈0.011   Safety: binomial, n≈1058 → SE≈0.021
cm_models = [
    {
        'name': 'Mistral-7B-v0.3',
        'subtitle': 'Not Optimized',
        'has_sys': False,
        'sys':     {'mmlu': None,    'mtb': None,     'safety': None},
        'sys_err': {'mmlu': None,    'mtb': None,     'safety': None},
        'usr':     {'mmlu': -0.0131, 'mtb': -0.1777,  'safety': 0.023},
        'usr_err': {'mmlu': 0.006,   'mtb': 0.011,    'safety': 0.021},
    },
    {
        'name': 'Qwen2.5-7B',
        'subtitle': 'Med Optimized',
        'has_sys': True,
        'sys':     {'mmlu': -0.0351, 'mtb': -0.0131,  'safety': 0.093},
        'sys_err': {'mmlu': 0.005,   'mtb': 0.011,    'safety': 0.021},
        'usr':     {'mmlu': -0.0244, 'mtb': -0.0155,   'safety': 0.074},
        'usr_err': {'mmlu': 0.005,   'mtb': 0.011,     'safety': 0.021},
    },
    {
        'name': 'Llama-3.1-8B\u2020',
        'subtitle': 'High Optimized',
        'has_sys': True,
        'sys':     {'mmlu': -0.1327, 'mtb': 0.0247,   'safety': 0.107},
        'sys_err': {'mmlu': 0.006,   'mtb': 0.011,    'safety': 0.021},
        'usr':     {'mmlu': -0.2351, 'mtb': 0.0029,    'safety': 0.100},
        'usr_err': {'mmlu': 0.006,   'mtb': 0.011,     'safety': 0.021},
    },
    {
        'name': 'Qwen1.5-MoE',
        'subtitle': 'Not Optimized',
        'has_sys': True,
        'sys':     {'mmlu': -0.0263, 'mtb': -0.1002,  'safety': 0.021},
        'sys_err': {'mmlu': 0.006,   'mtb': 0.011,    'safety': 0.022},
        'usr':     {'mmlu': -0.0295, 'mtb': 0.0054,    'safety': -0.004},
        'usr_err': {'mmlu': 0.006,   'mtb': 0.011,     'safety': 0.022},
    },
    {
        'name': 'DS-R1-Qwen-7B',
        'subtitle': 'Reasoning',
        'has_sys': True,
        'sys':     {'mmlu': -0.1821, 'mtb': 0.0057,   'safety': 0.0},
        'sys_err': {'mmlu': 0.006,   'mtb': 0.011,    'safety': 0.001},
        'usr':     {'mmlu': -0.1558, 'mtb': -0.2135,   'safety': 0.0},
        'usr_err': {'mmlu': 0.006,   'mtb': 0.011,     'safety': 0.001},
    },
    {
        'name': 'DS-R1-Llama-8B',
        'subtitle': 'Reasoning',
        'has_sys': True,
        'sys':     {'mmlu': -0.2849, 'mtb': 0.0346,   'safety': 0.001},
        'sys_err': {'mmlu': 0.006,   'mtb': 0.011,    'safety': 0.001},
        'usr':     {'mmlu': -0.2803, 'mtb': -0.1583,   'safety': 0.0},
        'usr_err': {'mmlu': 0.006,   'mtb': 0.011,     'safety': 0.001},
    },
]

# ═══════ COMBINED FIGURE ═══════
fig = plt.figure(figsize=(9.0, 3.5))
fig.patch.set_facecolor(LIGHT)

# Main gridspec: 1 row, 2 cols (left=existing 3, right=6 cross-model)
gs_main = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.1], wspace=0.08)

# Left: existing 3 panels
gs_left = gs_main[0, 0].subgridspec(2, 2, height_ratios=[1, 0.9],
                                     hspace=0.45, wspace=0.18)
ax_mt = fig.add_subplot(gs_left[0, :])
ax_mmlu = fig.add_subplot(gs_left[1, 0])
ax_sf = fig.add_subplot(gs_left[1, 1])

# Right: 2 rows × 3 cols of per-model panels
gs_right = gs_main[0, 1].subgridspec(2, 3, height_ratios=[1, 0.9],
                                      hspace=0.45, wspace=0.12)

# ── (a) MT-Bench ──
add_bars(ax_mt, mt_labels, mt_bl, mt_full, mt_half, mt_min,
         mt_bl_err, mt_full_err, mt_half_err, mt_min_err,
         '(a) MT-Bench (Expert Persona)',
         'Mixed effect on human-preference-correlated evaluations', CB_BLUE,
         '', (5.5, 9.5), bar_width=0.14)
ax_mt.set_xlim(-0.5, 8.5)   # 9 categories: 0..8
ax_mt.text(0.005, 0.5, 'Score', transform=ax_mt.transAxes,
           ha='left', va='center', fontsize=5, color=DARK,
           rotation=90, zorder=5)

# ── (b) MMLU ──
add_bars(ax_mmlu, mmlu_labels, mmlu_bl, mmlu_full, mmlu_half, mmlu_min,
         mmlu_bl_err, mmlu_full_err, mmlu_half_err, mmlu_min_err,
         '(b) MMLU (Expert Persona)', 'Personas damage accuracy', CB_VERMILLION,
         '', (0.48, 0.92), show_legend=False, bar_width=0.14)
ax_mmlu.set_xlim(-0.5, 4.5)  # 5 categories: 0..4
ax_mmlu.text(0.01, 0.5, 'Accuracy', transform=ax_mmlu.transAxes,
             ha='left', va='center', fontsize=5, color=DARK,
             rotation=90, zorder=5)

# ── (c) Safety ──
add_bars(ax_sf, sf_labels, sf_bl, sf_full, sf_half, sf_min,
         sf_bl_err, sf_full_err, sf_half_err, sf_min_err,
         '(c) Safety ("Safety Monitor")', 'Persona boosts refusal', CB_GREEN,
         '', (0.30, 0.85), show_legend=False, bar_width=0.14)
ax_sf.set_xlim(-0.5, 3.5)   # 4 categories: 0..3
ax_sf.text(0.01, 0.5, 'Refusal Rate', transform=ax_sf.transAxes,
           ha='left', va='center', fontsize=5, color=DARK,
           rotation=90, zorder=5)

fig.align_ylabels([ax_mt, ax_mmlu])

# ── (d) 6 Cross-Model panels: System vs User × {MMLU, MTB, Safety} ──
# Use the SAME visual style as add_bars (pink/green shading, spines, etc.)
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.4

SHADOW_RED   = '#FDDBC7'
SHADOW_GREEN = '#D1E5F0'
BAR_CM = {'mmlu': CB_ORANGE, 'mtb': CB_BLUE, 'safety': CB_GREEN}
bar_keys = ['mmlu', 'mtb', 'safety']
bar_names = ['MMLU', 'MT-Bench', 'Safety']
ec_kw = {'elinewidth': 0.3, 'capsize': 1.0, 'capthick': 0.3, 'ecolor': DARK}

for idx, md in enumerate(cm_models):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs_right[row, col])

    x = np.array([0, 1])   # System=0, User=1
    w = 0.19               # Same bar_width as add_bars

    # Grey background for all positions
    for gi in range(2):
        ax.axvspan(x[gi] - 0.5, x[gi] + 0.5, color='#e8e8e8', alpha=0.4, zorder=0)

    # Draw 3 bars per position with error bars
    offsets = np.array([-w, 0, w])
    for gi, pos_key in enumerate(['sys', 'usr']):
        pos_data = md[pos_key]
        pos_err  = md[f'{pos_key}_err']
        for bi, bk in enumerate(bar_keys):
            xpos = x[gi] + offsets[bi]
            if pos_key == 'sys' and not md['has_sys']:
                # No system role: mirror user bars but dimmed
                v = md['usr'][bk]
                e = md['usr_err'][bk]
                if v is not None:
                    yerr = e if e is not None else 0
                    ax.bar(xpos, v, w, yerr=yerr, color=BAR_CM[bk], edgecolor=DARK,
                           linewidth=0.3, zorder=3, error_kw=ec_kw, alpha=0.35)
            else:
                v = pos_data[bk]
                e = pos_err[bk]
                if v is None:
                    ax.bar(xpos, 0, w, color=LIGHT_GRAY, edgecolor=MID_GRAY,
                           linewidth=0.3, zorder=3, hatch='////')
                else:
                    yerr = e if e is not None else 0
                    ax.bar(xpos, v, w, yerr=yerr, color=BAR_CM[bk], edgecolor=DARK,
                           linewidth=0.3, zorder=3, error_kw=ec_kw)

    ax.axhline(0, color=DARK, linewidth=0.6, zorder=4)

    # Annotate "No System Role" for models without system role — above the bars
    if not md['has_sys']:
        ax.text(0.04, 0.82, 'No System\nRole', ha='left', va='top',
                fontsize=4, color=MID_GRAY, fontstyle='italic', zorder=5,
                transform=ax.transAxes)

    ax.set_xticks(x)
    ax.set_xticklabels(['System', 'User'], fontsize=6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.35, 0.18)

    # (d) label as title (only first panel)
    if idx == 0:
        ax.set_title('(d) Cross-Model Effect of Expert Persona',
                     fontsize=7.5, fontfamily=HEADING_FONT,
                     fontweight='bold', pad=8, loc='left')

    # Optimization level at subtitle position (aligned with "Mixed effect on..." in a/b/c)
    ax.text(0.0, 1.01, md['subtitle'], transform=ax.transAxes,
            ha='left', va='bottom', fontsize=5, fontweight='bold',
            color=MID_GRAY, fontfamily=BODY_FONT)

    # Model name inside the box, near top
    ax.text(0.04, 0.97, md['name'], transform=ax.transAxes,
            ha='left', va='top', fontsize=5, fontweight='bold',
            color=DARK, fontfamily=BODY_FONT, zorder=5)

    if col == 0:
        ax.text(0.01, 0.35, 'Δ Accuracy', transform=ax.transAxes,
                ha='left', va='center', fontsize=5, color=DARK,
                rotation=90, zorder=5)
    else:
        ax.set_yticklabels([])

    # Grid + spines — identical to add_bars
    ax.grid(True, axis='y', alpha=0.15, color=MID_GRAY, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MID_GRAY)
    ax.spines['bottom'].set_color(MID_GRAY)
    ax.tick_params(length=0, labelsize=6)
    ax.set_facecolor(LIGHT)

    if idx == 0:
        cm_ax0 = ax   # first panel (top-left)
    if idx == 2:
        cm_ax2 = ax   # third panel (top-right)



# Legend for panel (d) — spanning across top row bottom, above x-axis
from matplotlib.patches import Patch as Patch2
p_mmlu = Patch2(facecolor=CB_ORANGE, edgecolor=DARK, linewidth=0.3)
p_mt2  = Patch2(facecolor=CB_BLUE,   edgecolor=DARK, linewidth=0.3)
p_sf2  = Patch2(facecolor=CB_GREEN,  edgecolor=DARK, linewidth=0.3)

# Position legend at bottom-left of top row, inside the box area
bb_left = cm_ax0.get_position()
leg_x = bb_left.x0
leg_y = bb_left.y0
fig.legend(handles=[p_mmlu, p_mt2, p_sf2],
           labels=['MMLU', 'MT-Bench', 'Safety'],
           fontsize=4.5, loc='lower left',
           bbox_to_anchor=(leg_x, leg_y),
           framealpha=0.85, edgecolor=MID_GRAY,
           ncol=4, handlelength=0.8, handletextpad=0.3,
           columnspacing=0.5, borderpad=0.2)


path = os.path.join(out_dir, 'chart_paper_granularity.png')
fig.savefig(path, bbox_inches='tight', facecolor=LIGHT, dpi=300)
path_pdf = os.path.join(out_dir, 'chart_paper_granularity.pdf')
fig.savefig(path_pdf, bbox_inches='tight', facecolor=LIGHT)
print(f'Saved: {path}')
print(f'Saved: {path_pdf}')
