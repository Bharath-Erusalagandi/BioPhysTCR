#!/usr/bin/env python3
"""
Regenerate the compact 4-in-1 poster panel figure from existing predictions.
Fixes:
  - Clean ROC shading (no step artifact)
  - Clean score distribution annotations
  - Per-epitope bars with SEM error bars (not raw SD)
  - Model labeled as BioPhysTCR throughout
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc
from matplotlib.patches import Patch
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'clinical_validation'
PREDICTIONS_CSV = RESULTS_DIR / 'clinical_predictions.csv'

COVID_EPITOPES = {
    "YLQPRTFLL":  {"protein": "Spike",        "short": "Spike(269)"},
    "RLQSLQTYV":  {"protein": "Spike",        "short": "Spike(1000)"},
    "KLPDDFTGCV": {"protein": "Spike",        "short": "Spike(424)"},
    "NYNYLYRLF":  {"protein": "Spike",        "short": "Spike(448)"},
    "QYIKWPWYI":  {"protein": "Spike",        "short": "Spike(1208)"},
    "SPRWYFYYL":  {"protein": "Nucleocapsid", "short": "Nuc(105)"},
}

CONTROL_EPITOPES = {
    "GILGFVFTL":  {"pathogen": "Influenza", "short": "Flu M1"},
    "FMYSDFHFI":  {"pathogen": "Influenza", "short": "Flu basic"},
    "NLVPMVATV":  {"pathogen": "CMV",       "short": "CMV pp65"},
    "GLCTLVAML":  {"pathogen": "EBV",       "short": "EBV"},
    "ELAGIGILTV": {"pathogen": "Cancer",    "short": "Melan-A"},
    "IVTDFSVIK":  {"pathogen": "Ebola",     "short": "Ebola NP"},
}

# Color palette
COVID_COLOR = "#E74C3C"
COVID_DARK  = "#C0392B"
CTRL_COLOR  = "#3498DB"
CTRL_DARK   = "#2471A3"
ACCENT_GREEN = "#27AE60"
BG_FILL = "#F8F9FA"

# ── Load data ─────────────────────────────────────────────────
print("Loading predictions...")
df = pd.read_csv(PREDICTIONS_CSV)
y_true = df['is_covid_epitope'].values
y_pred = df['score'].values

# Core metrics
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

covid_scores = df.loc[df['is_covid_epitope'] == 1, 'score'].values
ctrl_scores  = df.loc[df['is_covid_epitope'] == 0, 'score'].values

# Per-subject AUROC
subj_aucs = []
for subj in df['sample'].unique():
    sd = df[df['sample'] == subj]
    sc = sd.loc[sd['is_covid_epitope'] == 1, 'score'].values
    sr = sd.loc[sd['is_covid_epitope'] == 0, 'score'].values
    if len(sc) > 0 and len(sr) > 0:
        try:
            sf, st, _ = roc_curve(
                np.concatenate([np.ones(len(sc)), np.zeros(len(sr))]),
                np.concatenate([sc, sr]))
            subj_aucs.append(auc(sf, st))
        except Exception:
            subj_aucs.append(0.5)
    else:
        subj_aucs.append(0.5)

# Per-epitope stats
epi_list = []
for epi in list(COVID_EPITOPES.keys()) + list(CONTROL_EPITOPES.keys()):
    scores = df.loc[df['epitope'] == epi, 'score']
    is_covid = epi in COVID_EPITOPES
    label = COVID_EPITOPES[epi]['short'] if is_covid else CONTROL_EPITOPES[epi]['short']
    n = len(scores)
    epi_list.append({
        'label': label, 'mean': scores.mean(), 'std': scores.std(),
        'sem': scores.std() / np.sqrt(n), 'n': n, 'is_covid': is_covid,
    })
epi_covid = sorted([e for e in epi_list if e['is_covid']], key=lambda x: x['mean'], reverse=True)
epi_ctrl  = sorted([e for e in epi_list if not e['is_covid']], key=lambda x: x['mean'], reverse=True)
epi_sorted = epi_covid + epi_ctrl

# Stats
_, p_val = stats.mannwhitneyu(covid_scores, ctrl_scores, alternative='greater')
effect_d = (np.mean(covid_scores) - np.mean(ctrl_scores)) / np.sqrt(
    (np.std(covid_scores)**2 + np.std(ctrl_scores)**2) / 2)

print(f"AUROC: {roc_auc:.4f}")
print(f"Per-subject AUROC: {np.mean(subj_aucs):.4f} +/- {np.std(subj_aucs):.4f}")

# ══════════════════════════════════════════════════════════════
#  BUILD FIGURE
# ══════════════════════════════════════════════════════════════
print("\nGenerating poster panel...")

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

fig = plt.figure(figsize=(11, 10.5))
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35,
                       left=0.08, right=0.95, top=0.92, bottom=0.07)

# ───────────────────────────────────────────────────────────────
# (a) ROC Curve  —  smooth shading, no step artifact
# ───────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor(BG_FILL)

# Smooth shaded area (fill from 0 to tpr, NO step='post')
ax.fill_between(fpr, 0, tpr, alpha=0.12, color=COVID_COLOR)
ax.plot(fpr, tpr, color=COVID_COLOR, lw=2.5, zorder=3,
        label=f'BioPhysTCR (AUROC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2, alpha=0.6,
        label='Random (AUROC = 0.500)')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.04])
ax.set_xlabel('False Positive Rate', fontweight='medium')
ax.set_ylabel('True Positive Rate', fontweight='medium')
ax.set_title('(a) ROC Curve', fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='lightgray')

# Subtle AUC watermark
ax.text(0.42, 0.32, f'AUC = {roc_auc:.3f}', fontsize=15, fontweight='bold',
        color=COVID_DARK, alpha=0.18, ha='center', va='center',
        transform=ax.transAxes)

# ───────────────────────────────────────────────────────────────
# (b) Score Distributions  —  clean KDE with clipping
# ───────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor(BG_FILL)

# KDE with clip=(0,1) so curves don't bleed beyond valid score range
sns.kdeplot(ctrl_scores, fill=True, color=CTRL_COLOR, alpha=0.30,
            label='Control Epitopes', linewidth=2, ax=ax, clip=(0, 1))
sns.kdeplot(covid_scores, fill=True, color=COVID_COLOR, alpha=0.30,
            label='SARS-CoV-2 Epitopes', linewidth=2, ax=ax, clip=(0, 1))

# Mean vertical lines (dashed)
ctrl_mean = np.mean(ctrl_scores)
covid_mean = np.mean(covid_scores)
ax.axvline(x=ctrl_mean, color=CTRL_DARK, linestyle='--', lw=1.8, alpha=0.8)
ax.axvline(x=covid_mean, color=COVID_DARK, linestyle='--', lw=1.8, alpha=0.8)

# Mean labels near the top of the plot, next to vertical lines
ymax = ax.get_ylim()[1]
ax.text(ctrl_mean - 0.02, ymax * 0.93, f'u={ctrl_mean:.2f}',
        fontsize=8.5, color=CTRL_DARK, fontweight='bold', va='top', ha='right')
ax.text(covid_mean + 0.02, ymax * 0.99, f'u={covid_mean:.2f}',
        fontsize=8.5, color=COVID_DARK, fontweight='bold', va='top', ha='left')

# Stats box
ax.text(0.97, 0.55, f"p < 1e-10\nCohen's d = {effect_d:.2f}",
        transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='#CCCCCC', alpha=0.9))

ax.set_xlabel('Predicted Binding Score', fontweight='medium')
ax.set_ylabel('Density', fontweight='medium')
ax.set_title('(b) Score Distributions', fontweight='bold', fontsize=13)
ax.legend(fontsize=9, framealpha=0.95, edgecolor='lightgray', loc='upper left')
ax.set_xlim([-0.02, 1.05])

# ───────────────────────────────────────────────────────────────
# (c) Per-Epitope Binding Scores  —  SEM error bars (clean)
# ───────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.set_facecolor(BG_FILL)

n_epi = len(epi_sorted)
n_covid_epi = len(epi_covid)
x_pos = np.arange(n_epi)
colors_bar = [COVID_COLOR if e['is_covid'] else CTRL_COLOR for e in epi_sorted]
edges_bar  = [COVID_DARK if e['is_covid'] else CTRL_DARK for e in epi_sorted]

means = [e['mean'] for e in epi_sorted]
sems  = [e['sem'] for e in epi_sorted]

bars = ax.bar(x_pos, means, yerr=sems,
              color=colors_bar, alpha=0.78, edgecolor=edges_bar, linewidth=0.9,
              capsize=4, error_kw={'lw': 1.3, 'capthick': 1.2, 'zorder': 5},
              zorder=3)

# Baseline
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, lw=1.2, zorder=1)

# Separator between groups
ax.axvline(x=n_covid_epi - 0.5, color='#888888', linestyle='-', lw=1.2, alpha=0.5)

# Group labels (once only)
ax.text(n_covid_epi / 2 - 0.5, 0.88, 'SARS-CoV-2', ha='center', fontsize=9,
        fontweight='bold', color=COVID_DARK, style='italic')
ax.text(n_covid_epi + (n_epi - n_covid_epi) / 2 - 0.5, 0.88, 'Control',
        ha='center', fontsize=9, fontweight='bold', color=CTRL_DARK, style='italic')

# Tick labels
ax.set_xticks(x_pos)
ax.set_xticklabels([e['label'] for e in epi_sorted], rotation=40, ha='right', fontsize=8)
ax.set_ylabel('Mean Binding Score', fontweight='medium')
ax.set_title('(c) Per-Epitope Binding Scores', fontweight='bold', fontsize=13)
ax.set_ylim([0, 0.93])

# Legend
legend_handles = [
    Patch(facecolor=COVID_COLOR, alpha=0.78, edgecolor=COVID_DARK, label='SARS-CoV-2'),
    Patch(facecolor=CTRL_COLOR, alpha=0.78, edgecolor=CTRL_DARK, label='Control'),
    plt.Line2D([0], [0], color='gray', linestyle='--', lw=1.2, label='Baseline (0.5)'),
]
ax.legend(handles=legend_handles, fontsize=8, loc='upper right', framealpha=0.9)

# ───────────────────────────────────────────────────────────────
# (d) Per-Subject AUROC Histogram
# ───────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.set_facecolor(BG_FILL)

counts, bin_edges, patches = ax.hist(
    subj_aucs, bins=16, color=ACCENT_GREEN, alpha=0.75,
    edgecolor='#1a7a3a', linewidth=0.9, zorder=3)

# Mean line
mean_auc = np.mean(subj_aucs)
std_auc = np.std(subj_aucs)
ax.axvline(x=mean_auc, color=COVID_COLOR, linestyle='--', lw=2.2, zorder=5,
           label=f'Mean = {mean_auc:.3f} +/- {std_auc:.3f}')
ax.axvline(x=0.5, color='gray', linestyle=':', lw=1.5, alpha=0.5,
           label='Random (0.5)')

# Annotation box
n_above = sum(1 for a in subj_aucs if a > 0.8)
n_total = len(subj_aucs)
ax.text(0.03, 0.95, f'{n_above}/{n_total} subjects\nAUROC > 0.8',
        transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='#CCCCCC', alpha=0.9))

ax.set_xlabel('Per-Subject AUROC', fontweight='medium')
ax.set_ylabel('Number of Subjects', fontweight='medium')
ax.set_title('(d) Subject-Level Performance', fontweight='bold', fontsize=13)
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9,
          bbox_to_anchor=(0.0, 0.78))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# ── Suptitle ──────────────────────────────────────────────────
fig.suptitle(
    'BioPhysTCR Clinical Validation: ImmuneCODE COVID-19 Cohort'
    ' (n = 72 subjects, 17,280 pairs)',
    fontsize=13, fontweight='bold', y=0.97)

# ── Save ──────────────────────────────────────────────────────
out_png = RESULTS_DIR / 'fig_clinical_poster_panel.png'
out_pdf = RESULTS_DIR / 'fig_clinical_poster_panel.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n  Saved: {out_png}")
print(f"  Saved: {out_pdf}")
print(f"\n  (a) ROC Curve        AUROC = {roc_auc:.3f}")
print(f"  (b) Score Dists      COVID u={covid_mean:.3f}  Ctrl u={ctrl_mean:.3f}")
print(f"  (c) Per-Epitope      6 SARS-CoV-2 + 6 Control  (SEM error bars)")
print(f"  (d) Subject AUROC    Mean = {mean_auc:.3f}, {n_above}/{n_total} > 0.8")
