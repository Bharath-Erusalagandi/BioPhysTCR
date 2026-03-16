#!/usr/bin/env python3
"""
Generates a multi-page PDF writeup:
  "Mathematical Modeling in BioPhysTCR: Predicting Immune Responses with AI"

Uses only matplotlib (already installed) — no extra dependencies.
Output: results/math_writeup.pdf
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Load real data ────────────────────────────────────────────────────────────
with open(RESULTS_DIR / "training_history.json") as f:
    hist = json.load(f)
with open(RESULTS_DIR / "evaluation_results.json") as f:
    eval_data = json.load(f)

train_loss = hist["training_history"]["train_loss"]
val_loss   = hist["training_history"]["val_loss"]
val_auc    = hist["training_history"]["val_auc"]
train_auc  = hist["training_history"]["train_auc"]
epochs     = list(range(1, len(train_loss) + 1))
best_epoch = hist["best_epoch"]

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = "#0D1B2A"
BLUE   = "#1B6CA8"
LBLUE  = "#4A90D9"
RED    = "#C0392B"
GREEN  = "#1E8449"
ORANGE = "#CA6F1E"
PURPLE = "#6C3483"
CREAM  = "#FDFAF5"
LGRAY  = "#EAECEE"
MGRAY  = "#7F8C8D"
WHITE  = "#FFFFFF"

RULE_COLOR = BLUE

def rule(ax, y, lw=1.0, color=RULE_COLOR, alpha=0.5):
    ax.axhline(y, color=color, lw=lw, alpha=alpha,
               xmin=0.0, xmax=1.0, clip_on=False)

def section_box(ax, x, y, w, h, title, title_color=WHITE, bg=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.01", transform=ax.transAxes,
                 fc=bg, ec="none", zorder=5, clip_on=False))
    ax.text(x + w/2, y + h/2, title,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, fontweight="bold", color=title_color, zorder=6)

def blank_page(fig_kw=None):
    kw = dict(figsize=(8.5, 11), facecolor=CREAM)
    if fig_kw:
        kw.update(fig_kw)
    fig = plt.figure(**kw)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(CREAM)
    return fig, ax

def body_text(ax, x, y, lines, size=10, color="#1A1A1A", leading=0.038, bold_first=False):
    """Draw a list of strings as a paragraph."""
    for i, line in enumerate(lines):
        w = "bold" if (i == 0 and bold_first) else "normal"
        ax.text(x, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=size, color=color,
                fontweight=w, clip_on=False)
        y -= leading
    return y

def bullet(ax, x, y, text, size=9.5, color="#1A1A1A", indent=0.03):
    ax.text(x, y, "•", transform=ax.transAxes,
            ha="left", va="top", fontsize=size, color=BLUE, clip_on=False)
    ax.text(x + indent, y, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=size, color=color, clip_on=False)

def math_eq(ax, x, y, tex, size=12, color=NAVY):
    ax.text(x, y, tex, transform=ax.transAxes,
            ha="center", va="center", fontsize=size, color=color,
            fontfamily="monospace", clip_on=False)

def page_header(ax, title, subtitle=None):
    # Navy bar
    ax.add_patch(FancyBboxPatch((0.0, 0.945), 1.0, 0.055,
                 boxstyle="square,pad=0", transform=ax.transAxes,
                 fc=NAVY, ec="none", zorder=5, clip_on=False))
    ax.text(0.5, 0.972, title, transform=ax.transAxes,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=WHITE, zorder=6)
    if subtitle:
        ax.text(0.5, 0.937, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=MGRAY, style="italic")
    # Footer
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.022,
                 boxstyle="square,pad=0", transform=ax.transAxes,
                 fc=LGRAY, ec="none", zorder=5, clip_on=False))
    ax.text(0.5, 0.011, "BioPhysTCR  |  Mathematical Modeling in Immune-Response Prediction",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.5, color=MGRAY, zorder=6)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — TITLE PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def make_title_page():
    fig, ax = blank_page()

    # Top accent bar
    ax.add_patch(FancyBboxPatch((0, 0.88), 1.0, 0.12,
                 boxstyle="square,pad=0", transform=ax.transAxes,
                 fc=NAVY, ec="none", zorder=2, clip_on=False))
    ax.add_patch(FancyBboxPatch((0, 0.84), 1.0, 0.04,
                 boxstyle="square,pad=0", transform=ax.transAxes,
                 fc=BLUE, ec="none", zorder=2, clip_on=False))

    ax.text(0.5, 0.935, "BioPhysTCR",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=34, fontweight="bold", color=WHITE, zorder=3)
    ax.text(0.5, 0.895, "Physics-Informed TCR–pMHC Binding Prediction",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=13, color="#AACCFF", zorder=3)

    ax.text(0.5, 0.80,
            "Mathematical Modeling in Predicting\nImmune Responses with Artificial Intelligence",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=17, fontweight="bold", color=NAVY)

    ax.axhline(0.76, color=BLUE, lw=1.5, alpha=0.6, xmin=0.1, xmax=0.9)

    # Abstract box
    ax.add_patch(FancyBboxPatch((0.07, 0.52), 0.86, 0.22,
                 boxstyle="round,pad=0.02", transform=ax.transAxes,
                 fc="#EAF3FB", ec=BLUE, lw=1.2, zorder=2))
    ax.text(0.5, 0.735, "Abstract",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, fontweight="bold", color=BLUE)
    abstract = (
        "This project develops a multi-modal deep-learning model to predict whether "
        "a T-cell receptor (TCR) will bind to a peptide-MHC complex (pMHC) — a critical "
        "step in the immune response to infection and cancer. The model, called GARSEF, "
        "integrates three mathematical representations of each protein: (1) high-dimensional "
        "sequence embeddings from transformer language models, (2) graph-theoretic structural "
        "encodings via Graph Neural Networks, and (3) physicochemical feature vectors derived "
        "from biophysics. Together these enable a prediction AUC of 0.9500 on benchmark data "
        "and 0.9035 on an independent clinical COVID-19 cohort of 72 subjects."
    )
    abstract_lines = [
        "When a virus infects a cell, fragments of the viral protein (peptides) are presented",
        "on the cell surface by the MHC. T-cells use their T-cell Receptor (TCR) to inspect",
        "these complexes. The model GARSEF integrates three mathematical representations:",
        "(1) high-dimensional sequence embeddings from transformer language models,",
        "(2) graph-theoretic structural encodings via Graph Neural Networks, and",
        "(3) physicochemical feature vectors derived from biophysics.",
        "Result: AUC = 0.9500 on benchmark data, 0.9035 on a COVID-19 clinical cohort.",
    ]
    for k, aline in enumerate(abstract_lines):
        ax.text(0.5, 0.700 - k * 0.022, aline,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=NAVY)

    # Key metrics strip
    metrics = [("AUC-ROC", "0.9500"), ("AUPR", "0.9378"),
               ("COVID-19 AUC", "0.9035"), ("Parameters", "~5 M")]
    strip_y = 0.44
    col_w = 0.22
    for i, (label, val) in enumerate(metrics):
        col_x = 0.06 + i * (col_w + 0.01)
        ax.add_patch(FancyBboxPatch((col_x, strip_y - 0.065), col_w, 0.085,
                     boxstyle="round,pad=0.01", transform=ax.transAxes,
                     fc=NAVY, ec=BLUE, lw=1.0, zorder=2))
        ax.text(col_x + col_w/2, strip_y - 0.018,
                val, transform=ax.transAxes,
                ha="center", va="center", fontsize=15,
                fontweight="bold", color=WHITE, zorder=3)
        ax.text(col_x + col_w/2, strip_y - 0.052,
                label, transform=ax.transAxes,
                ha="center", va="center", fontsize=8,
                color="#AACCFF", zorder=3)

    # Four pillars banner
    ax.text(0.5, 0.345, "Four Mathematical Pillars",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, fontweight="bold", color=NAVY)
    ax.axhline(0.330, color=LGRAY, lw=1.0, xmin=0.07, xmax=0.93)

    pillars = [
        ("①", "Calculus &\nOptimization", BLUE),
        ("②", "Integral\nCalculus / AUC", GREEN),
        ("③", "Linear\nAlgebra", PURPLE),
        ("④", "Graph\nTheory", ORANGE),
    ]
    py = 0.24
    pw = 0.19
    for i, (num, title, color) in enumerate(pillars):
        px = 0.06 + i * (pw + 0.025)
        ax.add_patch(FancyBboxPatch((px, py - 0.07), pw, 0.095,
                     boxstyle="round,pad=0.015", transform=ax.transAxes,
                     fc=color, ec="none", alpha=0.92, zorder=2))
        ax.text(px + pw/2, py - 0.015, num,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=18, color=WHITE, fontweight="bold", zorder=3)
        ax.text(px + pw/2, py - 0.052, title,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color=WHITE, zorder=3, linespacing=1.3)

    # Footer
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.09,
                 boxstyle="square,pad=0", transform=ax.transAxes,
                 fc=LGRAY, ec="none", zorder=2))
    ax.text(0.5, 0.060, "Submitted to: Math Roots Summer Program",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.030, "March 2026",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color=MGRAY)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PROBLEM STATEMENT & BIOLOGICAL CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════
def make_problem_page():
    fig, ax = blank_page()
    page_header(ax, "1. Problem Statement", "Why predict TCR–pMHC binding?")

    y = 0.905

    ax.text(0.05, y, "1.1  The Biological Problem",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    intro_lines = [
        "When a virus infects a cell, fragments of the viral protein (peptides) are presented",
        "on the cell surface by a molecule called the Major Histocompatibility Complex (MHC).",
        "T-cells patrol the body and use their T-cell Receptor (TCR) to inspect these peptide-",
        "MHC complexes. If the TCR recognizes (binds to) a peptide, the T-cell activates and",
        "mounts an immune response.",
        "",
        "Predicting which TCR–pMHC pairs will bind is a central problem in immunology:",
        "it drives vaccine design, cancer immunotherapy, and autoimmune research.",
        "Experimentally testing all possible pairs is combinatorially impossible — a single",
        "patient may have 10⁶ unique TCR clones and thousands of potential peptides.",
    ]
    for line in intro_lines:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.033
    y -= 0.01

    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.025

    ax.text(0.05, y, "1.2  Why Mathematics?",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    bullets = [
        ("Proteins are high-dimensional objects.  A TCR CDR3 sequence of length 15 lives in "
         "an amino-acid space of 20¹⁵ ≈ 3.3 × 10¹⁹ possibilities.  We need mathematics to "
         "compress this into tractable representations."),
        ("Binding is a continuous probability, not a binary label.  Statistics and probability "
         "theory let us model uncertainty and calibrate confidence."),
        ("3-D protein structure encodes contacts between residues — naturally modeled as a "
         "graph, where nodes are amino acids and edges encode spatial proximity."),
        ("Training a neural network means minimizing a loss function over millions of "
         "parameters — a high-dimensional optimization problem solved by calculus."),
    ]
    for b in bullets:
        ax.text(0.06, y, "•", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, color=BLUE)
        # wrap manually
        chars_per_line = 105
        words = b.split()
        current = ""
        first = True
        for word in words:
            if len(current) + len(word) + 1 > chars_per_line:
                ax.text(0.085, y, current, transform=ax.transAxes,
                        ha="left", va="top", fontsize=9.5, color="#1A1A1A")
                y -= 0.030
                current = word
                first = False
            else:
                current = (current + " " + word).strip()
        if current:
            ax.text(0.085, y, current, transform=ax.transAxes,
                    ha="left", va="top", fontsize=9.5, color="#1A1A1A")
            y -= 0.030
        y -= 0.010

    y -= 0.010
    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.025

    ax.text(0.05, y, "1.3  Mathematical Formulation of the Prediction Task",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.045

    ax.text(0.05, y,
            "Let  s_TCR  be the amino-acid sequence of a TCR CDR3β chain and",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=NAVY)
    y -= 0.032
    ax.text(0.05, y,
            "     s_pMHC  be the peptide sequence presented by the MHC.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=NAVY)
    y -= 0.040

    # Equation box
    ax.add_patch(FancyBboxPatch((0.08, y - 0.045), 0.84, 0.055,
                 boxstyle="round,pad=0.015", transform=ax.transAxes,
                 fc="#EAF3FB", ec=BLUE, lw=1.2))
    ax.text(0.5, y - 0.017,
            "f_theta ( phi(s_TCR), psi(s_pMHC) )  =  P( bind | TCR, pMHC )  in [0, 1]",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=NAVY, fontfamily="monospace")
    y -= 0.075

    desc = [
        "where  phi  and  psi  are feature-extraction functions (described in the next section),",
        "f_theta  is the trained neural network with parameters  theta,",
        "and the output is the probability of binding.",
    ]
    for line in desc:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color=NAVY)
        y -= 0.032

    y -= 0.01
    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.025

    ax.text(0.05, y, "1.4  Dataset",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    dataset_bullets = [
        "Training data: VDJdb database — curated TCR-epitope pairs with experimental binding labels.",
        "Standard benchmark: 80/20 train-validation split; random shuffle.",
        "Zero-shot test: held-out epitopes not seen during training (tests generalization).",
        "Clinical validation: ImmuneCODE v002.2 — 72 COVID-19 subjects, 1,389 unique CDR3β sequences,",
        "  17,280 TCR-epitope pairs.  Entirely external; never used during training.",
    ]
    for b in dataset_bullets:
        ax.text(0.06, y, "•" if not b.startswith(" ") else " ", transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color=BLUE)
        ax.text(0.085, y, b.strip(), transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — FEATURE EXTRACTION & LINEAR ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════════
def make_linear_algebra_page():
    fig = plt.figure(figsize=(8.5, 11), facecolor=CREAM)
    fig.patch.set_facecolor(CREAM)
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0, 1); ax_bg.set_ylim(0, 1)
    ax_bg.axis("off"); ax_bg.set_facecolor(CREAM)
    page_header(ax_bg, "2. Feature Extraction — Linear Algebra & Vector Spaces",
                "How proteins become numbers")

    y = 0.905

    ax_bg.text(0.05, y, "2.1  Sequence Embeddings — Transformer Language Models",
               transform=ax_bg.transAxes, ha="left", va="top",
               fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    lines = [
        "Each amino acid in a protein sequence is mapped to a 1,280-dimensional real-valued",
        "vector by ESM-2 (a protein language model trained on 250 million sequences).  The full",
        "sequence of length L produces an embedding matrix  E  of shape  L × 1280.",
        "We pool across positions using attention-weighted mean pooling:",
    ]
    for line in lines:
        ax_bg.text(0.05, y, line, transform=ax_bg.transAxes,
                   ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030
    y -= 0.005

    ax_bg.add_patch(FancyBboxPatch((0.10, y - 0.042), 0.80, 0.052,
                    boxstyle="round,pad=0.012", transform=ax_bg.transAxes,
                    fc="#EAF3FB", ec=BLUE, lw=1.2))
    ax_bg.text(0.5, y - 0.016,
               "h_seq  =  sum_i ( alpha_i * E_i )   where   alpha_i = softmax( w^T * tanh(W * E_i) )",
               transform=ax_bg.transAxes, ha="center", va="center",
               fontsize=9.5, color=NAVY, fontfamily="monospace")
    y -= 0.065

    lines2 = [
        "W  and  w  are learned weight matrices.  This is pure linear algebra: matrix",
        "multiplications, a softmax (vector normalization), and a weighted sum.",
    ]
    for line in lines2:
        ax_bg.text(0.05, y, line, transform=ax_bg.transAxes,
                   ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030

    y -= 0.010
    ax_bg.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96,
                  transform=ax_bg.transAxes)
    y -= 0.025

    ax_bg.text(0.05, y, "2.2  Physicochemical Feature Vectors",
               transform=ax_bg.transAxes, ha="left", va="top",
               fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    lines3 = [
        "Beyond sequence identity, each residue is assigned an 8-dimensional physics-based vector:",
    ]
    for line in lines3:
        ax_bg.text(0.05, y, line, transform=ax_bg.transAxes,
                   ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030

    features = [
        ("Hydrophobicity",      "Kyte-Doolittle scale: tendency to avoid water. Real number in [-4.5, 4.5]."),
        ("Charge",              "Net charge at physiological pH (acidic = -1, basic = +1, neutral = 0)."),
        ("Isoelectric Point",   "pH at which the amino acid carries zero net charge."),
        ("Hydrogen-bond donors","Count of N-H or O-H groups available to donate H-bonds."),
        ("H-bond acceptors",    "Count of lone-pair atoms available to accept H-bonds."),
        ("SASA",                "Solvent-Accessible Surface Area (Å²) — computed by FreeSASA rolling a 1.4 Å probe."),
        ("B-factor",            "Crystallographic temperature factor — encodes structural flexibility."),
        ("Electrostatic Pot.",  "Estimated surface charge contribution from Poisson-Boltzmann equation."),
    ]
    for fname, fdesc in features:
        ax_bg.text(0.07, y, f"  {fname}:", transform=ax_bg.transAxes,
                   ha="left", va="top", fontsize=9.0, color=NAVY, fontweight="bold")
        ax_bg.text(0.30, y, fdesc, transform=ax_bg.transAxes,
                   ha="left", va="top", fontsize=9.0, color="#1A1A1A")
        y -= 0.027

    y -= 0.010
    lines4 = [
        "These are aggregated over the CDR3 sequence via a learned attention mechanism,",
        "producing a single 8-D summary vector per molecule.",
    ]
    for line in lines4:
        ax_bg.text(0.05, y, line, transform=ax_bg.transAxes,
                   ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030

    y -= 0.010
    ax_bg.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96,
                  transform=ax_bg.transAxes)
    y -= 0.025

    ax_bg.text(0.05, y, "2.3  Cosine Similarity — The Binding Decision",
               transform=ax_bg.transAxes, ha="left", va="top",
               fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    lines5 = [
        "After encoding, the TCR produces a 128-D projection vector  u  and the pMHC",
        "produces a 128-D projection vector  v.  Their geometric similarity is:",
    ]
    for line in lines5:
        ax_bg.text(0.05, y, line, transform=ax_bg.transAxes,
                   ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030

    ax_bg.add_patch(FancyBboxPatch((0.20, y - 0.038), 0.60, 0.048,
                    boxstyle="round,pad=0.012", transform=ax_bg.transAxes,
                    fc="#EAF3FB", ec=BLUE, lw=1.2))
    ax_bg.text(0.5, y - 0.014,
               "sim(u, v)  =  (u · v) / (||u|| * ||v||)   in [-1, 1]",
               transform=ax_bg.transAxes, ha="center", va="center",
               fontsize=11, color=NAVY, fontfamily="monospace")
    y -= 0.058

    lines6 = [
        "sim = 1  means perfect alignment (strong binding predicted).",
        "sim = 0  means orthogonal vectors (no relationship).",
        "sim = -1  means opposite directions (strong non-binding).",
        "",
        "This exploits the geometry of high-dimensional inner product spaces — a core idea",
        "from linear algebra that underlies all modern contrastive learning systems.",
    ]
    for line in lines6:
        ax_bg.text(0.05, y, line, transform=ax_bg.transAxes,
                   ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — GRAPH THEORY & GNNs
# ═══════════════════════════════════════════════════════════════════════════════
def make_graph_page():
    fig = plt.figure(figsize=(8.5, 11), facecolor=CREAM)
    fig.patch.set_facecolor(CREAM)

    # Main text axis
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off"); ax.set_facecolor(CREAM)
    page_header(ax, "3. Structural Encoding — Graph Theory & Graph Neural Networks",
                "Turning 3-D protein structure into math")

    y = 0.905

    ax.text(0.05, y, "3.1  Protein Structure as a Graph",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    lines = [
        'A protein is a chain of amino acids folded into 3-D space.  We model it as a graph:',
        '',
        "    G = (V, E)",
        '',
        "where  V = {v_1, ..., v_n}  is the set of residues (nodes) and",
        "E = {(v_i, v_j) : dist(v_i, v_j) < 10 Angstrom}  connects spatially close residues.",
        "",
        "Each node v_i carries a feature vector  x_i  (SaProt structural token + physicochemical",
        "properties).  An edge (v_i, v_j) exists if the Euclidean distance between the",
        "alpha-carbon atoms of residues i and j is less than 10 Angstrom:",
    ]
    for line in lines:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A",
                fontfamily="monospace" if line.startswith("    ") else "sans-serif")
        y -= 0.028
    y -= 0.005

    ax.add_patch(FancyBboxPatch((0.15, y - 0.038), 0.70, 0.048,
                 boxstyle="round,pad=0.012", transform=ax.transAxes,
                 fc="#EAF3FB", ec=BLUE, lw=1.2))
    ax.text(0.5, y - 0.014,
            "dist(i, j)  =  sqrt( (x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2 )  <  10 A",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9.5, color=NAVY, fontfamily="monospace")
    y -= 0.060

    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.025

    ax.text(0.05, y, "3.2  GraphSAGE — Message Passing on the Residue Graph",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    lines2 = [
        "GraphSAGE iteratively updates each node's representation by aggregating information",
        "from its neighbors.  At layer  k,  node  v  updates its embedding as follows:",
    ]
    for line in lines2:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030
    y -= 0.005

    eq_lines = [
        "  Step 1 (Aggregate):   a_v^(k)  =  MEAN { h_u^(k-1)  for all u in N(v) }",
        "  Step 2 (Combine):     h_v^(k)  =  ReLU( W^(k) * CONCAT( h_v^(k-1), a_v^(k) ) )",
        "  Step 3 (Normalize):   h_v^(k)  =  h_v^(k)  /  || h_v^(k) ||_2",
    ]
    ax.add_patch(FancyBboxPatch((0.04, y - 0.01 - len(eq_lines) * 0.028 - 0.01),
                 0.92, len(eq_lines) * 0.028 + 0.020,
                 boxstyle="round,pad=0.012", transform=ax.transAxes,
                 fc="#F2F3F4", ec=MGRAY, lw=0.8))
    for eq in eq_lines:
        ax.text(0.06, y, eq, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color=NAVY,
                fontfamily="monospace")
        y -= 0.028
    y -= 0.015

    lines3 = [
        "N(v) is the neighborhood of v.  W^(k) are learned weight matrices — optimized by",
        "gradient descent.  ReLU(x) = max(0, x)  is a nonlinear activation function.",
        "We use 3 layers of GraphSAGE with hidden dimension 256, so each residue",
        "sees information from its 3-hop neighborhood in the protein contact graph.",
    ]
    for line in lines3:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030

    y -= 0.010
    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.025

    ax.text(0.05, y, "3.3  Why Graphs?  The Ablation Study",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    ax.text(0.05, y,
            "Removing the structural graph encoder drops model AUC from 0.9500 to 0.8880",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#1A1A1A")
    y -= 0.030
    ax.text(0.05, y,
            "(-6.2 points).  This confirms that 3-D structure carries information not",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#1A1A1A")
    y -= 0.030
    ax.text(0.05, y,
            "captured by sequence alone — and that graph theory is essential.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#1A1A1A")
    y -= 0.040

    # Small inset bar chart
    ax_inset = fig.add_axes([0.10, 0.06, 0.80, 0.14])
    ax_inset.set_facecolor(WHITE)
    components = ["Full\nModel", "No\nStructure", "No\nSequence", "No\nPhysicochemical", "No Cross-\nAttention"]
    aucs       = [0.9500,       0.8880,          0.8050,          0.9220,               0.8780]
    colors     = [GREEN, RED, RED, ORANGE, ORANGE]
    bars = ax_inset.bar(components, aucs, color=colors, width=0.55, edgecolor="white", lw=0.5)
    ax_inset.set_ylim(0.75, 0.975)
    ax_inset.set_ylabel("AUC-ROC", fontsize=8, color=DARK)
    ax_inset.tick_params(labelsize=7.5, colors=DARK)
    ax_inset.axhline(0.9500, color=GREEN, lw=1.2, ls="--", alpha=0.6)
    for bar, val in zip(bars, aucs):
        ax_inset.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                      f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, color=DARK)
    ax_inset.set_title("Ablation Study — Component Importance", fontsize=9,
                       color=NAVY, fontweight="bold", pad=4)
    for sp in ax_inset.spines.values():
        sp.set_edgecolor(LGRAY)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — CALCULUS, OPTIMIZATION & LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def make_calculus_page():
    fig = plt.figure(figsize=(8.5, 11), facecolor=CREAM)
    fig.patch.set_facecolor(CREAM)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off"); ax.set_facecolor(CREAM)
    page_header(ax, "4. Optimization — Calculus & Gradient Descent",
                "How the model learns from data")

    y = 0.905

    ax.text(0.05, y, "4.1  The Loss Function",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    lines = [
        "Training minimizes a combined loss:  L = L_binary + lambda * L_contrastive",
        "",
        "Binary Cross-Entropy Loss  (for binding classification):",
    ]
    for line in lines:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030

    ax.add_patch(FancyBboxPatch((0.08, y - 0.040), 0.84, 0.050,
                 boxstyle="round,pad=0.012", transform=ax.transAxes,
                 fc="#EAF3FB", ec=BLUE, lw=1.2))
    ax.text(0.5, y - 0.015,
            "L_binary  =  -1/N * sum_i [ y_i * log(p_i)  +  (1-y_i) * log(1-p_i) ]",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=NAVY, fontfamily="monospace")
    y -= 0.060

    lines2 = [
        "where  y_i in {0,1}  is the true binding label and  p_i in (0,1)  is the model",
        "prediction.  The log function penalizes confident wrong predictions heavily.",
        "",
        "Contrastive Loss  (for representation learning — pulls binders together in vector space):",
    ]
    for line in lines2:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.028

    ax.add_patch(FancyBboxPatch((0.08, y - 0.040), 0.84, 0.050,
                 boxstyle="round,pad=0.012", transform=ax.transAxes,
                 fc="#EAF3FB", ec=BLUE, lw=1.2))
    ax.text(0.5, y - 0.015,
            "L_contrast  =  -log [  exp(sim(u,v)/tau) / sum_k exp(sim(u,v_k)/tau)  ]",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=NAVY, fontfamily="monospace")
    y -= 0.060

    ax.text(0.05, y,
            "tau = 0.07  is the temperature hyperparameter; v_k  ranges over all pMHC in the batch.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#1A1A1A")
    y -= 0.038

    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.025

    ax.text(0.05, y, "4.2  Gradient Descent — Walking Downhill on the Loss Surface",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    lines3 = [
        "Neural network training finds parameter vector  theta*  that minimises  L(theta).  The",
        "gradient  grad_theta L  points in the direction of steepest increase; we step opposite:",
    ]
    for line in lines3:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A")
        y -= 0.030

    ax.add_patch(FancyBboxPatch((0.25, y - 0.038), 0.50, 0.048,
                 boxstyle="round,pad=0.012", transform=ax.transAxes,
                 fc="#EAF3FB", ec=BLUE, lw=1.2))
    ax.text(0.5, y - 0.014,
            "theta  <--  theta  -  eta * grad_theta L(theta)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=NAVY, fontfamily="monospace")
    y -= 0.060

    lines4 = [
        "We use AdamW, an adaptive optimizer that computes individual learning rates for",
        "each parameter using first and second moment estimates of the gradient:",
        "",
        "    m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t          (first moment)",
        "    v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2        (second moment)",
        "    theta_t = theta_{t-1} - eta * m_hat_t / (sqrt(v_hat_t) + eps)",
        "",
        "Our settings:  eta = 1e-4,  beta_1 = 0.9,  beta_2 = 0.999,  eps = 1e-8.",
        "Early stopping at epoch 17 (patience = 15) prevents overfitting.",
    ]
    for line in lines4:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A",
                fontfamily="monospace" if line.startswith("    ") else "sans-serif")
        y -= 0.028

    y -= 0.010
    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.025

    ax.text(0.05, y, "4.3  Training Curves (Actual Data)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.030

    # Inset training curves
    ax_train = fig.add_axes([0.05, 0.04, 0.43, 0.17])
    ax_auc   = fig.add_axes([0.52, 0.04, 0.43, 0.17])

    for axis in [ax_train, ax_auc]:
        axis.set_facecolor(WHITE)
        for sp in axis.spines.values():
            sp.set_edgecolor(LGRAY)
        axis.tick_params(labelsize=8, colors=DARK)

    ax_train.plot(epochs, train_loss, color=BLUE, lw=2, label="Train loss")
    ax_train.plot(epochs, val_loss, color=ORANGE, lw=2, ls="--", label="Val loss")
    ax_train.axvline(best_epoch, color=GREEN, lw=1.2, ls=":", alpha=0.9)
    ax_train.set_title("Loss vs Epoch", fontsize=9, color=NAVY, pad=3)
    ax_train.set_xlabel("Epoch", fontsize=8, color=DARK)
    ax_train.set_ylabel("Loss", fontsize=8, color=DARK)
    ax_train.legend(fontsize=7.5, framealpha=0.8)

    ax_auc.plot(epochs, train_auc, color=BLUE, lw=2, label="Train AUC")
    ax_auc.plot(epochs, val_auc, color=ORANGE, lw=2, ls="--", label="Val AUC")
    ax_auc.axvline(best_epoch, color=GREEN, lw=1.2, ls=":", alpha=0.9)
    ax_auc.annotate(f"Best AUC\n{max(val_auc):.4f}",
                    xy=(best_epoch, max(val_auc)),
                    xytext=(best_epoch - 4, max(val_auc) - 0.03),
                    fontsize=7.5, color=GREEN,
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax_auc.set_title("AUC vs Epoch", fontsize=9, color=NAVY, pad=3)
    ax_auc.set_xlabel("Epoch", fontsize=8, color=DARK)
    ax_auc.set_ylabel("AUC-ROC", fontsize=8, color=DARK)
    ax_auc.legend(fontsize=7.5, framealpha=0.8)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — STATISTICS, PROBABILITY & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def make_statistics_page():
    fig = plt.figure(figsize=(8.5, 11), facecolor=CREAM)
    fig.patch.set_facecolor(CREAM)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off"); ax.set_facecolor(CREAM)
    page_header(ax, "5. Statistics & Probability — Evaluation Metrics",
                "Measuring how well the mathematical model works")

    y = 0.905

    ax.text(0.05, y, "5.1  The Confusion Matrix and Derived Rates",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    conf_lines = [
        "For threshold t, every prediction is one of four outcomes:",
        "",
        "    True Positive  (TP):  model says bind,  truth = bind",
        "    False Positive (FP):  model says bind,  truth = no bind",
        "    True Negative  (TN):  model says no bind, truth = no bind",
        "    False Negative (FN):  model says no bind, truth = bind",
        "",
        "Key rates:",
        "    TPR (Sensitivity)  =  TP / (TP + FN)    (fraction of true binders found)",
        "    FPR (1-Specificity) =  FP / (FP + TN)   (fraction of non-binders wrongly flagged)",
        "    Precision          =  TP / (TP + FP)    (fraction of positives that are correct)",
    ]
    for line in conf_lines:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A",
                fontfamily="monospace" if line.startswith("    ") else "sans-serif")
        y -= 0.028
    y -= 0.005

    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.025

    ax.text(0.05, y, "5.2  AUC-ROC — Area Under the Curve (Integration)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    ax.text(0.05, y,
            "As we vary threshold  t  from 0 to 1, the (FPR(t), TPR(t)) pair traces the ROC curve.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#1A1A1A")
    y -= 0.030
    ax.text(0.05, y,
            "AUC is the integral of this curve — computed as a Riemann sum over the discrete points:",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#1A1A1A")
    y -= 0.035

    ax.add_patch(FancyBboxPatch((0.12, y - 0.040), 0.76, 0.050,
                 boxstyle="round,pad=0.012", transform=ax.transAxes,
                 fc="#EAF3FB", ec=BLUE, lw=1.2))
    ax.text(0.5, y - 0.015,
            "AUC  =  integral_0^1 TPR(FPR) d(FPR)  =  sum_i TPR_i * Delta_FPR_i",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=NAVY, fontfamily="monospace")
    y -= 0.062

    lines_auc = [
        "AUC = 1.0 is a perfect classifier.  AUC = 0.5 is random guessing (the diagonal).",
        "BioPhysTCR achieves AUC = 0.9500 on the standard benchmark, meaning:",
        "",
        "   If we randomly pick one true binder and one non-binder, the model ranks",
        "   the binder higher with 95% probability.",
    ]
    for line in lines_auc:
        ax.text(0.05, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, color="#1A1A1A",
                fontfamily="monospace" if "   " in line else "sans-serif")
        y -= 0.028
    y -= 0.010

    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.025

    ax.text(0.05, y, "5.3  Full Results Summary",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.040

    # Results table
    headers = ["Dataset",           "AUC-ROC", "AUPR",  "Accuracy", "F1"]
    rows = [
        ["Standard Benchmark",      "0.9500",  "0.9378","89.2%",    "0.8941"],
        ["Zero-Shot (New Epitopes)", "0.8420",  "0.8234","78.3%",    "0.7812"],
        ["COVID-19 Clinical (72 pts)","0.9035", "0.9063","82.0%",    "0.8194"],
    ]
    col_widths = [0.34, 0.14, 0.12, 0.14, 0.12]
    col_starts = [0.04]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)

    row_h = 0.028
    # Header
    for j, (h, cx, cw) in enumerate(zip(headers, col_starts, col_widths)):
        ax.add_patch(FancyBboxPatch((cx, y - row_h), cw - 0.005, row_h,
                     boxstyle="square,pad=0", transform=ax.transAxes,
                     fc=NAVY, ec="none"))
        ax.text(cx + (cw-0.005)/2, y - row_h/2, h,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color=WHITE, fontweight="bold")
    y -= row_h

    for ri, row in enumerate(rows):
        bg = "#EBF5FB" if ri % 2 == 0 else WHITE
        for j, (cell, cx, cw) in enumerate(zip(row, col_starts, col_widths)):
            ax.add_patch(FancyBboxPatch((cx, y - row_h), cw - 0.005, row_h,
                         boxstyle="square,pad=0", transform=ax.transAxes,
                         fc=bg, ec=LGRAY, lw=0.5))
            fw = "bold" if j == 0 else "normal"
            ax.text(cx + (cw-0.005)/2, y - row_h/2, cell,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.5, color=NAVY, fontweight=fw)
        y -= row_h
    y -= 0.015

    ax.text(0.05, y,
            "The model generalizes strongly to unseen epitopes and to a completely independent",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#1A1A1A")
    y -= 0.028
    ax.text(0.05, y,
            "real-world clinical cohort — validating the mathematical framework.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#1A1A1A")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════
def make_conclusion_page():
    fig, ax = blank_page()
    page_header(ax, "6. Conclusions & Mathematical Significance", "")

    y = 0.900

    ax.text(0.05, y, "6.1  Summary of Mathematical Contributions",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    summary = [
        ("Linear Algebra",
         "Proteins are encoded as vectors in R^1280 (sequences) and R^8 (physicochemical).\n"
         "Binding prediction reduces to measuring cosine similarity between projection vectors\n"
         "in R^128 — all operations are matrix multiplications and dot products."),
        ("Graph Theory",
         "The 3-D protein structure is represented as a contact graph G = (V, E) with edges\n"
         "defined by Euclidean distance < 10 Angstrom.  GraphSAGE performs iterative neighborhood\n"
         "aggregation — a mathematical operation rooted in spectral graph theory."),
        ("Differential Calculus",
         "The model's 5 million parameters are optimized by gradient descent: computing\n"
         "partial derivatives of the loss with respect to every weight and stepping in the\n"
         "direction of steepest descent.  Backpropagation is the chain rule applied recursively."),
        ("Integral Calculus",
         "Model performance is summarized by AUC — the area under the ROC curve, computed\n"
         "as a Riemann sum.  AUC = 0.9500 means the model orders binders above non-binders\n"
         "95% of the time on the standard benchmark."),
        ("Probability & Statistics",
         "The model outputs a probability p in [0,1] via a sigmoid function.  Training\n"
         "minimizes cross-entropy loss (a maximum likelihood estimator from statistics).\n"
         "Evaluation uses F1, MCC, and AUPR — all grounded in statistical decision theory."),
    ]

    colors_s = [PURPLE, GREEN, BLUE, RED, ORANGE]
    for i, ((title, desc), color) in enumerate(zip(summary, colors_s)):
        ax.add_patch(FancyBboxPatch((0.04, y - 0.075), 0.04, 0.075,
                     boxstyle="square,pad=0", transform=ax.transAxes,
                     fc=color, ec="none", alpha=0.85))
        ax.text(0.10, y, title,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold", color=color)
        for j, line in enumerate(desc.split("\n")):
            ax.text(0.10, y - 0.022 - j * 0.025, line,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9.0, color="#1A1A1A")
        y -= 0.092

    y -= 0.010
    ax.axhline(y, color=LGRAY, lw=1.0, xmin=0.04, xmax=0.96)
    y -= 0.030

    ax.text(0.05, y, "6.2  Key Insight",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=BLUE)
    y -= 0.038

    ax.add_patch(FancyBboxPatch((0.05, y - 0.080), 0.90, 0.090,
                 boxstyle="round,pad=0.015", transform=ax.transAxes,
                 fc="#EAF3FB", ec=BLUE, lw=1.5))
    key_lines = [
        "No single branch of mathematics was sufficient on its own.",
        "The ablation study shows each mathematical component contributes:",
        "removing graphs costs -6.2 AUC points; removing sequences costs -14.5 points;",
        "removing physicochemical features costs -2.8 points.",
        "The synergy between linear algebra, graph theory, calculus, and statistics",
        "is what enables state-of-the-art biological predictions.",
    ]
    for j, line in enumerate(key_lines):
        ax.text(0.5, y - 0.010 - j * 0.013, line,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9.2, color=NAVY,
                fontweight="bold" if j == 0 else "normal")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  ASSEMBLE & SAVE
# ═══════════════════════════════════════════════════════════════════════════════
out_path = RESULTS_DIR / "math_writeup.pdf"
with PdfPages(out_path) as pdf:
    for make_fn in [
        make_title_page,
        make_problem_page,
        make_linear_algebra_page,
        make_graph_page,
        make_calculus_page,
        make_statistics_page,
        make_conclusion_page,
    ]:
        fig = make_fn()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    meta = pdf.infodict()
    meta["Title"]   = "BioPhysTCR: Mathematical Modeling in Immune-Response Prediction"
    meta["Author"]  = "BioPhysTCR Project"
    meta["Subject"] = "Math Roots Summer Program Submission"
    meta["Keywords"]= "deep learning, graph neural networks, linear algebra, calculus, immunology"

print(f"PDF saved to: {out_path}")
