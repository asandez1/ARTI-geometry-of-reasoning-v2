"""
Generate Figure 8: Manifold Geometry of Reasoning Types (E19).

2x3 panel layout:
  (a) Scatter: R_eff vs ARTI accuracy — 10 colored points with Spearman rho
  (b) Scatter: D_eff vs ARTI accuracy
  (c) Scatter: Compactness ratio vs accuracy — expected strongest predictor
  (d) Heatmap: 10x10 center alignment matrix
  (e) Heatmap: 10x10 axes alignment matrix
  (f) Bar: Explained variance (R^2) for individual predictors + combined model

Outputs:
  docs/figures/figure8_manifold_geometry.png (300 DPI)
  docs/figures/figure8_manifold_geometry.pdf (300 DPI)

Usage:
    cd paper13_The_geometry_of_Machine_Reasoning
    python docs/figures/generate_e19_manifold_geometry.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
PAPER_DIR = BASE.parent.parent
RESULTS = PAPER_DIR / "experiments" / "experiment5_operators" / "results"
E19_RESULTS = RESULTS / "e19_manifold_geometry" / "e19_results.json"
OUTDIR = BASE

# ---------------------------------------------------------------------------
# Type metadata (same as generate_per_type_shapes.py)
# ---------------------------------------------------------------------------
TYPE_NAMES = [
    "PhysCause", "BehvCause", "SysCause", "Deduc", "Induc",
    "Analog", "Conserv", "Counter", "Abduc", "Decomp",
]

TYPE_FULL_NAMES = [
    "Physical\nCause", "Behavioral\nCause", "Systemic\nCause",
    "Deduction", "Induction", "Analogy", "Conservation",
    "Counter-\nfactual", "Abduction", "Decompo-\nsition",
]

TYPE_SHORT_LABELS = [
    "Phys", "Behv", "Sys", "Ded", "Ind",
    "Ana", "Con", "Ctr", "Abd", "Dec",
]

TYPE_COLORS = {
    0: "#d62728",  # Physical Cause — red
    1: "#8c564b",  # Behavioral Cause — brown
    2: "#e377c2",  # Systemic Cause — pink
    3: "#1f77b4",  # Deduction — blue
    4: "#2ca02c",  # Induction — green
    5: "#9467bd",  # Analogy — purple
    6: "#ff7f0e",  # Conservation — orange
    7: "#17becf",  # Counterfactual — cyan
    8: "#bcbd22",  # Abduction — yellow-green
    9: "#7f7f7f",  # Decomposition — gray
}

# Order by accuracy tier for heatmaps (highest to lowest)
ACC_ORDER = [7, 5, 2, 8, 4, 6, 9, 1, 3, 0]


def load_results():
    """Load E19 results JSON."""
    with open(E19_RESULTS) as f:
        return json.load(f)


def scatter_with_correlation(ax, x_vals, y_vals, colors, labels, xlabel, ylabel,
                             corr_label, rho, p_val, title, sc_pass):
    """Create a scatter plot with annotation of Spearman correlation."""
    for i in range(len(x_vals)):
        ax.scatter(x_vals[i], y_vals[i], c=colors[i], s=80, zorder=3,
                   edgecolors='white', linewidths=0.6)
        ax.annotate(labels[i], (x_vals[i], y_vals[i]),
                    fontsize=6.5, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points',
                    color=colors[i], fontweight='bold')

    # Trend line
    z = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    ax.plot(x_line, np.polyval(z, x_line), '--', color='#888888',
            linewidth=1.0, alpha=0.6, zorder=1)

    # Correlation annotation
    pass_color = '#059669' if sc_pass else '#DC2626'
    pass_marker = 'PASS' if sc_pass else 'FAIL'
    ax.text(0.03, 0.97,
            f'{corr_label}\n'
            f'$\\rho$ = {rho:.3f} (p = {p_val:.3f})\n'
            f'[{pass_marker}]',
            transform=ax.transAxes, fontsize=7, va='top', ha='left',
            color=pass_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=pass_color, alpha=0.9, linewidth=0.6))

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def heatmap_panel(ax, matrix, order, labels, title, cmap='RdBu_r', vmin=-1, vmax=1):
    """Create an alignment heatmap with reordered types."""
    n = len(order)
    reordered = np.zeros((n, n))
    for ii in range(n):
        for jj in range(n):
            reordered[ii, jj] = matrix[order[ii]][order[jj]]

    im = ax.imshow(reordered, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    ordered_labels = [labels[i] for i in order]
    ax.set_xticks(range(n))
    ax.set_xticklabels(ordered_labels, fontsize=6.5, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(ordered_labels, fontsize=6.5)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    # Add value annotations for off-diagonal
    for ii in range(n):
        for jj in range(n):
            if ii != jj:
                val = reordered[ii, jj]
                color = 'white' if abs(val) > 0.7 else 'black'
                ax.text(jj, ii, f'{val:.2f}', ha='center', va='center',
                        fontsize=4.5, color=color)

    return im


def bar_panel(ax, individual_r2, combined_r2, sc5_pass):
    """Create explained variance bar chart."""
    # Select key predictors
    bar_names = ['R_eff', 'D_eff', 'compact-\nness', 'center\nnorm', 'inter\nsep', 'mean\naxes', 'Combined\n(3-var)']
    bar_keys = ['R_eff', 'D_eff', 'compactness', 'center_norm', 'inter_sep', 'mean_axes_align']
    bar_vals = [individual_r2.get(k, 0.0) for k in bar_keys] + [combined_r2]

    colors = ['#4C78A8'] * 6 + ['#E45756']
    bars = ax.bar(range(len(bar_vals)), bar_vals, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.7)

    # SC5 target line
    ax.axhline(y=0.60, color='#DC2626', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.text(len(bar_vals) - 0.5, 0.62, 'SC5 target (R²=0.60)',
            fontsize=6.5, color='#DC2626', ha='right', va='bottom')

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, bar_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.set_xticks(range(len(bar_vals)))
    ax.set_xticklabels(bar_names, fontsize=7)
    ax.set_ylabel('$R^2$', fontsize=9)
    ax.set_title('(f) Explained Variance', fontsize=10, fontweight='bold', pad=8)
    ax.set_ylim(0, max(bar_vals) * 1.25)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, axis='y', alpha=0.2, linewidth=0.5)

    # Pass/fail badge
    pass_color = '#059669' if sc5_pass else '#DC2626'
    pass_marker = 'PASS' if sc5_pass else 'FAIL'
    ax.text(0.97, 0.97, f'[{pass_marker}]', transform=ax.transAxes,
            fontsize=8, fontweight='bold', color=pass_color,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=pass_color, alpha=0.9, linewidth=0.6))


def generate_figure():
    """Generate Figure 8: E19 Manifold Geometry."""
    results = load_results()

    # Extract data
    per_type = results['per_type_measures']
    correlations = results['correlations']
    sc = results['success_criteria']
    center_align_matrix = results['center_align_matrix']
    axes_align_matrix = results['axes_align_matrix']

    # Per-type arrays
    accs = [per_type[TYPE_NAMES[k]]['ensemble_accuracy'] * 100 for k in range(10)]
    R_effs = [per_type[TYPE_NAMES[k]]['R_eff'] for k in range(10)]
    D_effs = [per_type[TYPE_NAMES[k]]['D_eff'] for k in range(10)]
    compactness = [per_type[TYPE_NAMES[k]]['compactness_ratio'] for k in range(10)]
    colors = [TYPE_COLORS[k] for k in range(10)]
    labels = TYPE_SHORT_LABELS

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    # --- Panel (a): R_eff vs Accuracy ---
    sc1 = sc['SC1_reff_predicts_accuracy']
    scatter_with_correlation(
        axes[0, 0], R_effs, accs, colors, labels,
        xlabel='Effective Radius ($R_{eff}$)',
        ylabel='ARTI Ensemble Accuracy (%)',
        corr_label='SC1',
        rho=sc1['rho'], p_val=sc1['p'],
        title='(a) $R_{eff}$ vs Accuracy',
        sc_pass=sc1['pass'],
    )

    # --- Panel (b): D_eff vs Accuracy ---
    sc2 = sc['SC2_deff_predicts_accuracy']
    scatter_with_correlation(
        axes[0, 1], D_effs, accs, colors, labels,
        xlabel='Effective Dimension ($D_{eff}$)',
        ylabel='ARTI Ensemble Accuracy (%)',
        corr_label='SC2',
        rho=sc2['rho'], p_val=sc2['p'],
        title='(b) $D_{eff}$ vs Accuracy',
        sc_pass=sc2['pass'],
    )

    # --- Panel (c): Compactness vs Accuracy ---
    sc3 = sc['SC3_compactness_predicts_accuracy']
    scatter_with_correlation(
        axes[0, 2], compactness, accs, colors, labels,
        xlabel='Compactness Ratio ($R_{eff}$ / separation)',
        ylabel='ARTI Ensemble Accuracy (%)',
        corr_label='SC3',
        rho=sc3['rho'], p_val=sc3['p'],
        title='(c) Compactness vs Accuracy',
        sc_pass=sc3['pass'],
    )

    # --- Panel (d): Center Alignment Heatmap ---
    heatmap_panel(
        axes[1, 0], center_align_matrix, ACC_ORDER, TYPE_SHORT_LABELS,
        title='(d) Center Alignment cos($\\mu_i$, $\\mu_j$)',
        cmap='RdBu_r', vmin=-1, vmax=1,
    )

    # --- Panel (e): Axes Alignment Heatmap ---
    im = heatmap_panel(
        axes[1, 1], axes_align_matrix, ACC_ORDER, TYPE_SHORT_LABELS,
        title='(e) Axes Alignment mean|cos(PC$_i$, PC$_j$)|',
        cmap='YlOrRd', vmin=0, vmax=0.5,
    )

    # --- Panel (f): Explained Variance Bars ---
    individual_r2 = correlations['ols_regression']['individual_r2']
    combined_r2 = correlations['ols_regression']['r_squared']
    sc5 = sc['SC5_combined_model_r2']
    bar_panel(axes[1, 2], individual_r2, combined_r2, sc5['pass'])

    # Suptitle
    n_pass = sc['n_pass']
    n_total = sc['n_total']
    fig.suptitle(
        f'Figure 8: GCMC-Inspired Manifold Geometry of Reasoning Types (E19) '
        f'— {n_pass}/{n_total} SC met',
        fontsize=12, fontweight='bold', y=0.98,
    )

    # Save
    png_path = OUTDIR / "figure8_manifold_geometry.png"
    pdf_path = OUTDIR / "figure8_manifold_geometry.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    generate_figure()
