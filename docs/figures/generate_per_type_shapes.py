#!/usr/bin/env python3
"""
Generate per-type geometric signature visualization for Section 2.

2×5 grid: each panel highlights one reasoning type against a faded background
of all other points, showing the actual geometric "shape" of each type in
UMAP-projected embedding space. Includes density contours and per-type
ARTI ensemble accuracy.
"""

import numpy as np
import torch
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from pathlib import Path

RESULTS = Path("/home/qstar/AI/Research_papers/paper13_The_geometry_of_Machine_Reasoning/experiments/experiment5_operators/results")
OUTDIR = Path("/home/qstar/AI/Research_papers/paper13_The_geometry_of_Machine_Reasoning/docs/figures")

# Label ID → (display name, color, ARTI ensemble accuracy)
TYPE_INFO = {
    0: ("Physical Cause",    "#d62728", 38.8),
    1: ("Behavioral Cause",  "#8c564b", 76.7),
    2: ("Systemic Cause",    "#e377c2", 98.9),
    3: ("Deduction",         "#1f77b4", 56.8),
    4: ("Induction",         "#2ca02c", 96.4),
    5: ("Analogy",           "#9467bd", 99.3),
    6: ("Conservation",      "#ff7f0e", 90.3),
    7: ("Counterfactual",    "#17becf", 99.4),
    8: ("Abduction",         "#bcbd22", 98.7),
    9: ("Decomposition",     "#7f7f7f", 86.8),
}

# Ordering: group by accuracy tier for visual narrative
# Top row: 5 highest-accuracy types (strong geometric signal)
# Bottom row: 5 lower-accuracy types (weaker/overlapping signal)
PANEL_ORDER = [
    7,  # Counterfactual  99.4%
    5,  # Analogy         99.3%
    2,  # Systemic Cause  98.9%
    8,  # Abduction       98.7%
    4,  # Induction       96.4%
    6,  # Conservation    90.3%
    9,  # Decomposition   86.8%
    1,  # Behavioral C.   76.7%
    3,  # Deduction       56.8%
    0,  # Physical Cause  38.8%
]


def load_data():
    data = torch.load(RESULTS / "arti_v3/dataset.pt", weights_only=False)
    X = data['embeddings'].numpy()  # [7500, 384]
    y = data['labels'].numpy()      # [7500]
    return X, y


def compute_umap(X, n_neighbors=30, min_dist=0.25, random_state=42):
    """UMAP to 2D, matching the paper's Figure 1 settings."""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='euclidean',
        n_jobs=1,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return reducer.fit_transform(X_norm)


def draw_density_contour(ax, x, y, color, levels=3):
    """Draw KDE density contours for a point cloud."""
    if len(x) < 10:
        return
    try:
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method=0.3)
        xmin, xmax = x.min() - 1, x.max() + 1
        ymin, ymax = y.min() - 1, y.max() + 1
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z = kde(positions).reshape(xx.shape)
        ax.contour(xx, yy, z, levels=levels, colors=[color],
                   alpha=0.5, linewidths=0.8)
    except Exception:
        pass  # KDE can fail on degenerate distributions


def main():
    print("Loading ARTI v3 dataset...")
    X, y = load_data()
    print(f"  Embeddings: {X.shape}, Labels: {y.shape}")

    print("Computing UMAP (this takes ~30s)...")
    X_2d = compute_umap(X)
    print(f"  UMAP done: {X_2d.shape}")

    # Global axis limits with padding
    pad = 1.5
    xlim = (X_2d[:, 0].min() - pad, X_2d[:, 0].max() + pad)
    ylim = (X_2d[:, 1].min() - pad, X_2d[:, 1].max() + pad)

    # ── Create 2×5 figure ──
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.15)

    for idx, label_id in enumerate(PANEL_ORDER):
        row, col = divmod(idx, 5)
        ax = axes[row, col]

        name, color, acc = TYPE_INFO[label_id]
        mask = y == label_id
        n = mask.sum()

        # Background: all other points in light gray
        bg_mask = ~mask
        ax.scatter(X_2d[bg_mask, 0], X_2d[bg_mask, 1],
                   c='#E5E7EB', s=3, alpha=0.25, edgecolors='none',
                   rasterized=True, zorder=1)

        # Foreground: this type highlighted
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, s=14, alpha=0.7, edgecolors='none',
                   rasterized=True, zorder=3)

        # Density contours
        draw_density_contour(ax, X_2d[mask, 0], X_2d[mask, 1], color, levels=2)

        # Title with accuracy
        acc_color = '#059669' if acc >= 90 else ('#D97706' if acc >= 70 else '#DC2626')
        ax.set_title(f'{name}\n', fontsize=9.5, fontweight='bold', pad=2)
        # Accuracy badge below title
        ax.text(0.5, 1.01, f'ARTI: {acc:.0f}%',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=acc_color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=acc_color, linewidth=0.6, alpha=0.9))

        # Count annotation
        ax.text(0.03, 0.03, f'n={n}', transform=ax.transAxes,
                fontsize=7, color='#6B7280', va='bottom')

        # Styling
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#D1D5DB')
            spine.set_linewidth(0.5)

    # Row labels
    fig.text(0.01, 0.73, 'Strong geometric\nsignal (>90%)',
             ha='left', va='center', fontsize=9, fontstyle='italic',
             color='#059669', rotation=90)
    fig.text(0.01, 0.29, 'Weaker / overlapping\nsignal (<90%)',
             ha='left', va='center', fontsize=9, fontstyle='italic',
             color='#D97706', rotation=90)

    # Suptitle
    fig.suptitle('Geometric Signatures of 10 Reasoning Types (384D → UMAP 2D)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02,
             'Each panel highlights one reasoning type (color) against all others (gray). '
             'Density contours show cluster shape.\n'
             'Top row: types with strong, well-separated geometric clusters (ARTI >90%). '
             'Bottom row: types with diffuse or overlapping geometry.',
             ha='center', fontsize=8.5, color='#6B7280', fontstyle='italic',
             linespacing=1.5)

    plt.savefig(OUTDIR / "figure_per_type_shapes.png", dpi=300,
                bbox_inches='tight', pad_inches=0.2)
    plt.savefig(OUTDIR / "figure_per_type_shapes.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Per-type shapes figure saved to {OUTDIR}/figure_per_type_shapes.png/.pdf")


if __name__ == '__main__':
    main()
