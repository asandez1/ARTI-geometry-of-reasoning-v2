#!/usr/bin/env python3
"""
ARTI Laboratory — Research instrument for the Active Reasoning Type Identifier.

A professional analysis tool that:
  1. Identifies reasoning type from text with full probability distribution
  2. Projects the input onto the real 10D operator manifold (PCA → 2D)
     overlaid on the 1,500-sample validation reference distribution
  3. Displays the actual geometric feature vector as a radar chart
  4. Computes manifold distances to each type centroid
  5. Provides copyable classification reports with all numerical detail

Usage:
    cd experiments/experiment5_operators
    python arti_demo.py
    # Open http://localhost:7860
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
import gradio as gr

from shared.reasoning_types import (
    ReasoningType, N_REASONING_TYPES, REASONING_TYPES,
    TYPE_NAMES, TYPE_SHORT_NAMES, TYPE_COLORS,
)
from shared.arti import ARTI, ARTIConfig
from shared.arti_v2 import ARTIV2, ARTIV2Config
from shared.encoder import SentenceTransformerEncoder
from shared.text_utils import segment_text
from arti_data import load_dataset

from shared.controller import (
    GeometricReasoningController, ControllerConfig, TypeClassifier,
    CORE_TYPE_NAMES, N_CORE_TYPES,
)
from shared.model import COFRN, COFRNConfig

# ─── Global Constants ─────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "results" / "arti_v3" / "arti_model.pt"
V2_MODEL_PATH = BASE_DIR / "results" / "arti_v2_trajectory" / "arti_v2_model.pt"
COFRN_PATH = BASE_DIR / "results" / "controller" / "cofrn_quick.pt"
TYPE_CLF_PATH = BASE_DIR / "results" / "controller" / "type_clf.pt"
DATASET_PATH = BASE_DIR / "results" / "arti_v3" / "dataset.pt"
ENCODER_NAME = "all-MiniLM-L6-v2"
ENCODER_DIM = 384
MANIFOLD_DIM = 10
SEED = 42

# Per-type validation accuracy from training (for calibration context)
VAL_ACCURACY = {
    "PhysCause": 0.093, "BehvCause": 0.560, "SysCause": 0.740,
    "Deduc": 0.613, "Induc": 0.660, "Analog": 0.967,
    "Conserv": 0.813, "Counter": 0.927, "Abduc": 0.973, "Decomp": 0.800,
}

# ─── Load Model + Encoder + Reference Data ───────────────────────────────────

print("=" * 60)
print("ARTI Laboratory — loading components")
print("=" * 60)

print("[1/3] Loading ARTI classifier...")
config = ARTIConfig(
    encoder_dim=ENCODER_DIM, manifold_dim=MANIFOLD_DIM,
    n_geometric_features=32, classifier_type='mlp',
    hidden_dim=64, dropout=0.1,
)
arti = ARTI(config)
arti.load_state_dict(torch.load(str(MODEL_PATH), weights_only=True, map_location='cpu'))
arti.eval()
print(f"  {arti.trainable_params:,} trainable params")

print("[2/3] Loading sentence encoder...")
encoder = SentenceTransformerEncoder(
    model_name=ENCODER_NAME, hidden_dim=256, load_pretrained=True,
)
print(f"  {ENCODER_NAME} (384D)")

print("[3/3] Building reference distribution from validation set...")
full_dataset = load_dataset(str(DATASET_PATH))
_, val_ds = full_dataset.split(train_ratio=0.8, seed=SEED)

# Pre-compute manifold coords + features for all validation samples
with torch.no_grad():
    val_result = arti(val_ds.embeddings)
    ref_coords = val_result['manifold_coords'].numpy()    # [1500, 10]
    ref_features = val_result['features'].numpy()          # [1500, 32]
    ref_labels = val_ds.labels.numpy()                     # [1500]

# Fit PCA on reference coords once (reused for every query)
pca = PCA(n_components=2)
ref_coords_2d = pca.fit_transform(ref_coords)  # [1500, 2]
pca_var = pca.explained_variance_ratio_

# Per-type centroids in manifold space
type_centroids = {}
type_centroids_2d = {}
for rtype in ReasoningType:
    mask = ref_labels == int(rtype)
    type_centroids[rtype] = ref_coords[mask].mean(axis=0)       # [10]
    type_centroids_2d[rtype] = ref_coords_2d[mask].mean(axis=0) # [2]

print(f"  {len(val_ds)} reference samples, PCA variance: "
      f"PC1={pca_var[0]:.1%} PC2={pca_var[1]:.1%}")

# ─── Load ARTI v2 (trajectory-based) if available ────────────────────────────

arti_v2 = None
V2_AVAILABLE = False
if V2_MODEL_PATH.exists():
    print("[3b/3] Loading ARTI v2 (trajectory-based) classifier...")
    v2_config = ARTIV2Config(
        encoder_dim=ENCODER_DIM, manifold_dim=MANIFOLD_DIM,
        traj_feature_dim=60, traj_hidden=48,
        n_classes=N_REASONING_TYPES, classifier_hidden=96, dropout=0.1,
    )
    arti_v2 = ARTIV2(v2_config)
    arti_v2.load_state_dict(
        torch.load(str(V2_MODEL_PATH), weights_only=True, map_location='cpu')
    )
    arti_v2.eval()
    V2_AVAILABLE = True
    print(f"  {arti_v2.trainable_params:,} trainable params (trajectory-based)")
else:
    print("[3b/3] ARTI v2 not found — run run_arti_v2.py first to enable v2 mode")

# ─── Load Controller (if trained) ─────────────────────────────────────────────

controller = None
CONTROLLER_AVAILABLE = False
if COFRN_PATH.exists() and TYPE_CLF_PATH.exists():
    print("[3c/3] Loading GeometricReasoningController...")
    try:
        cofrn_config = COFRNConfig(
            use_precomputed=True, encoder_input_dim=ENCODER_DIM,
            hidden_dim=256, struct_dim=128, context_dim=128,
            manifold_dim=MANIFOLD_DIM, n_anchors=16, rank=16,
            task_type='single_step',
        )
        cofrn_model = COFRN(cofrn_config)
        cofrn_model.load_state_dict(
            torch.load(str(COFRN_PATH), weights_only=True, map_location='cpu')
        )
        cofrn_model.eval()

        ctrl_config = ControllerConfig(hidden_dim=256, manifold_dim=MANIFOLD_DIM, struct_dim=128)
        controller = GeometricReasoningController(
            config=ctrl_config,
            factorization=cofrn_model.factorization,
            reasoning_engine=cofrn_model.reasoning,
        )
        controller.type_clf.load_state_dict(
            torch.load(str(TYPE_CLF_PATH), weights_only=True, map_location='cpu')
        )
        controller.eval()
        CONTROLLER_AVAILABLE = True
        print(f"  Controller loaded ({controller.trainable_params:,} params)")
    except Exception as e:
        print(f"  Controller loading failed: {e}")
else:
    print("[3c/3] Controller not trained — run run_controller.py first")

print("=" * 60)
print("Ready.\n")


# ─── Visualization: Manifold Position ─────────────────────────────────────────

def plot_manifold_position(query_coords_10d, pred_idx):
    """
    Plot the query's real position on the 2D PCA projection of the 10D manifold,
    overlaid on the full validation reference distribution.
    """
    query_2d = pca.transform(query_coords_10d.reshape(1, -1))[0]  # [2]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Reference distribution (faded)
    for rtype in ReasoningType:
        mask = ref_labels == int(rtype)
        spec = REASONING_TYPES[rtype]
        ax.scatter(
            ref_coords_2d[mask, 0], ref_coords_2d[mask, 1],
            c=spec.color, alpha=0.15, s=12, linewidths=0,
            label=None,
        )

    # Type centroids
    for rtype in ReasoningType:
        spec = REASONING_TYPES[rtype]
        c2d = type_centroids_2d[rtype]
        ax.scatter(c2d[0], c2d[1], c=spec.color, s=80, marker='D',
                   edgecolors='white', linewidths=0.8, zorder=4)
        ax.annotate(spec.short_name, (c2d[0], c2d[1]),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points',
                    color=spec.color, fontweight='bold')

    # Query point (large star)
    pred_color = REASONING_TYPES[ReasoningType(pred_idx)].color
    ax.scatter(query_2d[0], query_2d[1], c=pred_color, s=280, marker='*',
               edgecolors='black', linewidths=1.0, zorder=5)
    ax.annotate('INPUT', (query_2d[0], query_2d[1]),
                fontsize=8, ha='center', va='bottom', fontweight='bold',
                xytext=(0, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=pred_color, alpha=0.9))

    ax.set_xlabel(f"PC1 ({pca_var[0]:.1%} variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca_var[1]:.1%} variance)", fontsize=10)
    ax.set_title("Manifold Position (10D → PCA 2D)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


# ─── Text Segmentation ────────────────────────────────────────────────────────
# segment_text() is imported from shared.text_utils


# ─── Visualization: Manifold Trajectory ───────────────────────────────────────

def plot_manifold_trajectory(text: str, pred_idx: int):
    """
    Encode each clause of the text independently, project through the 10D
    manifold, PCA to 2D, and draw the real trajectory the reasoning takes.

    Shows: path shape, direction arrows, per-step numbered points,
    delta magnitudes, curvature angles, and measured trajectory stats.
    """
    segments = segment_text(text)
    n_seg = len(segments)

    # Encode each segment independently
    with torch.no_grad():
        seg_embs = encoder.encode_texts(segments)  # [n_seg, 384]
        seg_result = arti(seg_embs)
    seg_coords_10d = seg_result['manifold_coords'].numpy()  # [n_seg, 10]
    seg_coords_2d = pca.transform(seg_coords_10d)           # [n_seg, 2]

    pred_color = REASONING_TYPES[ReasoningType(pred_idx)].color

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Light reference distribution
    for rtype in ReasoningType:
        mask = ref_labels == int(rtype)
        spec = REASONING_TYPES[rtype]
        ax.scatter(ref_coords_2d[mask, 0], ref_coords_2d[mask, 1],
                   c=spec.color, alpha=0.07, s=8, linewidths=0)

    # Type centroids (faded diamonds)
    for rtype in ReasoningType:
        spec = REASONING_TYPES[rtype]
        c2d = type_centroids_2d[rtype]
        ax.scatter(c2d[0], c2d[1], c=spec.color, s=50, marker='D',
                   edgecolors='white', linewidths=0.5, alpha=0.4, zorder=3)
        ax.annotate(spec.short_name, (c2d[0], c2d[1]),
                    fontsize=6, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points',
                    color=spec.color, alpha=0.5)

    # Draw the trajectory path with arrows
    for i in range(n_seg - 1):
        dx = seg_coords_2d[i+1, 0] - seg_coords_2d[i, 0]
        dy = seg_coords_2d[i+1, 1] - seg_coords_2d[i, 1]
        ax.annotate('', xy=seg_coords_2d[i+1], xytext=seg_coords_2d[i],
                     arrowprops=dict(
                         arrowstyle='->', color=pred_color,
                         lw=2.5, connectionstyle='arc3,rad=0.1',
                         alpha=0.5 + 0.5 * (i / max(n_seg - 2, 1)),  # fade in
                     ), zorder=4)

    # Draw numbered points at each segment
    for i, (x, y) in enumerate(seg_coords_2d):
        is_first = (i == 0)
        is_last = (i == n_seg - 1)
        ms = 13 if (is_first or is_last) else 10
        ec = 'black' if (is_first or is_last) else pred_color
        lw = 1.5 if (is_first or is_last) else 0.8
        ax.scatter(x, y, c=pred_color, s=ms**2, marker='o',
                   edgecolors=ec, linewidths=lw, zorder=5)
        ax.text(x, y, str(i + 1), ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=6)

    # Compute trajectory stats
    deltas = np.diff(seg_coords_10d, axis=0)  # [n_seg-1, 10]
    delta_mags = np.linalg.norm(deltas, axis=1)
    total_displacement = np.linalg.norm(seg_coords_10d[-1] - seg_coords_10d[0])
    total_path_length = delta_mags.sum()

    # Curvature: angle between consecutive deltas
    curvatures = []
    for i in range(len(deltas) - 1):
        cos = np.dot(deltas[i], deltas[i+1]) / (
            np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[i+1]) + 1e-9)
        curvatures.append(np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi)
    mean_curvature = np.mean(curvatures) if curvatures else 0.0

    # Direction consistency: mean pairwise cosine of deltas
    if len(deltas) >= 2:
        cosines = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                c = np.dot(deltas[i], deltas[j]) / (
                    np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j]) + 1e-9)
                cosines.append(c)
        consistency = np.mean(cosines)
    else:
        consistency = 1.0

    # Linearity ratio: direct displacement / total path length
    linearity = total_displacement / (total_path_length + 1e-9)

    # Stats annotation box
    stats_text = (
        f"Steps: {n_seg}   Path: {total_path_length:.2f}   "
        f"Displ: {total_displacement:.2f}\n"
        f"Linearity: {linearity:.2f}   "
        f"Curvature: {mean_curvature:.0f}°   "
        f"Consistency: {consistency:+.2f}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=7.5, family='monospace', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))

    ax.set_xlabel(f"PC1 ({pca_var[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca_var[1]:.1%} var)", fontsize=10)
    ax.set_title("Manifold Trajectory (clause-by-clause)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig, segments, delta_mags, mean_curvature, consistency, linearity


# ─── Visualization: Geometric Feature Radar ──────────────────────────────────

def plot_feature_radar(query_coords_10d, query_features_32d, pred_idx):
    """
    Radar chart of the interpretable raw geometric properties extracted from
    the real manifold coordinates, compared to the predicted type's centroid.
    """
    # Compute interpretable features from the raw 10D coords
    coords = query_coords_10d
    dist_from_origin = np.linalg.norm(coords)
    coord_variance = np.var(coords)
    coord_range = np.ptp(coords)
    coord_mean = np.mean(np.abs(coords))
    top3_energy = np.sort(np.abs(coords))[-3:].sum() / (np.abs(coords).sum() + 1e-9)
    entropy = -np.sum(np.abs(coords) / (np.abs(coords).sum() + 1e-9)
                      * np.log(np.abs(coords) / (np.abs(coords).sum() + 1e-9) + 1e-9))

    # Distances to all centroids
    centroid_dists = {}
    for rtype in ReasoningType:
        d = np.linalg.norm(coords - type_centroids[rtype])
        centroid_dists[rtype] = d
    nearest_dist = min(centroid_dists.values())
    farthest_dist = max(centroid_dists.values())

    # Labels and values for radar (short labels to avoid clipping)
    labels = [
        '||x||',
        'Var',
        'Range',
        'Top3 E',
        'Entropy',
        'Prox.',
        'Mean',
    ]
    values = [
        dist_from_origin,
        coord_variance,
        coord_range,
        top3_energy,
        entropy / 3.0,       # Normalize to ~[0, 1]
        1.0 - nearest_dist / (farthest_dist + 1e-9),  # Proximity (higher = closer)
        coord_mean,
    ]
    # Normalize to [0, 1] for display
    max_val = max(max(values), 1.0)
    values_norm = [v / max_val for v in values]

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_norm += values_norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    pred_color = REASONING_TYPES[ReasoningType(pred_idx)].color

    ax.fill(angles, values_norm, color=pred_color, alpha=0.2)
    ax.plot(angles, values_norm, color=pred_color, lw=2, marker='o', ms=5)

    # Value annotations at each vertex
    for a, v, vr in zip(angles[:-1], values, values_norm):
        ax.text(a, vr + 0.08, f'{v:.2f}', ha='center', va='center',
                fontsize=7, color='#555')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=6, color='#aaa')
    ax.set_title("Geometric Feature Profile", fontsize=11, fontweight='bold', y=1.08)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=1.5)
    return fig


# ─── Visualization: Probability Bar Chart ────────────────────────────────────

def plot_probabilities(probs, pred_idx):
    """Horizontal bar chart of type probabilities with val accuracy annotation."""
    fig, ax = plt.subplots(figsize=(6, 4))

    names = [REASONING_TYPES[t].short_name for t in ReasoningType]
    colors = [REASONING_TYPES[t].color for t in ReasoningType]

    bars = ax.barh(names, probs, color=colors, edgecolor='white', linewidth=0.5)
    bars[pred_idx].set_edgecolor('black')
    bars[pred_idx].set_linewidth(1.5)

    # Annotate: probability value + val accuracy in gray
    for i, v in enumerate(probs):
        va = VAL_ACCURACY.get(names[i], 0)
        if v > 0.015:
            ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=8.5, color='#333')
        # Val accuracy at right margin
        ax.text(0.98, i, f'val:{va:.0%}', va='center', ha='right',
                fontsize=7, color='#999', fontstyle='italic')

    ax.set_xlim(0, 1.02)
    ax.set_xlabel('Probability', fontsize=10)
    ax.set_title('Classification Distribution', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15, linestyle='--')
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    return fig


# ─── Visualization: 10D Coordinate Bar ───────────────────────────────────────

def plot_manifold_coords(query_coords_10d, pred_idx):
    """Bar chart of the raw 10D manifold coordinates with type centroid overlay."""
    fig, ax = plt.subplots(figsize=(6, 3.2))

    dims = np.arange(MANIFOLD_DIM)
    pred_color = REASONING_TYPES[ReasoningType(pred_idx)].color

    # Query coords as bars
    ax.bar(dims, query_coords_10d, color=pred_color, alpha=0.7,
           label='Input', edgecolor='white', linewidth=0.5, width=0.6)

    # Predicted type centroid as connected markers on bar centers
    centroid = type_centroids[ReasoningType(pred_idx)]
    ax.plot(dims, centroid, color='#333', lw=1.5, linestyle='--',
            marker='_', ms=12, mew=2, alpha=0.7, label='Type centroid')

    ax.set_xlabel('Manifold Dimension', fontsize=9)
    ax.set_ylabel('Coordinate Value', fontsize=9)
    ax.set_title('10D Manifold Coordinates', fontsize=11, fontweight='bold')
    ax.set_xticks(dims)
    ax.set_xticklabels([f'd{i}' for i in dims], fontsize=8)
    ax.legend(fontsize=8, loc='best', framealpha=0.8)
    ax.grid(axis='y', alpha=0.15, linestyle='--')
    ax.axhline(0, color='#aaa', lw=0.8)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


# ─── Report Generation ────────────────────────────────────────────────────────

# Per-type v2 validation accuracy (from training run)
V2_VAL_ACCURACY = {
    "PhysCause": 0.373, "BehvCause": 0.693, "SysCause": 0.987,
    "Deduc": 0.447, "Induc": 0.887, "Analog": 0.960,
    "Conserv": 0.713, "Counter": 1.000, "Abduc": 0.973, "Decomp": 0.760,
}
V2_OVERALL_ACC = 0.779


def build_report(text, probs, pred_idx, confidence, query_coords_10d,
                 query_features_32d, centroid_dists, elapsed_ms,
                 segments=None, delta_mags=None,
                 mean_curv=0.0, consistency=0.0, linearity=0.0,
                 model_version="v1", v1_pred=None, v2_pred=None):
    """Build a full-text classification report for copy/paste."""
    spec = REASONING_TYPES[ReasoningType(pred_idx)]
    sig = spec.geometric_signature

    # Select accuracy dict based on active model
    is_v2 = model_version == "v2"
    acc_dict = V2_VAL_ACCURACY if is_v2 else VAL_ACCURACY
    overall_acc = V2_OVERALL_ACC if is_v2 else 0.715

    # Entropy of distribution
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(N_REASONING_TYPES)
    norm_entropy = entropy / max_entropy

    # Sort by probability
    order = np.argsort(probs)[::-1]

    lines = []
    lines.append("=" * 60)
    lines.append(f"ARTI CLASSIFICATION REPORT  [{model_version}]")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Input:  {text}")
    lines.append("")
    lines.append("--- Identification ---")
    lines.append(f"Type:       {spec.name} ({spec.short_name})")
    lines.append(f"Confidence: {confidence:.1%}")
    lines.append(f"Entropy:    {entropy:.3f} / {max_entropy:.3f} "
                 f"(normalized: {norm_entropy:.2f})")
    val_acc = acc_dict.get(spec.short_name, 0)
    lines.append(f"Val Acc:    {val_acc:.1%} (model reliability for this type)")

    # v1/v2 comparison line
    if v1_pred is not None and v2_pred is not None:
        v1_name = REASONING_TYPES[ReasoningType(v1_pred['idx'])].short_name
        v2_name = REASONING_TYPES[ReasoningType(v2_pred['idx'])].short_name
        agree = "AGREE" if v1_pred['idx'] == v2_pred['idx'] else "DISAGREE"
        lines.append("")
        lines.append("--- v1/v2 Comparison ---")
        lines.append(f"  v1 (position):   {v1_name:>10}  ({v1_pred['conf']:.1%})")
        lines.append(f"  v2 (trajectory): {v2_name:>10}  ({v2_pred['conf']:.1%})")
        lines.append(f"  Models: {agree}")

    lines.append("")
    lines.append("--- Probability Distribution ---")
    for i in order:
        rtype = ReasoningType(i)
        s = REASONING_TYPES[rtype]
        marker = " <--" if i == pred_idx else ""
        lines.append(f"  {s.short_name:>10}  {probs[i]:6.1%}  "
                     f"(val acc: {acc_dict.get(s.short_name, 0):.0%}){marker}")
    lines.append("")
    lines.append("--- Geometric Signature ---")
    lines.append(f"  Expected:     {sig.trajectory_shape}")
    lines.append(f"  Curvature:    {sig.curvature:10s}  (measured: {mean_curv:.0f}°)")
    lines.append(f"  Displacement: {sig.displacement_magnitude}")
    lines.append(f"  Consistency:  {sig.direction_consistency:10s}  (measured: {consistency:+.2f})")
    lines.append(f"  Linearity:    {linearity:.2f} (1.0 = straight, 0.0 = winding)")
    lines.append(f"  Operators:    {', '.join(spec.level1_operators)}")
    lines.append(f"  Domains:      {', '.join(spec.common_domains)}")

    if segments:
        lines.append("")
        lines.append(f"--- Trajectory Segments ({len(segments)} steps) ---")
        for i, seg in enumerate(segments):
            d = f"  Δ={delta_mags[i-1]:.3f}" if (delta_mags is not None and i > 0) else ""
            lines.append(f"  [{i+1}] {seg[:70]}{d}")

    lines.append("")
    lines.append("--- Manifold Coordinates (10D) ---")
    coord_str = "  [" + ", ".join(f"{v:+.4f}" for v in query_coords_10d) + "]"
    lines.append(coord_str)
    lines.append(f"  ||x|| = {np.linalg.norm(query_coords_10d):.4f}   "
                 f"var = {np.var(query_coords_10d):.4f}")
    lines.append("")
    lines.append("--- Centroid Distances ---")
    sorted_dists = sorted(centroid_dists.items(), key=lambda x: x[1])
    for rtype, d in sorted_dists:
        s = REASONING_TYPES[rtype]
        marker = " <-- nearest" if d == sorted_dists[0][1] else ""
        lines.append(f"  {s.short_name:>10}  {d:.4f}{marker}")
    lines.append("")
    lines.append("--- Model Info ---")
    lines.append(f"  Model:      ARTI {model_version}")
    lines.append(f"  Encoder:    {ENCODER_NAME} (384D)")
    if is_v2:
        lines.append(f"  Classifier: Trajectory MLP (8,890 trainable + 7,910 frozen manifold)")
        lines.append(f"  Features:   60D trajectory -> 48D hidden -> 96D -> 10 types")
        lines.append(f"  Method:     Clause segmentation -> manifold projection -> trajectory geometry")
        lines.append(f"  Val Acc:    {V2_OVERALL_ACC:.1%} overall (+{V2_OVERALL_ACC - 0.715:.1%} vs v1)")
    else:
        lines.append(f"  Classifier: MLP (11,472 params)")
        lines.append(f"  Manifold:   10D projection")
        lines.append(f"  Features:   32D geometric vector")
        lines.append(f"  Val Acc:    71.5% overall")
    lines.append(f"  Latency:    {elapsed_ms:.0f} ms")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ─── v2 Classification ───────────────────────────────────────────────────────

def classify_v2(text: str):
    """
    Classify with ARTI v2 (trajectory-based).

    Returns: (pred_idx, confidence, probs, elapsed_ms) or None if v2 unavailable.
    """
    if arti_v2 is None:
        return None

    t0 = time.time()
    segments = segment_text(text)
    with torch.no_grad():
        clause_embs = encoder.encode_texts(segments)  # [n_clauses, 384]
        result = arti_v2([clause_embs])

    elapsed_ms = (time.time() - t0) * 1000

    probs = result['probabilities'].squeeze(0).cpu().numpy()
    pred_idx = int(result['type'].item())
    confidence = float(result['confidence'].item())

    return {
        'pred_idx': pred_idx,
        'confidence': confidence,
        'probs': probs,
        'elapsed_ms': elapsed_ms,
        'n_clauses': len(segments),
    }


# ─── Controller Classification ────────────────────────────────────────────────

def classify_controller(text: str):
    """
    Classify with GeometricReasoningController.

    Returns dict with type detection, routing decision, and type config,
    or None if controller unavailable.
    """
    if controller is None:
        return None

    t0 = time.time()
    with torch.no_grad():
        raw_emb = encoder.encode_texts([text])  # [1, 384]
        s0 = cofrn_model.encoder(raw_emb)  # [1, 256]

        # Factorize
        fact_out = controller.factorization(s0)
        structural = fact_out['structural']
        anchor_weights = fact_out['weights']

        # Detect type
        type_probs, detected_type, confidence, manifold_coords = controller.detect_type(structural)

        # Routing decision
        routing_weight = controller.compute_routing(type_probs, confidence, anchor_weights)

    elapsed_ms = (time.time() - t0) * 1000

    t = detected_type[0].item()
    depth, delta = controller.get_type_config(t)
    route = "structured" if (t in set(controller.config.structured_types) or
                             confidence[0].item() < controller.config.confidence_threshold) else "fast"

    # Top anchors
    aw = anchor_weights[0]
    topk_vals, topk_idx = aw.topk(controller.config.top_k_anchors)

    # Anchor entropy
    aw_np = aw.cpu().numpy()
    aw_np = aw_np / (aw_np.sum() + 1e-10)
    anchor_ent = float(-np.sum(aw_np * np.log(aw_np + 1e-10)))

    return {
        'core_type': CORE_TYPE_NAMES[t],
        'core_type_id': t,
        'confidence': confidence[0].item(),
        'type_probs': type_probs[0].cpu().numpy(),
        'route': route,
        'depth': depth,
        'delta': delta,
        'routing_weight': routing_weight[0, 0].item(),
        'top_anchors': topk_idx.tolist(),
        'anchor_entropy': anchor_ent,
        'elapsed_ms': elapsed_ms,
    }


# ─── Main Inference Pipeline ──────────────────────────────────────────────────

def identify_reasoning(text: str, model_choice: str = "v1 (position)"):
    """
    Full pipeline: text -> encoder -> ARTI v1 or v2 -> all outputs.
    Returns: (report, prob_fig, manifold_fig, traj_fig, radar_fig, coord_fig, type_info)
    """
    if not text or not text.strip():
        empty = plt.figure(figsize=(4, 3))
        empty.text(0.5, 0.5, "Enter a phrase to analyze", ha='center', va='center',
                   fontsize=12, color='#999', transform=empty.transFigure)
        plt.close(empty)
        return ("", empty, empty, empty, empty, empty, "")

    use_v2 = ("v2" in model_choice) and V2_AVAILABLE
    use_controller = ("Controller" in model_choice) and CONTROLLER_AVAILABLE

    t0 = time.time()

    # v1 classification (always run for visualizations + fallback)
    with torch.no_grad():
        raw_emb = encoder.encode_texts([text])  # [1, 384]
        v1_result = arti(raw_emb)

    v1_elapsed = (time.time() - t0) * 1000

    v1_probs = v1_result['probabilities'].squeeze(0).cpu().numpy()
    v1_pred_idx = int(v1_result['type'].item())
    v1_confidence = float(v1_result['confidence'].item())
    query_coords = v1_result['manifold_coords'].squeeze(0).cpu().numpy()
    query_features = v1_result['features'].squeeze(0).cpu().numpy()

    # v2 classification (if selected)
    v2_info = classify_v2(text) if V2_AVAILABLE else None

    # Controller classification (if selected)
    ctrl_info = classify_controller(text) if CONTROLLER_AVAILABLE else None

    # Use selected model's prediction for display
    if use_controller and ctrl_info:
        # Controller uses 6 core types; map back to closest ARTI 10-type for visualizations
        pred_idx = v1_pred_idx  # Visualizations use ARTI v1's 10-type prediction
        confidence = ctrl_info['confidence']
        probs = v1_probs  # Probability bars still show 10-type v1 distribution
        elapsed_ms = ctrl_info['elapsed_ms']
        model_label = "Controller"
    elif use_v2 and v2_info:
        pred_idx = v2_info['pred_idx']
        confidence = v2_info['confidence']
        probs = v2_info['probs']
        elapsed_ms = v2_info['elapsed_ms']
        model_label = "v2 (trajectory)"
    else:
        pred_idx = v1_pred_idx
        confidence = v1_confidence
        probs = v1_probs
        elapsed_ms = v1_elapsed
        model_label = "v1 (position)"

    pred_type = ReasoningType(pred_idx)
    spec = REASONING_TYPES[pred_type]

    # Centroid distances
    centroid_dists = {}
    for rtype in ReasoningType:
        centroid_dists[rtype] = float(np.linalg.norm(query_coords - type_centroids[rtype]))

    # Build visualizations
    prob_fig = plot_probabilities(probs, pred_idx)
    manifold_fig = plot_manifold_position(query_coords, pred_idx)
    traj_fig, segments, delta_mags, mean_curv, consist, linear = \
        plot_manifold_trajectory(text, pred_idx)
    radar_fig = plot_feature_radar(query_coords, query_features, pred_idx)
    coord_fig = plot_manifold_coords(query_coords, pred_idx)

    # Build v1/v2 comparison dicts for the report
    v1_pred = {'idx': v1_pred_idx, 'conf': v1_confidence}
    v2_pred = {'idx': v2_info['pred_idx'], 'conf': v2_info['confidence']} if v2_info else None
    active_version = "controller" if use_controller else ("v2" if use_v2 else "v1")

    # Build report (with trajectory stats + model-aware metadata)
    report = build_report(text, probs, pred_idx, confidence, query_coords,
                          query_features, centroid_dists, elapsed_ms,
                          segments, delta_mags, mean_curv, consist, linear,
                          model_version=active_version,
                          v1_pred=v1_pred, v2_pred=v2_pred)

    # Append controller-specific section to report if active
    if use_controller and ctrl_info:
        ctrl_lines = [
            "",
            "--- Controller Routing Decision ---",
            f"  Core Type:     {ctrl_info['core_type']} (6-type classification)",
            f"  Confidence:    {ctrl_info['confidence']:.1%}",
            f"  Route:         {ctrl_info['route']}",
            f"  Tree Depth:    {ctrl_info['depth']}",
            f"  Coherence (δ): {ctrl_info['delta']:.2f}",
            f"  Top Anchors:   {ctrl_info['top_anchors']}",
            f"  Anchor Entropy: {ctrl_info['anchor_entropy']:.3f}",
            f"  Routing Weight: {ctrl_info['routing_weight']:.3f} (0=fast, 1=structured)",
            "",
            "--- 6-Type Probability Distribution ---",
        ]
        for i, name in enumerate(CORE_TYPE_NAMES):
            marker = " <--" if i == ctrl_info['core_type_id'] else ""
            ctrl_lines.append(f"  {name:>16}  {ctrl_info['type_probs'][i]:.1%}{marker}")
        report += "\n" + "\n".join(ctrl_lines)

    # Compact type info header
    sig = spec.geometric_signature
    nearest = min(centroid_dists, key=centroid_dists.get)
    nearest_name = REASONING_TYPES[nearest].short_name
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    acc_dict = V2_VAL_ACCURACY if use_v2 else VAL_ACCURACY

    type_info = (
        f"### {spec.name} &nbsp; *({model_label})*\n"
        f"**Confidence:** {confidence:.1%} &nbsp; | &nbsp; "
        f"**Entropy:** {entropy:.2f} &nbsp; | &nbsp; "
        f"**Latency:** {elapsed_ms:.0f} ms\n\n"
    )

    # Controller routing info
    if use_controller and ctrl_info:
        type_info += (
            f"**Core Type:** {ctrl_info['core_type']} &nbsp; | &nbsp; "
            f"**Route:** {ctrl_info['route']} &nbsp; | &nbsp; "
            f"**Depth:** {ctrl_info['depth']} &nbsp; | &nbsp; "
            f"**Delta:** {ctrl_info['delta']:.2f} &nbsp; | &nbsp; "
            f"**Top Anchors:** {ctrl_info['top_anchors']}\n\n"
        )

    # Side-by-side comparison if v2 available
    if v2_info is not None and not use_controller:
        v1_name = REASONING_TYPES[ReasoningType(v1_pred_idx)].short_name
        v2_name = REASONING_TYPES[ReasoningType(v2_info['pred_idx'])].short_name
        agree = "agree" if v1_pred_idx == v2_info['pred_idx'] else "**disagree**"
        type_info += (
            f"**v1 (position):** {v1_name} ({v1_confidence:.1%}) &nbsp; | &nbsp; "
            f"**v2 (trajectory):** {v2_name} ({v2_info['confidence']:.1%}) &nbsp; | &nbsp; "
            f"Models {agree} &nbsp; | &nbsp; "
            f"**v2 clauses:** {v2_info['n_clauses']}\n\n"
        )

    type_info += (
        f"**Trajectory:** {sig.trajectory_shape} &nbsp; | &nbsp; "
        f"**Curvature:** {sig.curvature} (measured: {mean_curv:.0f}°) &nbsp; | &nbsp; "
        f"**Displacement:** {sig.displacement_magnitude} &nbsp; | &nbsp; "
        f"**Consistency:** {sig.direction_consistency} (measured: {consist:+.2f})\n\n"
        f"**Linearity:** {linear:.2f} &nbsp; | &nbsp; "
        f"**Steps:** {len(segments)} &nbsp; | &nbsp; "
        f"**Nearest centroid:** {nearest_name} (d={centroid_dists[nearest]:.3f}) &nbsp; | &nbsp; "
        f"**Domains:** {', '.join(spec.common_domains[:4])}\n\n"
        f"**Val accuracy for this type:** {acc_dict.get(spec.short_name, 0):.0%}"
    )

    return (report, prob_fig, manifold_fig, traj_fig, radar_fig, coord_fig, type_info)


# ─── Example Phrases ──────────────────────────────────────────────────────────

EXAMPLES = [
    ["The temperature dropped below freezing, so the pipes burst from ice expansion."],
    ["She studied every day for a month and scored 98% on the exam."],
    ["The drought killed crops, causing food shortages that triggered mass migration."],
    ["All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is warm-blooded."],
    ["Every patient given drug X improved. The evidence suggests drug X is effective."],
    ["A firewall protects a network just as a moat protects a castle."],
    ["$500 was transferred from checking to savings. The total stays the same."],
    ["What if the bridge had been inspected? The failure would have been prevented."],
    ["The lawn is wet but it hasn't rained. The best explanation is overnight dew."],
    ["First, calculate distance. Second, find fuel cost. Third, add tolls."],
]


# ─── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
.report-box textarea { font-family: monospace !important; font-size: 12px !important; }
"""

with gr.Blocks(
    title="ARTI Laboratory",
    theme=gr.themes.Base(),
    css=CSS,
) as demo:

    v2_status = "available" if V2_AVAILABLE else "not trained yet (run run_arti_v2.py)"
    ctrl_status = "available" if CONTROLLER_AVAILABLE else "not trained yet (run run_controller.py)"
    gr.Markdown(
        "# ARTI Laboratory\n"
        "*Active Reasoning Type Identifier — Research Instrument*\n\n"
        "Classifies reasoning type from text using a 10D operator manifold learned "
        "from 7,500 labeled reasoning traces. The visualizations show **real model "
        "vectors**, not schematics — each point is the actual manifold projection "
        "of the input phrase.\n\n"
        f"**v2 trajectory model:** {v2_status}\n\n"
        f"**Controller (routing):** {ctrl_status}"
    )

    # Build model choices list
    _model_choices = ["v1 (position)", "v2 (trajectory)"]
    if CONTROLLER_AVAILABLE:
        _model_choices.append("Controller (routing)")
    _default_model = "v2 (trajectory)" if V2_AVAILABLE else "v1 (position)"
    _interactive = V2_AVAILABLE or CONTROLLER_AVAILABLE

    # ── Input Row ──
    with gr.Row():
        with gr.Column(scale=4):
            text_input = gr.Textbox(
                label="Input reasoning phrase",
                placeholder="Type or paste a reasoning statement...",
                lines=3,
            )
        with gr.Column(scale=1, min_width=180):
            model_choice = gr.Radio(
                choices=_model_choices,
                value=_default_model,
                label="Model",
                interactive=_interactive,
            )
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")

    # ── Type Identification Header ──
    type_info = gr.Markdown(value="*Results will appear here after analysis.*")

    # ── Main Visualization Panels ──
    with gr.Row():
        with gr.Column():
            manifold_plot = gr.Plot(label="Manifold Position (real 10D → PCA 2D)")
        with gr.Column():
            traj_plot = gr.Plot(label="Manifold Trajectory (clause-by-clause)")

    with gr.Row():
        with gr.Column():
            prob_plot = gr.Plot(label="Classification Distribution")
        with gr.Column():
            radar_plot = gr.Plot(label="Geometric Feature Profile")

    with gr.Row():
        coord_plot = gr.Plot(label="10D Manifold Coordinates")

    # ── Full Report (copyable) ──
    with gr.Accordion("Full Classification Report (copy-paste)", open=False):
        report_box = gr.Textbox(
            label="Classification Report",
            lines=30,
            max_lines=50,
            buttons=["copy"],
            interactive=False,
            elem_classes=["report-box"],
        )

    # ── Model Card ──
    with gr.Accordion("Model Card", open=False):
        gr.Markdown(f"""
| Property | Value |
|----------|-------|
| **Encoder** | `{ENCODER_NAME}` (384D, frozen) |
| **Manifold** | 10D learned projection (384 → 20 → 10) |
| **Features** | 32D geometric vector (coords + magnitude + direction + curvature + variance) |
| **Classifier** | 2-layer MLP (32 → 64 → 10), {arti.trainable_params:,} trainable params |
| **Training** | 6,000 samples, 68 epochs, AdamW + cosine LR, early stop patience=15 |
| **Validation** | 1,500 samples (balanced, 150/type), **71.5% accuracy** |
| **Random chance** | 10% (10 types) |
| **Reference** | Paper 11 E4: operator space is ~10D continuous manifold (silhouette=0.33) |

**Per-Type Validation Accuracy:**

| Type | Acc | Type | Acc |
|------|-----|------|-----|
| PhysCause | 9.3% | Analog | 96.7% |
| BehvCause | 56.0% | Conserv | 81.3% |
| SysCause | 74.0% | Counter | 92.7% |
| Deduc | 61.3% | Abduc | 97.3% |
| Induc | 66.0% | Decomp | 80.0% |

**Architecture:**
`text → SentenceTransformer(384D) → ManifoldProj(10D) → GeometricFeatures(32D) → MLP → P(type)`

**Geometric Features Extracted:**
Manifold coordinates (10D) + delta magnitude (1D) + delta direction (10D) +
curvature (1D) + distance from origin (1D) + coordinate variance (1D) → MLP → 32D

**Reference Distribution:**
{len(val_ds)} validation samples shown as background in manifold plot.
PCA explains {pca_var[0]:.1%} + {pca_var[1]:.1%} = {sum(pca_var):.1%} of 10D variance.
        """)

    # ── Examples ──
    all_inputs = [text_input, model_choice]
    all_outputs = [report_box, prob_plot, manifold_plot, traj_plot,
                   radar_plot, coord_plot, type_info]

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        outputs=all_outputs,
        fn=identify_reasoning,
        cache_examples=False,
    )

    # ── Event Binding ──
    submit_btn.click(fn=identify_reasoning, inputs=all_inputs, outputs=all_outputs)
    text_input.submit(fn=identify_reasoning, inputs=all_inputs, outputs=all_outputs)


if __name__ == "__main__":
    print("\nStarting ARTI Laboratory at http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)
