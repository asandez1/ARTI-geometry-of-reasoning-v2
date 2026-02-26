#!/usr/bin/env python3
"""
Generate Figure for Section 6.1: ARTIRoutedScorer Architecture Diagram.

Publication-quality architecture diagram showing:
  - Left column: ARTI routing path (frozen type detection → alpha)
  - Right column: CO-FRN reasoning path (factorize → manifold → transform)
  - Bottom: dual scoring heads blended by alpha
  - Color coding: frozen (blue), trainable (green), blend (gold)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUTDIR = Path("/home/qstar/AI/Research_papers/paper13_The_geometry_of_Machine_Reasoning/docs/figures")

# ── Color palette ──────────────────────────────────────────────
FROZEN_FILL   = '#DBEAFE'   # light blue  — frozen components
FROZEN_EDGE   = '#3B82F6'   # blue border
TRAIN_FILL    = '#D1FAE5'   # light green — trainable components
TRAIN_EDGE    = '#059669'   # green border
BLEND_FILL    = '#FEF3C7'   # light amber — blending mechanism
BLEND_EDGE    = '#D97706'   # amber border
INPUT_FILL    = '#F3F4F6'   # light gray  — input/output
INPUT_EDGE    = '#6B7280'   # gray border
ARROW_COLOR   = '#374151'   # dark gray arrows
LABEL_COLOR   = '#1F2937'   # near-black text
DIM_COLOR     = '#6B7280'   # dimension annotations
HIGHLIGHT     = '#DC2626'   # red for key callouts


def draw_box(ax, cx, cy, w, h, label, sublabel=None, fill=TRAIN_FILL,
             edge=TRAIN_EDGE, fontsize=9, sublabel_size=7.5, bold=False,
             linestyle='-', linewidth=1.2):
    """Draw a rounded box centered at (cx, cy) with label."""
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=fill, edgecolor=edge,
        linewidth=linewidth, linestyle=linestyle,
        zorder=3,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + 0.13, label, ha='center', va='center',
                fontsize=fontsize, color=LABEL_COLOR,
                fontweight='bold' if bold else 'normal', zorder=4)
        ax.text(cx, cy - 0.16, sublabel, ha='center', va='center',
                fontsize=sublabel_size, color=DIM_COLOR,
                fontstyle='italic', zorder=4)
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fontsize, color=LABEL_COLOR,
                fontweight='bold' if bold else 'normal', zorder=4)
    return box


def draw_arrow(ax, x0, y0, x1, y1, color=ARROW_COLOR, lw=1.3,
               connectionstyle='arc3,rad=0', style='-|>'):
    """Draw an arrow from (x0,y0) to (x1,y1)."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        connectionstyle=connectionstyle,
        arrowstyle=style,
        color=color, linewidth=lw,
        mutation_scale=14, zorder=2,
    )
    ax.add_patch(arrow)
    return arrow


def draw_tensor_label(ax, x, y, text, fontsize=7.5):
    """Draw a small tensor-shape annotation."""
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=DIM_COLOR,
            fontstyle='italic', zorder=4,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#D1D5DB', linewidth=0.5, alpha=0.9))


def main():
    fig, ax = plt.subplots(figsize=(11.5, 13.5))
    ax.set_xlim(-5.8, 5.8)
    ax.set_ylim(-0.5, 14.5)
    ax.set_aspect('equal')
    ax.axis('off')

    BW = 3.0     # box width
    BH = 0.7     # box height
    LEFT = -3.0  # left column center (routing)
    RIGHT = 3.0  # right column center (reasoning)

    # ── Column headers ──
    ax.text(LEFT, 14.15, 'ROUTING PATH', ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=FROZEN_EDGE,
            bbox=dict(boxstyle='round,pad=0.22', facecolor=FROZEN_FILL,
                      edgecolor=FROZEN_EDGE, linewidth=0.8))
    ax.text(RIGHT, 14.15, 'REASONING PATH', ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=TRAIN_EDGE,
            bbox=dict(boxstyle='round,pad=0.22', facecolor=TRAIN_FILL,
                      edgecolor=TRAIN_EDGE, linewidth=0.8))

    # ── Row 0: Input (y=13.3) ──
    draw_box(ax, 0, 13.3, 3.2, BH, 'Input Question', fill=INPUT_FILL,
             edge=INPUT_EDGE, fontsize=10.5, bold=True)

    # ── Row 1: Frozen Encoder (y=12.1) ──
    draw_box(ax, 0, 12.1, 3.2, 0.8, 'Frozen GPT-2', '124M params',
             fill=FROZEN_FILL, edge=FROZEN_EDGE, fontsize=10.5, bold=True,
             linestyle='--', linewidth=1.5)
    draw_arrow(ax, 0, 13.3 - BH/2, 0, 12.1 + 0.8/2)

    # ── s0 tensor label (y=11.2) ──
    draw_tensor_label(ax, 0, 11.25, 's0  [B, 256]', fontsize=9)
    draw_arrow(ax, 0, 12.1 - 0.8/2, 0, 11.25 + 0.25)

    # ── Split arrows from s0 to both columns ──
    draw_arrow(ax, -0.5, 11.25 - 0.25, LEFT, 10.25 + BH/2,
               connectionstyle='arc3,rad=0.15')
    draw_arrow(ax, 0.5, 11.25 - 0.25, RIGHT, 10.25 + BH/2,
               connectionstyle='arc3,rad=-0.15')

    # ════════════════════════════════════════════════════════════
    # LEFT COLUMN: ARTI Routing Path
    # ════════════════════════════════════════════════════════════

    # Row 3: ARTI Adapter (y=10.25)
    draw_box(ax, LEFT, 10.25, BW, BH, 'ARTI Adapter', 'Linear 256\u2192384  (98K)',
             fill=TRAIN_FILL, edge=TRAIN_EDGE)
    draw_arrow(ax, LEFT, 10.25 - BH/2, LEFT, 9.05 + 0.8/2)

    # Row 4: ARTI (y=9.05)
    draw_box(ax, LEFT, 9.05, BW, 0.8, 'ARTI (frozen)', '84.2% accuracy \u00b7 20K params',
             fill=FROZEN_FILL, edge=FROZEN_EDGE, fontsize=10, bold=True,
             linestyle='--', linewidth=1.5)

    # type_probs label (y=8.1)
    draw_tensor_label(ax, LEFT, 8.1, 'type_probs  [B, 10]', fontsize=8.5)
    draw_arrow(ax, LEFT, 9.05 - 0.8/2, LEFT, 8.1 + 0.25)

    # Row 6: Router (y=7.15)
    draw_box(ax, LEFT, 7.15, BW, BH, 'Router', 'Linear(10, 1) + Sigmoid  (11 params)',
             fill=TRAIN_FILL, edge=TRAIN_EDGE)
    draw_arrow(ax, LEFT, 8.1 - 0.25, LEFT, 7.15 + BH/2)

    # alpha label (y=6.2)
    draw_tensor_label(ax, LEFT, 6.2, '\u03b1  [B, 1]', fontsize=9)
    draw_arrow(ax, LEFT, 7.15 - BH/2, LEFT, 6.2 + 0.25)

    # Alpha annotation — positioned to the left, no overlap risk
    ax.text(LEFT - 2.0, 5.6, '\u03b1 \u2248 0.48\n(learned, fixed)',
            ha='center', va='center',
            fontsize=8.5, color=HIGHLIGHT, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEE2E2',
                      edgecolor=HIGHLIGHT, linewidth=0.6, alpha=0.9),
            zorder=5)
    draw_arrow(ax, LEFT - 1.3, 5.75, LEFT - 0.5, 6.2,
               color=HIGHLIGHT, lw=1.0, style='-|>')

    # ════════════════════════════════════════════════════════════
    # RIGHT COLUMN: CO-FRN Reasoning Path
    # ════════════════════════════════════════════════════════════

    # Row 3: Factorize (y=10.25)
    draw_box(ax, RIGHT, 10.25, BW, BH, 'MI Factorization', '256D \u2192 128D structural',
             fill=TRAIN_FILL, edge=TRAIN_EDGE)
    draw_arrow(ax, RIGHT, 10.25 - BH/2, RIGHT, 9.1 + BH/2)

    # Row 4: Manifold (y=9.1)
    draw_box(ax, RIGHT, 9.1, BW, BH, 'Continuous Operator', '16 anchors on ~10D manifold',
             fill=TRAIN_FILL, edge=TRAIN_EDGE)
    draw_arrow(ax, RIGHT, 9.1 - BH/2, RIGHT, 7.95 + BH/2)

    # Row 5: Transform (y=7.95)
    draw_box(ax, RIGHT, 7.95, BW, BH, 'HilbertTree Transform', 'beam=2, depth=3',
             fill=TRAIN_FILL, edge=TRAIN_EDGE)

    # transformed label (y=7.0)
    draw_tensor_label(ax, RIGHT, 7.0, 'transformed  [B, 256]', fontsize=8.5)
    draw_arrow(ax, RIGHT, 7.95 - BH/2, RIGHT, 7.0 + 0.25)

    # ════════════════════════════════════════════════════════════
    # DUAL SCORING HEADS (centered at bottom)
    # ════════════════════════════════════════════════════════════

    COS_X = 0.8    # cosine scoring head
    DIR_X = 4.0    # direct scoring head
    SCORE_Y = 5.4  # scoring head y-position

    # Split arrows from transformed to dual scoring heads
    draw_arrow(ax, RIGHT - 0.5, 7.0 - 0.25, COS_X + 0.2, SCORE_Y + 0.8/2,
               connectionstyle='arc3,rad=0.15')
    draw_arrow(ax, RIGHT + 0.5, 7.0 - 0.25, DIR_X - 0.2, SCORE_Y + 0.8/2,
               connectionstyle='arc3,rad=-0.15')

    # Cosine scoring head
    draw_box(ax, COS_X, SCORE_Y, 2.6, 0.8, 'Cosine Scoring',
             'cos(h, ans_enc) / \u03c4',
             fill=TRAIN_FILL, edge=TRAIN_EDGE, fontsize=9)

    # Direct classification head
    draw_box(ax, DIR_X, SCORE_Y, 2.6, 0.8, 'Direct MLP',
             '256\u2192128\u2192GELU\u21924  (33K)',
             fill=TRAIN_FILL, edge=TRAIN_EDGE, fontsize=9)

    # Answer embeddings annotation (small, to the left of cosine box)
    ax.text(COS_X - 2.1, SCORE_Y + 0.15, 'answer',
            ha='center', va='center', fontsize=7, color=FROZEN_EDGE,
            fontstyle='italic')
    ax.text(COS_X - 2.1, SCORE_Y - 0.15, 'embeddings',
            ha='center', va='center', fontsize=7, color=FROZEN_EDGE,
            fontstyle='italic')
    # Small frozen box around it
    ans_box = FancyBboxPatch(
        (COS_X - 2.7, SCORE_Y - 0.35), 1.2, 0.7,
        boxstyle="round,pad=0.05",
        facecolor=FROZEN_FILL, edgecolor=FROZEN_EDGE,
        linewidth=0.6, linestyle='--', alpha=0.7, zorder=2,
    )
    ax.add_patch(ans_box)
    draw_arrow(ax, COS_X - 1.5, SCORE_Y, COS_X - 2.6/2, SCORE_Y,
               color=FROZEN_EDGE, lw=0.9, style='-|>')

    # Score labels
    draw_tensor_label(ax, COS_X, SCORE_Y - 0.7, 'cos_scores', fontsize=7.5)
    draw_tensor_label(ax, DIR_X, SCORE_Y - 0.7, 'dir_scores', fontsize=7.5)
    draw_arrow(ax, COS_X, SCORE_Y - 0.8/2, COS_X, SCORE_Y - 0.7 + 0.2)
    draw_arrow(ax, DIR_X, SCORE_Y - 0.8/2, DIR_X, SCORE_Y - 0.7 + 0.2)

    # ════════════════════════════════════════════════════════════
    # BLEND BOX
    # ════════════════════════════════════════════════════════════

    BLEND_X = (COS_X + DIR_X) / 2   # = 2.4
    BLEND_Y = 3.3

    draw_box(ax, BLEND_X, BLEND_Y, 4.0, 0.9,
             '\u03b1 \u00b7 cosine + (1\u2212\u03b1) \u00b7 direct', None,
             fill=BLEND_FILL, edge=BLEND_EDGE, fontsize=11, bold=True,
             linewidth=2.0)

    # Arrows from scores to blend
    draw_arrow(ax, COS_X, SCORE_Y - 0.7 - 0.2, BLEND_X - 0.8, BLEND_Y + 0.9/2,
               connectionstyle='arc3,rad=0.12')
    draw_arrow(ax, DIR_X, SCORE_Y - 0.7 - 0.2, BLEND_X + 0.8, BLEND_Y + 0.9/2,
               connectionstyle='arc3,rad=-0.12')

    # Arrow from alpha to blend — long curve from left column to blend center
    draw_arrow(ax, LEFT, 6.2 - 0.25, BLEND_X - 1.8, BLEND_Y + 0.9/2,
               color=BLEND_EDGE, lw=2.0,
               connectionstyle='arc3,rad=0.4', style='-|>')
    # Label on the alpha arrow
    ax.text(-1.5, 4.6, '\u03b1', fontsize=11, fontweight='bold',
            color=BLEND_EDGE, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.12', facecolor=BLEND_FILL,
                      edgecolor=BLEND_EDGE, linewidth=0.6, alpha=0.9),
            zorder=5)

    # ── Output (y=2.1) ──
    draw_box(ax, BLEND_X, 2.1, 3.2, BH, 'final_scores  [B, C]',
             fill=INPUT_FILL, edge=INPUT_EDGE, fontsize=10.5, bold=True)
    draw_arrow(ax, BLEND_X, BLEND_Y - 0.9/2, BLEND_X, 2.1 + BH/2)

    # ════════════════════════════════════════════════════════════
    # DASHED REGION BOXES
    # ════════════════════════════════════════════════════════════

    # Left routing region
    region_left = FancyBboxPatch(
        (LEFT - BW/2 - 0.3, 5.75), BW + 0.6, 5.2,
        boxstyle="round,pad=0.15",
        facecolor='none', edgecolor=FROZEN_EDGE,
        linewidth=0.6, linestyle=':', alpha=0.35, zorder=1,
    )
    ax.add_patch(region_left)

    # Right reasoning region
    region_right = FancyBboxPatch(
        (RIGHT - BW/2 - 0.3, 6.6), BW + 0.6, 4.35,
        boxstyle="round,pad=0.15",
        facecolor='none', edgecolor=TRAIN_EDGE,
        linewidth=0.6, linestyle=':', alpha=0.35, zorder=1,
    )
    ax.add_patch(region_right)

    # Scoring region (both heads)
    region_scoring = FancyBboxPatch(
        (COS_X - 2.6/2 - 0.3, SCORE_Y - 1.1), DIR_X - COS_X + 2.6 + 0.6, 2.2,
        boxstyle="round,pad=0.15",
        facecolor='none', edgecolor=BLEND_EDGE,
        linewidth=0.5, linestyle=':', alpha=0.3, zorder=1,
    )
    ax.add_patch(region_scoring)
    ax.text(DIR_X + 1.6, SCORE_Y, 'Dual\nScoring', ha='center', va='center',
            fontsize=8, color=BLEND_EDGE, fontstyle='italic', alpha=0.7)

    # ════════════════════════════════════════════════════════════
    # LEGEND
    # ════════════════════════════════════════════════════════════
    legend_y = 0.8
    legend_items = [
        (FROZEN_FILL, FROZEN_EDGE, '--', 'Frozen (not trained)'),
        (TRAIN_FILL, TRAIN_EDGE, '-', 'Trainable (~132K new params)'),
        (BLEND_FILL, BLEND_EDGE, '-', 'Blend mechanism'),
    ]
    for i, (fill, edge, ls, label) in enumerate(legend_items):
        lx = -3.5 + i * 3.5
        box = FancyBboxPatch(
            (lx, legend_y - 0.2), 0.55, 0.4,
            boxstyle="round,pad=0.04",
            facecolor=fill, edgecolor=edge,
            linewidth=1.0, linestyle=ls, zorder=3,
        )
        ax.add_patch(box)
        ax.text(lx + 0.78, legend_y, label, va='center', fontsize=8.5,
                color=LABEL_COLOR)

    # ── Title ──
    ax.text(0, 14.9, 'ARTIRoutedScorer Architecture',
            ha='center', va='center', fontsize=15, fontweight='bold',
            color=LABEL_COLOR)

    # ── Caption ──
    ax.text(BLEND_X, 0.05,
            'Frozen ARTI (84.2%) produces type probabilities that route between '
            'cosine scoring (content-rich answers)\n'
            'and direct classification (generic labels). '
            'The ~50/50 blend is self-balancing via complementary variance profiles.',
            ha='center', va='center', fontsize=8.5, color=DIM_COLOR,
            fontstyle='italic', linespacing=1.5)

    # ── Save ──
    plt.savefig(OUTDIR / "figure_architecture_arti_routed.png", dpi=300,
                bbox_inches='tight', pad_inches=0.2)
    plt.savefig(OUTDIR / "figure_architecture_arti_routed.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Architecture diagram saved to {OUTDIR}/figure_architecture_arti_routed.png/.pdf")


if __name__ == '__main__':
    main()
