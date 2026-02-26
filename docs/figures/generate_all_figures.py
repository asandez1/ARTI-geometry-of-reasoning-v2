#!/usr/bin/env python3
"""Generate all 5 figures for Paper 13 v2."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
RESULTS = Path("/home/qstar/AI/Research_papers/paper13_The_geometry_of_Machine_Reasoning/experiments/experiment5_operators/results")
OUTDIR = Path("/home/qstar/AI/Research_papers/paper13_The_geometry_of_Machine_Reasoning/docs/figures")

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    'blue': '#2563EB',
    'green': '#059669',
    'red': '#DC2626',
    'orange': '#EA580C',
    'purple': '#7C3AED',
    'gray': '#6B7280',
    'teal': '#0D9488',
    'pink': '#DB2777',
    'amber': '#D97706',
    'indigo': '#4F46E5',
}


# ============================================================
# FIGURE 1: ARTI Ensemble — Confusion matrix + per-type bars
# ============================================================
def figure1():
    with open(RESULTS / "arti_ensemble/ensemble_results.json") as f:
        data = json.load(f)

    types = ['Deduc', 'Induc', 'Abduc', 'Analog',
             'Counter', 'Conserv', 'Decomp',
             'PhysCause', 'BehvCause', 'SysCause']
    type_labels = ['Deduct.', 'Induct.', 'Abduct.', 'Analogy',
                   'Counter.', 'Conserv.', 'Decomp.',
                   'PhysCause', 'BehvCause', 'SysCause']

    # Per-type ensemble accuracy
    ensemble_acc = []
    v1_acc = []
    v2_acc = []
    oracle_acc = []
    for t in types:
        td = data['per_type'][t]
        ensemble_acc.append(td['ensemble_acc'] * 100)
        v1_acc.append(td['v1_acc'] * 100)
        v2_acc.append(td['v2_acc'] * 100)
        oracle_acc.append(td['oracle_acc'] * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.3, 1]})

    # (a) Per-type accuracy comparison (v1, v2, ensemble, oracle)
    x = np.arange(len(types))
    w = 0.2
    bars_v1 = ax1.bar(x - 1.5*w, v1_acc, w, label='v1-retrained (83.5%)', color=COLORS['blue'], alpha=0.7)
    bars_v2 = ax1.bar(x - 0.5*w, v2_acc, w, label='v2-retrained (82.9%)', color=COLORS['teal'], alpha=0.7)
    bars_ens = ax1.bar(x + 0.5*w, ensemble_acc, w, label='Ensemble (84.2%)', color=COLORS['green'], edgecolor='black', linewidth=0.5)
    bars_ora = ax1.bar(x + 1.5*w, oracle_acc, w, label='Oracle (89.0%)', color=COLORS['gray'], alpha=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(type_labels, rotation=35, ha='right')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 108)
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=0.8, label='Chance (10%)')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax1.set_title('(a) Per-Type Accuracy: ARTI v1, v2, Ensemble, Oracle')
    ax1.grid(axis='y', alpha=0.3)

    # Add accuracy labels on ensemble bars
    for bar, acc in zip(bars_ens, ensemble_acc):
        if acc > 15:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{acc:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # (b) ARTI version progression
    versions = ['v1\n(8-type)', 'v2\n(8-type)', 'v3\n(10-type)', 'Traj.\n(10-type)',
                'v1-ret.\n(10-type)', 'v2-ret.\n(10-type)', 'Ensemble']
    accuracies = [54.0, 70.8, 71.5, 77.9, 83.5, 82.9, 84.2]
    bar_colors = [COLORS['gray']]*4 + [COLORS['blue'], COLORS['teal'], COLORS['green']]

    bars = ax2.bar(range(len(versions)), accuracies, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(versions)))
    ax2.set_xticklabels(versions, fontsize=8)
    ax2.set_ylabel('Overall Accuracy (%)')
    ax2.set_ylim(0, 95)
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=0.8, label='Chance (10%)')
    ax2.set_title('(b) ARTI Evolution: 54% → 84.2%')
    ax2.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTDIR / "figure1_arti_ensemble.png")
    plt.close()
    print("Figure 1 saved.")


# ============================================================
# FIGURE 2: Pre-factorization signal retention (3.5x gap)
# ============================================================
def figure2():
    features = ['s0\n(pre-factorization)\n256D', 'Structural\n(post-MI)\n128D',
                'Manifold\ncoordinates\n10D', 'Random\nbaseline']
    accuracies = [76.7, 21.9, 18.8, 16.7]
    ratios = ['1.0x', '0.29x', '0.25x', '0.22x']
    bar_colors = [COLORS['green'], COLORS['orange'], COLORS['amber'], COLORS['gray']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]})

    # (a) Overall comparison
    bars = ax1.bar(range(len(features)), accuracies, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.6)
    ax1.set_xticks(range(len(features)))
    ax1.set_xticklabels(features, fontsize=9)
    ax1.set_ylabel('Type Classification Accuracy (%)')
    ax1.set_ylim(0, 90)
    ax1.set_title('(a) Type Detection Accuracy by Feature Source')
    ax1.grid(axis='y', alpha=0.3)

    for bar, acc, ratio in zip(bars, accuracies, ratios):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{acc}%\n({ratio})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Arrow showing 3.5x gap — offset to avoid label overlap
    ax1.annotate('', xy=(0.1, 76.7), xytext=(0.9, 28),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(0.5, 46, '3.5x', ha='center', fontsize=14, fontweight='bold', color='red')

    # (b) Per-type breakdown on s0
    per_type = {
        'Analogy': 97.9, 'Induction': 88.6, 'Counterfactual': 83.6,
        'Conservation': 77.8, 'Cause-Effect': 66.9, 'Deduction': 55.5
    }
    types_sorted = list(per_type.keys())
    accs_sorted = list(per_type.values())
    colors_per = [COLORS['green'] if a > 75 else COLORS['amber'] if a > 60 else COLORS['orange'] for a in accs_sorted]

    bars2 = ax2.barh(range(len(types_sorted)), accs_sorted, color=colors_per, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(types_sorted)))
    ax2.set_yticklabels(types_sorted)
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_xlim(0, 108)
    ax2.set_title('(b) Per-Type Accuracy on s0 (256D)')
    ax2.axvline(x=16.7, color='red', linestyle='--', alpha=0.5, linewidth=0.8, label='Random')
    ax2.grid(axis='x', alpha=0.3)

    for bar, acc in zip(bars2, accs_sorted):
        ax2.text(acc + 1.5, bar.get_y() + bar.get_height()/2,
                f'{acc}%', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTDIR / "figure2_prefactorization_gap.png")
    plt.close()
    print("Figure 2 saved.")


# ============================================================
# FIGURE 3: E15 ARTI-routed scoring results
# ============================================================
def figure3():
    with open(RESULTS / "e15_routed_scoring/e15_results.json") as f:
        data = json.load(f)

    benchmarks = ['GSM8K', 'ARC', 'StrategyQA', 'FOLIO']
    bench_keys = ['gsm8k', 'arc_challenge', 'strategyqa', 'folio']
    conditions = ['C0_baseline', 'C1_arti_routed',
                  'C2_direct_only', 'C3_cosine_3bench']
    cond_labels = ['C0: Cosine (2-bench)', 'C1: ARTI-Routed (3-bench)',
                   'C2: Direct-only (3-bench)', 'C3: Cosine (3-bench)']
    cond_colors = [COLORS['gray'], COLORS['green'], COLORS['red'], COLORS['blue']]
    chance_levels = [25.0, 25.0, 50.0, 33.3]

    # 2-row layout: (a) spans top full width; (b) and (c) share bottom row
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])   # (a) full top row
    ax2 = fig.add_subplot(gs[1, 0])  # (b) bottom-left
    ax3 = fig.add_subplot(gs[1, 1])  # (c) bottom-right

    # (a) 4-condition comparison across benchmarks
    x = np.arange(len(benchmarks))
    w = 0.18

    for i, (cond, label, color) in enumerate(zip(conditions, cond_labels, cond_colors)):
        means = []
        stds = []
        for bk in bench_keys:
            agg = data['aggregate'][cond][bk]
            means.append(agg['mean_acc'] * 100)
            stds.append(agg['std_acc'] * 100)
        bars = ax.bar(x + (i - 1.5)*w, means, w, yerr=stds, label=label,
                      color=color, edgecolor='black', linewidth=0.3, capsize=3,
                      alpha=0.85 if i == 1 else 0.6)

    # Chance lines per benchmark
    for j, chance in enumerate(chance_levels):
        ax.plot([j - 2*w, j + 2*w], [chance, chance], 'r--', alpha=0.4, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 95)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    ax.set_title('(a) E15: Four Conditions Across Benchmarks', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    # Highlight SQA breakthrough
    ax.annotate('+32.6pp', xy=(2 - 0.5*w, 81), fontsize=11, fontweight='bold',
               color=COLORS['green'], ha='center',
               xytext=(2.8, 89), arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.5))

    # (b) Router alpha per benchmark
    alphas_c1 = []
    alpha_stds_c1 = []
    for bk in bench_keys:
        seeds = data['per_condition_per_seed']['C1_arti_routed']
        a_vals = [seeds[str(s)][bk]['router_alpha_mean'] for s in [42, 123, 7]]
        alphas_c1.append(np.mean(a_vals))
        alpha_stds_c1.append(np.std(a_vals))

    bars2 = ax2.bar(range(len(benchmarks)), alphas_c1, yerr=alpha_stds_c1,
                    color=[COLORS['blue'], COLORS['purple'], COLORS['orange'], COLORS['teal']],
                    edgecolor='black', linewidth=0.5, capsize=4, width=0.55)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax2.set_xticks(range(len(benchmarks)))
    ax2.set_xticklabels(benchmarks, fontsize=10)
    ax2.set_ylabel('Router Alpha (α)', fontsize=10)
    ax2.set_ylim(0.35, 0.60)
    ax2.set_title('(b) Router α ≈ 0.48 (not task-adaptive)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # Add mean alpha label on each bar
    for i, (a, s) in enumerate(zip(alphas_c1, alpha_stds_c1)):
        ax2.text(i, a + s + 0.008, f'{a:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (c) Why the fixed blend works — score variance (schematic)
    bench_short = ['GSM8K', 'ARC', 'SQA', 'FOLIO']
    cosine_variance = [2.0, 0.5, 0.07, 0.09]  # approximate score ranges from E13
    direct_variance = [0.8, 0.6, 0.8, 0.7]    # approximate

    x3 = np.arange(len(bench_short))
    w3 = 0.3
    ax3.bar(x3 - w3/2, cosine_variance, w3, label='Cosine score range', color=COLORS['blue'], alpha=0.7,
            edgecolor='black', linewidth=0.3)
    ax3.bar(x3 + w3/2, direct_variance, w3, label='Direct score range', color=COLORS['orange'], alpha=0.7,
            edgecolor='black', linewidth=0.3)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(bench_short, fontsize=10)
    ax3.set_ylabel('Score Range (approx.)', fontsize=10)
    ax3.set_title('(c) Complementary Variance Profiles (schematic)', fontsize=12)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    # Annotations — positioned to avoid bar overlap
    ax3.annotate('Cosine\ndominates', xy=(0, 2.0), fontsize=9, ha='center',
                xytext=(0.6, 2.3), color=COLORS['blue'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=1, alpha=0.6))
    ax3.annotate('Direct\ndominates', xy=(2, 0.8), fontsize=9, ha='center',
                xytext=(2.6, 1.4), color=COLORS['orange'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1, alpha=0.6))

    plt.savefig(OUTDIR / "figure3_e15_routed_scoring.png")
    plt.close()
    print("Figure 3 saved.")


# ============================================================
# FIGURE 4: E16→E17→E18 Falsification chain
# ============================================================
def figure4():
    fig, ax = plt.subplots(figsize=(12, 5))

    experiments = ['E15\n(frozen)', 'E16\n(unfrozen)', 'E17\n(two-encoder)', 'E18\n(freeze answers)']
    sqa_means = [81.0, 54.0, 54.0, 54.0]
    sqa_stds = [3.8, 0.0, 0.0, 0.0]
    gsm8k_means = [48.1, 49.8, 49.2, 42.6]
    gsm8k_stds = [2.7, 1.8, 1.5, 0.8]

    x = np.arange(len(experiments))
    w = 0.3

    # StrategyQA bars
    bars_sqa = ax.bar(x - w/2, sqa_means, w, yerr=sqa_stds,
                      label='StrategyQA', color=[COLORS['green'], COLORS['red'], COLORS['red'], COLORS['red']],
                      edgecolor='black', linewidth=0.5, capsize=4)
    # GSM8K bars
    bars_gsm = ax.bar(x + w/2, gsm8k_means, w, yerr=gsm8k_stds,
                      label='GSM8K', color=[COLORS['blue'], COLORS['blue'], COLORS['blue'], COLORS['orange']],
                      edgecolor='black', linewidth=0.5, capsize=4, alpha=0.7)

    # Chance lines
    ax.axhline(y=50.0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axhline(y=25.0, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.text(3.6, 51, 'SQA chance (50%)', fontsize=7, color='red', alpha=0.5)
    ax.text(3.6, 26, 'GSM8K chance (25%)', fontsize=7, color='blue', alpha=0.5)

    # Annotations for the falsification chain
    ax.annotate('', xy=(1, 55), xytext=(0, 79),
               arrowprops=dict(arrowstyle='->', color='red', lw=2.5, linestyle='-'))
    ax.text(0.5, 70, '−27pp\ncollapse', ha='center', fontsize=9, color='red', fontweight='bold')

    ax.annotate('ARTI shift\nhypothesis\nfalsified', xy=(2, 55), xytext=(2, 68),
               fontsize=8, ha='center', color=COLORS['purple'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.2))

    ax.annotate('Answer-grad\nhypothesis\nfalsified\n+ GSM8K −5.5pp', xy=(3, 44), xytext=(3, 68),
               fontsize=8, ha='center', color=COLORS['orange'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.2))

    # Labels — offset E15 SQA label to avoid error bar cap
    for bars, means, stds in [(bars_sqa, sqa_means, sqa_stds), (bars_gsm, gsm8k_means, gsm8k_stds)]:
        for bar, m, s in zip(bars, means, stds):
            offset = max(s, 1.5) + 1.5
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                   f'{m:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=10)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 95)
    ax.set_title('E16→E17→E18: Three-Hypothesis Falsification Chain\n'
                 'The frozen encoder is structurally necessary', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Bottom text
    ax.text(0.5, -0.12,
            'SQA 54.0% with zero std across all 9 seeds (3 experiments × 3 seeds) — identical constant-prediction failure',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic', color=COLORS['gray'])

    plt.tight_layout()
    plt.savefig(OUTDIR / "figure4_falsification_chain.png")
    plt.close()
    print("Figure 4 saved.")


# ============================================================
# FIGURE 5: Entropy trajectory 0.99996 → 0.937
# ============================================================
def figure5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.3, 1]})

    # (a) Entropy trajectory
    stages = ['Paper 12\nbaseline', 'E10\ncosine fix', 'E11\nentropy fix', 'E12\n−sqrt(d)']
    entropy_vals = [0.99996, 0.9995, 0.996, 0.937]
    max_weights = [0.0625, 0.0625, 0.08, 0.16]

    color_map = [COLORS['red'], COLORS['orange'], COLORS['amber'], COLORS['green']]

    ax1.plot(range(len(stages)), entropy_vals, 'o-', color=COLORS['blue'], linewidth=2.5,
             markersize=12, zorder=5)
    # Label positions: nudge first two apart to prevent collision with each other and title
    # Point 0 (0.99996): label below-left; Point 1 (0.9995): label below-right
    # Point 2 (0.996): label above; Point 3 (0.937): label above
    label_y = [0.99996 - 0.012, 0.9995 - 0.012, 0.996 + 0.006, 0.937 + 0.006]
    label_ha = ['center', 'center', 'center', 'center']
    label_va = ['top', 'top', 'bottom', 'bottom']
    label_xnudge = [-0.15, 0.15, 0, 0]
    for i, (s, e) in enumerate(zip(stages, entropy_vals)):
        ax1.scatter(i, e, color=color_map[i], s=150, zorder=6, edgecolors='black', linewidth=0.5)
        lbl = f'{e:.5f}' if e > 0.999 else (f'{e:.4f}' if e > 0.99 else f'{e:.3f}')
        ax1.text(i + label_xnudge[i], label_y[i], lbl,
                ha=label_ha[i], va=label_va[i], fontsize=9, fontweight='bold')

    ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='Target (<0.95)')
    ax1.fill_between(range(len(stages)), 0.90, 0.95, alpha=0.1, color='green')
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages, fontsize=9)
    ax1.set_ylabel('Anchor Entropy Ratio')
    ax1.set_ylim(0.90, 1.005)
    ax1.set_title('(a) Entropy Trajectory: Three Independent Fixes')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Cause annotations
    causes = ['MLP scorer\nignored tree', 'Constant entropy\nregularization', 'sqrt(d)=4\nsuppressed\ndifferentiation']
    for i, cause in enumerate(zip(causes)):
        pass  # annotations already visible via labels

    # (b) Per-benchmark entropy at E12 C3
    benchmarks = ['GSM8K', 'ARC', 'StrategyQA', 'FOLIO']
    e12_entropy = [0.961, 0.921, 0.934, 0.932]
    e11_entropy = [0.997, 0.996, 0.996, 0.996]
    pass_fail = ['FAIL', 'PASS', 'PASS', 'PASS']
    colors_pf = [COLORS['red'], COLORS['green'], COLORS['green'], COLORS['green']]

    x = np.arange(len(benchmarks))
    w = 0.3

    bars_e11 = ax2.bar(x - w/2, e11_entropy, w, label='E11 (before)', color=COLORS['gray'], alpha=0.5,
                       edgecolor='black', linewidth=0.3)
    bars_e12 = ax2.bar(x + w/2, e12_entropy, w, label='E12 C3 (after)', color=colors_pf,
                       edgecolor='black', linewidth=0.5)

    ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='Target (<0.95)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks, fontsize=10)
    ax2.set_ylabel('Anchor Entropy Ratio')
    ax2.set_ylim(0.88, 1.005)
    ax2.set_title('(b) Per-Benchmark Entropy: E11 vs E12')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    for bar, e, pf in zip(bars_e12, e12_entropy, pass_fail):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{e:.3f}\n{pf}', ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=COLORS['green'] if pf == 'PASS' else COLORS['red'])

    plt.tight_layout()
    plt.savefig(OUTDIR / "figure5_entropy_trajectory.png")
    plt.close()
    print("Figure 5 saved.")


if __name__ == '__main__':
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print(f"\nAll figures saved to {OUTDIR}/")
