#!/usr/bin/env python3
"""
E11 3-Seed Full-Benchmark Evaluation — Breaking Uniform Attention.

Same setup as E10 (cosine scoring, curriculum training, 3 seeds, full test sets)
but with 4 targeted fixes to the initialization and entropy schedule:

1. Trainer delegates entropy weight to model's anneal schedule (was constant 0.01)
2. Xavier-normal manifold_proj init with gain=2.0 (query magnitude ~0.76 vs ~0.025)
3. Orthogonal anchor init via QR (unit-norm, max separation vs ~0.316 magnitude)
4. Operator U/V init at 0.05 scale (~16% perturbation vs 0.006%)

Diagnostics added:
- Per-benchmark entropy ratio (target: < 0.95, was 0.9999)
- Max anchor weight distribution
- Operator perturbation magnitude
- Per-epoch entropy trajectory

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_e11_3seed.py
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from shared.model import COFRN, COFRNConfig
from shared.train_utils import TrainConfig, CurriculumPhase, curriculum_train
from shared.data_utils import (
    precompute_embeddings_st, MixedDomainDataset, BenchmarkDataset, collate_benchmark,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "e11_break_uniform"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 7]

BENCHMARKS = [
    ('gsm8k',         'test',       4),
    ('arc_challenge',  'test',       4),
    ('strategyqa',    'train',      2),
    ('folio',         'validation', 3),
]

# E10 baselines for comparison
E10_BASELINES = {
    'gsm8k': {'mean': 0.474, 'std': 0.033},
    'arc_challenge': {'mean': 0.281, 'std': 0.003},
    'strategyqa': {'mean': 0.492, 'std': 0.023},
    'folio': {'mean': 0.330, 'std': 0.010},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_operator_perturbation(model: COFRN) -> dict:
    """Measure how much the operator transforms inputs vs identity."""
    manifold = model.factorization.manifold
    hidden_dim = manifold.hidden_dim

    # Generate random unit vectors
    x = torch.randn(100, hidden_dim, device=next(model.parameters()).device)
    x = torch.nn.functional.normalize(x, dim=-1)

    # Apply each anchor's operator independently
    perturbations = []
    for i in range(manifold.n_anchors):
        U_i = manifold.U_all[i]  # [hidden_dim, rank]
        V_i = manifold.V_all[i]  # [hidden_dim, rank]
        Vx = x @ V_i             # [100, rank]
        UVx = Vx @ U_i.T         # [100, hidden_dim]
        # Perturbation magnitude relative to input
        pert = UVx.norm(dim=-1) / x.norm(dim=-1)
        perturbations.append(pert.mean().item())

    return {
        'mean_perturbation': float(np.mean(perturbations)),
        'std_perturbation': float(np.std(perturbations)),
        'max_perturbation': float(np.max(perturbations)),
        'min_perturbation': float(np.min(perturbations)),
        'per_anchor': perturbations,
    }


@torch.no_grad()
def measure_anchor_diagnostics(model: COFRN) -> dict:
    """Measure anchor separation and projection magnitude."""
    manifold = model.factorization.manifold

    # Anchor norms
    anchor_norms = manifold.anchors.norm(dim=-1)

    # Pairwise cosine similarity
    anchors_norm = torch.nn.functional.normalize(manifold.anchors, dim=-1)
    sim = anchors_norm @ anchors_norm.T
    off_diag = sim[~torch.eye(sim.size(0), dtype=bool, device=sim.device)]

    # Projection weight magnitude
    proj_weight_norm = manifold.manifold_proj.weight.norm().item()

    return {
        'anchor_norm_mean': anchor_norms.mean().item(),
        'anchor_norm_std': anchor_norms.std().item(),
        'anchor_cosine_sim_mean': off_diag.mean().item(),
        'anchor_cosine_sim_max': off_diag.max().item(),
        'proj_weight_norm': proj_weight_norm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_e11_cofrn(seed: int) -> COFRN:
    """Train COFRN with E11 fixes for a given seed."""
    saved_path = RESULTS_DIR / f"cofrn_e11_seed{seed}.pt"

    config = COFRNConfig(
        use_precomputed=True,
        encoder_input_dim=384,
        hidden_dim=256,
        struct_dim=128,
        context_dim=128,
        manifold_dim=10,
        n_anchors=16,
        rank=16,
        task_type='multi_step',
        use_cosine_scoring=True,
        entropy_schedule='anneal',
        entropy_anneal_epochs=10,
    )

    # Check for cached model
    if saved_path.exists():
        logger.info(f"[seed={seed}] Loading cached model from {saved_path}")
        model = COFRN(config)
        model.load_state_dict(
            torch.load(str(saved_path), weights_only=True, map_location=DEVICE)
        )
        model.to(DEVICE).eval()
        return model

    logger.info(f"[seed={seed}] Training COFRN with E11 fixes...")

    # Load training data
    gsm8k_ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'gsm8k', 'train', seed=seed)
    arc_ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'arc_challenge', 'train', seed=seed)
    logger.info(f"[seed={seed}] GSM8K: {len(gsm8k_ds)}, ARC: {len(arc_ds)}")

    mixed_ds = MixedDomainDataset({'gsm8k': gsm8k_ds, 'arc_challenge': arc_ds})
    n = len(mixed_ds)

    val_size = min(500, n // 5)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    train_ds = BenchmarkDataset(
        question_embeddings=mixed_ds.question_embeddings[indices[val_size:]],
        answer_embeddings=mixed_ds.answer_embeddings[indices[val_size:]],
        labels=mixed_ds.labels[indices[val_size:]],
        domain='mixed',
    )
    val_ds = BenchmarkDataset(
        question_embeddings=mixed_ds.question_embeddings[indices[:val_size]],
        answer_embeddings=mixed_ds.answer_embeddings[indices[:val_size]],
        labels=mixed_ds.labels[indices[:val_size]],
        domain='mixed',
    )
    logger.info(f"[seed={seed}] Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = COFRN(config).to(DEVICE)
    logger.info(f"[seed={seed}] Trainable params: {model.trainable_params:,}")

    # Log initial diagnostics
    init_diag = measure_anchor_diagnostics(model)
    logger.info(f"[seed={seed}] Init anchor norms: {init_diag['anchor_norm_mean']:.3f} ± {init_diag['anchor_norm_std']:.3f}")
    logger.info(f"[seed={seed}] Init anchor cosine sim: {init_diag['anchor_cosine_sim_mean']:.3f} (max={init_diag['anchor_cosine_sim_max']:.3f})")
    logger.info(f"[seed={seed}] Init proj weight norm: {init_diag['proj_weight_norm']:.3f}")

    init_pert = measure_operator_perturbation(model)
    logger.info(f"[seed={seed}] Init operator perturbation: {init_pert['mean_perturbation']:.4f} ± {init_pert['std_perturbation']:.4f}")

    base_cfg = TrainConfig(
        learning_rate=3e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        batch_size=32,
        max_epochs=30,
        patience=5,
        device=DEVICE,
        seed=seed,
    )

    phases = [
        CurriculumPhase(
            depth=1, epochs=5,
            lambda_factorize=0.01, lambda_coherence=0.0,
            learning_rate=3e-4,
        ),
        CurriculumPhase(
            depth=2, epochs=5,
            lambda_factorize=0.05, lambda_coherence=0.005,
            learning_rate=1e-4,
        ),
        CurriculumPhase(
            depth=3, epochs=10,
            lambda_factorize=0.1, lambda_coherence=0.01,
            learning_rate=5e-5,
        ),
    ]

    model, train_result = curriculum_train(
        model, train_ds, val_ds, base_cfg,
        phases=phases,
        model_name=f'cofrn_e11_s{seed}',
    )

    best_acc = train_result.get('best_val_acc', 0.0)
    logger.info(f"[seed={seed}] Training complete. Best val acc: {best_acc:.2%}")

    torch.save(model.state_dict(), str(saved_path))
    logger.info(f"[seed={seed}] Saved to {saved_path}")

    # Save training history for entropy trajectory analysis
    hist_path = RESULTS_DIR / f"train_history_seed{seed}.json"
    with open(hist_path, 'w') as f:
        json.dump(train_result, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    model.to(DEVICE).eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Full-benchmark evaluation with diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full(model: COFRN, seed: int) -> dict:
    """Evaluate on FULL test sets for all 4 benchmarks with anchor diagnostics."""
    results = {}

    for bench, split, n_choices in BENCHMARKS:
        ds = precompute_embeddings_st('all-MiniLM-L6-v2', bench, split, seed=seed)
        n = len(ds)

        q = ds.question_embeddings.to(DEVICE)
        a = ds.answer_embeddings.to(DEVICE)
        lab = ds.labels.to(DEVICE)

        correct = 0
        total = 0
        all_entropies = []
        all_max_weights = []
        all_weights = []

        for start in range(0, n, 64):
            end = min(start + 64, n)
            out = model(
                embeddings=q[start:end],
                answer_embeddings=a[start:end],
                labels=lab[start:end],
            )
            pred = out['scores'].argmax(-1)
            correct += (pred == lab[start:end]).sum().item()
            total += end - start

            # Anchor weight diagnostics
            aw = out['anchor_weights']  # [B, n_anchors]
            ent = -(aw * torch.log(aw + 1e-10)).sum(dim=-1)  # [B]
            all_entropies.append(ent.cpu())
            all_max_weights.append(aw.max(dim=-1).values.cpu())
            all_weights.append(aw.cpu())

        acc = correct / total
        random_baseline = 1.0 / n_choices
        max_ent = np.log(16)

        cat_ent = torch.cat(all_entropies)
        cat_max_w = torch.cat(all_max_weights)
        cat_weights = torch.cat(all_weights, dim=0)

        # Per-anchor utilization across all examples
        mean_weights = cat_weights.mean(dim=0).numpy()

        results[bench] = {
            'n_examples': total,
            'n_choices': n_choices,
            'random_baseline': random_baseline,
            'accuracy': acc,
            'lift_over_random': acc - random_baseline,
            # Entropy diagnostics
            'anchor_entropy_mean': cat_ent.mean().item(),
            'anchor_entropy_std': cat_ent.std().item(),
            'anchor_entropy_ratio': cat_ent.mean().item() / max_ent,
            # Max weight diagnostics
            'max_anchor_weight_mean': cat_max_w.mean().item(),
            'max_anchor_weight_std': cat_max_w.std().item(),
            # Per-anchor utilization
            'per_anchor_mean_weight': mean_weights.tolist(),
            'anchor_weight_gini': float(_gini(mean_weights)),
        }

        logger.info(
            f"[seed={seed}] {bench:15s}: {acc:.1%} ({total} ex, "
            f"random={random_baseline:.1%}, lift={acc-random_baseline:+.1%})  "
            f"entropy_ratio={results[bench]['anchor_entropy_ratio']:.4f}  "
            f"max_w={cat_max_w.mean():.3f}±{cat_max_w.std():.3f}"
        )

    return results


def _gini(weights: np.ndarray) -> float:
    """Compute Gini coefficient of anchor weight distribution. 0=equal, 1=concentrated."""
    w = np.sort(weights)
    n = len(w)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * w) / (n * np.sum(w))) - (n + 1) / n)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("  E11 3-SEED FULL-BENCHMARK: BREAKING UNIFORM ATTENTION")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("  Fixes: entropy anneal, xavier proj, orthogonal anchors, 0.05 U/V")
    logger.info("=" * 70)

    all_seed_results = {}
    all_diagnostics = {}

    for seed in SEEDS:
        logger.info(f"\n{'='*60}")
        logger.info(f"  SEED {seed}")
        logger.info(f"{'='*60}")

        model = train_e11_cofrn(seed)

        # Post-training diagnostics
        diag = {
            'anchor': measure_anchor_diagnostics(model),
            'perturbation': measure_operator_perturbation(model),
        }
        all_diagnostics[str(seed)] = diag
        logger.info(f"[seed={seed}] Post-train perturbation: "
                     f"{diag['perturbation']['mean_perturbation']:.4f}")
        logger.info(f"[seed={seed}] Post-train anchor cosine sim: "
                     f"{diag['anchor']['anchor_cosine_sim_mean']:.3f}")

        seed_results = evaluate_full(model, seed)
        all_seed_results[str(seed)] = seed_results

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─── Aggregate across seeds ───────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  AGGREGATE RESULTS (mean ± std across 3 seeds)")
    logger.info(f"{'='*70}")

    aggregate = {}
    for bench, split, n_choices in BENCHMARKS:
        accs = [all_seed_results[str(s)][bench]['accuracy'] for s in SEEDS]
        lifts = [all_seed_results[str(s)][bench]['lift_over_random'] for s in SEEDS]
        ent_ratios = [all_seed_results[str(s)][bench]['anchor_entropy_ratio'] for s in SEEDS]
        max_ws = [all_seed_results[str(s)][bench]['max_anchor_weight_mean'] for s in SEEDS]
        n_ex = all_seed_results[str(SEEDS[0])][bench]['n_examples']

        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)
        mean_lift = np.mean(lifts)
        std_lift = np.std(lifts, ddof=1)
        mean_ent = np.mean(ent_ratios)
        std_ent = np.std(ent_ratios, ddof=1)
        mean_max_w = np.mean(max_ws)

        # Delta vs E10
        e10 = E10_BASELINES.get(bench, {'mean': 0, 'std': 0})
        delta_vs_e10 = mean_acc - e10['mean']

        aggregate[bench] = {
            'n_examples': n_ex,
            'n_choices': n_choices,
            'random_baseline': 1.0 / n_choices,
            'per_seed_acc': {str(s): a for s, a in zip(SEEDS, accs)},
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_lift': mean_lift,
            'std_lift': std_lift,
            'mean_entropy_ratio': mean_ent,
            'std_entropy_ratio': std_ent,
            'mean_max_anchor_weight': mean_max_w,
            'delta_vs_e10': delta_vs_e10,
        }

        logger.info(
            f"  {bench:15s}: {mean_acc:.1%} ± {std_acc:.1%}  "
            f"(lift={mean_lift:+.1%} ± {std_lift:.1%})  "
            f"entropy_ratio={mean_ent:.4f}±{std_ent:.4f}  "
            f"max_w={mean_max_w:.3f}  "
            f"[seeds: {', '.join(f'{a:.1%}' for a in accs)}]  "
            f"n={n_ex}"
        )

    # ─── Comparison with E10 ──────────────────────────────────────────
    logger.info(f"\n  {'Benchmark':15s} {'Random':>7} {'E10(3-seed)':>14} {'E11(3-seed)':>14} {'Delta':>8} {'Ent.Ratio':>10}")
    logger.info(f"  {'-'*72}")
    for bench, _, n_choices in BENCHMARKS:
        rb = 1.0 / n_choices
        e10 = E10_BASELINES.get(bench, {'mean': 0, 'std': 0})
        e11_m = aggregate[bench]['mean_acc']
        e11_s = aggregate[bench]['std_acc']
        delta = aggregate[bench]['delta_vs_e10']
        ent = aggregate[bench]['mean_entropy_ratio']
        logger.info(
            f"  {bench:15s} {rb:>6.1%} "
            f"{e10['mean']:>6.1%}±{e10['std']:.1%} "
            f"{e11_m:>6.1%}±{e11_s:.1%} "
            f"{delta:>+7.1%} "
            f"{ent:>9.4f}"
        )

    # ─── Verification checks ─────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  VERIFICATION CHECKS")
    logger.info(f"{'='*70}")

    any_below_95 = False
    for bench, _, _ in BENCHMARKS:
        ent = aggregate[bench]['mean_entropy_ratio']
        status = "PASS" if ent < 0.95 else "FAIL"
        if ent < 0.95:
            any_below_95 = True
        logger.info(f"  {bench:15s} entropy_ratio={ent:.4f}  [{status}] (target < 0.95, E10 was ~0.9999)")

    logger.info(f"\n  Entropy broken in at least one benchmark: {'YES' if any_below_95 else 'NO'}")

    # ─── Save ─────────────────────────────────────────────────────────
    output = {
        'experiment': 'E11_break_uniform_attention',
        'description': 'E11: 4 fixes to break uniform anchor attention (entropy anneal, xavier proj, ortho anchors, 0.05 U/V)',
        'seeds': SEEDS,
        'device': DEVICE,
        'fixes': [
            'Trainer delegates entropy weight to model._get_effective_lambda_entropy()',
            'manifold_proj: xavier_normal(gain=2.0) init',
            'anchors: orthogonal QR init (unit-norm, max separation)',
            'U_all/V_all: 0.05 scale (was 0.01)',
        ],
        'e10_baselines': E10_BASELINES,
        'per_seed': all_seed_results,
        'aggregate': aggregate,
        'diagnostics': all_diagnostics,
        'verification': {
            'any_entropy_below_0.95': any_below_95,
            'per_benchmark_entropy': {
                bench: aggregate[bench]['mean_entropy_ratio']
                for bench, _, _ in BENCHMARKS
            },
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': time.time() - t0,
    }

    out_path = RESULTS_DIR / "e11_3seed_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
