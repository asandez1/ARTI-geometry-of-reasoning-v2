#!/usr/bin/env python3
"""
E10 3-Seed Full-Benchmark Evaluation.

Trains CO-FRN with cosine scoring from scratch with 3 seeds (42, 123, 7),
evaluates each on all 4 benchmarks using the FULL test sets (not 100-example
slices), and reports mean ± std across seeds.

This replaces the single-seed 100-example numbers with publication-ready stats.

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_e10_3seed.py
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

RESULTS_DIR = BASE_DIR / "results" / "cosine_fix_3seed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 7]

BENCHMARKS = [
    ('gsm8k',         'test',       4),
    ('arc_challenge',  'test',       4),
    ('strategyqa',    'train',      2),
    ('folio',         'validation', 3),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_cosine_cofrn(seed: int) -> COFRN:
    """Train COFRN with use_cosine_scoring=True for a given seed."""
    saved_path = RESULTS_DIR / f"cofrn_cosine_seed{seed}.pt"

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

    logger.info(f"[seed={seed}] Training COFRN with cosine scoring...")

    # Load training data (use seed for train/val split, but always same data)
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
        model_name=f'cofrn_cosine_s{seed}',
    )

    best_acc = train_result.get('best_val_acc', 0.0)
    logger.info(f"[seed={seed}] Training complete. Best val acc: {best_acc:.2%}")

    torch.save(model.state_dict(), str(saved_path))
    logger.info(f"[seed={seed}] Saved to {saved_path}")

    model.to(DEVICE).eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Full-benchmark evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_full(model: COFRN, seed: int) -> dict:
    """Evaluate on FULL test sets for all 4 benchmarks."""
    results = {}

    for bench, split, n_choices in BENCHMARKS:
        ds = precompute_embeddings_st('all-MiniLM-L6-v2', bench, split, seed=seed)
        n = len(ds)

        q = ds.question_embeddings.to(DEVICE)
        a = ds.answer_embeddings.to(DEVICE)
        lab = ds.labels.to(DEVICE)

        correct = 0
        total = 0
        anchor_entropies = []

        with torch.no_grad():
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

                # Anchor entropy
                aw = out['anchor_weights']
                ent = -(aw * torch.log(aw + 1e-10)).sum(dim=-1)
                anchor_entropies.append(ent.cpu())

        acc = correct / total
        random_baseline = 1.0 / n_choices

        all_ent = torch.cat(anchor_entropies)
        mean_ent = all_ent.mean().item()
        max_ent = np.log(16)

        results[bench] = {
            'n_examples': total,
            'n_choices': n_choices,
            'random_baseline': random_baseline,
            'accuracy': acc,
            'lift_over_random': acc - random_baseline,
            'anchor_entropy': mean_ent,
            'anchor_entropy_ratio': mean_ent / max_ent,
        }
        logger.info(f"[seed={seed}] {bench:15s}: {acc:.1%} ({total} ex, "
                     f"random={random_baseline:.1%}, lift={acc-random_baseline:+.1%})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("  E10 3-SEED FULL-BENCHMARK EVALUATION")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 70)

    all_seed_results = {}

    for seed in SEEDS:
        logger.info(f"\n{'='*60}")
        logger.info(f"  SEED {seed}")
        logger.info(f"{'='*60}")

        model = train_cosine_cofrn(seed)
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
        n_ex = all_seed_results[str(SEEDS[0])][bench]['n_examples']

        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)  # sample std
        mean_lift = np.mean(lifts)
        std_lift = np.std(lifts, ddof=1)
        mean_ent = np.mean(ent_ratios)

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
        }

        logger.info(f"  {bench:15s}: {mean_acc:.1%} ± {std_acc:.1%}  "
                     f"(lift={mean_lift:+.1%} ± {std_lift:.1%})  "
                     f"[seeds: {', '.join(f'{a:.1%}' for a in accs)}]  "
                     f"n={n_ex}")

    # ─── Comparison with old MLP single-run numbers ───────────────────
    old_mlp = {'gsm8k': 0.49, 'arc_challenge': 0.31, 'strategyqa': 0.58, 'folio': 0.32}

    logger.info(f"\n  {'Benchmark':15s} {'Random':>7} {'MLP(1-seed)':>12} {'Cosine(3-seed)':>16} {'Delta':>8}")
    logger.info(f"  {'-'*60}")
    for bench, _, n_choices in BENCHMARKS:
        rb = 1.0 / n_choices
        old = old_mlp.get(bench, 0)
        new_m = aggregate[bench]['mean_acc']
        new_s = aggregate[bench]['std_acc']
        delta = new_m - old
        logger.info(f"  {bench:15s} {rb:>6.1%} {old:>11.1%} "
                     f"{new_m:>7.1%}±{new_s:.1%} {delta:>+7.1%}")

    # ─── Save ─────────────────────────────────────────────────────────
    output = {
        'experiment': 'E10_3seed_full_benchmark',
        'seeds': SEEDS,
        'device': DEVICE,
        'per_seed': all_seed_results,
        'aggregate': aggregate,
        'old_mlp_single_seed': old_mlp,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': time.time() - t0,
    }

    out_path = RESULTS_DIR / "e10_3seed_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
