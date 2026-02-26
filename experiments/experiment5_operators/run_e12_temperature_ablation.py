#!/usr/bin/env python3
"""
E12 Temperature Ablation — Breaking the Softmax Bottleneck.

E11 proved uniform attention was a training artifact (entropy 0.99996 → 0.996)
but the softmax temperature/scaling regime keeps logits too small for sharp
attention. E12 ablates four conditions to isolate the remaining bottleneck:

Condition 1 (E11 baseline): tau_floor=0.1, with sqrt(d) scaling
Condition 2 (lower floor):  tau_floor=0.01
Condition 3 (no sqrt(d)):   remove sqrt(manifold_dim) from attention logits
Condition 4 (contrastive):  add anchor separation loss that pushes anchors apart

All conditions use the E11 fixes (entropy anneal, xavier proj, ortho anchors,
0.05 U/V init). Same 3 seeds, curriculum, full-benchmark eval.

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_e12_temperature_ablation.py
"""

import sys
import time
import json
import copy
import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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

RESULTS_DIR = BASE_DIR / "results" / "e12_temperature_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 7]

BENCHMARKS = [
    ('gsm8k',         'test',       4),
    ('arc_challenge',  'test',       4),
    ('strategyqa',    'train',      2),
    ('folio',         'validation', 3),
]

# E10 and E11 baselines for comparison
E10_BASELINES = {
    'gsm8k': {'mean': 0.474, 'std': 0.033},
    'arc_challenge': {'mean': 0.281, 'std': 0.003},
    'strategyqa': {'mean': 0.492, 'std': 0.023},
    'folio': {'mean': 0.330, 'std': 0.010},
}
E11_BASELINES = {
    'gsm8k': {'mean': 0.485, 'std': 0.035},
    'arc_challenge': {'mean': 0.290, 'std': 0.005},
    'strategyqa': {'mean': 0.486, 'std': 0.019},
    'folio': {'mean': 0.350, 'std': 0.005},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Condition-specific patches
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    'C1_e11_baseline': {
        'description': 'E11 baseline (tau_floor=0.1, sqrt(d) scaling)',
        'tau_floor': 0.1,
        'remove_sqrt_d': False,
        'contrastive_loss': False,
    },
    'C2_low_tau': {
        'description': 'Lower temperature floor (tau_floor=0.01)',
        'tau_floor': 0.01,
        'remove_sqrt_d': False,
        'contrastive_loss': False,
    },
    'C3_no_sqrt_d': {
        'description': 'Remove sqrt(d) scaling from attention',
        'tau_floor': 0.1,
        'remove_sqrt_d': True,
        'contrastive_loss': False,
    },
    'C4_contrastive': {
        'description': 'Add contrastive anchor separation loss',
        'tau_floor': 0.1,
        'remove_sqrt_d': False,
        'contrastive_loss': True,
    },
}


def patch_manifold_no_sqrt_d(manifold):
    """Monkey-patch compute_attention_weights to remove sqrt(d) scaling."""
    original_fn = manifold.compute_attention_weights

    def patched_compute_attention_weights(structural):
        query = manifold.manifold_proj(structural)  # [B, manifold_dim]
        # No sqrt(d) scaling — just divide by tau
        logits = query @ manifold.anchors.T / manifold.tau  # [B, n_anchors]
        weights = F.softmax(logits, dim=-1)
        return weights

    manifold.compute_attention_weights = patched_compute_attention_weights
    return manifold


def compute_contrastive_anchor_loss(manifold) -> torch.Tensor:
    """
    Push anchors apart when they are too similar.

    Loss = mean(max(0, cos(a_i, a_j) - margin)) for all i != j.
    Margin = 0.0 means push all pairs to be orthogonal or anti-correlated.
    """
    anchors_norm = F.normalize(manifold.anchors, dim=-1)  # [n_anchors, manifold_dim]
    sim = anchors_norm @ anchors_norm.T  # [n_anchors, n_anchors]
    # Mask diagonal
    mask = ~torch.eye(manifold.n_anchors, dtype=bool, device=sim.device)
    off_diag = sim[mask]
    # Hinge loss: penalize positive cosine similarity (push toward orthogonal)
    margin = 0.0
    loss = F.relu(off_diag - margin).mean()
    return loss


class ContrastiveTrainerWrapper:
    """Wraps the forward step to add contrastive anchor loss."""

    def __init__(self, model, lambda_contrastive=0.1):
        self.model = model
        self.lambda_contrastive = lambda_contrastive

    def add_contrastive_loss(self, output):
        """Add contrastive loss to output dict."""
        manifold = self.model.factorization.manifold
        c_loss = compute_contrastive_anchor_loss(manifold)
        output['contrastive_loss'] = c_loss
        output['total_loss'] = output['total_loss'] + self.lambda_contrastive * c_loss
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(condition: dict, seed: int) -> COFRN:
    """Build COFRN with condition-specific config."""
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
        tau_floor=condition['tau_floor'],
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = COFRN(config)

    # Apply condition-specific patches
    if condition['remove_sqrt_d']:
        patch_manifold_no_sqrt_d(model.factorization.manifold)

    return model


def train_condition(cond_name: str, condition: dict, seed: int) -> COFRN:
    """Train COFRN for a specific condition and seed."""
    saved_path = RESULTS_DIR / f"cofrn_{cond_name}_seed{seed}.pt"

    # Check for cached model
    if saved_path.exists():
        logger.info(f"[{cond_name}|seed={seed}] Loading cached model")
        model = build_model(condition, seed)
        model.load_state_dict(
            torch.load(str(saved_path), weights_only=True, map_location=DEVICE)
        )
        # Re-apply patches (not saved in state_dict)
        if condition['remove_sqrt_d']:
            patch_manifold_no_sqrt_d(model.factorization.manifold)
        model.to(DEVICE).eval()
        return model

    logger.info(f"[{cond_name}|seed={seed}] Training: {condition['description']}")

    # Load training data
    gsm8k_ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'gsm8k', 'train', seed=seed)
    arc_ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'arc_challenge', 'train', seed=seed)

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

    model = build_model(condition, seed).to(DEVICE)

    # For contrastive condition, wrap the model's forward to add the loss
    if condition['contrastive_loss']:
        wrapper = ContrastiveTrainerWrapper(model, lambda_contrastive=0.1)
        original_forward = model.forward

        def wrapped_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            return wrapper.add_contrastive_loss(output)

        model.forward = wrapped_forward

    logger.info(f"[{cond_name}|seed={seed}] Trainable params: {model.trainable_params:,}")

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
        model_name=f'{cond_name}_s{seed}',
    )

    best_acc = train_result.get('best_val_acc', 0.0)
    logger.info(f"[{cond_name}|seed={seed}] Best val acc: {best_acc:.2%}")

    # Restore original forward before saving (unwrap contrastive)
    if condition['contrastive_loss']:
        model.forward = original_forward

    torch.save(model.state_dict(), str(saved_path))

    # Re-apply patches for eval
    if condition['remove_sqrt_d']:
        patch_manifold_no_sqrt_d(model.factorization.manifold)

    model.to(DEVICE).eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full(model: COFRN, seed: int) -> dict:
    """Evaluate on full test sets with anchor diagnostics."""
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

            aw = out['anchor_weights']
            ent = -(aw * torch.log(aw + 1e-10)).sum(dim=-1)
            all_entropies.append(ent.cpu())
            all_max_weights.append(aw.max(dim=-1).values.cpu())

        acc = correct / total
        random_baseline = 1.0 / n_choices
        max_ent = np.log(16)

        cat_ent = torch.cat(all_entropies)
        cat_max_w = torch.cat(all_max_weights)

        results[bench] = {
            'n_examples': total,
            'n_choices': n_choices,
            'random_baseline': random_baseline,
            'accuracy': acc,
            'lift_over_random': acc - random_baseline,
            'anchor_entropy_mean': cat_ent.mean().item(),
            'anchor_entropy_ratio': cat_ent.mean().item() / max_ent,
            'max_anchor_weight_mean': cat_max_w.mean().item(),
            'max_anchor_weight_std': cat_max_w.std().item(),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("  E12 TEMPERATURE ABLATION — BREAKING THE SOFTMAX BOTTLENECK")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Conditions: {list(CONDITIONS.keys())}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 70)

    all_results = {}

    for cond_name, condition in CONDITIONS.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"  CONDITION: {cond_name}")
        logger.info(f"  {condition['description']}")
        logger.info(f"{'='*70}")

        cond_seed_results = {}

        for seed in SEEDS:
            logger.info(f"\n--- {cond_name} | seed={seed} ---")
            model = train_condition(cond_name, condition, seed)
            seed_results = evaluate_full(model, seed)
            cond_seed_results[str(seed)] = seed_results

            for bench, _, n_choices in BENCHMARKS:
                r = seed_results[bench]
                logger.info(
                    f"  [{cond_name}|s{seed}] {bench:15s}: {r['accuracy']:.1%}  "
                    f"ent_ratio={r['anchor_entropy_ratio']:.4f}  "
                    f"max_w={r['max_anchor_weight_mean']:.3f}"
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[cond_name] = cond_seed_results

    # ─── Aggregate ────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  AGGREGATE RESULTS")
    logger.info(f"{'='*70}")

    aggregate = {}
    for cond_name in CONDITIONS:
        aggregate[cond_name] = {}
        for bench, _, n_choices in BENCHMARKS:
            accs = [all_results[cond_name][str(s)][bench]['accuracy'] for s in SEEDS]
            ent_ratios = [all_results[cond_name][str(s)][bench]['anchor_entropy_ratio'] for s in SEEDS]
            max_ws = [all_results[cond_name][str(s)][bench]['max_anchor_weight_mean'] for s in SEEDS]

            aggregate[cond_name][bench] = {
                'mean_acc': float(np.mean(accs)),
                'std_acc': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                'mean_entropy_ratio': float(np.mean(ent_ratios)),
                'std_entropy_ratio': float(np.std(ent_ratios, ddof=1)) if len(ent_ratios) > 1 else 0.0,
                'mean_max_weight': float(np.mean(max_ws)),
                'per_seed_acc': {str(s): a for s, a in zip(SEEDS, accs)},
            }

    # ─── Print comparison table ───────────────────────────────────────
    for bench, _, n_choices in BENCHMARKS:
        logger.info(f"\n  {bench} (random={1/n_choices:.1%}):")
        logger.info(f"  {'Condition':25s} {'Acc':>12s} {'Ent.Ratio':>12s} {'Max W':>8s} {'Δ vs E10':>8s}")
        logger.info(f"  {'-'*68}")

        e10_acc = E10_BASELINES[bench]['mean']
        for cond_name in CONDITIONS:
            a = aggregate[cond_name][bench]
            delta = a['mean_acc'] - e10_acc
            logger.info(
                f"  {cond_name:25s} "
                f"{a['mean_acc']:>5.1%}±{a['std_acc']:.1%} "
                f"{a['mean_entropy_ratio']:>11.4f} "
                f"{a['mean_max_weight']:>7.3f} "
                f"{delta:>+7.1%}"
            )

    # ─── Cross-condition summary ──────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  CROSS-CONDITION SUMMARY (mean across benchmarks)")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Condition':25s} {'Mean Ent.Ratio':>14s} {'Mean Max W':>10s} {'Mean Acc':>10s}")
    logger.info(f"  {'-'*62}")

    for cond_name in CONDITIONS:
        ent_ratios = [aggregate[cond_name][b]['mean_entropy_ratio'] for b, _, _ in BENCHMARKS]
        max_ws = [aggregate[cond_name][b]['mean_max_weight'] for b, _, _ in BENCHMARKS]
        accs = [aggregate[cond_name][b]['mean_acc'] for b, _, _ in BENCHMARKS]
        logger.info(
            f"  {cond_name:25s} "
            f"{np.mean(ent_ratios):>13.4f} "
            f"{np.mean(max_ws):>9.3f} "
            f"{np.mean(accs):>9.1%}"
        )

    # ─── Verification ─────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  VERIFICATION: Entropy ratio < 0.95 in any condition/benchmark?")
    logger.info(f"{'='*70}")

    any_pass = False
    for cond_name in CONDITIONS:
        for bench, _, _ in BENCHMARKS:
            ent = aggregate[cond_name][bench]['mean_entropy_ratio']
            if ent < 0.95:
                any_pass = True
                logger.info(f"  PASS: {cond_name} / {bench}: entropy_ratio = {ent:.4f}")

    if not any_pass:
        logger.info("  No condition achieved entropy_ratio < 0.95")
        best_cond = min(CONDITIONS.keys(),
                       key=lambda c: np.mean([aggregate[c][b]['mean_entropy_ratio'] for b, _, _ in BENCHMARKS]))
        best_ent = np.mean([aggregate[best_cond][b]['mean_entropy_ratio'] for b, _, _ in BENCHMARKS])
        logger.info(f"  Best condition: {best_cond} (mean entropy_ratio = {best_ent:.4f})")

    # ─── Save ─────────────────────────────────────────────────────────
    output = {
        'experiment': 'E12_temperature_ablation',
        'description': 'E12: Temperature ablation on top of E11 fixes',
        'conditions': {k: v['description'] for k, v in CONDITIONS.items()},
        'seeds': SEEDS,
        'device': DEVICE,
        'e10_baselines': E10_BASELINES,
        'e11_baselines': E11_BASELINES,
        'per_condition_per_seed': all_results,
        'aggregate': aggregate,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': time.time() - t0,
    }

    out_path = RESULTS_DIR / "e12_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
