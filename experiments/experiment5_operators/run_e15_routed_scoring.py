#!/usr/bin/env python3
"""
E15 ARTI-Routed Scoring Heads — Closing the Detection-Application Gap.

E13 gradient diagnostic confirmed cosine scoring with generic answer labels
(Yes/No, True/False) is the root cause of constant predictions on non-math
benchmarks. Answer embeddings are nearly identical → score range ~0.07 →
argmax always picks the same class.

Solution: Use ARTI (84.2% type detection, frozen) as a router to soft-blend
cosine scoring (works for content-rich answers like GSM8K) and direct
classification heads (works for generic labels like StrategyQA).

4 Conditions:
  C0: E12 C3 baseline (cosine only, 2 benchmarks)
  C1: ARTI-routed (blended cosine + direct, 3 benchmarks)
  C2: Direct-only (direct classification, 3 benchmarks)
  C3: Cosine + 3bench (cosine only, 3 benchmarks — data ablation)

Success Criteria:
  SC1: C1 StrategyQA > 53% (> +3pp vs C0's ~49%)
  SC2: C1 FOLIO > 35% zero-shot (above C0's ~34%)
  SC3: C1 GSM8K within 2pp of C0 (no regression)
  SC4: Router alpha task-adaptive (mean alpha differs by >0.1 between GSM8K/SQA)

Usage:
    cd paper13_The_geometry_of_Machine_Reasoning
    python experiments/experiment5_operators/run_e15_routed_scoring.py
"""

import sys
import time
import json
import logging
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
    precompute_embeddings_st, MixedDomainDataset, BenchmarkDataset,
    collate_benchmark, collate_mixed_domain,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "e15_routed_scoring"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 7]

# ARTI v1 pretrained checkpoint
ARTI_CHECKPOINT = str(
    BASE_DIR / "results" / "arti_ensemble" / "arti_v1.pt"
)

BENCHMARKS = [
    ('gsm8k',         'test',       4),
    ('arc_challenge',  'test',       4),
    ('strategyqa',    'train',      2),
    ('folio',         'validation', 3),
]

# E12 C3 baselines for comparison
E12_C3_BASELINES = {
    'gsm8k': {'mean': 0.496, 'std': 0.028},
    'arc_challenge': {'mean': 0.300, 'std': 0.007},
    'strategyqa': {'mean': 0.488, 'std': 0.021},
    'folio': {'mean': 0.342, 'std': 0.016},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Conditions
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    'C0_baseline': {
        'description': 'E12 C3 baseline: cosine only, 2 benchmarks (GSM8K + ARC)',
        'use_routed_scoring': False,
        'use_direct_only': False,
        'train_benchmarks': ['gsm8k', 'arc_challenge'],
    },
    'C1_arti_routed': {
        'description': 'ARTI-routed: blended cosine + direct, 3 benchmarks',
        'use_routed_scoring': True,
        'use_direct_only': False,
        'train_benchmarks': ['gsm8k', 'arc_challenge', 'strategyqa'],
    },
    'C2_direct_only': {
        'description': 'Direct classification only, 3 benchmarks',
        'use_routed_scoring': False,
        'use_direct_only': True,
        'train_benchmarks': ['gsm8k', 'arc_challenge', 'strategyqa'],
    },
    'C3_cosine_3bench': {
        'description': 'Cosine only, 3 benchmarks (data ablation)',
        'use_routed_scoring': False,
        'use_direct_only': False,
        'train_benchmarks': ['gsm8k', 'arc_challenge', 'strategyqa'],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# E12 C3 patch: remove sqrt(d) scaling (applied to all conditions)
# ═══════════════════════════════════════════════════════════════════════════════

def patch_manifold_no_sqrt_d(manifold):
    """Monkey-patch compute_attention_weights to remove sqrt(d) scaling."""
    original_fn = manifold.compute_attention_weights

    def patched_compute_attention_weights(structural):
        query = manifold.manifold_proj(structural)
        logits = query @ manifold.anchors.T / manifold.tau
        weights = F.softmax(logits, dim=-1)
        return weights

    manifold.compute_attention_weights = patched_compute_attention_weights
    return manifold


# ═══════════════════════════════════════════════════════════════════════════════
# Direct-only scoring patch (C2)
# ═══════════════════════════════════════════════════════════════════════════════

class DirectOnlyScorer(nn.Module):
    """
    Direct classification scorer that ignores answer embeddings entirely.
    For C2 ablation: tests if learned classification alone is sufficient.
    """

    def __init__(self, hidden_dim: int = 256, max_choices: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_choices),
        )

    def forward(self, transformed, answer_encodings, valid_choices=None):
        scores = self.head(transformed)  # [B, max_choices]
        n_answers = answer_encodings.shape[1]
        scores = scores[:, :n_answers]
        if valid_choices is not None:
            valid = valid_choices[:, :n_answers]
            scores = scores.masked_fill(~valid, -1e9)
        return scores


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
        tau_floor=0.1,
        # E15 routing
        use_routed_scoring=condition['use_routed_scoring'],
        arti_checkpoint=ARTI_CHECKPOINT if condition['use_routed_scoring'] else '',
        arti_encoder_dim=384,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = COFRN(config)

    # Apply E12 C3 fix to all conditions
    patch_manifold_no_sqrt_d(model.factorization.manifold)

    return model


def build_training_data(condition: dict, seed: int):
    """Load and prepare training data for a condition."""
    datasets = {}
    for bench_name in condition['train_benchmarks']:
        ds = precompute_embeddings_st('all-MiniLM-L6-v2', bench_name, 'train', seed=seed)
        datasets[bench_name] = ds

    mixed_ds = MixedDomainDataset(datasets)
    n = len(mixed_ds)

    val_size = min(500, n // 5)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_ds = BenchmarkDataset(
        question_embeddings=mixed_ds.question_embeddings[train_idx],
        answer_embeddings=mixed_ds.answer_embeddings[train_idx],
        labels=mixed_ds.labels[train_idx],
        domain='mixed',
    )
    val_ds = BenchmarkDataset(
        question_embeddings=mixed_ds.question_embeddings[val_idx],
        answer_embeddings=mixed_ds.answer_embeddings[val_idx],
        labels=mixed_ds.labels[val_idx],
        domain='mixed',
    )

    # Attach valid_choices to the BenchmarkDatasets
    train_ds.valid_choices = mixed_ds.valid_choices[train_idx]
    val_ds.valid_choices = mixed_ds.valid_choices[val_idx]

    return train_ds, val_ds


def collate_with_valid_choices(batch):
    """Collate that preserves valid_choices if present in dataset items."""
    result = collate_benchmark(batch)
    # Check if valid_choices is in the batch items
    if 'valid_choices' in batch[0]:
        result['valid_choices'] = torch.stack([b['valid_choices'] for b in batch])
    return result


class ValidChoicesDataset(BenchmarkDataset):
    """BenchmarkDataset that includes valid_choices in __getitem__."""

    def __init__(self, question_embeddings, answer_embeddings, labels,
                 valid_choices, domain='mixed'):
        super().__init__(question_embeddings, answer_embeddings, labels, domain)
        self.valid_choices = valid_choices

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['valid_choices'] = self.valid_choices[idx]
        return item


def build_training_datasets(condition: dict, seed: int):
    """Load and prepare training data with valid_choices support."""
    datasets = {}
    for bench_name in condition['train_benchmarks']:
        ds = precompute_embeddings_st('all-MiniLM-L6-v2', bench_name, 'train', seed=seed)
        datasets[bench_name] = ds

    mixed_ds = MixedDomainDataset(datasets)
    n = len(mixed_ds)

    val_size = min(500, n // 5)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_ds = ValidChoicesDataset(
        question_embeddings=mixed_ds.question_embeddings[train_idx],
        answer_embeddings=mixed_ds.answer_embeddings[train_idx],
        labels=mixed_ds.labels[train_idx],
        valid_choices=mixed_ds.valid_choices[train_idx],
        domain='mixed',
    )
    val_ds = ValidChoicesDataset(
        question_embeddings=mixed_ds.question_embeddings[val_idx],
        answer_embeddings=mixed_ds.answer_embeddings[val_idx],
        labels=mixed_ds.labels[val_idx],
        valid_choices=mixed_ds.valid_choices[val_idx],
        domain='mixed',
    )

    return train_ds, val_ds


def train_condition(cond_name: str, condition: dict, seed: int) -> COFRN:
    """Train COFRN for a specific condition and seed."""
    saved_path = RESULTS_DIR / f"cofrn_{cond_name}_seed{seed}.pt"

    # Check for cached model
    if saved_path.exists():
        logger.info(f"[{cond_name}|seed={seed}] Loading cached model")
        model = build_model(condition, seed)
        state = torch.load(str(saved_path), weights_only=True, map_location=DEVICE)
        model.load_state_dict(state)
        # Re-apply patches (not saved in state_dict)
        patch_manifold_no_sqrt_d(model.factorization.manifold)
        model.to(DEVICE).eval()
        return model

    logger.info(f"[{cond_name}|seed={seed}] Training: {condition['description']}")

    model = build_model(condition, seed)

    # C2: Replace scoring with direct-only (patch after model build)
    direct_only_scorer = None
    if condition['use_direct_only']:
        direct_only_scorer = DirectOnlyScorer(hidden_dim=256, max_choices=4)
        # Monkey-patch the reasoning engine's forward_direct and forward_multistep
        original_direct = model.reasoning.forward_direct
        original_multistep = model.reasoning.forward_multistep

        def patched_forward_direct(transformed, answer_encodings, **kwargs):
            vc = kwargs.get('valid_choices')
            scores = direct_only_scorer(transformed, answer_encodings, vc)
            return {
                'scores': scores,
                'coherence_loss': torch.tensor(0.0, device=transformed.device),
            }

        def patched_forward_multistep(transformed, evidence, operator_fn,
                                       structural, answer_encodings, depth=None, **kwargs):
            # Still run tree for coherence, but score with direct head
            tree_out = model.reasoning.tree(
                initial_state=transformed, evidence=evidence,
                operator_fn=operator_fn, structural=structural, depth=depth,
            )
            h_weights = F.softmax(tree_out['hypothesis_scores'], dim=-1)
            best_hyp = (h_weights.unsqueeze(-1) * tree_out['hypothesis_states']).sum(dim=1)
            vc = kwargs.get('valid_choices')
            scores = direct_only_scorer(best_hyp, answer_encodings, vc)
            return {
                'scores': scores,
                'coherence_loss': tree_out['coherence_loss'],
                'hypothesis_states': tree_out['hypothesis_states'],
                'hypothesis_scores': tree_out['hypothesis_scores'],
            }

        model.reasoning.forward_direct = patched_forward_direct
        model.reasoning.forward_multistep = patched_forward_multistep
        # Register direct_only_scorer's params with optimizer
        model.direct_only_scorer = direct_only_scorer

    model.to(DEVICE)

    # Build training data
    train_ds, val_ds = build_training_datasets(condition, seed)

    logger.info(f"[{cond_name}|seed={seed}] Train: {len(train_ds)}, Val: {len(val_ds)}")
    logger.info(f"[{cond_name}|seed={seed}] Trainable params: {model.trainable_params:,}")
    if direct_only_scorer is not None:
        n_direct = sum(p.numel() for p in direct_only_scorer.parameters())
        logger.info(f"[{cond_name}|seed={seed}] + DirectOnlyScorer: {n_direct:,}")

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
        collate_fn=collate_with_valid_choices,
    )

    best_acc = train_result.get('best_val_acc', 0.0)
    logger.info(f"[{cond_name}|seed={seed}] Best val acc: {best_acc:.2%}")

    # Restore original forward methods before saving (for C2)
    if condition['use_direct_only']:
        model.reasoning.forward_direct = original_direct
        model.reasoning.forward_multistep = original_multistep

    torch.save(model.state_dict(), str(saved_path))

    # Re-apply patches for eval
    patch_manifold_no_sqrt_d(model.factorization.manifold)
    if condition['use_direct_only']:
        model.reasoning.forward_direct = patched_forward_direct
        model.reasoning.forward_multistep = patched_forward_multistep

    model.to(DEVICE).eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full(model: COFRN, seed: int, condition: dict) -> dict:
    """Evaluate on full test sets with anchor + router diagnostics."""
    results = {}

    for bench, split, n_choices in BENCHMARKS:
        ds = precompute_embeddings_st('all-MiniLM-L6-v2', bench, split, seed=seed)
        n = len(ds)

        q = ds.question_embeddings.to(DEVICE)
        a = ds.answer_embeddings.to(DEVICE)
        lab = ds.labels.to(DEVICE)

        # Build valid_choices mask for this benchmark
        max_choices = 4  # padded to max
        if a.shape[1] < max_choices:
            pad = torch.zeros(n, max_choices - a.shape[1], a.shape[2], device=DEVICE)
            a_padded = torch.cat([a, pad], dim=1)
            vc = torch.zeros(n, max_choices, dtype=torch.bool, device=DEVICE)
            vc[:, :a.shape[1]] = True
        else:
            a_padded = a
            vc = torch.ones(n, max_choices, dtype=torch.bool, device=DEVICE)

        correct = 0
        total = 0
        all_entropies = []
        all_max_weights = []
        all_alphas = []

        for start in range(0, n, 64):
            end = min(start + 64, n)
            out = model(
                embeddings=q[start:end],
                answer_embeddings=a_padded[start:end],
                labels=lab[start:end],
                valid_choices=vc[start:end],
            )

            # Predictions: mask invalid choices before argmax
            scores = out['scores']
            if scores.shape[1] > n_choices:
                scores = scores[:, :n_choices]
            pred = scores.argmax(-1)
            correct += (pred == lab[start:end]).sum().item()
            total += end - start

            # Anchor diagnostics
            aw = out['anchor_weights']
            ent = -(aw * torch.log(aw + 1e-10)).sum(dim=-1)
            all_entropies.append(ent.cpu())
            all_max_weights.append(aw.max(dim=-1).values.cpu())

            # Router alpha diagnostic (E15)
            if 'router_alpha' in out:
                all_alphas.append(out['router_alpha'].cpu())

        acc = correct / total
        random_baseline = 1.0 / n_choices
        max_ent = np.log(16)

        cat_ent = torch.cat(all_entropies)
        cat_max_w = torch.cat(all_max_weights)

        bench_result = {
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

        # Router alpha stats
        if all_alphas:
            cat_alpha = torch.cat(all_alphas)
            bench_result['router_alpha_mean'] = cat_alpha.mean().item()
            bench_result['router_alpha_std'] = cat_alpha.std().item()

        results[bench] = bench_result

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("  E15 ARTI-ROUTED SCORING HEADS")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Conditions: {list(CONDITIONS.keys())}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  ARTI checkpoint: {ARTI_CHECKPOINT}")
    logger.info("=" * 70)

    # Verify ARTI checkpoint exists
    if not Path(ARTI_CHECKPOINT).exists():
        logger.error(f"ARTI checkpoint not found: {ARTI_CHECKPOINT}")
        logger.error("Run run_arti_ensemble.py first to train ARTI v1.")
        return

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
            seed_results = evaluate_full(model, seed, condition)
            cond_seed_results[str(seed)] = seed_results

            for bench, _, n_choices in BENCHMARKS:
                r = seed_results[bench]
                alpha_str = ""
                if 'router_alpha_mean' in r:
                    alpha_str = f"  alpha={r['router_alpha_mean']:.3f}"
                logger.info(
                    f"  [{cond_name}|s{seed}] {bench:15s}: {r['accuracy']:.1%}  "
                    f"ent_ratio={r['anchor_entropy_ratio']:.4f}  "
                    f"max_w={r['max_anchor_weight_mean']:.3f}"
                    f"{alpha_str}"
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[cond_name] = cond_seed_results

    # ─── Aggregate ────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  AGGREGATE RESULTS (3-seed mean ± std)")
    logger.info(f"{'='*70}")

    aggregate = {}
    for cond_name in CONDITIONS:
        aggregate[cond_name] = {}
        for bench, _, n_choices in BENCHMARKS:
            accs = [all_results[cond_name][str(s)][bench]['accuracy'] for s in SEEDS]
            ent_ratios = [all_results[cond_name][str(s)][bench]['anchor_entropy_ratio'] for s in SEEDS]
            max_ws = [all_results[cond_name][str(s)][bench]['max_anchor_weight_mean'] for s in SEEDS]

            entry = {
                'mean_acc': float(np.mean(accs)),
                'std_acc': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                'mean_entropy_ratio': float(np.mean(ent_ratios)),
                'mean_max_weight': float(np.mean(max_ws)),
                'per_seed_acc': {str(s): a for s, a in zip(SEEDS, accs)},
            }

            # Router alpha aggregate
            alphas = []
            for s in SEEDS:
                r = all_results[cond_name][str(s)][bench]
                if 'router_alpha_mean' in r:
                    alphas.append(r['router_alpha_mean'])
            if alphas:
                entry['mean_router_alpha'] = float(np.mean(alphas))
                entry['std_router_alpha'] = float(np.std(alphas, ddof=1)) if len(alphas) > 1 else 0.0

            aggregate[cond_name][bench] = entry

    # ─── Print comparison table ───────────────────────────────────────
    for bench, _, n_choices in BENCHMARKS:
        logger.info(f"\n  {bench} (random={1/n_choices:.1%}):")
        header = f"  {'Condition':25s} {'Acc':>12s} {'Ent.Ratio':>12s} {'Max W':>8s} {'Alpha':>8s} {'Δ vs E12':>8s}"
        logger.info(header)
        logger.info(f"  {'-'*78}")

        e12_acc = E12_C3_BASELINES[bench]['mean']
        for cond_name in CONDITIONS:
            a = aggregate[cond_name][bench]
            delta = a['mean_acc'] - e12_acc
            alpha_str = f"{a.get('mean_router_alpha', 0):>7.3f}" if 'mean_router_alpha' in a else "    N/A"
            logger.info(
                f"  {cond_name:25s} "
                f"{a['mean_acc']:>5.1%}±{a['std_acc']:.1%} "
                f"{a['mean_entropy_ratio']:>11.4f} "
                f"{a['mean_max_weight']:>7.3f} "
                f"{alpha_str} "
                f"{delta:>+7.1%}"
            )

    # ─── Success criteria check ───────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  SUCCESS CRITERIA CHECK")
    logger.info(f"{'='*70}")

    c1 = aggregate.get('C1_arti_routed', {})
    c0 = aggregate.get('C0_baseline', {})

    # SC1: C1 StrategyQA > 53%
    sqa_acc = c1.get('strategyqa', {}).get('mean_acc', 0)
    sc1 = sqa_acc > 0.53
    logger.info(f"  SC1: C1 StrategyQA > 53%: {sqa_acc:.1%} → {'PASS' if sc1 else 'FAIL'}")

    # SC2: C1 FOLIO > 35% (zero-shot)
    folio_acc = c1.get('folio', {}).get('mean_acc', 0)
    sc2 = folio_acc > 0.35
    logger.info(f"  SC2: C1 FOLIO > 35% (zero-shot): {folio_acc:.1%} → {'PASS' if sc2 else 'FAIL'}")

    # SC3: C1 GSM8K within 2pp of C0
    c1_gsm = c1.get('gsm8k', {}).get('mean_acc', 0)
    c0_gsm = c0.get('gsm8k', {}).get('mean_acc', 0)
    delta_gsm = c1_gsm - c0_gsm
    sc3 = delta_gsm > -0.02
    logger.info(f"  SC3: C1 GSM8K within 2pp of C0: Δ={delta_gsm:+.1%} → {'PASS' if sc3 else 'FAIL'}")

    # SC4: Router alpha task-adaptive
    c1_alpha_gsm = c1.get('gsm8k', {}).get('mean_router_alpha', 0.5)
    c1_alpha_sqa = c1.get('strategyqa', {}).get('mean_router_alpha', 0.5)
    alpha_diff = abs(c1_alpha_gsm - c1_alpha_sqa)
    sc4 = alpha_diff > 0.1
    logger.info(
        f"  SC4: Router alpha adaptive (|Δ|>0.1): "
        f"GSM8K={c1_alpha_gsm:.3f}, SQA={c1_alpha_sqa:.3f}, |Δ|={alpha_diff:.3f} "
        f"→ {'PASS' if sc4 else 'FAIL'}"
    )

    n_pass = sum([sc1, sc2, sc3, sc4])
    logger.info(f"\n  Summary: {n_pass}/4 success criteria met")

    # ─── Cross-condition summary ──────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  CROSS-CONDITION SUMMARY (mean across benchmarks)")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Condition':25s} {'Mean Acc':>10s} {'Mean Ent.R':>10s} {'Mean Alpha':>10s}")
    logger.info(f"  {'-'*58}")

    for cond_name in CONDITIONS:
        accs = [aggregate[cond_name][b]['mean_acc'] for b, _, _ in BENCHMARKS]
        ent_ratios = [aggregate[cond_name][b]['mean_entropy_ratio'] for b, _, _ in BENCHMARKS]
        alphas = [aggregate[cond_name][b].get('mean_router_alpha', float('nan'))
                  for b, _, _ in BENCHMARKS]
        alpha_str = f"{np.nanmean(alphas):>9.3f}" if any(not np.isnan(a) for a in alphas) else "      N/A"
        logger.info(
            f"  {cond_name:25s} "
            f"{np.mean(accs):>9.1%} "
            f"{np.mean(ent_ratios):>9.4f} "
            f"{alpha_str}"
        )

    # ─── Save ─────────────────────────────────────────────────────────
    output = {
        'experiment': 'E15_arti_routed_scoring',
        'description': 'E15: ARTI-routed scoring heads to close detection-application gap',
        'conditions': {k: v['description'] for k, v in CONDITIONS.items()},
        'seeds': SEEDS,
        'device': DEVICE,
        'arti_checkpoint': ARTI_CHECKPOINT,
        'e12_c3_baselines': E12_C3_BASELINES,
        'per_condition_per_seed': all_results,
        'aggregate': aggregate,
        'success_criteria': {
            'SC1_sqa_above_53': sc1,
            'SC2_folio_above_35': sc2,
            'SC3_gsm8k_no_regression': sc3,
            'SC4_alpha_adaptive': sc4,
            'n_pass': n_pass,
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': time.time() - t0,
    }

    out_path = RESULTS_DIR / "e15_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
