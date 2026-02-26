#!/usr/bin/env python3
"""
E13 Unfrozen Encoder — Breaking the Frozen-Encoder Ceiling.

Paper 12 showed CO-FRN achieves +23pp on GSM8K but 0pp on all non-math
benchmarks. E10-E12 diagnosed and fixed the uniform-attention paradox, but
non-math accuracy stayed at chance. The remaining hypothesis: the frozen
GPT-2 encoder simply does not produce embeddings that distinguish science,
logic, or commonsense questions.

E13 tests this by selectively unfreezing the last N GPT-2 blocks so
gradients from the task loss can reshape the encoder representations.

All conditions include the three pipeline fixes:
  - E10: cosine scoring (no MLP scorer collapse)
  - E11: entropy annealing (no training artifact)
  - E12 C3: no sqrt(d) scaling (proper softmax sharpness)

Conditions (4 × 3 seeds × 4 benchmarks):
  C0: Frozen baseline (precomputed ST embeddings, reproduces E12 C3)
  C1: Unfreeze last 2 GPT-2 blocks (encoder_lr=1e-5)
  C2: Unfreeze last 6 GPT-2 blocks (encoder_lr=5e-6)
  C3: Unfreeze all 12 GPT-2 blocks (encoder_lr=2e-6)

Memory: ~2-3 GB for C3 (GPT-2 is 124M params). Fits in 24GB.
Runtime: C0 ~7min/seed, C1 ~33min/seed, C2 ~53min/seed, C3 ~83min/seed.
         Total ~9h across all 4 conditions × 3 seeds.

Usage:
    cd paper13_The_geometry_of_Machine_Reasoning
    python experiments/experiment5_operators/run_e13_unfrozen.py
"""

import sys
import time
import json
import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from shared.model import COFRN, COFRNConfig
from shared.train_utils import (
    TrainConfig, CurriculumPhase, Trainer, curriculum_train, save_results,
)
from shared.data_utils import (
    precompute_embeddings_st, MixedDomainDataset, BenchmarkDataset,
    TextBenchmarkDataset, collate_benchmark, collate_text_benchmark,
    load_text_benchmark, LOADERS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "e13_unfrozen"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 7]

BENCHMARKS = [
    ('gsm8k',         'test',       4),
    ('arc_challenge',  'test',       4),
    ('strategyqa',    'train',      2),
    ('folio',         'validation', 3),
]

# E12 C3 baselines for comparison
E12_C3_BASELINES = {
    'gsm8k':          {'mean': 0.496, 'std': 0.028},
    'arc_challenge':  {'mean': 0.300, 'std': 0.007},
    'strategyqa':     {'mean': 0.488, 'std': 0.021},
    'folio':          {'mean': 0.342, 'std': 0.016},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Conditions
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    'C0_frozen': {
        'description': 'Frozen baseline (precomputed ST embeddings, E12 C3)',
        'use_precomputed': True,
        'unfreeze_layers': 0,
        'encoder_lr': 0.0,
        'batch_size': 32,
        'gradient_accumulation_steps': 1,
    },
    'C1_unfreeze_2': {
        'description': 'Unfreeze last 2 GPT-2 blocks',
        'use_precomputed': False,
        'unfreeze_layers': 2,
        'encoder_lr': 1e-5,
        'batch_size': 16,
        'gradient_accumulation_steps': 2,
    },
    'C2_unfreeze_6': {
        'description': 'Unfreeze last 6 GPT-2 blocks',
        'use_precomputed': False,
        'unfreeze_layers': 6,
        'encoder_lr': 5e-6,
        'batch_size': 16,
        'gradient_accumulation_steps': 2,
    },
    'C3_unfreeze_all': {
        'description': 'Unfreeze all 12 GPT-2 blocks',
        'use_precomputed': False,
        'unfreeze_layers': 12,
        'encoder_lr': 2e-6,
        'batch_size': 8,
        'gradient_accumulation_steps': 4,
    },
}


def patch_manifold_no_sqrt_d(manifold):
    """Monkey-patch compute_attention_weights to remove sqrt(d) scaling (E12 C3)."""
    def patched_compute_attention_weights(structural):
        query = manifold.manifold_proj(structural)
        logits = query @ manifold.anchors.T / manifold.tau
        weights = F.softmax(logits, dim=-1)
        return weights
    manifold.compute_attention_weights = patched_compute_attention_weights
    return manifold


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_precomputed_training_data(seed: int) -> Tuple[BenchmarkDataset, BenchmarkDataset]:
    """Load precomputed ST training data (for C0 frozen baseline)."""
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
    return train_ds, val_ds


def load_text_training_data(
    tokenizer,
    seed: int,
    max_length: int = 128,
) -> Tuple[TextBenchmarkDataset, TextBenchmarkDataset]:
    """Load tokenized text training data (for C1-C3 unfrozen conditions)."""
    logger.info(f"  Loading tokenized GSM8K + ARC training data...")

    gsm8k_ds = load_text_benchmark(
        'gsm8k', 'train', tokenizer=tokenizer, max_length=max_length, seed=seed,
    )
    arc_ds = load_text_benchmark(
        'arc_challenge', 'train', tokenizer=tokenizer, max_length=max_length, seed=seed,
    )

    # Combine: concatenate along first dimension
    # Need to pad answer choices to max (GSM8K=4, ARC=4 — same)
    n_gsm = len(gsm8k_ds)
    n_arc = len(arc_ds)
    N = n_gsm + n_arc

    q_ids = torch.cat([gsm8k_ds.question_input_ids, arc_ds.question_input_ids], dim=0)
    q_mask = torch.cat([gsm8k_ds.question_attention_mask, arc_ds.question_attention_mask], dim=0)
    a_ids = torch.cat([gsm8k_ds.answer_input_ids, arc_ds.answer_input_ids], dim=0)
    a_mask = torch.cat([gsm8k_ds.answer_attention_mask, arc_ds.answer_attention_mask], dim=0)
    labels = torch.cat([gsm8k_ds.labels, arc_ds.labels], dim=0)

    # Shuffle and split
    val_size = min(500, N // 5)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    train_idx = torch.tensor(indices[val_size:], dtype=torch.long)
    val_idx = torch.tensor(indices[:val_size], dtype=torch.long)

    train_ds = TextBenchmarkDataset(
        question_input_ids=q_ids[train_idx],
        question_attention_mask=q_mask[train_idx],
        answer_input_ids=a_ids[train_idx],
        answer_attention_mask=a_mask[train_idx],
        labels=labels[train_idx],
        domain='mixed',
    )
    val_ds = TextBenchmarkDataset(
        question_input_ids=q_ids[val_idx],
        question_attention_mask=q_mask[val_idx],
        answer_input_ids=a_ids[val_idx],
        answer_attention_mask=a_mask[val_idx],
        labels=labels[val_idx],
        domain='mixed',
    )

    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(condition: dict, seed: int) -> COFRN:
    """Build COFRN with condition-specific config."""
    if condition['use_precomputed']:
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
            unfreeze_encoder_layers=0,
        )
    else:
        config = COFRNConfig(
            use_precomputed=False,
            encoder_model='gpt2',
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
            unfreeze_encoder_layers=condition['unfreeze_layers'],
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = COFRN(config)

    # Apply E12 C3 fix: remove sqrt(d) scaling
    patch_manifold_no_sqrt_d(model.factorization.manifold)

    return model


def train_condition(cond_name: str, condition: dict, seed: int) -> COFRN:
    """Train COFRN for a specific condition and seed."""
    saved_path = RESULTS_DIR / f"cofrn_{cond_name}_seed{seed}.pt"

    # Check for cached model
    if saved_path.exists():
        logger.info(f"[{cond_name}|seed={seed}] Loading cached model")
        model = build_model(condition, seed)
        state = torch.load(str(saved_path), weights_only=True, map_location=DEVICE)
        model.load_state_dict(state)
        patch_manifold_no_sqrt_d(model.factorization.manifold)
        model.to(DEVICE).eval()
        return model

    logger.info(f"[{cond_name}|seed={seed}] Training: {condition['description']}")

    model = build_model(condition, seed)

    # Log parameter counts
    total_trainable = model.trainable_params
    encoder_trainable = model.encoder.trainable_params
    downstream_trainable = total_trainable - encoder_trainable
    logger.info(
        f"  Params: total_trainable={total_trainable:,}  "
        f"encoder={encoder_trainable:,}  downstream={downstream_trainable:,}"
    )

    # Load training data
    if condition['use_precomputed']:
        train_ds, val_ds = load_precomputed_training_data(seed)
        collate_fn = collate_benchmark
    else:
        train_ds, val_ds = load_text_training_data(
            tokenizer=model.encoder.tokenizer,
            seed=seed,
            max_length=128,
        )
        collate_fn = collate_text_benchmark

    model.to(DEVICE)

    base_cfg = TrainConfig(
        learning_rate=3e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        batch_size=condition['batch_size'],
        encoder_lr=condition['encoder_lr'],
        gradient_accumulation_steps=condition['gradient_accumulation_steps'],
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

    # Build data loaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds, batch_size=base_cfg.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=base_cfg.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Use curriculum_train for frozen (same as E12), manual for unfrozen
    model, train_result = curriculum_train(
        model, train_ds, val_ds, base_cfg,
        phases=phases,
        model_name=f'{cond_name}_s{seed}',
        collate_fn=collate_fn,
    )

    best_acc = train_result.get('best_val_acc', 0.0)
    logger.info(f"[{cond_name}|seed={seed}] Best val acc: {best_acc:.2%}")

    # Verify gradient flow for unfrozen conditions
    if condition['unfreeze_layers'] > 0:
        _verify_gradient_flow(model, condition)

    # Save model
    torch.save(model.state_dict(), str(saved_path))
    logger.info(f"  Saved to {saved_path}")

    patch_manifold_no_sqrt_d(model.factorization.manifold)
    model.to(DEVICE).eval()
    return model


def _verify_gradient_flow(model: COFRN, condition: dict):
    """Verify encoder layers have non-zero grads (smoke test)."""
    if not hasattr(model.encoder, 'transformer') or model.encoder.transformer is None:
        return

    encoder_grad_params = sum(
        1 for p in model.encoder.transformer.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    encoder_trainable = sum(
        1 for p in model.encoder.transformer.parameters()
        if p.requires_grad
    )
    logger.info(
        f"  Gradient flow: {encoder_grad_params}/{encoder_trainable} "
        f"encoder params had non-zero grads (last training step)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full(model: COFRN, seed: int, use_precomputed: bool = True) -> dict:
    """Evaluate on full test sets with anchor diagnostics.

    For frozen conditions (use_precomputed=True): uses precomputed ST embeddings.
    For unfrozen conditions (use_precomputed=False): tokenizes text and passes
    input_ids through the GPT-2 encoder.
    """
    results = {}

    for bench, split, n_choices in BENCHMARKS:
        if use_precomputed:
            ds = precompute_embeddings_st('all-MiniLM-L6-v2', bench, split, seed=seed)
            n = len(ds)
            q = ds.question_embeddings.to(DEVICE)
            a = ds.answer_embeddings.to(DEVICE)
            lab = ds.labels.to(DEVICE)
        else:
            ds = load_text_benchmark(
                bench, split, tokenizer=model.encoder.tokenizer,
                max_length=128, seed=seed,
            )
            n = len(ds)
            lab = ds.labels.to(DEVICE)

        correct = 0
        total = 0
        all_entropies = []
        all_max_weights = []

        batch_size = 32 if use_precomputed else 16
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            if use_precomputed:
                out = model(
                    embeddings=q[start:end],
                    answer_embeddings=a[start:end],
                    labels=lab[start:end],
                )
            else:
                batch_items = [ds[i] for i in range(start, end)]
                batch = collate_text_benchmark(batch_items)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    answer_input_ids=batch['answer_input_ids'],
                    answer_attention_mask=batch['answer_attention_mask'],
                    labels=batch['labels'],
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
    logger.info("  E13 UNFROZEN ENCODER — BREAKING THE FROZEN-ENCODER CEILING")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Conditions: {list(CONDITIONS.keys())}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 70)

    all_results = {}

    for cond_name, condition in CONDITIONS.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"  CONDITION: {cond_name}")
        logger.info(f"  {condition['description']}")
        logger.info(f"  unfreeze_layers={condition['unfreeze_layers']}, "
                     f"encoder_lr={condition['encoder_lr']}, "
                     f"batch={condition['batch_size']}, "
                     f"accum={condition['gradient_accumulation_steps']}")
        logger.info(f"{'='*70}")

        cond_seed_results = {}

        for seed in SEEDS:
            logger.info(f"\n--- {cond_name} | seed={seed} ---")
            t_seed = time.time()

            model = train_condition(cond_name, condition, seed)
            seed_results = evaluate_full(model, seed, use_precomputed=condition['use_precomputed'])
            cond_seed_results[str(seed)] = seed_results

            for bench, _, n_choices in BENCHMARKS:
                r = seed_results[bench]
                logger.info(
                    f"  [{cond_name}|s{seed}] {bench:15s}: {r['accuracy']:.1%}  "
                    f"ent_ratio={r['anchor_entropy_ratio']:.4f}  "
                    f"max_w={r['max_anchor_weight_mean']:.3f}"
                )

            elapsed = time.time() - t_seed
            logger.info(f"  Seed {seed} elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[cond_name] = cond_seed_results

    # ─── Aggregate ────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  AGGREGATE RESULTS (3-seed means)")
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
        e12_acc = E12_C3_BASELINES[bench]['mean']
        logger.info(f"\n  {bench} (random={1/n_choices:.1%}, E12_C3={e12_acc:.1%}):")
        logger.info(
            f"  {'Condition':25s} {'Acc':>12s} {'Δ vs C0':>8s} {'Δ vs E12':>8s} "
            f"{'Ent.Ratio':>10s} {'Max W':>8s}"
        )
        logger.info(f"  {'-'*78}")

        c0_acc = aggregate['C0_frozen'][bench]['mean_acc']
        for cond_name in CONDITIONS:
            a = aggregate[cond_name][bench]
            delta_c0 = a['mean_acc'] - c0_acc
            delta_e12 = a['mean_acc'] - e12_acc
            marker = ""
            if cond_name != 'C0_frozen' and delta_c0 > 0.02:
                marker = " ✓"
            logger.info(
                f"  {cond_name:25s} "
                f"{a['mean_acc']:>5.1%}±{a['std_acc']:.1%} "
                f"{delta_c0:>+7.1%}{marker} "
                f"{delta_e12:>+7.1%} "
                f"{a['mean_entropy_ratio']:>9.4f} "
                f"{a['mean_max_weight']:>7.3f}"
            )

    # ─── Verification ─────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  VERIFICATION")
    logger.info(f"{'='*70}")

    # Check 1: C0 reproduces E12 C3
    logger.info("\n  1) C0 reproduces E12 C3 (within 2pp)?")
    for bench, _, _ in BENCHMARKS:
        c0_acc = aggregate['C0_frozen'][bench]['mean_acc']
        e12_acc = E12_C3_BASELINES[bench]['mean']
        diff = abs(c0_acc - e12_acc)
        status = "PASS" if diff < 0.02 else f"WARN (diff={diff:.1%})"
        logger.info(f"    {bench:15s}: C0={c0_acc:.1%}, E12={e12_acc:.1%} → {status}")

    # Check 2: Any unfrozen condition > C0 on non-math by >2pp
    logger.info("\n  2) Any unfrozen > C0 on non-math benchmark by >2pp?")
    non_math = ['arc_challenge', 'strategyqa', 'folio']
    any_pass = False
    for cond_name in ['C1_unfreeze_2', 'C2_unfreeze_6', 'C3_unfreeze_all']:
        for bench in non_math:
            c0 = aggregate['C0_frozen'][bench]['mean_acc']
            cx = aggregate[cond_name][bench]['mean_acc']
            delta = cx - c0
            if delta > 0.02:
                any_pass = True
                logger.info(f"    PASS: {cond_name} / {bench}: {cx:.1%} vs C0={c0:.1%} (Δ={delta:+.1%})")

    if not any_pass:
        logger.info("    No unfrozen condition beats C0 on non-math by >2pp")

        # Report best deltas
        logger.info("\n  Best deltas (unfrozen - C0):")
        for bench in non_math:
            c0 = aggregate['C0_frozen'][bench]['mean_acc']
            best_cond = max(
                ['C1_unfreeze_2', 'C2_unfreeze_6', 'C3_unfreeze_all'],
                key=lambda c: aggregate[c][bench]['mean_acc'],
            )
            best = aggregate[best_cond][bench]['mean_acc']
            logger.info(f"    {bench:15s}: {best_cond} = {best:.1%} (Δ={best-c0:+.1%})")

    # Check 3: GSM8K not degraded
    logger.info("\n  3) GSM8K not degraded (unfrozen ≥ C0 - 2pp)?")
    for cond_name in ['C1_unfreeze_2', 'C2_unfreeze_6', 'C3_unfreeze_all']:
        c0 = aggregate['C0_frozen']['gsm8k']['mean_acc']
        cx = aggregate[cond_name]['gsm8k']['mean_acc']
        delta = cx - c0
        status = "PASS" if delta > -0.02 else f"DEGRADED (Δ={delta:+.1%})"
        logger.info(f"    {cond_name}: {cx:.1%} vs C0={c0:.1%} → {status}")

    # ─── Summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  CROSS-CONDITION SUMMARY (mean across all benchmarks)")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Condition':25s} {'Mean Acc':>10s} {'Mean Ent.Ratio':>14s}")
    logger.info(f"  {'-'*52}")

    for cond_name in CONDITIONS:
        accs = [aggregate[cond_name][b]['mean_acc'] for b, _, _ in BENCHMARKS]
        ent_ratios = [aggregate[cond_name][b]['mean_entropy_ratio'] for b, _, _ in BENCHMARKS]
        logger.info(
            f"  {cond_name:25s} "
            f"{np.mean(accs):>9.1%} "
            f"{np.mean(ent_ratios):>13.4f}"
        )

    # ─── Decision ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  DECISION")
    logger.info(f"{'='*70}")

    if any_pass:
        logger.info("  ✓ FROZEN ENCODER WAS THE CEILING — unfreezing enables cross-domain reasoning")
        logger.info("  → Paper contribution: unfrozen encoder + geometric routing")
    else:
        logger.info("  ✗ Unfreezing does not help non-math benchmarks")
        logger.info("  → The bottleneck may not be the encoder (or CO-FRN is wrong abstraction)")

    # ─── Save ─────────────────────────────────────────────────────────
    output = {
        'experiment': 'E13_unfrozen_encoder',
        'description': 'E13: Selective encoder unfreezing to break frozen-encoder ceiling',
        'conditions': {k: v['description'] for k, v in CONDITIONS.items()},
        'seeds': SEEDS,
        'device': DEVICE,
        'e12_c3_baselines': E12_C3_BASELINES,
        'per_condition_per_seed': all_results,
        'aggregate': aggregate,
        'any_non_math_pass': any_pass,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': time.time() - t0,
    }

    out_path = RESULTS_DIR / "e13_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total elapsed: {time.time() - t0:.1f}s ({(time.time() - t0)/3600:.1f}h)")


if __name__ == "__main__":
    main()
