#!/usr/bin/env python3
"""
E16 Unfrozen Encoder + ARTI-Routed Scoring — Combining E13 and E15.

E13 showed that unfreezing the encoder alone (cosine-only scoring) FAILS:
constant predictions on StrategyQA/FOLIO (54.0%/30.5% with zero std).
E15 showed that ARTI-routed scoring (frozen encoder) achieves StrategyQA 81%
but ARC/FOLIO remain at frozen-encoder ceiling.

E16 combines both: unfrozen encoder + ARTI routing + multi-benchmark training.
This is the first test with all pipeline fixes simultaneously:
  - E10: cosine scoring (no MLP collapse)
  - E11: entropy annealing (no training artifact)
  - E12 C3: no sqrt(d) scaling
  - E15: ARTI-routed scoring (cosine + direct blend)
  - E13: unfrozen encoder (gradient flow into transformer)

Conditions (2 × 3 seeds × 4 benchmarks):
  C1: Unfreeze last 2 + ARTI-routed + 3 benchmarks (GSM8K + ARC + SQA)
  C2: Unfreeze last 2 + ARTI-routed + 4 benchmarks (add FOLIO training)

References:
  E15 C1 (frozen + routed): GSM8K 48.1%, SQA 81.0%, ARC 28.6%, FOLIO 33.5%
  E13 C1 (unfreeze 2, cosine): GSM8K 49.2%, SQA 54.0%, ARC 27.1%, FOLIO 30.5%

Success Criteria:
  SC1: C1 GSM8K >= 48% (no regression vs E15 C1)
  SC2: C1 StrategyQA >= 75% (preserve E15 breakthrough)
  SC3: C1 ARC > 32% (break frozen-encoder ceiling of ~29%)
  SC4: C2 FOLIO > 38% (benefit from training data + unfreezing)
  SC5: C1 or C2 mean across 4 benchmarks > E15 C1 mean

Usage:
    cd paper13_The_geometry_of_Machine_Reasoning
    python experiments/experiment5_operators/run_e16_unfrozen_routed.py
"""

import sys
import time
import json
import logging
import inspect
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
    TrainConfig, CurriculumPhase, curriculum_train,
)
from shared.data_utils import (
    TextBenchmarkDataset, collate_text_benchmark,
    load_text_benchmark, LOADERS, BENCHMARK_CONFIGS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "e16_unfrozen_routed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 7]

ARTI_CHECKPOINT = str(BASE_DIR / "results" / "arti_ensemble" / "arti_v1.pt")

BENCHMARKS = [
    ('gsm8k',         'test',       4),
    ('arc_challenge',  'test',       4),
    ('strategyqa',    'train',      2),
    ('folio',         'validation', 3),
]

# E15 C1 baselines (frozen + ARTI-routed, 3 benchmarks)
E15_C1_BASELINES = {
    'gsm8k':         {'mean': 0.481, 'std': 0.027},
    'arc_challenge': {'mean': 0.286, 'std': 0.014},
    'strategyqa':    {'mean': 0.810, 'std': 0.038},
    'folio':         {'mean': 0.335, 'std': 0.023},
}

# E13 C1 baselines (unfreeze 2, cosine-only, 2 benchmarks)
E13_C1_BASELINES = {
    'gsm8k':         {'mean': 0.492, 'std': 0.024},
    'arc_challenge': {'mean': 0.271, 'std': 0.010},
    'strategyqa':    {'mean': 0.540, 'std': 0.000},
    'folio':         {'mean': 0.305, 'std': 0.000},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Conditions
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    'C1_unfreeze2_routed_3bench': {
        'description': 'Unfreeze last 2 + ARTI-routed + 3 benchmarks (GSM8K+ARC+SQA)',
        'unfreeze_layers': 2,
        'encoder_lr': 1e-5,
        'train_benchmarks': ['gsm8k', 'arc_challenge', 'strategyqa'],
        'batch_size': 16,
        'gradient_accumulation_steps': 2,
    },
    'C2_unfreeze2_routed_4bench': {
        'description': 'Unfreeze last 2 + ARTI-routed + 4 benchmarks (add FOLIO)',
        'unfreeze_layers': 2,
        'encoder_lr': 1e-5,
        'train_benchmarks': ['gsm8k', 'arc_challenge', 'strategyqa', 'folio'],
        'batch_size': 16,
        'gradient_accumulation_steps': 2,
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
# Data loading — Text-based with mixed-domain padding + valid_choices
# ═══════════════════════════════════════════════════════════════════════════════

class TextMixedDomainDataset(TextBenchmarkDataset):
    """
    Concatenates multiple TextBenchmarkDatasets with padding to max_choices.

    Handles different n_choices across benchmarks (GSM8K=4, ARC=4, SQA=2,
    FOLIO=3) by padding answer_input_ids/mask to max_choices with pad tokens,
    and providing a valid_choices boolean mask.
    """

    def __init__(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        valid_choices: torch.Tensor,
        domain: str = 'mixed',
    ):
        super().__init__(
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            answer_input_ids=answer_input_ids,
            answer_attention_mask=answer_attention_mask,
            labels=labels,
            domain=domain,
        )
        self._valid_choices = valid_choices

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['valid_choices'] = self._valid_choices[idx]
        return item


def collate_text_with_valid_choices(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for TextMixedDomainDataset (text + valid_choices)."""
    result = {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'answer_input_ids': torch.stack([b['answer_input_ids'] for b in batch]),
        'answer_attention_mask': torch.stack([b['answer_attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }
    if 'valid_choices' in batch[0]:
        result['valid_choices'] = torch.stack([b['valid_choices'] for b in batch])
    return result


def load_text_training_data(
    tokenizer,
    train_benchmarks: List[str],
    seed: int,
    max_length: int = 128,
) -> Tuple[TextMixedDomainDataset, TextMixedDomainDataset]:
    """Load tokenized text training data for multiple benchmarks with padding."""
    logger.info(f"  Loading tokenized training data for: {train_benchmarks}")

    datasets = {}
    for bench_name in train_benchmarks:
        ds = load_text_benchmark(
            bench_name, 'train', tokenizer=tokenizer,
            max_length=max_length, seed=seed,
        )
        datasets[bench_name] = ds
        logger.info(f"    {bench_name}: {len(ds)} examples, "
                     f"n_choices={ds.answer_input_ids.shape[1]}")

    # Find max choices across all benchmarks
    max_choices = max(ds.answer_input_ids.shape[1] for ds in datasets.values())
    L = next(iter(datasets.values())).question_input_ids.shape[1]  # max_length
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    all_q_ids = []
    all_q_mask = []
    all_a_ids = []
    all_a_mask = []
    all_labels = []
    all_valid = []

    for name, ds in datasets.items():
        n = len(ds)
        n_choices = ds.answer_input_ids.shape[1]

        all_q_ids.append(ds.question_input_ids)
        all_q_mask.append(ds.question_attention_mask)
        all_labels.append(ds.labels)

        # Pad answer choices if needed
        if n_choices < max_choices:
            pad_count = max_choices - n_choices
            a_pad_ids = torch.full((n, pad_count, L), pad_token_id, dtype=torch.long)
            a_pad_mask = torch.zeros(n, pad_count, L, dtype=torch.long)
            all_a_ids.append(torch.cat([ds.answer_input_ids, a_pad_ids], dim=1))
            all_a_mask.append(torch.cat([ds.answer_attention_mask, a_pad_mask], dim=1))
            valid = torch.zeros(n, max_choices, dtype=torch.bool)
            valid[:, :n_choices] = True
            all_valid.append(valid)
        else:
            all_a_ids.append(ds.answer_input_ids)
            all_a_mask.append(ds.answer_attention_mask)
            all_valid.append(torch.ones(n, max_choices, dtype=torch.bool))

    q_ids = torch.cat(all_q_ids, dim=0)
    q_mask = torch.cat(all_q_mask, dim=0)
    a_ids = torch.cat(all_a_ids, dim=0)
    a_mask = torch.cat(all_a_mask, dim=0)
    labels = torch.cat(all_labels, dim=0)
    valid_choices = torch.cat(all_valid, dim=0)

    N = len(labels)
    logger.info(f"  Combined: {N} examples, max_choices={max_choices}")

    # Shuffle and split
    val_size = min(500, N // 5)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    train_idx = torch.tensor(indices[val_size:], dtype=torch.long)
    val_idx = torch.tensor(indices[:val_size], dtype=torch.long)

    train_ds = TextMixedDomainDataset(
        question_input_ids=q_ids[train_idx],
        question_attention_mask=q_mask[train_idx],
        answer_input_ids=a_ids[train_idx],
        answer_attention_mask=a_mask[train_idx],
        labels=labels[train_idx],
        valid_choices=valid_choices[train_idx],
    )
    val_ds = TextMixedDomainDataset(
        question_input_ids=q_ids[val_idx],
        question_attention_mask=q_mask[val_idx],
        answer_input_ids=a_ids[val_idx],
        answer_attention_mask=a_mask[val_idx],
        labels=labels[val_idx],
        valid_choices=valid_choices[val_idx],
    )

    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(condition: dict, seed: int) -> COFRN:
    """Build COFRN with unfrozen encoder + ARTI-routed scoring."""
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
        # Unfreezing
        unfreeze_encoder_layers=condition['unfreeze_layers'],
        # E15 ARTI routing
        use_routed_scoring=True,
        arti_checkpoint=ARTI_CHECKPOINT,
        arti_encoder_dim=384,
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
        f"  Params: total={total_trainable:,}  "
        f"encoder={encoder_trainable:,}  downstream={downstream_trainable:,}"
    )

    # Load training data
    train_ds, val_ds = load_text_training_data(
        tokenizer=model.encoder.tokenizer,
        train_benchmarks=condition['train_benchmarks'],
        seed=seed,
        max_length=128,
    )

    model.to(DEVICE)

    base_cfg = TrainConfig(
        learning_rate=3e-4,
        encoder_lr=condition['encoder_lr'],
        weight_decay=0.01,
        grad_clip=1.0,
        batch_size=condition['batch_size'],
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

    model, train_result = curriculum_train(
        model, train_ds, val_ds, base_cfg,
        phases=phases,
        model_name=f'{cond_name}_s{seed}',
        collate_fn=collate_text_with_valid_choices,
    )

    best_acc = train_result.get('best_val_acc', 0.0)
    logger.info(f"[{cond_name}|seed={seed}] Best val acc: {best_acc:.2%}")

    # Verify gradient flow
    _verify_gradient_flow(model)

    # Save model
    torch.save(model.state_dict(), str(saved_path))
    logger.info(f"  Saved to {saved_path}")

    patch_manifold_no_sqrt_d(model.factorization.manifold)
    model.to(DEVICE).eval()
    return model


def _verify_gradient_flow(model: COFRN):
    """Verify encoder layers have non-zero grads."""
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
        f"encoder params had non-zero grads"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full(model: COFRN, seed: int) -> dict:
    """Evaluate on full test sets with anchor + router diagnostics.

    Uses text-based evaluation (unfrozen encoder — embeddings differ from
    frozen model, so precomputed ST embeddings are not valid).
    """
    results = {}

    for bench, split, n_choices in BENCHMARKS:
        ds = load_text_benchmark(
            bench, split, tokenizer=model.encoder.tokenizer,
            max_length=128, seed=seed,
        )
        n = len(ds)
        lab = ds.labels.to(DEVICE)

        # Build valid_choices mask for this benchmark
        max_choices = 4
        actual_choices = ds.answer_input_ids.shape[1]
        L = ds.answer_input_ids.shape[2]
        pad_token_id = model.encoder.tokenizer.pad_token_id or 0

        if actual_choices < max_choices:
            pad_count = max_choices - actual_choices
            a_pad_ids = torch.full((n, pad_count, L), pad_token_id, dtype=torch.long)
            a_pad_mask = torch.zeros(n, pad_count, L, dtype=torch.long)
            a_ids_padded = torch.cat([ds.answer_input_ids, a_pad_ids], dim=1)
            a_mask_padded = torch.cat([ds.answer_attention_mask, a_pad_mask], dim=1)
            vc = torch.zeros(n, max_choices, dtype=torch.bool)
            vc[:, :actual_choices] = True
        else:
            a_ids_padded = ds.answer_input_ids
            a_mask_padded = ds.answer_attention_mask
            vc = torch.ones(n, max_choices, dtype=torch.bool)

        correct = 0
        total = 0
        all_entropies = []
        all_max_weights = []
        all_alphas = []

        batch_size = 16  # smaller for unfrozen encoder memory
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            out = model(
                input_ids=ds.question_input_ids[start:end].to(DEVICE),
                attention_mask=ds.question_attention_mask[start:end].to(DEVICE),
                answer_input_ids=a_ids_padded[start:end].to(DEVICE),
                answer_attention_mask=a_mask_padded[start:end].to(DEVICE),
                labels=lab[start:end],
                valid_choices=vc[start:end].to(DEVICE),
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

            # Router alpha diagnostic
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
    logger.info("  E16 UNFROZEN ENCODER + ARTI-ROUTED SCORING")
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
        logger.info(f"  unfreeze={condition['unfreeze_layers']}, "
                     f"encoder_lr={condition['encoder_lr']}, "
                     f"benchmarks={condition['train_benchmarks']}")
        logger.info(f"{'='*70}")

        cond_seed_results = {}

        for seed in SEEDS:
            logger.info(f"\n--- {cond_name} | seed={seed} ---")
            t_seed = time.time()

            model = train_condition(cond_name, condition, seed)
            seed_results = evaluate_full(model, seed)
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

            elapsed = time.time() - t_seed
            logger.info(f"  Seed {seed} elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

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
        e15_acc = E15_C1_BASELINES[bench]['mean']
        e13_acc = E13_C1_BASELINES[bench]['mean']
        logger.info(f"\n  {bench} (random={1/n_choices:.1%}, "
                     f"E15_C1={e15_acc:.1%}, E13_C1={e13_acc:.1%}):")
        header = (f"  {'Condition':35s} {'Acc':>12s} "
                  f"{'Δ vs E15':>8s} {'Δ vs E13':>8s} "
                  f"{'Ent.R':>8s} {'Alpha':>8s}")
        logger.info(header)
        logger.info(f"  {'-'*86}")

        for cond_name in CONDITIONS:
            a = aggregate[cond_name][bench]
            delta_e15 = a['mean_acc'] - e15_acc
            delta_e13 = a['mean_acc'] - e13_acc
            alpha_str = f"{a.get('mean_router_alpha', 0):>7.3f}" if 'mean_router_alpha' in a else "    N/A"
            logger.info(
                f"  {cond_name:35s} "
                f"{a['mean_acc']:>5.1%}±{a['std_acc']:.1%} "
                f"{delta_e15:>+7.1%} "
                f"{delta_e13:>+7.1%} "
                f"{a['mean_entropy_ratio']:>7.4f} "
                f"{alpha_str}"
            )

    # ─── Success criteria check ───────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  SUCCESS CRITERIA CHECK")
    logger.info(f"{'='*70}")

    c1 = aggregate.get('C1_unfreeze2_routed_3bench', {})
    c2 = aggregate.get('C2_unfreeze2_routed_4bench', {})

    # SC1: C1 GSM8K >= 48%
    c1_gsm = c1.get('gsm8k', {}).get('mean_acc', 0)
    sc1 = c1_gsm >= 0.48
    logger.info(f"  SC1: C1 GSM8K >= 48%: {c1_gsm:.1%} -> {'PASS' if sc1 else 'FAIL'}")

    # SC2: C1 StrategyQA >= 75%
    c1_sqa = c1.get('strategyqa', {}).get('mean_acc', 0)
    sc2 = c1_sqa >= 0.75
    logger.info(f"  SC2: C1 StrategyQA >= 75%: {c1_sqa:.1%} -> {'PASS' if sc2 else 'FAIL'}")

    # SC3: C1 ARC > 32%
    c1_arc = c1.get('arc_challenge', {}).get('mean_acc', 0)
    sc3 = c1_arc > 0.32
    logger.info(f"  SC3: C1 ARC > 32%: {c1_arc:.1%} -> {'PASS' if sc3 else 'FAIL'}")

    # SC4: C2 FOLIO > 38%
    c2_folio = c2.get('folio', {}).get('mean_acc', 0)
    sc4 = c2_folio > 0.38
    logger.info(f"  SC4: C2 FOLIO > 38%: {c2_folio:.1%} -> {'PASS' if sc4 else 'FAIL'}")

    # SC5: Best condition mean > E15 C1 mean
    e15_mean = np.mean([v['mean'] for v in E15_C1_BASELINES.values()])
    c1_mean = np.mean([c1.get(b, {}).get('mean_acc', 0) for b, _, _ in BENCHMARKS])
    c2_mean = np.mean([c2.get(b, {}).get('mean_acc', 0) for b, _, _ in BENCHMARKS])
    best_mean = max(c1_mean, c2_mean)
    sc5 = best_mean > e15_mean
    logger.info(
        f"  SC5: Best mean > E15 C1 mean ({e15_mean:.1%}): "
        f"C1={c1_mean:.1%}, C2={c2_mean:.1%} -> {'PASS' if sc5 else 'FAIL'}"
    )

    n_pass = sum([sc1, sc2, sc3, sc4, sc5])
    logger.info(f"\n  Summary: {n_pass}/5 success criteria met")

    # ─── Save ─────────────────────────────────────────────────────────
    output = {
        'experiment': 'E16_unfrozen_routed',
        'description': 'E16: Unfrozen encoder + ARTI-routed scoring (combining E13 + E15)',
        'conditions': {k: v['description'] for k, v in CONDITIONS.items()},
        'seeds': SEEDS,
        'device': DEVICE,
        'arti_checkpoint': ARTI_CHECKPOINT,
        'e15_c1_baselines': E15_C1_BASELINES,
        'e13_c1_baselines': E13_C1_BASELINES,
        'per_condition_per_seed': all_results,
        'aggregate': aggregate,
        'success_criteria': {
            'SC1_gsm8k_no_regression': sc1,
            'SC2_sqa_preserved': sc2,
            'SC3_arc_above_32': sc3,
            'SC4_folio_above_38': sc4,
            'SC5_mean_above_e15': sc5,
            'n_pass': n_pass,
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': time.time() - t0,
    }

    out_path = RESULTS_DIR / "e16_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n  Results saved to {out_path}")
    logger.info(f"  Total elapsed: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}min)")


if __name__ == '__main__':
    main()
