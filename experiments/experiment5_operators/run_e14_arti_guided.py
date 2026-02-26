#!/usr/bin/env python3
"""
E14 ARTI-Guided Operator Selection — Inference-Only Experiment.

Tests whether ARTI-guided operator selection beats uniform/learned attention.
Uses pretrained CO-FRN (E12 C3) and ARTI v1 models — NO training, NO changes
to shared modules.

Procedure:
1. Discovery phase: run ARTI + CO-FRN on ARTI dataset to find type→anchor mapping
2. For each benchmark: ARTI detects type → replace attention weights → score answers
3. Compare 5 conditions across 3 seeds × 4 benchmarks

Conditions:
  C0: Baseline — learned attention (E12 C3 no sqrt(d), reproduces E12)
  C1: ARTI-forced — 80% weight on top-2 anchors per detected type
  C2: ARTI-soft — add type bias to logits before softmax
  C3: Random-forced — 80% on random 2 anchors (controls for non-uniformity alone)
  C4: Oracle-forced — heuristic labeler → anchors (upper bound)

Success: C1 > C0 > C3 on any non-math benchmark by >2pp = operators specialized
         AND ARTI guidance helps.

Usage:
    cd paper13_The_geometry_of_Machine_Reasoning
    python experiments/experiment5_operators/run_e14_arti_guided.py
"""

import sys
import time
import json
import copy
import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from shared.model import COFRN, COFRNConfig
from shared.arti import ARTI, ARTIConfig
from shared.reasoning_types import (
    ReasoningType, N_REASONING_TYPES, REASONING_TYPES,
    HeuristicLabeler, TYPE_SHORT_NAMES,
)
from shared.data_utils import precompute_embeddings_st, BenchmarkDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "e14_arti_guided"
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
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

E12_DIR = BASE_DIR / "results" / "e12_temperature_ablation"
ARTI_DIR = BASE_DIR / "results" / "arti_ensemble"
ARTI_V3_DIR = BASE_DIR / "results" / "arti_v3"


def patch_manifold_no_sqrt_d(manifold):
    """Monkey-patch compute_attention_weights to remove sqrt(d) scaling (E12 C3)."""
    def patched_compute_attention_weights(structural):
        query = manifold.manifold_proj(structural)
        logits = query @ manifold.anchors.T / manifold.tau
        weights = F.softmax(logits, dim=-1)
        return weights
    manifold.compute_attention_weights = patched_compute_attention_weights
    return manifold


def load_cofrn(seed: int) -> COFRN:
    """Load pretrained CO-FRN from E12 C3 (no sqrt(d))."""
    path = E12_DIR / f"cofrn_C3_no_sqrt_d_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"E12 C3 model not found: {path}")

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
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = COFRN(config)
    model.load_state_dict(
        torch.load(str(path), weights_only=True, map_location=DEVICE)
    )
    patch_manifold_no_sqrt_d(model.factorization.manifold)
    model.to(DEVICE).eval()
    return model


def load_arti() -> ARTI:
    """Load pretrained ARTI v1 from ensemble results."""
    path = ARTI_DIR / "arti_v1.pt"
    if not path.exists():
        raise FileNotFoundError(f"ARTI v1 not found: {path}")

    config = ARTIConfig(
        encoder_dim=384,
        manifold_dim=10,
        n_geometric_features=32,
        classifier_type='mlp',
        hidden_dim=64,
    )
    arti = ARTI(config)
    arti.load_state_dict(
        torch.load(str(path), weights_only=True, map_location=DEVICE)
    )
    arti.to(DEVICE).eval()
    return arti


def load_arti_dataset() -> dict:
    """Load ARTI v3 dataset (7500 labeled samples)."""
    path = ARTI_V3_DIR / "dataset.pt"
    if not path.exists():
        raise FileNotFoundError(f"ARTI dataset not found: {path}")
    return torch.load(str(path), weights_only=False, map_location='cpu')


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Discover type → anchor mapping
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def discover_type_anchor_mapping(
    cofrn: COFRN,
    arti: ARTI,
    arti_dataset: dict,
) -> Dict[int, List[int]]:
    """
    Run ARTI + CO-FRN on labeled dataset to find which anchors fire most per type.

    Returns:
        type_to_top_anchors: {type_id: [anchor_idx_1, anchor_idx_2]} top-2 per type
    """
    embeddings = arti_dataset['embeddings']  # [N, 384]
    labels = arti_dataset['labels']          # [N]

    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.long)

    N = embeddings.shape[0]
    n_anchors = cofrn.config.n_anchors
    batch_size = 256

    # Accumulate anchor weights per type
    type_anchor_sums = defaultdict(lambda: torch.zeros(n_anchors))
    type_counts = defaultdict(int)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        emb_batch = embeddings[start:end].to(DEVICE)
        lab_batch = labels[start:end]

        # Get ARTI type predictions
        arti_out = arti(emb_batch)
        pred_types = arti_out['type'].cpu()

        # Get CO-FRN anchor weights (use s0 embeddings through the encoder+factorization)
        s0 = cofrn.encoder(emb_batch)
        fact_out = cofrn.factorization(s0)
        anchor_weights = fact_out['weights'].cpu()  # [B, n_anchors]

        # Accumulate by predicted type
        for i in range(end - start):
            t = pred_types[i].item()
            type_anchor_sums[t] += anchor_weights[i]
            type_counts[t] += 1

    # Find top-2 anchors per type
    type_to_top_anchors = {}
    logger.info("\n  Type → Anchor Mapping (top-2 per ARTI-predicted type):")
    for t in range(N_REASONING_TYPES):
        if type_counts[t] > 0:
            mean_weights = type_anchor_sums[t] / type_counts[t]
            top2 = mean_weights.topk(2).indices.tolist()
            type_to_top_anchors[t] = top2
            name = TYPE_SHORT_NAMES[t]
            logger.info(
                f"    {name:>12s} (n={type_counts[t]:4d}): "
                f"anchors [{top2[0]:2d}, {top2[1]:2d}]  "
                f"weights [{mean_weights[top2[0]]:.3f}, {mean_weights[top2[1]]:.3f}]  "
                f"max_any={mean_weights.max():.3f}"
            )
        else:
            type_to_top_anchors[t] = [0, 1]  # fallback
            logger.info(f"    {TYPE_SHORT_NAMES[t]:>12s}: no samples, fallback [0, 1]")

    return type_to_top_anchors


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Evaluate with attention override
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_with_attention_override(
    cofrn: COFRN,
    arti: ARTI,
    ds: BenchmarkDataset,
    condition: str,
    type_to_anchors: Dict[int, List[int]],
    heuristic_labeler: Optional[HeuristicLabeler] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate CO-FRN with modified attention weights based on condition.

    Args:
        cofrn: pretrained CO-FRN model
        arti: pretrained ARTI model
        ds: benchmark dataset
        condition: one of C0, C1, C2, C3, C4
        type_to_anchors: type → top-2 anchor mapping
        heuristic_labeler: for C4 (oracle) condition
        seed: for C3 (random) reproducibility
    """
    n = len(ds)
    n_anchors = cofrn.config.n_anchors
    rng = np.random.RandomState(seed)
    batch_size = 64

    q = ds.question_embeddings.to(DEVICE)
    a = ds.answer_embeddings.to(DEVICE)
    lab = ds.labels.to(DEVICE)

    correct = 0
    total = 0
    all_entropies = []
    all_max_weights = []
    type_distribution = defaultdict(int)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        q_batch = q[start:end]
        a_batch = a[start:end]
        lab_batch = lab[start:end]
        B = end - start

        # Step 1: Encode through CO-FRN pipeline
        s0 = cofrn.encoder(q_batch)
        fact_out = cofrn.factorization(s0)
        original_weights = fact_out['weights']  # [B, n_anchors]

        # Step 2: Detect types
        if condition in ('C1', 'C2'):
            arti_out = arti(q_batch)
            detected_types = arti_out['type'].cpu().numpy()
        elif condition == 'C4' and heuristic_labeler is not None and ds.metadata is not None:
            detected_types = []
            for i in range(start, end):
                meta = ds.metadata[i]
                text = meta.get('question', '')
                rtype, _ = heuristic_labeler.label_hard(text)
                detected_types.append(rtype.value)
            detected_types = np.array(detected_types)
        else:
            detected_types = None

        # Step 3: Compute modified weights based on condition
        if condition == 'C0':
            # Baseline: use learned weights as-is
            weights = original_weights

        elif condition == 'C1':
            # ARTI-forced: 80% weight on top-2 anchors, 20% spread over rest
            weights = torch.full((B, n_anchors), 0.2 / (n_anchors - 2), device=DEVICE)
            for i in range(B):
                t = detected_types[i]
                top2 = type_to_anchors.get(int(t), [0, 1])
                weights[i, top2[0]] = 0.40
                weights[i, top2[1]] = 0.40
                # Zero out the top2 from the uniform remainder
                weights[i, top2[0]] = 0.40
                weights[i, top2[1]] = 0.40
                # Renormalize the non-top-2 to sum to 0.2
                mask = torch.ones(n_anchors, dtype=torch.bool, device=DEVICE)
                mask[top2[0]] = False
                mask[top2[1]] = False
                weights[i, mask] = 0.2 / mask.sum()

        elif condition == 'C2':
            # ARTI-soft: add type bias to logits before softmax
            structural = fact_out['structural']
            query = cofrn.factorization.manifold.manifold_proj(structural)
            logits = query @ cofrn.factorization.manifold.anchors.T / cofrn.factorization.manifold.tau

            # Add bias: +2.0 to top-2 anchors for detected type
            bias = torch.zeros((B, n_anchors), device=DEVICE)
            for i in range(B):
                t = detected_types[i]
                top2 = type_to_anchors.get(int(t), [0, 1])
                bias[i, top2[0]] = 2.0
                bias[i, top2[1]] = 2.0

            weights = F.softmax(logits + bias, dim=-1)

        elif condition == 'C3':
            # Random-forced: 80% on random 2 anchors
            weights = torch.full((B, n_anchors), 0.2 / (n_anchors - 2), device=DEVICE)
            for i in range(B):
                random_2 = rng.choice(n_anchors, size=2, replace=False).tolist()
                weights[i, random_2[0]] = 0.40
                weights[i, random_2[1]] = 0.40
                mask = torch.ones(n_anchors, dtype=torch.bool, device=DEVICE)
                mask[random_2[0]] = False
                mask[random_2[1]] = False
                weights[i, mask] = 0.2 / mask.sum()

        elif condition == 'C4':
            # Oracle-forced: heuristic labeler → anchors
            weights = torch.full((B, n_anchors), 0.2 / (n_anchors - 2), device=DEVICE)
            for i in range(B):
                t = detected_types[i]
                top2 = type_to_anchors.get(int(t), [0, 1])
                weights[i, top2[0]] = 0.40
                weights[i, top2[1]] = 0.40
                mask = torch.ones(n_anchors, dtype=torch.bool, device=DEVICE)
                mask[top2[0]] = False
                mask[top2[1]] = False
                weights[i, mask] = 0.2 / mask.sum()
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Step 4: Apply operator with override weights and score
        transformed = cofrn.factorization.manifold.apply_operator(s0, weights)

        # Encode answers
        answer_enc = cofrn.encode_answers(a_batch)

        # Score via reasoning engine
        reason_out = cofrn.reasoning.forward_direct(
            transformed=transformed,
            answer_encodings=answer_enc,
        )
        scores = reason_out['scores']
        pred = scores.argmax(dim=-1)

        correct += (pred == lab_batch).sum().item()
        total += B

        # Diagnostics
        ent = -(weights * torch.log(weights + 1e-10)).sum(dim=-1)
        all_entropies.append(ent.cpu())
        all_max_weights.append(weights.max(dim=-1).values.cpu())

        # Type distribution
        if detected_types is not None:
            for t in detected_types:
                type_distribution[int(t)] += 1

    acc = correct / total
    max_ent = np.log(n_anchors)
    cat_ent = torch.cat(all_entropies)
    cat_max_w = torch.cat(all_max_weights)

    result = {
        'accuracy': acc,
        'n_examples': total,
        'anchor_entropy_mean': cat_ent.mean().item(),
        'anchor_entropy_ratio': cat_ent.mean().item() / max_ent,
        'max_anchor_weight_mean': cat_max_w.mean().item(),
        'max_anchor_weight_std': cat_max_w.std().item(),
    }

    if type_distribution:
        result['type_distribution'] = {
            TYPE_SHORT_NAMES[t]: c for t, c in sorted(type_distribution.items())
        }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = ['C0', 'C1', 'C2', 'C3', 'C4']
CONDITION_NAMES = {
    'C0': 'Baseline (E12 C3 learned attention)',
    'C1': 'ARTI-forced (80% on top-2 anchors per type)',
    'C2': 'ARTI-soft (logit bias +2.0 on top-2)',
    'C3': 'Random-forced (80% on random 2 anchors)',
    'C4': 'Oracle-forced (heuristic labeler → anchors)',
}


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("  E14 ARTI-GUIDED OPERATOR SELECTION (INFERENCE-ONLY)")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Conditions: {CONDITIONS}")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 70)

    # Load ARTI model (shared across seeds)
    logger.info("\nLoading ARTI v1...")
    arti = load_arti()
    logger.info(f"  ARTI params: {arti.trainable_params:,}")

    # Load ARTI dataset for type→anchor discovery
    logger.info("Loading ARTI v3 dataset...")
    arti_dataset = load_arti_dataset()
    logger.info(f"  Dataset size: {len(arti_dataset['labels'])} samples")

    # Heuristic labeler for C4
    heuristic_labeler = HeuristicLabeler()

    all_results = {}

    for seed in SEEDS:
        logger.info(f"\n{'='*70}")
        logger.info(f"  SEED = {seed}")
        logger.info(f"{'='*70}")

        # Load CO-FRN for this seed
        logger.info(f"\n  Loading CO-FRN (E12 C3, seed={seed})...")
        cofrn = load_cofrn(seed)
        logger.info(f"  CO-FRN trainable params: {cofrn.trainable_params:,}")

        # Phase 1: Discover type→anchor mapping
        logger.info(f"\n  Phase 1: Discovering type→anchor mapping...")
        type_to_anchors = discover_type_anchor_mapping(cofrn, arti, arti_dataset)

        # Phase 2: Evaluate each condition on each benchmark
        seed_results = {}

        for cond in CONDITIONS:
            logger.info(f"\n  --- {cond}: {CONDITION_NAMES[cond]} ---")
            cond_results = {}

            for bench, split, n_choices in BENCHMARKS:
                ds = precompute_embeddings_st(
                    'all-MiniLM-L6-v2', bench, split, seed=seed
                )

                result = evaluate_with_attention_override(
                    cofrn=cofrn,
                    arti=arti,
                    ds=ds,
                    condition=cond,
                    type_to_anchors=type_to_anchors,
                    heuristic_labeler=heuristic_labeler,
                    seed=seed,
                )

                random_baseline = 1.0 / n_choices
                result['random_baseline'] = random_baseline
                result['lift_over_random'] = result['accuracy'] - random_baseline
                cond_results[bench] = result

                logger.info(
                    f"    {bench:15s}: {result['accuracy']:.1%}  "
                    f"(Δ random={result['lift_over_random']:+.1%})  "
                    f"ent_ratio={result['anchor_entropy_ratio']:.4f}  "
                    f"max_w={result['max_anchor_weight_mean']:.3f}"
                )

            seed_results[cond] = cond_results

        all_results[str(seed)] = seed_results

        # Clean up
        del cofrn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─── Aggregate across seeds ───────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  AGGREGATE RESULTS (3-seed means)")
    logger.info(f"{'='*70}")

    aggregate = {}
    for cond in CONDITIONS:
        aggregate[cond] = {}
        for bench, _, n_choices in BENCHMARKS:
            accs = [all_results[str(s)][cond][bench]['accuracy'] for s in SEEDS]
            ent_ratios = [all_results[str(s)][cond][bench]['anchor_entropy_ratio'] for s in SEEDS]
            max_ws = [all_results[str(s)][cond][bench]['max_anchor_weight_mean'] for s in SEEDS]

            aggregate[cond][bench] = {
                'mean_acc': float(np.mean(accs)),
                'std_acc': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                'mean_entropy_ratio': float(np.mean(ent_ratios)),
                'mean_max_weight': float(np.mean(max_ws)),
                'per_seed_acc': {str(s): a for s, a in zip(SEEDS, accs)},
            }

    # ─── Print comparison table ───────────────────────────────────────
    for bench, _, n_choices in BENCHMARKS:
        logger.info(f"\n  {bench} (random={1/n_choices:.1%}, E12_C3={E12_C3_BASELINES[bench]['mean']:.1%}):")
        logger.info(
            f"  {'Condition':45s} {'Acc':>12s} {'Δ vs C0':>8s} {'Ent.Ratio':>10s}"
        )
        logger.info(f"  {'-'*78}")

        c0_acc = aggregate['C0'][bench]['mean_acc']
        for cond in CONDITIONS:
            a = aggregate[cond][bench]
            delta = a['mean_acc'] - c0_acc
            marker = ""
            if cond != 'C0' and delta > 0.02:
                marker = " ✓"
            elif cond != 'C0' and delta < -0.02:
                marker = " ✗"
            logger.info(
                f"  {CONDITION_NAMES[cond]:45s} "
                f"{a['mean_acc']:>5.1%}±{a['std_acc']:.1%} "
                f"{delta:>+7.1%}{marker} "
                f"{a['mean_entropy_ratio']:>9.4f}"
            )

    # ─── Success criteria check ───────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  SUCCESS CRITERIA CHECK")
    logger.info(f"{'='*70}")

    # Check: C0 reproduces E12 C3 within 1pp
    logger.info("\n  1) C0 reproduces E12 C3 (within 1pp)?")
    for bench, _, _ in BENCHMARKS:
        c0_acc = aggregate['C0'][bench]['mean_acc']
        e12_acc = E12_C3_BASELINES[bench]['mean']
        diff = abs(c0_acc - e12_acc)
        status = "PASS" if diff < 0.01 else f"WARN (diff={diff:.1%})"
        logger.info(f"    {bench:15s}: C0={c0_acc:.1%}, E12={e12_acc:.1%}, diff={diff:.1%} → {status}")

    # Check: C1 > C0 > C3 on any non-math benchmark by >2pp
    logger.info("\n  2) C1 > C0 > C3 on any non-math benchmark by >2pp?")
    non_math_benchmarks = ['arc_challenge', 'strategyqa', 'folio']
    any_pass = False
    for bench in non_math_benchmarks:
        c0 = aggregate['C0'][bench]['mean_acc']
        c1 = aggregate['C1'][bench]['mean_acc']
        c3 = aggregate['C3'][bench]['mean_acc']
        c1_vs_c0 = c1 - c0
        c0_vs_c3 = c0 - c3
        c1_vs_c3 = c1 - c3
        passed = c1_vs_c0 > 0.02 and c0_vs_c3 > 0.0
        if passed:
            any_pass = True
        logger.info(
            f"    {bench:15s}: C1={c1:.1%}, C0={c0:.1%}, C3={c3:.1%}  "
            f"(C1-C0={c1_vs_c0:+.1%}, C0-C3={c0_vs_c3:+.1%})  "
            f"→ {'PASS' if passed else 'FAIL'}"
        )

    if any_pass:
        logger.info("\n  ✓ OPERATORS ARE SPECIALIZED AND ARTI GUIDANCE HELPS")
    else:
        logger.info("\n  ✗ No non-math benchmark shows C1 > C0 > C3 by >2pp")

    # Check: C2 (soft) vs C0
    logger.info("\n  3) C2 (soft guidance) vs C0?")
    for bench in non_math_benchmarks:
        c0 = aggregate['C0'][bench]['mean_acc']
        c2 = aggregate['C2'][bench]['mean_acc']
        diff = c2 - c0
        logger.info(f"    {bench:15s}: C2-C0={diff:+.1%}")

    # Check: C4 (oracle) vs C1
    logger.info("\n  4) C4 (oracle) vs C1 (ARTI-guided)?")
    for bench in non_math_benchmarks:
        c1 = aggregate['C1'][bench]['mean_acc']
        c4 = aggregate['C4'][bench]['mean_acc']
        diff = c4 - c1
        logger.info(f"    {bench:15s}: C4-C1={diff:+.1%}")

    # ─── Type distribution per benchmark ──────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("  TYPE DISTRIBUTION PER BENCHMARK (seed=42, C1)")
    logger.info(f"{'='*70}")

    for bench, _, _ in BENCHMARKS:
        r = all_results['42']['C1'][bench]
        if 'type_distribution' in r:
            logger.info(f"\n  {bench}:")
            dist = r['type_distribution']
            total = sum(dist.values())
            for name, count in sorted(dist.items(), key=lambda x: -x[1]):
                logger.info(f"    {name:>12s}: {count:4d} ({count/total:.0%})")

    # ─── Save ─────────────────────────────────────────────────────────
    output = {
        'experiment': 'E14_arti_guided_operator_selection',
        'description': 'Inference-only: ARTI-guided attention override vs learned attention',
        'conditions': CONDITION_NAMES,
        'seeds': SEEDS,
        'device': DEVICE,
        'e12_c3_baselines': E12_C3_BASELINES,
        'per_seed': all_results,
        'aggregate': aggregate,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': time.time() - t0,
    }

    out_path = RESULTS_DIR / "e14_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
