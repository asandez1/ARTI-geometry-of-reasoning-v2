#!/usr/bin/env python3
"""
Quick benchmark: 100 StrategyQA + 100 FOLIO with controller-enabled CO-FRN.

Compares:
  1. Random baseline (expected accuracy)
  2. COFRN baseline (direct scoring, no controller)
  3. Controller-enabled (type-aware routing: fast vs structured path)

Both COFRN and controller use the SAME trained answer projection for fairness.
The controller adds type detection + routing but doesn't change the scoring head.

Previous results (E1, trained per-benchmark): StrategyQA ~51%, FOLIO ~35%
(Both essentially chance-level: 50% binary, 33% 3-way)

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_benchmark.py
"""

import sys
import time
import json
import logging
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np

# Path setup
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))

from shared.model import COFRN, COFRNConfig
from shared.controller import (
    GeometricReasoningController, ControllerConfig, TypeClassifier,
    CORE_TYPE_NAMES, N_CORE_TYPES,
)
from shared.data_utils import precompute_embeddings_st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "controller"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


def load_models():
    """Load trained COFRN (Phase 0) and TypeClassifier (Phase 1)."""
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
    )

    cofrn = COFRN(config)
    cofrn.load_state_dict(
        torch.load(str(RESULTS_DIR / "cofrn_quick.pt"),
                    weights_only=True, map_location=DEVICE)
    )
    cofrn.to(DEVICE).eval()
    logger.info(f"COFRN loaded: {cofrn.trainable_params:,} params")

    ctrl_config = ControllerConfig(
        hidden_dim=config.hidden_dim,
        manifold_dim=config.manifold_dim,
        struct_dim=config.struct_dim,
    )

    controller = GeometricReasoningController(
        config=ctrl_config,
        factorization=cofrn.factorization,
        reasoning_engine=cofrn.reasoning,
    )

    # Load trained TypeClassifier weights
    type_clf_state = torch.load(
        str(RESULTS_DIR / "type_clf.pt"),
        weights_only=True, map_location=DEVICE,
    )
    controller.type_clf.load_state_dict(type_clf_state)
    controller.to(DEVICE).eval()
    logger.info(f"Controller loaded: TypeClassifier {controller.type_clf.trainable_params:,} params")

    return cofrn, controller, ctrl_config


def evaluate_benchmark(benchmark, split, n_examples, cofrn, controller, ctrl_config,
                       force_fast=False):
    """
    Evaluate both COFRN baseline and controller on a benchmark slice.

    COFRN baseline: direct scoring (task_type='single_step')
    Controller: type-aware routing (fast=direct, struct=HilbertTree with type depth/delta)

    Both use COFRN's trained answer_proj for fair comparison.

    Args:
        force_fast: if True, controller always uses fast path (isolates type
                    detection from structured path damage)
    """
    mode = "UNIFORM DEPTH" if force_fast else "TYPE-SPECIFIC DEPTH"
    print(f"\n{'='*65}")
    print(f"  {benchmark.upper()} — {split} split, {n_examples} ex — {mode}")
    print(f"{'='*65}")

    # Load data
    ds = precompute_embeddings_st('all-MiniLM-L6-v2', benchmark, split, seed=SEED)
    n_available = len(ds)

    # Subsample
    n = min(n_examples, n_available)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n_available)[:n]

    q_emb = ds.question_embeddings[indices].to(DEVICE)   # [N, 384]
    a_emb = ds.answer_embeddings[indices].to(DEVICE)     # [N, n_ans, 384]
    labels = ds.labels[indices].to(DEVICE)                # [N]

    n_choices = a_emb.shape[1]
    random_baseline = 1.0 / n_choices
    logger.info(f"  {n}/{n_available} examples, {n_choices} choices, "
                f"random baseline: {random_baseline:.1%}")

    # Label distribution
    label_dist = Counter(labels.cpu().numpy().tolist())
    logger.info(f"  Label distribution: {dict(sorted(label_dist.items()))}")

    # ─── COFRN Baseline ──────────────────────────────────────────────
    cofrn_correct = 0
    with torch.no_grad():
        for start in range(0, n, 32):
            end = min(start + 32, n)
            out = cofrn(
                embeddings=q_emb[start:end],
                answer_embeddings=a_emb[start:end],
                labels=labels[start:end],
            )
            pred = out['scores'].argmax(dim=-1)
            cofrn_correct += (pred == labels[start:end]).sum().item()

    cofrn_acc = cofrn_correct / n

    # ─── Controller-Enabled ──────────────────────────────────────────
    # Manual pipeline using COFRN's trained answer projection
    ctrl_correct = 0
    type_counts = Counter()
    route_counts = Counter()
    per_type_correct = Counter()
    per_type_total = Counter()

    structured_set = set(ctrl_config.structured_types)

    with torch.no_grad():
        for start in range(0, n, 32):
            end = min(start + 32, n)
            batch_q = q_emb[start:end]     # [B, 384]
            batch_a = a_emb[start:end]     # [B, n_ans, 384]
            batch_l = labels[start:end]    # [B]
            B = batch_q.shape[0]

            # 1. Encode question → s0 [B, 256]
            s0 = cofrn.encode_text(embeddings=batch_q)

            # 2. Factorize
            fact_out = cofrn.factorization(s0)
            structural = fact_out['structural']
            transformed = fact_out['transformed']
            anchor_weights = fact_out['weights']

            # 3. Detect reasoning type (from s0, pre-factorization)
            type_probs, detected_type, confidence, _ = \
                controller.detect_type(s0, structural)

            # 4. Encode answers using COFRN's trained projection (384 → 256)
            answer_enc = cofrn.encode_answers(batch_a)  # [B, n_ans, 256]

            # 5. Route per example: type-specific depth via forward_multistep
            # Both fast and structured types use the trained beam search —
            # the controller's value comes from TYPE-SPECIFIC DEPTH, not
            # switching between different scoring pathways.
            operator_fn = cofrn.factorization.manifold.forward

            for i in range(B):
                t = detected_type[i].item()
                conf = confidence[i].item()
                t_name = CORE_TYPE_NAMES[t]
                type_counts[t_name] += 1
                per_type_total[t_name] += 1

                if force_fast:
                    # Uniform depth (same as COFRN baseline)
                    depth = cofrn.reasoning.tree.max_depth
                    delta = cofrn.reasoning.tree.delta
                    route_counts['uniform'] += 1
                else:
                    # Type-specific depth and delta
                    depth, delta = controller.get_type_config(t)
                    label = 'fast' if t in set(ctrl_config.fast_types) else 'struct'
                    route_counts[label] += 1

                    # Top-k masking for structured types
                    if t in structured_set or conf < ctrl_config.confidence_threshold:
                        w = anchor_weights[i:i+1]
                        topk_vals, topk_idx = w.topk(
                            ctrl_config.top_k_anchors, dim=-1)
                        mask = torch.zeros_like(w)
                        mask.scatter_(1, topk_idx, 1.0)
                        masked_w = w * mask
                        masked_w = masked_w / (
                            masked_w.sum(dim=-1, keepdim=True) + 1e-10)

                        def hard_masked_op(x, s, _mw=masked_w):
                            Ox = cofrn.factorization.manifold.apply_operator(
                                x, _mw.expand(x.shape[0], -1))
                            return Ox, _mw
                        op = hard_masked_op
                    else:
                        op = operator_fn

                old_delta = cofrn.reasoning.tree.delta
                cofrn.reasoning.tree.delta = delta

                out = cofrn.reasoning.forward_multistep(
                    transformed=transformed[i:i+1],
                    evidence=transformed[i:i+1],
                    operator_fn=op if not force_fast else operator_fn,
                    structural=structural[i:i+1],
                    answer_encodings=answer_enc[i:i+1],
                    depth=depth,
                )
                cofrn.reasoning.tree.delta = old_delta

                pred = out['scores'].argmax(dim=-1)
                correct = pred.item() == batch_l[i].item()
                if correct:
                    ctrl_correct += 1
                    per_type_correct[t_name] += 1

    ctrl_acc = ctrl_correct / n
    lift = ctrl_acc - cofrn_acc

    # ─── Report ──────────────────────────────────────────────────────
    print(f"\n  Random baseline:   {random_baseline:>6.1%}")
    print(f"  COFRN baseline:    {cofrn_acc:>6.1%}  ({cofrn_correct}/{n})")
    print(f"  Controller:        {ctrl_acc:>6.1%}  ({ctrl_correct}/{n})")
    print(f"  Lift:              {lift:>+6.1%}")

    print(f"\n  Type distribution:")
    for t_name in CORE_TYPE_NAMES:
        cnt = type_counts.get(t_name, 0)
        corr = per_type_correct.get(t_name, 0)
        tot = per_type_total.get(t_name, 0)
        acc_str = f"{corr/tot:.0%}" if tot > 0 else "n/a"
        print(f"    {t_name:<16} {cnt:>4}  acc={acc_str}")

    print(f"\n  Routing: {route_counts['fast']} fast, {route_counts['struct']} structured")

    return {
        'benchmark': benchmark,
        'split': split,
        'n_examples': n,
        'n_available': n_available,
        'n_choices': n_choices,
        'random_baseline': random_baseline,
        'cofrn_acc': cofrn_acc,
        'controller_acc': ctrl_acc,
        'lift': lift,
        'type_distribution': dict(type_counts),
        'routing': dict(route_counts),
        'per_type_accuracy': {
            t: per_type_correct[t] / max(per_type_total[t], 1)
            for t in CORE_TYPE_NAMES if per_type_total[t] > 0
        },
    }


def main():
    print("=" * 65)
    print("  Controller Benchmark: 4 Benchmarks x 2 Modes")
    print(f"  Device: {DEVICE}")
    print("=" * 65)
    t0 = time.time()

    cofrn, controller, ctrl_config = load_models()

    # Benchmarks: (name, split, n_examples)
    benchmarks = [
        ('gsm8k', 'test', 100),
        ('arc_challenge', 'test', 100),
        ('strategyqa', 'train', 100),
        ('folio', 'validation', 100),
    ]

    results = []

    for bm, split, n in benchmarks:
        # Full routing (fast + structured)
        r_full = evaluate_benchmark(
            bm, split, n, cofrn, controller, ctrl_config,
            force_fast=False,
        )
        r_full['mode'] = 'full_routing'

        # Fast-only (isolates type detection from structured path)
        r_fast = evaluate_benchmark(
            bm, split, n, cofrn, controller, ctrl_config,
            force_fast=True,
        )
        r_fast['mode'] = 'fast_only'

        results.extend([r_full, r_fast])

    # ─── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Benchmark':<15} {'Random':>7} {'COFRN':>7}  "
          f"{'TypeDepth':>11} {'Uniform':>11}  {'Notes'}")
    print(f"  {'-'*72}")

    for i in range(0, len(results), 2):
        r_typed = results[i]
        r_uniform = results[i+1]
        bm = r_typed['benchmark']
        trained = 'in-domain' if bm in ('gsm8k', 'arc_challenge') else 'OOD'
        print(f"  {bm:<15} {r_typed['random_baseline']:>6.1%} {r_typed['cofrn_acc']:>6.1%}  "
              f"{r_typed['controller_acc']:>6.1%} ({r_typed['lift']:>+.1%})  "
              f"{r_uniform['controller_acc']:>6.1%} ({r_uniform['lift']:>+.1%})  "
              f"{trained}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  TypeDepth = type-specific depth + masking; Uniform = same depth for all")
    print(f"  in-domain = trained on; OOD = zero-shot transfer")

    # Save
    save_path = RESULTS_DIR / "benchmark_results.json"
    with open(str(save_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {save_path}")


if __name__ == "__main__":
    main()
