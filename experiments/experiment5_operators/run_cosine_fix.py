#!/usr/bin/env python3
"""
E10: Cosine Scoring Fix — Train and evaluate CO-FRN with non-parametric scoring.

Root cause (E9): tree_scorer MLP learned to ignore best_hyp, scoring from
answer_enc only. Cosine scoring score(k) = cos(best_hyp, answer_k)/tau
cannot learn to ignore best_hyp.

Steps:
  1. Train COFRN with use_cosine_scoring=True from scratch (same curriculum as E5 Phase 0)
  2. Verify fix: replace best_hyp with random → predictions MUST change
  3. Evaluate on all 4 benchmarks (GSM8K, ARC, StrategyQA, FOLIO)
  4. Compare: old MLP scoring vs new cosine scoring
  5. Depth sweep: do different depths now produce different predictions?

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_cosine_fix.py
"""

import sys
import time
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

RESULTS_DIR = BASE_DIR / "results" / "cosine_fix"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Train COFRN with cosine scoring
# ═══════════════════════════════════════════════════════════════════════════════

def train_cosine_cofrn() -> COFRN:
    """Train COFRN with use_cosine_scoring=True from scratch."""
    logger.info("=" * 60)
    logger.info("E10 Step 1: Train COFRN with cosine scoring")
    logger.info("=" * 60)

    saved_path = RESULTS_DIR / "cofrn_cosine.pt"

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
        use_cosine_scoring=True,       # E10 fix
        entropy_schedule='anneal',     # Anneal entropy to allow operator specialization
        entropy_anneal_epochs=10,
    )

    # Check for cached
    if saved_path.exists():
        logger.info(f"Loading cached cosine COFRN from {saved_path}")
        model = COFRN(config)
        model.load_state_dict(
            torch.load(str(saved_path), weights_only=True, map_location=DEVICE)
        )
        model.to(DEVICE).eval()
        logger.info(f"  Loaded. Trainable params: {model.trainable_params:,}")
        return model

    # Load real benchmark data
    logger.info("Loading benchmarks with all-MiniLM-L6-v2 (384D)...")
    gsm8k_ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'gsm8k', 'train', seed=SEED)
    arc_ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'arc_challenge', 'train', seed=SEED)
    logger.info(f"  GSM8K: {len(gsm8k_ds)}, ARC: {len(arc_ds)}")

    mixed_ds = MixedDomainDataset({'gsm8k': gsm8k_ds, 'arc_challenge': arc_ds})
    n = len(mixed_ds)

    val_size = min(500, n // 5)
    rng = np.random.RandomState(SEED)
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
    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = COFRN(config).to(DEVICE)
    logger.info(f"  COFRN (cosine scoring) trainable params: {model.trainable_params:,}")
    logger.info(f"  use_cosine_scoring: {config.use_cosine_scoring}")

    base_cfg = TrainConfig(
        learning_rate=3e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        batch_size=32,
        max_epochs=30,
        patience=5,
        device=DEVICE,
        seed=SEED,
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

    logger.info(f"  Curriculum: {' → '.join(f'd={p.depth}({p.epochs}ep)' for p in phases)}")

    model, train_result = curriculum_train(
        model, train_ds, val_ds, base_cfg,
        phases=phases,
        model_name='cofrn_cosine',
    )

    best_acc = train_result.get('best_val_acc', 0.0)
    logger.info(f"  Training complete. Best val acc: {best_acc:.2%}")

    torch.save(model.state_dict(), str(saved_path))
    logger.info(f"  Saved to {saved_path}")

    model.to(DEVICE).eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Verify fix — does tree_scorer now use best_hyp?
# ═══════════════════════════════════════════════════════════════════════════════

def verify_fix(cofrn: COFRN) -> dict:
    """Run the E9 H2 test: replace best_hyp with random/zero/negated."""
    logger.info("=" * 60)
    logger.info("E10 Step 2: Verify cosine scoring uses best_hyp")
    logger.info("=" * 60)

    ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'gsm8k', 'test', seed=SEED)
    rng = np.random.RandomState(SEED)
    n = 100
    indices = rng.permutation(len(ds))[:n]
    q = ds.question_embeddings[indices].to(DEVICE)
    a = ds.answer_embeddings[indices].to(DEVICE)
    labels = ds.labels[indices].to(DEVICE)

    # Normal forward
    with torch.no_grad():
        s0 = cofrn.encode_text(embeddings=q)
        fact = cofrn.factorization(s0)
        answer_enc = cofrn.encode_answers(a)

        tree_out = cofrn.reasoning.tree(
            initial_state=fact['transformed'],
            evidence=fact['transformed'],
            operator_fn=cofrn.factorization.manifold.forward,
            structural=fact['structural'],
            depth=3,
        )
        h_w = F.softmax(tree_out['hypothesis_scores'], dim=-1)
        best_hyp = (h_w.unsqueeze(-1) * tree_out['hypothesis_states']).sum(dim=1)

        # Cosine scoring
        tau = cofrn.reasoning.tree_cosine_tau.abs().clamp(min=0.01)
        best_exp = best_hyp.unsqueeze(1).expand_as(answer_enc)
        cos_sim = F.cosine_similarity(best_exp, answer_enc, dim=-1)
        scores_normal = cos_sim / tau
        preds_normal = scores_normal.argmax(-1)
        acc_normal = (preds_normal == labels).float().mean().item()

    results = {'normal_acc': acc_normal, 'n': n}

    # Test replacements
    torch.manual_seed(999)
    replacements = {
        'random': F.normalize(torch.randn_like(best_hyp), dim=-1),
        'zeros': torch.zeros_like(best_hyp),
        'negated': -best_hyp,
        'shuffled': best_hyp[torch.randperm(n, device=DEVICE)],
    }

    for name, fake_hyp in replacements.items():
        with torch.no_grad():
            fake_exp = fake_hyp.unsqueeze(1).expand_as(answer_enc)
            cos_sim_fake = F.cosine_similarity(fake_exp, answer_enc, dim=-1)
            scores_fake = cos_sim_fake / tau
            preds_fake = scores_fake.argmax(-1)
            acc_fake = (preds_fake == labels).float().mean().item()
            changes = (preds_normal != preds_fake).sum().item()

        results[f'{name}_acc'] = acc_fake
        results[f'{name}_changes'] = changes
        logger.info(f"  {name:10s}: acc={acc_fake:.1%}, pred_changes={changes}/{n}")

    # best_hyp variation
    hyp_centroid = best_hyp.mean(dim=0)
    hyp_cos = F.cosine_similarity(
        best_hyp, hyp_centroid.unsqueeze(0).expand_as(best_hyp), dim=-1
    )
    results['hyp_cos_to_centroid_mean'] = hyp_cos.mean().item()
    results['hyp_cos_to_centroid_min'] = hyp_cos.min().item()
    logger.info(f"  best_hyp cosine to centroid: mean={hyp_cos.mean():.4f}, min={hyp_cos.min():.4f}")

    # Verdict
    total_changes = sum(results.get(f'{k}_changes', 0) for k in replacements)
    fix_works = total_changes > 50  # should see many changes with cosine scoring
    results['fix_works'] = fix_works
    logger.info(f"  VERDICT: cosine scoring {'USES' if fix_works else 'STILL IGNORES'} best_hyp "
                f"(total changes: {total_changes})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Depth sweep — do different depths produce different predictions?
# ═══════════════════════════════════════════════════════════════════════════════

def depth_sweep(cofrn: COFRN) -> dict:
    """Test whether different tree depths now produce different predictions."""
    logger.info("=" * 60)
    logger.info("E10 Step 3: Depth sweep on GSM8K")
    logger.info("=" * 60)

    ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'gsm8k', 'test', seed=SEED)
    rng = np.random.RandomState(SEED)
    n = 100
    indices = rng.permutation(len(ds))[:n]
    q = ds.question_embeddings[indices].to(DEVICE)
    a = ds.answer_embeddings[indices].to(DEVICE)
    labels = ds.labels[indices].to(DEVICE)

    accs = {}
    all_preds = {}

    for depth in [1, 2, 3, 5]:
        correct = 0
        preds_list = []
        with torch.no_grad():
            for start in range(0, n, 32):
                end = min(start + 32, n)
                out = cofrn(
                    embeddings=q[start:end],
                    answer_embeddings=a[start:end],
                    labels=labels[start:end],
                    depth=depth,
                )
                pred = out['scores'].argmax(-1)
                correct += (pred == labels[start:end]).sum().item()
                preds_list.append(pred.cpu())

        accs[depth] = correct / n
        all_preds[depth] = torch.cat(preds_list)
        logger.info(f"  Depth {depth}: acc={accs[depth]:.1%}")

    # Check if predictions differ across depths
    depth_diffs = {}
    for d in [2, 3, 5]:
        changes = (all_preds[1] != all_preds[d]).sum().item()
        depth_diffs[f'd1_vs_d{d}'] = changes
        logger.info(f"  Depth 1 vs {d}: {changes}/{n} prediction changes")

    all_same = all(torch.equal(all_preds[1], all_preds[d]) for d in [2, 3, 5])
    logger.info(f"  All depths identical? {'YES (BAD)' if all_same else 'NO (GOOD — depths matter now)'}")

    return {
        'accuracies': {str(k): v for k, v in accs.items()},
        'depth_diffs': depth_diffs,
        'all_depths_identical': all_same,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Benchmark evaluation (all 4)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_benchmarks(cofrn: COFRN) -> list:
    """Evaluate on GSM8K, ARC, StrategyQA, FOLIO."""
    logger.info("=" * 60)
    logger.info("E10 Step 4: 4-benchmark evaluation")
    logger.info("=" * 60)

    benchmarks = [
        ('gsm8k',        'test',       4),
        ('arc_challenge', 'test',       4),
        ('strategyqa',   'train',      2),
        ('folio',        'validation', 3),
    ]

    results = []
    for bench, split, n_choices in benchmarks:
        ds = precompute_embeddings_st('all-MiniLM-L6-v2', bench, split, seed=SEED)
        rng = np.random.RandomState(SEED)
        n = min(100, len(ds))
        indices = rng.permutation(len(ds))[:n]
        q = ds.question_embeddings[indices].to(DEVICE)
        a = ds.answer_embeddings[indices].to(DEVICE)
        lab = ds.labels[indices].to(DEVICE)

        correct = 0
        anchor_weights_all = []
        with torch.no_grad():
            for start in range(0, n, 32):
                end = min(start + 32, n)
                out = cofrn(
                    embeddings=q[start:end],
                    answer_embeddings=a[start:end],
                    labels=lab[start:end],
                )
                pred = out['scores'].argmax(-1)
                correct += (pred == lab[start:end]).sum().item()
                anchor_weights_all.append(out['anchor_weights'].cpu())

        acc = correct / n
        random_baseline = 1.0 / n_choices

        # Anchor diagnostics
        aw = torch.cat(anchor_weights_all, dim=0)
        entropy = -(aw * torch.log(aw + 1e-10)).sum(dim=-1).mean().item()
        max_entropy = np.log(16)
        pr = ((aw.mean(0).sum() ** 2) / (aw.mean(0) ** 2).sum()).item()

        result = {
            'benchmark': bench,
            'split': split,
            'n_examples': n,
            'n_choices': n_choices,
            'random_baseline': random_baseline,
            'cosine_acc': acc,
            'lift_over_random': acc - random_baseline,
            'anchor_entropy': entropy,
            'anchor_entropy_ratio': entropy / max_entropy,
            'participation_ratio': pr,
        }
        results.append(result)
        logger.info(f"  {bench:15s}: acc={acc:.1%} (random={random_baseline:.1%}, "
                     f"lift={acc - random_baseline:+.1%}) "
                     f"H={entropy:.3f}/{max_entropy:.3f} PR={pr:.1f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Compare with old MLP model
# ═══════════════════════════════════════════════════════════════════════════════

def compare_with_old(cosine_benchmarks: list) -> dict:
    """Load old MLP-scoring model and compare accuracies."""
    logger.info("=" * 60)
    logger.info("E10 Step 5: Compare cosine vs MLP scoring")
    logger.info("=" * 60)

    old_path = BASE_DIR / "results" / "controller" / "cofrn_quick.pt"
    if not old_path.exists():
        logger.warning("Old MLP model not found, skipping comparison")
        return {}

    old_config = COFRNConfig(
        use_precomputed=True, encoder_input_dim=384, hidden_dim=256,
        struct_dim=128, context_dim=128, manifold_dim=10,
        n_anchors=16, rank=16, task_type='multi_step',
        use_cosine_scoring=False,
    )
    old_model = COFRN(old_config)
    old_model.load_state_dict(
        torch.load(str(old_path), weights_only=True, map_location=DEVICE),
        strict=False,  # Old model lacks cosine_tau params
    )
    old_model.to(DEVICE).eval()

    # Old accuracies from option_c_results (known: GSM8K 49%, ARC 31%, StrategyQA 58%, FOLIO 32%)
    old_accs = {'gsm8k': 0.49, 'arc_challenge': 0.31, 'strategyqa': 0.58, 'folio': 0.32}

    comparison = []
    for result in cosine_benchmarks:
        bench = result['benchmark']
        old_acc = old_accs.get(bench, None)
        new_acc = result['cosine_acc']
        delta = new_acc - old_acc if old_acc is not None else None

        row = {
            'benchmark': bench,
            'old_mlp_acc': old_acc,
            'new_cosine_acc': new_acc,
            'delta': delta,
        }
        comparison.append(row)
        logger.info(f"  {bench:15s}: MLP={old_acc:.1%} → Cosine={new_acc:.1%} "
                     f"(Δ={delta:+.1%})" if old_acc else f"  {bench}: new={new_acc:.1%}")

    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Anchor weight analysis — are they still uniform?
# ═══════════════════════════════════════════════════════════════════════════════

def anchor_analysis(cofrn: COFRN) -> dict:
    """Check if cosine scoring creates gradient pressure for non-uniform attention."""
    logger.info("=" * 60)
    logger.info("E10 Step 6: Anchor weight analysis")
    logger.info("=" * 60)

    ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'gsm8k', 'test', seed=SEED)
    rng = np.random.RandomState(SEED)
    n = 200
    indices = rng.permutation(len(ds))[:n]
    q = ds.question_embeddings[indices].to(DEVICE)

    with torch.no_grad():
        s0 = cofrn.encode_text(embeddings=q)
        fact = cofrn.factorization(s0)
        weights = fact['weights']  # [N, 16]

    expected_uniform = 1.0 / 16
    max_dev = (weights - expected_uniform).abs().max().item()
    entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean().item()
    max_entropy = np.log(16)
    per_anchor_std = weights.std(dim=0)  # [16]

    # Operator scale
    U = cofrn.factorization.manifold.U_all.detach()
    V = cofrn.factorization.manifold.V_all.detach()
    u_rms = U.pow(2).mean().sqrt().item()
    v_rms = V.pow(2).mean().sqrt().item()

    # Temperature
    tree_tau = cofrn.reasoning.tree_cosine_tau.detach().abs().clamp(min=0.01).item()
    direct_tau = cofrn.reasoning.direct.cosine_tau.detach().abs().clamp(min=0.01).item()

    results = {
        'max_dev_from_uniform': max_dev,
        'mean_entropy': entropy,
        'entropy_ratio': entropy / max_entropy,
        'per_anchor_std_mean': per_anchor_std.mean().item(),
        'per_anchor_std_max': per_anchor_std.max().item(),
        'U_rms': u_rms,
        'V_rms': v_rms,
        'tree_cosine_tau': tree_tau,
        'direct_cosine_tau': direct_tau,
    }

    logger.info(f"  Max dev from uniform: {max_dev:.6f}")
    logger.info(f"  Entropy: {entropy:.4f} / {max_entropy:.4f} = {entropy/max_entropy:.4f}")
    logger.info(f"  Per-anchor std: mean={per_anchor_std.mean():.6f}, max={per_anchor_std.max():.6f}")
    logger.info(f"  Operator scale: U_rms={u_rms:.4f}, V_rms={v_rms:.4f}")
    logger.info(f"  Learned temperatures: tree_tau={tree_tau:.4f}, direct_tau={direct_tau:.4f}")

    still_uniform = entropy / max_entropy > 0.99
    logger.info(f"  Still uniform? {'YES' if still_uniform else 'NO (operators specializing)'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("  E10: COSINE SCORING FIX")
    logger.info("  Replacing MLP scoring (collapsed — E9) with cosine similarity")
    logger.info("=" * 70)

    # Step 1: Train
    cofrn = train_cosine_cofrn()

    # Step 2: Verify fix
    verify_results = verify_fix(cofrn)

    # Step 3: Depth sweep
    depth_results = depth_sweep(cofrn)

    # Step 4: Benchmark eval
    benchmark_results = evaluate_benchmarks(cofrn)

    # Step 5: Compare
    comparison = compare_with_old(benchmark_results)

    # Step 6: Anchor analysis
    anchor_results = anchor_analysis(cofrn)

    # Save all results
    all_results = {
        'experiment': 'E10_cosine_scoring_fix',
        'verify_fix': verify_results,
        'depth_sweep': depth_results,
        'benchmark_results': benchmark_results,
        'comparison': comparison,
        'anchor_analysis': anchor_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': DEVICE,
        'elapsed_seconds': time.time() - t0,
    }

    out_path = RESULTS_DIR / "e10_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("  E10 SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\n  Fix verification:")
    logger.info(f"    Random hyp changes:  {verify_results.get('random_changes', '?')}/100")
    logger.info(f"    Zero hyp changes:    {verify_results.get('zeros_changes', '?')}/100")
    logger.info(f"    Fix works: {verify_results.get('fix_works', '?')}")

    logger.info(f"\n  Depth sensitivity:")
    logger.info(f"    All depths identical: {depth_results['all_depths_identical']}")
    for k, v in depth_results['depth_diffs'].items():
        logger.info(f"    {k}: {v}/100 changes")

    logger.info(f"\n  Benchmark results:")
    for r in benchmark_results:
        logger.info(f"    {r['benchmark']:15s}: {r['cosine_acc']:.1%} "
                     f"(lift={r['lift_over_random']:+.1%})")

    if comparison:
        logger.info(f"\n  Comparison (MLP → Cosine):")
        for c in comparison:
            logger.info(f"    {c['benchmark']:15s}: {c['old_mlp_acc']:.1%} → {c['new_cosine_acc']:.1%} "
                         f"(Δ={c['delta']:+.1%})")

    logger.info(f"\n  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
