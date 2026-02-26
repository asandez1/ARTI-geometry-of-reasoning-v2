#!/usr/bin/env python3
"""
E6: Rank Ablation Experiment.

Tests whether increasing operator rank closes the detection-application gap.

The CO-FRN ContinuousOperatorManifold uses low-rank operators O_i = I + U_i V_i^T
where U_i, V_i are (hidden_dim, rank). Current default rank=16 produces 0% accuracy
lift from type-specific routing. This experiment tests whether the bottleneck is
insufficient operator expressiveness.

Ranks tested: {4, 8, 16, 32, 64}
Benchmarks: GSM8K (train), ARC Challenge (train)  -> eval on test splits
Seeds: 3 per configuration
Metrics: accuracy, parameter count, anchor entropy, anchor utilization

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_rank_ablation.py
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
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from shared.model import COFRN, COFRNConfig
from shared.train_utils import TrainConfig, curriculum_train, CurriculumPhase, save_results
from shared.data_utils import (
    precompute_embeddings_st, BenchmarkDataset, MixedDomainDataset,
    collate_benchmark,
)
from shared.controller import (
    GeometricReasoningController, ControllerConfig, TypeClassifier,
    CORE_TYPE_NAMES, N_CORE_TYPES,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "rank_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANKS = [4, 8, 16, 32, 64]
SEEDS = [42, 123, 456]
N_EVAL = 200  # examples per benchmark for evaluation


def load_training_data(seed: int):
    """Load GSM8K + ARC training data with sentence-transformer embeddings."""
    logger.info(f"Loading training data (seed={seed})...")

    gsm8k_ds = precompute_embeddings_st(
        'all-MiniLM-L6-v2', 'gsm8k', 'train', seed=seed,
    )
    arc_ds = precompute_embeddings_st(
        'all-MiniLM-L6-v2', 'arc_challenge', 'train', seed=seed,
    )

    mixed_ds = MixedDomainDataset({'gsm8k': gsm8k_ds, 'arc_challenge': arc_ds})
    n = len(mixed_ds)

    # Train/val split
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

    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds


def load_test_data(benchmark: str, split: str, n_examples: int, seed: int):
    """Load test data for evaluation."""
    ds = precompute_embeddings_st('all-MiniLM-L6-v2', benchmark, split, seed=seed)
    n = min(n_examples, len(ds))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(ds))[:n]

    q_emb = ds.question_embeddings[indices]
    a_emb = ds.answer_embeddings[indices]
    labels = ds.labels[indices]
    return q_emb, a_emb, labels


def train_cofrn_at_rank(rank: int, train_ds, val_ds, seed: int) -> COFRN:
    """Train CO-FRN with a specific operator rank."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = COFRNConfig(
        use_precomputed=True,
        encoder_input_dim=384,
        hidden_dim=256,
        struct_dim=128,
        context_dim=128,
        manifold_dim=10,
        n_anchors=16,
        rank=rank,
        task_type='multi_step',
    )

    model = COFRN(config).to(DEVICE)

    # Count operator params specifically
    op_params = model.factorization.manifold.U_all.numel() + \
                model.factorization.manifold.V_all.numel()

    logger.info(f"  Rank={rank}: total trainable={model.trainable_params:,}, "
                f"operator params={op_params:,}")

    # Curriculum training (same as Phase 0 in run_controller.py)
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
        CurriculumPhase(depth=1, epochs=5, lambda_factorize=0.01,
                        lambda_coherence=0.0, learning_rate=3e-4),
        CurriculumPhase(depth=2, epochs=5, lambda_factorize=0.05,
                        lambda_coherence=0.005, learning_rate=1e-4),
        CurriculumPhase(depth=3, epochs=10, lambda_factorize=0.1,
                        lambda_coherence=0.01, learning_rate=5e-5),
    ]

    model, train_result = curriculum_train(
        model, train_ds, val_ds, base_cfg,
        phases=phases,
        model_name=f'cofrn_rank{rank}_s{seed}',
    )

    model.eval()
    return model, train_result


@torch.no_grad()
def evaluate_cofrn(model: COFRN, q_emb, a_emb, labels):
    """Evaluate CO-FRN accuracy and collect anchor diagnostics."""
    model.eval()
    q_emb = q_emb.to(DEVICE)
    a_emb = a_emb.to(DEVICE)
    labels = labels.to(DEVICE)
    n = len(labels)

    total_correct = 0
    all_weights = []

    for start in range(0, n, 32):
        end = min(start + 32, n)
        out = model(
            embeddings=q_emb[start:end],
            answer_embeddings=a_emb[start:end],
            labels=labels[start:end],
        )
        pred = out['scores'].argmax(dim=-1)
        total_correct += (pred == labels[start:end]).sum().item()
        all_weights.append(out['anchor_weights'].cpu())

    accuracy = total_correct / n

    # Anchor diagnostics
    all_weights = torch.cat(all_weights, dim=0)
    mean_w = all_weights.mean(dim=0)
    mean_w = mean_w / (mean_w.sum() + 1e-10)

    entropy = -(mean_w * torch.log(mean_w + 1e-10)).sum().item()
    max_entropy = np.log(model.config.n_anchors)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Participation ratio
    pr = ((mean_w.sum() ** 2) / ((mean_w ** 2).sum() + 1e-10)).item()

    # Max weight concentration
    max_w = mean_w.max().item()

    return {
        'accuracy': accuracy,
        'anchor_entropy': entropy,
        'anchor_norm_entropy': norm_entropy,
        'anchor_participation_ratio': pr,
        'anchor_max_weight': max_w,
    }


@torch.no_grad()
def evaluate_with_routing(model: COFRN, type_clf: TypeClassifier,
                          q_emb, a_emb, labels, ctrl_config: ControllerConfig):
    """Evaluate with type-specific depth routing (controller-enabled)."""
    model.eval()
    type_clf.eval()
    q_emb = q_emb.to(DEVICE)
    a_emb = a_emb.to(DEVICE)
    labels = labels.to(DEVICE)
    n = len(labels)

    total_correct = 0
    structured_set = set(ctrl_config.structured_types)
    operator_fn = model.factorization.manifold.forward

    for start in range(0, n, 32):
        end = min(start + 32, n)
        batch_q = q_emb[start:end]
        batch_a = a_emb[start:end]
        batch_l = labels[start:end]
        B = batch_q.shape[0]

        s0 = model.encode_text(embeddings=batch_q)
        fact_out = model.factorization(s0)
        structural = fact_out['structural']
        transformed = fact_out['transformed']
        anchor_weights = fact_out['weights']
        answer_enc = model.encode_answers(batch_a)

        # Detect type from s0
        type_probs, detected_type, confidence = type_clf(s0)

        for i in range(B):
            t = detected_type[i].item()
            depth = ctrl_config.type_depths.get(t, 2)
            delta = ctrl_config.type_deltas.get(t, 0.8)

            old_delta = model.reasoning.tree.delta
            model.reasoning.tree.delta = delta

            out = model.reasoning.forward_multistep(
                transformed=transformed[i:i+1],
                evidence=transformed[i:i+1],
                operator_fn=operator_fn,
                structural=structural[i:i+1],
                answer_encodings=answer_enc[i:i+1],
                depth=depth,
            )
            model.reasoning.tree.delta = old_delta

            pred = out['scores'].argmax(dim=-1)
            if pred.item() == batch_l[i].item():
                total_correct += 1

    return total_correct / n


def train_type_classifier(model: COFRN, seed: int) -> TypeClassifier:
    """Quick-train TypeClassifier on ARTI v3 dataset through this model's encoder."""
    dataset_path = BASE_DIR / "results" / "arti_v3" / "dataset.pt"

    if dataset_path.exists():
        data = torch.load(str(dataset_path), weights_only=False)
        embeddings = data['embeddings']
        labels_10 = data['labels']
    else:
        # Synthetic fallback
        logger.warning("ARTI v3 dataset not found, using synthetic data")
        torch.manual_seed(seed)
        n_per_type = 500
        centers = torch.randn(N_CORE_TYPES, 256) * 0.5
        all_features = []
        all_labels = []
        for t in range(N_CORE_TYPES):
            noise = torch.randn(n_per_type, 256) * 0.3
            all_features.append(centers[t].unsqueeze(0) + noise)
            all_labels.append(torch.full((n_per_type,), t, dtype=torch.long))
        s0_features = torch.cat(all_features, dim=0)
        labels_6 = torch.cat(all_labels, dim=0)
        return _train_clf(s0_features, labels_6, seed)

    # Extract s0 features through model's encoder
    from shared.controller import merge_labels
    labels_6 = merge_labels(labels_10)

    model.eval()
    all_features = []
    with torch.no_grad():
        for start in range(0, len(embeddings), 64):
            batch_emb = embeddings[start:start+64].to(DEVICE)
            s0 = model.encode_text(embeddings=batch_emb)
            all_features.append(s0.cpu())
    s0_features = torch.cat(all_features, dim=0)

    return _train_clf(s0_features, labels_6, seed)


def _train_clf(features, labels_6, seed):
    """Train TypeClassifier on features."""
    n = len(features)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    split = int(0.8 * n)

    train_f = features[indices[:split]].to(DEVICE)
    train_l = labels_6[indices[:split]].to(DEVICE)
    val_f = features[indices[split:]].to(DEVICE)
    val_l = labels_6[indices[split:]].to(DEVICE)

    counts = torch.bincount(train_l, minlength=N_CORE_TYPES).float()
    class_weights = (1.0 / (counts + 1.0))
    class_weights = class_weights / class_weights.sum() * N_CORE_TYPES

    clf = TypeClassifier(input_dim=features.shape[1], n_classes=N_CORE_TYPES).to(DEVICE)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    best_val_acc = 0.0
    best_state = None

    for epoch in range(50):
        clf.train()
        perm = torch.randperm(len(train_f), device=DEVICE)
        for start in range(0, len(train_f), 128):
            idx = perm[start:start+128]
            logits = clf.mlp(train_f[idx])
            loss = criterion(logits, train_l[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        clf.eval()
        with torch.no_grad():
            _, pred, _ = clf(val_f)
            val_acc = (pred == val_l).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}

    if best_state:
        clf.load_state_dict(best_state)
    clf.to(DEVICE).eval()

    logger.info(f"  TypeClassifier val acc: {best_val_acc:.1%}")
    return clf


def main():
    logger.info("=" * 65)
    logger.info("E6: Rank Ablation Experiment")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Ranks: {RANKS}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info("=" * 65)
    t_start = time.time()

    all_results = []

    # Load test data once (shared across all configs)
    logger.info("Loading test data...")
    gsm8k_test = load_test_data('gsm8k', 'test', N_EVAL, seed=42)
    arc_test = load_test_data('arc_challenge', 'test', N_EVAL, seed=42)

    for rank in RANKS:
        logger.info(f"\n{'='*65}")
        logger.info(f"  RANK = {rank}")
        logger.info(f"{'='*65}")

        rank_results = {
            'rank': rank,
            'seeds': [],
        }

        for seed in SEEDS:
            logger.info(f"\n  --- Seed {seed} ---")
            torch.manual_seed(seed)

            # Load training data
            train_ds, val_ds = load_training_data(seed)

            # Train COFRN at this rank
            model, train_result = train_cofrn_at_rank(rank, train_ds, val_ds, seed)

            # Count parameters
            op_params = model.factorization.manifold.U_all.numel() + \
                        model.factorization.manifold.V_all.numel()
            total_params = model.trainable_params

            # Evaluate on GSM8K
            gsm8k_result = evaluate_cofrn(model, *gsm8k_test)
            logger.info(f"  GSM8K: {gsm8k_result['accuracy']:.1%}")

            # Evaluate on ARC
            arc_result = evaluate_cofrn(model, *arc_test)
            logger.info(f"  ARC:   {arc_result['accuracy']:.1%}")

            # Train TypeClassifier for this model
            type_clf = train_type_classifier(model, seed)

            # Evaluate with type-specific routing
            ctrl_config = ControllerConfig(
                hidden_dim=256, manifold_dim=10, struct_dim=128,
            )

            gsm8k_routed = evaluate_with_routing(
                model, type_clf, *gsm8k_test, ctrl_config)
            arc_routed = evaluate_with_routing(
                model, type_clf, *arc_test, ctrl_config)

            gsm8k_lift = gsm8k_routed - gsm8k_result['accuracy']
            arc_lift = arc_routed - arc_result['accuracy']
            logger.info(f"  GSM8K routed: {gsm8k_routed:.1%} (lift: {gsm8k_lift:+.1%})")
            logger.info(f"  ARC   routed: {arc_routed:.1%} (lift: {arc_lift:+.1%})")

            seed_result = {
                'seed': seed,
                'total_trainable_params': total_params,
                'operator_params': op_params,
                'best_val_acc': train_result['best_val_acc'],
                'epochs_trained': train_result['epochs_trained'],
                'gsm8k': {
                    'baseline_acc': gsm8k_result['accuracy'],
                    'routed_acc': gsm8k_routed,
                    'lift': gsm8k_lift,
                    'anchor_entropy': gsm8k_result['anchor_norm_entropy'],
                    'anchor_pr': gsm8k_result['anchor_participation_ratio'],
                },
                'arc_challenge': {
                    'baseline_acc': arc_result['accuracy'],
                    'routed_acc': arc_routed,
                    'lift': arc_lift,
                    'anchor_entropy': arc_result['anchor_norm_entropy'],
                    'anchor_pr': arc_result['anchor_participation_ratio'],
                },
            }
            rank_results['seeds'].append(seed_result)

            # Free memory
            del model, type_clf
            torch.cuda.empty_cache() if DEVICE == 'cuda' else None

        # Compute rank-level aggregates
        gsm8k_accs = [s['gsm8k']['baseline_acc'] for s in rank_results['seeds']]
        arc_accs = [s['arc_challenge']['baseline_acc'] for s in rank_results['seeds']]
        gsm8k_lifts = [s['gsm8k']['lift'] for s in rank_results['seeds']]
        arc_lifts = [s['arc_challenge']['lift'] for s in rank_results['seeds']]

        rank_results['aggregate'] = {
            'operator_params': rank_results['seeds'][0]['operator_params'],
            'total_params': rank_results['seeds'][0]['total_trainable_params'],
            'gsm8k_acc_mean': np.mean(gsm8k_accs),
            'gsm8k_acc_std': np.std(gsm8k_accs),
            'arc_acc_mean': np.mean(arc_accs),
            'arc_acc_std': np.std(arc_accs),
            'gsm8k_lift_mean': np.mean(gsm8k_lifts),
            'gsm8k_lift_std': np.std(gsm8k_lifts),
            'arc_lift_mean': np.mean(arc_lifts),
            'arc_lift_std': np.std(arc_lifts),
        }
        all_results.append(rank_results)

    # ─── Summary Table ────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  E6: RANK ABLATION SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Rank':>5} {'OpParams':>10} {'TotalParams':>12} "
          f"{'GSM8K':>12} {'ARC':>12} {'GSM8K Lift':>12} {'ARC Lift':>12}")
    print(f"  {'-'*85}")

    for r in all_results:
        agg = r['aggregate']
        print(f"  {r['rank']:>5} {agg['operator_params']:>10,} {agg['total_params']:>12,} "
              f"{agg['gsm8k_acc_mean']:>5.1%}+/-{agg['gsm8k_acc_std']:.1%} "
              f"{agg['arc_acc_mean']:>5.1%}+/-{agg['arc_acc_std']:.1%} "
              f"{agg['gsm8k_lift_mean']:>+5.1%}+/-{agg['gsm8k_lift_std']:.1%} "
              f"{agg['arc_lift_mean']:>+5.1%}+/-{agg['arc_lift_std']:.1%}")

    print(f"\n  Key question: Does higher rank produce non-zero routing lift?")

    # Anchor entropy comparison
    print(f"\n  {'Rank':>5} {'GSM8K Entropy':>15} {'ARC Entropy':>15} "
          f"{'GSM8K PR':>10} {'ARC PR':>10}")
    print(f"  {'-'*60}")
    for r in all_results:
        seeds = r['seeds']
        ge = np.mean([s['gsm8k']['anchor_entropy'] for s in seeds])
        ae = np.mean([s['arc_challenge']['anchor_entropy'] for s in seeds])
        gp = np.mean([s['gsm8k']['anchor_pr'] for s in seeds])
        ap = np.mean([s['arc_challenge']['anchor_pr'] for s in seeds])
        print(f"  {r['rank']:>5} {ge:>15.3f} {ae:>15.3f} {gp:>10.2f} {ap:>10.2f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save
    save_path = RESULTS_DIR / "rank_ablation_results.json"
    save_results(all_results, str(save_path))
    logger.info(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
