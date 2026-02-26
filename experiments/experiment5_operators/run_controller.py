#!/usr/bin/env python3
"""
GeometricReasoningController — Training and Diagnostic Pipeline.

Phase 0: Quick-train COFRN on GSM8K + ARC (~5 min) for meaningful manifold
Phase 1: Train TypeClassifier on ARTI v3 dataset manifold coordinates (~30-60s)
Phase 2: End-to-end pipeline test with 12 canonical examples
Phase 3: Routing diagnostics — participation ratio, type distribution, route verification

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_controller.py
"""

import sys
import time
import json
import logging
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Path setup
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR.parent))
sys.path.insert(0, str(BASE_DIR))

from shared.model import COFRN, COFRNConfig
from shared.controller import (
    GeometricReasoningController, ControllerConfig, TypeClassifier,
    TYPE_MERGE_MAP, CORE_TYPE_NAMES, N_CORE_TYPES, merge_labels,
)
from shared.reasoning_types import ReasoningType, REASONING_TYPES, TYPE_SHORT_NAMES
from shared.train_utils import TrainConfig, save_results
from shared.data_utils import BenchmarkDataset, collate_benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "controller"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0: Quick-train COFRN
# ═══════════════════════════════════════════════════════════════════════════════

def phase0_quick_train_cofrn() -> COFRN:
    """
    Train COFRN on real GSM8K + ARC sentence-transformer embeddings
    (all-MiniLM-L6-v2, 384D) so the manifold produces meaningful coordinates.

    Uses the same encoder as the ARTI v3 dataset for consistency.
    Falls back to random embeddings if datasets/encoder unavailable.
    """
    logger.info("=" * 60)
    logger.info("Phase 0: Train COFRN on real benchmark data")
    logger.info("=" * 60)

    saved_path = RESULTS_DIR / "cofrn_quick.pt"

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

    # Check for cached model
    if saved_path.exists():
        logger.info(f"Loading cached COFRN from {saved_path}")
        model = COFRN(config)
        model.load_state_dict(
            torch.load(str(saved_path), weights_only=True, map_location=DEVICE)
        )
        model.to(DEVICE)
        model.eval()
        logger.info(f"  Loaded (multi_step). Trainable params: {model.trainable_params:,}")
        return model

    # --- Load real benchmark data with sentence-transformer ---
    has_real_data = False
    try:
        from shared.data_utils import precompute_embeddings_st, MixedDomainDataset

        logger.info("Loading benchmarks with all-MiniLM-L6-v2 (384D)...")

        gsm8k_ds = precompute_embeddings_st(
            'all-MiniLM-L6-v2', 'gsm8k', 'train', seed=SEED,
        )
        logger.info(f"  GSM8K: {len(gsm8k_ds)} examples, "
                     f"emb_dim={gsm8k_ds.question_embeddings.shape[1]}")

        arc_ds = precompute_embeddings_st(
            'all-MiniLM-L6-v2', 'arc_challenge', 'train', seed=SEED,
        )
        logger.info(f"  ARC:   {len(arc_ds)} examples, "
                     f"emb_dim={arc_ds.question_embeddings.shape[1]}")

        # Combine into mixed domain dataset (handles padding if n_choices differ)
        mixed_ds = MixedDomainDataset({'gsm8k': gsm8k_ds, 'arc_challenge': arc_ds})
        n = len(mixed_ds)
        logger.info(f"  Combined: {n} examples")

        # Train/val split
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
        has_real_data = True
        logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
        logger.warning("Falling back to random embeddings (manifold will be weak)")

    if not has_real_data:
        logger.info("Training with random embeddings (fallback)...")
        n_train, n_val, n_answers = 800, 200, 4
        encoder_dim = config.encoder_input_dim  # 384
        torch.manual_seed(SEED)
        train_ds = BenchmarkDataset(
            question_embeddings=torch.randn(n_train, encoder_dim),
            answer_embeddings=torch.randn(n_train, n_answers, encoder_dim),
            labels=torch.randint(0, n_answers, (n_train,)),
            domain='mixed',
        )
        val_ds = BenchmarkDataset(
            question_embeddings=torch.randn(n_val, encoder_dim),
            answer_embeddings=torch.randn(n_val, n_answers, encoder_dim),
            labels=torch.randint(0, n_answers, (n_val,)),
            domain='mixed',
        )

    model = COFRN(config).to(DEVICE)
    logger.info(f"  COFRN trainable params: {model.trainable_params:,}")
    logger.info(f"  task_type: {config.task_type} (HilbertTree trained)")

    # Use curriculum training: depth 1 → 2 → 3
    # This trains both the direct scorer AND the HilbertTree beam search
    from shared.train_utils import (
        curriculum_train, TrainConfig, CurriculumPhase,
    )

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

    # Shorter curriculum than E1 default (speed: ~3-5 min total)
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
        model_name='cofrn_multistep',
    )

    best_acc = train_result.get('best_val_acc', 0.0)
    logger.info(f"  Curriculum complete. Best val acc: {best_acc:.2%}")

    # Save
    torch.save(model.state_dict(), str(saved_path))
    logger.info(f"  Saved to {saved_path}")

    model.to(DEVICE)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Train TypeClassifier
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_train_type_classifier(cofrn: COFRN) -> TypeClassifier:
    """
    Train TypeClassifier on manifold coordinates from ARTI v3 dataset.

    Steps:
    1. Load ARTI v3 dataset (7,500 traces with 10-type labels)
    2. Apply TYPE_MERGE_MAP to get 6 core labels
    3. Encode all traces through trained COFRN -> manifold coordinates [N, 10]
    4. Train TypeClassifier: AdamW, 50 epochs, cross-entropy
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Train TypeClassifier on manifold coordinates")
    logger.info("=" * 60)

    saved_path = RESULTS_DIR / "type_clf.pt"

    # Load ARTI v3 dataset
    dataset_path = BASE_DIR / "results" / "arti_v3" / "dataset.pt"
    if not dataset_path.exists():
        logger.warning(f"ARTI v3 dataset not found at {dataset_path}")
        logger.info("Generating synthetic manifold data for TypeClassifier training...")
        return _train_type_clf_synthetic(cofrn, saved_path)

    logger.info(f"Loading ARTI v3 dataset from {dataset_path}")
    data = torch.load(str(dataset_path), weights_only=False)
    embeddings = data['embeddings']  # [N, 384]
    labels_10 = data['labels']  # [N] (10-type)
    texts = data.get('texts', [])

    logger.info(f"  {len(embeddings)} traces, {embeddings.shape[1]}D embeddings")

    # Merge to 6 core types
    labels_6 = merge_labels(labels_10)
    dist = Counter(labels_6.numpy().tolist())
    logger.info(f"  6-type distribution: {dict(sorted(dist.items()))}")

    # Extract s0 features (256D projected embeddings, before factorization)
    # s0 retains the full semantic signal; post-factorization features are
    # shaped by MI discriminator for QA, not reasoning type discrimination
    logger.info("Extracting s0 features (256D, pre-factorization) through COFRN...")
    cofrn.eval()
    all_features = []

    with torch.no_grad():
        for start in range(0, len(embeddings), 64):
            batch_emb = embeddings[start:start+64].to(DEVICE)
            s0 = cofrn.encode_text(embeddings=batch_emb)  # [B, 256]
            all_features.append(s0.cpu())

    s0_features = torch.cat(all_features, dim=0)  # [N, 256]
    logger.info(f"  s0 features: {s0_features.shape}")

    # Train TypeClassifier on s0 features
    return _train_type_clf(s0_features, labels_6, saved_path)


def _train_type_clf_synthetic(cofrn: COFRN, saved_path: Path) -> TypeClassifier:
    """Fallback: train on synthetic s0 features when ARTI dataset unavailable."""
    logger.info("Using synthetic data for TypeClassifier training")
    torch.manual_seed(SEED)

    n_per_type = 500
    struct_dim = 256  # Match COFRN hidden_dim (s0 before factorization)

    # Generate cluster centers and samples
    centers = torch.randn(N_CORE_TYPES, struct_dim) * 0.5
    all_features = []
    all_labels = []

    for t in range(N_CORE_TYPES):
        noise = torch.randn(n_per_type, struct_dim) * 0.3
        features = centers[t].unsqueeze(0) + noise
        all_features.append(features)
        all_labels.append(torch.full((n_per_type,), t, dtype=torch.long))

    structural_features = torch.cat(all_features, dim=0)
    labels_6 = torch.cat(all_labels, dim=0)

    return _train_type_clf(structural_features, labels_6, saved_path)


def _train_type_clf(
    features: torch.Tensor,
    labels_6: torch.Tensor,
    saved_path: Path,
) -> TypeClassifier:
    """Train TypeClassifier on structural features with 6 core labels."""

    # Check for cached model
    if saved_path.exists():
        logger.info(f"Loading cached TypeClassifier from {saved_path}")
        clf = TypeClassifier(input_dim=features.shape[1], n_classes=N_CORE_TYPES)
        clf.load_state_dict(
            torch.load(str(saved_path), weights_only=True, map_location=DEVICE)
        )
        clf.to(DEVICE)
        clf.eval()
        # Quick accuracy check
        with torch.no_grad():
            probs, pred, conf = clf(features.to(DEVICE))
            acc = (pred == labels_6.to(DEVICE)).float().mean().item()
        logger.info(f"  Loaded. Overall accuracy: {acc:.1%}")
        return clf

    n = len(features)

    # Train/val split (80/20, stratified)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_coords = features[train_idx].to(DEVICE)
    train_labels = labels_6[train_idx].to(DEVICE)
    val_coords = features[val_idx].to(DEVICE)
    val_labels = labels_6[val_idx].to(DEVICE)

    logger.info(f"  Train: {len(train_coords)}, Val: {len(val_coords)}")

    # Class weights for imbalanced data
    counts = torch.bincount(train_labels, minlength=N_CORE_TYPES).float()
    class_weights = (1.0 / (counts + 1.0))
    class_weights = class_weights / class_weights.sum() * N_CORE_TYPES

    # Train
    clf = TypeClassifier(
        input_dim=features.shape[1],
        n_classes=N_CORE_TYPES,
    ).to(DEVICE)
    logger.info(f"  TypeClassifier params: {clf.trainable_params}")

    optimizer = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    best_val_acc = 0.0
    best_state = None
    n_epochs = 50
    batch_size = 128

    for epoch in range(n_epochs):
        clf.train()
        perm = torch.randperm(len(train_coords), device=DEVICE)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_coords), batch_size):
            idx = perm[start:start+batch_size]
            logits = clf.mlp(train_coords[idx])
            loss = criterion(logits, train_labels[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        clf.eval()
        with torch.no_grad():
            _, pred, conf = clf(val_coords)
            val_acc = (pred == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: loss={total_loss/max(n_batches,1):.4f}, "
                         f"val_acc={val_acc:.1%}")

    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.to(DEVICE)
    clf.eval()

    # Per-type accuracy
    with torch.no_grad():
        _, pred, conf = clf(val_coords)
        val_acc = (pred == val_labels).float().mean().item()
        logger.info(f"\n  Overall val accuracy: {val_acc:.1%} (target: 65-78%)")

        for t in range(N_CORE_TYPES):
            mask = val_labels == t
            if mask.sum() > 0:
                t_acc = (pred[mask] == t).float().mean().item()
                logger.info(f"    {CORE_TYPE_NAMES[t]:>15}: {t_acc:.1%} ({mask.sum().item()} samples)")

    # Save
    torch.save(clf.state_dict(), str(saved_path))
    logger.info(f"  Saved to {saved_path}")

    return clf


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: End-to-End Pipeline Test
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical examples: 2 per core type
CANONICAL_EXAMPLES = [
    # CauseEffect (0)
    ("The temperature dropped below freezing, causing the pipes to burst.",
     0, "CauseEffect"),
    ("She raised prices by 30%, which halved sales volume.",
     0, "CauseEffect"),
    # Deduction (1)
    ("All mammals are warm-blooded. Whales are mammals. Therefore, whales are warm-blooded.",
     1, "Deduction"),
    ("If it rains, the ground gets wet. It is raining. Therefore, the ground is wet.",
     1, "Deduction"),
    # Induction (2)
    ("Every patient given drug X improved within 3 days. Evidence suggests drug X is effective.",
     2, "Induction"),
    ("Sales increased in Q1, Q2, and Q3 after the campaign. The campaign consistently drives growth.",
     2, "Induction"),
    # Analogy (3)
    ("A firewall protects a network just as a moat protects a castle.",
     3, "Analogy"),
    ("The atom has electrons orbiting a nucleus, just as planets orbit the sun.",
     3, "Analogy"),
    # Conservation (4)
    ("$500 transferred from checking to savings. The total remains the same.",
     4, "Conservation"),
    ("Mass is conserved: atoms in equals atoms out.",
     4, "Conservation"),
    # Counterfactual (5)
    ("What if the bridge had been inspected? The failure would have been prevented.",
     5, "Counterfactual"),
    ("The best explanation for the wet lawn is overnight dew.",
     5, "Counterfactual"),  # Abduction -> merged to Counterfactual
]


def phase2_pipeline_test(
    cofrn: COFRN,
    type_clf: TypeClassifier,
):
    """
    Run 12 canonical examples through the full GeometricReasoningController pipeline.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: End-to-End Pipeline Test")
    logger.info("=" * 60)

    config = ControllerConfig(
        hidden_dim=cofrn.config.hidden_dim,
        manifold_dim=cofrn.config.manifold_dim,
        struct_dim=cofrn.config.struct_dim,
    )

    controller = GeometricReasoningController(
        config=config,
        factorization=cofrn.factorization,
        reasoning_engine=cofrn.reasoning,
    )

    # Load trained TypeClassifier weights
    controller.type_clf.load_state_dict(type_clf.state_dict())
    controller.to(DEVICE)
    controller.eval()

    logger.info(f"  Controller total params: {controller.trainable_params:,}")

    # We need to encode texts via the encoder to get s0 vectors
    # Since COFRN uses precomputed embeddings, we simulate with sentence-transformers
    # or use random embeddings to demonstrate the pipeline structure
    try:
        from shared.encoder import SentenceTransformerEncoder
        encoder = SentenceTransformerEncoder(
            model_name='all-MiniLM-L6-v2', hidden_dim=256, load_pretrained=True,
        )
        has_encoder = True
        logger.info("  Using real sentence-transformer encoder")
    except Exception:
        has_encoder = False
        logger.info("  Using random embeddings (sentence-transformers not available)")

    n_answers = 4
    results_table = []

    print("\n" + "=" * 130)
    print(f"{'Text':<55} {'Expected':>14} {'Detected':>14} {'Conf':>6} {'Route':>8} "
          f"{'Depth':>5} {'Delta':>6} {'TopAnchors':>12} {'AncEnt':>7}")
    print("-" * 130)

    for text, expected_type, expected_name in CANONICAL_EXAMPLES:
        with torch.no_grad():
            if has_encoder:
                raw_emb = encoder.encode_texts([text])  # [1, 384]
                s0 = cofrn.encode_text(embeddings=raw_emb.to(DEVICE))
            else:
                torch.manual_seed(hash(text) % (2**32))
                raw_emb = torch.randn(1, 384)
                s0 = cofrn.encode_text(embeddings=raw_emb.to(DEVICE))

            # Random answer embeddings for demonstration
            # Controller's answer_proj expects hidden_dim (256), not encoder_input_dim
            answer_emb = torch.randn(1, n_answers, config.hidden_dim, device=DEVICE)

            result = controller.forward_inference(s0, answer_emb)

        detected = result['detected_type'][0].item()
        detected_name = CORE_TYPE_NAMES[detected]
        conf = result['confidence'][0].item()
        route = "struct" if result['route_decisions'][0].item() == 1 else "fast"
        depth, delta = controller.get_type_config(detected)

        # Top anchors
        aw = result['anchor_weights'][0]
        topk_vals, topk_idx = aw.topk(config.top_k_anchors)
        top_anchors_str = ",".join([str(i.item()) for i in topk_idx])

        # Anchor entropy
        aw_np = aw.cpu().numpy()
        aw_np = aw_np / (aw_np.sum() + 1e-10)
        anchor_ent = -np.sum(aw_np * np.log(aw_np + 1e-10))

        match = "ok" if detected == expected_type else "MISS"
        short_text = text[:53] + ".." if len(text) > 55 else text

        print(f"{short_text:<55} {expected_name:>14} {detected_name:>14} {conf:>5.1%} "
              f"{route:>8} {depth:>5} {delta:>6.2f} {top_anchors_str:>12} {anchor_ent:>7.3f}  {match}")

        results_table.append({
            'text': text,
            'expected_type': expected_name,
            'detected_type': detected_name,
            'confidence': conf,
            'route': route,
            'depth': depth,
            'delta': delta,
            'top_anchors': topk_idx.tolist(),
            'anchor_entropy': float(anchor_ent),
            'match': detected == expected_type,
        })

    print("=" * 130)

    # Summary
    n_correct = sum(1 for r in results_table if r['match'])
    n_total = len(results_table)
    n_structured = sum(1 for r in results_table if r['route'] == 'struct')
    logger.info(f"\n  Type detection: {n_correct}/{n_total} correct ({n_correct/n_total:.0%})")
    logger.info(f"  Routing: {n_structured}/{n_total} structured, {n_total - n_structured}/{n_total} fast")

    # Save results
    save_results(
        {'phase2_results': results_table, 'accuracy': n_correct / n_total},
        str(RESULTS_DIR / "phase2_pipeline.json"),
    )

    return controller, results_table


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Routing Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_routing_diagnostics(
    controller: GeometricReasoningController,
    cofrn: COFRN,
):
    """
    Diagnostic analysis of the controller's routing behavior.

    Checks:
    1. Participation ratio before/after masking
    2. Type distribution of routing decisions
    3. Structured types actually trigger tree, fast types don't
    """
    logger.info("=" * 60)
    logger.info("Phase 3: Routing Diagnostics")
    logger.info("=" * 60)

    controller.eval()
    config = controller.config
    n_samples = 200
    n_answers = 4
    torch.manual_seed(SEED)

    # Generate diverse random inputs
    random_embs = torch.randn(n_samples, 384, device=DEVICE)
    answer_emb = torch.randn(n_samples, n_answers, config.hidden_dim, device=DEVICE)

    with torch.no_grad():
        s0 = cofrn.encode_text(embeddings=random_embs)
        result = controller.forward_inference(s0, answer_emb)

    # Get diagnostics
    diag = controller.get_diagnostics(result)

    print("\n--- Routing Diagnostics ---")
    print(f"  Mean confidence:  {diag['mean_confidence']:.3f}")
    print(f"  Min confidence:   {diag['min_confidence']:.3f}")
    print(f"  Mean type entropy: {diag['mean_type_entropy']:.3f}")
    print(f"  % structured:    {diag.get('pct_structured', 0):.1%}")

    # Anchor participation ratio
    aw = result['anchor_weights']
    mean_aw = aw.mean(dim=0)
    pr_before = ((mean_aw.sum() ** 2) / ((mean_aw ** 2).sum() + 1e-10)).item()

    # After top-k masking
    k = config.top_k_anchors
    topk_vals, topk_idx = aw.topk(k, dim=-1)
    mask = torch.zeros_like(aw)
    mask.scatter_(1, topk_idx, 1.0)
    masked_aw = aw * mask
    masked_aw = masked_aw / (masked_aw.sum(dim=-1, keepdim=True) + 1e-10)
    mean_masked = masked_aw.mean(dim=0)
    pr_after = ((mean_masked.sum() ** 2) / ((mean_masked ** 2).sum() + 1e-10)).item()

    print(f"\n--- Anchor Participation ---")
    print(f"  Before masking: PR = {pr_before:.2f} / {config.n_core_types}")
    print(f"  After top-{k}:   PR = {pr_after:.2f} / {k}")

    # Type distribution
    detected = result['detected_type']
    route = result['route_decisions']

    print(f"\n--- Per-Type Routing ---")
    print(f"  {'Type':<16} {'Count':>6} {'% Fast':>8} {'% Struct':>8} {'Expected':>10}")
    structured_set = set(config.structured_types)

    type_results = {}
    for t in range(N_CORE_TYPES):
        mask = detected == t
        count = mask.sum().item()
        if count > 0:
            fast_pct = (route[mask] == 0).float().mean().item()
            struct_pct = (route[mask] == 1).float().mean().item()
        else:
            fast_pct = struct_pct = 0.0
        expected = "struct" if t in structured_set else "fast"
        primary = "struct" if struct_pct > fast_pct else "fast"
        status = "ok" if primary == expected else "MISMATCH"
        print(f"  {CORE_TYPE_NAMES[t]:<16} {count:>6} {fast_pct:>7.0%} {struct_pct:>8.0%} "
              f"{expected:>10}  {status}")
        type_results[CORE_TYPE_NAMES[t]] = {
            'count': count,
            'fast_pct': fast_pct,
            'struct_pct': struct_pct,
            'expected': expected,
            'correct_route': primary == expected,
        }

    # Confidence breakdown by route
    fast_mask = route == 0
    struct_mask = route == 1
    if fast_mask.sum() > 0 and struct_mask.sum() > 0:
        fast_conf = result['confidence'][fast_mask].mean().item()
        struct_conf = result['confidence'][struct_mask].mean().item()
        print(f"\n--- Confidence by Route ---")
        print(f"  Fast path:       {fast_conf:.3f} (n={fast_mask.sum().item()})")
        print(f"  Structured path: {struct_conf:.3f} (n={struct_mask.sum().item()})")

    # Save diagnostics
    save_results(
        {
            'diagnostics': diag,
            'participation_ratio_before': pr_before,
            'participation_ratio_after': pr_after,
            'type_routing': type_results,
        },
        str(RESULTS_DIR / "phase3_diagnostics.json"),
    )

    return diag


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("GeometricReasoningController — Full Pipeline")
    logger.info(f"Device: {DEVICE}")
    t_start = time.time()

    # Phase 0: Quick-train COFRN
    cofrn = phase0_quick_train_cofrn()

    # Phase 1: Train TypeClassifier
    type_clf = phase1_train_type_classifier(cofrn)

    # Phase 2: End-to-end pipeline
    controller, pipeline_results = phase2_pipeline_test(cofrn, type_clf)

    # Phase 3: Routing diagnostics
    diagnostics = phase3_routing_diagnostics(controller, cofrn)

    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"All phases complete in {elapsed:.1f}s")
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
