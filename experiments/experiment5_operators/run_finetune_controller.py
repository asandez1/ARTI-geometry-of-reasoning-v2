#!/usr/bin/env python3
"""
Option C: Fine-Tune TypeClassifier + route_gate on Benchmark Data.

Tests whether fine-tuning the controller's routing components on
pseudo-labeled benchmark data can improve type detection and downstream
accuracy on non-math benchmarks.

Protocol:
    1. Pseudo-label all benchmark questions via HeuristicLabeler → 10-type
       → merge to 6 core types
    2. Fine-tune TypeClassifier (16,774 params):
       - Start from E5 pre-trained weights
       - Train on 50% pseudo-labeled benchmark + 50% ARTI data
       - 30 epochs, lr=5e-4, confidence-weighted loss
    3. Fine-tune route_gate (~145 params):
       - Train with task loss (cross-entropy on blended scores)
       - 20 epochs, lr=1e-3
    4. Evaluate on 100 StrategyQA + 100 FOLIO

Comparison:
    - TypeClassifier accuracy on benchmark pseudo-labels: target > 76.7%
    - Benchmark accuracy before/after fine-tuning
    - Anchor entropy before/after

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_finetune_controller.py
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
import numpy as np

# Path setup
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from shared.model import COFRN, COFRNConfig
from shared.controller import (
    GeometricReasoningController, ControllerConfig, TypeClassifier,
    TYPE_MERGE_MAP, CORE_TYPE_NAMES, N_CORE_TYPES, merge_labels,
)
from shared.reasoning_types import (
    ReasoningType, HeuristicLabeler, REASONING_TYPES, TYPE_SHORT_NAMES,
)
from shared.data_utils import precompute_embeddings_st, BenchmarkDataset
from shared.train_utils import save_results
from shared.metrics import anchor_utilization_entropy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "finetune_controller"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONTROLLER_DIR = BASE_DIR / "results" / "controller"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Load Models
# ═══════════════════════════════════════════════════════════════════════════════

def load_models():
    """Load pre-trained COFRN and controller (from run_controller.py)."""
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

    model_path = CONTROLLER_DIR / "cofrn_quick.pt"
    clf_path = CONTROLLER_DIR / "type_clf.pt"

    for p in [model_path, clf_path]:
        if not p.exists():
            logger.error(f"Required file not found: {p}")
            logger.error("Run `python experiments/experiment5_operators/run_controller.py` first.")
            sys.exit(1)

    cofrn = COFRN(config)
    cofrn.load_state_dict(
        torch.load(str(model_path), weights_only=True, map_location=DEVICE)
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

    controller.type_clf.load_state_dict(
        torch.load(str(clf_path), weights_only=True, map_location=DEVICE)
    )
    controller.to(DEVICE).eval()
    logger.info(f"Controller loaded: TypeClassifier {controller.type_clf.trainable_params:,} params")

    return cofrn, controller, ctrl_config


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Pseudo-Label Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def pseudo_label_benchmark(
    benchmark: str,
    split: str,
) -> dict:
    """
    Pseudo-label benchmark questions using HeuristicLabeler.

    Returns dict with:
        - questions: list of question texts
        - labels_10: [N] 10-type labels
        - labels_6: [N] 6-type labels
        - confidences: [N] labeling confidence
        - distribution: per-type counts
    """
    from shared.data_utils import LOADERS
    import inspect

    loader_fn = LOADERS[benchmark]
    kwargs = {'split': split}
    if 'seed' in inspect.signature(loader_fn).parameters:
        kwargs['seed'] = SEED
    questions, choices, labels = loader_fn(**kwargs)

    labeler = HeuristicLabeler()
    labels_10 = []
    confidences = []

    for q in questions:
        rtype, conf = labeler.label_hard(q)
        labels_10.append(int(rtype))
        confidences.append(conf)

    labels_10_t = torch.tensor(labels_10, dtype=torch.long)
    labels_6_t = merge_labels(labels_10_t)
    confidences_t = torch.tensor(confidences, dtype=torch.float)

    dist_6 = Counter(labels_6_t.numpy().tolist())

    logger.info(f"  Pseudo-labeled {benchmark}/{split}: {len(questions)} examples")
    logger.info(f"    6-type distribution: {dict(sorted(dist_6.items()))}")
    logger.info(f"    Mean confidence: {confidences_t.mean():.3f}")

    return {
        'questions': questions,
        'labels_10': labels_10_t,
        'labels_6': labels_6_t,
        'confidences': confidences_t,
        'distribution': dict(sorted(dist_6.items())),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Fine-Tune TypeClassifier
# ═══════════════════════════════════════════════════════════════════════════════

def finetune_type_classifier(
    cofrn: COFRN,
    controller: GeometricReasoningController,
    pseudo_data: dict,
) -> TypeClassifier:
    """
    Fine-tune TypeClassifier on pseudo-labeled benchmark data + ARTI data.

    Protocol:
        - 50% pseudo-labeled benchmark data (confidence-weighted)
        - 50% ARTI data (if available, else synthetic)
        - 30 epochs, lr=5e-4
        - Start from E5 pre-trained weights
    """
    logger.info(f"\n{'='*60}")
    logger.info("  Step 2: Fine-Tune TypeClassifier")
    logger.info(f"{'='*60}")

    # Get s0 features for pseudo-labeled data
    questions = pseudo_data['questions']
    labels_6 = pseudo_data['labels_6']
    confidences = pseudo_data['confidences']

    # Encode questions to s0 features
    logger.info("  Encoding benchmark questions to s0 features...")
    try:
        from shared.encoder import SentenceTransformerEncoder
        st_enc = SentenceTransformerEncoder(
            model_name='all-MiniLM-L6-v2', hidden_dim=256, load_pretrained=True,
        )
        raw_embs = st_enc.encode_texts(questions, batch_size=64)
        del st_enc
    except Exception:
        logger.warning("  Could not encode texts, using random embeddings")
        torch.manual_seed(SEED)
        raw_embs = torch.randn(len(questions), 384)

    # Get s0 from COFRN encoder
    cofrn.eval()
    all_s0 = []
    with torch.no_grad():
        for start in range(0, len(raw_embs), 64):
            batch = raw_embs[start:start+64].to(DEVICE)
            s0 = cofrn.encode_text(embeddings=batch)
            all_s0.append(s0.cpu())
    bench_s0 = torch.cat(all_s0, dim=0)  # [N_bench, 256]
    logger.info(f"  Benchmark s0 features: {bench_s0.shape}")

    # Load ARTI data (if available)
    arti_path = BASE_DIR / "results" / "arti_v3" / "dataset.pt"
    has_arti = False
    if arti_path.exists():
        logger.info(f"  Loading ARTI data from {arti_path}")
        arti_data = torch.load(str(arti_path), weights_only=False)
        arti_embs = arti_data['embeddings']
        arti_labels_10 = arti_data['labels']
        arti_labels_6 = merge_labels(arti_labels_10)

        # Get s0 features for ARTI data
        arti_s0_list = []
        with torch.no_grad():
            for start in range(0, len(arti_embs), 64):
                batch = arti_embs[start:start+64].to(DEVICE)
                s0 = cofrn.encode_text(embeddings=batch)
                arti_s0_list.append(s0.cpu())
        arti_s0 = torch.cat(arti_s0_list, dim=0)
        has_arti = True
        logger.info(f"  ARTI s0 features: {arti_s0.shape}")
    else:
        logger.info("  ARTI data not found, generating synthetic data")
        torch.manual_seed(SEED + 1)
        n_synth = len(bench_s0)
        centers = torch.randn(N_CORE_TYPES, 256) * 0.5
        arti_s0_parts = []
        arti_labels_parts = []
        per_type = n_synth // N_CORE_TYPES
        for t in range(N_CORE_TYPES):
            noise = torch.randn(per_type, 256) * 0.3
            arti_s0_parts.append(centers[t].unsqueeze(0) + noise)
            arti_labels_parts.append(torch.full((per_type,), t, dtype=torch.long))
        arti_s0 = torch.cat(arti_s0_parts)
        arti_labels_6 = torch.cat(arti_labels_parts)
        has_arti = True

    # Balance: subsample to 50/50
    n_bench = len(bench_s0)
    n_arti = len(arti_s0)
    n_target = min(n_bench, n_arti)

    rng = np.random.RandomState(SEED)
    bench_idx = rng.permutation(n_bench)[:n_target]
    arti_idx = rng.permutation(n_arti)[:n_target]

    # Combined dataset
    combined_s0 = torch.cat([bench_s0[bench_idx], arti_s0[arti_idx]], dim=0)
    combined_labels = torch.cat([labels_6[bench_idx], arti_labels_6[arti_idx]], dim=0)

    # Confidence weights: benchmark data gets confidence weight, ARTI gets 1.0
    bench_weights = confidences[bench_idx]
    arti_weights = torch.ones(len(arti_idx))
    combined_weights = torch.cat([bench_weights, arti_weights], dim=0)

    # Shuffle
    perm = rng.permutation(len(combined_s0))
    combined_s0 = combined_s0[perm]
    combined_labels = combined_labels[perm]
    combined_weights = combined_weights[perm]

    logger.info(f"  Combined dataset: {len(combined_s0)} examples "
                f"({n_target} bench + {n_target} ARTI)")

    # Train/val split (80/20)
    n_total = len(combined_s0)
    split_pt = int(0.8 * n_total)
    train_s0 = combined_s0[:split_pt].to(DEVICE)
    train_labels = combined_labels[:split_pt].to(DEVICE)
    train_weights = combined_weights[:split_pt].to(DEVICE)
    val_s0 = combined_s0[split_pt:].to(DEVICE)
    val_labels = combined_labels[split_pt:].to(DEVICE)

    logger.info(f"  Train: {len(train_s0)}, Val: {len(val_s0)}")

    # Evaluate BEFORE fine-tuning
    controller.type_clf.eval()
    with torch.no_grad():
        _, pred_before, conf_before = controller.type_clf(val_s0)
        acc_before = (pred_before == val_labels).float().mean().item()
    logger.info(f"  TypeClassifier accuracy BEFORE fine-tuning: {acc_before:.1%}")

    # Fine-tune
    clf = controller.type_clf
    optimizer = torch.optim.AdamW(clf.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    batch_size = 128
    n_epochs = 30
    best_val_acc = acc_before
    best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}

    for epoch in range(n_epochs):
        clf.train()
        perm_t = torch.randperm(len(train_s0), device=DEVICE)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_s0), batch_size):
            idx = perm_t[start:start + batch_size]
            logits = clf.mlp(train_s0[idx])
            # Confidence-weighted cross-entropy
            per_sample_loss = F.cross_entropy(logits, train_labels[idx], reduction='none')
            weighted_loss = (per_sample_loss * train_weights[idx]).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        clf.eval()
        with torch.no_grad():
            _, pred, conf = clf(val_s0)
            val_acc = (pred == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}/{n_epochs}: loss={total_loss/max(n_batches,1):.4f} "
                        f"val_acc={val_acc:.1%}")

    # Restore best
    clf.load_state_dict(best_state)
    clf.to(DEVICE).eval()

    # Evaluate AFTER
    with torch.no_grad():
        _, pred_after, conf_after = clf(val_s0)
        acc_after = (pred_after == val_labels).float().mean().item()

    logger.info(f"\n  TypeClassifier accuracy AFTER fine-tuning: {acc_after:.1%}")
    logger.info(f"  Improvement: {acc_after - acc_before:+.1%}")

    # Per-type accuracy
    for t in range(N_CORE_TYPES):
        mask = val_labels == t
        if mask.sum() > 0:
            t_acc = (pred_after[mask] == t).float().mean().item()
            logger.info(f"    {CORE_TYPE_NAMES[t]:>15}: {t_acc:.1%} ({mask.sum().item()} samples)")

    # Save
    save_path = RESULTS_DIR / "type_clf_finetuned.pt"
    torch.save(clf.state_dict(), str(save_path))
    logger.info(f"  Saved to {save_path}")

    return clf, {
        'acc_before': acc_before,
        'acc_after': acc_after,
        'improvement': acc_after - acc_before,
        'n_train': len(train_s0),
        'n_val': len(val_s0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Fine-Tune route_gate
# ═══════════════════════════════════════════════════════════════════════════════

def finetune_route_gate(
    cofrn: COFRN,
    controller: GeometricReasoningController,
) -> dict:
    """
    Fine-tune the route_gate (~145 params) with task loss on benchmark data.

    Uses the controller's blended forward pass (soft blend of fast/structured),
    training only the route_gate parameters while keeping everything else frozen.
    """
    logger.info(f"\n{'='*60}")
    logger.info("  Step 3: Fine-Tune route_gate")
    logger.info(f"{'='*60}")

    gate_params = sum(p.numel() for p in controller.route_gate.parameters())
    logger.info(f"  route_gate params: {gate_params}")

    # Load training data
    try:
        gsm8k_ds = precompute_embeddings_st(
            'all-MiniLM-L6-v2', 'gsm8k', 'train', seed=SEED,
        )
        arc_ds = precompute_embeddings_st(
            'all-MiniLM-L6-v2', 'arc_challenge', 'train', seed=SEED,
        )
    except Exception as e:
        logger.error(f"  Could not load data: {e}")
        return {'status': 'failed', 'error': str(e)}

    # Subsample for speed
    rng = np.random.RandomState(SEED)
    n_gsm = min(1000, len(gsm8k_ds))
    n_arc = min(1000, len(arc_ds))
    gsm_idx = rng.permutation(len(gsm8k_ds))[:n_gsm]
    arc_idx = rng.permutation(len(arc_ds))[:n_arc]

    # Pad ARC answers to match GSM8K n_choices if needed
    gsm_q = gsm8k_ds.question_embeddings[gsm_idx]
    gsm_a = gsm8k_ds.answer_embeddings[gsm_idx]
    gsm_l = gsm8k_ds.labels[gsm_idx]

    arc_q = arc_ds.question_embeddings[arc_idx]
    arc_a = arc_ds.answer_embeddings[arc_idx]
    arc_l = arc_ds.labels[arc_idx]

    # Ensure same n_choices
    max_choices = max(gsm_a.shape[1], arc_a.shape[1])
    if gsm_a.shape[1] < max_choices:
        pad = torch.zeros(gsm_a.shape[0], max_choices - gsm_a.shape[1], gsm_a.shape[2])
        gsm_a = torch.cat([gsm_a, pad], dim=1)
    if arc_a.shape[1] < max_choices:
        pad = torch.zeros(arc_a.shape[0], max_choices - arc_a.shape[1], arc_a.shape[2])
        arc_a = torch.cat([arc_a, pad], dim=1)

    train_q = torch.cat([gsm_q, arc_q], dim=0)
    train_a = torch.cat([gsm_a, arc_a], dim=0)
    train_l = torch.cat([gsm_l, arc_l], dim=0)

    logger.info(f"  Training data: {len(train_q)} examples")

    # Freeze everything except route_gate
    for param in controller.parameters():
        param.requires_grad = False
    for param in controller.route_gate.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        controller.route_gate.parameters(),
        lr=1e-3, weight_decay=0.01,
    )

    batch_size = 32
    n_epochs = 20

    # Evaluate BEFORE
    controller.eval()
    routing_weights_before = []
    with torch.no_grad():
        for start in range(0, min(200, len(train_q)), batch_size):
            end = min(start + batch_size, len(train_q))
            q = train_q[start:end].to(DEVICE)
            s0 = cofrn.encode_text(embeddings=q)
            fact_out = cofrn.factorization(s0)
            tp, dt, conf, _ = controller.detect_type(s0, fact_out['structural'])
            rw = controller.compute_routing(tp, conf, fact_out['weights'])
            routing_weights_before.append(rw.cpu())
    rw_before = torch.cat(routing_weights_before).squeeze(-1)
    mean_rw_before = rw_before.mean().item()
    logger.info(f"  Mean routing_weight BEFORE: {mean_rw_before:.3f} "
                f"(0=fast, 1=structured)")

    # Fine-tune
    for epoch in range(n_epochs):
        controller.train()
        perm = torch.randperm(len(train_q))
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for start in range(0, len(train_q), batch_size):
            idx = perm[start:start + batch_size]
            q = train_q[idx].to(DEVICE)
            a = train_a[idx].to(DEVICE)
            l = train_l[idx].to(DEVICE)

            # Get s0 from COFRN encoder and project answers to 256D
            with torch.no_grad():
                s0 = cofrn.encode_text(embeddings=q)
                # COFRN answer_proj: Linear(384→256), controller expects 256D
                answer_enc = cofrn.encode_answers(a)  # [B, n_ans, 256]

            result = controller(
                s0=s0,
                answer_embeddings=answer_enc,
                labels=l,
            )

            loss = result['task_loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.route_gate.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(idx)
            if 'correct' in result:
                total_correct += result['correct'].sum().item()
            total_seen += len(idx)

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)

        if (epoch + 1) % 5 == 0:
            logger.info(f"    Epoch {epoch+1}/{n_epochs}: loss={train_loss:.4f} "
                        f"acc={train_acc:.1%}")

    # Unfreeze all (restore original state for later eval)
    for param in controller.parameters():
        param.requires_grad = True

    # Evaluate AFTER
    controller.eval()
    routing_weights_after = []
    with torch.no_grad():
        for start in range(0, min(200, len(train_q)), batch_size):
            end = min(start + batch_size, len(train_q))
            q = train_q[start:end].to(DEVICE)
            s0 = cofrn.encode_text(embeddings=q)
            fact_out = cofrn.factorization(s0)
            tp, dt, conf, _ = controller.detect_type(s0, fact_out['structural'])
            rw = controller.compute_routing(tp, conf, fact_out['weights'])
            routing_weights_after.append(rw.cpu())
    rw_after = torch.cat(routing_weights_after).squeeze(-1)
    mean_rw_after = rw_after.mean().item()
    logger.info(f"  Mean routing_weight AFTER: {mean_rw_after:.3f}")
    logger.info(f"  Δ routing_weight: {mean_rw_after - mean_rw_before:+.3f}")

    # Save
    save_path = RESULTS_DIR / "route_gate_finetuned.pt"
    torch.save(controller.route_gate.state_dict(), str(save_path))

    return {
        'mean_rw_before': mean_rw_before,
        'mean_rw_after': mean_rw_after,
        'delta_rw': mean_rw_after - mean_rw_before,
        'gate_params': gate_params,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Evaluate on Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_controller(
    cofrn: COFRN,
    controller: GeometricReasoningController,
    ctrl_config: ControllerConfig,
    benchmark: str,
    split: str,
    n_examples: int,
    label: str = "",
) -> dict:
    """
    Evaluate controller on a benchmark slice.

    Uses the same evaluation logic as run_benchmark.py but with the
    fine-tuned controller.
    """
    ds = precompute_embeddings_st('all-MiniLM-L6-v2', benchmark, split, seed=SEED)
    n_available = len(ds)
    n = min(n_examples, n_available)

    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n_available)[:n]

    q_emb = ds.question_embeddings[indices].to(DEVICE)
    a_emb = ds.answer_embeddings[indices].to(DEVICE)
    labels = ds.labels[indices].to(DEVICE)

    n_choices = a_emb.shape[1]
    random_baseline = 1.0 / n_choices

    # COFRN baseline
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

    # Controller (type-specific routing)
    controller.eval()
    ctrl_correct = 0
    type_counts = Counter()
    route_counts = Counter()
    all_anchor_weights = []
    structured_set = set(ctrl_config.structured_types)

    with torch.no_grad():
        for start in range(0, n, 32):
            end = min(start + 32, n)
            batch_q = q_emb[start:end]
            batch_a = a_emb[start:end]
            batch_l = labels[start:end]
            B = batch_q.shape[0]

            s0 = cofrn.encode_text(embeddings=batch_q)
            fact_out = cofrn.factorization(s0)
            structural = fact_out['structural']
            transformed = fact_out['transformed']
            anchor_weights = fact_out['weights']
            all_anchor_weights.append(anchor_weights.cpu())

            type_probs, detected_type, confidence, _ = \
                controller.detect_type(s0, structural)

            answer_enc = cofrn.encode_answers(batch_a)
            operator_fn = cofrn.factorization.manifold.forward

            for i in range(B):
                t = detected_type[i].item()
                t_name = CORE_TYPE_NAMES[t]
                type_counts[t_name] += 1

                depth, delta = controller.get_type_config(t)
                route_label = 'struct' if t in structured_set else 'fast'
                route_counts[route_label] += 1

                old_delta = cofrn.reasoning.tree.delta
                cofrn.reasoning.tree.delta = delta

                out = cofrn.reasoning.forward_multistep(
                    transformed=transformed[i:i+1],
                    evidence=transformed[i:i+1],
                    operator_fn=operator_fn,
                    structural=structural[i:i+1],
                    answer_encodings=answer_enc[i:i+1],
                    depth=depth,
                )
                cofrn.reasoning.tree.delta = old_delta

                pred = out['scores'].argmax(dim=-1)
                if pred.item() == batch_l[i].item():
                    ctrl_correct += 1

    ctrl_acc = ctrl_correct / n
    lift = ctrl_acc - cofrn_acc

    # Anchor entropy
    aw = torch.cat(all_anchor_weights, dim=0).numpy()
    ent_stats = anchor_utilization_entropy(aw, n_anchors=16)

    return {
        'benchmark': benchmark,
        'split': split,
        'label': label,
        'n_examples': n,
        'n_choices': n_choices,
        'random_baseline': random_baseline,
        'cofrn_acc': cofrn_acc,
        'controller_acc': ctrl_acc,
        'lift': lift,
        'type_distribution': dict(type_counts),
        'routing': dict(route_counts),
        'anchor_entropy': ent_stats['entropy'],
        'participation_ratio': ent_stats['participation_ratio'],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 65)
    logger.info("  Option C: Fine-Tune TypeClassifier + route_gate")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 65)
    t0 = time.time()

    # Load models
    cofrn, controller, ctrl_config = load_models()

    # ─── Evaluate BEFORE fine-tuning ──────────────────────────────────
    benchmarks = [
        ('gsm8k', 'test', 100),
        ('arc_challenge', 'test', 100),
        ('strategyqa', 'train', 100),
        ('folio', 'validation', 100),
    ]

    logger.info("\n--- Evaluating BEFORE fine-tuning ---")
    results_before = []
    for bm, split, n in benchmarks:
        try:
            r = evaluate_controller(cofrn, controller, ctrl_config, bm, split, n,
                                    label='before')
            results_before.append(r)
            logger.info(f"  {bm:<15}: COFRN={r['cofrn_acc']:.1%} "
                        f"Ctrl={r['controller_acc']:.1%} "
                        f"lift={r['lift']:+.1%} "
                        f"ent={r['anchor_entropy']:.3f}")
        except Exception as e:
            logger.error(f"  {bm}: FAILED ({e})")

    # ─── Step 1: Pseudo-label ─────────────────────────────────────────
    logger.info("\n--- Step 1: Pseudo-label benchmarks ---")
    all_pseudo_data = {}
    pseudo_benchmarks = [
        ('gsm8k', 'train'),
        ('arc_challenge', 'train'),
        ('strategyqa', 'train'),
        ('folio', 'validation'),
    ]
    for bm, split in pseudo_benchmarks:
        try:
            all_pseudo_data[bm] = pseudo_label_benchmark(bm, split)
        except Exception as e:
            logger.error(f"  {bm}: FAILED ({e})")

    # Merge all pseudo-labeled questions
    all_questions = []
    all_labels_6 = []
    all_confidences = []
    for bm, data in all_pseudo_data.items():
        all_questions.extend(data['questions'])
        all_labels_6.append(data['labels_6'])
        all_confidences.append(data['confidences'])

    if all_labels_6:
        merged_pseudo = {
            'questions': all_questions,
            'labels_6': torch.cat(all_labels_6),
            'confidences': torch.cat(all_confidences),
        }
        logger.info(f"  Total pseudo-labeled: {len(all_questions)}")
    else:
        logger.error("No pseudo-labeled data available")
        return

    # ─── Step 2: Fine-tune TypeClassifier ─────────────────────────────
    clf_finetuned, clf_results = finetune_type_classifier(
        cofrn, controller, merged_pseudo,
    )

    # ─── Step 3: Fine-tune route_gate ─────────────────────────────────
    gate_results = finetune_route_gate(cofrn, controller)

    # ─── Step 4: Evaluate AFTER fine-tuning ───────────────────────────
    logger.info("\n--- Evaluating AFTER fine-tuning ---")
    results_after = []
    for bm, split, n in benchmarks:
        try:
            r = evaluate_controller(cofrn, controller, ctrl_config, bm, split, n,
                                    label='after')
            results_after.append(r)
            logger.info(f"  {bm:<15}: COFRN={r['cofrn_acc']:.1%} "
                        f"Ctrl={r['controller_acc']:.1%} "
                        f"lift={r['lift']:+.1%} "
                        f"ent={r['anchor_entropy']:.3f}")
        except Exception as e:
            logger.error(f"  {bm}: FAILED ({e})")

    # ─── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  OPTION C RESULTS: Fine-Tune TypeClassifier + route_gate")
    print(f"{'='*80}")

    print(f"\n  TypeClassifier:")
    print(f"    Accuracy before: {clf_results['acc_before']:.1%}")
    print(f"    Accuracy after:  {clf_results['acc_after']:.1%}")
    print(f"    Improvement:     {clf_results['improvement']:+.1%}")

    if gate_results.get('status') != 'failed':
        print(f"\n  route_gate:")
        print(f"    Mean routing_weight before: {gate_results['mean_rw_before']:.3f}")
        print(f"    Mean routing_weight after:  {gate_results['mean_rw_after']:.3f}")
        print(f"    Δ routing_weight:           {gate_results['delta_rw']:+.3f}")

    print(f"\n  {'Benchmark':<15} {'Random':>7} {'COFRN':>7} "
          f"{'Before':>8} {'After':>8} {'Δ':>7}")
    print(f"  {'-'*55}")

    for rb, ra in zip(results_before, results_after):
        bm = rb['benchmark']
        delta = ra['controller_acc'] - rb['controller_acc']
        print(f"  {bm:<15} {rb['random_baseline']:>6.1%} {rb['cofrn_acc']:>6.1%} "
              f"{rb['controller_acc']:>7.1%} {ra['controller_acc']:>7.1%} "
              f"{delta:>+6.1%}")

    # Anchor entropy comparison
    print(f"\n  Anchor Entropy:")
    for rb, ra in zip(results_before, results_after):
        bm = rb['benchmark']
        print(f"    {bm:<15}: before={rb['anchor_entropy']:.3f} "
              f"after={ra['anchor_entropy']:.3f} "
              f"Δ={ra['anchor_entropy'] - rb['anchor_entropy']:+.3f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save all results
    save_results(
        {
            'clf_results': clf_results,
            'gate_results': gate_results,
            'results_before': results_before,
            'results_after': results_after,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': DEVICE,
        },
        str(RESULTS_DIR / "option_c_results.json"),
    )
    logger.info(f"Results saved to {RESULTS_DIR / 'option_c_results.json'}")


if __name__ == "__main__":
    main()
