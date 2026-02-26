#!/usr/bin/env python3
"""
E13 Gradient Diagnostic — Verify encoder weight changes and gradient flow.

Three checks:
  1. Do encoder weights change after 1 epoch of training? (L2 diff)
  2. Do gradients reach encoder params? (check after backward, before zero_grad)
  3. Are predictions on StrategyQA/FOLIO diverse? (unique prediction count)

Fast: ~2-3 min on GPU.
"""

import sys
import time
import logging
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))

from shared.model import COFRN, COFRNConfig
from shared.data_utils import (
    load_text_benchmark, collate_text_benchmark,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def patch_manifold_no_sqrt_d(manifold):
    """E12 C3 fix."""
    def patched(structural):
        query = manifold.manifold_proj(structural)
        logits = query @ manifold.anchors.T / manifold.tau
        weights = F.softmax(logits, dim=-1)
        return weights
    manifold.compute_attention_weights = patched
    return manifold


def build_model(unfreeze_layers: int, seed: int = 42) -> COFRN:
    """Build COFRN with specified unfreezing."""
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
        unfreeze_encoder_layers=unfreeze_layers,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = COFRN(config)
    patch_manifold_no_sqrt_d(model.factorization.manifold)
    return model


def check_1_weight_change(model: COFRN, train_ds, seed: int = 42):
    """Check 1: Do encoder weights change after 1 epoch?"""
    logger.info("=" * 70)
    logger.info("  CHECK 1: Encoder weight change after 1 epoch")
    logger.info("=" * 70)

    model.to(DEVICE).train()

    # Snapshot encoder weights before training
    encoder_snapshot = {}
    for name, param in model.encoder.transformer.named_parameters():
        if param.requires_grad:
            encoder_snapshot[name] = param.data.clone()

    # Also snapshot downstream weights for comparison
    downstream_snapshot = {}
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith('encoder.transformer.'):
            downstream_snapshot[name] = param.data.clone()

    n_encoder_params = len(encoder_snapshot)
    n_downstream_params = len(downstream_snapshot)
    logger.info(f"  Unfrozen encoder params: {n_encoder_params}")
    logger.info(f"  Downstream params: {n_downstream_params}")

    # Build optimizer (same as Trainer)
    encoder_params = [
        p for p in model.encoder.transformer.parameters() if p.requires_grad
    ]
    downstream_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith('encoder.transformer.')
    ]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-5},
        {'params': downstream_params, 'lr': 3e-4},
    ], weight_decay=0.01)

    # Train for 1 epoch (up to 50 steps to keep it fast)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=8, shuffle=True,
        collate_fn=collate_text_benchmark, num_workers=0,
    )

    max_steps = 50
    step = 0
    optimizer.zero_grad()
    for batch in loader:
        if step >= max_steps:
            break
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        output = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            answer_input_ids=batch['answer_input_ids'],
            answer_attention_mask=batch['answer_attention_mask'],
            labels=batch['labels'],
        )
        loss = output['total_loss']
        loss.backward()

        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    logger.info(f"  Trained {step} steps")

    # Compare weights
    logger.info("\n  Encoder weight changes:")
    encoder_changed = 0
    encoder_total_diff = 0.0
    for name, before in encoder_snapshot.items():
        param = dict(model.encoder.transformer.named_parameters())[name]
        diff = (param.data - before).abs().sum().item()
        rel_diff = diff / (before.abs().sum().item() + 1e-10)
        encoder_total_diff += diff
        if diff > 0:
            encoder_changed += 1
            if encoder_changed <= 5:
                logger.info(f"    {name}: L1_diff={diff:.6f}, rel={rel_diff:.6f}")

    logger.info(f"\n  Encoder: {encoder_changed}/{n_encoder_params} params changed, total L1 diff={encoder_total_diff:.6f}")

    logger.info("\n  Downstream weight changes (sample):")
    downstream_changed = 0
    downstream_total_diff = 0.0
    for name, before in downstream_snapshot.items():
        # Navigate to the parameter
        obj = model
        for part in name.split('.'):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        diff = (obj.data - before).abs().sum().item()
        downstream_total_diff += diff
        if diff > 0:
            downstream_changed += 1

    logger.info(f"  Downstream: {downstream_changed}/{n_downstream_params} params changed, total L1 diff={downstream_total_diff:.6f}")

    if encoder_changed > 0:
        logger.info("\n  RESULT: ENCODER WEIGHTS CHANGE ✓")
    else:
        logger.info("\n  RESULT: ENCODER WEIGHTS DID NOT CHANGE ✗")
        logger.info("  → Gradients may flow but are too small, or there's a deeper issue")

    return encoder_changed, downstream_changed


def check_2_gradient_flow(model: COFRN, train_ds):
    """Check 2: Do gradients reach encoder params? (check right after backward)"""
    logger.info("\n" + "=" * 70)
    logger.info("  CHECK 2: Gradient flow after backward (before zero_grad)")
    logger.info("=" * 70)

    model.to(DEVICE).train()

    # Zero all grads first
    model.zero_grad()

    # Single forward + backward
    batch_items = [train_ds[i] for i in range(min(8, len(train_ds)))]
    batch = collate_text_benchmark(batch_items)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    output = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        answer_input_ids=batch['answer_input_ids'],
        answer_attention_mask=batch['answer_attention_mask'],
        labels=batch['labels'],
    )
    loss = output['total_loss']
    logger.info(f"  Loss: {loss.item():.4f}")
    loss.backward()

    # Check encoder gradients BEFORE zero_grad
    encoder_grad_count = 0
    encoder_total = 0
    max_grad_norm = 0.0
    grad_norms = {}
    for name, param in model.encoder.transformer.named_parameters():
        if param.requires_grad:
            encoder_total += 1
            if param.grad is not None:
                gnorm = param.grad.abs().sum().item()
                if gnorm > 0:
                    encoder_grad_count += 1
                    max_grad_norm = max(max_grad_norm, gnorm)
                    grad_norms[name] = gnorm

    logger.info(f"\n  Encoder: {encoder_grad_count}/{encoder_total} params have non-zero gradients")
    logger.info(f"  Max grad L1 norm: {max_grad_norm:.6f}")

    if grad_norms:
        logger.info("\n  Top 5 gradient norms:")
        for name, gnorm in sorted(grad_norms.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"    {name}: {gnorm:.6f}")

    # Also check downstream for comparison
    downstream_grad_count = 0
    downstream_total = 0
    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith('encoder.transformer.'):
            downstream_total += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                downstream_grad_count += 1

    logger.info(f"\n  Downstream: {downstream_grad_count}/{downstream_total} params have non-zero gradients")

    # Check projection layer specifically (bridge between encoder and rest)
    for name, param in model.encoder.projection.named_parameters():
        if param.grad is not None:
            gnorm = param.grad.abs().sum().item()
            logger.info(f"  Projection {name}: grad L1={gnorm:.6f}")

    model.zero_grad()

    if encoder_grad_count > 0:
        logger.info("\n  RESULT: GRADIENTS FLOW TO ENCODER ✓")
    else:
        logger.info("\n  RESULT: NO GRADIENTS REACH ENCODER ✗")
        logger.info("  → There IS a gradient-breaking bug in the forward/backward path")

    return encoder_grad_count, encoder_total


def check_3_prediction_diversity(model: COFRN):
    """Check 3: Are predictions diverse on StrategyQA/FOLIO?"""
    logger.info("\n" + "=" * 70)
    logger.info("  CHECK 3: Prediction diversity on StrategyQA and FOLIO")
    logger.info("=" * 70)

    model.to(DEVICE).eval()

    benchmarks = [
        ('strategyqa', 'train', 2),
        ('folio', 'validation', 3),
    ]

    for bench, split, n_choices in benchmarks:
        ds = load_text_benchmark(
            bench, split, tokenizer=model.encoder.tokenizer,
            max_length=128, seed=42,
        )
        n = min(200, len(ds))  # Check first 200

        all_preds = []
        all_scores = []
        with torch.no_grad():
            for start in range(0, n, 16):
                end = min(start + 16, n)
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
                preds = out['scores'].argmax(-1)
                all_preds.extend(preds.cpu().tolist())
                all_scores.append(out['scores'].cpu())

        preds = np.array(all_preds)
        scores = torch.cat(all_scores, dim=0)

        # Prediction distribution
        pred_counts = np.bincount(preds, minlength=n_choices)
        logger.info(f"\n  {bench} ({n} examples, {n_choices} choices):")
        logger.info(f"    Prediction distribution: {dict(enumerate(pred_counts.tolist()))}")
        logger.info(f"    Unique predictions: {len(np.unique(preds))}/{n_choices}")
        logger.info(f"    Most common: {pred_counts.max()}/{n} ({pred_counts.max()/n:.1%})")

        # Score statistics
        mean_scores = scores.mean(dim=0)
        std_scores = scores.std(dim=0)
        logger.info(f"    Mean scores per choice: {mean_scores.tolist()}")
        logger.info(f"    Std scores per choice:  {std_scores.tolist()}")

        # Score range across examples
        score_range = scores.max(dim=-1).values - scores.min(dim=-1).values
        logger.info(f"    Score range (max-min) mean: {score_range.mean():.4f}, "
                     f"std: {score_range.std():.4f}")

        if pred_counts.max() / n > 0.95:
            logger.info(f"    → CONSTANT PREDICTION DETECTED (>{95}% same answer)")
        else:
            logger.info(f"    → Predictions are diverse ✓")


def main():
    t0 = time.time()

    logger.info("E13 GRADIENT DIAGNOSTIC")
    logger.info(f"Device: {DEVICE}")
    logger.info("")

    # Build model with 2 unfrozen layers (C1)
    logger.info("Building C1 model (unfreeze last 2 GPT-2 blocks)...")
    model = build_model(unfreeze_layers=2, seed=42)
    model.to(DEVICE)

    # Count trainable encoder params
    enc_trainable = sum(
        p.numel() for p in model.encoder.transformer.parameters() if p.requires_grad
    )
    enc_total = sum(p.numel() for p in model.encoder.transformer.parameters())
    logger.info(f"Encoder: {enc_trainable:,} trainable / {enc_total:,} total "
                f"({enc_trainable/enc_total:.1%})")

    # Load small training set
    logger.info("Loading training data...")
    train_ds = load_text_benchmark(
        'gsm8k', 'train', tokenizer=model.encoder.tokenizer,
        max_length=128, seed=42,
    )
    logger.info(f"Training examples: {len(train_ds)}")

    # Run checks
    check_2_gradient_flow(model, train_ds)
    check_1_weight_change(model, train_ds)

    # Rebuild fresh model for prediction diversity check (untrained, to see baseline)
    logger.info("\n\nNow checking prediction diversity on UNTRAINED model:")
    model_fresh = build_model(unfreeze_layers=2, seed=42)
    model_fresh.to(DEVICE)
    check_3_prediction_diversity(model_fresh)

    # Also check a C0 cached model if available
    cached_path = BASE_DIR / "results" / "e13_unfrozen" / "cofrn_C1_unfreeze_2_seed42.pt"
    if cached_path.exists():
        logger.info("\n\nLoading TRAINED C1 model for prediction diversity check:")
        model_trained = build_model(unfreeze_layers=2, seed=42)
        state = torch.load(str(cached_path), weights_only=True, map_location=DEVICE)
        model_trained.load_state_dict(state)
        patch_manifold_no_sqrt_d(model_trained.factorization.manifold)
        model_trained.to(DEVICE)
        check_3_prediction_diversity(model_trained)
    else:
        logger.info(f"\n  No cached C1 model at {cached_path}")

    # Also check C0 frozen model
    cached_c0 = BASE_DIR / "results" / "e13_unfrozen" / "cofrn_C0_frozen_seed42.pt"
    if cached_c0.exists():
        logger.info("\n\nLoading C0 FROZEN model for prediction diversity comparison:")
        config_c0 = COFRNConfig(
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
        torch.manual_seed(42)
        model_c0 = COFRN(config_c0)
        state = torch.load(str(cached_c0), weights_only=True, map_location=DEVICE)
        model_c0.load_state_dict(state)
        patch_manifold_no_sqrt_d(model_c0.factorization.manifold)
        model_c0.to(DEVICE).eval()
        # C0 uses precomputed embeddings, can't run text through it
        logger.info("  (C0 uses precomputed embeddings — skipping text-based eval)")

    elapsed = time.time() - t0
    logger.info(f"\n\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
