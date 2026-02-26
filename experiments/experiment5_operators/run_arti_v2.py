#!/usr/bin/env python3
"""
ARTI v2 Training: Trajectory-Based Reasoning Type Classification.

Trains ARTI v2 which classifies reasoning types from trajectory geometry
on the 10D operator manifold, rather than single-point position (v1).

Pipeline:
    1. Load v1 ManifoldProjection weights (freeze them)
    2. Build TrajectoryDataset from v1 dataset texts
    3. Train TrajectoryFeatureExtractor + Classifier
    4. Compare per-type accuracy with v1

Usage:
    python run_arti_v2.py --output results/arti_v2_trajectory/
    python run_arti_v2.py --traj-dataset results/arti_v2_trajectory/traj_dataset.pt
"""

import argparse
import logging
import json
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.reasoning_types import (
    ReasoningType, N_REASONING_TYPES, REASONING_TYPES,
    TYPE_SHORT_NAMES,
)
from shared.arti import ARTI, ARTIConfig
from shared.arti_v2 import ARTIV2, ARTIV2Config
from shared.encoder import SentenceTransformerEncoder

from arti_v2_data import (
    TrajectoryDataset, collate_trajectory,
    build_trajectory_dataset, save_trajectory_dataset,
    load_trajectory_dataset,
)
from arti_data import compute_class_weights, load_dataset

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
V1_MODEL_PATH = BASE_DIR / "results" / "arti_v3" / "arti_model.pt"
V1_DATASET_PATH = BASE_DIR / "results" / "arti_v3" / "dataset.pt"
ENCODER_NAME = "all-MiniLM-L6-v2"
SEED = 42


# ─── Training ────────────────────────────────────────────────────────────────

def train_arti_v2(
    model: ARTIV2,
    train_ds: TrajectoryDataset,
    val_ds: TrajectoryDataset,
    max_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    patience: int = 15,
    device: str = 'cpu',
    class_weights: torch.Tensor = None,
) -> dict:
    """
    Train the ARTI v2 trajectory classifier.

    Args:
        model: ARTIV2 model (manifold frozen)
        train_ds: training trajectory dataset
        val_ds: validation trajectory dataset
        max_epochs: maximum epochs
        batch_size: batch size
        lr: learning rate
        weight_decay: AdamW weight decay
        patience: early stopping patience
        device: torch device
        class_weights: inverse-frequency class weights

    Returns:
        Dict with training history and best metrics
    """
    model = model.to(device)
    model.train()

    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Using class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_trajectory, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_trajectory, num_workers=0,
    )

    # Only optimize trainable parameters (manifold is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_per_type_acc': [],
    }
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            clause_embs_list = [c.to(device) for c in batch['clause_embeddings_list']]
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            result = model(clause_embs_list)
            logits = result['logits']

            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            bs = labels.shape[0]
            total_loss += loss.item() * bs
            predicted = logits.argmax(dim=-1)
            total_correct += (predicted == labels).sum().item()
            total_samples += bs

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Validate
        val_metrics = evaluate_arti_v2(model, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_per_type_acc'].append(val_metrics['per_type_accuracy'])

        logger.info(
            f"  Epoch {epoch+1}/{max_epochs}: "
            f"train_acc={train_acc:.1%} val_acc={val_metrics['accuracy']:.1%} "
            f"loss={train_loss:.4f}"
        )

        # Early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'epochs_trained': len(history['train_loss']),
    }


@torch.no_grad()
def evaluate_arti_v2(
    model: ARTIV2,
    loader: DataLoader,
    device: str = 'cpu',
) -> dict:
    """Evaluate ARTI v2 on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_type_correct = Counter()
    per_type_total = Counter()

    all_preds = []
    all_labels = []

    for batch in loader:
        clause_embs_list = [c.to(device) for c in batch['clause_embeddings_list']]
        labels = batch['labels'].to(device)

        result = model(clause_embs_list)
        logits = result['logits']
        loss = F.cross_entropy(logits, labels)

        predicted = logits.argmax(dim=-1)
        bs = labels.shape[0]
        total_loss += loss.item() * bs
        total_correct += (predicted == labels).sum().item()
        total_samples += bs

        for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
            per_type_total[int(label)] += 1
            if pred == label:
                per_type_correct[int(label)] += 1

        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = max(total_samples, 1)

    per_type_acc = {}
    for rtype in ReasoningType:
        total = per_type_total.get(int(rtype), 0)
        correct = per_type_correct.get(int(rtype), 0)
        per_type_acc[REASONING_TYPES[rtype].short_name] = (
            correct / total if total > 0 else 0.0
        )

    return {
        'loss': total_loss / n,
        'accuracy': total_correct / n,
        'per_type_accuracy': per_type_acc,
        'predictions': all_preds,
        'labels': all_labels,
    }


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, output_path: str):
    """Plot training loss and accuracy curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('ARTI v2 Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax2.axhline(y=1.0/N_REASONING_TYPES, color='gray', linestyle='--',
                label=f'Chance ({1.0/N_REASONING_TYPES:.1%})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('ARTI v2 Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved to {output_path}")


def plot_comparison_bar(v1_acc: dict, v2_acc: dict, output_path: str):
    """Bar chart comparing v1 vs v2 per-type accuracy."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    types = list(v1_acc.keys())
    v1_vals = [v1_acc[t] for t in types]
    v2_vals = [v2_acc.get(t, 0.0) for t in types]

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, v1_vals, width, label='ARTI v1 (position)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, v2_vals, width, label='ARTI v2 (trajectory)', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Reasoning Type', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('ARTI v1 vs v2: Per-Type Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.0%}',
                ha='center', va='bottom', fontsize=8, color='#3498db')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.0%}',
                ha='center', va='bottom', fontsize=8, color='#e74c3c')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison chart saved to {output_path}")


def plot_confusion_matrix(predictions: list, labels: list, output_path: str):
    """Plot confusion matrix for ARTI v2 predictions."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return

    cm = confusion_matrix(labels, predictions, labels=list(range(N_REASONING_TYPES)))
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(cm_normalized, cmap='Reds', vmin=0, vmax=1)

    ax.set_xticks(range(N_REASONING_TYPES))
    ax.set_yticks(range(N_REASONING_TYPES))
    ax.set_xticklabels(TYPE_SHORT_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(TYPE_SHORT_NAMES)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("ARTI v2 Confusion Matrix", fontsize=14)

    for i in range(N_REASONING_TYPES):
        for j in range(N_REASONING_TYPES):
            val = cm_normalized[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f"{val:.0%}", ha='center', va='center',
                    color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='Accuracy')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


# ─── Test on Specific Phrases ────────────────────────────────────────────────

def test_specific_phrases(model: ARTIV2, encoder, device: str = 'cpu'):
    """Test the 5 originally-failing phrases + typical examples."""
    from shared.text_utils import segment_text

    test_phrases = [
        # Originally-failing phrases for v1 (especially PhysCause)
        ("The temperature dropped below freezing, so the pipes burst from ice expansion.",
         "PhysCause"),
        ("She studied every day for a month and scored 98% on the exam.",
         "BehvCause"),
        ("The drought killed crops, causing food shortages that triggered mass migration.",
         "SysCause"),
        ("All mammals are warm-blooded. A whale is a mammal. Therefore, a whale is warm-blooded.",
         "Deduc"),
        ("Every patient given drug X improved. The evidence suggests drug X is effective.",
         "Induc"),
        # Additional test cases
        ("A firewall protects a network just as a moat protects a castle.",
         "Analog"),
        ("$500 was transferred from checking to savings. The total stays the same.",
         "Conserv"),
        ("What if the bridge had been inspected? The failure would have been prevented.",
         "Counter"),
        ("The lawn is wet but it hasn't rained. The best explanation is overnight dew.",
         "Abduc"),
        ("First, calculate distance. Second, find fuel cost. Third, add tolls.",
         "Decomp"),
    ]

    model.eval()
    results = []

    for text, expected in test_phrases:
        # Segment and encode
        segments = segment_text(text)
        with torch.no_grad():
            clause_embs = encoder.encode_texts(segments).to(device)  # [n_clauses, 384]
            result = model([clause_embs])

        pred_idx = result['type'].item()
        pred_name = REASONING_TYPES[ReasoningType(pred_idx)].short_name
        confidence = result['confidence'].item()
        correct = pred_name == expected

        results.append({
            'text': text[:60],
            'expected': expected,
            'predicted': pred_name,
            'confidence': confidence,
            'correct': correct,
            'n_clauses': len(segments),
        })

        status = "OK" if correct else "MISS"
        logger.info(
            f"  [{status}] {text[:55]}..."
            f"\n       Expected: {expected:>10}  Got: {pred_name:>10} "
            f"({confidence:.1%}, {len(segments)} clauses)"
        )

    n_correct = sum(1 for r in results if r['correct'])
    logger.info(f"\n  Test phrases: {n_correct}/{len(results)} correct")
    return results


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARTI v2 Training Pipeline")
    parser.add_argument('--output', type=str, default='results/arti_v2_trajectory/',
                        help='Output directory')
    parser.add_argument('--traj-dataset', type=str, default=None,
                        help='Path to cached trajectory dataset')
    parser.add_argument('--v1-model', type=str, default=None,
                        help='Path to v1 ARTI model (for manifold weights)')
    parser.add_argument('--v1-dataset', type=str, default=None,
                        help='Path to v1 dataset.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    v1_model_path = Path(args.v1_model) if args.v1_model else V1_MODEL_PATH
    v1_dataset_path = Path(args.v1_dataset) if args.v1_dataset else V1_DATASET_PATH

    logger.info("=" * 60)
    logger.info("ARTI v2: Trajectory-Based Reasoning Type Classification")
    logger.info("=" * 60)

    # ── Step 1: Load encoder ──────────────────────────────────────────

    logger.info("\n[1/6] Loading sentence encoder...")
    encoder = SentenceTransformerEncoder(
        model_name=ENCODER_NAME, hidden_dim=256, load_pretrained=True,
    )
    logger.info(f"  {ENCODER_NAME} (384D)")

    # ── Step 2: Build or load trajectory dataset ──────────────────────

    logger.info("\n[2/6] Building trajectory dataset...")
    traj_dataset_path = args.traj_dataset
    if traj_dataset_path and Path(traj_dataset_path).exists():
        logger.info(f"  Loading cached trajectory dataset from {traj_dataset_path}")
        full_traj_ds = load_trajectory_dataset(traj_dataset_path)
    else:
        full_traj_ds = build_trajectory_dataset(
            v1_dataset_path=str(v1_dataset_path),
            encoder=encoder,
            batch_size=args.batch_size,
        )
        # Save for reuse
        traj_save_path = output_dir / 'traj_dataset.pt'
        save_trajectory_dataset(full_traj_ds, str(traj_save_path))

    # Split (matching v1: 80/20, same seed)
    train_ds, val_ds = full_traj_ds.split(train_ratio=0.8, seed=args.seed)
    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Label distribution
    train_dist = Counter(train_ds.labels.numpy().tolist())
    logger.info("  Train label distribution:")
    for rtype in ReasoningType:
        count = train_dist.get(int(rtype), 0)
        logger.info(f"    {REASONING_TYPES[rtype].short_name:>8}: {count}")

    # ── Step 3: Initialize ARTI v2 with v1 manifold ───────────────────

    logger.info("\n[3/6] Initializing ARTI v2...")

    config = ARTIV2Config(
        encoder_dim=384,
        manifold_dim=10,
        traj_feature_dim=60,
        traj_hidden=48,
        n_classes=N_REASONING_TYPES,
        classifier_hidden=96,
        dropout=0.1,
    )
    model = ARTIV2(config)

    # Load v1 ManifoldProjection weights
    logger.info(f"  Loading v1 manifold weights from {v1_model_path}")
    v1_state = torch.load(str(v1_model_path), weights_only=True, map_location='cpu')
    # Extract manifold_proj weights from v1
    manifold_keys = {k: v for k, v in v1_state.items() if k.startswith('manifold_proj.')}
    model.manifold_proj.load_state_dict(
        {k.replace('manifold_proj.', ''): v for k, v in manifold_keys.items()}
    )
    logger.info(f"  Loaded {len(manifold_keys)} manifold weight tensors")

    # Freeze manifold
    model.freeze_manifold()

    breakdown = model.get_param_breakdown()
    logger.info(f"\n  Parameter breakdown:")
    for k, v in breakdown.items():
        logger.info(f"    {k}: {v:,}")

    # ── Step 4: Train ─────────────────────────────────────────────────

    logger.info(f"\n[4/6] Training for up to {args.epochs} epochs...")

    # Compute class weights from v1 dataset (same distribution)
    v1_full_ds = load_dataset(str(v1_dataset_path))
    v1_train_ds, _ = v1_full_ds.split(train_ratio=0.8, seed=args.seed)
    class_weights = compute_class_weights(v1_train_ds)
    logger.info(f"  Class weights computed from training distribution")

    train_results = train_arti_v2(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=0.01,
        patience=args.patience,
        device=device,
        class_weights=class_weights,
    )

    logger.info(f"\n  Best val accuracy: {train_results['best_val_acc']:.1%}")
    logger.info(f"  Epochs trained: {train_results['epochs_trained']}")

    # Save model
    model_path = output_dir / 'arti_v2_model.pt'
    torch.save(model.state_dict(), model_path)
    logger.info(f"  Model saved to {model_path}")

    # Plot training curves
    plot_training_curves(
        train_results['history'],
        str(output_dir / 'training_curves.png'),
    )

    # ── Step 5: Final evaluation + comparison ─────────────────────────

    logger.info("\n[5/6] Final evaluation...")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_trajectory, num_workers=0,
    )
    final_metrics = evaluate_arti_v2(model, val_loader, device)

    # Load v1 results for comparison
    v1_results_path = BASE_DIR / "results" / "arti_v3" / "arti_results.json"
    v1_per_type = {}
    v1_overall = 0.0
    if v1_results_path.exists():
        with open(v1_results_path) as f:
            v1_results = json.load(f)
        v1_per_type = v1_results.get('arti_results', {}).get('per_type_accuracy', {})
        v1_overall = v1_results.get('arti_results', {}).get('accuracy', 0.0)
    else:
        # Fallback to hardcoded v1 values
        v1_per_type = {
            "PhysCause": 0.093, "BehvCause": 0.560, "SysCause": 0.740,
            "Deduc": 0.613, "Induc": 0.660, "Analog": 0.967,
            "Conserv": 0.813, "Counter": 0.927, "Abduc": 0.973, "Decomp": 0.800,
        }
        v1_overall = 0.715

    logger.info(f"\n{'='*60}")
    logger.info(f"  COMPARISON: ARTI v1 (position) vs v2 (trajectory)")
    logger.info(f"{'='*60}")
    logger.info(f"  Overall accuracy:")
    logger.info(f"    v1: {v1_overall:.1%}")
    logger.info(f"    v2: {final_metrics['accuracy']:.1%}")
    delta = final_metrics['accuracy'] - v1_overall
    logger.info(f"    delta: {delta:+.1%}")

    logger.info(f"\n  Per-type accuracy:")
    logger.info(f"  {'Type':>10}  {'v1':>6}  {'v2':>6}  {'delta':>7}  {'winner':>6}")
    logger.info(f"  {'-'*45}")
    v2_wins = 0
    v1_wins = 0
    for name in TYPE_SHORT_NAMES:
        v1a = v1_per_type.get(name, 0.0)
        v2a = final_metrics['per_type_accuracy'].get(name, 0.0)
        d = v2a - v1a
        winner = "v2" if d > 0.01 else ("v1" if d < -0.01 else "tie")
        if d > 0.01:
            v2_wins += 1
        elif d < -0.01:
            v1_wins += 1
        logger.info(f"  {name:>10}  {v1a:5.1%}  {v2a:5.1%}  {d:+6.1%}  {winner:>6}")

    logger.info(f"\n  v2 wins on {v2_wins} types, v1 wins on {v1_wins} types")

    # ── Step 5b: Test specific phrases ────────────────────────────────

    logger.info(f"\n  Testing specific phrases...")
    model.to(device)
    phrase_results = test_specific_phrases(model, encoder, device)

    # ── Step 6: Save results ──────────────────────────────────────────

    logger.info(f"\n[6/6] Saving results...")

    # Plot comparison
    plot_comparison_bar(
        v1_per_type, final_metrics['per_type_accuracy'],
        str(output_dir / 'v1_vs_v2_comparison.png'),
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        final_metrics['predictions'], final_metrics['labels'],
        str(output_dir / 'confusion_matrix.png'),
    )

    results = {
        'config': {
            'encoder': ENCODER_NAME,
            'encoder_dim': config.encoder_dim,
            'manifold_dim': config.manifold_dim,
            'traj_feature_dim': config.traj_feature_dim,
            'traj_hidden': config.traj_hidden,
            'classifier_hidden': config.classifier_hidden,
            'n_trainable_params': model.trainable_params,
        },
        'dataset': {
            'total': len(full_traj_ds),
            'train': len(train_ds),
            'val': len(val_ds),
        },
        'training': {
            'epochs': train_results['epochs_trained'],
            'best_val_acc': train_results['best_val_acc'],
        },
        'v2_results': {
            'accuracy': final_metrics['accuracy'],
            'per_type_accuracy': final_metrics['per_type_accuracy'],
        },
        'v1_results': {
            'accuracy': v1_overall,
            'per_type_accuracy': v1_per_type,
        },
        'comparison': {
            'delta_overall': delta,
            'v2_type_wins': v2_wins,
            'v1_type_wins': v1_wins,
        },
        'phrase_tests': phrase_results,
    }

    results_path = output_dir / 'arti_v2_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Results saved to {results_path}")

    # ── Summary ───────────────────────────────────────────────────────

    logger.info(f"\n{'='*60}")
    logger.info(f"ARTI v2 RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Architecture:    Trajectory-based (60D raw -> 48D -> 96 -> 10)")
    logger.info(f"  Parameters:      {model.trainable_params:,} trainable "
                f"(+ {sum(p.numel() for p in model.manifold_proj.parameters()):,} frozen manifold)")
    logger.info(f"  v2 Accuracy:     {final_metrics['accuracy']:.1%}")
    logger.info(f"  v1 Accuracy:     {v1_overall:.1%}")
    logger.info(f"  Delta:           {delta:+.1%}")
    logger.info(f"  Random Chance:   {1.0/N_REASONING_TYPES:.1%}")
    logger.info(f"  v2 type wins:    {v2_wins}/{N_REASONING_TYPES}")
    logger.info(f"  Phrase tests:    {sum(1 for r in phrase_results if r['correct'])}/{len(phrase_results)}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
