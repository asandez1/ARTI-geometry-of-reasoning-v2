#!/usr/bin/env python3
"""
ARTI Training, Evaluation, and Visualization Pipeline.

Trains the Active Reasoning Type Identifier on mixed reasoning traces
and visualizes the learned type clusters on the 10D manifold.

Usage:
    # Full pipeline (collect data, train, evaluate, visualize)
    python run_arti.py --output results/arti/

    # Quick mode (no benchmark data, small synthetic set)
    python run_arti.py --quick --output results/arti_quick/

    # Use cached dataset
    python run_arti.py --dataset results/arti/dataset.pt --output results/arti_v2/

    # Evaluate only (load trained model)
    python run_arti.py --eval-only --checkpoint results/arti/arti_model.pt
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
    TYPE_NAMES, TYPE_SHORT_NAMES, TYPE_COLORS,
    HeuristicLabeler,
)
from shared.arti import ARTI, ARTIConfig, StreamingARTI

from arti_data import (
    ReasoningTraceDataset, build_training_dataset,
    save_dataset, load_dataset,
    collect_builtin_examples, generate_expanded_traces,
    compute_class_weights,
)

logger = logging.getLogger(__name__)


# ─── Training ──────────────────────────────────────────────────────────────────

def collate_arti(batch):
    """Collate function for ARTI datasets."""
    result = {
        'embeddings': torch.stack([b['embeddings'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }
    if 'soft_labels' in batch[0] and batch[0]['soft_labels'] is not None:
        result['soft_labels'] = torch.stack([b['soft_labels'] for b in batch])
    return result


def train_arti(
    arti: ARTI,
    train_ds: ReasoningTraceDataset,
    val_ds: ReasoningTraceDataset,
    max_epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    patience: int = 15,
    device: str = 'cpu',
    use_soft_labels: bool = True,
    class_weights: torch.Tensor = None,
) -> dict:
    """
    Train the ARTI classifier.

    Args:
        arti: ARTI model
        train_ds: training dataset
        val_ds: validation dataset
        max_epochs: maximum training epochs
        batch_size: training batch size
        lr: learning rate
        weight_decay: weight decay
        patience: early stopping patience
        device: torch device
        use_soft_labels: if True, use KL divergence with soft labels
        class_weights: optional inverse-frequency class weights [N_REASONING_TYPES]

    Returns:
        Dict with training history and metrics
    """
    arti = arti.to(device)
    arti.train()

    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Using class weights: {class_weights.tolist()}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_arti, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_arti, num_workers=0,
    )

    optimizer = AdamW(arti.parameters(), lr=lr, weight_decay=weight_decay)
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
        arti.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            result = arti(embeddings)
            logits = result['logits']

            if use_soft_labels and 'soft_labels' in batch:
                soft_labels = batch['soft_labels'].to(device)
                if class_weights is not None:
                    # Weighted KL: weight each sample by its true class weight
                    log_probs = F.log_softmax(logits, dim=-1)
                    kl_per_sample = F.kl_div(
                        log_probs, soft_labels, reduction='none'
                    ).sum(dim=-1)
                    sample_weights = class_weights[labels]
                    loss = (kl_per_sample * sample_weights).mean()
                else:
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
            else:
                loss = F.cross_entropy(logits, labels, weight=class_weights)

            loss.backward()
            nn.utils.clip_grad_norm_(arti.parameters(), 1.0)
            optimizer.step()

            bs = labels.shape[0]
            total_loss += loss.item() * bs
            predicted = logits.argmax(dim=-1)
            total_correct += (predicted == labels).sum().item()
            total_samples += bs

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Validate
        val_metrics = evaluate_arti(arti, val_loader, device)
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
            best_state = {k: v.cpu().clone() for k, v in arti.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        arti.load_state_dict(best_state)

    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'epochs_trained': len(history['train_loss']),
    }


@torch.no_grad()
def evaluate_arti(
    arti: ARTI,
    loader: DataLoader,
    device: str = 'cpu',
) -> dict:
    """Evaluate ARTI on a dataset."""
    arti.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_type_correct = Counter()
    per_type_total = Counter()

    all_preds = []
    all_labels = []
    all_coords = []

    for batch in loader:
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)

        result = arti(embeddings)
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
        all_coords.append(result['manifold_coords'].cpu().numpy())

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
        'manifold_coords': np.concatenate(all_coords, axis=0) if all_coords else np.array([]),
    }


# ─── Visualization ─────────────────────────────────────────────────────────────

def plot_manifold_clusters(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "Reasoning Types on Operator Manifold",
):
    """
    2D PCA visualization of reasoning type clusters on the manifold.

    Args:
        coords: [N, manifold_dim] manifold coordinates
        labels: [N] type labels (int)
        output_path: path to save the figure
        title: plot title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("matplotlib or sklearn not available, skipping visualization")
        return

    # PCA to 2D
    if coords.shape[1] > 2:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(coords)
        var_explained = pca.explained_variance_ratio_
        xlabel = f"PC1 ({var_explained[0]:.1%})"
        ylabel = f"PC2 ({var_explained[1]:.1%})"
    else:
        coords_2d = coords
        xlabel, ylabel = "Dim 1", "Dim 2"

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    for rtype in ReasoningType:
        mask = labels == int(rtype)
        if not mask.any():
            continue
        spec = REASONING_TYPES[rtype]
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=spec.color, label=spec.short_name,
            alpha=0.6, s=40, edgecolors='white', linewidth=0.5,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Manifold plot saved to {output_path}")


def plot_confusion_matrix(
    predictions: list,
    labels: list,
    output_path: str,
):
    """Plot confusion matrix for ARTI predictions."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        logger.warning("matplotlib or sklearn not available, skipping visualization")
        return

    cm = confusion_matrix(labels, predictions, labels=list(range(N_REASONING_TYPES)))
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(N_REASONING_TYPES))
    ax.set_yticks(range(N_REASONING_TYPES))
    ax.set_xticklabels(TYPE_SHORT_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(TYPE_SHORT_NAMES)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("ARTI Confusion Matrix", fontsize=14)

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


def plot_training_curves(
    history: dict,
    output_path: str,
):
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
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax2.axhline(y=1.0/N_REASONING_TYPES, color='gray', linestyle='--',
                label=f'Chance ({1.0/N_REASONING_TYPES:.1%})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved to {output_path}")


# ─── Heuristic Baseline ───────────────────────────────────────────────────────

def evaluate_heuristic_baseline(
    dataset: ReasoningTraceDataset,
) -> dict:
    """
    Evaluate the keyword-based heuristic labeler as a baseline.

    Returns accuracy metrics comparable to the trained ARTI classifier.
    """
    labeler = HeuristicLabeler()
    correct = 0
    total = 0
    per_type_correct = Counter()
    per_type_total = Counter()

    for i in range(len(dataset)):
        text = dataset.texts[i]
        true_label = dataset.labels[i].item()
        pred_label, conf = labeler.label_hard(text)

        per_type_total[true_label] += 1
        if int(pred_label) == true_label:
            correct += 1
            per_type_correct[true_label] += 1
        total += 1

    per_type_acc = {}
    for rtype in ReasoningType:
        t = per_type_total.get(int(rtype), 0)
        c = per_type_correct.get(int(rtype), 0)
        per_type_acc[REASONING_TYPES[rtype].short_name] = c / t if t > 0 else 0.0

    return {
        'accuracy': correct / max(total, 1),
        'per_type_accuracy': per_type_acc,
    }


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARTI Training Pipeline")
    parser.add_argument('--output', type=str, default='results/arti/',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: no benchmarks, small synthetic set')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to cached dataset')
    parser.add_argument('--eval-only', action='store_true',
                        help='Evaluate only (needs --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained ARTI model')
    parser.add_argument('--encoder', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    parser.add_argument('--classifier', type=str, default='mlp',
                        choices=['centroid', 'linear', 'mlp'],
                        help='Classifier type')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-synthetic', type=int, default=50,
                        help='Expanded synthetic examples per type')
    parser.add_argument('--max-per-benchmark', type=int, default=2000,
                        help='Max examples per benchmark')
    parser.add_argument('--target-per-type', type=int, default=750,
                        help='Target samples per type after balancing')
    parser.add_argument('--no-balance', action='store_true',
                        help='Disable class-balanced sampling')
    parser.add_argument('--no-class-weights', action='store_true',
                        help='Disable class-weighted loss')
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

    logger.info("=" * 60)
    logger.info("ARTI: Active Reasoning Type Identifier")
    logger.info("=" * 60)

    # ── Step 1: Build or load dataset ──────────────────────────────────

    if args.dataset and Path(args.dataset).exists():
        logger.info(f"Loading cached dataset from {args.dataset}")
        full_dataset = load_dataset(args.dataset)
    else:
        logger.info("Building training dataset...")
        from shared.encoder import SentenceTransformerEncoder
        encoder = SentenceTransformerEncoder(
            model_name=args.encoder,
            hidden_dim=256,
            load_pretrained=True,
        )

        if args.quick:
            full_dataset = build_training_dataset(
                encoder=encoder,
                n_synthetic_per_type=10,
                benchmarks=None,
                max_per_benchmark=0,
                target_per_type=50,
                balance=False,
                seed=args.seed,
            )
        else:
            full_dataset = build_training_dataset(
                encoder=encoder,
                n_synthetic_per_type=args.n_synthetic,
                benchmarks=['gsm8k', 'arc_challenge', 'strategyqa', 'folio'],
                max_per_benchmark=args.max_per_benchmark,
                target_per_type=args.target_per_type,
                balance=not args.no_balance,
                seed=args.seed,
            )

        # Save dataset
        dataset_path = output_dir / 'dataset.pt'
        save_dataset(full_dataset, str(dataset_path))

    # Split
    train_ds, val_ds = full_dataset.split(train_ratio=0.8, seed=args.seed)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Label distribution
    train_dist = Counter(train_ds.labels.numpy().tolist())
    logger.info("Train label distribution:")
    for rtype in ReasoningType:
        count = train_dist.get(int(rtype), 0)
        logger.info(f"  {REASONING_TYPES[rtype].short_name:>8}: {count}")

    # ── Step 2: Heuristic baseline ─────────────────────────────────────

    logger.info("\nHeuristic baseline evaluation...")
    heuristic_results = evaluate_heuristic_baseline(val_ds)
    logger.info(f"Heuristic accuracy: {heuristic_results['accuracy']:.1%}")
    for name, acc in heuristic_results['per_type_accuracy'].items():
        logger.info(f"  {name:>8}: {acc:.1%}")

    # ── Step 3: Initialize ARTI ────────────────────────────────────────

    encoder_dim = full_dataset.embeddings.shape[1]
    config = ARTIConfig(
        encoder_dim=encoder_dim,
        manifold_dim=10,
        n_geometric_features=32,
        classifier_type=args.classifier,
        hidden_dim=64,
        dropout=0.1,
    )

    arti = ARTI(config)
    logger.info(f"\nARTI model ({args.classifier} classifier):")
    breakdown = arti.get_param_breakdown()
    for k, v in breakdown.items():
        logger.info(f"  {k}: {v:,}")

    # ── Step 4: Train ──────────────────────────────────────────────────

    if not args.eval_only:
        # Compute class weights for balanced loss
        weights = None
        if not args.no_class_weights:
            weights = compute_class_weights(train_ds)
            logger.info(f"\nClass weights computed from training distribution")

        logger.info(f"\nTraining for up to {args.epochs} epochs...")
        train_results = train_arti(
            arti=arti,
            train_ds=train_ds,
            val_ds=val_ds,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=15,
            device=device,
            use_soft_labels=True,
            class_weights=weights,
        )

        logger.info(f"\nBest val accuracy: {train_results['best_val_acc']:.1%}")
        logger.info(f"Epochs trained: {train_results['epochs_trained']}")

        # Save model
        model_path = output_dir / 'arti_model.pt'
        torch.save(arti.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # Plot training curves
        plot_training_curves(
            train_results['history'],
            str(output_dir / 'training_curves.png'),
        )
    elif args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        arti.load_state_dict(torch.load(args.checkpoint, weights_only=True))

    # ── Step 5: Final evaluation ───────────────────────────────────────

    logger.info("\nFinal evaluation on validation set...")
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_arti, num_workers=0,
    )
    final_metrics = evaluate_arti(arti, val_loader, device)

    logger.info(f"\nARTI Accuracy: {final_metrics['accuracy']:.1%}")
    logger.info(f"Heuristic Baseline: {heuristic_results['accuracy']:.1%}")
    logger.info(f"Random Chance: {1.0/N_REASONING_TYPES:.1%}")
    logger.info(f"\nPer-type accuracy:")
    for name, acc in final_metrics['per_type_accuracy'].items():
        heur = heuristic_results['per_type_accuracy'].get(name, 0.0)
        delta = acc - heur
        marker = "+" if delta > 0 else ""
        logger.info(f"  {name:>8}: ARTI={acc:.1%}  Heuristic={heur:.1%}  ({marker}{delta:.1%})")

    # ── Step 6: Visualizations ─────────────────────────────────────────

    logger.info("\nGenerating visualizations...")

    if len(final_metrics['manifold_coords']) > 0:
        plot_manifold_clusters(
            coords=final_metrics['manifold_coords'],
            labels=np.array(final_metrics['labels']),
            output_path=str(output_dir / 'manifold_clusters.png'),
            title="Reasoning Types on 10D Operator Manifold (PCA→2D)",
        )

        plot_confusion_matrix(
            predictions=final_metrics['predictions'],
            labels=final_metrics['labels'],
            output_path=str(output_dir / 'confusion_matrix.png'),
        )

    # ── Step 7: Save results ───────────────────────────────────────────

    results = {
        'config': {
            'encoder': args.encoder,
            'classifier': args.classifier,
            'encoder_dim': encoder_dim,
            'manifold_dim': config.manifold_dim,
            'n_params': arti.trainable_params,
        },
        'dataset': {
            'total': len(full_dataset),
            'train': len(train_ds),
            'val': len(val_ds),
        },
        'heuristic_baseline': heuristic_results,
        'arti_results': {
            'accuracy': final_metrics['accuracy'],
            'per_type_accuracy': final_metrics['per_type_accuracy'],
        },
        'improvement_over_heuristic': (
            final_metrics['accuracy'] - heuristic_results['accuracy']
        ),
        'improvement_over_chance': (
            final_metrics['accuracy'] - 1.0 / N_REASONING_TYPES
        ),
    }

    if not args.eval_only:
        results['training'] = {
            'epochs': train_results['epochs_trained'],
            'best_val_acc': train_results['best_val_acc'],
        }

    results_path = output_dir / 'arti_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    # ── Summary ────────────────────────────────────────────────────────

    logger.info("\n" + "=" * 60)
    logger.info("ARTI RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Parameters:     {arti.trainable_params:,}")
    logger.info(f"  ARTI Accuracy:  {final_metrics['accuracy']:.1%}")
    logger.info(f"  Heuristic:      {heuristic_results['accuracy']:.1%}")
    logger.info(f"  Random Chance:  {1.0/N_REASONING_TYPES:.1%}")
    logger.info(f"  Delta vs Heur:  {results['improvement_over_heuristic']:+.1%}")
    logger.info(f"  Delta vs Chance:{results['improvement_over_chance']:+.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
