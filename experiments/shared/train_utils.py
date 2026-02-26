#!/usr/bin/env python3
"""
Training Utilities for CO-FRN Experiments (Section 3.4 of draft).

Adapted from Paper 11 E3 train.py.

Key features:
- Three-term loss: L_task + lambda_1 * L_factorize + lambda_2 * L_coherence
- Curriculum learning for multi-step tasks (depth 1 -> 2 -> 3+)
- Temperature annealing per epoch
- Early stopping on validation accuracy
- Loss weight annealing (ramp lambda_1 from 0.01 to 0.1)
- Inference latency measurement
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time
import logging

from .data_utils import collate_benchmark

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration for CO-FRN and baselines."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 32

    # E13: Differential LR for unfrozen encoder layers
    encoder_lr: float = 0.0  # 0 = use learning_rate for all params
    gradient_accumulation_steps: int = 1  # >1 for larger effective batch size

    # Schedule
    max_epochs: int = 30
    patience: int = 5
    warmup_epochs: int = 2

    # Loss weights
    lambda_factorize: float = 0.1
    lambda_coherence: float = 0.01
    lambda_entropy: float = 0.01
    loss_anneal_epochs: int = 5  # epochs to ramp lambda_factorize from 0.01 to target

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42


@dataclass
class CurriculumPhase:
    """A single phase of curriculum learning for multi-step tasks."""
    depth: int               # tree reasoning depth for this phase
    epochs: int              # epochs in this phase
    lambda_factorize: float  # factorization loss weight
    lambda_coherence: float  # coherence loss weight
    learning_rate: float     # LR for this phase


DEFAULT_CURRICULUM = [
    CurriculumPhase(depth=1, epochs=5, lambda_factorize=0.01, lambda_coherence=0.0, learning_rate=3e-4),
    CurriculumPhase(depth=2, epochs=5, lambda_factorize=0.05, lambda_coherence=0.005, learning_rate=1e-4),
    CurriculumPhase(depth=3, epochs=20, lambda_factorize=0.1, lambda_coherence=0.01, learning_rate=5e-5),
]


class Trainer:
    """
    Trains CO-FRN or baseline models.

    Handles the three-term loss, temperature annealing, early stopping,
    and detailed metric tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        model_name: str = 'cofrn',
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name
        self.device = config.device

        # Build optimizer with optional differential LR for encoder params
        if config.encoder_lr > 0 and hasattr(model, 'encoder') and hasattr(model.encoder, 'transformer'):
            encoder_params = [
                p for p in model.encoder.transformer.parameters() if p.requires_grad
            ]
            downstream_params = [
                p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith('encoder.transformer.')
            ]
            param_groups = [
                {'params': encoder_params, 'lr': config.encoder_lr},
                {'params': downstream_params, 'lr': config.learning_rate},
            ]
        else:
            trainable = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{'params': trainable, 'lr': config.learning_rate}]

        self.optimizer = AdamW(
            param_groups,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs
        )

        # LR warmup
        if config.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=config.warmup_epochs,
            )
        else:
            self.warmup_scheduler = None

        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_state = None

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_task_loss': [], 'val_factorize_loss': [],
            'val_coherence_loss': [], 'val_entropy_loss': [],
            'lr': [], 'temperature': [],
        }

    def _to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _get_loss_weights(self) -> Tuple[float, float, float]:
        """Get annealed loss weights."""
        anneal = self.config.loss_anneal_epochs
        if anneal > 0:
            ramp = min(1.0, self.current_epoch / anneal)
        else:
            ramp = 1.0

        # Delegate entropy weight to model's schedule when available
        # (fixes bug where Trainer always used constant lambda_entropy,
        # ignoring model's entropy_schedule='anneal' config)
        if hasattr(self.model, '_get_effective_lambda_entropy'):
            lam_e = self.model._get_effective_lambda_entropy()
        else:
            lam_e = self.config.lambda_entropy

        return (
            self.config.lambda_factorize * ramp,
            self.config.lambda_coherence,
            lam_e,
        )

    def _forward_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Run model forward and compute total loss."""
        output = self.model(
            embeddings=batch.get('embeddings'),
            input_ids=batch.get('input_ids'),
            attention_mask=batch.get('attention_mask'),
            answer_embeddings=batch.get('answer_embeddings'),
            answer_input_ids=batch.get('answer_input_ids'),
            answer_attention_mask=batch.get('answer_attention_mask'),
            labels=batch.get('labels'),
            valid_choices=batch.get('valid_choices'),
            st_embeddings=batch.get('st_embeddings'),
        )

        # Compute weighted total loss
        lam_f, lam_c, lam_e = self._get_loss_weights()
        total_loss = output.get('task_loss', torch.tensor(0.0, device=self.device))

        fact_loss = output.get('factorization_loss', torch.tensor(0.0, device=self.device))
        coh_loss = output.get('coherence_loss', torch.tensor(0.0, device=self.device))
        ent_loss = output.get('entropy_loss', torch.tensor(0.0, device=self.device))

        total_loss = total_loss + lam_f * fact_loss + lam_c * coh_loss + lam_e * ent_loss
        output['total_loss'] = total_loss

        return output

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with optional gradient accumulation. Returns (loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        accum_steps = max(1, self.config.gradient_accumulation_steps)

        self.optimizer.zero_grad()
        for step_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            output = self._forward_step(batch)
            loss = output['total_loss'] / accum_steps

            loss.backward()

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(self.train_loader):
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.grad_clip,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            bs = batch['labels'].shape[0] if 'labels' in batch else 1
            total_loss += output['total_loss'].item() * bs  # unscaled loss for logging

            if 'predicted' in output:
                correct = (output['predicted'] == batch['labels']).sum().item()
                total_correct += correct
            elif 'scores' in output and output['scores'] is not None:
                predicted = output['scores'].argmax(dim=-1)
                total_correct += (predicted == batch['labels']).sum().item()

            total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        metrics = {
            'loss': 0.0, 'accuracy': 0.0, 'n': 0,
            'task_loss': 0.0, 'factorize_loss': 0.0,
            'coherence_loss': 0.0, 'entropy_loss': 0.0,
        }

        for batch in self.val_loader:
            batch = self._to_device(batch)
            output = self._forward_step(batch)

            bs = batch['labels'].shape[0] if 'labels' in batch else 1
            metrics['loss'] += output['total_loss'].item() * bs
            metrics['task_loss'] += output.get('task_loss', torch.tensor(0.0)).item() * bs
            metrics['factorize_loss'] += output.get('factorization_loss', torch.tensor(0.0)).item() * bs
            metrics['coherence_loss'] += output.get('coherence_loss', torch.tensor(0.0)).item() * bs
            metrics['entropy_loss'] += output.get('entropy_loss', torch.tensor(0.0)).item() * bs

            if 'predicted' in output:
                metrics['accuracy'] += (output['predicted'] == batch['labels']).sum().item()
            elif 'scores' in output and output['scores'] is not None:
                predicted = output['scores'].argmax(dim=-1)
                metrics['accuracy'] += (predicted == batch['labels']).sum().item()

            metrics['n'] += bs

        n = max(metrics['n'], 1)
        return {
            'loss': metrics['loss'] / n,
            'accuracy': metrics['accuracy'] / n,
            'task_loss': metrics['task_loss'] / n,
            'factorize_loss': metrics['factorize_loss'] / n,
            'coherence_loss': metrics['coherence_loss'] / n,
            'entropy_loss': metrics['entropy_loss'] / n,
        }

    def train(self) -> Dict:
        """Full training loop with early stopping."""
        logger.info(f"Training {self.model_name} for up to {self.config.max_epochs} epochs")

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch

            # Temperature annealing
            if hasattr(self.model, 'anneal_temperature'):
                self.model.anneal_temperature(epoch)

            train_loss, train_acc = self.train_epoch()
            val_metrics = self.evaluate()

            # LR scheduling
            if self.warmup_scheduler is not None and epoch < self.config.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_task_loss'].append(val_metrics['task_loss'])
            self.history['val_factorize_loss'].append(val_metrics['factorize_loss'])
            self.history['val_coherence_loss'].append(val_metrics['coherence_loss'])
            self.history['val_entropy_loss'].append(val_metrics['entropy_loss'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if hasattr(self.model, 'factorization') and hasattr(self.model.factorization, 'manifold'):
                self.history['temperature'].append(
                    self.model.factorization.manifold.tau.item()
                )
            else:
                self.history['temperature'].append(1.0)

            logger.info(
                f"  [{self.model_name}] Epoch {epoch+1}/{self.config.max_epochs}: "
                f"train_acc={train_acc:.2%} val_acc={val_metrics['accuracy']:.2%} "
                f"val_loss={val_metrics['loss']:.4f}"
            )

            # Early stopping
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return {
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'epochs_trained': len(self.history['train_loss']),
        }


def train_model(
    model: nn.Module,
    train_ds,
    val_ds,
    config: TrainConfig,
    model_name: str = 'model',
) -> Tuple[nn.Module, Dict]:
    """Train a model and return it with results."""
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_benchmark, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_benchmark, num_workers=0,
    )

    trainer = Trainer(model, train_loader, val_loader, config, model_name)
    results = trainer.train()
    return model, results


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_ds,
    config: TrainConfig,
) -> Dict[str, float]:
    """Evaluate a trained model on a test set."""
    model.eval()
    loader = DataLoader(
        test_ds, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_benchmark, num_workers=0,
    )

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for batch in loader:
        batch = {
            k: v.to(config.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        output = model(
            embeddings=batch.get('embeddings'),
            answer_embeddings=batch.get('answer_embeddings'),
            labels=batch.get('labels'),
        )

        if 'predicted' in output:
            predicted = output['predicted']
        elif 'scores' in output and output['scores'] is not None:
            predicted = output['scores'].argmax(dim=-1)
        else:
            continue

        total_correct += (predicted == batch['labels']).sum().item()
        total_loss += output.get('task_loss', torch.tensor(0.0)).item() * batch['labels'].shape[0]
        total_samples += batch['labels'].shape[0]

    n = max(total_samples, 1)
    return {
        'accuracy': total_correct / n,
        'loss': total_loss / n,
        'n_samples': total_samples,
    }


def curriculum_train(
    model: nn.Module,
    train_ds,
    val_ds,
    base_config: TrainConfig,
    phases: Optional[List[CurriculumPhase]] = None,
    model_name: str = 'model',
    collate_fn=None,
) -> Tuple[nn.Module, Dict]:
    """
    Train with curriculum learning over multiple phases.

    Each phase trains with different depth, loss weights, and LR.
    Model weights carry over between phases.
    """
    if phases is None:
        phases = DEFAULT_CURRICULUM

    total_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_task_loss': [], 'val_factorize_loss': [],
        'val_coherence_loss': [], 'val_entropy_loss': [],
        'lr': [], 'temperature': [],
        'phase_boundaries': [],
    }
    best_val_acc = 0.0
    best_state = None
    total_epochs = 0

    for phase_idx, phase in enumerate(phases):
        logger.info(
            f"  Curriculum Phase {phase_idx+1}/{len(phases)}: "
            f"depth={phase.depth}, epochs={phase.epochs}"
        )

        # Set model depth if applicable — propagate to both config and tree
        if hasattr(model, 'config'):
            model.config.max_depth = phase.depth
        if hasattr(model, 'reasoning') and hasattr(model.reasoning, 'tree'):
            model.reasoning.tree.max_depth = phase.depth

        # Phase-specific training config
        phase_config = TrainConfig(
            learning_rate=phase.learning_rate,
            weight_decay=base_config.weight_decay,
            grad_clip=base_config.grad_clip,
            batch_size=base_config.batch_size,
            encoder_lr=base_config.encoder_lr,
            gradient_accumulation_steps=base_config.gradient_accumulation_steps,
            max_epochs=phase.epochs,
            patience=max(3, phase.epochs // 3),
            warmup_epochs=2 if phase_idx == 0 else 0,
            lambda_factorize=phase.lambda_factorize,
            lambda_coherence=phase.lambda_coherence,
            lambda_entropy=base_config.lambda_entropy,
            loss_anneal_epochs=0,
            device=base_config.device,
            seed=base_config.seed,
        )

        _collate = collate_fn if collate_fn is not None else collate_benchmark
        train_loader = DataLoader(
            train_ds, batch_size=phase_config.batch_size,
            shuffle=True, collate_fn=_collate, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=phase_config.batch_size,
            shuffle=False, collate_fn=_collate, num_workers=0,
        )

        trainer = Trainer(
            model, train_loader, val_loader, phase_config,
            f"{model_name}_P{phase_idx+1}",
        )
        results = trainer.train()

        # Track best from latest phase
        if phase_idx == len(phases) - 1 or results['best_val_acc'] > best_val_acc:
            best_val_acc = results['best_val_acc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Append history
        total_history['phase_boundaries'].append(total_epochs)
        for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                     'val_task_loss', 'val_factorize_loss',
                     'val_coherence_loss', 'val_entropy_loss', 'lr', 'temperature']:
            total_history[key].extend(results['history'].get(key, []))

        total_epochs += results['epochs_trained']

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        'history': total_history,
        'best_val_acc': best_val_acc,
        'epochs_trained': total_epochs,
    }


@torch.no_grad()
def measure_inference_latency(
    model: nn.Module,
    batch_size: int = 1,
    hidden_dim: int = 256,
    n_answers: int = 4,
    n_warmup: int = 10,
    n_measure: int = 100,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Measure inference latency.

    Args:
        model: trained model
        batch_size: inference batch size
        hidden_dim: embedding dimension
        n_answers: number of answer choices
        n_warmup: warmup iterations
        n_measure: measurement iterations
        device: torch device

    Returns:
        Dict with mean_ms, std_ms, throughput
    """
    model.eval()
    model = model.to(device)

    # Synthetic input
    embeddings = torch.randn(batch_size, hidden_dim * 3, device=device)  # approximate encoder dim
    answer_emb = torch.randn(batch_size, n_answers, hidden_dim, device=device)

    # Check what input the model actually expects
    try:
        # Try with hidden_dim (for precomputed)
        test_emb = torch.randn(batch_size, 768, device=device)
        model(embeddings=test_emb, answer_embeddings=answer_emb)
        embeddings = test_emb
    except Exception:
        pass

    # Warmup
    for _ in range(n_warmup):
        try:
            model(embeddings=embeddings, answer_embeddings=answer_emb)
        except Exception:
            break

    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_measure):
        start = time.perf_counter()
        try:
            model(embeddings=embeddings, answer_embeddings=answer_emb)
        except Exception:
            break
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    if not latencies:
        return {'mean_ms': float('nan'), 'std_ms': float('nan'), 'throughput': 0.0}

    latencies = np.array(latencies)
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'throughput': float(batch_size * 1000 / np.mean(latencies)),
    }


def save_results(results: Dict, path: str):
    """Save results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    serializable = json.loads(json.dumps(results, default=convert))
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    from .data_utils import BenchmarkDataset, collate_benchmark
    from .model import COFRN, COFRNConfig
    from .baselines import LinearProbe

    logging.basicConfig(level=logging.INFO)
    print("Testing train_utils...")

    # Create synthetic dataset
    n_train, n_val = 100, 20
    hidden_dim = 256
    n_classes = 4

    train_ds = BenchmarkDataset(
        question_embeddings=torch.randn(n_train, 768),
        answer_embeddings=torch.randn(n_train, n_classes, hidden_dim),
        labels=torch.randint(0, n_classes, (n_train,)),
        domain='test',
    )
    val_ds = BenchmarkDataset(
        question_embeddings=torch.randn(n_val, 768),
        answer_embeddings=torch.randn(n_val, n_classes, hidden_dim),
        labels=torch.randint(0, n_classes, (n_val,)),
        domain='test',
    )

    config = TrainConfig(
        batch_size=16,
        max_epochs=3,
        patience=2,
        device='cpu',
    )

    # Test with LinearProbe
    model = LinearProbe(
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        encoder_input_dim=768,
    )
    model, results = train_model(model, train_ds, val_ds, config, 'linear_probe')
    print(f"\nLinearProbe: best_val_acc={results['best_val_acc']:.2%}, "
          f"epochs={results['epochs_trained']}")

    # Test latency measurement
    latency = measure_inference_latency(model, hidden_dim=hidden_dim)
    print(f"Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")

    print("\ntrain_utils tests passed!")
